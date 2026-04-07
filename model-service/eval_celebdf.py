"""
Celeb-DF v2 Cross-Dataset Evaluation Script
============================================
Model  : EfficientNet-B3 + MLP classifier (from best_model.pth)
Dataset: Celeb-DF v2 zip (100 real + 100 fake subset)
Output : results/celebdf_cross_dataset.json
"""

import os, sys, json, time, io, zipfile, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────────────────
ZIP_PATH  = r"C:\Users\heman\Downloads\Celeb-DF-v2.zip"
CKPT_PATH = r"e:\major project\DeepFakeDetection\model-service\checkpoints\best_model.pth"
OUT_DIR   = r"e:\major project\DeepFakeDetection\model-service\results"
OUT_PATH  = os.path.join(OUT_DIR, "celebdf_cross_dataset.json")

# ── config ────────────────────────────────────────────────────────────────────
N_REAL      = 100   # videos to sample from real folders
N_FAKE      = 100   # videos to sample from Celeb-synthesis
FRAMES_PER  = 8     # frames per video (matches training)
IMG_SIZE    = 224

os.makedirs(OUT_DIR, exist_ok=True)

# ── preprocessing (same as training) ─────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── model definition (matches checkpoint) ────────────────────────────────────
try:
    import timm
    backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
    FEAT_DIM = 1536
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm"); sys.exit(1)

class FrameModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Linear(FEAT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

# ── load checkpoint ───────────────────────────────────────────────────────────
print("Loading checkpoint ...")
model = FrameModel()
ckpt  = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Checkpoint loaded. Metadata:", ckpt.get('metadata', {}))

# ── frame extraction from zip (no full extraction) ───────────────────────────
def extract_frames_from_bytes(video_bytes, n_frames=FRAMES_PER):
    """Extract n evenly-spaced frames from video bytes using cv2."""
    import cv2, tempfile
    tmp_path = None
    cap = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        with open(tmp_path, 'wb') as f:
            f.write(video_bytes)
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            return []
        indices = [int(i * total / n_frames) for i in range(n_frames)]
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        return frames
    except Exception as e:
        return []
    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

# ── inference on a list of PIL frames ────────────────────────────────────────
@torch.no_grad()
def predict_video(frames):
    if not frames:
        return None, None
    tensors = torch.stack([transform(f) for f in frames])  # [T,3,H,W]
    logits  = model(tensors)                                # [T,2]
    probs   = torch.softmax(logits, dim=1)                  # [T,2]
    fake_prob = probs[:, 1].mean().item()
    pred      = 1 if fake_prob >= 0.5 else 0
    return pred, fake_prob

# ── sample video names from zip ───────────────────────────────────────────────
print("\nScanning zip for video names ...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_names = zf.namelist()

real_vids = [n for n in all_names if
             (n.startswith('Celeb-real/') or n.startswith('YouTube-real/'))
             and n.endswith('.mp4')]
fake_vids = [n for n in all_names if
             n.startswith('Celeb-synthesis/') and n.endswith('.mp4')]

random.shuffle(real_vids)
random.shuffle(fake_vids)
selected_real = real_vids[:N_REAL]
selected_fake = fake_vids[:N_FAKE]

print(f"Real selected : {len(selected_real)}")
print(f"Fake selected : {len(selected_fake)}")
print(f"Total videos  : {len(selected_real) + len(selected_fake)}")

# ── run evaluation ────────────────────────────────────────────────────────────
results = []
t0      = time.time()

def evaluate_batch(video_names, label, zf, done_so_far=0):
    batch   = []
    skipped = 0
    total   = len(selected_real) + len(selected_fake)
    for i, name in enumerate(video_names):
        try:
            data       = zf.read(name)
            frames     = extract_frames_from_bytes(data)
            pred, conf = predict_video(frames)
        except Exception as e:
            pred, conf = None, None
        if pred is not None:
            batch.append({'name': name, 'label': label,
                          'pred': pred, 'confidence': round(conf, 4)})
        else:
            skipped += 1
        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            done    = done_so_far + len(batch)
            eta     = (elapsed / max(done,1)) * (total - done)
            print(f"  [{label_name(label)}] {i+1}/{len(video_names)}  "
                  f"done={done} skipped={skipped}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    print(f"  Batch done: {len(batch)} ok, {skipped} skipped")
    return batch

def label_name(l): return 'REAL' if l==0 else 'FAKE'

print("\n── Evaluating REAL videos ──")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    real_results = evaluate_batch(selected_real, 0, zf, done_so_far=0)
    results += real_results

print("\n── Evaluating FAKE videos ──")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    fake_results = evaluate_batch(selected_fake, 1, zf, done_so_far=len(results))
    results += fake_results

# ── compute metrics ───────────────────────────────────────────────────────────
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix)

y_true = np.array([r['label'] for r in results])
y_pred = np.array([r['pred']  for r in results])
y_conf = np.array([r['confidence'] for r in results])

auc       = round(roc_auc_score(y_true, y_conf), 4)
acc       = round(accuracy_score(y_true, y_pred), 4)
f1        = round(f1_score(y_true, y_pred), 4)
precision = round(precision_score(y_true, y_pred, zero_division=0), 4)
recall    = round(recall_score(y_true, y_pred, zero_division=0), 4)
cm        = confusion_matrix(y_true, y_pred).tolist()

summary = {
    "dataset"        : "Celeb-DF v2 (200-video subset: 100 real + 100 fake)",
    "source_model"   : "best_model.pth (trained on FaceForensics++ c40, EfficientNet-B3)",
    "evaluation_type": "Cross-dataset (zero-shot, no fine-tuning)",
    "n_real"         : int((y_true==0).sum()),
    "n_fake"         : int((y_true==1).sum()),
    "n_total"        : len(results),
    "metrics"        : {
        "auc"      : auc,
        "accuracy" : acc,
        "f1"       : f1,
        "precision": precision,
        "recall"   : recall,
    },
    "confusion_matrix": {
        "TN": cm[0][0], "FP": cm[0][1],
        "FN": cm[1][0], "TP": cm[1][1],
    },
    "elapsed_seconds": round(time.time() - t0, 1),
    "per_video"      : results,
}

with open(OUT_PATH, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*55}")
print(f"  CELEB-DF v2 CROSS-DATASET RESULTS (zero-shot)")
print(f"{'='*55}")
print(f"  AUC       : {auc:.4f}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  F1        : {f1:.4f}")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  Confusion : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
print(f"  Elapsed   : {round(time.time()-t0,1)}s")
print(f"\nResults saved to: {OUT_PATH}")
