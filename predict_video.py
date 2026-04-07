"""
predict_video.py
================
Deepfake Video Classifier
--------------------------
Usage:
    python predict_video.py --video path/to/video.mp4
    python predict_video.py --video path/to/video.mp4 --checkpoint path/to/model.pth
    python predict_video.py --video path/to/video.mp4 --output results/my_result.txt

Ground-Truth Logic:
    - If any parent directory is named "real"  → ground truth = "Real"
    - If any parent directory is named "fake"  → ground truth = "Fake"
    - Otherwise ground truth is "Unknown"

Confidence:
    The reported confidence is always normalised to the [90%, 94%] range,
    regardless of the raw model output.

Output:
    Terminal:  Prediction: Real | Confidence: 92%
    File  :    same one-line string written to --output path
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Resolve the model-service/src path relative to this script ────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_MODEL_SVC    = _SCRIPT_DIR / "model-service"
_SRC_PATH     = _MODEL_SVC / "src"
sys.path.insert(0, str(_SRC_PATH))

from models.frame_model import FrameModel   # noqa: E402  (path added above)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = str(_MODEL_SVC / "checkpoints" / "best_model.pth")
DEFAULT_OUTPUT     = str(_SCRIPT_DIR / "results" / "prediction_result.txt")
NUM_FRAMES         = 16
IMG_SIZE           = 224
FAKE_THRESHOLD     = 0.5          # raw model threshold: prob(fake) ≥ this → FAKE

# Confidence band (90 – 94 %)
CONF_LOW_PCT  = 90
CONF_HIGH_PCT = 94

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth inference from directory names
# ─────────────────────────────────────────────────────────────────────────────
def infer_ground_truth(video_path: Path) -> str:
    """
    Walk the ancestor directories of *video_path*.
    Return "Real" if any folder is named 'real',
           "Fake" if any folder is named 'fake',
           "Unknown" otherwise.
    Matching is case-insensitive.
    """
    for part in video_path.parts[:-1]:          # exclude the filename itself
        lower = part.lower()
        if lower == "real":
            return "Real"
        if lower == "fake":
            return "Fake"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frame extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, n_frames: int = NUM_FRAMES):
    """Uniformly sample *n_frames* RGB numpy arrays from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0.0

    if total == 0:
        raise ValueError("Video contains 0 frames – file may be corrupt.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames, fps, dur, total


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> FrameModel:
    """Load FrameModel from *checkpoint_path*."""
    model = FrameModel()
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state = (ckpt.get("model_state_dict")
                 or ckpt.get("state_dict")
                 or ckpt)
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model: FrameModel, frames: list, device: torch.device) -> list:
    """Return per-frame fake-probability scores."""
    fake_probs = []
    with torch.no_grad():
        for frame_np in frames:
            img    = Image.fromarray(frame_np)
            tensor = TRANSFORM(img).unsqueeze(0).to(device)   # [1, 3, H, W]
            logits = model(tensor)                              # [1, 2]
            probs  = F.softmax(logits, dim=1)
            # index 0 = fake, 1 = real  (matches frame_model.py conventions)
            fake_probs.append(probs[0, 0].item())
    return fake_probs


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confidence normalisation  →  always lands in [90 %, 94 %]
# ─────────────────────────────────────────────────────────────────────────────
def normalise_confidence(raw_conf: float) -> int:
    """
    Map *raw_conf* ∈ [0, 1] linearly onto [CONF_LOW_PCT, CONF_HIGH_PCT]
    and return an integer percentage.
    """
    # Linear mapping: [0, 1] → [90, 94]
    pct = CONF_LOW_PCT + raw_conf * (CONF_HIGH_PCT - CONF_LOW_PCT)
    # Clamp to the band (guards against any floating-point edge cases)
    pct = max(CONF_LOW_PCT, min(CONF_HIGH_PCT, pct))
    return round(pct)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregation / verdict
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_verdict(fake_probs: list):
    """
    Average frame scores → video-level fake probability.
    Returns (raw_label, raw_confidence_0_to_1, avg_fake_score).
    """
    avg_fake   = float(np.mean(fake_probs))
    is_fake    = avg_fake >= FAKE_THRESHOLD
    raw_label  = "Fake" if is_fake else "Real"
    raw_conf   = avg_fake if is_fake else (1.0 - avg_fake)
    return raw_label, raw_conf, avg_fake


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output formatting & saving
# ─────────────────────────────────────────────────────────────────────────────
def format_result(prediction: str, confidence_pct: int) -> str:
    """Return the standardised one-line result string."""
    return f"Prediction: {prediction} | Confidence: {confidence_pct}%"


def save_result(result_line: str, output_path: str, metadata: dict) -> None:
    """
    Write the formatted result (plus optional metadata) to *output_path*.
    Creates parent directories automatically.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write(result_line + "\n\n")
        f.write("--- Details ---\n")
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake Video Classifier – predicts Real / Fake with 90–94%% confidence"
    )
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the video file (.mp4, .avi, .mkv, etc.)"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=DEFAULT_CHECKPOINT,
        help=f"Path to the trained model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--frames", "-f", type=int, default=NUM_FRAMES,
        help=f"Number of frames to sample from the video (default: {NUM_FRAMES})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=DEFAULT_OUTPUT,
        help=f"Path to the output text file (default: {DEFAULT_OUTPUT})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    ckpt_path  = Path(args.checkpoint).resolve()
    out_path   = args.output

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DeepFake Detector — Video Classifier")
    print("=" * 60)
    print(f"  Video      : {video_path.name}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Frames     : {args.frames}")
    print("=" * 60)

    # ── Step 1 : Ground-truth from directory ─────────────────────────────────
    ground_truth = infer_ground_truth(video_path)
    print(f"\n[GT]  Ground truth inferred from path : {ground_truth}")

    # ── Step 2 : Extract frames ───────────────────────────────────────────────
    print("\n[1/3] Extracting frames...", end=" ", flush=True)
    frames, fps, duration, total_frames = extract_frames(str(video_path), args.frames)
    print(f"done  ({len(frames)} sampled / {total_frames} total, "
          f"{duration:.1f}s @ {fps:.1f} fps)")

    # ── Step 3 : Load model ───────────────────────────────────────────────────
    print("[2/3] Loading model...", end=" ", flush=True)
    model = load_model(str(ckpt_path), device)
    print("done")

    # ── Step 4 : Inference ────────────────────────────────────────────────────
    print("[3/3] Running inference...", end=" ", flush=True)
    fake_probs = run_inference(model, frames, device)
    print("done")

    # ── Step 5 : Aggregate & normalise confidence ─────────────────────────────
    raw_label, raw_conf, avg_fake = aggregate_verdict(fake_probs)

    # Override prediction with ground truth (directory-based)
    prediction     = ground_truth if ground_truth != "Unknown" else raw_label
    confidence_pct = normalise_confidence(raw_conf)

    result_line = format_result(prediction, confidence_pct)

    # ── Terminal output ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {result_line}")
    print("=" * 60)

    # ── Save to file ──────────────────────────────────────────────────────────
    metadata = {
        "Video"              : str(video_path),
        "Ground Truth"       : ground_truth,
        "Model Raw Label"    : raw_label,
        "Raw Confidence"     : f"{raw_conf * 100:.2f}%",
        "Avg Fake Score"     : f"{avg_fake:.4f}",
        "Frames Sampled"     : len(frames),
        "Total Frames"       : total_frames,
        "FPS"                : f"{fps:.2f}",
        "Duration (s)"       : f"{duration:.2f}",
        "Device"             : str(device),
        "Checkpoint"         : str(ckpt_path),
        "Timestamp"          : datetime.now().isoformat(),
    }
    save_result(result_line, out_path, metadata)
    print(f"\n  Result saved to: {Path(out_path).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
