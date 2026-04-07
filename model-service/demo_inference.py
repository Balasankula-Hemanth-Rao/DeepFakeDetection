"""
DeepFake Detection — Professor Demo
=====================================
Downloads sample face images, runs them through the trained EfficientNet-B3
model and prints a clear REAL / FAKE verdict with confidence scores.

Usage:
    python demo_inference.py                     # auto-download sample faces
    python demo_inference.py --image my_face.jpg # your own image
"""

import sys, os, json, argparse, urllib.request, io
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.frame_model import FrameModel

CHECKPOINT = Path(__file__).parent / "checkpoints" / "best_model.pth"
THRESHOLD  = 0.5   # fake score above this → FAKE
OUT_DIR    = Path(__file__).parent / "results" / "demo"

# ── Sample faces (Creative-Commons / public domain via ThisPersonDoesNotExist)
# We'll use known-real portrait images from Wikipedia Commons as "real" samples
# and note the corrupted FakeAVCeleb files as "fake" samples
SAMPLE_URLS = {
    "sample_A_real": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/500px-Gatto_europeo4.jpg",
    "sample_B_real": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/500px-Cute_dog.jpg",
}

# ── transform ─────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
def load_model(device):
    if not CHECKPOINT.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)
    model = FrameModel()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def predict_image(model, img: Image.Image, device) -> dict:
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    fake_score = probs[0].item()   # index 0 = fake
    real_score = probs[1].item()   # index 1 = real
    label      = "FAKE" if fake_score >= THRESHOLD else "REAL"
    confidence = fake_score if label == "FAKE" else real_score
    return {"label": label, "confidence": confidence,
            "fake_score": fake_score, "real_score": real_score}


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def make_result_card(img: Image.Image, result: dict, name: str) -> Image.Image:
    """Draw a nice result card on the image for professor display."""
    W, H   = 640, 480
    canvas = img.copy().resize((W, H))
    draw   = ImageDraw.Draw(canvas)

    label   = result["label"]
    conf    = result["confidence"] * 100
    f_score = result["fake_score"]
    r_score = result["real_score"]

    # Semi-transparent overlay at bottom
    overlay = Image.new("RGBA", (W, 160), (0, 0, 0, 180))
    canvas  = canvas.convert("RGBA")
    canvas.paste(overlay, (0, H - 160), overlay)
    canvas  = canvas.convert("RGB")
    draw    = ImageDraw.Draw(canvas)

    # Verdict colour
    color = (255, 80, 80) if label == "FAKE" else (80, 220, 80)

    # Title
    draw.text((20, H - 150), f"Verdict:  {label}", fill=color)
    draw.text((20, H - 115), f"Confidence: {conf:.1f}%", fill=(255, 255, 255))
    draw.text((20, H - 85),  f"Fake score: {f_score:.3f}   Real score: {r_score:.3f}",
              fill=(200, 200, 200))
    draw.text((20, H - 55),  f"Model: EfficientNet-B3  |  Threshold: {THRESHOLD}",
              fill=(150, 150, 150))
    draw.text((20, H - 28),  f"File: {name}",
              fill=(150, 150, 150))

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 380, H - 120, 240, 18
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                   fill=(60, 60, 60), outline=(120, 120, 120))
    fill_w = int(bar_w * f_score)
    draw.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=color)
    draw.text((bar_x, bar_y - 20), "Fake probability:", fill=(200, 200, 200))

    return canvas


def print_result(name: str, result: dict):
    label = result["label"]
    conf  = result["confidence"] * 100
    bar   = "█" * int(result["fake_score"] * 30)
    icon  = "🚨 FAKE" if label == "FAKE" else "✅ REAL"
    print(f"\n  {'─'*50}")
    print(f"  Image     : {name}")
    print(f"  Verdict   : {icon}")
    print(f"  Confidence: {conf:.1f}%")
    print(f"  Fake ▶ [{bar:<30}] {result['fake_score']:.3f}")
    print(f"  Real ▶ [{('█'*int(result['real_score']*30)):<30}] {result['real_score']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepFake detection demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a local image file to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  🔍  DeepFake Detector — EfficientNet-B3  (Professor Demo)")
    print("="*60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {CHECKPOINT.name}  ({CHECKPOINT.stat().st_size/1e6:.1f} MB)")
    print(f"  Threshold  : fake_score ≥ {THRESHOLD} → FAKE")
    print("="*60)

    # Load model
    print("\n[1/3] Loading model...", end=" ", flush=True)
    model = load_model(device)
    print("done ✓")

    results_log = []

    # ── Case 1: user supplied an image ───────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERROR] Image not found: {img_path}")
            sys.exit(1)
        samples = {img_path.stem: Image.open(img_path).convert("RGB")}

    # ── Case 2: create synthetic test images (no download needed) ────────────
    else:
        print("\n[2/3] Creating synthetic test samples...")
        print("      (Using synthetic face-like patches — no internet needed)\n")
        samples = {}

        # ── Synthetic "REAL-like" image: smooth gradient face shape
        rng = np.random.default_rng(42)
        real_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        # skin-tone background
        real_arr[:, :] = [220, 180, 140]
        # add smooth noise (natural texture)
        noise = (rng.normal(0, 8, (224, 224, 3))).astype(np.int16)
        real_arr = np.clip(real_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        samples["synthetic_natural_face"] = Image.fromarray(real_arr)

        # ── Synthetic "FAKE-like" image: blocked artifacts + unnatural colours
        fake_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for bx in range(0, 224, 16):
            for by in range(0, 224, 16):
                col = rng.integers(80, 200, 3).astype(np.uint8)
                fake_arr[by:by+16, bx:bx+16] = col
        # Add sharp high-frequency noise (typical deepfake artifact)
        hf = rng.integers(-60, 60, (224, 224, 3)).astype(np.int16)
        fake_arr = np.clip(fake_arr.astype(np.int16) + hf, 0, 255).astype(np.uint8)
        samples["synthetic_artifact_face"] = Image.fromarray(fake_arr)

        print("      ✓ synthetic_natural_face  (smooth, skin-tone — expected REAL)")
        print("      ✓ synthetic_artifact_face (blocky + noisy — expected FAKE)")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n[3/3] Running inference...")
    for name, img in samples.items():
        result = predict_image(model, img, device)
        print_result(name, result)

        # Save annotated card
        card     = make_result_card(img, result, name)
        out_path = OUT_DIR / f"{name}_result.png"
        card.save(out_path)
        print(f"            → saved: {out_path}")

        results_log.append({"image": name, **result})

    # ── Save JSON log ─────────────────────────────────────────────────────────
    log_path = OUT_DIR / "demo_results.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model"    : "EfficientNet-B3",
            "device"   : str(device),
            "threshold": THRESHOLD,
            "samples"  : results_log,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✓ Result images saved to : {OUT_DIR}")
    print(f"  ✓ JSON log saved to      : {log_path}")
    print(f"{'='*60}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"  {'Name':<35} {'Verdict':<8} {'Confidence':>10}")
    print(f"  {'─'*35} {'─'*8} {'─'*10}")
    for r in results_log:
        print(f"  {r['image']:<35} {r['label']:<8} {r['confidence']*100:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
