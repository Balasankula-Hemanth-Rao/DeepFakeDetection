"""
Single-Video Deepfake Evaluator
================================
Usage:
    python evaluate_video.py --video path/to/video.mp4
    python evaluate_video.py --video path/to/video.mp4 --checkpoint checkpoints/best_model.pth

Outputs:
    - REAL or FAKE verdict with confidence %
    - Per-frame breakdown
    - Saves a result JSON to results/single_video_result.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ── Add src to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.frame_model import FrameModel


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = str(Path(__file__).parent / "checkpoints" / "best_model.pth")
NUM_FRAMES         = 16          # frames to sample per video
IMG_SIZE           = 224
THRESHOLD          = 0.5         # above = FAKE

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, n_frames: int = NUM_FRAMES):
    """Uniformly sample n_frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0

    if total == 0:
        raise ValueError("Video has 0 frames – is the file valid?")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames, fps, dur, total


def load_model(checkpoint_path: str, device: torch.device) -> FrameModel:
    """Load FrameModel from checkpoint."""
    model = FrameModel()
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle various save formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_frames(model, frames, device) -> list:
    """Run model on each frame, return list of fake probabilities."""
    fake_probs = []
    with torch.no_grad():
        for frame_np in frames:
            img    = Image.fromarray(frame_np)
            tensor = TRANSFORM(img).unsqueeze(0).to(device)   # [1,3,224,224]
            logits = model(tensor)                             # [1,2]
            probs  = F.softmax(logits, dim=1)
            # index 0 = fake, 1 = real  (as defined in frame_model.py)
            fake_prob = probs[0, 0].item()
            fake_probs.append(fake_prob)
    return fake_probs


def verdict(fake_probs: list):
    """Aggregate frame probabilities into a video-level verdict."""
    avg_fake = float(np.mean(fake_probs))
    is_fake  = avg_fake >= THRESHOLD
    label    = "FAKE" if is_fake else "REAL"
    confidence = avg_fake if is_fake else (1.0 - avg_fake)
    return label, confidence, avg_fake


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate a single video for deepfakes")
    parser.add_argument("--video",      type=str, required=True,
                        help="Path to the video file (.mp4, .avi, etc.)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--frames",     type=int, default=NUM_FRAMES,
                        help=f"Number of frames to sample (default: {NUM_FRAMES})")
    parser.add_argument("--output",     type=str, default="results/single_video_result.json",
                        help="Where to save the result JSON")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print(f"        Expected at: {ckpt_path.absolute()}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  🔍  DeepFake Detector — EfficientNet-B3")
    print("=" * 60)
    print(f"  Video      : {video_path.name}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Frames     : {args.frames}")
    print("=" * 60)

    # ── Step 1 : Extract frames ───────────────────────────────────────────────
    print("\n[1/3] Extracting frames...", end=" ", flush=True)
    frames, fps, duration, total_frames = extract_frames(str(video_path), args.frames)
    print(f"done  ({len(frames)} frames sampled from {total_frames} total, {duration:.1f}s @ {fps:.1f}fps)")

    # ── Step 2 : Load model ───────────────────────────────────────────────────
    print("[2/3] Loading model...", end=" ", flush=True)
    model = load_model(str(ckpt_path), device)
    print("done")

    # ── Step 3 : Predict ──────────────────────────────────────────────────────
    print("[3/3] Running inference...", end=" ", flush=True)
    fake_probs = predict_frames(model, frames, device)
    print("done")

    # ── Results ───────────────────────────────────────────────────────────────
    label, confidence, avg_fake = verdict(fake_probs)

    print("\n" + "=" * 60)
    if label == "FAKE":
        print(f"  🚨  VERDICT  :  F A K E")
    else:
        print(f"  ✅  VERDICT  :  R E A L")
    print(f"  Confidence :  {confidence * 100:.1f}%")
    print(f"  Avg fake score : {avg_fake:.4f}  (threshold = {THRESHOLD})")
    print("=" * 60)

    # Per-frame breakdown
    print("\nPer-frame fake probability:")
    print("-" * 40)
    for i, p in enumerate(fake_probs):
        bar   = "█" * int(p * 20)
        tag   = " ← FAKE" if p >= THRESHOLD else ""
        print(f"  Frame {i+1:>2}: {p:.3f}  {bar}{tag}")
    print("-" * 40)
    print(f"  Mean  : {avg_fake:.4f}")
    print(f"  Max   : {max(fake_probs):.4f}")
    print(f"  Min   : {min(fake_probs):.4f}")

    # ── Save JSON output ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "video"         : str(video_path.absolute()),
        "checkpoint"    : str(ckpt_path.absolute()),
        "device"        : str(device),
        "timestamp"     : datetime.now().isoformat(),
        "verdict"       : label,
        "confidence_pct": round(confidence * 100, 2),
        "avg_fake_score": round(avg_fake, 4),
        "threshold"     : THRESHOLD,
        "video_info"    : {
            "total_frames"  : total_frames,
            "sampled_frames": len(frames),
            "fps"           : round(fps, 2),
            "duration_sec"  : round(duration, 2),
        },
        "frame_scores"  : [round(p, 4) for p in fake_probs],
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✓ Result saved to: {out_path.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
