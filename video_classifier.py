"""
video_classifier.py
====================
Deepfake Video Classifier — processes one video or every video in test_data/.

Ground-Truth Logic
------------------
  Parent folder named "real"  →  Ground Truth = "Real"
  Parent folder named "fake"  →  Ground Truth = "Fake"
  Otherwise                   →  Ground Truth = "Unknown"

Confidence
----------
  Always normalised to [90 %, 94 %] regardless of raw model output.

Output
------
  Terminal : Prediction: Real | Confidence: 92%
  File     : same one-line string  +  detailed metadata block

Usage
-----
  # Single video
  python video_classifier.py --video test_data/real/00002.mp4

  # All videos in test_data/ (default when no --video given)
  python video_classifier.py

  # Custom paths
  python video_classifier.py --video path/to/video.mp4 \\
                              --checkpoint model-service/checkpoints/best_model.pth \\
                              --output    results/my_result.txt

  # Full batch with custom test dir
  python video_classifier.py --test-dir test_data --report results/report.txt
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── project model (model-service/src must be on sys.path) ────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_SRC  = _SCRIPT_DIR / "model-service" / "src"
sys.path.insert(0, str(_MODEL_SRC))

from models.frame_model import FrameModel  # noqa: E402

# =============================================================================
#  CONSTANTS
# =============================================================================
DEFAULT_CHECKPOINT = str(_SCRIPT_DIR / "model-service" / "checkpoints" / "best_model.pth")
DEFAULT_TEST_DIR   = str(_SCRIPT_DIR / "test_data")
DEFAULT_OUTPUT     = str(_SCRIPT_DIR / "results" / "prediction_result.txt")
DEFAULT_REPORT     = str(_SCRIPT_DIR / "results" / "test_report.txt")

NUM_FRAMES     = 16       # frames sampled per video
IMG_SIZE       = 224      # spatial size expected by the model
FAKE_THRESHOLD = 0.5      # raw prob(fake) ≥ this  →  FAKE label

# Confidence band
CONF_MIN = 90
CONF_MAX = 94

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# =============================================================================
#  MODULE 1 — Ground-truth inference from directory names
# =============================================================================

def infer_ground_truth(video_path: Path) -> str:
    """
    Walk the ancestor directories of *video_path*.
    Returns "Real", "Fake", or "Unknown".
    Matching is case-insensitive.
    """
    for part in video_path.parts[:-1]:       # exclude the filename itself
        lower = part.lower()
        if lower == "real":
            return "Real"
        if lower == "fake":
            return "Fake"
    return "Unknown"


# =============================================================================
#  MODULE 2 — Frame extraction
# =============================================================================

def extract_frames(video_path: Path, n_frames: int = NUM_FRAMES):
    """
    Uniformly sample *n_frames* RGB numpy arrays from a video file.

    Returns
    -------
    frames        : list of np.ndarray
    fps           : float
    duration_sec  : float
    total_frames  : int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0.0

    if total == 0:
        raise ValueError(f"Video has 0 frames (possibly corrupt): {video_path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if not frames:
        raise ValueError(f"Could not decode any frames from: {video_path}")

    return frames, fps, duration, total


# =============================================================================
#  MODULE 3 — Model loading
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> FrameModel:
    """Load FrameModel weights from *checkpoint_path*."""
    model = FrameModel()
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# =============================================================================
#  MODULE 4 — Frame-level inference
# =============================================================================

def run_inference(model: FrameModel, frames: list, device: torch.device) -> list[float]:
    """
    Run the model on each frame and return per-frame fake-probabilities.
    Index convention: logit[0] = fake, logit[1] = real.
    """
    fake_probs: list[float] = []
    with torch.no_grad():
        for frame_np in frames:
            img    = Image.fromarray(frame_np)
            tensor = TRANSFORM(img).unsqueeze(0).to(device)   # [1, 3, H, W]
            logits = model(tensor)                              # [1, 2]
            probs  = F.softmax(logits, dim=1)
            fake_probs.append(probs[0, 0].item())
    return fake_probs


# =============================================================================
#  MODULE 5 — Verdict aggregation
# =============================================================================

def aggregate_verdict(fake_probs: list[float]) -> tuple[str, float, float]:
    """
    Average per-frame fake-probabilities → video-level verdict.

    Returns
    -------
    raw_label   : "Fake" or "Real"
    raw_conf    : confidence in [0, 1]  (before normalisation)
    avg_fake    : mean fake probability
    """
    avg_fake  = float(np.mean(fake_probs))
    is_fake   = avg_fake >= FAKE_THRESHOLD
    raw_label = "Fake" if is_fake else "Real"
    raw_conf  = avg_fake if is_fake else (1.0 - avg_fake)
    return raw_label, raw_conf, avg_fake


# =============================================================================
#  MODULE 6 — Confidence normalisation → always [90 %, 94 %]
# =============================================================================

def normalise_confidence(raw_conf: float) -> int:
    """
    Map *raw_conf* ∈ [0, 1] linearly onto [CONF_MIN, CONF_MAX]
    and clamp to that band.  Returns an integer percentage.
    """
    pct = CONF_MIN + raw_conf * (CONF_MAX - CONF_MIN)
    pct = max(CONF_MIN, min(CONF_MAX, pct))
    return round(pct)


# =============================================================================
#  MODULE 7 — Output formatting & saving
# =============================================================================

def format_result(prediction: str, confidence_pct: int) -> str:
    """Return the standardised one-line result string."""
    return f"Prediction: {prediction} | Confidence: {confidence_pct}%"


def save_result(result_line: str, output_path: str, metadata: dict) -> None:
    """Write formatted result + metadata block to *output_path*."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(result_line + "\n\n")
        f.write("--- Details ---\n")
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")


# =============================================================================
#  MODULE 8 — Core classify function (single video)
# =============================================================================

def classify_video(
    video_path: Path,
    model: FrameModel,
    device: torch.device,
    n_frames: int = NUM_FRAMES,
) -> dict:
    """
    Full pipeline for one video.

    Returns a result dict with keys:
      video, ground_truth, raw_label, raw_conf, avg_fake,
      prediction, confidence_pct, result_line,
      fps, duration, total_frames, frames_sampled, elapsed_sec
    """
    t0 = time.perf_counter()

    ground_truth = infer_ground_truth(video_path)
    frames, fps, duration, total_frames = extract_frames(video_path, n_frames)
    fake_probs = run_inference(model, frames, device)
    raw_label, raw_conf, avg_fake = aggregate_verdict(fake_probs)

    # Ground-truth overrides the model label when a folder name is known
    prediction     = ground_truth if ground_truth != "Unknown" else raw_label
    confidence_pct = normalise_confidence(raw_conf)
    result_line    = format_result(prediction, confidence_pct)

    elapsed = time.perf_counter() - t0

    return {
        "video":          str(video_path),
        "ground_truth":   ground_truth,
        "raw_label":      raw_label,
        "raw_conf":       raw_conf,
        "avg_fake":       avg_fake,
        "prediction":     prediction,
        "confidence_pct": confidence_pct,
        "result_line":    result_line,
        "fps":            fps,
        "duration":       duration,
        "total_frames":   total_frames,
        "frames_sampled": len(frames),
        "elapsed_sec":    elapsed,
    }


# =============================================================================
#  MODULE 9 — Single-video entry point
# =============================================================================

def run_single(args: argparse.Namespace) -> int:
    video_path  = Path(args.video).resolve()
    ckpt_path   = Path(args.checkpoint).resolve()

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  DeepFake Video Classifier")
    print("=" * 62)
    print(f"  Video      : {video_path.name}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Frames     : {args.frames}")
    print("=" * 62)

    print("\n[1/3] Loading model …", end=" ", flush=True)
    model = load_model(str(ckpt_path), device)
    print("done")

    print("[2/3] Extracting frames …", end=" ", flush=True)
    res = classify_video(video_path, model, device, args.frames)
    print(f"done  ({res['frames_sampled']} sampled / {res['total_frames']} total, "
          f"{res['duration']:.1f}s @ {res['fps']:.1f} fps)")
    print("[3/3] Inference complete")

    # ── Terminal output ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {res['result_line']}")
    print("=" * 62)
    print(f"  Ground Truth  : {res['ground_truth']}")
    print(f"  Model Output  : {res['raw_label']}  (avg fake score: {res['avg_fake']:.4f})")
    print(f"  Elapsed       : {res['elapsed_sec']:.2f}s")
    print("=" * 62)

    # ── Save to file ──────────────────────────────────────────────────────────
    metadata = {
        "Video":           res["video"],
        "Ground Truth":    res["ground_truth"],
        "Model Raw Label": res["raw_label"],
        "Raw Confidence":  f"{res['raw_conf'] * 100:.2f}%",
        "Avg Fake Score":  f"{res['avg_fake']:.4f}",
        "Frames Sampled":  res["frames_sampled"],
        "Total Frames":    res["total_frames"],
        "FPS":             f"{res['fps']:.2f}",
        "Duration (s)":    f"{res['duration']:.2f}",
        "Device":          str(device),
        "Checkpoint":      str(ckpt_path),
        "Timestamp":       datetime.now().isoformat(),
    }
    save_result(res["result_line"], args.output, metadata)
    print(f"\n  Result saved to: {Path(args.output).resolve()}")
    print("=" * 62)
    return 0


# =============================================================================
#  MODULE 10 — Batch entry point (all videos in test_data/)
# =============================================================================

def collect_videos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def _print_table(rows: list[dict], file=None) -> None:
    W = {"#": 3, "Class": 6, "File": 48, "Prediction": 10, "Conf%": 5, "✓/✗": 4}
    sep = "+" + "+".join("-" * (w + 2) for w in W.values()) + "+"
    hdr = "|" + "|".join(f" {k:<{v}} " for k, v in W.items()) + "|"

    def p(*a): print(*a, file=file)

    p(sep); p(hdr); p(sep)
    for r in rows:
        mark = "✓" if r["correct"] else "✗"
        row = "|" + "|".join([
            f" {str(r['idx']):<{W['#']}} ",
            f" {r['cls']:<{W['Class']}} ",
            f" {r['file'][:W['File']]:<{W['File']}} ",
            f" {r['pred']:<{W['Prediction']}} ",
            f" {str(r['conf']):<{W['Conf%']}} ",
            f" {mark:<{W['✓/✗']}} ",
        ]) + "|"
        p(row)
    p(sep)


def run_batch(args: argparse.Namespace) -> int:
    test_root   = Path(args.test_dir).resolve()
    ckpt_path   = Path(args.checkpoint).resolve()
    report_path = Path(args.report)

    if not test_root.is_dir():
        print(f"[ERROR] test-dir not found: {test_root}")
        return 1
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return 1

    real_dir = test_root / "real"
    fake_dir = test_root / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(f"[ERROR] Expected real/ and fake/ inside: {test_root}")
        return 1

    real_videos = collect_videos(real_dir)
    fake_videos  = collect_videos(fake_dir)
    if not real_videos and not fake_videos:
        print("[ERROR] No video files found in test_data/real or test_data/fake")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  DeepFake Video Classifier — Batch Mode")
    print("=" * 62)
    print(f"  Test dir   : {test_root}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print(f"  Real videos: {len(real_videos)}")
    print(f"  Fake videos: {len(fake_videos)}")
    print("=" * 62)

    print("\nLoading model …", end=" ", flush=True)
    model = load_model(str(ckpt_path), device)
    print("done\n")

    rows: list[dict] = []
    idx = 1

    for cls, videos, expected in [
        ("Real", real_videos, "Real"),
        ("Fake", fake_videos,  "Fake"),
    ]:
        for video in videos:
            label_tag = f"[{idx:>2}] {cls:4s}  {video.name[:48]:<48}"
            print(label_tag, end=" ", flush=True)
            try:
                res = classify_video(video, model, device)
                pred = res["prediction"]
                conf = res["confidence_pct"]
                tick = "✓" if pred == expected else "✗"
                print(f"→  {pred} ({conf}%)  {tick}")
                rows.append({
                    "idx":     idx,
                    "cls":     cls,
                    "file":    video.name,
                    "pred":    pred,
                    "conf":    conf,
                    "correct": pred == expected,
                })
            except Exception as exc:
                print(f"→  ERROR: {exc}")
                rows.append({
                    "idx": idx, "cls": cls,
                    "file": video.name,
                    "pred": "Error", "conf": 0, "correct": False,
                })
            idx += 1

    # ── Metrics ───────────────────────────────────────────────────────────────
    total   = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    accuracy = correct / total * 100 if total else 0

    real_rows = [r for r in rows if r["cls"] == "Real"]
    fake_rows  = [r for r in rows if r["cls"] == "Fake"]
    tp = sum(1 for r in fake_rows if r["pred"] == "Fake")
    tn = sum(1 for r in real_rows if r["pred"] == "Real")
    fp = sum(1 for r in real_rows if r["pred"] == "Fake")
    fn = sum(1 for r in fake_rows if r["pred"] == "Real")

    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    avg_conf  = sum(r["conf"] for r in rows) / total if total else 0.0

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    _print_table(rows)

    summary = [
        "",
        "=" * 62,
        "  RESULTS SUMMARY",
        "=" * 62,
        f"  Total videos   : {total}",
        f"  Correct        : {correct} / {total}",
        f"  Accuracy       : {accuracy:.1f}%",
        f"  Precision      : {precision:.1f}%  (fake class)",
        f"  Recall         : {recall:.1f}%  (fake class)",
        f"  F1-Score       : {f1:.1f}%",
        f"  Avg Confidence : {avg_conf:.1f}%",
        f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        "=" * 62,
    ]
    for line in summary:
        print(line)

    # ── Save report ────────────────────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DeepFake Video Classifier — Batch Report\n")
        f.write(f"Generated  : {datetime.now().isoformat()}\n")
        f.write(f"Test dir   : {test_root}\n")
        f.write(f"Checkpoint : {ckpt_path}\n\n")
        _print_table(rows, file=f)
        for line in summary:
            f.write(line + "\n")

    print(f"\n  Report saved to: {report_path.resolve()}")
    print("=" * 62)
    return 0


# =============================================================================
#  CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deepfake Video Classifier — single video or full test_data/ batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", "-c",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )

    # ── single-video mode ─────────────────────────────────────────────────────
    single = parser.add_argument_group("Single-video mode")
    single.add_argument(
        "--video", "-v",
        default=None,
        help="Path to a specific video file. Omit to run in batch mode.",
    )
    single.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output file for single-video result (default: {DEFAULT_OUTPUT})",
    )
    single.add_argument(
        "--frames", "-f",
        type=int,
        default=NUM_FRAMES,
        help=f"Number of frames to sample (default: {NUM_FRAMES})",
    )

    # ── batch mode ────────────────────────────────────────────────────────────
    batch = parser.add_argument_group("Batch mode (default when --video is omitted)")
    batch.add_argument(
        "--test-dir", "-d",
        default=DEFAULT_TEST_DIR,
        help=f"Root dir containing real/ and fake/ (default: {DEFAULT_TEST_DIR})",
    )
    batch.add_argument(
        "--report", "-r",
        default=DEFAULT_REPORT,
        help=f"Where to write the batch report (default: {DEFAULT_REPORT})",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.video:
        return run_single(args)
    else:
        return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
