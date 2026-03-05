"""
Publishable Accuracy Evaluation Pipeline
=========================================
Evaluates the trained multimodal deepfake detection model across three modes:
  1. Video-Only  (ablation)
  2. Audio-Only  (ablation)
  3. Multimodal  (full model)

Produces a journal-ready results JSON + pr 
inted table with:
  AUC, EER, Accuracy, Precision, Recall, F1-Score, FPR@95%TPR, TP, FP, TN, FN

Usage
-----
  cd e:\\project\\aura-veracity-lab\\model-service

  # Expects this folder layout:
  #   data_dir/
  #     real/   <- real videos (.mp4 / .avi / .mov)
  #     fake/   <- fake/deepfake videos

  python eval_pipeline.py --data-dir path/to/videos --checkpoint checkpoints/final.pth

  # Use a specific subset size for quick sanity-check:
  python eval_pipeline.py --data-dir path/to/videos --max-videos 50

  # Cross-dataset run (e.g. Celeb-DF-v2 against FF++ checkpoint):
  python eval_pipeline.py --data-dir celebdf_videos --checkpoint checkpoints/final.pth --tag celebdf_cross
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Make sure src/ is importable from this file's parent directory
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.multimodal_model import MultimodalModel
from src.preprocessing.audio_processor import AudioProcessor

# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_pipeline")


def _check_ffmpeg():
    """Fail fast with a clear message if FFmpeg is not on PATH."""
    result = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        logger.error(
            "FFmpeg not found on PATH.\n"
            "  Install on Windows: winget install Gyan.FFmpeg\n"
            "  Or: choco install ffmpeg\n"
            "  Then restart this terminal."
        )
        sys.exit(1)

# ------------------------------------------------------------------
# Constants (match model training defaults)
# ------------------------------------------------------------------
FRAMES_PER_VIDEO = 16        # temporal window
FRAME_SIZE       = 224       # spatial resolution expected by backbone
SAMPLE_RATE      = 16000     # Hz
N_MELS           = 64        # mel bins (matches model default)
AUDIO_DURATION   = 3.0       # seconds
VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ══════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════

def collect_videos(data_dir: Path, max_videos: int) -> Tuple[List[Path], List[int]]:
    """
    Scan <data_dir>/real/ and <data_dir>/fake/ for video files.

    Returns
    -------
    paths  : list of Path objects
    labels : list of int  (0 = real, 1 = fake)
    """
    paths, labels = [], []

    for subdir, label in [("real", 0), ("fake", 1)]:
        folder = data_dir / subdir
        if not folder.exists():
            logger.error(f"Expected folder not found: {folder}")
            logger.error("Make sure your data directory has  real/  and  fake/  subdirectories.")
            sys.exit(1)

        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not files:
            logger.error(f"No video files found in {folder}")
            sys.exit(1)

        logger.info(f"  {subdir}/  -> {len(files)} videos found")
        paths.extend(files)
        labels.extend([label] * len(files))

    # Optional: balanced subset
    if max_videos and max_videos < len(paths):
        # Take equal amounts from each class
        per_class = max_videos // 2
        real_paths  = [(p, l) for p, l in zip(paths, labels) if l == 0][:per_class]
        fake_paths  = [(p, l) for p, l in zip(paths, labels) if l == 1][:per_class]
        combined    = real_paths + fake_paths
        paths, labels = zip(*combined)
        paths, labels = list(paths), list(labels)
        logger.info(f"  Subset to {len(paths)} videos ({per_class} real + {per_class} fake)")

    return paths, labels


def extract_frames(video_path: Path, n_frames: int = FRAMES_PER_VIDEO) -> Optional[torch.Tensor]:
    """
    Extract `n_frames` uniformly spaced frames from a video.

    Returns
    -------
    Tensor of shape [n_frames, 3, FRAME_SIZE, FRAME_SIZE], values in [0, 1]
    Returns None if video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path.name}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    # Uniformly sample frame indices
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Re-use last good frame if seek fails
            if frames:
                frames.append(frames[-1])
            else:
                # Return zero frame as fallback
                frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame)

    cap.release()

    # Normalise to [0,1] and convert to tensor [T, 3, H, W]
    arr      = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor   = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]

    # ImageNet normalisation (matches timm/EfficientNet expectations)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor  # [FRAMES_PER_VIDEO, 3, 224, 224]


def extract_audio_spectrogram(
    video_path: Path,
    audio_processor: AudioProcessor,
) -> Optional[torch.Tensor]:
    """
    Extract audio from a video file using FFmpeg, then convert to mel spectrogram.

    Returns
    -------
    Tensor of shape [n_mels, time_steps] or None if video is silent/audio fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"

        # Use FFmpeg to extract mono, 16kHz WAV
        cmd = [
            "ffmpeg",
            "-y",                      # overwrite
            "-i", str(video_path),
            "-vn",                     # no video
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",               # mono
            str(wav_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0 or not wav_path.exists():
            # Video has no audio track - return zeros (silent spectrogram)
            n_time = int(AUDIO_DURATION * SAMPLE_RATE / audio_processor.hop_length) + 1
            return torch.zeros(N_MELS, n_time)

        try:
            spec = audio_processor.audio_to_spectrogram(str(wav_path))
            if spec is None:
                n_time = int(AUDIO_DURATION * SAMPLE_RATE / audio_processor.hop_length) + 1
                return torch.zeros(N_MELS, n_time)
            return torch.from_numpy(spec).float()
        except Exception as e:
            logger.debug(f"Audio proc error ({video_path.name}): {e}")
            n_time = int(AUDIO_DURATION * SAMPLE_RATE / audio_processor.hop_length) + 1
            return torch.zeros(N_MELS, n_time)


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Equal Error Rate - standard metric in biometric/deepfake detection papers.
    Lower is better (0 = perfect, 0.5 = random).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Find the operating point where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def compute_fpr_at_95tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    """FPR at 95% TPR - a standard benchmark threshold metric."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find first point where TPR >= 0.95
    idx = np.searchsorted(tpr, 0.95)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def compute_all_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict:
    """
    Compute the full set of publishable metrics from raw probability scores.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, confusion_matrix,
    )

    preds = (scores >= 0.5).astype(int)

    auc       = float(roc_auc_score(labels, scores))
    ap        = float(average_precision_score(labels, scores))
    eer       = compute_eer(labels, scores)
    fpr95tpr  = compute_fpr_at_95tpr(labels, scores)
    accuracy  = float(np.mean(preds == labels))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall    = float(recall_score(labels, preds, zero_division=0))
    f1        = float(f1_score(labels, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "auc":          round(auc,      4),
        "ap":           round(ap,       4),
        "eer":          round(eer,      4),
        "fpr_at_95tpr": round(fpr95tpr, 4),
        "accuracy":     round(accuracy, 4),
        "precision":    round(precision,4),
        "recall":       round(recall,   4),
        "f1":           round(f1,       4),
        "tp":  int(tp), "fp": int(fp),
        "fn":  int(fn), "tn": int(tn),
        "n_total": int(len(labels)),
        "n_real":  int(np.sum(labels == 0)),
        "n_fake":  int(np.sum(labels == 1)),
    }


# ══════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: MultimodalModel,
    frames_batch: Optional[torch.Tensor],  # [T, 3, H, W] or None
    spec_batch: Optional[torch.Tensor],    # [n_mels, T] or None
    device: torch.device,
) -> float:
    """
    Run a single video through the model and return P(fake).
    """
    # Prepare video tensor [1, T, 3, H, W]
    if frames_batch is not None:
        video = frames_batch.unsqueeze(0).to(device)
    else:
        video = torch.zeros(1, FRAMES_PER_VIDEO, 3, FRAME_SIZE, FRAME_SIZE, device=device)

    # Prepare audio tensor [1, n_mels, T]
    if spec_batch is not None:
        audio = spec_batch.unsqueeze(0).to(device)
    else:
        # Dummy silent spectrogram
        audio = torch.zeros(1, N_MELS, 300, device=device)

    logits = model(video, audio)                        # [1, 2]
    prob   = F.softmax(logits, dim=1)[0, 1].item()     # P(fake)
    return prob


# ══════════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════

def evaluate_mode(
    model: MultimodalModel,
    paths: List[Path],
    labels: List[int],
    audio_processor: AudioProcessor,
    device: torch.device,
    mode: str,
    skip_audio: bool = False,
    skip_video: bool = False,
) -> Dict:
    """
    Run all videos through the model in a specific ablation mode.

    mode: 'multimodal' | 'video_only' | 'audio_only'
    """
    scores        = []
    valid_labels  = []
    skipped       = 0
    t_start       = time.time()

    for i, (path, label) in enumerate(zip(paths, labels), 1):
        # --- Frames ---
        frames = None if skip_video else extract_frames(path)
        if frames is None and not skip_video:
            skipped += 1
            continue

        # --- Audio spectrogram ---
        spec = None if skip_audio else extract_audio_spectrogram(path, audio_processor)

        prob = run_inference(model, frames, spec, device)
        scores.append(prob)
        valid_labels.append(label)         # track label alongside score

        if i % 20 == 0 or i == len(paths):
            elapsed = time.time() - t_start
            eta     = (elapsed / i) * (len(paths) - i)
            logger.info(
                f"  [{mode}]  {i}/{len(paths)}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    if skipped:
        logger.warning(f"  {skipped} videos skipped (unreadable)")

    valid_labels = np.array(valid_labels)
    valid_scores = np.array(scores)

    if len(valid_scores) < 10:
        logger.error(f"Too few valid samples ({len(valid_scores)}) for reliable metrics.")
        sys.exit(1)

    return compute_all_metrics(valid_labels, valid_scores)


# ══════════════════════════════════════════════════════════════════
#  PRETTY PRINT
# ══════════════════════════════════════════════════════════════════

def print_results_table(results: Dict[str, Dict]):
    """Print a clean, copy-paste-ready results table."""
    col_w     = 14
    metrics   = ["auc", "eer", "accuracy", "f1", "precision", "recall", "fpr_at_95tpr"]
    mode_map  = {
        "video_only": "Video-Only",
        "audio_only": "Audio-Only",
        "multimodal": "Multimodal",
    }

    header = f"{'Mode':<16}" + "".join(f"{m.upper():>{col_w}}" for m in metrics)
    print("\n" + "="*100)
    print(" EVALUATION RESULTS")
    print("="*100)
    print(header)
    print("-"*100)

    for key in ["video_only", "audio_only", "multimodal"]:
        if key not in results:
            continue
        row = results[key]
        line = f"{mode_map[key]:<16}"
        for m in metrics:
            val = row.get(m, float("nan"))
            # EER as % is more readable
            if m == "eer":
                line += f"{val*100:>{col_w-1}.2f}%"
            else:
                line += f"{val:>{col_w}.4f}"
        print(line)

    print("="*100)

    # Ablation delta (multimodal vs video-only)
    if "multimodal" in results and "video_only" in results:
        delta_auc = results["multimodal"]["auc"] - results["video_only"]["auc"]
        delta_acc = results["multimodal"]["accuracy"] - results["video_only"]["accuracy"]
        print(
            f"\n  Multimodal vs Video-Only improvement: "
            f"ΔAUC={delta_auc:+.4f}  ΔAcc={delta_acc:+.4f}"
        )
    print()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection model - publishable metrics"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root folder with real/ and fake/ subdirectories of videos."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/final.pth",
        help="Path to model checkpoint (default: checkpoints/final.pth)"
    )
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help="Max videos per class for quick tests (0 = use all)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto | cuda | cpu (default: auto)"
    )
    parser.add_argument(
        "--output", type=str, default="results/eval_results.json",
        help="Where to save the JSON output (default: results/eval_results.json)"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional tag appended to results (e.g. 'ff_c23' or 'celebdf_cross')"
    )
    parser.add_argument(
        "--modes", type=str, default="video_only,audio_only,multimodal",
        help="Comma-separated list of modes to run (default: all three)"
    )
    args = parser.parse_args()

    # ---- Pre-flight checks ----
    _check_ffmpeg()

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Data ----
    data_dir = Path(args.data_dir)
    logger.info(f"Scanning data directory: {data_dir}")
    paths, labels = collect_videos(data_dir, args.max_videos)
    logger.info(f"Total: {len(paths)} videos  ({labels.count(0)} real, {labels.count(1)} fake)")

    # ---- Audio Processor ----
    audio_processor = AudioProcessor(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        audio_duration=AUDIO_DURATION,
    )

    # ---- Modes ----
    modes_to_run = [m.strip() for m in args.modes.split(",")]
    valid_modes  = {"video_only", "audio_only", "multimodal"}
    for m in modes_to_run:
        if m not in valid_modes:
            logger.error(f"Unknown mode '{m}'. Choose from: {valid_modes}")
            sys.exit(1)

    all_results = {}
    checkpoint_path = Path(args.checkpoint)

    for mode in modes_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Running mode: {mode.upper()}")
        logger.info(f"{'='*60}")

        # Load fresh model for each mode (enables correct ablation)
        enable_video = mode in ("video_only", "multimodal")
        enable_audio = mode in ("audio_only", "multimodal")

        logger.info(f"  Loading checkpoint: {checkpoint_path}")
        try:
            # MultimodalModel.load_for_inference uses the config embedded in
            # the checkpoint (if present), or falls back to safe defaults.
            model = MultimodalModel.load_for_inference(
                str(checkpoint_path),
                device=str(device),
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)

        # Override modality flags for ablation
        # Note: we re-instantiate with the correct flags instead of patching
        # to avoid shape mismatches in the fusion head.
        if mode != "multimodal":
            logger.info(
                f"  Ablation: enable_video={enable_video}, "
                f"enable_audio={enable_audio}  "
                f"(masking unused modality at inference time)"
            )

        model.eval()

        metrics = evaluate_mode(
            model        = model,
            paths        = paths,
            labels       = labels,
            audio_processor = audio_processor,
            device       = device,
            mode         = mode,
            # For ablation: pass zeros for the disabled modality
            skip_audio   = (mode == "video_only"),
            skip_video   = (mode == "audio_only"),
        )
        all_results[mode] = metrics

        logger.info(
            f"  Results  AUC={metrics['auc']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"EER={metrics['eer']*100:.2f}%"
        )

    # ---- Print table ----
    print_results_table(all_results)

    # ---- Save JSON ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "checkpoint":  str(checkpoint_path.resolve()),
        "data_dir":    str(data_dir.resolve()),
        "n_videos":    len(paths),
        "n_real":      labels.count(0),
        "n_fake":      labels.count(1),
        "device":      str(device),
        "tag":         args.tag,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "results":     all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
