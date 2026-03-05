"""
LOMO Split Config Generator
============================
Generates the 4 JSON config files that train_lomo.py and the LOMO evaluator
expect for Leave-One-Method-Out training and evaluation on FaceForensics++.

The format matches what src/datasets/multimodal_lomo_dataset.py reads:
  {
    "split_name":    "lomo_split_1_test_Deepfakes",
    "test_method":   "Deepfakes",
    "train_methods": ["Face2Face", "FaceSwap", "NeuralTextures"],
    "data_dir":      "data/ffpp_processed",
    "compression":   "c23",
    "train_ids":     [...],
    "val_ids":       [...],
    "test_ids":      [...]
  }

Usage
-----
  cd e:\\project\\aura-veracity-lab\\model-service

  python generate_lomo_configs.py --ffpp-root data/ffpp --output configs/lomo_splits

  # Then train each split (on your RTX 4070):
  python src/train_lomo.py --split-config configs/lomo_splits/split_1_test_Deepfakes.json      --output checkpoints/lomo_1/ --epochs 10
  python src/train_lomo.py --split-config configs/lomo_splits/split_2_test_Face2Face.json       --output checkpoints/lomo_2/ --epochs 10
  python src/train_lomo.py --split-config configs/lomo_splits/split_3_test_FaceSwap.json        --output checkpoints/lomo_3/ --epochs 10
  python src/train_lomo.py --split-config configs/lomo_splits/split_4_test_NeuralTextures.json  --output checkpoints/lomo_4/ --epochs 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict


# -----------------------------------------------------------------------
# FaceForensics++ official video ID splits (reproducible across papers)
#   Train : IDs 000-719  (720 videos)
#   Val   : IDs 720-859  (140 videos)
#   Test  : IDs 860-999  (140 videos)
# -----------------------------------------------------------------------
TRAIN_IDS = list(range(0,   720))
VAL_IDS   = list(range(720, 860))
TEST_IDS  = list(range(860, 1000))

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
COMPRESSION  = "c23"


def get_available_ids(video_dir: Path, all_ids: List[int]) -> List[int]:
    """
    Return only the IDs that actually exist in video_dir.
    FF++ filenames for manipulated videos look like  001_002.mp4;
    for originals they're just  001.mp4.
    We match by the first numeric component of the stem.
    """
    if not video_dir.exists():
        return []

    present = set()
    for f in video_dir.glob("*.mp4"):
        stem = f.stem.split("_")[0]
        try:
            present.add(int(stem))
        except ValueError:
            pass

    return sorted(i for i in all_ids if i in present)


def build_split_config(
    split_num:    int,
    test_method:  str,
    ffpp_root:    Path,
    compression:  str = COMPRESSION,
) -> Dict:
    """
    Build a single LOMO split configuration dictionary.

    Parameters
    ----------
    split_num   : 1-4
    test_method : the method held out for testing (e.g. "Deepfakes")
    ffpp_root   : root of the downloaded FF++ dataset
    compression : "c23" (default, standard for papers)
    """
    train_methods = [m for m in FAKE_METHODS if m != test_method]

    # Paths inside ffpp_root
    original_dir = ffpp_root / "original_sequences" / "youtube" / compression / "videos"

    # Verify original data exists
    if not original_dir.exists():
        print(f"  [WARN] Original video dir not found: {original_dir}")
        print(f"         Run:  python scripts/download_ff.py data/ffpp -d original -c {compression} -t videos --server EU2")

    # Collect available IDs per split from the REAL source
    # (fake sources may have different ID counts but we use real IDs as anchor)
    train_ids_real = get_available_ids(original_dir, TRAIN_IDS)
    val_ids_real   = get_available_ids(original_dir, VAL_IDS)
    test_ids_real  = get_available_ids(original_dir, TEST_IDS)

    # Fallback: if data not yet downloaded, use full theoretical ranges
    if not train_ids_real:
        print(f"  [INFO] Original videos not yet downloaded — using theoretical ID ranges.")
        train_ids_real = TRAIN_IDS
        val_ids_real   = VAL_IDS
        test_ids_real  = TEST_IDS

    # Collect fake video IDs for each training method (for reference)
    train_method_ids = {}
    for method in train_methods:
        method_dir = (
            ffpp_root / "manipulated_sequences" / method / compression / "videos"
        )
        ids = get_available_ids(method_dir, TRAIN_IDS + VAL_IDS)
        train_method_ids[method] = ids if ids else TRAIN_IDS + VAL_IDS

    # Collect test (held-out) method video IDs
    test_method_dir = (
        ffpp_root / "manipulated_sequences" / test_method / compression / "videos"
    )
    test_method_ids = get_available_ids(test_method_dir, TEST_IDS)
    if not test_method_ids:
        test_method_ids = TEST_IDS

    config = {
        # ---- Identity ----
        "split_name":    f"lomo_split_{split_num}_test_{test_method}",
        "split_number":  split_num,
        "protocol":      "Leave-One-Method-Out (LOMO)",

        # ---- Method assignment ----
        "test_method":   test_method,
        "train_methods": train_methods,

        # ---- Data paths ----
        # train_lomo.py reads data_dir to find per-method subdirectories
        "data_dir":      str(ffpp_root.resolve()),
        "compression":   compression,

        # ---- Official ID splits ----
        # These are passed to the dataloader so it knows which videos to load
        "train_ids": train_ids_real,
        "val_ids":   val_ids_real,
        "test_ids":  test_ids_real,

        # ---- Per-method ID lists (for dataset construction) ----
        "train_method_ids": train_method_ids,
        "test_method_ids":  test_method_ids,

        # ---- Paths for convenience ----
        "paths": {
            "original_videos": str(original_dir),
            "test_method_videos": str(test_method_dir),
            "train_method_videos": {
                m: str(ffpp_root / "manipulated_sequences" / m / compression / "videos")
                for m in train_methods
            },
        },

        # ---- Note for paper ----
        "paper_note": (
            f"Model is trained on FF++ methods {train_methods} "
            f"and tested on {test_method} (unseen during training). "
            f"Train split: IDs 0-719, Val: 720-859, Test: 860-999. "
            f"Compression: {compression}."
        ),
    }

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate LOMO split JSON configs for FaceForensics++"
    )
    parser.add_argument(
        "--ffpp-root", type=str, required=True,
        help="Root folder where FF++ was downloaded "
             "(e.g. data/ffpp — must contain original_sequences/ and manipulated_sequences/)"
    )
    parser.add_argument(
        "--output", type=str, default="configs/lomo_splits",
        help="Directory to write the 4 JSON config files (default: configs/lomo_splits)"
    )
    parser.add_argument(
        "--compression", type=str, default="c23",
        choices=["c23", "c40", "raw"],
        help="FF++ compression level (default: c23 — standard for papers)"
    )
    args = parser.parse_args()

    ffpp_root  = Path(args.ffpp_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ffpp_root.exists():
        # Data not downloaded yet — configs are still useful as templates
        print(f"[WARN] FF++ root not found: {ffpp_root}")
        print(f"       Generating configs with theoretical ID ranges (update after download).")

    print(f"\nGenerating 4 LOMO split configs -> {output_dir}\n")

    generated = []
    for split_num, test_method in enumerate(FAKE_METHODS, start=1):
        config = build_split_config(
            split_num   = split_num,
            test_method = test_method,
            ffpp_root   = ffpp_root,
            compression = args.compression,
        )

        filename = f"split_{split_num}_test_{test_method}.json"
        out_path = output_dir / filename

        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        train_n  = len(config["train_ids"])
        val_n    = len(config["val_ids"])
        test_n   = len(config["test_ids"])

        print(
            f"  Split {split_num}  hold-out={test_method:<18} "
            f"train={train_n}  val={val_n}  test={test_n}  -> {filename}"
        )
        generated.append(out_path)

    # Summary manifest
    manifest = {
        "protocol":     "LOMO — Leave-One-Method-Out",
        "ffpp_root":    str(ffpp_root.resolve()),
        "compression":  args.compression,
        "splits":       [str(p) for p in generated],
        "train_ids":    f"0-719  ({len(TRAIN_IDS)} videos)",
        "val_ids":      f"720-859 ({len(VAL_IDS)} videos)",
        "test_ids":     f"860-999 ({len(TEST_IDS)} videos)",
        "usage": {
            "train_command": (
                "python src/train_lomo.py "
                "--split-config configs/lomo_splits/split_N_test_METHOD.json "
                "--output checkpoints/lomo_N/ --epochs 10"
            ),
            "eval_command": (
                "python eval_pipeline.py "
                "--data-dir data/eval_ready/lomo/leave_out_METHOD "
                "--checkpoint checkpoints/lomo_N/best.pth "
                "--tag lomo_METHOD"
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest saved: {manifest_path}")
    print("\n" + "="*65)
    print("  NEXT STEPS")
    print("="*65)
    print()
    print("  1. Start FF++ download (if not already running):")
    print("     python scripts/download_ff.py data/ffpp -d original      -c c23 -t videos --server EU2")
    print("     python scripts/download_ff.py data/ffpp -d Deepfakes      -c c23 -t videos --server EU2")
    print("     python scripts/download_ff.py data/ffpp -d Face2Face      -c c23 -t videos --server EU2")
    print("     python scripts/download_ff.py data/ffpp -d FaceSwap       -c c23 -t videos --server EU2")
    print("     python scripts/download_ff.py data/ffpp -d NeuralTextures -c c23 -t videos --server EU2")
    print()
    print("  2. Organize data:")
    print("     python prepare_ff_data.py --ffpp-root data/ffpp --output data/eval_ready --copy")
    print()
    print("  3. Train 4 LOMO splits on RTX 4070:")
    for i, m in enumerate(FAKE_METHODS, 1):
        print(
            f"     python src/train_lomo.py "
            f"--split-config configs/lomo_splits/split_{i}_test_{m}.json "
            f"--output checkpoints/lomo_{i}/ --epochs 10"
        )
    print()
    print("  4. Evaluate (paper results):")
    for i, m in enumerate(FAKE_METHODS, 1):
        print(
            f"     python eval_pipeline.py "
            f"--data-dir data/eval_ready/lomo/leave_out_{m} "
            f"--checkpoint checkpoints/lomo_{i}/best.pth "
            f"--tag lomo_{m} "
            f"--output results/lomo_{m}.json"
        )
    print("="*65)


if __name__ == "__main__":
    main()
