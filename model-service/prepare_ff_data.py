"""
FaceForensics++ Data Preparation for eval_pipeline.py
=======================================================
Organizes the downloaded FF++ dataset into the flat real/ fake/ structure
that eval_pipeline.py expects.

After running the FF++ download script, your data will look like:
  ffpp_root/
    original_sequences/youtube/c23/videos/*.mp4        (real)
    manipulated_sequences/Deepfakes/c23/videos/*.mp4   (fake)
    manipulated_sequences/Face2Face/c23/videos/*.mp4   (fake)
    manipulated_sequences/FaceSwap/c23/videos/*.mp4    (fake)
    manipulated_sequences/NeuralTextures/c23/videos/*.mp4 (fake)

This script produces:
  output_dir/
    real/    <- symlinks or copies of real videos
    fake/    <- symlinks or copies of all fake videos (or per-method subdirs)

  Plus LOMO test splits:
  output_dir/lomo/
    leave_out_Deepfakes/
      real/
      fake/          <- only Deepfakes videos (unseen method at test time)
    leave_out_Face2Face/
      ...

Usage
-----
  # Step 1 - Download FF++ (c23, videos only):
  python scripts/download_ff.py data/ffpp -d original    -c c23 -t videos
  python scripts/download_ff.py data/ffpp -d Deepfakes   -c c23 -t videos
  python scripts/download_ff.py data/ffpp -d Face2Face   -c c23 -t videos
  python scripts/download_ff.py data/ffpp -d FaceSwap    -c c23 -t videos
  python scripts/download_ff.py data/ffpp -d NeuralTextures -c c23 -t videos

  # Step 2 - Prepare for evaluation:
  python prepare_ff_data.py --ffpp-root data/ffpp --output data/eval_ready

  # Step 3 - Run accuracy evaluation:
  python eval_pipeline.py --data-dir data/eval_ready --tag ff_c23

  # LOMO evaluation (one split at a time):
  python eval_pipeline.py \\
      --data-dir data/eval_ready/lomo/leave_out_Deepfakes \\
      --tag lomo_leave_out_Deepfakes
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------
# FaceForensics++ structure constants (c23 compression, standard for papers)
# -----------------------------------------------------------------------
REAL_SUBPATH    = "original_sequences/youtube/{comp}/videos"
FAKE_METHODS    = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FAKE_SUBPATH    = "manipulated_sequences/{method}/{comp}/videos"
VIDEO_EXTS      = {".mp4", ".avi", ".mov"}

# Official test split: FF++ uses a fixed 720/140/140 train/val/test split
# defined by video IDs 000–719 (train), 720–859 (val), 860–999 (test).
# We use the test set for benchmark reporting (standard for papers).
TEST_ID_RANGE   = range(860, 1000)   # IDs 860–999
VAL_ID_RANGE    = range(720, 860)    # IDs 720–859 (optional)


def collect_test_ids(video_paths: List[Path]) -> List[Path]:
    """
    Filter to the official FF++ test split by video ID.
    FF++ video filenames are like  001_003.mp4  (source_target pairs) or
    001.mp4 (for original videos). We extract the first numeric ID.
    """
    test_videos = []
    for p in video_paths:
        stem = p.stem.split("_")[0]          # e.g. "001_003" -> "001"
        try:
            vid_id = int(stem)
            if vid_id in TEST_ID_RANGE:
                test_videos.append(p)
        except ValueError:
            # Non-numeric filename - include it anyway (non-standard dataset)
            test_videos.append(p)
    return test_videos


def link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    """Create a symlink (fast) or copy (safe) from src -> dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def prepare_flat(
    ffpp_root: Path,
    output_dir: Path,
    compression: str,
    use_symlinks: bool,
    all_splits: bool,
    methods: List[str],
) -> Tuple[int, int]:
    """
    Build the flat  real/  fake/  structure for eval_pipeline.py.
    If all_splits=False, uses only the official test split IDs.

    Returns (n_real, n_fake).
    """
    real_source = ffpp_root / REAL_SUBPATH.format(comp=compression)
    if not real_source.exists():
        print(f"[ERROR] Real videos not found at: {real_source}")
        print("        Did you download the 'original' dataset first?")
        sys.exit(1)

    real_videos = sorted(
        p for p in real_source.iterdir() if p.suffix.lower() in VIDEO_EXTS
    )

    if not all_splits:
        real_videos = collect_test_ids(real_videos)

    print(f"  Real videos : {len(real_videos)}")

    n_real = 0
    for vid in real_videos:
        dst = output_dir / "real" / vid.name
        link_or_copy(vid, dst, use_symlinks)
        n_real += 1

    n_fake = 0
    for method in methods:
        fake_source = ffpp_root / FAKE_SUBPATH.format(method=method, comp=compression)
        if not fake_source.exists():
            print(f"  [WARN] Fake videos for {method} not found at: {fake_source}  (skipping)")
            continue

        fake_videos = sorted(
            p for p in fake_source.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not all_splits:
            fake_videos = collect_test_ids(fake_videos)

        print(f"  {method:<20}: {len(fake_videos)} fake videos")

        for vid in fake_videos:
            # Prefix with method name to avoid filename collisions
            dst_name = f"{method}_{vid.name}"
            dst = output_dir / "fake" / dst_name
            link_or_copy(vid, dst, use_symlinks)
            n_fake += 1

    return n_real, n_fake


def prepare_lomo(
    ffpp_root: Path,
    output_dir: Path,
    compression: str,
    use_symlinks: bool,
    methods: List[str],
) -> None:
    """
    Build one test folder per LOMO split.

    For each held-out method M:
      output_dir/lomo/leave_out_M/
        real/   <- test-split real videos
        fake/   <- ONLY videos from method M (test split)

    This is what goes into eval_pipeline.py for the LOMO table in your paper.
    """
    real_source = ffpp_root / REAL_SUBPATH.format(comp=compression)
    if not real_source.exists():
        print(f"[ERROR] Real videos not found at: {real_source}")
        sys.exit(1)

    all_real = sorted(
        p for p in real_source.iterdir() if p.suffix.lower() in VIDEO_EXTS
    )
    test_real = collect_test_ids(all_real)

    for held_out in methods:
        split_dir    = output_dir / "lomo" / f"leave_out_{held_out}"
        split_real   = split_dir / "real"
        split_fake   = split_dir / "fake"

        # Real videos (same for every split)
        for vid in test_real:
            link_or_copy(vid, split_real / vid.name, use_symlinks)

        # Fake: only the held-out method
        fake_source = ffpp_root / FAKE_SUBPATH.format(
            method=held_out, comp=compression
        )
        if not fake_source.exists():
            print(f"  [WARN] {held_out} not found, skipping LOMO split.")
            continue

        test_fake = collect_test_ids(sorted(
            p for p in fake_source.iterdir() if p.suffix.lower() in VIDEO_EXTS
        ))

        for vid in test_fake:
            link_or_copy(vid, split_fake / f"{held_out}_{vid.name}", use_symlinks)

        print(
            f"  LOMO leave_out_{held_out}: "
            f"{len(test_real)} real  +  {len(test_fake)} {held_out} fake"
        )

    # Save split manifest
    manifest_path = output_dir / "lomo" / "splits_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol":    "Leave-One-Method-Out (LOMO)",
        "compression": compression,
        "test_id_range": [TEST_ID_RANGE.start, TEST_ID_RANGE.stop],
        "splits":      [f"leave_out_{m}" for m in methods],
        "note":        "Each split tests generalization to an unseen manipulation method."
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  LOMO manifest saved to: {manifest_path}")


def print_next_steps(output_dir: Path) -> None:
    print("\n" + "="*65)
    print("  DATA READY — next steps")
    print("="*65)
    print()
    print("  1. FULL BENCHMARK (all fake methods combined):")
    print(f"     python eval_pipeline.py \\")
    print(f"         --data-dir {output_dir} \\")
    print(f"         --tag ff_c23")
    print()
    print("  2. LOMO EVALUATION (one command per held-out method):")
    for method in FAKE_METHODS:
        split = output_dir / "lomo" / f"leave_out_{method}"
        print(f"     python eval_pipeline.py \\")
        print(f"         --data-dir {split} \\")
        print(f"         --tag lomo_{method} \\")
        print(f"         --output results/lomo_{method}.json")
        print()
    print("  Results JSON files -> paste numbers straight into your paper.")
    print("="*65)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FF++ dataset for eval_pipeline.py"
    )
    parser.add_argument(
        "--ffpp-root", type=str, required=True,
        help="Root folder where FF++ was downloaded (contains original_sequences/ etc.)"
    )
    parser.add_argument(
        "--output", type=str, default="data/eval_ready",
        help="Output directory for organized data (default: data/eval_ready)"
    )
    parser.add_argument(
        "--compression", type=str, default="c23",
        choices=["c23", "c40", "raw"],
        help="FF++ compression level to use (default: c23 — standard for papers)"
    )
    parser.add_argument(
        "--methods", type=str,
        default="Deepfakes,Face2Face,FaceSwap,NeuralTextures",
        help="Comma-separated fake methods to include (default: all 4)"
    )
    parser.add_argument(
        "--all-splits", action="store_true",
        help="Use ALL videos (not just test split). Default: test split only."
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking (use if on Windows without symlink permissions)"
    )
    parser.add_argument(
        "--skip-lomo", action="store_true",
        help="Skip creating per-method LOMO split folders"
    )
    args = parser.parse_args()

    ffpp_root  = Path(args.ffpp_root)
    output_dir = Path(args.output)
    methods    = [m.strip() for m in args.methods.split(",")]
    # On Windows, symlinks require admin — default to copy
    use_symlinks = not args.copy and sys.platform != "win32"

    if not ffpp_root.exists():
        print(f"[ERROR] FF++ root not found: {ffpp_root}")
        sys.exit(1)

    print(f"\nFF++ root  : {ffpp_root}")
    print(f"Output dir : {output_dir}")
    print(f"Compression: {args.compression}")
    print(f"Methods    : {methods}")
    print(f"Split      : {'all' if args.all_splits else 'test only (IDs 860-999)'}")
    print(f"File mode  : {'copy' if not use_symlinks else 'symlink'}")
    print()

    # ---- Flat structure ----
    print("Building flat  real/  fake/  structure...")
    n_real, n_fake = prepare_flat(
        ffpp_root, output_dir, args.compression,
        use_symlinks, args.all_splits, methods
    )
    print(f"  Done: {n_real} real  +  {n_fake} fake videos")

    # ---- LOMO splits ----
    if not args.skip_lomo:
        print("\nBuilding LOMO split folders...")
        prepare_lomo(ffpp_root, output_dir, args.compression, use_symlinks, methods)

    print_next_steps(output_dir)


if __name__ == "__main__":
    main()
