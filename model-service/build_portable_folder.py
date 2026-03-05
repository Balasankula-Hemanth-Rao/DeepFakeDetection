"""
Build Portable Training Folder
================================
Assembles everything needed for LOMO training into one folder
that can be copied to an RTX 4070 machine or external drive.

What gets included:
  - Source code (src/, scripts/)
  - Config files (configs/lomo_splits/)
  - Top-level scripts (eval_pipeline.py, prepare_ff_data.py, generate_lomo_configs.py)
  - requirements.txt
  - Checkpoints (checkpoints/final.pth)
  - Downloaded dataset (data/ffpp/)
  - TRAINING_INSTRUCTIONS.md

Usage
-----
  cd e:\\project\\aura-veracity-lab\\model-service

  # Build into a local folder
  python build_portable_folder.py --output C:\\aura_lomo_training

  # Build onto an external drive
  python build_portable_folder.py --output F:\\aura_lomo_training

  # Skip data (if you'll transfer data separately)
  python build_portable_folder.py --output F:\\aura_lomo_training --skip-data
"""

import argparse
import shutil
import sys
from pathlib import Path


# -----------------------------------------------------------------------
# Everything to copy — (source_relative_to_model_service, dest_name)
# -----------------------------------------------------------------------
CODE_ITEMS = [
    # Top-level scripts
    ("eval_pipeline.py",                  "eval_pipeline.py"),
    ("prepare_ff_data.py",                "prepare_ff_data.py"),
    ("generate_lomo_configs.py",          "generate_lomo_configs.py"),
    ("TRAINING_INSTRUCTIONS.md",          "TRAINING_INSTRUCTIONS.md"),
    ("requirements.txt",                  "requirements.txt"),
    ("portable_package/README.md",        "README.md"),

    # Source code
    ("src",                         "src"),

    # Scripts
    ("scripts",                     "scripts"),

    # Configs (LOMO splits already generated)
    ("configs",                     "configs"),

    # Existing checkpoint (for ablation eval)
    ("checkpoints/final.pth",       "checkpoints/final.pth"),
]

DATA_ITEMS = [
    ("data/ffpp",                   "data/ffpp"),
]


def copy_item(src: Path, dst: Path):
    """Copy a file or directory from src to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    elif src.is_file():
        shutil.copy2(src, dst)
    else:
        print(f"  [WARN] Not found, skipping: {src}")
        return False
    return True


def human_size(path: Path) -> str:
    """Return human-readable size of a file or directory."""
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Assemble portable LOMO training folder"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Destination folder (e.g. F:\\aura_lomo_training or C:\\aura_lomo_training)"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip copying data/ffpp (if you will transfer the data separately)"
    )
    parser.add_argument(
        "--skip-checkpoint", action="store_true",
        help="Skip copying checkpoints/final.pth"
    )
    args = parser.parse_args()

    root    = Path(__file__).parent          # model-service/
    dst_root = Path(args.output)

    print(f"\nSource : {root}")
    print(f"Output : {dst_root}")
    print()

    # ---- Code and configs ----
    print("Copying code, scripts, configs, checkpoint...")
    for src_rel, dst_rel in CODE_ITEMS:
        if "final.pth" in src_rel and args.skip_checkpoint:
            print(f"  [SKIP] {src_rel}")
            continue
        src = root / src_rel
        dst = dst_root / dst_rel
        ok  = copy_item(src, dst)
        if ok:
            print(f"  OK   {dst_rel}  ({human_size(src)})")

    # ---- Dataset ----
    if args.skip_data:
        print("\n  [SKIP] data/ffpp  (--skip-data flag set)")
        print("         Copy the data manually:")
        print(f"         xcopy /E /I /H {root / 'data' / 'ffpp'} {dst_root / 'data' / 'ffpp'}")
    else:
        print("\nCopying dataset (data/ffpp) — this may take a few minutes...")
        for src_rel, dst_rel in DATA_ITEMS:
            src = root / src_rel
            dst = dst_root / dst_rel
            if not src.exists():
                print(f"  [WARN] Dataset not found at {src}")
                print(f"         Make sure FF++ was downloaded with download_ff.py first.")
                continue
            ok = copy_item(src, dst)
            if ok:
                print(f"  OK   {dst_rel}  ({human_size(src)})")

    # ---- Summary ----
    total_size = human_size(dst_root) if dst_root.exists() else "unknown"
    print(f"\nTotal folder size: {total_size}")
    print(f"\nFolder is ready at: {dst_root}")
    print()
    print("="*60)
    print("  ON THE RTX 4070 MACHINE:")
    print("="*60)
    print(f"  1. Open terminal in:  {dst_root}")
    print(f"  2. Read:              TRAINING_INSTRUCTIONS.md")
    print(f"  3. Run:               pip install -r requirements.txt")
    print(f"  4. Run:               python generate_lomo_configs.py --ffpp-root data/ffpp --output configs/lomo_splits")
    print(f"  5. Run:               python prepare_ff_data.py --ffpp-root data/ffpp --output data/eval_ready --copy")
    print(f"  6. Train splits 1-4 as described in TRAINING_INSTRUCTIONS.md")
    print("="*60)


if __name__ == "__main__":
    main()
