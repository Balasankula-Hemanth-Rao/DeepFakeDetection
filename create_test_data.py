"""
create_test_data.py
===================
Creates a test_data/ directory with real/ and fake/ subfolders by
copying ~10 sample videos from each class out of the FakeAVCeleb v1.2 dataset
(or a legacy FaceForensics++ / flat real-fake layout).

FakeAVCeleb v1.2 layout (primary / auto-detected)
---------------------------------------------------
  FakeAVCeleb_v1.2/
    RealVideo-RealAudio/<race>/<gender>/<id>/*.mp4   ← REAL (category A)
    FakeVideo-FakeAudio/<race>/<gender>/<id>/*.mp4   ← FAKE (category D)
    FakeVideo-RealAudio/<race>/<gender>/<id>/*.mp4   ← FAKE (category C)
    RealVideo-FakeAudio/<race>/<gender>/<id>/*.mp4   ← FAKE (category B)

Legacy fallback layouts
------------------------
1. FaceForensics++ / DeepFakeDetection (FF++) layout:
     <data_root>/
       manipulated_sequences/DeepFakeDetection/c23/videos/*.mp4   ← fake
       original_sequences/actors/c23/videos/*.mp4                 ← real

2. Flat layout with real/ and fake/ folders anywhere under <data_root>:
     <data_root>/
       real/*.mp4
       fake/*.mp4

Usage
------
  python create_test_data.py
  python create_test_data.py --data-root "E:/major project/DeepFakeDetection/FakeAVCeleb_v1.2"
  python create_test_data.py --data-root "E:/path/to/dataset" --count 10 --seed 42
  python create_test_data.py --output test_data --count 10

Arguments
---------
  --data-root   Root directory of the dataset (default: auto-detect FakeAVCeleb_v1.2)
  --output      Output directory name/path (default: test_data)
  --count       Number of videos per class (default: 10)
  --seed        Random seed for reproducibility (default: 42)
  --fake-cats   Comma-separated fake categories to pool from (default: all fake categories)
                Choices: FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio
"""

from __future__ import annotations
import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("create_test_data")

# ── Constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# FakeAVCeleb v1.2 ─ folder names
FAKEAVCELEB_REAL_DIR = "RealVideo-RealAudio"          # category A  → real
FAKEAVCELEB_FAKE_DIRS = [                              # categories B/C/D → fake
    "FakeVideo-FakeAudio",   # D – fake video + fake audio
    "FakeVideo-RealAudio",   # C – fake video + real audio
    "RealVideo-FakeAudio",   # B – real video + fake audio
]

# FaceForensics++ / DeepFakeDetection candidate sub-paths (legacy fallback)
FF_FAKE_CANDIDATES = [
    ("manipulated_sequences", "DeepFakeDetection", "c23", "videos"),
    ("manipulated_sequences", "DeepFakeDetection", "c40", "videos"),
    ("manipulated_sequences", "DeepFakeDetection", "raw", "videos"),
    ("manipulated_sequences", "Deepfakes",         "c23", "videos"),
    ("manipulated_sequences", "Deepfakes",         "c40", "videos"),
    ("manipulated_sequences", "FaceSwap",          "c23", "videos"),
    ("manipulated_sequences", "Face2Face",         "c23", "videos"),
    ("manipulated_sequences", "NeuralTextures",    "c23", "videos"),
    ("fake",),   # flat layout: <root>/fake/
]

FF_REAL_CANDIDATES = [
    ("original_sequences", "actors",  "c23", "videos"),
    ("original_sequences", "actors",  "c40", "videos"),
    ("original_sequences", "youtube", "c23", "videos"),
    ("original_sequences", "youtube", "c40", "videos"),
    ("real",),   # flat layout: <root>/real/
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_videos(directory: Path) -> list[Path]:
    """Return all video files under *directory* (recursive)."""
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def find_videos_in_candidates(data_root: Path, candidates: list[tuple]) -> list[Path]:
    """
    Try each candidate sub-path under data_root.
    Return the first non-empty list of videos found.
    """
    for parts in candidates:
        candidate_dir = data_root.joinpath(*parts)
        if candidate_dir.is_dir():
            videos = collect_videos(candidate_dir)
            if videos:
                log.info("Found %d video(s) in: %s", len(videos), candidate_dir)
                return videos
            else:
                log.debug("Directory exists but is empty: %s", candidate_dir)
    return []


# ── FakeAVCeleb v1.2 detection ─────────────────────────────────────────────────

def is_fakeavceleb(path: Path) -> bool:
    """Return True if *path* looks like a FakeAVCeleb v1.2 root."""
    return (path / FAKEAVCELEB_REAL_DIR).is_dir() and any(
        (path / d).is_dir() for d in FAKEAVCELEB_FAKE_DIRS
    )


def collect_fakeavceleb_real(data_root: Path) -> list[Path]:
    """Collect real videos from FakeAVCeleb_v1.2/RealVideo-RealAudio/."""
    real_dir = data_root / FAKEAVCELEB_REAL_DIR
    videos = collect_videos(real_dir)
    log.info(
        "FakeAVCeleb | REAL  (%s): %d video(s)",
        FAKEAVCELEB_REAL_DIR, len(videos)
    )
    return videos


def collect_fakeavceleb_fake(data_root: Path, categories: list[str]) -> list[Path]:
    """
    Pool fake videos from the requested fake category folders.
    *categories* should be a subset of FAKEAVCELEB_FAKE_DIRS.
    """
    all_fake: list[Path] = []
    for cat in categories:
        cat_dir = data_root / cat
        if not cat_dir.is_dir():
            log.warning("FakeAVCeleb | category folder not found: %s", cat_dir)
            continue
        videos = collect_videos(cat_dir)
        log.info("FakeAVCeleb | FAKE  (%s): %d video(s)", cat, len(videos))
        all_fake.extend(videos)
    return sorted(set(all_fake))


# ── Auto-detect ────────────────────────────────────────────────────────────────

def auto_detect_data_root() -> Path | None:
    """
    Try to locate the dataset automatically.
    Priority: FakeAVCeleb_v1.2 next to this script → legacy paths.
    """
    # Primary: FakeAVCeleb v1.2 sitting beside the script
    fakeavceleb_default = SCRIPT_DIR / "FakeAVCeleb_v1.2"
    if is_fakeavceleb(fakeavceleb_default):
        log.info("Auto-detected FakeAVCeleb v1.2 dataset: %s", fakeavceleb_default)
        return fakeavceleb_default

    # Legacy locations
    legacy_candidates = [
        SCRIPT_DIR / "model-service" / "data" / "deepfake",
        SCRIPT_DIR / "model-service" / "data",
        SCRIPT_DIR / "data" / "deepfake",
        SCRIPT_DIR / "data",
        SCRIPT_DIR / "dataset",
    ]
    for path in legacy_candidates:
        if path.is_dir():
            log.info("Auto-detected legacy data root: %s", path)
            return path

    return None


# ── Sampling & copying ─────────────────────────────────────────────────────────

def sample_videos(videos: list[Path], count: int, seed: int) -> list[Path]:
    """Return a random sample of *count* videos (or all if fewer available)."""
    rng = random.Random(seed)
    if len(videos) <= count:
        log.warning(
            "Only %d video(s) available (requested %d) — using all.",
            len(videos), count,
        )
        return videos.copy()
    return rng.sample(videos, count)


def copy_videos(videos: list[Path], dest_dir: Path) -> None:
    """Copy each video in *videos* to *dest_dir*, preserving filenames."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in videos:
        dst = dest_dir / src.name
        counter = 1
        while dst.exists():
            dst = dest_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        log.info(
            "  Copying  %-55s  →  %s/%s",
            src.name, dest_dir.name, dst.name,
        )
        shutil.copy2(src, dst)


def print_summary(output_dir: Path, n_real: int, n_fake: int) -> None:
    log.info("=" * 60)
    log.info("test_data creation complete!")
    log.info("  Output  : %s", output_dir.resolve())
    log.info("  real/   : %d video(s) copied", n_real)
    log.info("  fake/   : %d video(s) copied", n_fake)
    log.info("=" * 60)
    log.info("Directory layout:")
    log.info("  %s/", output_dir.name)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    if real_dir.is_dir():
        log.info("  ├── real/")
        for p in sorted(real_dir.iterdir()):
            log.info("  │   └── %s", p.name)
    if fake_dir.is_dir():
        log.info("  └── fake/")
        for p in sorted(fake_dir.iterdir()):
            log.info("      └── %s", p.name)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create test_data/real and test_data/fake by sampling from "
            "FakeAVCeleb v1.2 (or a legacy FaceForensics++/flat layout)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root", "-d",
        type=str,
        default=None,
        help=(
            "Root directory of the dataset. "
            "Auto-detects FakeAVCeleb_v1.2/ next to this script if not specified."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(SCRIPT_DIR / "test_data"),
        help="Output directory (default: test_data/ next to this script).",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of videos to copy per class (default: 10).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--fake-cats",
        type=str,
        default=None,
        help=(
            "Comma-separated list of FakeAVCeleb fake categories to pool from. "
            "Choices: FakeVideo-FakeAudio, FakeVideo-RealAudio, RealVideo-FakeAudio. "
            "Defaults to all three."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)

    # ── Locate data root ───────────────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root).resolve()
        if not data_root.is_dir():
            log.error("Specified --data-root does not exist: %s", data_root)
            return 1
    else:
        data_root = auto_detect_data_root()
        if data_root is None:
            log.error(
                "Could not auto-detect the dataset directory.\n"
                "  Please specify it explicitly:\n"
                "    python create_test_data.py --data-root <path_to_dataset>\n\n"
                "  Expected locations (in priority order):\n"
                "    %s/FakeAVCeleb_v1.2          (FakeAVCeleb v1.2)\n"
                "    %s/model-service/data/deepfake\n"
                "    %s/data\n"
                "  or a folder containing  real/  and  fake/  subdirectories.",
                SCRIPT_DIR, SCRIPT_DIR, SCRIPT_DIR,
            )
            return 1

    log.info("=" * 60)
    log.info("  DeepFake Detection — Test Data Creator")
    log.info("=" * 60)
    log.info("  Dataset root : %s", data_root)
    log.info("  Output dir   : %s", output_dir.resolve())
    log.info("  Count        : %d per class", args.count)
    log.info("  Seed         : %d", args.seed)
    log.info("=" * 60)

    # ── Determine source strategy ─────────────────────────────────────────────
    fakeavceleb_mode = is_fakeavceleb(data_root)

    if fakeavceleb_mode:
        log.info("Dataset type : FakeAVCeleb v1.2")

        # Parse --fake-cats
        if args.fake_cats:
            requested = [c.strip() for c in args.fake_cats.split(",")]
            invalid = [c for c in requested if c not in FAKEAVCELEB_FAKE_DIRS]
            if invalid:
                log.error(
                    "Unknown --fake-cats value(s): %s\n"
                    "  Valid choices: %s",
                    ", ".join(invalid),
                    ", ".join(FAKEAVCELEB_FAKE_DIRS),
                )
                return 1
            fake_categories = requested
        else:
            fake_categories = FAKEAVCELEB_FAKE_DIRS

        log.info("  Fake cats  : %s", ", ".join(fake_categories))
        log.info("=" * 60)

        # ── Find videos ───────────────────────────────────────────────────────
        log.info("Searching for REAL videos (FakeAVCeleb: %s) …", FAKEAVCELEB_REAL_DIR)
        real_videos = collect_fakeavceleb_real(data_root)
        if not real_videos:
            log.error(
                "No real videos found in: %s/%s",
                data_root, FAKEAVCELEB_REAL_DIR,
            )
            return 1

        log.info("Searching for FAKE videos (FakeAVCeleb: %s) …", ", ".join(fake_categories))
        fake_videos = collect_fakeavceleb_fake(data_root, fake_categories)
        if not fake_videos:
            log.error(
                "No fake videos found in the selected FakeAVCeleb categories: %s",
                ", ".join(fake_categories),
            )
            return 1

    else:
        # Legacy FF++ / flat layout
        log.info("Dataset type : FaceForensics++ / flat layout (legacy fallback)")
        log.info("=" * 60)

        log.info("Searching for FAKE videos …")
        fake_videos = find_videos_in_candidates(data_root, FF_FAKE_CANDIDATES)
        if not fake_videos:
            log.error(
                "No fake videos found under: %s\n"
                "Check that the dataset is downloaded and the folder structure matches\n"
                "the FaceForensics++/DeepFakeDetection layout or a flat real/fake layout.",
                data_root,
            )
            return 1

        log.info("Searching for REAL videos …")
        real_videos = find_videos_in_candidates(data_root, FF_REAL_CANDIDATES)
        if not real_videos:
            log.error(
                "No real videos found under: %s\n"
                "Check that the dataset is downloaded and the folder structure is correct.",
                data_root,
            )
            return 1

    # ── Sample ────────────────────────────────────────────────────────────────
    fake_sample = sample_videos(fake_videos, args.count, seed=args.seed)
    real_sample = sample_videos(real_videos, args.count, seed=args.seed + 1)

    # ── Copy ──────────────────────────────────────────────────────────────────
    log.info("\nCopying FAKE videos  (%d) …", len(fake_sample))
    copy_videos(fake_sample, output_dir / "fake")

    log.info("Copying REAL videos  (%d) …", len(real_sample))
    copy_videos(real_sample, output_dir / "real")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(output_dir, n_real=len(real_sample), n_fake=len(fake_sample))
    return 0


if __name__ == "__main__":
    sys.exit(main())
