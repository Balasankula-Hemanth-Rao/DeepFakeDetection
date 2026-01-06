"""
Frame Extraction Module

Extracts frames from video files at a consistent 3 FPS using ffmpeg,
computes checksums, and generates metadata for reproducibility.

Usage:
    python extract_frames.py --input path/to/video.mp4 --output path/to/output/
"""

import argparse
import json
import logging
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_ffmpeg_installed() -> bool:
    """
    Check if ffmpeg is installed and accessible.

    Returns:
        bool: True if ffmpeg is available, False otherwise.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        str: Hexadecimal SHA256 checksum.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_frames(
    video_path: Path, output_dir: Path, fps: float = 3.0
) -> Tuple[int, str]:
    """
    Extract frames from a video file using ffmpeg at specified FPS.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (default: 3.0).

    Returns:
        Tuple of (frame_count, ffmpeg_output).

    Raises:
        subprocess.CalledProcessError: If ffmpeg extraction fails.
        FileNotFoundError: If video file does not exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created frames directory: {frames_dir}")

    # Prepare ffmpeg command
    frame_pattern = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",  # Quality (2 = high quality for JPEG)
        frame_pattern,
    ]

    logger.info(f"Extracting frames at {fps} FPS from: {video_path}")
    logger.debug(f"ffmpeg command: {' '.join(cmd)}")

    # Run ffmpeg
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg extraction failed: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    # Count extracted frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frame_count = len(frame_files)
    logger.info(f"Successfully extracted {frame_count} frames")

    return frame_count, result.stderr


def compute_frame_checksums(frames_dir: Path) -> Dict[str, str]:
    """
    Compute SHA256 checksums for all extracted frames.

    Args:
        frames_dir: Directory containing extracted frames.

    Returns:
        Dict mapping frame filename to SHA256 checksum.
    """
    checksums = {}
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))

    logger.info(f"Computing checksums for {len(frame_files)} frames...")

    for i, frame_path in enumerate(frame_files, 1):
        checksum = compute_sha256(frame_path)
        checksums[frame_path.name] = checksum

        if i % 50 == 0 or i == len(frame_files):
            logger.debug(f"Checksummed {i}/{len(frame_files)} frames")

    logger.info(f"Completed checksums for all {len(frame_files)} frames")
    return checksums


def generate_metadata(
    video_path: Path,
    frame_count: int,
    checksums: Dict[str, str],
    output_dir: Path,
) -> Dict:
    """
    Generate metadata JSON containing video info and frame checksums.

    Args:
        video_path: Path to original video file.
        frame_count: Total number of extracted frames.
        checksums: Dictionary of frame filenames to SHA256 checksums.
        output_dir: Output directory for metadata file.

    Returns:
        Dict containing metadata.
    """
    metadata = {
        "video_name": video_path.name,
        "video_path": str(video_path.absolute()),
        "num_frames": frame_count,
        "fps": 3.0,
        "frames": [
            {"filename": filename, "checksum": checksums[filename]}
            for filename in sorted(checksums.keys())
        ],
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_path}")
    return metadata


def main():
    """Main entry point for frame extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video at 3 FPS with checksums.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_frames.py --input video.mp4 --output ./output/
  python extract_frames.py --input /path/to/video.mp4 --output /path/to/output/ --verbose
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for frames and metadata.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Convert to Path objects
    video_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        # Check prerequisites
        if not check_ffmpeg_installed():
            logger.error("ffmpeg is not installed or not in PATH")
            logger.error("Install ffmpeg: https://ffmpeg.org/download.html")
            sys.exit(1)

        logger.info("=" * 70)
        logger.info("Frame Extraction Started")
        logger.info("=" * 70)

        # Extract frames
        frame_count, ffmpeg_output = extract_frames(video_path, output_dir)

        # Compute checksums
        frames_dir = output_dir / "frames"
        checksums = compute_frame_checksums(frames_dir)

        # Generate metadata
        metadata = generate_metadata(video_path, frame_count, checksums, output_dir)

        logger.info("=" * 70)
        logger.info("Frame Extraction Completed Successfully")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info(f"Frames saved to: {frames_dir.absolute()}")
        logger.info(f"Metadata saved to: {output_dir / 'metadata.json'}")
        logger.info(f"Total frames extracted: {frame_count}")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg execution failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
