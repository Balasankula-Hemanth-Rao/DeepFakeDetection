"""
FaceForensics Dataset Preprocessing Script (OpenCV Version)

This script processes the downloaded FaceForensics dataset by:
1. Extracting frames from all videos at 3 FPS using OpenCV (cv2)
2. Organizing frames into train/val/test splits (70/15/15)
3. Creating balanced fake/real distributions
4. Generating metadata for reproducibility

Usage:
    python scripts/preprocess_faceforensics_cv2.py --input data/ --output data/processed/
    python scripts/preprocess_faceforensics_cv2.py --input data/ --output data/processed/ --max-videos 10 --debug
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames_from_video_cv2(
    video_path: Path, output_dir: Path, fps: float = 3.0
) -> Tuple[int, str]:
    """
    Extract frames from a video file using OpenCV at specified FPS.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 3.0)
        
    Returns:
        Tuple of (frame_count, status_message)
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create frames directory
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    logger.debug(f"Video FPS: {video_fps}, extracting every {frame_interval} frames")
    
    frame_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_idx % frame_interval == 0:
            frame_filename = frames_dir / f'frame_{frame_count + 1:05d}.jpg'
            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    logger.debug(f"Extracted {frame_count} frames from {video_path.name}")
    
    return frame_count, f"Extracted {frame_count} frames"


def find_videos(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find all fake and real videos in the FaceForensics dataset.
    
    Args:
        data_dir: Root data directory containing FaceForensics dataset
        
    Returns:
        Tuple of (fake_videos, real_videos) as lists of Paths
    """
    logger.info("Scanning for videos...")
    
    # Find fake videos (Deepfakes)
    fake_dir = data_dir / 'manipulated_sequences' / 'Deepfakes' / 'c40' / 'videos'
    fake_videos = []
    if fake_dir.exists():
        fake_videos = sorted(list(fake_dir.glob('*.mp4')))
        logger.info(f"Found {len(fake_videos)} fake videos in {fake_dir}")
    else:
        logger.warning(f"Fake videos directory not found: {fake_dir}")
    
    # Find real videos (original)
    real_dir = data_dir / 'original_sequences' / 'youtube' / 'c40' / 'videos'
    real_videos = []
    if real_dir.exists():
        real_videos = sorted(list(real_dir.glob('*.mp4')))
        logger.info(f"Found {len(real_videos)} real videos in {real_dir}")
    else:
        logger.warning(f"Real videos directory not found: {real_dir}")
    
    return fake_videos, real_videos


def create_splits(
    fake_videos: List[Path],
    real_videos: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        fake_videos: List of fake video paths
        real_videos: List of real video paths
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with splits: {'train': {'fake': [...], 'real': [...]}, ...}
    """
    random.seed(seed)
    
    # Shuffle videos
    fake_shuffled = fake_videos.copy()
    real_shuffled = real_videos.copy()
    random.shuffle(fake_shuffled)
    random.shuffle(real_shuffled)
    
    # Calculate split sizes
    n_fake = len(fake_shuffled)
    n_real = len(real_shuffled)
    
    fake_train_size = int(n_fake * train_ratio)
    fake_val_size = int(n_fake * val_ratio)
    
    real_train_size = int(n_real * train_ratio)
    real_val_size = int(n_real * val_ratio)
    
    # Create splits
    splits = {
        'train': {
            'fake': fake_shuffled[:fake_train_size],
            'real': real_shuffled[:real_train_size]
        },
        'val': {
            'fake': fake_shuffled[fake_train_size:fake_train_size + fake_val_size],
            'real': real_shuffled[real_train_size:real_train_size + real_val_size]
        },
        'test': {
            'fake': fake_shuffled[fake_train_size + fake_val_size:],
            'real': real_shuffled[real_train_size + real_val_size:]
        }
    }
    
    # Log split statistics
    logger.info("Dataset splits created:")
    for split_name, split_data in splits.items():
        n_fake_split = len(split_data['fake'])
        n_real_split = len(split_data['real'])
        total = n_fake_split + n_real_split
        logger.info(f"  {split_name}: {total} videos ({n_fake_split} fake, {n_real_split} real)")
    
    return splits


def process_video(
    video_path: Path,
    output_dir: Path,
    split_name: str,
    class_name: str,
    video_idx: int,
    fps: float = 3.0
) -> Dict:
    """
    Process a single video: extract frames and organize them.
    
    Args:
        video_path: Path to video file
        output_dir: Root output directory
        split_name: Split name ('train', 'val', or 'test')
        class_name: Class name ('fake' or 'real')
        video_idx: Video index for naming
        fps: Frames per second to extract
        
    Returns:
        Dictionary with processing metadata
    """
    # Create temporary directory for frame extraction
    temp_dir = output_dir / 'temp' / f'{split_name}_{class_name}_{video_idx:04d}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract frames
        frame_count, status_msg = extract_frames_from_video_cv2(video_path, temp_dir, fps=fps)
        
        # Move frames to final location with proper naming
        frames_dir = temp_dir / 'frames'
        target_dir = output_dir / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted(frames_dir.glob('frame_*.jpg'))
        moved_frames = []
        
        for frame_file in frame_files:
            # New filename: video_0000_frame_00001.jpg
            new_name = f'video_{video_idx:04d}_{frame_file.name}'
            target_path = target_dir / new_name
            shutil.move(str(frame_file), str(target_path))
            moved_frames.append(new_name)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        return {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'split': split_name,
            'class': class_name,
            'frame_count': frame_count,
            'frames': moved_frames,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to process {video_path}: {e}")
        # Clean up temp directory on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'split': split_name,
            'class': class_name,
            'status': 'failed',
            'error': str(e)
        }


def process_all_videos(
    splits: Dict[str, Dict[str, List[Path]]],
    output_dir: Path,
    fps: float = 3.0,
    max_videos: int = None
) -> List[Dict]:
    """
    Process all videos in all splits.
    
    Args:
        splits: Dictionary of splits with video lists
        output_dir: Output directory for processed frames
        fps: Frames per second to extract
        max_videos: Maximum videos to process (for debugging)
        
    Returns:
        List of processing metadata dictionaries
    """
    metadata = []
    video_idx = 0
    total_videos = sum(
        len(split_data['fake']) + len(split_data['real'])
        for split_data in splits.values()
    )
    
    if max_videos:
        total_videos = min(total_videos, max_videos)
    
    logger.info(f"Processing {total_videos} videos...")
    
    with tqdm(total=total_videos, desc="Processing videos") as pbar:
        for split_name, split_data in splits.items():
            for class_name in ['fake', 'real']:
                videos = split_data[class_name]
                
                for video_path in videos:
                    if max_videos and video_idx >= max_videos:
                        logger.info(f"Reached max_videos limit: {max_videos}")
                        return metadata
                    
                    result = process_video(
                        video_path=video_path,
                        output_dir=output_dir,
                        split_name=split_name,
                        class_name=class_name,
                        video_idx=video_idx,
                        fps=fps
                    )
                    
                    metadata.append(result)
                    video_idx += 1
                    pbar.update(1)
                    
                    # Log progress every 50 videos
                    if video_idx % 50 == 0:
                        successful = sum(1 for m in metadata if m['status'] == 'success')
                        logger.info(f"Processed {video_idx} videos ({successful} successful)")
    
    return metadata


def save_metadata(metadata: List[Dict], output_dir: Path):
    """
    Save processing metadata to JSON file.
    
    Args:
        metadata: List of processing metadata dictionaries
        output_dir: Output directory
    """
    metadata_path = output_dir / 'preprocessing_metadata.json'
    
    # Calculate statistics
    total = len(metadata)
    successful = sum(1 for m in metadata if m['status'] == 'success')
    failed = total - successful
    total_frames = sum(m.get('frame_count', 0) for m in metadata if m['status'] == 'success')
    
    summary = {
        'total_videos': total,
        'successful': successful,
        'failed': failed,
        'total_frames': total_frames,
        'videos': metadata
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Summary: {successful}/{total} videos processed successfully, {total_frames} frames extracted")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess FaceForensics dataset for training (OpenCV version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos
  python scripts/preprocess_faceforensics_cv2.py --input data/ --output data/processed/
  
  # Debug mode (process only 10 videos)
  python scripts/preprocess_faceforensics_cv2.py --input data/ --output data/processed/ --max-videos 10 --debug
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing FaceForensics dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed frames'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=3.0,
        help='Frames per second to extract (default: 3.0)'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (for debugging)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splits (default: 42)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (verbose logging)'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    try:
        logger.info("=" * 70)
        logger.info("FaceForensics Preprocessing Started (OpenCV)")
        logger.info("=" * 70)
        
        # Find videos
        fake_videos, real_videos = find_videos(input_dir)
        
        if not fake_videos and not real_videos:
            logger.error("No videos found! Check input directory structure.")
            sys.exit(1)
        
        # Create splits
        splits = create_splits(
            fake_videos=fake_videos,
            real_videos=real_videos,
            seed=args.seed
        )
        
        # Process videos
        metadata = process_all_videos(
            splits=splits,
            output_dir=output_dir,
            fps=args.fps,
            max_videos=args.max_videos
        )
        
        # Save metadata
        save_metadata(metadata, output_dir)
        
        logger.info("=" * 70)
        logger.info("Preprocessing Completed Successfully")
        logger.info("=" * 70)
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info(f"Processed {len(metadata)} videos")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
