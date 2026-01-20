"""
Dataset Verification Script

Verifies the integrity and statistics of downloaded datasets.

Usage:
    python scripts/verify_dataset.py --data-root data/deepfake/
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter
import cv2

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_video_file(video_path: Path) -> dict:
    """
    Verify a single video file.
    
    Returns:
        dict: Video metadata (duration, fps, resolution, etc.)
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {'valid': False, 'error': 'Cannot open video'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'valid': True,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'size_mb': video_path.stat().st_size / (1024 * 1024),
        }
    
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def verify_dataset(data_root: Path, check_videos: bool = False):
    """
    Verify dataset structure and statistics.
    
    Args:
        data_root: Root directory of dataset
        check_videos: Whether to verify video files (slow)
    """
    logger.info("=" * 80)
    logger.info("Dataset Verification")
    logger.info("=" * 80)
    
    data_root = Path(data_root)
    
    if not data_root.exists():
        logger.error(f"Dataset directory not found: {data_root}")
        return False
    
    # Check splits
    splits = ['train', 'val', 'test']
    split_stats = {}
    
    for split in splits:
        split_dir = data_root / split
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split}")
            continue
        
        # Count videos and labels
        video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        labels = []
        valid_videos = 0
        invalid_videos = 0
        total_duration = 0
        total_size_mb = 0
        
        logger.info(f"\nVerifying {split} split ({len(video_dirs)} videos)...")
        
        for video_dir in video_dirs:
            video_file = video_dir / 'video.mp4'
            meta_file = video_dir / 'meta.json'
            
            # Check files exist
            if not video_file.exists():
                logger.warning(f"Missing video: {video_dir.name}")
                invalid_videos += 1
                continue
            
            if not meta_file.exists():
                logger.warning(f"Missing metadata: {video_dir.name}")
                invalid_videos += 1
                continue
            
            # Read metadata
            try:
                meta = json.loads(meta_file.read_text())
                labels.append(meta.get('label', -1))
            except Exception as e:
                logger.warning(f"Invalid metadata in {video_dir.name}: {e}")
                invalid_videos += 1
                continue
            
            # Verify video file (optional, slow)
            if check_videos:
                video_info = verify_video_file(video_file)
                
                if not video_info['valid']:
                    logger.warning(f"Invalid video {video_dir.name}: {video_info.get('error')}")
                    invalid_videos += 1
                    continue
                
                total_duration += video_info['duration']
                total_size_mb += video_info['size_mb']
            
            valid_videos += 1
        
        # Compute statistics
        label_counts = Counter(labels)
        
        split_stats[split] = {
            'total': len(video_dirs),
            'valid': valid_videos,
            'invalid': invalid_videos,
            'real': label_counts.get(0, 0),
            'fake': label_counts.get(1, 0),
            'total_duration': total_duration,
            'total_size_mb': total_size_mb,
        }
        
        # Print statistics
        logger.info(f"\n{split.upper()} Split Statistics:")
        logger.info(f"  Total videos: {len(video_dirs)}")
        logger.info(f"  Valid videos: {valid_videos}")
        logger.info(f"  Invalid videos: {invalid_videos}")
        logger.info(f"  Real videos: {label_counts.get(0, 0)}")
        logger.info(f"  Fake videos: {label_counts.get(1, 0)}")
        
        if check_videos and total_duration > 0:
            logger.info(f"  Total duration: {total_duration / 60:.1f} minutes")
            logger.info(f"  Total size: {total_size_mb / 1024:.1f} GB")
            logger.info(f"  Avg duration: {total_duration / valid_videos:.1f} seconds")
    
    # Overall statistics
    logger.info("\n" + "=" * 80)
    logger.info("Overall Dataset Statistics")
    logger.info("=" * 80)
    
    total_videos = sum(s['total'] for s in split_stats.values())
    total_valid = sum(s['valid'] for s in split_stats.values())
    total_real = sum(s['real'] for s in split_stats.values())
    total_fake = sum(s['fake'] for s in split_stats.values())
    
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Valid videos: {total_valid}")
    logger.info(f"Real videos: {total_real} ({total_real/total_valid*100:.1f}%)")
    logger.info(f"Fake videos: {total_fake} ({total_fake/total_valid*100:.1f}%)")
    
    # Check balance
    if total_real > 0 and total_fake > 0:
        balance_ratio = min(total_real, total_fake) / max(total_real, total_fake)
        logger.info(f"Balance ratio: {balance_ratio:.2f}")
        
        if balance_ratio < 0.5:
            logger.warning("⚠️  Dataset is imbalanced! Consider rebalancing.")
        else:
            logger.info("✓ Dataset is reasonably balanced")
    
    # Validation checks
    logger.info("\n" + "=" * 80)
    logger.info("Validation Checks")
    logger.info("=" * 80)
    
    checks_passed = True
    
    # Check 1: All splits present
    if len(split_stats) < 3:
        logger.warning("⚠️  Not all splits (train/val/test) are present")
        checks_passed = False
    else:
        logger.info("✓ All splits present")
    
    # Check 2: Sufficient data
    if total_valid < 100:
        logger.warning("⚠️  Dataset too small (< 100 videos)")
        checks_passed = False
    else:
        logger.info(f"✓ Sufficient data ({total_valid} videos)")
    
    # Check 3: No invalid videos
    total_invalid = sum(s['invalid'] for s in split_stats.values())
    if total_invalid > 0:
        logger.warning(f"⚠️  {total_invalid} invalid videos found")
        checks_passed = False
    else:
        logger.info("✓ No invalid videos")
    
    # Check 4: Both classes present
    if total_real == 0 or total_fake == 0:
        logger.error("❌ Missing real or fake videos!")
        checks_passed = False
    else:
        logger.info("✓ Both classes (real/fake) present")
    
    if checks_passed:
        logger.info("\n✅ Dataset verification PASSED!")
    else:
        logger.info("\n⚠️  Dataset verification completed with warnings")
    
    return checks_passed


def main():
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--check-videos', action='store_true',
                        help='Verify video files (slow)')
    
    args = parser.parse_args()
    
    verify_dataset(Path(args.data_root), args.check_videos)


if __name__ == '__main__':
    main()
