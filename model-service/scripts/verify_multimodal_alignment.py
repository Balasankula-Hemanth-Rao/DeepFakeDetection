"""
Audio-Video Alignment Verification Script

Verifies that extracted audio and video frames are properly synchronized
and aligned for multimodal training. Detects mismatches and provides diagnostics.

Usage:
    python scripts/verify_multimodal_alignment.py \
        --frame-dir data/processed/train \
        --audio-dir data/processed/audio/train \
        --output-report alignment_report.json

    python scripts/verify_multimodal_alignment.py \
        --frame-dir data/processed \
        --audio-dir data/processed/audio \
        --all-splits
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import subprocess

import numpy as np
import torch
import torchaudio
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MultimodalAlignmentVerifier:
    """Verify audio-video alignment in multimodal datasets."""
    
    def __init__(self, frame_rate: float = 3.0, sample_rate: int = 16000):
        """
        Initialize verifier.
        
        Args:
            frame_rate: Frames per second in video extraction (default: 3.0)
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.frame_duration = 1.0 / frame_rate  # Duration per frame in seconds
    
    def get_video_duration_from_frames(self, frame_files: List[Path]) -> Tuple[float, int]:
        """
        Calculate video duration from frame count.
        
        Args:
            frame_files: List of frame file paths
            
        Returns:
            Tuple of (duration_seconds, frame_count)
        """
        frame_count = len(frame_files)
        duration = frame_count * self.frame_duration
        return duration, frame_count

    def get_video_duration_from_file(self, video_path: Path) -> Tuple[float, int]:
        """
        Calculate video duration from video file using OpenCV.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (duration_seconds, frame_count)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return 0.0, 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0.0
                
            cap.release()
            return duration, frame_count
        except Exception as e:
            logger.error(f"Error reading video {video_path}: {e}")
            return 0.0, 0
    
    def get_audio_duration(self, audio_path: Path) -> Tuple[float, int]:
        """
        Get audio duration and sample count.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (duration_seconds, sample_count)
        """
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            n_samples = waveform.shape[1]
            duration = n_samples / self.sample_rate
            
            return duration, n_samples
        except Exception as e:
            logger.error(f"Error reading audio {audio_path}: {e}")
            return 0, 0
    
    def verify_alignment(
        self,
        video_id: str,
        frame_files: List[Path],
        audio_path: Path,
        tolerance: float = 0.5
    ) -> Dict:
        """
        Verify alignment between video and audio.
        
        Args:
            video_id: Video identifier
            frame_files: List of frame file paths
            audio_path: Path to audio file
            tolerance: Maximum time difference in seconds (default: 0.5s)
            
        Returns:
            Alignment report dictionary
        """
        
        if isinstance(frame_files, list):
             # Extracted frames mode
            video_duration, frame_count = self.get_video_duration_from_frames(frame_files)
        else:
             # Direct video file mode
            video_duration, frame_count = self.get_video_duration_from_file(frame_files)
            
        audio_duration, audio_samples = self.get_audio_duration(audio_path)
        
        time_diff = abs(video_duration - audio_duration)
        is_aligned = time_diff <= tolerance
        
        return {
            'video_id': video_id,
            'video_duration': video_duration,
            'frame_count': frame_count,
            'audio_duration': audio_duration,
            'audio_samples': audio_samples,
            'time_difference': time_diff,
            'is_aligned': is_aligned,
            'tolerance': tolerance,
            'status': 'OK' if is_aligned else 'MISMATCH'
        }
    
    def verify_directory(
        self,
        frame_dir: Path,
        audio_dir: Path,
        label: str = None,
        tolerance: float = 0.5,
        max_videos: int = None
    ) -> Dict:
        """
        Verify alignment for all videos in a directory.
        
        Args:
            frame_dir: Directory containing video frames (fake/ or real/ subdirs)
            audio_dir: Directory containing audio files (fake/ or real/ subdirs)
            label: Label name ('fake' or 'real', auto-detected if None)
            tolerance: Maximum time difference in seconds
            max_videos: Maximum videos to check (for quick testing)
            
        Returns:
            Comprehensive alignment report
        """
        frame_dir = Path(frame_dir)
        audio_dir = Path(audio_dir)
        
        # Auto-detect label
        if label is None:
            if 'fake' in str(frame_dir).lower():
                label = 'fake'
            else:
                label = 'real'
        
        report = {
            'label': label,
            'frame_dir': str(frame_dir),
            'audio_dir': str(audio_dir),
            'tolerance_seconds': tolerance,
            'videos': [],
            'summary': {
                'total': 0,
                'aligned': 0,
                'misaligned': 0,
                'missing_audio': 0,
                'missing_frames': 0,
                'avg_time_diff': 0.0,
                'max_time_diff': 0.0
            }
        }
        
        # Group frames by video OR list video files
        video_frames = defaultdict(list)
        
        # Check if we have extracted frames or video files
        has_frames = len(list(frame_dir.glob('*.jpg'))) > 0
        has_videos = len(list(frame_dir.glob('*.mp4'))) > 0
        
        if has_frames:
            logger.info("Detected extracted frames (.jpg)")
            for frame_file in sorted(frame_dir.glob('*.jpg')):
                video_id = frame_file.stem.rsplit('_frame', 1)[0]
                video_frames[video_id].append(frame_file)
        elif has_videos:
            logger.info("Detected video files (.mp4)")
            for video_file in sorted(frame_dir.glob('*.mp4')):
                video_id = video_file.stem
                # Store the video path itself instead of a list of frames
                video_frames[video_id] = video_file
        else:
            logger.warning(f"No .jpg frames or .mp4 videos found in {frame_dir}")
            return report
        
        video_ids = list(video_frames.keys())
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        logger.info(f"Verifying {len(video_ids)} videos from {label}/")
        
        time_diffs = []
        
        for video_id in tqdm(video_ids, desc=f"Verifying {label} videos"):
            frame_files = video_frames[video_id]
            audio_file = audio_dir / f"{video_id}.wav"
            
            # Check for missing files
            if not audio_file.exists():
                # Count frames or just put 0 if it's a video file
                f_count = len(frame_files) if isinstance(frame_files, list) else 0
                report['videos'].append({
                    'video_id': video_id,
                    'status': 'MISSING_AUDIO',
                    'frame_count': f_count
                })
                report['summary']['missing_audio'] += 1
                continue
            
            if not frame_files:
                report['videos'].append({
                    'video_id': video_id,
                    'status': 'MISSING_FRAMES'
                })
                report['summary']['missing_frames'] += 1
                continue
            
            # Verify alignment
            alignment = self.verify_alignment(
                video_id,
                frame_files,
                audio_file,
                tolerance=tolerance
            )
            
            report['videos'].append(alignment)
            time_diffs.append(alignment['time_difference'])
            
            if alignment['is_aligned']:
                report['summary']['aligned'] += 1
            else:
                report['summary']['misaligned'] += 1
            
            report['summary']['total'] += 1
        
        # Calculate summary statistics
        if time_diffs:
            report['summary']['avg_time_diff'] = float(np.mean(time_diffs))
            report['summary']['max_time_diff'] = float(np.max(time_diffs))
        
        return report
    
    def verify_all_splits(
        self,
        data_dir: Path,
        tolerance: float = 0.5,
        max_videos_per_split: int = None
    ) -> Dict:
        """
        Verify alignment across all splits (train/val/test).
        
        Args:
            data_dir: Root data directory
            tolerance: Maximum time difference in seconds
            max_videos_per_split: Max videos per split
            
        Returns:
            Combined report for all splits
        """
        all_reports = {
            'splits': {}
        }
        
        for split_name in ['train', 'val', 'test']:
            split_dir = data_dir / split_name
            audio_dir = data_dir / 'audio' / split_name
            
            if not split_dir.exists():
                logger.warning(f"Split not found: {split_dir}")
                continue
            
            split_reports = {}
            for label in ['fake', 'real']:
                label_dir = split_dir / label
                label_audio_dir = audio_dir / label if audio_dir.exists() else None
                
                if not label_dir.exists():
                    continue
                
                if label_audio_dir and label_audio_dir.exists():
                    report = self.verify_directory(
                        label_dir,
                        label_audio_dir,
                        label=label,
                        tolerance=tolerance,
                        max_videos=max_videos_per_split
                    )
                    split_reports[label] = report
            
            all_reports['splits'][split_name] = split_reports
        
        return all_reports
    
    def print_report(self, report: Dict):
        """Pretty print alignment report."""
        print("\n" + "="*70)
        print(f"ALIGNMENT REPORT: {report.get('label', 'ALL')}")
        print("="*70)
        
        summary = report.get('summary', {})
        print(f"\nSummary:")
        print(f"  Total videos: {summary.get('total', 0)}")
        print(f"  ✓ Aligned: {summary.get('aligned', 0)}")
        print(f"  ✗ Misaligned: {summary.get('misaligned', 0)}")
        print(f"  ⚠ Missing audio: {summary.get('missing_audio', 0)}")
        print(f"  ⚠ Missing frames: {summary.get('missing_frames', 0)}")
        print(f"  Average time diff: {summary.get('avg_time_diff', 0):.3f}s")
        print(f"  Max time diff: {summary.get('max_time_diff', 0):.3f}s")
        
        # Show misaligned videos
        misaligned = [v for v in report.get('videos', []) if v.get('status') != 'OK']
        if misaligned:
            print(f"\nMisaligned videos (top 5):")
            for video in misaligned[:5]:
                print(f"  - {video['video_id']}: "
                      f"{video.get('time_difference', 0):.3f}s diff")


def main():
    parser = argparse.ArgumentParser(
        description='Verify audio-video alignment in multimodal datasets'
    )
    parser.add_argument(
        '--frame-dir',
        type=Path,
        help='Directory containing video frames'
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed'),
        help='Root data directory (for --all-splits)'
    )
    parser.add_argument(
        '--all-splits',
        action='store_true',
        help='Verify all splits (train/val/test)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.5,
        help='Tolerance for time difference in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum videos to verify (for quick testing)'
    )
    parser.add_argument(
        '--output-report',
        type=Path,
        default=None,
        help='Save report to JSON file'
    )
    parser.add_argument(
        '--frame-rate',
        type=float,
        default=3.0,
        help='Frame extraction rate in FPS (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    verifier = MultimodalAlignmentVerifier(frame_rate=args.frame_rate)
    
    # Verify alignment
    if args.all_splits:
        report = verifier.verify_all_splits(
            args.data_dir,
            tolerance=args.tolerance,
            max_videos_per_split=args.max_videos
        )
        
        # Print reports for each split/label
        for split_name, split_reports in report.get('splits', {}).items():
            for label, label_report in split_reports.items():
                verifier.print_report(label_report)
    else:
        if not args.frame_dir or not args.audio_dir:
            parser.error("Provide --frame-dir and --audio-dir, or use --all-splits")
        
        report = verifier.verify_directory(
            args.frame_dir,
            args.audio_dir,
            tolerance=args.tolerance,
            max_videos=args.max_videos
        )
        
        verifier.print_report(report)
    
    # Save report
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ Report saved: {args.output_report}")


if __name__ == '__main__':
    main()
