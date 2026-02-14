"""
Extract Audio from Videos for Multimodal Deepfake Detection

This script extracts audio tracks from video files and organizes them
into train/val/test splits with fake/real labels for multimodal training.

Features:
- Extracts audio at 16kHz sample rate (optimal for speech models)
- Mono audio (1 channel)
- Organized directory structure matching video frames
- Tracks synchronization with extracted frames

Usage:
    python scripts/extract_audio_multimodal.py \
        --video-dir data/FaceForensics++/original_sequences/youtube/raw/videos \
        --output-dir data/processed/audio \
        --sample-rate 16000

    python scripts/extract_audio_multimodal.py \
        --video-dir data/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos \
        --output-dir data/processed/audio \
        --label fake \
        --sample-rate 16000
"""

import argparse
import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

# Add FFmpeg to PATH
ffmpeg_path = r"C:\Users\SAI-RAM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if ffmpeg_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract and process audio from video files."""
    
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        Initialize audio extractor.
        
        Args:
            sample_rate: Target sample rate in Hz (default: 16000 for speech)
            mono: Convert to mono audio (default: True)
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self):
        """Verify ffmpeg is installed."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            logger.info("✓ FFmpeg verified")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ FFmpeg not found. Install with: pip install ffmpeg-python")
            raise RuntimeError("FFmpeg is required for audio extraction")
    
    def extract_audio(self, video_path: Path, output_path: Path) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save WAV audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return False
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command to extract audio
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-q:a', '9',  # Quality (0=best, 9=worst, smaller file)
            '-ac', '1' if self.mono else '2',  # 1=mono, 2=stereo
            '-ar', str(self.sample_rate),  # Sample rate
            '-loglevel', 'error',  # Suppress verbose output
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"Extracted audio: {video_path.name} → {output_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract audio from {video_path.name}: {e.stderr}")
            return False
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
                str(audio_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get duration for {audio_path}: {e}")
            return 0.0


def process_video_directory(
    video_dir: Path,
    output_dir: Path,
    label: str = None,
    sample_rate: int = 16000,
    max_videos: int = None
) -> Dict[str, any]:
    """
    Process all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted audio
        label: Label for audio files ('fake' or 'real', auto-detected if None)
        sample_rate: Target sample rate
        max_videos: Maximum number of videos to process (for testing)
        
    Returns:
        Dictionary with statistics
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return {}
    
    # Detect label from path if not provided
    if label is None:
        if 'manipulated' in str(video_dir) or 'fake' in str(video_dir).lower():
            label = 'fake'
        elif 'original' in str(video_dir) or 'real' in str(video_dir).lower():
            label = 'real'
        else:
            label = 'unknown'
    
    logger.info(f"Processing videos from: {video_dir}")
    logger.info(f"Label: {label}")
    logger.info(f"Output: {output_dir / label}")
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.flv', '.mkv', '.webm'}
    video_files = [
        f for f in video_dir.glob('*') 
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    logger.info(f"Found {len(video_files)} videos")
    
    # Initialize extractor
    extractor = AudioExtractor(sample_rate=sample_rate)
    
    # Process videos
    stats = {
        'label': label,
        'total_videos': len(video_files),
        'successful': 0,
        'failed': 0,
        'total_duration': 0.0,
        'files': []
    }
    
    output_label_dir = output_dir / label
    
    for video_file in tqdm(video_files, desc=f"Extracting {label} audio"):
        audio_file = output_label_dir / f"{video_file.stem}.wav"
        
        if extractor.extract_audio(video_file, audio_file):
            duration = extractor.get_audio_duration(audio_file)
            stats['successful'] += 1
            stats['total_duration'] += duration
            stats['files'].append({
                'video': video_file.name,
                'audio': audio_file.name,
                'duration': duration
            })
        else:
            stats['failed'] += 1
    
    # Log statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Audio Extraction Complete for {label.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"✓ Successful: {stats['successful']}")
    logger.info(f"✗ Failed: {stats['failed']}")
    logger.info(f"⏱  Total duration: {stats['total_duration']/3600:.2f} hours")
    logger.info(f"📁 Output: {output_label_dir}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from videos for multimodal deepfake detection'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        required=True,
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/audio'),
        help='Directory to save extracted audio (default: data/processed/audio)'
    )
    parser.add_argument(
        '--label',
        choices=['real', 'fake'],
        default=None,
        help='Label for audio files (auto-detected if not provided)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum videos to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Process videos
    stats = process_video_directory(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        label=args.label,
        sample_rate=args.sample_rate,
        max_videos=args.max_videos
    )
    
    # Save metadata
    if stats:
        metadata_file = args.output_dir / 'extraction_metadata.json'
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"📋 Metadata saved: {metadata_file}")


if __name__ == '__main__':
    main()
