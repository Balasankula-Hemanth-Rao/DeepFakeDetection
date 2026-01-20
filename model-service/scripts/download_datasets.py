"""
Dataset Download and Preparation Script

This script helps download and prepare deepfake datasets for training:
- FaceForensics++ (primary dataset)
- Celeb-DF (validation dataset)

IMPORTANT: Both datasets require academic access and signed agreements.
This script automates the download process after you obtain credentials.

Usage:
    python scripts/download_datasets.py --dataset faceforensics --output data/
    python scripts/download_datasets.py --dataset celebdf --output data/
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Base class for dataset downloaders."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self):
        """Download dataset."""
        raise NotImplementedError
    
    def verify(self):
        """Verify downloaded dataset."""
        raise NotImplementedError


class FaceForensicsPPDownloader(DatasetDownloader):
    """
    FaceForensics++ dataset downloader.
    
    Dataset Info:
    - URL: https://github.com/ondyari/FaceForensics
    - Size: ~500GB (full), ~38GB (compressed c23)
    - Videos: 1000 real + 4000 fake (4 methods)
    - Access: Requires academic email and signed agreement
    """
    
    
    DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download-FaceForensics.py"
    
    def __init__(self, output_dir: Path, compression: str = 'c23', dataset_type: str = 'all', server: str = 'EU2'):
        super().__init__(output_dir)
        self.compression = compression  # c0 (raw), c23 (light), c40 (heavy)
        self.dataset_type = dataset_type  # all, Deepfakes, Face2Face, FaceSwap, NeuralTextures
        self.server = server
        self.download_script = self.output_dir / 'download-FaceForensics.py'
    
    def download(self):
        """Download FaceForensics++ dataset."""
        logger.info("=" * 80)
        logger.info("FaceForensics++ Dataset Download")
        logger.info("=" * 80)
        
        # Check if credentials are available
        if not self._check_credentials():
            logger.error("FaceForensics++ credentials not found!")
            logger.info("\nTo download FaceForensics++:")
            logger.info("1. Visit: https://github.com/ondyari/FaceForensics")
            logger.info("2. Fill out the access form with your academic email")
            logger.info("3. Wait for approval (usually 1-2 days)")
            logger.info("4. You'll receive a download script and credentials")
            logger.info("5. Save credentials to: .env or pass via --username and --password")
            return False
        
        # Download the official download script
        logger.info("Downloading FaceForensics++ download script...")
        self._download_script()
        
        # Run download script
        logger.info(f"Starting download (compression: {self.compression}, type: {self.dataset_type})...")
        logger.info("This may take several hours depending on your connection...")
        
        cmd = [
            sys.executable,
            str(self.download_script),
            '-d', self.dataset_type,
            '-c', self.compression,
            '-o', str(self.output_dir / 'FaceForensics++'),
            '--server', self.server,
        ]
        
        # Add credentials if available
        username, password = self._get_credentials()
        if username and password:
            cmd.extend(['--username', username, '--password', password])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Download completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _download_script(self):
        """Download the official FaceForensics++ download script."""
        if self.download_script.exists():
            logger.info("Download script already exists, skipping...")
            return
        
        response = requests.get(self.DOWNLOAD_SCRIPT_URL)
        response.raise_for_status()
        
        self.download_script.write_text(response.text)
        logger.info(f"✓ Downloaded script to: {self.download_script}")
    
    def _check_credentials(self) -> bool:
        """Check if credentials are available."""
        username, password = self._get_credentials()
        return username is not None and password is not None
    
    def _get_credentials(self) -> tuple:
        """Get credentials from environment or config."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        username = os.getenv('FACEFORENSICS_USERNAME')
        password = os.getenv('FACEFORENSICS_PASSWORD')
        
        return username, password
    
    def verify(self):
        """Verify downloaded dataset."""
        dataset_dir = self.output_dir / 'FaceForensics++'
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        # Count videos
        video_files = list(dataset_dir.rglob('*.mp4'))
        logger.info(f"Found {len(video_files)} video files")
        
        # Expected counts (approximate)
        expected_counts = {
            'all': 5000,
            'Deepfakes': 1000,
            'Face2Face': 1000,
            'FaceSwap': 1000,
            'NeuralTextures': 1000,
        }
        
        expected = expected_counts.get(self.dataset_type, 0)
        if len(video_files) < expected * 0.9:  # Allow 10% tolerance
            logger.warning(f"Expected ~{expected} videos, found {len(video_files)}")
        else:
            logger.info("✓ Dataset verification passed!")
        
        return True


class CelebDFDownloader(DatasetDownloader):
    """
    Celeb-DF dataset downloader.
    
    Dataset Info:
    - URL: https://github.com/yuezunli/celeb-deepfakeforensics
    - Size: ~15GB
    - Videos: 590 real + 5639 fake
    - Access: Requires signed agreement
    """
    
    GITHUB_URL = "https://github.com/yuezunli/celeb-deepfakeforensics"
    
    def download(self):
        """Download Celeb-DF dataset."""
        logger.info("=" * 80)
        logger.info("Celeb-DF Dataset Download")
        logger.info("=" * 80)
        
        logger.info("\nCeleb-DF requires manual download:")
        logger.info("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
        logger.info("2. Read and sign the agreement")
        logger.info("3. Download links will be provided via Google Drive")
        logger.info("4. Download all parts and extract to:")
        logger.info(f"   {self.output_dir / 'Celeb-DF'}")
        logger.info("\nAfter downloading, run this script with --verify flag")
        
        return False
    
    def verify(self):
        """Verify downloaded dataset."""
        dataset_dir = self.output_dir / 'Celeb-DF'
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        # Count videos
        video_files = list(dataset_dir.rglob('*.mp4'))
        logger.info(f"Found {len(video_files)} video files")
        
        # Expected: ~6229 videos
        if len(video_files) < 5000:
            logger.warning(f"Expected ~6229 videos, found {len(video_files)}")
        else:
            logger.info("✓ Dataset verification passed!")
        
        return True


def organize_dataset(dataset_dir: Path, output_dir: Path, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        dataset_dir: Source dataset directory
        output_dir: Output directory for organized dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    """
    import shutil
    import random
    
    logger.info("Organizing dataset into train/val/test splits...")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all videos
    video_files = list(dataset_dir.rglob('*.mp4'))
    random.shuffle(video_files)
    
    # Calculate split sizes
    total = len(video_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split videos
    train_videos = video_files[:train_size]
    val_videos = video_files[train_size:train_size + val_size]
    test_videos = video_files[train_size + val_size:]
    
    # Copy videos to splits
    for split_name, videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
        logger.info(f"Copying {len(videos)} videos to {split_name}...")
        
        for i, video_path in enumerate(tqdm(videos, desc=split_name)):
            # Create video directory
            video_dir = output_dir / split_name / f'video_{i:05d}'
            video_dir.mkdir(exist_ok=True)
            
            # Copy video
            shutil.copy2(video_path, video_dir / 'video.mp4')
            
            # Create metadata
            # Determine label from path (customize based on dataset structure)
            label = 1 if 'fake' in str(video_path).lower() or 'manipulated' in str(video_path).lower() else 0
            
            meta = {
                'label': label,
                'source': video_path.parent.name,
                'original_path': str(video_path),
            }
            
            (video_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    
    logger.info("✓ Dataset organization complete!")
    logger.info(f"  Train: {len(train_videos)} videos")
    logger.info(f"  Val: {len(val_videos)} videos")
    logger.info(f"  Test: {len(test_videos)} videos")


def main():
    parser = argparse.ArgumentParser(description='Download and prepare deepfake datasets')
    parser.add_argument('--dataset', choices=['faceforensics', 'celebdf', 'both'], required=True,
                        help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/',
                        help='Output directory')
    parser.add_argument('--compression', type=str, default='c23', choices=['c0', 'c23', 'c40'],
                        help='FaceForensics++ compression level')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
                        help='FaceForensics++ dataset type')
    parser.add_argument('--server', type=str, default='EU2',
                        help='Download server (default: EU2)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing dataset')
    parser.add_argument('--organize', action='store_true',
                        help='Organize dataset into train/val/test splits')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.dataset in ['faceforensics', 'both']:
        downloader = FaceForensicsPPDownloader(output_dir, args.compression, args.type, args.server)
        
        if args.verify:
            downloader.verify()
        else:
            success = downloader.download()
            if success and args.organize:
                organize_dataset(
                    output_dir / 'FaceForensics++',
                    output_dir / 'deepfake',
                )
    
    if args.dataset in ['celebdf', 'both']:
        downloader = CelebDFDownloader(output_dir)
        
        if args.verify:
            downloader.verify()
        else:
            downloader.download()
            if args.organize:
                organize_dataset(
                    output_dir / 'Celeb-DF',
                    output_dir / 'deepfake_celebdf',
                )


if __name__ == '__main__':
    main()
