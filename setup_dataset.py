#!/usr/bin/env python
"""
FaceForensics++ Dataset Downloader & Organizer

This script helps you download and organize the FaceForensics++ dataset
for cross-dataset deepfake detection validation.

Requirements:
- Python 3.8+
- Internet connection
- ~5-10 GB disk space (depending on options)

Usage:
    python setup_dataset.py --help
    python setup_dataset.py download --type mini
    python setup_dataset.py organize
    python setup_dataset.py validate
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List
import argparse
import hashlib


class DatasetManager:
    """Manage FaceForensics++ dataset downloads and organization."""
    
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.faceforensics_dir = workspace_dir / 'FaceForensics-master'
        self.model_service_dir = workspace_dir / 'model-service'
    
    def guide_manual_download(self) -> None:
        """
        Show instructions for manual download from official FaceForensics++ repo.
        This is the BEST way to get the data.
        """
        print("\n" + "="*70)
        print("OFFICIAL FaceForensics++ DOWNLOAD INSTRUCTIONS")
        print("="*70)
        
        print("""
The official FaceForensics++ dataset must be downloaded from the official
source with proper Terms of Service acknowledgment.

STEP 1: Get the Official Download Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Visit: https://github.com/ondyari/FaceForensics

Clone the repository:
    git clone https://github.com/ondyari/FaceForensics.git

This gives you the official download.py script.


STEP 2: Download Full Dataset (1-3 hours)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Navigate to FaceForensics directory:
    cd FaceForensics/

Run the official download script:

    # Download ALL manipulated videos (Deepfakes, Face2Face, FaceSwap, etc.)
    python download.py . -d all -c c40 -t videos
    
    # OR download just Deepfakes (easiest for testing)
    python download.py . -d deepfakes -c c40 -t videos
    
    # Download original/real videos
    python download.py . -d original -c c40 -t videos


OPTIONS:
  -d/--deepfake_method  all, deepfakes, face2face, faceswap, neuraltextures
  -c/--compression      c40 (recommended), c23, raw
  -t/--filetype        videos, masks, models
  --server             EU, EU2, CA (pick closest to you)


STEP 3: Copy to Aura Veracity Project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After downloading, copy data to our project:

On Windows PowerShell:
    $sourceDir = "path/to/FaceForensics/manipulated_sequences"
    $destDir = "E:\\major project\\DeepFakeDetection\\model-service\\data"
    Copy-Item -Recurse -Force $sourceDir $destDir


EXPECTED OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After download and copy, your directory structure should be:

model-service/data/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   └── c40/
│   │       └── videos/
│   │           ├── 000_000.mp4
│   │           ├── 000_001.mp4
│   │           └── ... (~1000+ videos)
│   ├── Face2Face/
│   │   └── c40/
│   │       └── videos/ ... (for ablation studies)
│   ├── FaceSwap/
│   │   └── c40/
│   │       └── videos/ ...
│   └── NeuralTextures/
│       └── c40/
│           └── videos/ ...
└── original_sequences/
    └── youtube/
        └── c40/
            └── videos/
                ├── 000.mp4
                ├── 003.mp4
                └── ... (~1000+ videos)


Cross-Dataset Evaluation Support:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Once you have the data, you can run:

1. LOMO (Leave-One-Method-Out) Evaluation
   Train on 3 methods, test on held-out method
   
2. Cross-Dataset Validation
   Train on FaceForensics++ → Test on FakeAVCeleb
   
3. Modality Ablation Studies
   Video-only vs. Audio-only vs. Multimodal

        """)
        
        print("="*70)
        print("NEXT STEPS")
        print("="*70)
        print("""
1. Visit: https://github.com/ondyari/FaceForensics
2. Follow their download instructions
3. Copy downloaded data to: model-service/data/
4. Run cross-dataset evaluation: python evaluate_cross_dataset.py

        """)


def setup_directories(workspace_dir: Path) -> None:
    """Create required directories."""
    dirs_to_create = [
        workspace_dir / 'model-service' / 'data' / 'manipulated_sequences' / 'Deepfakes' / 'c40' / 'videos',
        workspace_dir / 'model-service' / 'data' / 'original_sequences' / 'youtube' / 'c40' / 'videos',
        workspace_dir / 'model-service' / 'checkpoints',
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {dir_path.relative_to(workspace_dir)}")


def check_dataset_status(workspace_dir: Path) -> None:
    """Check what data is currently available."""
    print("\n" + "="*70)
    print("DATASET STATUS")
    print("="*70 + "\n")
    
    # Check Deepfakes
    deepfakes_dir = workspace_dir / 'model-service' / 'data' / 'manipulated_sequences' / 'Deepfakes' / 'c40' / 'videos'
    if deepfakes_dir.exists():
        videos = list(deepfakes_dir.glob('*.mp4'))
        print(f"✓ Deepfakes videos: {len(videos)} files")
    else:
        print(f"✗ Deepfakes videos: NOT YET DOWNLOADED")
    
    # Check original
    original_dir = workspace_dir / 'model-service' / 'data' / 'original_sequences' / 'youtube' / 'c40' / 'videos'
    if original_dir.exists():
        videos = list(original_dir.glob('*.mp4'))
        print(f"✓ Original (real) videos: {len(videos)} files")
    else:
        print(f"✗ Original (real) videos: NOT YET DOWNLOADED")
    
    # Check model checkpoint
    checkpoint = workspace_dir / 'model-service' / 'checkpoints' / 'best_model.pth'
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"✓ Model checkpoint: {size_mb:.1f} MB")
    else:
        print(f"✗ Model checkpoint: NOT YET TRAINED")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Setup FaceForensics++ dataset for cross-dataset validation'
    )
    
    parser.add_argument(
        'command',
        choices=['download', 'organize', 'status', 'guide'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--workspace',
        type=str,
        default='e:\\major project\\DeepFakeDetection',
        help='Workspace directory'
    )
    
    args = parser.parse_args()
    
    workspace_dir = Path(args.workspace)
    
    if not workspace_dir.exists():
        print(f"✗ Workspace not found: {workspace_dir}")
        sys.exit(1)
    
    manager = DatasetManager(workspace_dir)
    
    if args.command == 'guide':
        manager.guide_manual_download()
    
    elif args.command == 'status':
        check_dataset_status(workspace_dir)
    
    elif args.command == 'organize':
        print("Setting up directory structure...")
        setup_directories(workspace_dir)
        print("\n✓ Directories created. Ready for data download.")


if __name__ == '__main__':
    main()
