#!/usr/bin/env python3
"""
FakeAVCeleb Dataset Downloader & Organizer

FakeAVCeleb is a benchmark dataset for audio-visual deepfake detection.
This script downloads and organizes the dataset for cross-dataset evaluation.

Dataset Info:
- Source: https://github.com/ICTMCG/FakeAVCeleb
- Type: Audio-Visual (multimodal) deepfakes
- Fake Videos: ~1,000 synthesized by Deepfakes or Face2Face
- Real Videos: ~500 real celebrity videos from YouTube
- Video Resolution: 128×128 (low-res, but good for testing robustness)
- Total Size: ~2-3 GB

Usage:
    python download_fakeavceleb.py download     # Start download
    python download_fakeavceleb.py organize     # Organize into project structure
    python download_fakeavceleb.py status       # Check download status
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import argparse
import shutil


class FakeAVCelebDownloader:
    """Download and organize FakeAVCeleb dataset."""
    
    def __init__(self, workspace_dir: Optional[Path] = None):
        if workspace_dir is None:
            workspace_dir = Path.cwd()
        self.workspace_dir = Path(workspace_dir)
        self.project_root = self.workspace_dir if (self.workspace_dir / 'model-service').exists() else self.workspace_dir.parent
        self.model_service_dir = self.project_root / 'model-service'
        self.data_dir = self.model_service_dir / 'data' / 'fakeavceleb'
        self.download_dir = self.project_root / 'FakeAVCeleb-download'
        
    def show_download_instructions(self):
        """Display step-by-step download instructions."""
        print("\n" + "="*80)
        print("🎬 FakeAVCeleb Dataset Download Instructions")
        print("="*80)
        print("""
FakeAVCeleb is an audio-visual deepfake dataset with ~1,500 videos.
This is IDEAL for cross-dataset evaluation of your FF++ trained model.

┌─ OPTION 1: OFFICIAL GOOGLE DRIVE LINK (Easiest) ────────────────────────────┐
│                                                                                │
│ 1. Visit: https://www.dropbox.com/sh/3ejf7uc0zmhzg6d/AABzaL5kRYkMQK7iNlpkMt0xa
│                                                                                │
│ 2. Download the files:                                                       │
│    - download-aligned-faces.zip          (~2 GB)                            │
│    - metadata.json                       (~100 KB)                          │
│                                                                                │
│ 3. Extract to: FakeAVCeleb-download/                                         │
│    unzip download-aligned-faces.zip -d FakeAVCeleb-download/                │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

┌─ OPTION 2: GITHUB DIRECT (If Dropbox fails) ──────────────────────────────────┐
│                                                                                │
│ Repository: https://github.com/ICTMCG/FakeAVCeleb                            │
│                                                                                │
│ Quick Steps:                                                                 │
│   1. Clone or download: https://github.com/ICTMCG/FakeAVCeleb/archive/refs/heads/main.zip
│   2. Extract to: FakeAVCeleb-download/                                      │
│   3. Navigate to: FakeAVCeleb-download/FakeAVCeleb-main/                     │
│   4. Run download script (requires ffmpeg):                                  │
│      python download.py --phase all --format aligned                        │
│                                                                                │
│ Note: The GitHub download is slower but more reliable.                       │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘

📊 DATASET STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After extraction, the directory should look like:

FakeAVCeleb-download/
├── aligned-faces/                      (Main video directory)
│   ├── aaa_000.mp4                     (Fake video - DeepFakes or Face2Face)
│   ├── aaa_001.mp4
│   ├── ...
│   ├── real_000.mp4                    (Real video - YouTube source)
│   ├── real_001.mp4
│   └── ... (total ~1500 videos)
│
├── metadata.json                        (Video metadata + labels)
│
└── README.md                            (Dataset documentation)


📥 EXPECTED DOWNLOAD SIZE & TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full Dataset (All Videos): 2-3 GB
├── Fake videos: ~1,100 files (~1.8 GB)
└── Real videos: ~500 files (~1.2 GB)

Estimated Download Time: 30-60 minutes (depends on internet speed)

💾 STORAGE REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total needed: 4-5 GB free space
├── Downloaded files: 2-3 GB
├── Extracted files: 2-3 GB (same as downloaded)
└── Buffer: 1 GB for safety


🚀 NEXT STEPS AFTER DOWNLOAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Once downloaded:

Step 1: Verify download
    python download_fakeavceleb.py status

Step 2: Organize to project structure
    python download_fakeavceleb.py organize

Step 3: Run zero-shot evaluation
    python -m pytest model-service/tests/test_cross_dataset_e2e.py -v -k fakeavceleb

Step 4: Generate cross-dataset metrics
    python model-service/src/train.py --eval-only --dataset fakeavceleb


📚 WHAT IS FakeAVCeleb?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper: "FakeAVCeleb: A Novel Dataset for Multimodal Deepfake Detection"
Authors: Jie Li, Yali Song, Jiaying Liu (Peking University)
Published: 2023

Key Features:
✓ Audio-Visual (multimodal) deepfakes - tests model robustness
✓ Two manipulation types: DeepFakes (GAN) and Face2Face (traditional)
✓ Real videos from YouTube celebrities (diverse faces)
✓ Aligned face crops (128×128) - optimized for neural networks
✓ High quality synthesis - modern generation techniques
✓ Metadata includes: fake method, source, duration, quality

Why Use It?
→ Test zero-shot transfer (train on FF++ → evaluate on FakeAVCeleb)
→ Different compression (128×128 aligned) - different artifacts
→ Audio-visual manipulation - tests multimodal robustness
→ Recently published dataset - state-of-the-art baseline


🔗 USEFUL LINKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GitHub: https://github.com/ICTMCG/FakeAVCeleb
Paper: https://arxiv.org/abs/2301.01212
Dropbox (Direct): https://www.dropbox.com/sh/.../download-aligned-faces.zip
Dataset Stats: ~1,500 videos, ~2-3 GB, 128×128 resolution


✅ DOWNLOAD CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- [ ] Download download-aligned-faces.zip (~2 GB) from Dropbox/GitHub
- [ ] Extract to FakeAVCeleb-download/ (creates aligned-faces/)
- [ ] Verify metadata.json exists
- [ ] Run: python download_fakeavceleb.py status
- [ ] Run: python download_fakeavceleb.py organize
- [ ] Verify in: model-service/data/fakeavceleb/
- [ ] Run evaluation tests


🎯 QUICK COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# After manual download & extraction:
python download_fakeavceleb.py status      # Check what you have
python download_fakeavceleb.py organize    # Move to project
python download_fakeavceleb.py validate    # Verify integrity

# Then test model on it:
pytest model-service/tests/test_cross_dataset_e2e.py -v -k fakeavceleb


⏱️  ESTIMATE: 60 minutes download + 5 minutes setup = 65 minutes total
""")

    def download_official(self):
        """Guide to official download - manual step since it requires browser."""
        print("\n✅ To download FakeAVCeleb:")
        print(f"""
1. Visit Dropbox link:
   https://www.dropbox.com/sh/3ejf7uc0zmhzg6d/AABzaL5kRYkMQK7iNlpkMt0xa

2. Download: download-aligned-faces.zip (~2 GB)

3. Extract to: {self.download_dir}/
   mkdir -p {self.download_dir}
   unzip download-aligned-faces.zip -d {self.download_dir}/

4. Then run:
   python download_fakeavceleb.py organize
""")

    def organize_dataset(self):
        """Organize downloaded FakeAVCeleb into project structure."""
        print("\n📁 Organizing FakeAVCeleb dataset...")
        
        # Check if download directory exists
        if not self.download_dir.exists():
            print(f"❌ Download directory not found: {self.download_dir}")
            print("\nPlease download FakeAVCeleb first:")
            self.show_download_instructions()
            return False
        
        # Find aligned-faces directory
        aligned_dir = None
        for item in self.download_dir.rglob('aligned-faces'):
            if item.is_dir():
                aligned_dir = item
                break
        
        if not aligned_dir:
            print(f"❌ Could not find 'aligned-faces' directory in {self.download_dir}")
            print("\nExpected structure:")
            print("  FakeAVCeleb-download/")
            print("  ├── aligned-faces/")
            print("  └── metadata.json")
            return False
        
        # Create project data structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fake_dir = self.data_dir / 'fake'
        real_dir = self.data_dir / 'real'
        fake_dir.mkdir(exist_ok=True)
        real_dir.mkdir(exist_ok=True)
        
        print(f"✓ Created: {fake_dir}")
        print(f"✓ Created: {real_dir}")
        
        # Organize videos (assuming naming convention: real_*.mp4 vs others)
        total_fake = 0
        total_real = 0
        
        for video in aligned_dir.glob('*.mp4'):
            if video.name.startswith('real_'):
                # Copy real videos
                dest = real_dir / video.name
                if not dest.exists():
                    shutil.copy2(video, dest)
                    total_real += 1
            else:
                # Copy fake videos
                dest = fake_dir / video.name
                if not dest.exists():
                    shutil.copy2(video, dest)
                    total_fake += 1
        
        # Copy metadata
        metadata_file = self.download_dir / 'metadata.json'
        for potential_metadata in self.download_dir.rglob('metadata.json'):
            metadata_file = potential_metadata
            break
        
        if metadata_file.exists():
            dest_metadata = self.data_dir / 'metadata.json'
            shutil.copy2(metadata_file, dest_metadata)
            print(f"✓ Copied: metadata.json")
        
        print(f"\n✅ Organization complete:")
        print(f"  Fake videos: {total_fake}")
        print(f"  Real videos: {total_real}")
        print(f"  Location: {self.data_dir}")
        
        return True

    def check_status(self):
        """Check download and organization status."""
        print("\n📊 FakeAVCeleb Status Report")
        print("="*60)
        
        # Check download directory
        has_download = self.download_dir.exists()
        download_size = 0
        if has_download:
            for file in self.download_dir.rglob('*'):
                if file.is_file():
                    download_size += file.stat().st_size
        
        print(f"\n1️⃣ Download Directory: {self.download_dir}")
        if has_download:
            print(f"   ✓ Exists (Size: {download_size / 1e9:.2f} GB)")
        else:
            print(f"   ✗ Not found - run download first")
        
        # Check organized data
        has_data = self.data_dir.exists()
        fake_count = 0
        real_count = 0
        data_size = 0
        
        if has_data:
            for video in (self.data_dir / 'fake').glob('*.mp4'):
                fake_count += 1
                data_size += video.stat().st_size
            for video in (self.data_dir / 'real').glob('*.mp4'):
                real_count += 1
                data_size += video.stat().st_size
        
        print(f"\n2️⃣ Organized Data: {self.data_dir}")
        if has_data and (fake_count > 0 or real_count > 0):
            print(f"   ✓ Fake videos: {fake_count}")
            print(f"   ✓ Real videos: {real_count}")
            print(f"   ✓ Total size: {data_size / 1e9:.2f} GB")
        else:
            print(f"   ✗ Not organized - run 'organize' command")
        
        # Check metadata
        metadata_file = self.data_dir / 'metadata.json'
        print(f"\n3️⃣ Metadata: metadata.json")
        if metadata_file.exists():
            print(f"   ✓ Found")
        else:
            print(f"   ✗ Not found")
        
        print("\n" + "="*60)
        if fake_count > 0 and real_count > 0:
            print("✅ FakeAVCeleb ready for evaluation!")
            print(f"\nNext: Run cross-dataset tests")
            print(f"  pytest model-service/tests/test_cross_dataset_e2e.py -v -k fakeavceleb")
        elif has_download:
            print("⏳ Data downloaded but not organized")
            print(f"\nNext: Run organization")
            print(f"  python download_fakeavceleb.py organize")
        else:
            print("📥 Ready to download FakeAVCeleb")
            print(f"\nNext: Download and extract dataset")
            print(f"  python download_fakeavceleb.py download")
        
        return (fake_count, real_count)

    def validate_dataset(self):
        """Validate dataset integrity."""
        print("\n🔍 Validating FakeAVCeleb dataset...")
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return False
        
        issues = []
        fake_videos = list((self.data_dir / 'fake').glob('*.mp4'))
        real_videos = list((self.data_dir / 'real').glob('*.mp4'))
        
        print(f"\n✓ Fake videos: {len(fake_videos)}")
        print(f"✓ Real videos: {len(real_videos)}")
        
        # Check for corrupted files
        print("\nChecking file integrity...")
        for video_list, label in [(fake_videos, 'Fake'), (real_videos, 'Real')]:
            for video in video_list[:5]:  # Check first 5
                size = video.stat().st_size
                if size < 50000:  # Less than 50KB is likely corrupted
                    issues.append(f"  ⚠️  Small file: {video.name} ({size} bytes)")
        
        # Check metadata
        metadata_file = self.data_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"✓ Metadata valid: {len(metadata)} entries")
            except json.JSONDecodeError:
                issues.append("  ❌ Corrupted metadata.json")
        else:
            issues.append("  ⚠️  metadata.json not found")
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(issue)
            return False
        
        print("\n✅ Validation complete - dataset is ready!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='FakeAVCeleb Dataset Downloader & Organizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_fakeavceleb.py download        Show download instructions
  python download_fakeavceleb.py organize        Organize downloaded files
  python download_fakeavceleb.py status          Check current status
  python download_fakeavceleb.py validate        Validate dataset
        """
    )
    
    parser.add_argument(
        'command',
        choices=['download', 'organize', 'status', 'validate'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    # Determine workspace directory
    workspace = Path.cwd()
    if not (workspace / 'model-service').exists():
        workspace = workspace.parent
    
    downloader = FakeAVCelebDownloader(workspace)
    
    if args.command == 'download':
        downloader.show_download_instructions()
        downloader.download_official()
    
    elif args.command == 'organize':
        downloader.organize_dataset()
    
    elif args.command == 'status':
        downloader.check_status()
    
    elif args.command == 'validate':
        downloader.validate_dataset()


if __name__ == '__main__':
    main()
