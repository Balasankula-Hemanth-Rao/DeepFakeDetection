#!/usr/bin/env python3
"""
FakeAVCeleb Dataset Downloader
Downloads from Dropbox mirror or GitHub
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm
import json


class DownloadProgressBar(urllib.request.FancyHTTPHandler):
    """Simple progress bar for downloads"""
    pass


def download_file(url, destination, filename):
    """Download file with progress bar"""
    filepath = Path(destination) / filename
    
    print(f"\n📥 Downloading: {filename}")
    print(f"   URL: {url}")
    print(f"   Size: ~2 GB (estimated)")
    print(f"   Destination: {filepath}")
    
    class ProgressBar(urllib.request.FancyHTTPHandler):
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) / total_size)
                bar_length = 40
                filled = int(bar_length * downloaded / total_size)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{bar}] {percent:.1f}% ({downloaded / 1e9:.2f}/{total_size / 1e9:.2f} GB)", end='\r')
        return progress_hook
    
    try:
        # For Dropbox: use direct download link
        if 'dropbox' in url:
            # Convert sharing link to direct download
            if '?dl=0' in url:
                url = url.replace('?dl=0', '?dl=1')
        
        print("\n⏳ This may take 30-60 minutes depending on your connection...")
        print("   (You can pause/resume with Ctrl+C - the file will be saved)\n")
        
        urllib.request.urlretrieve(
            url, 
            filepath,
            reporthook=ProgressBar().progress_hook(0, 0, 0)
        )
        
        print(f"\n✅ Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nManual Download Instructions:")
        print(f"1. Visit: {url}")
        print(f"2. Download to: {filepath}")
        print(f"3. Then run: python download_fakeavceleb.py organize")
        return None


def main():
    workspace = Path.cwd()
    if not (workspace / 'model-service').exists():
        workspace = workspace.parent
    
    download_dir = workspace / 'FakeAVCeleb-download'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🎬 FakeAVCeleb Dataset Downloader")
    print("="*80)
    
    # Dropbox mirror (official)
    dropbox_url = "https://www.dropbox.com/sh/3ejf7uc0zmhzg6d/AABzaL5kRYkMQK7iNlpkMt0xa/download-aligned-faces.zip?dl=1"
    
    print("\n📊 Dataset Information:")
    print("  - Name: FakeAVCeleb (Aligned Faces)")
    print("  - Type: Audio-Visual Deepfakes")
    print("  - Size: ~2-3 GB")
    print("  - Videos: ~1,600 (1,100 fake + 500 real)")
    print("  - Resolution: 128×128 aligned faces")
    print("  - Download Time: 30-60 minutes")
    
    print("\n" + "="*80)
    print("⚠️  MANUAL DOWNLOAD REQUIRED")
    print("="*80)
    
    print(f"""
The dataset must be downloaded manually due to authentication requirements.

STEP 1: Click the link below (or copy to browser):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   {dropbox_url.split('?')[0]}

STEP 2: Download the file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   File: download-aligned-faces.zip
   Size: ~2 GB
   Destination: {download_dir}/

STEP 3: Extract the file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   PowerShell:
   Expand-Archive -Path "{download_dir}/download-aligned-faces.zip" -DestinationPath "{download_dir}/" -Force

   Linux/Mac:
   unzip "{download_dir}/download-aligned-faces.zip" -d "{download_dir}/"

STEP 4: Organize into project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   cd {workspace}
   python download_fakeavceleb.py organize

STEP 5: Verify download
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   python download_fakeavceleb.py status

STEP 6: Run evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   cd {workspace}/model-service
   pytest tests/test_cross_dataset_e2e.py -v -k fakeavceleb

""")
    
    print("\n💾 Total Storage Needed: 5 GB")
    print("   - Download: ~2 GB")
    print("   - Extracted: ~2 GB")
    print("   - Buffer: ~1 GB")
    
    print("\n⏱️  Estimated Time: 60-90 minutes total")
    print("   - Download: 30-60 min")
    print("   - Extract: 5 min")
    print("   - Organize: 5 min")
    print("   - Evaluate: 15-30 min")
    
    print("\n" + "="*80)
    print("✅ Ready to download!")
    print("="*80)
    
    # Create status file
    status_file = download_dir / 'DOWNLOAD_STATUS.json'
    status = {
        'download_url': dropbox_url.split('?')[0],
        'destination': str(download_dir),
        'expected_files': ['download-aligned-faces.zip', 'metadata.json'],
        'size_gb': 2,
        'status': 'Ready for download'
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"\n📝 Status saved to: {status_file}")


if __name__ == '__main__':
    main()
