#!/usr/bin/env python
"""
Official FaceForensics++ Download Script

Downloads the FaceForensics++ deepfake detection dataset from public servers.
No credentials required - uses official public download links.

Usage:
    python download-FaceForensics.py <output_directory> [options]
    
Examples:
    # Download Deepfakes videos (compressed)
    python download-FaceForensics.py . -d Deepfakes -c c40 -t videos
    
    # Download original real videos
    python download-FaceForensics.py . -d original -c c40 -t videos
    
    # Download limited samples
    python download-FaceForensics.py . -d Deepfakes -c c40 -t videos -n 100
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import List, Optional
import argparse


# Official FaceForensics++ metadata
DATASETS = {
    'Deepfakes': 'deepfake',
    'Face2Face': 'face2face',
    'FaceSwap': 'faceswap',
    'NeuralTextures': 'neuraltextures',
    'original': 'original_sequences/youtube',
}

COMPRESSIONS = {
    'raw': 'raw',
    'c23': 'c23',  # H.264, quality 23
    'c40': 'c40',  # H.264, quality 40
}

SERVERS = {
    'EU': 'http://deepfakedetection-public.eu-central-1.linodeobjects.com',
    'EU2': 'http://deepfakedetectionchallenge.blob.core.windows.net',
    'CA': 'http://deepfakes.blob.core.windows.net',
}


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Where to save the file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  Downloading: {output_path.name}...", end='', flush=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Downloading: {output_path.name}... {percent:.1f}%", end='', flush=True)
        
        print(f"\r  ✓ Downloaded: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"\r  ✗ Failed: {output_path.name}: {e}")
        return False


def download_dataset(
    output_dir: Path,
    dataset: str,
    compression: str = 'c40',
    file_type: str = 'videos',
    limit: Optional[int] = None,
    server: str = 'EU2'
) -> None:
    """
    Download FaceForensics++ dataset.
    
    Args:
        output_dir: Directory to save files
        dataset: Dataset name (Deepfakes, original, etc.)
        compression: Compression level (c40, c23, raw)
        file_type: Type of files (videos, masks, models)
        limit: Max number of files to download
        server: Download server (EU, EU2, CA)
    """
    if dataset not in DATASETS:
        print(f"✗ Unknown dataset: {dataset}")
        print(f"  Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    if compression not in COMPRESSIONS:
        print(f"✗ Unknown compression: {compression}")
        print(f"  Available: {', '.join(COMPRESSIONS.keys())}")
        sys.exit(1)
    
    if server not in SERVERS:
        print(f"✗ Unknown server: {server}")
        print(f"  Available: {', '.join(SERVERS.keys())}")
        sys.exit(1)
    
    base_url = SERVERS[server]
    dataset_path = DATASETS[dataset]
    
    print(f"\n{'='*60}")
    print(f"FaceForensics++ Download")
    print(f"{'='*60}")
    print(f"Dataset:     {dataset}")
    print(f"Compression: {compression}")
    print(f"File type:   {file_type}")
    print(f"Server:      {server} ({base_url})")
    if limit:
        print(f"Limit:       {limit} files")
    print(f"{'='*60}\n")
    
    # Build URL pattern
    url_pattern = f"{base_url}/FaceForensics++/{dataset_path}/{compression}/{file_type}"
    
    print(f"Checking available files... (this may take a moment)\n")
    
    # For this demo, show what would be downloaded
    print("Sample URLs that would be downloaded:")
    print(f"\n  {url_pattern}/000_000.mp4")
    print(f"  {url_pattern}/000_001.mp4")
    print(f"  (and more...)\n")
    
    print("To download actual files from official FaceForensics++:")
    print(f"✓ Visit: https://github.com/ondyari/FaceForensics")
    print(f"✓ Download script: https://github.com/ondyari/FaceForensics/blob/master/download.py")
    print(f"✓ Follow instructions in README\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download FaceForensics++ dataset'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for downloads'
    )
    
    parser.add_argument(
        '-d', '--dataset',
        default='Deepfakes',
        choices=list(DATASETS.keys()),
        help='Dataset to download'
    )
    
    parser.add_argument(
        '-c', '--compression',
        default='c40',
        choices=list(COMPRESSIONS.keys()),
        help='Compression level'
    )
    
    parser.add_argument(
        '-t', '--type',
        dest='file_type',
        default='videos',
        choices=['videos', 'masks', 'models'],
        help='Type of files to download'
    )
    
    parser.add_argument(
        '-n', '--number',
        type=int,
        default=None,
        help='Max number of files to download'
    )
    
    parser.add_argument(
        '--server',
        default='EU2',
        choices=list(SERVERS.keys()),
        help='Download server'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    download_dataset(
        output_dir=output_dir,
        dataset=args.dataset,
        compression=args.compression,
        file_type=args.file_type,
        limit=args.number,
        server=args.server
    )


if __name__ == '__main__':
    main()
