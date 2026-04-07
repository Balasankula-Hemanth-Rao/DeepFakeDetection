#!/usr/bin/env python3
"""
FakeAVCeleb Test Data Generator
Creates sample multipedia videos for testing cross-dataset evaluation
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime


def create_test_dataset():
    """Create sample FakeAVCeleb-like data structure for testing"""
    
    workspace = Path.cwd()
    if not (workspace / 'model-service').exists():
        workspace = workspace.parent
    
    data_dir = workspace / 'model-service' / 'data' / 'fakeavceleb'
    fake_dir = data_dir / 'fake'
    real_dir = data_dir / 'real'
    
    # Create directories
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("📁 Creating FakeAVCeleb test data structure...")
    
    metadata = {
        'dataset': 'FakeAVCeleb',
        'split': 'test',
        'created': datetime.now().isoformat(),
        'videos': {}
    }
    
    # Create fake video metadata (simulated)
    print("   Creating fake videos metadata...")
    for i in range(50):  # 50 fake samples for quick testing
        video_name = f"fake_{i:04d}.mp4"
        video_path = fake_dir / video_name
        
        # Create dummy file (100 KB placeholder)
        if not video_path.exists():
            video_path.write_bytes(b'FAKE_VIDEO_DATA' * 7000)
        
        metadata['videos'][video_name] = {
            'label': 0,  # 0 = fake
            'method': 'Deepfakes' if i % 2 == 0 else 'Face2Face',
            'source': f'source_{i}',
            'duration': 5.0,
            'fps': 30,
            'resolution': '128x128',
            'faces': 1
        }
    
    # Create real video metadata (simulated)
    print("   Creating real videos metadata...")
    for i in range(20):  # 20 real samples for quick testing
        video_name = f"real_{i:04d}.mp4"
        video_path = real_dir / video_name
        
        # Create dummy file (100 KB placeholder)
        if not video_path.exists():
            video_path.write_bytes(b'REAL_VIDEO_DATA' * 7000)
        
        metadata['videos'][video_name] = {
            'label': 1,  # 1 = real
            'method': 'Original',
            'source': f'youtube_{i}',
            'duration': 5.0,
            'fps': 30,
            'resolution': '128x128',
            'faces': 1
        }
    
    # Save metadata
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    fake_count = len(list(fake_dir.glob('*.mp4')))
    real_count = len(list(real_dir.glob('*.mp4')))
    
    print("\n✅ Test data created:")
    print(f"   Fake videos: {fake_count}")
    print(f"   Real videos: {real_count}")
    print(f"   Metadata: {metadata_file}")
    print(f"   Total size: {sum(p.stat().st_size for p in data_dir.rglob('*')) / 1e6:.1f} MB")
    
    print("\n📝 This is TEST data (smaller video files)")
    print("   For FULL evaluation, download real FakeAVCeleb dataset:")
    print("   https://github.com/ICTMCG/FakeAVCeleb/releases")
    
    return metadata


if __name__ == '__main__':
    create_test_dataset()
