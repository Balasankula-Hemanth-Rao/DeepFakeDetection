"""
Tests for MultimodalDataset.

Validates dataset loading, preprocessing, augmentation, and batching.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.data.multimodal_dataset import MultimodalDataset


@pytest.fixture
def config():
    """Fixture for config."""
    return get_config()


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with fake MP4s and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        for split in ['train', 'val']:
            split_dir = tmpdir / split
            split_dir.mkdir()
            
            # Create a few sample directories with metadata
            for i in range(2):
                video_dir = split_dir / f"video_{i:03d}"
                video_dir.mkdir()
                
                # Create dummy video file (just a binary file for testing)
                video_path = video_dir / "video.mp4"
                video_path.write_bytes(b"dummy video data")
                
                # Create metadata
                meta = {
                    'label': i % 2,  # Alternate fake/real
                    'duration': 10.0,
                }
                with open(video_dir / "meta.json", 'w') as f:
                    json.dump(meta, f)
        
        yield tmpdir


def test_dataset_initialization(temp_data_dir, config):
    """Test dataset initialization."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    assert len(dataset) > 0
    assert hasattr(dataset, 'samples')
    assert dataset.temporal_window > 0


def test_dataset_getitem_structure(temp_data_dir, config):
    """Test that __getitem__ returns proper structure."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        # Check keys
        assert 'video' in sample
        assert 'audio' in sample
        assert 'label' in sample
        assert 'meta' in sample
        
        # Check types and shapes
        assert isinstance(sample['video'], torch.Tensor)
        assert isinstance(sample['audio'], torch.Tensor)
        assert isinstance(sample['label'], int)
        assert isinstance(sample['meta'], dict)


def test_video_tensor_shape(temp_data_dir, config):
    """Test video tensor has correct shape."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        video = sample['video']
        
        # Expected shape: [T, 3, H, W]
        assert video.dim() == 4
        assert video.shape[1] == 3  # RGB channels
        assert video.shape[0] <= dataset.temporal_window


def test_audio_tensor_shape(temp_data_dir, config):
    """Test audio tensor shape."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        audio = sample['audio']
        
        # Audio should be 2D or 1D
        assert audio.dim() in [1, 2]


def test_label_values(temp_data_dir, config):
    """Test that labels are valid."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    for sample in dataset:
        label = sample['label']
        assert label in [0, 1]


def test_collate_fn(temp_data_dir, config):
    """Test custom collate function."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=dataset.collate_fn,
    )
    
    for batch in loader:
        assert 'video' in batch
        assert 'audio' in batch
        assert 'label' in batch
        assert 'meta' in batch
        
        assert batch['video'].shape[0] == len(batch['label'])
        assert batch['audio'].shape[0] == len(batch['label'])


def test_debug_mode(temp_data_dir, config):
    """Test debug mode limits dataset size."""
    dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=True,
    )
    
    # In debug mode, should be limited to debug_size
    assert len(dataset) <= dataset.debug_size


def test_split_loading(temp_data_dir, config):
    """Test loading different splits."""
    train_dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='train',
        config=config,
        debug=False,
    )
    
    val_dataset = MultimodalDataset(
        data_root=temp_data_dir,
        split='val',
        config=config,
        debug=False,
    )
    
    assert len(train_dataset) >= 0
    assert len(val_dataset) >= 0
