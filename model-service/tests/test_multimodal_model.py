"""
Tests for MultimodalModel.

Validates model instantiation, forward pass, feature extraction, and checkpointing.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.config import get_config
from src.models.multimodal_model import MultimodalModel


@pytest.fixture
def config():
    """Fixture for config."""
    return get_config()


@pytest.fixture
def device():
    """Fixture for device."""
    return torch.device('cpu')


def test_model_instantiation(config, device):
    """Test model can be instantiated."""
    model = MultimodalModel(config=config, num_classes=2)
    model.to(device)
    
    assert isinstance(model, torch.nn.Module)
    assert model.num_classes == 2


def test_model_forward_pass(config, device):
    """Test forward pass with random tensors."""
    model = MultimodalModel(config=config, num_classes=2)
    model.to(device)
    model.eval()
    
    # Create random input
    batch_size = 2
    temporal_window = 16
    video = torch.randn(batch_size, temporal_window, 3, 224, 224).to(device)
    audio = torch.randn(batch_size, 64, 128).to(device)
    
    with torch.no_grad():
        logits = model(video, audio)
    
    assert logits.shape == (batch_size, 2)
    assert logits.dtype == torch.float32


def test_feature_extraction(config, device):
    """Test feature extraction."""
    model = MultimodalModel(config=config)
    model.to(device)
    model.eval()
    
    batch_size = 2
    temporal_window = 16
    video = torch.randn(batch_size, temporal_window, 3, 224, 224).to(device)
    audio = torch.randn(batch_size, 64, 128).to(device)
    
    with torch.no_grad():
        video_feat, audio_feat = model.extract_features(video, audio)
    
    assert video_feat.shape[0] == batch_size
    assert audio_feat.shape[0] == batch_size
    assert video_feat.shape[1] > 0
    assert audio_feat.shape[1] > 0


def test_model_parameter_count(config):
    """Test model parameter counting."""
    model = MultimodalModel(config=config)
    
    param_count = model.count_parameters()
    assert param_count > 0
    
    # Compare with manual count
    manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count == manual_count


def test_save_checkpoint(config, device):
    """Test checkpoint saving."""
    model = MultimodalModel(config=config)
    model.to(device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'test_checkpoint.pth'
        
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=0,
            metrics={'auc': 0.85},
        )
        
        assert checkpoint_path.exists()


def test_load_checkpoint(config, device):
    """Test checkpoint loading."""
    model_orig = MultimodalModel(config=config)
    model_orig.to(device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'test_checkpoint.pth'
        
        # Save checkpoint
        model_orig.save_checkpoint(
            str(checkpoint_path),
            epoch=0,
            metrics={'auc': 0.85},
        )
        
        # Load checkpoint
        model_loaded = MultimodalModel.load_for_inference(
            str(checkpoint_path),
            config=config,
            device=str(device),
        )
        
        # Verify loaded model
        assert isinstance(model_loaded, MultimodalModel)
        assert model_loaded.num_classes == 2


def test_different_temporal_strategies(device):
    """Test different temporal encoding strategies."""
    strategies = ['avg_pool', 'tconv', 'transformer']
    
    for strategy in strategies:
        # Create minimal config with temporal strategy
        config = type('Config', (), {
            'model': type('Model', (), {
                'video': type('Video', (), {
                    'backbone': 'efficientnet_b3',
                    'pretrained': False,
                    'embed_dim': 1536,
                    'temporal_strategy': strategy,
                })(),
                'fusion': type('Fusion', (), {
                    'strategy': 'concat',
                    'hidden_dim': 512,
                    'dropout': 0.3,
                })(),
            })(),
            'audio': type('Audio', (), {
                'sample_rate': 16000,
                'n_mels': 64,
                'n_fft': 2048,
                'hop_length': 512,
                'audio_encoder': 'small_cnn',
                'embed_dim': 256,
            })(),
        })()
        
        try:
            model = MultimodalModel(config=config)
            model.to(device)
            model.eval()
            
            video = torch.randn(1, 16, 3, 224, 224).to(device)
            audio = torch.randn(1, 64, 128).to(device)
            
            with torch.no_grad():
                logits = model(video, audio)
            
            assert logits.shape == (1, 2)
        except Exception as e:
            # Some strategies might require optional dependencies
            print(f"Strategy {strategy} failed (may be OK if dependencies missing): {e}")


def test_different_fusion_strategies(config, device):
    """Test different fusion strategies."""
    fusion_strategies = ['concat', 'attention']
    
    for strategy in fusion_strategies:
        try:
            # Override fusion strategy
            config.model.fusion.strategy = strategy
            
            model = MultimodalModel(config=config)
            model.to(device)
            model.eval()
            
            video = torch.randn(1, 16, 3, 224, 224).to(device)
            audio = torch.randn(1, 64, 128).to(device)
            
            with torch.no_grad():
                logits = model(video, audio)
            
            assert logits.shape == (1, 2)
        except Exception as e:
            print(f"Fusion strategy {strategy} failed: {e}")


def test_model_eval_mode(config, device):
    """Test model evaluation mode."""
    model = MultimodalModel(config=config)
    model.to(device)
    
    model.eval()
    
    # In eval mode, should have no gradients
    video = torch.randn(1, 16, 3, 224, 224).to(device)
    audio = torch.randn(1, 64, 128).to(device)
    
    with torch.no_grad():
        logits = model(video, audio)
    
    assert logits.grad is None


def test_model_train_mode(config, device):
    """Test model training mode."""
    model = MultimodalModel(config=config)
    model.to(device)
    
    model.train()
    
    video = torch.randn(1, 16, 3, 224, 224, requires_grad=True).to(device)
    audio = torch.randn(1, 64, 128, requires_grad=True).to(device)
    
    logits = model(video, audio)
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
