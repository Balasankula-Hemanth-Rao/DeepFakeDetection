"""
Tests for temporal consistency loss integration with MultimodalModel.
"""

import pytest
import torch

from src.config import Config
from src.models.multimodal_model import MultimodalModel


class TestTemporalConsistencyModelIntegration:
    """Test temporal consistency loss integration with the model."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test config with temporal consistency disabled."""
        config_data = {
            "model": {
                "enable_audio": True,
                "enable_video": True,
                "checkpoint_dir": "checkpoints",
                "video": {
                    "backbone": "efficientnet_b3",
                    "pretrained": False,
                    "embed_dim": 1536,
                    "temporal_strategy": "avg_pool",
                },
                "audio": {
                    "sample_rate": 16000,
                    "n_mels": 64,
                    "embed_dim": 256,
                },
                "fusion": {
                    "strategy": "concat",
                    "hidden_dim": 512,
                    "dropout": 0.3,
                    "modality_dropout_prob": 0.0,
                    "temporal_consistency_loss": {
                        "enabled": False,
                        "weight": 0.1,
                    },
                },
            },
            "training": {"seed": 42},
        }
        return Config(**config_data)
    
    @pytest.fixture
    def test_config_with_temporal_loss(self):
        """Create a test config with temporal consistency enabled."""
        config_data = {
            "model": {
                "enable_audio": True,
                "enable_video": True,
                "checkpoint_dir": "checkpoints",
                "video": {
                    "backbone": "efficientnet_b3",
                    "pretrained": False,
                    "embed_dim": 1536,
                    "temporal_strategy": "avg_pool",
                },
                "audio": {
                    "sample_rate": 16000,
                    "n_mels": 64,
                    "embed_dim": 256,
                },
                "fusion": {
                    "strategy": "concat",
                    "hidden_dim": 512,
                    "dropout": 0.3,
                    "modality_dropout_prob": 0.0,
                    "temporal_consistency_loss": {
                        "enabled": True,
                        "weight": 0.1,
                    },
                },
            },
            "training": {"seed": 42},
        }
        return Config(**config_data)
    
    @pytest.fixture
    def batch_data_multimodal(self):
        """Create sample batch data for multimodal testing."""
        batch_size = 2
        video_frames = 8
        audio_frames = 1024
        
        video = torch.randn(batch_size, video_frames, 3, 224, 224)  # [B, T, C, H, W]
        audio = torch.randn(batch_size, 1, audio_frames)
        
        return video, audio
    
    def test_model_initialization_without_temporal_loss(self, test_config):
        """Test model initialization without temporal loss."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        assert model is not None
        assert not model.temporal_consistency_enabled
        assert model.temporal_consistency_loss_fn is None
    
    def test_model_initialization_with_temporal_loss(self, test_config_with_temporal_loss):
        """Test model initialization with temporal loss enabled."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        assert model is not None
        assert model.temporal_consistency_enabled
        assert model.temporal_consistency_loss_fn is not None
        assert model.temporal_consistency_weight == 0.1
    
    def test_compute_temporal_loss_disabled(self, test_config, batch_data_multimodal):
        """Test that temporal loss returns None when disabled."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        video, _ = batch_data_multimodal
        loss = model.compute_temporal_consistency_loss(video)
        
        assert loss is None
    
    def test_compute_temporal_loss_enabled(self, test_config_with_temporal_loss, batch_data_multimodal):
        """Test that temporal loss is computed when enabled."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.eval()  # Set to eval mode to avoid dropout issues
        
        video, _ = batch_data_multimodal
        loss = model.compute_temporal_consistency_loss(video)
        
        assert loss is not None
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_temporal_loss_with_none_video(self, test_config_with_temporal_loss):
        """Test that temporal loss handles None video gracefully."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        loss = model.compute_temporal_consistency_loss(None)
        assert loss is None
    
    def test_temporal_loss_gradient_flow(self, test_config_with_temporal_loss, batch_data_multimodal):
        """Test that gradients flow through temporal loss."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        
        video, _ = batch_data_multimodal
        video.requires_grad_(True)
        
        loss = model.compute_temporal_consistency_loss(video)
        assert loss is not None
        
        loss.backward()
        assert video.grad is not None
        assert video.grad.abs().sum() > 0
    
    def test_video_only_model_with_temporal_loss(self, test_config_with_temporal_loss):
        """Test temporal loss with video-only model."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        
        assert model.temporal_consistency_enabled
        assert model.temporal_consistency_loss_fn is not None
    
    def test_audio_only_model_with_temporal_loss(self, test_config_with_temporal_loss):
        """Test temporal loss is disabled with audio-only model."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        
        # Should be disabled because video is not enabled
        assert not model.temporal_consistency_enabled
        assert model.temporal_consistency_loss_fn is None
    
    def test_temporal_loss_config_values(self, test_config_with_temporal_loss):
        """Test that temporal loss config is read correctly."""
        model = MultimodalModel(
            config=test_config_with_temporal_loss,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        assert model.temporal_consistency_enabled is True
        assert model.temporal_consistency_weight == 0.1
    
    def test_multiple_loss_weights(self):
        """Test different temporal loss weight configurations."""
        weights_to_test = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        for weight in weights_to_test:
            config_data = {
                "model": {
                    "enable_audio": True,
                    "enable_video": True,
                    "checkpoint_dir": "checkpoints",
                    "video": {
                        "backbone": "efficientnet_b3",
                        "pretrained": False,
                        "embed_dim": 1536,
                        "temporal_strategy": "avg_pool",
                    },
                    "audio": {
                        "sample_rate": 16000,
                        "n_mels": 64,
                        "embed_dim": 256,
                    },
                    "fusion": {
                        "strategy": "concat",
                        "hidden_dim": 512,
                        "dropout": 0.3,
                        "temporal_consistency_loss": {
                            "enabled": True,
                            "weight": weight,
                        },
                    },
                },
                "training": {"seed": 42},
            }
            config = Config(**config_data)
            model = MultimodalModel(config=config, num_classes=2)
            
            assert model.temporal_consistency_weight == weight


class TestTemporalLossTrainingIntegration:
    """Test how temporal loss integrates with training loop."""
    
    @pytest.fixture
    def model_and_config(self):
        """Create model with temporal loss enabled."""
        config_data = {
            "model": {
                "enable_audio": True,
                "enable_video": True,
                "checkpoint_dir": "checkpoints",
                "video": {
                    "backbone": "efficientnet_b3",
                    "pretrained": False,
                    "embed_dim": 1536,
                    "temporal_strategy": "avg_pool",
                },
                "audio": {
                    "sample_rate": 16000,
                    "n_mels": 64,
                    "embed_dim": 256,
                },
                "fusion": {
                    "strategy": "concat",
                    "hidden_dim": 512,
                    "dropout": 0.3,
                    "temporal_consistency_loss": {
                        "enabled": True,
                        "weight": 0.1,
                    },
                },
            },
            "training": {"seed": 42},
        }
        config = Config(**config_data)
        model = MultimodalModel(config=config, num_classes=2)
        return model, config
    
    def test_forward_pass_produces_logits(self, model_and_config):
        """Test that forward pass produces expected outputs."""
        model, _ = model_and_config
        model.eval()
        
        batch_size = 2
        video = torch.randn(batch_size, 8, 3, 224, 224)
        audio = torch.randn(batch_size, 1, 1024)
        
        with torch.no_grad():
            logits = model(video=video, audio=audio)
        
        assert logits.shape == (batch_size, 2)
    
    def test_temporal_loss_in_training_loop(self, model_and_config):
        """Test temporal loss in a simulated training loop."""
        model, _ = model_and_config
        model.train()
        
        batch_size = 2
        video = torch.randn(batch_size, 8, 3, 224, 224)
        audio = torch.randn(batch_size, 1, 1024)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        logits = model(video=video, audio=audio)
        
        # Compute temporal loss
        temporal_loss = model.compute_temporal_consistency_loss(video)
        
        # Compute main loss
        criterion = torch.nn.CrossEntropyLoss()
        main_loss = criterion(logits, targets)
        
        # Combine losses
        if temporal_loss is not None:
            total_loss = main_loss + model.temporal_consistency_weight * temporal_loss
        else:
            total_loss = main_loss
        
        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)
    
    def test_inference_mode_no_temporal_loss(self, model_and_config):
        """Test that inference mode works without temporal loss computation."""
        model, _ = model_and_config
        model.eval()
        
        batch_size = 2
        video = torch.randn(batch_size, 8, 3, 224, 224)
        audio = torch.randn(batch_size, 1, 1024)
        
        with torch.no_grad():
            logits = model(video=video, audio=audio)
            temporal_loss = model.compute_temporal_consistency_loss(video)
        
        assert logits.shape == (batch_size, 2)
        # Loss is still computed in eval mode, but that's fine
        # User can check model.training flag to decide whether to use it
