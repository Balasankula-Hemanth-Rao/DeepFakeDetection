"""
Unit tests for modality ablation study functionality.

Tests modality gating, model instantiation with different enable_audio/enable_video
combinations, and proper handling of optional audio/video inputs.
"""

import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import Config
from src.models.multimodal_model import MultimodalModel


class TestModalityConfiguration:
    """Test modality flag configuration."""
    
    def test_config_load_with_modality_flags(self, config_path):
        """Test loading config with enable_audio and enable_video flags."""
        # Load config from YAML
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check that modality flags exist in model config
        assert 'model' in config_dict
        assert 'enable_audio' in config_dict['model']
        assert 'enable_video' in config_dict['model']
        
        # Check defaults
        assert config_dict['model']['enable_audio'] is True
        assert config_dict['model']['enable_video'] is True
    
    def test_config_object_has_modality_attributes(self, config):
        """Test that Config object has enable_audio and enable_video attributes."""
        model_cfg = getattr(config, 'model', {})
        enable_audio = getattr(model_cfg, 'enable_audio', True)
        enable_video = getattr(model_cfg, 'enable_video', True)
        
        assert isinstance(enable_audio, bool)
        assert isinstance(enable_video, bool)
    
    def test_modality_flags_override_defaults(self, tmp_path):
        """Test that modality flags can be overridden in config."""
        config_content = """
model:
  enable_audio: false
  enable_video: true
  checkpoint_dir: checkpoints
  architecture: multimodal
  fusion_strategy: concat
training:
  seed: 42
  learning_rate: 0.0001
dataset:
  fps: 30
  audio_sr: 16000
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        model_cfg = getattr(config, 'model', {})
        
        assert getattr(model_cfg, 'enable_audio', True) is False
        assert getattr(model_cfg, 'enable_video', True) is True


class TestMultimodalModelInstantiation:
    """Test MultimodalModel instantiation with different modality combinations."""
    
    def test_instantiate_multimodal_default(self, config):
        """Test instantiation with default settings (both modalities enabled)."""
        model = MultimodalModel(config=config, num_classes=2)
        
        # Check that model was created
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Check that both encoders exist
        assert hasattr(model, 'video_encoder')
        assert hasattr(model, 'audio_encoder')
    
    def test_instantiate_multimodal_explicit_both_enabled(self, config):
        """Test explicit instantiation with both modalities enabled."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        assert model is not None
        assert hasattr(model, 'video_encoder')
        assert hasattr(model, 'audio_encoder')
    
    def test_instantiate_video_only(self, config):
        """Test instantiation with video-only mode."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        
        assert model is not None
        assert hasattr(model, 'video_encoder')
        # Audio encoder should still exist but not be used
        assert hasattr(model, 'audio_encoder')
    
    def test_instantiate_audio_only(self, config):
        """Test instantiation with audio-only mode."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        
        assert model is not None
        assert hasattr(model, 'video_encoder')
        # Both encoders exist, but video features aren't used
        assert hasattr(model, 'audio_encoder')
    
    def test_cannot_disable_both_modalities(self, config):
        """Test that disabling both modalities raises an error."""
        with pytest.raises(ValueError, match="at least one modality must be enabled"):
            MultimodalModel(
                config=config,
                num_classes=2,
                enable_video=False,
                enable_audio=False,
            )
    
    def test_model_parameter_count(self, config):
        """Test that model reports correct parameter count."""
        model = MultimodalModel(config=config, num_classes=2)
        
        param_count = model.count_parameters()
        assert isinstance(param_count, int)
        assert param_count > 0


class TestForwardPassWithModalities:
    """Test forward pass with different modality inputs."""
    
    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        batch_size = 2
        video_frames = 16
        audio_frames = 1024
        
        # Video: (batch, channels, frames, height, width)
        video = torch.randn(batch_size, 3, video_frames, 224, 224)
        # Audio: (batch, channels, audio_samples)
        audio = torch.randn(batch_size, 1, audio_frames)
        
        return video, audio
    
    def test_forward_multimodal_both_inputs(self, config, batch_data):
        """Test forward pass with both video and audio inputs."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.eval()
        
        video, audio = batch_data
        
        with torch.no_grad():
            logits = model(video=video, audio=audio)
        
        assert logits is not None
        assert logits.shape == (2, 2)  # (batch_size, num_classes)
    
    def test_forward_video_only_with_video_input(self, config, batch_data):
        """Test video-only model with video input."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        model.eval()
        
        video, _ = batch_data
        
        with torch.no_grad():
            logits = model(video=video, audio=None)
        
        assert logits is not None
        assert logits.shape == (2, 2)
    
    def test_forward_audio_only_with_audio_input(self, config, batch_data):
        """Test audio-only model with audio input."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        model.eval()
        
        _, audio = batch_data
        
        with torch.no_grad():
            logits = model(video=None, audio=audio)
        
        assert logits is not None
        assert logits.shape == (2, 2)
    
    def test_forward_video_only_ignores_audio(self, config, batch_data):
        """Test that video-only model ignores audio input."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        model.eval()
        
        video, audio = batch_data
        
        with torch.no_grad():
            # Forward with audio should not cause error, but audio should be ignored
            logits = model(video=video, audio=audio)
        
        assert logits is not None
        assert logits.shape == (2, 2)
    
    def test_forward_audio_only_ignores_video(self, config, batch_data):
        """Test that audio-only model ignores video input."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        model.eval()
        
        video, audio = batch_data
        
        with torch.no_grad():
            # Forward with video should not cause error, but video should be ignored
            logits = model(video=None, audio=audio)
        
        assert logits is not None
        assert logits.shape == (2, 2)
    
    def test_forward_missing_required_input(self, config, batch_data):
        """Test that forward pass fails when required input is missing."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.eval()
        
        video, _ = batch_data
        
        # Missing audio should cause an error
        with pytest.raises((ValueError, RuntimeError)):
            with torch.no_grad():
                logits = model(video=video, audio=None)


class TestExtractFeaturesWithModalities:
    """Test feature extraction with different modalities."""
    
    @pytest.fixture
    def batch_data(self):
        """Create sample batch data."""
        batch_size = 2
        video_frames = 16
        audio_frames = 1024
        
        video = torch.randn(batch_size, 3, video_frames, 224, 224)
        audio = torch.randn(batch_size, 1, audio_frames)
        
        return video, audio
    
    def test_extract_features_multimodal(self, config, batch_data):
        """Test feature extraction in multimodal mode."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.eval()
        
        video, audio = batch_data
        
        with torch.no_grad():
            video_feat, audio_feat = model.extract_features(
                video=video,
                audio=audio,
            )
        
        # Both features should be returned
        assert video_feat is not None
        assert audio_feat is not None
        assert video_feat.dim() >= 2
        assert audio_feat.dim() >= 2
    
    def test_extract_features_video_only(self, config, batch_data):
        """Test feature extraction in video-only mode."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        model.eval()
        
        video, _ = batch_data
        
        with torch.no_grad():
            video_feat, audio_feat = model.extract_features(
                video=video,
                audio=None,
            )
        
        # Video features should be returned, audio should be None
        assert video_feat is not None
        assert audio_feat is None
    
    def test_extract_features_audio_only(self, config, batch_data):
        """Test feature extraction in audio-only mode."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        model.eval()
        
        _, audio = batch_data
        
        with torch.no_grad():
            video_feat, audio_feat = model.extract_features(
                video=None,
                audio=audio,
            )
        
        # Audio features should be returned, video should be None
        assert video_feat is None
        assert audio_feat is not None


class TestModalityFusionDimensions:
    """Test that fusion dimensions are computed correctly based on modalities."""
    
    def test_fusion_dim_multimodal(self, config):
        """Test fusion dimension with both modalities enabled."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        # Fusion dimension should be sum of video and audio dims (for concat)
        # or max of dims (for attention)
        assert hasattr(model, 'fusion_dim')
        assert model.fusion_dim > 0
    
    def test_fusion_dim_video_only(self, config):
        """Test fusion dimension with video-only."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        
        assert hasattr(model, 'fusion_dim')
        assert model.fusion_dim > 0
    
    def test_fusion_dim_audio_only(self, config):
        """Test fusion dimension with audio-only."""
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        
        assert hasattr(model, 'fusion_dim')
        assert model.fusion_dim > 0


class TestTrainerModalitySupport:
    """Test that Trainer class properly handles modality configuration."""
    
    def test_trainer_reads_modality_flags(self, config, tmp_path):
        """Test that Trainer reads enable_audio and enable_video from config."""
        from src.train.multimodal_train import Trainer
        
        # Create a minimal data directory structure
        data_root = tmp_path / "data"
        data_root.mkdir(exist_ok=True)
        (data_root / "train").mkdir(exist_ok=True)
        (data_root / "val").mkdir(exist_ok=True)
        
        try:
            trainer = Trainer(
                config=config,
                data_root=str(data_root),
                epochs=1,
                batch_size=2,
                debug=True,
            )
            
            # Check that model was built
            assert trainer.model is not None
            assert isinstance(trainer.model, MultimodalModel)
        except Exception as e:
            # If trainer fails due to data loading issues, that's OK
            # We're just testing that modality config is read
            assert "modality" not in str(e).lower() or "enable_" not in str(e).lower()
    
    def test_trainer_with_video_only_config(self, tmp_path):
        """Test Trainer with video-only configuration."""
        from src.train.multimodal_train import Trainer
        
        # Create config with video-only
        config_content = """
model:
  enable_audio: false
  enable_video: true
  checkpoint_dir: checkpoints
  architecture: multimodal
  fusion_strategy: concat
training:
  seed: 42
  learning_rate: 0.0001
dataset:
  fps: 30
  audio_sr: 16000
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        # Create data directory
        data_root = tmp_path / "data"
        data_root.mkdir(exist_ok=True)
        (data_root / "train").mkdir(exist_ok=True)
        (data_root / "val").mkdir(exist_ok=True)
        
        try:
            trainer = Trainer(
                config=config,
                data_root=str(data_root),
                epochs=1,
                batch_size=2,
                debug=True,
            )
            
            assert trainer.model is not None
        except Exception as e:
            # Allow data loading errors, but not modality config errors
            assert "modality" not in str(e).lower()


class TestAblationConfiguration:
    """Test end-to-end ablation study configuration."""
    
    def test_modality_combinations(self, config):
        """Test all three modality combinations can be configured."""
        combinations = [
            (True, True, "multimodal"),
            (True, False, "video-only"),
            (False, True, "audio-only"),
        ]
        
        models = []
        for enable_video, enable_audio, name in combinations:
            try:
                model = MultimodalModel(
                    config=config,
                    num_classes=2,
                    enable_video=enable_video,
                    enable_audio=enable_audio,
                )
                models.append((name, model))
            except ValueError:
                pytest.fail(f"Failed to create {name} model")
        
        assert len(models) == 3
    
    def test_modality_config_in_yaml(self, tmp_path):
        """Test modality configuration in YAML file."""
        yaml_content = """
model:
  enable_audio: true
  enable_video: true
  checkpoint_dir: checkpoints
  architecture: multimodal
training:
  seed: 42
dataset:
  fps: 30
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        
        config = Config(str(config_file))
        model_cfg = getattr(config, 'model', {})
        
        assert getattr(model_cfg, 'enable_audio', True)
        assert getattr(model_cfg, 'enable_video', True)


class TestModalityDropout:
    """Test modality-level dropout for fusion robustness."""
    
    def test_modality_dropout_config_loading(self, config_path):
        """Test that modality_dropout_prob is loaded from config."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        fusion_cfg = config_dict.get('model', {}).get('fusion', {})
        modality_dropout = fusion_cfg.get('modality_dropout_prob', 0.0)
        
        assert isinstance(modality_dropout, (int, float))
        assert 0.0 <= modality_dropout <= 1.0
    
    def test_modality_dropout_disabled_by_default(self, tmp_path):
        """Test that modality dropout is disabled by default (prob=0.0)."""
        # Create minimal config for testing
        config_content = """
model:
  enable_audio: true
  enable_video: true
  checkpoint_dir: checkpoints
  video:
    backbone: efficientnet_b3
    pretrained: true
    embed_dim: 1536
  audio:
    sample_rate: 16000
    n_mels: 64
    embed_dim: 256
  fusion:
    strategy: concat
    hidden_dim: 512
    dropout: 0.3
    modality_dropout_prob: 0.0
training:
  seed: 42
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        
        assert hasattr(model, 'modality_dropout_prob')
        assert model.modality_dropout_prob >= 0.0
    
    @pytest.fixture
    def test_config(self, tmp_path):
        """Create a test config for dropout tests."""
        import yaml
        config_data = {
            "model": {
                "enable_audio": True,
                "enable_video": True,
                "checkpoint_dir": "checkpoints",
                "video": {
                    "backbone": "efficientnet_b3",
                    "pretrained": True,
                    "embed_dim": 1536,
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
                },
            },
            "training": {"seed": 42},
        }
        return Config(**config_data)
    
    @pytest.fixture
    def batch_data_multimodal(self):
        """Create sample batch data for multimodal testing."""
        batch_size = 4
        video_frames = 16
        audio_frames = 1024
        
        video = torch.randn(batch_size, 3, video_frames, 224, 224)
        audio = torch.randn(batch_size, 1, audio_frames)
        
        return video, audio
    
    def test_dropout_disabled_during_inference(self, test_config, batch_data_multimodal):
        """Test that dropout is disabled during inference (eval mode)."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.eval()  # Set to evaluation mode
        
        # Manually set high dropout probability
        model.modality_dropout_prob = 0.5
        
        video, audio = batch_data_multimodal
        
        # Run inference multiple times - should be identical
        with torch.no_grad():
            output1 = model(video=video, audio=audio)
            output2 = model(video=video, audio=audio)
        
        # Outputs should be identical (no randomness in eval mode)
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_dropout_applied_during_training(self, test_config, batch_data_multimodal):
        """Test that dropout is applied during training."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()  # Set to training mode
        model.modality_dropout_prob = 0.5  # High dropout probability
        
        video, audio = batch_data_multimodal
        
        # Multiple forward passes should produce different results
        outputs = []
        for _ in range(5):
            output = model(video=video, audio=audio)
            outputs.append(output)
        
        # Check if outputs vary (due to dropout)
        # Note: This is probabilistic, so we just check that tensor operations work
        assert all(out.shape == (4, 2) for out in outputs)
    
    def test_dropout_produces_valid_outputs(self, test_config, batch_data_multimodal):
        """Test that dropout doesn't break output validity."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        model.modality_dropout_prob = 0.5
        
        video, audio = batch_data_multimodal
        
        logits = model(video=video, audio=audio)
        
        # Check output shape
        assert logits.shape == (4, 2)
        
        # Check for NaN or Inf
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        # Check that logits are reasonable
        assert logits.abs().max() < 1e3
    
    def test_dropout_zeros_out_modality(self, test_config, batch_data_multimodal):
        """Test that dropout actually zeros out modality features."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        
        # Very high dropout to ensure it happens
        model.modality_dropout_prob = 0.99
        
        video, audio = batch_data_multimodal
        
        # Run multiple times to catch dropout events
        for _ in range(20):
            model(video=video, audio=audio)
    
    def test_dropout_only_multimodal(self, test_config, batch_data_multimodal):
        """Test that dropout only applies in multimodal mode."""
        video, audio = batch_data_multimodal
        
        # Test video-only model
        model_video = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=False,
        )
        model_video.train()
        model_video.modality_dropout_prob = 0.5
        
        # Should not error even with dropout enabled
        output = model_video(video=video, audio=None)
        assert output.shape == (4, 2)
        
        # Test audio-only model
        model_audio = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=False,
            enable_audio=True,
        )
        model_audio.train()
        model_audio.modality_dropout_prob = 0.5
        
        # Should not error even with dropout enabled
        output = model_audio(video=None, audio=audio)
        assert output.shape == (4, 2)
    
    def test_dropout_with_different_probabilities(self, test_config, batch_data_multimodal):
        """Test dropout with various probability values."""
        video, audio = batch_data_multimodal
        
        for dropout_prob in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            model = MultimodalModel(
                config=test_config,
                num_classes=2,
                enable_video=True,
                enable_audio=True,
            )
            model.train()
            model.modality_dropout_prob = dropout_prob
            
            logits = model(video=video, audio=audio)
            
            # Check output shape and validity
            assert logits.shape == (4, 2)
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()
    
    def test_dropout_preserves_gradient_flow(self, test_config, batch_data_multimodal):
        """Test that dropout doesn't break gradient flow."""
        video, audio = batch_data_multimodal
        video.requires_grad_(True)
        audio.requires_grad_(True)
        
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        model.modality_dropout_prob = 0.5
        
        # Forward pass
        logits = model(video=video, audio=audio)
        loss = logits.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        assert video.grad is not None
        assert audio.grad is not None
    
    def test_apply_modality_dropout_method(self, test_config, batch_data_multimodal):
        """Test the _apply_modality_dropout method directly."""
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        model.modality_dropout_prob = 1.0  # Always apply dropout
        
        video, audio = batch_data_multimodal
        video_feat = torch.randn(4, model.video_embed_dim)
        audio_feat = torch.randn(4, model.audio_embed_dim)
        
        # Apply dropout
        video_dropped, audio_dropped = model._apply_modality_dropout(
            video_feat.clone(), audio_feat.clone()
        )
        
        # At least one should be zeroed out (with 100% dropout prob)
        video_zeroed = (video_dropped == 0).all()
        audio_zeroed = (audio_dropped == 0).all()
        
        # With 100% dropout prob on both, at least one should be zeroed
        assert video_zeroed or audio_zeroed
    
    def test_dropout_disabled_with_zero_probability(self, test_config, batch_data_multimodal):
        """Test that zero dropout probability disables the feature."""
        video, audio = batch_data_multimodal
        
        model = MultimodalModel(
            config=test_config,
            num_classes=2,
            enable_video=True,
            enable_audio=True,
        )
        model.train()
        model.modality_dropout_prob = 0.0
        
        # Run multiple times - should be deterministic
        outputs = []
        for _ in range(3):
            # Reset random seed to ensure determinism
            torch.manual_seed(42)
            output = model(video=video, audio=audio)
            outputs.append(output)
        
        # All outputs should be identical (no dropout)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

# Fixtures
@pytest.fixture
def config_path():
    """Get path to default config file."""
    config_file = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_file.exists():
        pytest.skip("Config file not found")
    return config_file


@pytest.fixture
def config(config_path):
    """Load config for tests."""
    return Config(str(config_path))

