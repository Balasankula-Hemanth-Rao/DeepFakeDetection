"""
Unit tests for audio encoder module.

Tests both Wav2Vec2AudioEncoder and SimpleAudioEncoder.
"""

import pytest
import torch
import numpy as np

from src.models.audio_encoder import Wav2Vec2AudioEncoder, SimpleAudioEncoder


class TestWav2Vec2AudioEncoder:
    """Tests for Wav2Vec2AudioEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return Wav2Vec2AudioEncoder(
            model_name="facebook/wav2vec2-base",
            embed_dim=512,
            freeze_layers=8,
            sample_rate=16000,
        )
    
    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embed_dim == 512
        assert encoder.freeze_layers == 8
        assert encoder.sample_rate == 16000
    
    def test_forward_pass(self, encoder):
        """Test forward pass with valid input."""
        # Create sample waveform (1 second at 16kHz)
        batch_size = 4
        waveform = torch.randn(batch_size, 16000)
        
        # Forward pass
        features = encoder(waveform)
        
        # Check output shape
        assert features.shape == (batch_size, 512)
        assert not torch.isnan(features).any()
    
    def test_single_sample(self, encoder):
        """Test with single sample (1D input)."""
        waveform = torch.randn(16000)
        features = encoder(waveform)
        
        assert features.shape == (1, 512)
    
    def test_variable_length(self, encoder):
        """Test with variable length audio."""
        # Different length (2 seconds)
        waveform = torch.randn(4, 32000)
        features = encoder(waveform)
        
        assert features.shape == (4, 512)
    
    def test_parameter_count(self, encoder):
        """Test parameter counting."""
        total_params = encoder.count_parameters(trainable_only=False)
        trainable_params = encoder.count_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params < total_params  # Some layers are frozen
    
    def test_pooling_strategies(self):
        """Test different pooling strategies."""
        strategies = ['mean', 'max', 'cls']
        
        for strategy in strategies:
            encoder = Wav2Vec2AudioEncoder(
                embed_dim=512,
                pooling_strategy=strategy
            )
            waveform = torch.randn(2, 16000)
            features = encoder(waveform)
            
            assert features.shape == (2, 512)


class TestSimpleAudioEncoder:
    """Tests for SimpleAudioEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return SimpleAudioEncoder(n_mels=64, embed_dim=256)
    
    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.n_mels == 64
        assert encoder.embed_dim == 256
    
    def test_forward_pass_3d(self, encoder):
        """Test forward pass with 3D input."""
        # Mel-spectrogram: [B, n_mels, T]
        batch_size = 4
        mel_spec = torch.randn(batch_size, 64, 100)
        
        features = encoder(mel_spec)
        
        assert features.shape == (batch_size, 256)
        assert not torch.isnan(features).any()
    
    def test_forward_pass_4d(self, encoder):
        """Test forward pass with 4D input."""
        # Mel-spectrogram: [B, 1, n_mels, T]
        mel_spec = torch.randn(4, 1, 64, 100)
        
        features = encoder(mel_spec)
        
        assert features.shape == (4, 256)
    
    def test_different_time_lengths(self, encoder):
        """Test with different temporal lengths."""
        for time_steps in [50, 100, 200]:
            mel_spec = torch.randn(2, 64, time_steps)
            features = encoder(mel_spec)
            
            assert features.shape == (2, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAudioEncoderGPU:
    """GPU-specific tests."""
    
    def test_wav2vec2_gpu(self):
        """Test Wav2Vec2 on GPU."""
        encoder = Wav2Vec2AudioEncoder(embed_dim=512)
        encoder = encoder.cuda()
        
        waveform = torch.randn(2, 16000).cuda()
        features = encoder(waveform)
        
        assert features.device.type == 'cuda'
        assert features.shape == (2, 512)
    
    def test_simple_encoder_gpu(self):
        """Test SimpleAudioEncoder on GPU."""
        encoder = SimpleAudioEncoder(n_mels=64, embed_dim=256)
        encoder = encoder.cuda()
        
        mel_spec = torch.randn(2, 64, 100).cuda()
        features = encoder(mel_spec)
        
        assert features.device.type == 'cuda'
        assert features.shape == (2, 256)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
