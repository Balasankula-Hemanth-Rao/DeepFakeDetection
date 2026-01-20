"""
Integration test for end-to-end video inference pipeline.

Tests the complete flow from video input to prediction output.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models import MultimodalModel
from src.serve import VideoInferenceEngine
from src.preprocess import AudioExtractor, FrameExtractor


class TestEndToEndInference:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample video frames."""
        # Simulate 16 frames of 224x224 RGB images
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
        return frames
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio waveform."""
        # 1 second of audio at 16kHz
        waveform = np.random.randn(16000).astype(np.float32)
        return waveform
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration."""
        class Config:
            class Model:
                class Video:
                    backbone = 'efficientnet_b3'
                    pretrained = False  # Don't download for testing
                    embed_dim = 1536
                    temporal_strategy = 'avg_pool'
                
                class Audio:
                    encoder_type = 'simple'  # Use simple for faster testing
                    sample_rate = 16000
                    n_mels = 64
                    embed_dim = 256
                
                class Fusion:
                    strategy = 'concat'
                    hidden_dim = 512
                    dropout = 0.3
                    modality_dropout_prob = 0.0
                
                video = Video()
                audio = Audio()
                fusion = Fusion()
            
            model = Model()
        
        return Config()
    
    def test_model_initialization(self, model_config):
        """Test model can be initialized."""
        model = MultimodalModel(config=model_config)
        
        assert model is not None
        assert model.enable_video
        assert model.enable_audio
    
    def test_forward_pass(self, model_config, sample_frames):
        """Test forward pass through model."""
        model = MultimodalModel(config=model_config)
        model.eval()
        
        # Prepare inputs
        # Video: [B, T, C, H, W]
        video_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            for frame in sample_frames
        ]).unsqueeze(0)
        
        # Audio: [B, n_mels, T]
        audio_tensor = torch.randn(1, 64, 100)
        
        # Forward pass
        with torch.no_grad():
            logits = model(video=video_tensor, audio=audio_tensor)
        
        assert logits.shape == (1, 2)  # Binary classification
        assert not torch.isnan(logits).any()
    
    @pytest.mark.asyncio
    async def test_inference_engine(self, model_config, sample_frames, sample_audio):
        """Test inference engine."""
        model = MultimodalModel(config=model_config)
        model.eval()
        
        config = {
            'fps': 3,
            'sample_rate': 16000,
            'aggregation': 'mean',
        }
        
        engine = VideoInferenceEngine(model, config)
        
        # Run inference
        result = await engine.analyze_video(
            frames=sample_frames,
            audio=sample_audio,
            sample_rate=16000,
        )
        
        # Check result structure
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'frame_predictions' in result
        assert 'anomalous_frames' in result
        assert 'metadata' in result
        
        # Check prediction values
        assert result['prediction'] in ['real', 'fake']
        assert 0 <= result['confidence'] <= 1
        assert len(result['frame_predictions']) > 0
    
    def test_model_save_load(self, model_config, tmp_path):
        """Test model saving and loading."""
        # Create and save model
        model = MultimodalModel(config=model_config)
        checkpoint_path = tmp_path / "test_model.pth"
        
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=1,
            metrics={'auc': 0.85},
        )
        
        assert checkpoint_path.exists()
        
        # Load model
        loaded_model = MultimodalModel.load_for_inference(
            str(checkpoint_path),
            config=model_config,
            device='cpu',
        )
        
        assert loaded_model is not None
        
        # Test loaded model works
        video_tensor = torch.randn(1, 16, 3, 224, 224)
        audio_tensor = torch.randn(1, 64, 100)
        
        with torch.no_grad():
            logits = loaded_model(video=video_tensor, audio=audio_tensor)
        
        assert logits.shape == (1, 2)
    
    def test_different_fusion_strategies(self, model_config):
        """Test model with different fusion strategies."""
        strategies = ['concat', 'attention']
        
        for strategy in strategies:
            model_config.model.fusion.strategy = strategy
            model = MultimodalModel(config=model_config)
            
            video_tensor = torch.randn(2, 8, 3, 224, 224)
            audio_tensor = torch.randn(2, 64, 100)
            
            with torch.no_grad():
                logits = model(video=video_tensor, audio=audio_tensor)
            
            assert logits.shape == (2, 2)
    
    def test_modality_ablation(self, model_config):
        """Test video-only and audio-only modes."""
        # Video-only
        video_model = MultimodalModel(
            config=model_config,
            enable_video=True,
            enable_audio=False,
        )
        
        video_tensor = torch.randn(2, 8, 3, 224, 224)
        
        with torch.no_grad():
            logits = video_model(video=video_tensor, audio=None)
        
        assert logits.shape == (2, 2)
        
        # Audio-only
        audio_model = MultimodalModel(
            config=model_config,
            enable_video=False,
            enable_audio=True,
        )
        
        audio_tensor = torch.randn(2, 64, 100)
        
        with torch.no_grad():
            logits = audio_model(video=None, audio=audio_tensor)
        
        assert logits.shape == (2, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUInference:
    """GPU-specific integration tests."""
    
    def test_gpu_inference(self):
        """Test inference on GPU."""
        class SimpleConfig:
            class Model:
                class Video:
                    backbone = 'efficientnet_b3'
                    pretrained = False
                    embed_dim = 1536
                    temporal_strategy = 'avg_pool'
                
                class Audio:
                    encoder_type = 'simple'
                    sample_rate = 16000
                    n_mels = 64
                    embed_dim = 256
                
                class Fusion:
                    strategy = 'concat'
                    hidden_dim = 512
                    dropout = 0.3
                    modality_dropout_prob = 0.0
                
                video = Video()
                audio = Audio()
                fusion = Fusion()
            
            model = Model()
        
        config = SimpleConfig()
        model = MultimodalModel(config=config)
        model = model.cuda()
        model.eval()
        
        video_tensor = torch.randn(2, 8, 3, 224, 224).cuda()
        audio_tensor = torch.randn(2, 64, 100).cuda()
        
        with torch.no_grad():
            logits = model(video=video_tensor, audio=audio_tensor)
        
        assert logits.device.type == 'cuda'
        assert logits.shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
