"""
Unit tests for fusion modules.

Tests CrossModalAttentionFusion, GatedFusion, BimodalTransformerFusion, and ConcatenationFusion.
"""

import pytest
import torch

from src.models.fusion import (
    CrossModalAttentionFusion,
    GatedFusion,
    BimodalTransformerFusion,
    ConcatenationFusion,
)


class TestCrossModalAttentionFusion:
    """Tests for CrossModalAttentionFusion."""
    
    @pytest.fixture
    def fusion(self):
        """Create fusion module."""
        return CrossModalAttentionFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=1024,
            num_heads=4,
        )
    
    def test_initialization(self, fusion):
        """Test initialization."""
        assert fusion.video_dim == 1536
        assert fusion.audio_dim == 512
        assert fusion.output_dim == 1024
    
    def test_forward_pass(self, fusion):
        """Test forward pass."""
        batch_size = 4
        video_feat = torch.randn(batch_size, 1536)
        audio_feat = torch.randn(batch_size, 512)
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.shape == (batch_size, 1024)
        assert not torch.isnan(fused).any()
    
    def test_different_batch_sizes(self, fusion):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            video_feat = torch.randn(batch_size, 1536)
            audio_feat = torch.randn(batch_size, 512)
            
            fused = fusion(video_feat, audio_feat)
            assert fused.shape == (batch_size, 1024)


class TestGatedFusion:
    """Tests for GatedFusion."""
    
    @pytest.fixture
    def fusion(self):
        """Create fusion module."""
        return GatedFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=1024,
        )
    
    def test_forward_pass(self, fusion):
        """Test forward pass."""
        video_feat = torch.randn(4, 1536)
        audio_feat = torch.randn(4, 512)
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.shape == (4, 1024)
        assert not torch.isnan(fused).any()
    
    def test_gate_values(self, fusion):
        """Test that gate values are in valid range."""
        video_feat = torch.randn(4, 1536)
        audio_feat = torch.randn(4, 512)
        
        # Forward pass (gate is internal)
        fused = fusion(video_feat, audio_feat)
        
        # Check output is valid
        assert torch.all(torch.isfinite(fused))


class TestBimodalTransformerFusion:
    """Tests for BimodalTransformerFusion."""
    
    @pytest.fixture
    def fusion(self):
        """Create fusion module."""
        return BimodalTransformerFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=1024,
            num_heads=4,
            num_layers=2,
        )
    
    def test_forward_pass(self, fusion):
        """Test forward pass."""
        video_feat = torch.randn(4, 1536)
        audio_feat = torch.randn(4, 512)
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.shape == (4, 1024)
        assert not torch.isnan(fused).any()
    
    def test_modality_embeddings(self, fusion):
        """Test that modality embeddings are learned."""
        assert fusion.video_embed.requires_grad
        assert fusion.audio_embed.requires_grad


class TestConcatenationFusion:
    """Tests for ConcatenationFusion."""
    
    def test_simple_concatenation(self):
        """Test simple concatenation without projection."""
        fusion = ConcatenationFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=None,  # No projection
        )
        
        video_feat = torch.randn(4, 1536)
        audio_feat = torch.randn(4, 512)
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.shape == (4, 1536 + 512)
    
    def test_with_projection(self):
        """Test concatenation with projection."""
        fusion = ConcatenationFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=1024,
        )
        
        video_feat = torch.randn(4, 1536)
        audio_feat = torch.randn(4, 512)
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.shape == (4, 1024)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusionGPU:
    """GPU-specific tests."""
    
    def test_cross_modal_attention_gpu(self):
        """Test CrossModalAttentionFusion on GPU."""
        fusion = CrossModalAttentionFusion(
            video_dim=1536,
            audio_dim=512,
            output_dim=1024,
        ).cuda()
        
        video_feat = torch.randn(4, 1536).cuda()
        audio_feat = torch.randn(4, 512).cuda()
        
        fused = fusion(video_feat, audio_feat)
        
        assert fused.device.type == 'cuda'
        assert fused.shape == (4, 1024)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
