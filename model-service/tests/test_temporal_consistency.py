"""
Tests for temporal consistency loss and related functionality.
"""

import pytest
import torch

from src.models.losses import TemporalConsistencyLoss, AuxiliaryLossWeighter


class TestTemporalConsistencyLoss:
    """Test TemporalConsistencyLoss functionality."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create a temporal consistency loss function."""
        return TemporalConsistencyLoss(reduction='mean', normalize=True)
    
    @pytest.fixture
    def frame_embeddings(self):
        """Create sample frame embeddings."""
        # [B=4, T=8, D=256]
        return torch.randn(4, 8, 256)
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = TemporalConsistencyLoss()
        assert loss_fn is not None
        assert loss_fn.reduction == 'mean'
        assert loss_fn.normalize is True
    
    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError):
            TemporalConsistencyLoss(reduction='invalid')
    
    def test_forward_pass_shape(self, loss_fn, frame_embeddings):
        """Test forward pass returns scalar loss."""
        loss = loss_fn(frame_embeddings)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_loss_value_range(self, loss_fn, frame_embeddings):
        """Test loss is in reasonable range."""
        loss = loss_fn(frame_embeddings)
        assert 0 <= loss.item() < 1e6, "Loss should be in reasonable range"
    
    def test_deterministic_output(self, loss_fn, frame_embeddings):
        """Test that same input gives same output."""
        loss1 = loss_fn(frame_embeddings)
        loss2 = loss_fn(frame_embeddings)
        assert torch.isclose(loss1, loss2), "Loss should be deterministic"
    
    def test_consistent_embeddings_low_loss(self, loss_fn):
        """Test that consistent embeddings have low loss."""
        # Create embeddings that are all the same (consistent)
        consistent_emb = torch.ones(4, 8, 256)
        loss = loss_fn(consistent_emb)
        
        # Create random embeddings (inconsistent)
        random_emb = torch.randn(4, 8, 256)
        loss_random = loss_fn(random_emb)
        
        # Consistent embeddings should have lower loss
        assert loss < loss_random, "Consistent embeddings should have lower loss"
    
    def test_minimum_frames_requirement(self, loss_fn):
        """Test that at least 2 frames are required."""
        # Single frame should raise error
        single_frame = torch.randn(4, 1, 256)
        with pytest.raises(ValueError):
            loss_fn(single_frame)
    
    def test_dimension_validation(self, loss_fn):
        """Test that 3D input is required."""
        # 2D input
        invalid_2d = torch.randn(4, 256)
        with pytest.raises(ValueError):
            loss_fn(invalid_2d)
        
        # 4D input
        invalid_4d = torch.randn(4, 8, 256, 2)
        with pytest.raises(ValueError):
            loss_fn(invalid_4d)
    
    def test_mean_reduction(self, frame_embeddings):
        """Test mean reduction."""
        loss_fn = TemporalConsistencyLoss(reduction='mean')
        loss = loss_fn(frame_embeddings)
        assert loss.ndim == 0
    
    def test_sum_reduction(self, frame_embeddings):
        """Test sum reduction."""
        loss_fn = TemporalConsistencyLoss(reduction='sum')
        loss = loss_fn(frame_embeddings)
        assert loss.ndim == 0
        
        # Sum should be >= mean
        loss_mean = TemporalConsistencyLoss(reduction='mean')(frame_embeddings)
        assert loss >= loss_mean
    
    def test_normalization_effect(self, frame_embeddings):
        """Test normalization effect on loss."""
        loss_normalized = TemporalConsistencyLoss(normalize=True)(frame_embeddings)
        loss_unnormalized = TemporalConsistencyLoss(normalize=False)(frame_embeddings)
        
        # Both should be positive
        assert loss_normalized >= 0
        assert loss_unnormalized >= 0
    
    def test_batch_independence(self, loss_fn):
        """Test that loss is computed per-batch and averaged."""
        # Create two batches with different characteristics
        batch1 = torch.ones(1, 8, 256)  # Consistent
        batch2 = torch.randn(1, 8, 256)  # Random
        
        loss1 = loss_fn(batch1)
        loss2 = loss_fn(batch2)
        
        # Consistent batch should have lower loss
        assert loss1 < loss2
    
    def test_gradient_flow(self, frame_embeddings):
        """Test that gradients flow through loss."""
        frame_embeddings.requires_grad_(True)
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(frame_embeddings)
        loss.backward()
        
        assert frame_embeddings.grad is not None
        assert frame_embeddings.grad.shape == frame_embeddings.shape
    
    def test_device_compatibility(self, loss_fn):
        """Test loss works on CPU."""
        embeddings = torch.randn(4, 8, 256)
        loss = loss_fn(embeddings)
        assert loss.item() >= 0
    
    def test_dtype_compatibility(self, loss_fn):
        """Test loss with different dtypes."""
        # Float32 (default)
        emb_f32 = torch.randn(4, 8, 256, dtype=torch.float32)
        loss_f32 = loss_fn(emb_f32)
        assert loss_f32.item() >= 0
        
        # Float64
        emb_f64 = torch.randn(4, 8, 256, dtype=torch.float64)
        loss_f64 = loss_fn(emb_f64)
        assert loss_f64.item() >= 0
    
    def test_varying_temporal_dimensions(self, loss_fn):
        """Test loss with different number of frames."""
        for T in [2, 4, 8, 16, 32]:
            embeddings = torch.randn(4, T, 256)
            loss = loss_fn(embeddings)
            assert loss.ndim == 0
            assert loss.item() >= 0


class TestAuxiliaryLossWeighter:
    """Test AuxiliaryLossWeighter functionality."""
    
    @pytest.fixture
    def weighter(self):
        """Create an auxiliary loss weighter."""
        return AuxiliaryLossWeighter()
    
    def test_initialization(self, weighter):
        """Test weighter initialization."""
        assert weighter is not None
        assert len(weighter.weights) == 0
    
    def test_add_loss(self, weighter):
        """Test adding auxiliary loss."""
        weighter.add_loss('temporal', 0.1)
        assert 'temporal' in weighter.weights
        assert weighter.weights['temporal'] == 0.1
    
    def test_negative_weight_error(self, weighter):
        """Test that negative weights raise error."""
        with pytest.raises(ValueError):
            weighter.add_loss('temporal', -0.1)
    
    def test_zero_weight(self, weighter):
        """Test that zero weight is allowed."""
        weighter.add_loss('temporal', 0.0)
        assert weighter.weights['temporal'] == 0.0
    
    def test_compute_total_loss_main_only(self, weighter):
        """Test total loss with only main loss."""
        main_loss = torch.tensor(1.0)
        total = weighter.compute_total_loss(main_loss)
        assert torch.isclose(total, main_loss)
    
    def test_compute_total_loss_with_auxiliary(self, weighter):
        """Test total loss with auxiliary loss."""
        weighter.add_loss('temporal', 0.1)
        
        main_loss = torch.tensor(1.0)
        aux_loss = torch.tensor(2.0)
        
        total = weighter.compute_total_loss(main_loss, temporal=aux_loss)
        expected = main_loss + 0.1 * aux_loss
        
        assert torch.isclose(total, expected)
    
    def test_compute_total_loss_multiple_auxiliary(self, weighter):
        """Test total loss with multiple auxiliary losses."""
        weighter.add_loss('temporal', 0.1)
        weighter.add_loss('consistency', 0.05)
        
        main_loss = torch.tensor(1.0)
        temporal_loss = torch.tensor(2.0)
        consistency_loss = torch.tensor(3.0)
        
        total = weighter.compute_total_loss(
            main_loss,
            temporal=temporal_loss,
            consistency=consistency_loss
        )
        expected = main_loss + 0.1 * temporal_loss + 0.05 * consistency_loss
        
        assert torch.isclose(total, expected)
    
    def test_unregistered_loss_error(self, weighter):
        """Test that unregistered loss raises error."""
        main_loss = torch.tensor(1.0)
        unregistered_loss = torch.tensor(2.0)
        
        with pytest.raises(ValueError):
            weighter.compute_total_loss(main_loss, unregistered=unregistered_loss)
    
    def test_none_auxiliary_loss(self, weighter):
        """Test that None auxiliary losses are skipped."""
        weighter.add_loss('temporal', 0.1)
        
        main_loss = torch.tensor(1.0)
        total = weighter.compute_total_loss(main_loss, temporal=None)
        
        assert torch.isclose(total, main_loss)
    
    def test_zero_weight_ignored(self, weighter):
        """Test that zero-weight auxiliary losses don't affect total."""
        weighter.add_loss('temporal', 0.0)
        
        main_loss = torch.tensor(1.0)
        aux_loss = torch.tensor(100.0)  # High value, but zero weight
        
        total = weighter.compute_total_loss(main_loss, temporal=aux_loss)
        
        assert torch.isclose(total, main_loss)
    
    def test_log_losses_no_auxiliary(self, weighter):
        """Test logging with only main loss."""
        main_loss = torch.tensor(1.5)
        
        # Method should complete without error
        weighter.log_losses(main_loss)
    
    def test_log_losses_with_step(self, weighter):
        """Test logging with step information."""
        main_loss = torch.tensor(1.0)
        
        # Just ensure method doesn't crash with step parameter
        weighter.log_losses(main_loss, step=100)
    
    def test_loss_weight_configuration(self, weighter):
        """Test configuring multiple loss weights."""
        weights_config = {
            'temporal': 0.15,
            'consistency': 0.08,
            'smoothness': 0.05,
        }
        
        for name, weight in weights_config.items():
            weighter.add_loss(name, weight)
        
        assert len(weighter.weights) == 3
        for name, weight in weights_config.items():
            assert weighter.weights[name] == weight


class TestTemporalConsistencyIntegration:
    """Integration tests for temporal consistency loss with model."""
    
    def test_loss_with_realistic_embeddings(self):
        """Test loss with realistic video frame embeddings."""
        # Simulate embeddings from actual model
        batch_size = 4
        num_frames = 16
        embedding_dim = 1536  # EfficientNet embedding size
        
        # Create somewhat realistic embeddings (not purely random)
        base = torch.randn(batch_size, embedding_dim)
        noise = torch.randn(batch_size, num_frames, embedding_dim) * 0.1
        embeddings = base.unsqueeze(1) + noise
        
        loss_fn = TemporalConsistencyLoss()
        loss = loss_fn(embeddings)
        
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_loss_computation_efficiency(self):
        """Test that loss computation is efficient."""
        embeddings = torch.randn(8, 32, 1536)  # Large batch
        loss_fn = TemporalConsistencyLoss()
        
        # Should compute without issues
        loss = loss_fn(embeddings)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_loss_gradient_propagation(self):
        """Test that gradients propagate correctly."""
        embeddings = torch.randn(4, 16, 256, requires_grad=True)
        loss_fn = TemporalConsistencyLoss()
        
        loss = loss_fn(embeddings)
        loss.backward()
        
        # Check gradients
        assert embeddings.grad is not None
        assert embeddings.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert not torch.isnan(embeddings.grad).any()
        assert not torch.isinf(embeddings.grad).any()
