"""
Loss functions for video deepfake detection.

This module provides auxiliary losses to improve model training:
- TemporalConsistencyLoss: Penalizes high variance across adjacent frame embeddings
"""

import logging
from typing import Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss for video understanding.
    
    Penalizes high variance in embeddings across adjacent video frames.
    This encourages smooth, consistent feature representations across time,
    which is important for deepfake detection where authentic videos tend to
    have more temporal consistency than artificially generated ones.
    
    The loss is computed as the mean squared difference between embeddings
    of adjacent frames, normalized by the variance of all frame embeddings.
    
    Args:
        reduction (str): How to reduce the loss. Options: 'mean', 'sum'. Default: 'mean'
        normalize (bool): Whether to normalize by frame embedding variance. Default: True
    
    Example:
        >>> loss_fn = TemporalConsistencyLoss()
        >>> frame_feats = torch.randn(4, 16, 1536)  # [B, T, D]
        >>> loss = loss_fn(frame_feats)
        >>> print(loss.item())  # scalar loss value
    """
    
    def __init__(self, reduction: str = 'mean', normalize: bool = True):
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize
        
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    
    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            frame_embeddings: [B, T, D] where:
                - B: batch size
                - T: number of frames (must be >= 2)
                - D: embedding dimension
        
        Returns:
            loss: scalar tensor representing temporal consistency loss
        
        Raises:
            ValueError: If frame_embeddings has fewer than 2 frames
        """
        if frame_embeddings.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor [B, T, D], got shape {frame_embeddings.shape}"
            )
        
        B, T, D = frame_embeddings.shape
        
        if T < 2:
            raise ValueError(f"Need at least 2 frames for temporal consistency, got {T}")
        
        # Compute differences between adjacent frames: [B, T-1, D]
        frame_diffs = frame_embeddings[:, 1:, :] - frame_embeddings[:, :-1, :]
        
        # Mean squared difference: [B, T-1, D] -> [B, T-1] -> scalar
        mse_diffs = (frame_diffs ** 2).mean(dim=2)  # [B, T-1]
        
        # Optionally normalize by variance across frames
        if self.normalize:
            # Compute per-frame variance across embedding dimension
            frame_vars = frame_embeddings.var(dim=2)  # [B, T]
            
            # Mean variance across all frames: [B]
            mean_vars = frame_vars.mean(dim=1, keepdim=True)  # [B, 1]
            
            # Avoid division by zero
            mean_vars = torch.clamp(mean_vars, min=1e-8)
            
            # Normalize differences by variance
            mse_diffs = mse_diffs / mean_vars  # [B, T-1]
        
        # Reduce across batch and temporal dimensions
        if self.reduction == 'mean':
            loss = mse_diffs.mean()
        else:  # sum
            loss = mse_diffs.sum()
        
        return loss


class AuxiliaryLossWeighter:
    """
    Helper class to manage and log auxiliary losses.
    
    Maintains a dictionary of auxiliary loss names and their weights,
    providing utilities to compute weighted sums and log separately.
    
    Example:
        >>> weighter = AuxiliaryLossWeighter()
        >>> weighter.add_loss('temporal', 0.1)
        >>> main_loss = criterion(pred, target)
        >>> temporal_loss = temporal_loss_fn(frame_feats)
        >>> total_loss = weighter.compute_total_loss(main_loss, temporal=temporal_loss)
    """
    
    def __init__(self):
        self.weights = {}
        self.logger = logging.getLogger(__name__)
    
    def add_loss(self, name: str, weight: float):
        """Register an auxiliary loss with its weight."""
        if weight < 0:
            raise ValueError(f"Loss weight must be non-negative, got {weight}")
        self.weights[name] = weight
    
    def compute_total_loss(self, main_loss: torch.Tensor, **auxiliary_losses) -> torch.Tensor:
        """
        Compute weighted sum of main loss and auxiliary losses.
        
        Args:
            main_loss: The primary classification loss (tensor)
            **auxiliary_losses: Named auxiliary losses as keyword arguments
        
        Returns:
            total_loss: Weighted sum of all losses
        """
        total = main_loss
        
        for name, loss in auxiliary_losses.items():
            if loss is None:
                continue
            
            if name not in self.weights:
                raise ValueError(
                    f"Auxiliary loss '{name}' not registered. "
                    f"Available: {list(self.weights.keys())}"
                )
            
            weight = self.weights[name]
            if weight > 0:
                total = total + weight * loss
        
        return total
    
    def log_losses(
        self,
        main_loss: torch.Tensor,
        step: Optional[int] = None,
        **auxiliary_losses
    ):
        """
        Log main loss and all auxiliary losses separately.
        
        Args:
            main_loss: The primary classification loss
            step: Optional training step number for logging context
            **auxiliary_losses: Named auxiliary losses
        """
        step_str = f"[Step {step}] " if step is not None else ""
        
        self.logger.info(f"{step_str}Main Loss: {main_loss.item():.4f}")
        
        for name, loss in auxiliary_losses.items():
            if loss is None:
                continue
            
            weight = self.weights.get(name, 0.0)
            weighted_value = (weight * loss).item() if weight > 0 else 0.0
            
            self.logger.info(
                f"{step_str}Auxiliary Loss - {name}: {loss.item():.4f} "
                f"(weight={weight:.4f}, weighted={weighted_value:.4f})"
            )
