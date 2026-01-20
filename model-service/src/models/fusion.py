"""
Cross-Modal Attention Fusion for Multimodal Deepfake Detection

This module implements advanced fusion strategies for combining video and audio
features, including cross-modal attention and gated fusion mechanisms.

Expected Performance Gain: +2-5% AUC improvement over simple concatenation

Usage:
    fusion = CrossModalAttentionFusion(video_dim=1536, audio_dim=512, output_dim=1024)
    video_feat = torch.randn(4, 1536)
    audio_feat = torch.randn(4, 512)
    fused = fusion(video_feat, audio_feat)  # [4, 1024]
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion for video and audio features.
    
    Uses video features as queries and audio features as keys/values
    to learn complementary information between modalities.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        output_dim: Output fused feature dimension
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Project video and audio to common dimension for attention
        self.common_dim = max(video_dim, audio_dim)
        
        self.video_proj = nn.Linear(video_dim, self.common_dim)
        self.audio_proj = nn.Linear(audio_dim, self.common_dim)
        
        # Multi-head cross-attention: video queries, audio keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.common_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        logger.info(
            f"Initialized CrossModalAttentionFusion: "
            f"video_dim={video_dim}, audio_dim={audio_dim}, "
            f"output_dim={output_dim}, num_heads={num_heads}"
        )
    
    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse video and audio features using cross-modal attention.
        
        Args:
            video_feat: Video features [B, video_dim]
            audio_feat: Audio features [B, audio_dim]
        
        Returns:
            fused_feat: Fused features [B, output_dim]
        """
        # Project to common dimension
        video_proj = self.video_proj(video_feat)  # [B, common_dim]
        audio_proj = self.audio_proj(audio_feat)  # [B, common_dim]
        
        # Add sequence dimension for attention [B, 1, common_dim]
        video_query = video_proj.unsqueeze(1)
        audio_kv = audio_proj.unsqueeze(1)
        
        # Cross-attention: video attends to audio
        attended, attention_weights = self.cross_attention(
            query=video_query,
            key=audio_kv,
            value=audio_kv,
        )
        
        # Remove sequence dimension [B, common_dim]
        attended = attended.squeeze(1)
        
        # Concatenate original video features with attended audio
        combined = torch.cat([video_proj, attended], dim=1)  # [B, common_dim * 2]
        
        # Final fusion
        fused = self.fusion(combined)  # [B, output_dim]
        
        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for adaptive modality weighting.
    
    Learns to weight video and audio contributions based on input content.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        output_dim: Output fused feature dimension
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Project modalities to common dimension
        self.video_proj = nn.Linear(video_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(video_dim + audio_dim, output_dim),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"Initialized GatedFusion: "
            f"video_dim={video_dim}, audio_dim={audio_dim}, output_dim={output_dim}"
        )
    
    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse video and audio features using gating.
        
        Args:
            video_feat: Video features [B, video_dim]
            audio_feat: Audio features [B, audio_dim]
        
        Returns:
            fused_feat: Fused features [B, output_dim]
        """
        # Project to common dimension
        video_proj = self.video_proj(video_feat)  # [B, output_dim]
        audio_proj = self.audio_proj(audio_feat)  # [B, output_dim]
        
        # Compute gate weights
        concat_feat = torch.cat([video_feat, audio_feat], dim=1)
        gate_weights = self.gate(concat_feat)  # [B, output_dim]
        
        # Weighted combination
        fused = gate_weights * video_proj + (1 - gate_weights) * audio_proj
        fused = self.dropout(fused)
        
        return fused


class BimodalTransformerFusion(nn.Module):
    """
    Transformer-based fusion with bidirectional cross-attention.
    
    Both video and audio attend to each other, allowing rich cross-modal
    interaction before fusion.
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        output_dim: Output fused feature dimension
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim
        
        # Common embedding dimension
        self.embed_dim = max(video_dim, audio_dim)
        
        # Modality projections
        self.video_proj = nn.Linear(video_dim, self.embed_dim)
        self.audio_proj = nn.Linear(audio_dim, self.embed_dim)
        
        # Modality embeddings (learnable)
        self.video_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.audio_embed = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Transformer encoder for cross-modal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        logger.info(
            f"Initialized BimodalTransformerFusion: "
            f"video_dim={video_dim}, audio_dim={audio_dim}, "
            f"output_dim={output_dim}, num_heads={num_heads}, num_layers={num_layers}"
        )
    
    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse video and audio features using transformer.
        
        Args:
            video_feat: Video features [B, video_dim]
            audio_feat: Audio features [B, audio_dim]
        
        Returns:
            fused_feat: Fused features [B, output_dim]
        """
        batch_size = video_feat.size(0)
        
        # Project to common dimension
        video_proj = self.video_proj(video_feat)  # [B, embed_dim]
        audio_proj = self.audio_proj(audio_feat)  # [B, embed_dim]
        
        # Add modality embeddings
        video_proj = video_proj.unsqueeze(1) + self.video_embed  # [B, 1, embed_dim]
        audio_proj = audio_proj.unsqueeze(1) + self.audio_embed  # [B, 1, embed_dim]
        
        # Concatenate as sequence [B, 2, embed_dim]
        multimodal_seq = torch.cat([video_proj, audio_proj], dim=1)
        
        # Apply transformer
        transformed = self.transformer(multimodal_seq)  # [B, 2, embed_dim]
        
        # Extract video and audio representations
        video_out = transformed[:, 0, :]  # [B, embed_dim]
        audio_out = transformed[:, 1, :]  # [B, embed_dim]
        
        # Concatenate and project
        combined = torch.cat([video_out, audio_out], dim=1)  # [B, embed_dim * 2]
        fused = self.output_proj(combined)  # [B, output_dim]
        
        return fused


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation-based fusion (baseline).
    
    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        output_dim: Output fused feature dimension (optional, defaults to video_dim + audio_dim)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.output_dim = output_dim or (video_dim + audio_dim)
        
        # Optional projection if output_dim differs from concat dim
        if self.output_dim != (video_dim + audio_dim):
            self.projection = nn.Sequential(
                nn.Linear(video_dim + audio_dim, self.output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()
        
        logger.info(
            f"Initialized ConcatenationFusion: "
            f"video_dim={video_dim}, audio_dim={audio_dim}, output_dim={self.output_dim}"
        )
    
    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate video and audio features.
        
        Args:
            video_feat: Video features [B, video_dim]
            audio_feat: Audio features [B, audio_dim]
        
        Returns:
            fused_feat: Fused features [B, output_dim]
        """
        concatenated = torch.cat([video_feat, audio_feat], dim=1)
        fused = self.projection(concatenated)
        return fused
