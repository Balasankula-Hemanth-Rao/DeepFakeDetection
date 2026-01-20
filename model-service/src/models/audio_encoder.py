"""
Audio Encoder using Wav2Vec2 for Deepfake Detection

This module provides a pre-trained wav2vec2-based audio encoder for extracting
high-quality audio features from raw waveforms. The encoder replaces the naive
AudioCNN with a state-of-the-art pre-trained model.

Architecture:
- Base model: facebook/wav2vec2-base (768-dim output)
- Fine-tuning strategy: Freeze first 8 layers, fine-tune last 4
- Output projection: 768 -> 512 dimensions

Expected Performance Gain: +5-10% AUC improvement over naive CNN

Usage:
    encoder = Wav2Vec2AudioEncoder(embed_dim=512, freeze_layers=8)
    waveform = torch.randn(4, 16000)  # 1 second at 16kHz
    features = encoder(waveform)  # [4, 512]
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except ImportError:
    Wav2Vec2Model = None
    Wav2Vec2Config = None

logger = logging.getLogger(__name__)


class Wav2Vec2AudioEncoder(nn.Module):
    """
    Wav2Vec2-based audio encoder for deepfake detection.
    
    Uses pre-trained wav2vec2 model to extract audio features from raw waveforms.
    Supports partial fine-tuning and dimensionality reduction.
    
    Args:
        model_name: HuggingFace model name (default: facebook/wav2vec2-base)
        embed_dim: Output embedding dimension (default: 512)
        freeze_layers: Number of transformer layers to freeze (default: 8)
        sample_rate: Expected audio sample rate in Hz (default: 16000)
        pooling_strategy: How to pool sequence outputs ('mean', 'max', 'cls')
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        embed_dim: int = 512,
        freeze_layers: int = 8,
        sample_rate: int = 16000,
        pooling_strategy: str = "mean",
    ):
        super().__init__()
        
        if Wav2Vec2Model is None:
            raise ImportError(
                "transformers library required for Wav2Vec2AudioEncoder. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.freeze_layers = freeze_layers
        self.sample_rate = sample_rate
        self.pooling_strategy = pooling_strategy
        
        # Load pre-trained wav2vec2 model
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
            logger.info(f"Loaded pre-trained model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Get model hidden size (768 for base, 1024 for large)
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # Freeze feature extractor (CNN layers)
        self._freeze_feature_extractor()
        
        # Freeze first N transformer layers
        self._freeze_transformer_layers(freeze_layers)
        
        # Projection layer to reduce dimensionality
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        logger.info(
            f"Initialized Wav2Vec2AudioEncoder: "
            f"hidden_size={self.hidden_size}, embed_dim={embed_dim}, "
            f"frozen_layers={freeze_layers}, pooling={pooling_strategy}"
        )
    
    def _freeze_feature_extractor(self):
        """Freeze the CNN feature extractor layers."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        logger.debug("Froze feature extractor (CNN layers)")
    
    def _freeze_transformer_layers(self, num_layers: int):
        """
        Freeze first N transformer encoder layers.
        
        Args:
            num_layers: Number of layers to freeze (0 = freeze none)
        """
        if num_layers <= 0:
            logger.debug("No transformer layers frozen")
            return
        
        total_layers = len(self.wav2vec2.encoder.layers)
        num_layers = min(num_layers, total_layers)
        
        for i in range(num_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False
        
        logger.debug(f"Froze first {num_layers}/{total_layers} transformer layers")
    
    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract audio features from raw waveform.
        
        Args:
            waveform: Raw audio waveform [B, T] where T is number of samples
                     Expected sample rate: 16kHz
            attention_mask: Optional mask for variable-length audio [B, T]
        
        Returns:
            features: Audio embeddings [B, embed_dim]
        
        Example:
            >>> encoder = Wav2Vec2AudioEncoder(embed_dim=512)
            >>> waveform = torch.randn(4, 16000)  # 4 samples, 1 second each
            >>> features = encoder(waveform)
            >>> features.shape
            torch.Size([4, 512])
        """
        # Ensure waveform is 2D [B, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Forward through wav2vec2
        outputs = self.wav2vec2(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        
        # Extract last hidden state [B, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling_strategy == "mean":
            # Mean pooling over time dimension
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling over time dimension
            pooled = hidden_states.max(dim=1)[0]
        
        elif self.pooling_strategy == "cls":
            # Use first token (CLS-like)
            pooled = hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Project to target embedding dimension
        features = self.projection(pooled)
        
        return features
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_trainable_layers(self) -> list:
        """
        Get list of trainable layer names.
        
        Returns:
            List of layer names that are trainable
        """
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
        return trainable


class SimpleAudioEncoder(nn.Module):
    """
    Fallback simple CNN-based audio encoder for mel-spectrograms.
    
    This is a lightweight alternative when wav2vec2 is not available
    or when computational resources are limited.
    
    Args:
        n_mels: Number of mel frequency bins (default: 64)
        embed_dim: Output embedding dimension (default: 256)
    """
    
    def __init__(self, n_mels: int = 64, embed_dim: int = 256):
        super().__init__()
        
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        
        self.conv_layers = nn.Sequential(
            # Conv1: [B, 1, n_mels, T] -> [B, 32, n_mels, T]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Conv2: [B, 32, n_mels, T] -> [B, 64, n_mels/2, T/2]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv3: [B, 64, n_mels/2, T/2] -> [B, 128, n_mels/4, T/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        logger.info(f"Initialized SimpleAudioEncoder: n_mels={n_mels}, embed_dim={embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from mel-spectrogram.
        
        Args:
            x: Mel-spectrogram [B, n_mels, T] or [B, 1, n_mels, T]
        
        Returns:
            features: Audio embeddings [B, embed_dim]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, n_mels, T]
        
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
