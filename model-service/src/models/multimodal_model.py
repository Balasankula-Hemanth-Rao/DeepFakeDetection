"""
Multimodal Deepfake Detection Model

Combines video and audio modalities for robust deepfake detection.

Architecture:
- Video backbone: EfficientNet or other timm model
- Video temporal encoder: Average pooling, temporal convolution, or transformer
- Audio encoder: Small CNN on mel-spectrogram
- Fusion head: Concatenation, attention, or cross-modal fusion
- Classification head: 2-class output (real/fake)

Usage:
    config = get_config()
    model = MultimodalModel(config)
    model.to(device)
    
    # Forward pass
    logits = model(video, audio)
    
    # Feature extraction
    video_feat, audio_feat = model.extract_features(video, audio)
    
    # Load checkpoint
    model = MultimodalModel.load_for_inference('checkpoint.pth', device='cuda')
    
    # Computing auxiliary losses
    loss_main = criterion(logits, targets)
    temporal_loss = model.compute_temporal_consistency_loss(video)
    total_loss = loss_main + 0.1 * temporal_loss  # Weighted combination
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

from .losses import TemporalConsistencyLoss


logger = logging.getLogger(__name__)


class TemporalConv(nn.Module):
    """Small 1D temporal convolution module."""
    
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] where T is temporal dimension
        
        Returns:
            [B, out_dim]
        """
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        # [B, D, T] -> [B, D, 1] -> [B, D]
        x = self.pool(x)
        x = x.squeeze(-1)
        return x


class AudioCNN(nn.Module):
    """Small CNN for mel-spectrogram audio encoding."""
    
    def __init__(self, n_mels: int = 64, embed_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Estimate flattened size after convs
        # Rough estimate: (n_mels -> n_mels/4) x (time_steps/4)
        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_mels, T] or [B, 1, n_mels, T]
        
        Returns:
            [B, embed_dim]
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, n_mels, T]
        
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MultimodalModel(nn.Module):
    """
    Multimodal deepfake detection model combining video and audio.
    
    Supports modality ablation studies via enable_audio and enable_video flags.
    
    Args:
        config: Configuration object with model parameters
        num_classes: Number of output classes (default 2: real/fake)
        enable_video: Whether to use video modality (default True)
        enable_audio: Whether to use audio modality (default True)
    """
    
    def __init__(self, config=None, num_classes: int = 2, enable_video: bool = True, enable_audio: bool = True):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.enable_video = enable_video
        self.enable_audio = enable_audio
        
        # Validate: at least one modality must be enabled
        if not (enable_video or enable_audio):
            raise ValueError("At least one modality (video or audio) must be enabled")
        
        # Extract config parameters
        self._setup_config()
        
        # Video backbone (optional)
        if self.enable_video:
            self.video_backbone = self._build_video_backbone()
            self.temporal_encoder = self._build_temporal_encoder()
        else:
            self.video_backbone = None
            self.temporal_encoder = None
        
        # Audio encoder (optional)
        if self.enable_audio:
            self.audio_encoder = self._build_audio_encoder()
        else:
            self.audio_encoder = None
        
        # Fusion head (may be identity if only one modality)
        self.fusion_head = self._build_fusion_head()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        
        # Temporal consistency loss (optional auxiliary loss)
        if self.temporal_consistency_enabled and self.enable_video:
            self.temporal_consistency_loss_fn = TemporalConsistencyLoss()
            logger.info(
                f"Temporal consistency loss enabled with weight={self.temporal_consistency_weight}"
            )
        else:
            self.temporal_consistency_loss_fn = None
        
        modality_str = f"video={enable_video}, audio={enable_audio}"
        logger.info(f"Initialized MultimodalModel ({modality_str}) with {self.count_parameters()} parameters")
    
    def _setup_config(self):
        """Extract configuration parameters."""
        if self.config is None:
            # Fallback defaults
            self.video_backbone_name = 'efficientnet_b3'
            self.video_pretrained = True
            self.video_embed_dim = 1536
            self.temporal_strategy = 'avg_pool'
            self.audio_encoder_type = 'simple'  # NEW: 'simple' or 'wav2vec2'
            self.audio_sample_rate = 16000
            self.audio_n_mels = 64
            self.audio_embed_dim = 256
            self.fusion_strategy = 'concat'
            self.hidden_dim = 512
            self.dropout = 0.3
            self.modality_dropout_prob = 0.0
            self.temporal_consistency_enabled = False
            self.temporal_consistency_weight = 0.1
        else:
            # Extract from config
            try:
                # Try to get model config (could be object or dict)
                model_cfg = getattr(self.config, 'model', None)
                if model_cfg is None:
                    model_cfg = {}
                
                # Handle case where model_cfg is an object
                if hasattr(model_cfg, '__dict__'):
                    model_cfg_dict = vars(model_cfg)
                elif isinstance(model_cfg, dict):
                    model_cfg_dict = model_cfg
                else:
                    model_cfg_dict = {}
                
                # Get video config
                video_cfg = model_cfg_dict.get('video', {})
                if hasattr(video_cfg, '__dict__'):
                    video_cfg_dict = vars(video_cfg)
                elif isinstance(video_cfg, dict):
                    video_cfg_dict = video_cfg
                else:
                    video_cfg_dict = {}
                
                self.video_backbone_name = video_cfg_dict.get('backbone', 'efficientnet_b3')
                self.video_pretrained = video_cfg_dict.get('pretrained', True)
                self.video_embed_dim = video_cfg_dict.get('embed_dim', 1536)
                self.temporal_strategy = video_cfg_dict.get('temporal_strategy', 'avg_pool')
                
                # Get audio config
                audio_cfg = getattr(self.config, 'audio', {}) if hasattr(self.config, 'audio') else {}
                if hasattr(audio_cfg, '__dict__'):
                    audio_cfg_dict = vars(audio_cfg)
                elif isinstance(audio_cfg, dict):
                    audio_cfg_dict = audio_cfg
                else:
                    audio_cfg_dict = {}
                    
                self.audio_encoder_type = audio_cfg_dict.get('encoder_type', 'simple')  # NEW
                self.audio_sample_rate = audio_cfg_dict.get('sample_rate', 16000)
                self.audio_n_mels = audio_cfg_dict.get('n_mels', 64)
                self.audio_embed_dim = audio_cfg_dict.get('embed_dim', 256)
                
                # Get fusion config
                fusion_cfg = model_cfg_dict.get('fusion', {})
                if hasattr(fusion_cfg, '__dict__'):
                    fusion_cfg_dict = vars(fusion_cfg)
                elif isinstance(fusion_cfg, dict):
                    fusion_cfg_dict = fusion_cfg
                else:
                    fusion_cfg_dict = {}
                
                self.fusion_strategy = fusion_cfg_dict.get('strategy', 'concat')
                self.hidden_dim = fusion_cfg_dict.get('hidden_dim', 512)
                self.dropout = fusion_cfg_dict.get('dropout', 0.3)
                self.modality_dropout_prob = fusion_cfg_dict.get('modality_dropout_prob', 0.0)
                
                # Get temporal consistency loss config
                temporal_cfg = fusion_cfg_dict.get('temporal_consistency_loss', {})
                if hasattr(temporal_cfg, '__dict__'):
                    temporal_cfg_dict = vars(temporal_cfg)
                elif isinstance(temporal_cfg, dict):
                    temporal_cfg_dict = temporal_cfg
                else:
                    temporal_cfg_dict = {}
                    
                self.temporal_consistency_enabled = temporal_cfg_dict.get('enabled', False)
                self.temporal_consistency_weight = temporal_cfg_dict.get('weight', 0.1)
            except Exception as e:
                # Fallback to defaults if anything goes wrong
                logger.warning(f"Error reading config, using defaults: {e}")
                self.video_backbone_name = 'efficientnet_b3'
                self.video_pretrained = True
                self.video_embed_dim = 1536
                self.temporal_strategy = 'avg_pool'
                self.audio_encoder_type = 'simple'
                self.audio_sample_rate = 16000
                self.audio_n_mels = 64
                self.audio_embed_dim = 256
                self.fusion_strategy = 'concat'
                self.hidden_dim = 512
                self.dropout = 0.3
                self.modality_dropout_prob = 0.0
                self.temporal_consistency_enabled = False
                self.temporal_consistency_weight = 0.1
        
        # Compute fusion input dimension based on enabled modalities
        if self.enable_video and self.enable_audio:
            # Both modalities enabled
            if self.fusion_strategy == 'concat':
                self.fusion_dim = self.video_embed_dim + self.audio_embed_dim
            elif self.fusion_strategy == 'attention':
                self.fusion_dim = max(self.video_embed_dim, self.audio_embed_dim)
            else:
                self.fusion_dim = self.video_embed_dim + self.audio_embed_dim
        elif self.enable_video:
            # Video-only
            self.fusion_dim = self.video_embed_dim
        elif self.enable_audio:
            # Audio-only
            self.fusion_dim = self.audio_embed_dim
        else:
            raise ValueError("At least one modality must be enabled")
    
    def _build_video_backbone(self) -> nn.Module:
        """Build video backbone using timm."""
        if timm is None:
            raise ImportError("timm required for video backbone. Install: pip install timm")
        
        try:
            model = timm.create_model(
                self.video_backbone_name,
                pretrained=self.video_pretrained,
                num_classes=0,  # Remove classification head
            )
            logger.info(f"Loaded video backbone: {self.video_backbone_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load backbone {self.video_backbone_name}: {e}")
            # Fallback to efficientnet_b3
            model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
            logger.warning(f"Fallback to efficientnet_b3. Original error: {e}")
            return model
    
    def _build_temporal_encoder(self) -> nn.Module:
        """Build temporal encoder for video frames."""
        if self.temporal_strategy == 'avg_pool':
            return nn.AdaptiveAvgPool1d(1)
        elif self.temporal_strategy == 'tconv':
            return TemporalConv(self.video_embed_dim, self.video_embed_dim)
        elif self.temporal_strategy == 'transformer':
            # Simple transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.video_embed_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            )
            return nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            logger.warning(f"Unknown temporal strategy: {self.temporal_strategy}. Using avg_pool.")
            return nn.AdaptiveAvgPool1d(1)
    
    def _build_audio_encoder(self) -> nn.Module:
        """Build audio encoder."""
        if self.audio_encoder_type == 'wav2vec2':
            # Use pre-trained Wav2Vec2 encoder
            try:
                from .audio_encoder import Wav2Vec2AudioEncoder
                logger.info("Using Wav2Vec2 audio encoder")
                return Wav2Vec2AudioEncoder(
                    model_name="facebook/wav2vec2-base",
                    embed_dim=self.audio_embed_dim,
                    freeze_layers=8,
                    sample_rate=self.audio_sample_rate,
                )
            except ImportError:
                logger.warning("Wav2Vec2AudioEncoder not available, falling back to AudioCNN")
                return AudioCNN(n_mels=self.audio_n_mels, embed_dim=self.audio_embed_dim)
        else:
            # Use simple CNN encoder
            return AudioCNN(n_mels=self.audio_n_mels, embed_dim=self.audio_embed_dim)
    
    def _build_fusion_head(self) -> nn.Module:
        """Build fusion module."""
        if self.fusion_strategy == 'concat':
            return nn.Identity()  # No fusion module needed
        elif self.fusion_strategy == 'attention':
            # Use advanced cross-modal attention fusion
            try:
                from .fusion import CrossModalAttentionFusion
                logger.info("Using CrossModalAttentionFusion")
                return CrossModalAttentionFusion(
                    video_dim=self.video_embed_dim,
                    audio_dim=self.audio_embed_dim,
                    output_dim=self.fusion_dim,
                    num_heads=4,
                    dropout=self.dropout,
                )
            except ImportError:
                logger.warning("CrossModalAttentionFusion not available, using simple attention")
                return nn.MultiheadAttention(
                    embed_dim=max(self.video_embed_dim, self.audio_embed_dim),
                    num_heads=4,
                    dropout=self.dropout,
                    batch_first=True,
                )
        elif self.fusion_strategy == 'gated':
            # Use gated fusion
            try:
                from .fusion import GatedFusion
                logger.info("Using GatedFusion")
                return GatedFusion(
                    video_dim=self.video_embed_dim,
                    audio_dim=self.audio_embed_dim,
                    output_dim=self.fusion_dim,
                    dropout=self.dropout,
                )
            except ImportError:
                logger.warning("GatedFusion not available, falling back to concat")
                return nn.Identity()
        elif self.fusion_strategy == 'transformer':
            # Use transformer-based fusion
            try:
                from .fusion import BimodalTransformerFusion
                logger.info("Using BimodalTransformerFusion")
                return BimodalTransformerFusion(
                    video_dim=self.video_embed_dim,
                    audio_dim=self.audio_embed_dim,
                    output_dim=self.fusion_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=self.dropout,
                )
            except ImportError:
                logger.warning("BimodalTransformerFusion not available, falling back to concat")
                return nn.Identity()
        else:
            return nn.Identity()
    
    def forward(self, video: Optional[torch.Tensor] = None, audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            video: [B, T, 3, H, W] where T is temporal dimension (required if enable_video=True)
            audio: [B, n_mels, time_steps] or [B, 1, n_mels, time_steps] (required if enable_audio=True)
        
        Returns:
            logits: [B, num_classes]
        """
        video_feat, audio_feat = self.extract_features(video, audio)
        
        # Apply modality-level dropout during training only
        if self.training and self.modality_dropout_prob > 0.0:
            video_feat, audio_feat = self._apply_modality_dropout(video_feat, audio_feat)
        
        # Fusion based on enabled modalities
        if self.enable_video and self.enable_audio:
            # Both modalities: fuse them
            if self.fusion_strategy == 'concat':
                fused = torch.cat([video_feat, audio_feat], dim=1)
            elif self.fusion_strategy == 'attention':
                # Simple attention-based fusion
                video_feat_exp = video_feat.unsqueeze(1)
                audio_feat_exp = audio_feat.unsqueeze(1)
                fused, _ = self.fusion_head(video_feat_exp, audio_feat_exp, audio_feat_exp)
                fused = fused.squeeze(1)
            else:
                fused = torch.cat([video_feat, audio_feat], dim=1)
        elif self.enable_video:
            # Video-only
            fused = video_feat
        elif self.enable_audio:
            # Audio-only
            fused = audio_feat
        else:
            raise RuntimeError("At least one modality must be enabled")
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def extract_features(self, video: Optional[torch.Tensor] = None, audio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from video and audio separately.
        
        Args:
            video: [B, T, 3, H, W] or None if enable_video=False
            audio: [B, n_mels, T] or None if enable_audio=False
        
        Returns:
            video_feat: [B, video_embed_dim] (None if enable_video=False)
            audio_feat: [B, audio_embed_dim] (None if enable_audio=False)
        """
        video_feat = None
        audio_feat = None
        
        # Process video frames if enabled
        if self.enable_video:
            if video is None:
                raise ValueError("Video tensor required when enable_video=True")
            video_feat = self._process_video(video)
        
        # Process audio if enabled
        if self.enable_audio:
            if audio is None:
                raise ValueError("Audio tensor required when enable_audio=True")
            audio_feat = self.audio_encoder(audio)
        
        return video_feat, audio_feat
    
    def _process_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Process video frames through backbone and temporal encoder.
        
        Args:
            video: [B, T, 3, H, W]
        
        Returns:
            features: [B, video_embed_dim]
        """
        B, T, C, H, W = video.shape
        
        # Reshape to process all frames together
        video = video.reshape(B * T, C, H, W)
        
        # Extract per-frame features
        frame_feats = self.video_backbone(video)  # [B*T, D]
        
        # Reshape back for temporal encoding
        frame_feats = frame_feats.reshape(B, T, -1)  # [B, T, D]
        
        # Apply temporal encoder
        if self.temporal_strategy == 'avg_pool':
            # [B, T, D] -> [B, D, T] -> [B, D, 1] -> [B, D]
            video_feat = frame_feats.transpose(1, 2)
            video_feat = self.temporal_encoder(video_feat)
            video_feat = video_feat.squeeze(-1)
        elif self.temporal_strategy == 'transformer':
            # Transformer expects [B, T, D]
            video_feat = self.temporal_encoder(frame_feats)
            # Take mean over temporal dimension
            video_feat = video_feat.mean(dim=1)
        else:
            # tconv or other
            video_feat = self.temporal_encoder(frame_feats)  # [B, D]
        
        return video_feat
    
    def _apply_modality_dropout(
        self,
        video_feat: Optional[torch.Tensor],
        audio_feat: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply modality-level dropout during training.
        
        Randomly drops audio or video features to prevent over-reliance on a single modality.
        Only applied during training (self.training=True). Inference is deterministic.
        
        Args:
            video_feat: Video features [B, video_embed_dim] or None
            audio_feat: Audio features [B, audio_embed_dim] or None
        
        Returns:
            video_feat: Video features (possibly masked) or None
            audio_feat: Audio features (possibly masked) or None
        """
        # Only apply during training
        if not self.training or self.modality_dropout_prob <= 0.0:
            return video_feat, audio_feat
        
        # Only apply if both modalities are enabled
        if not (self.enable_video and self.enable_audio):
            return video_feat, audio_feat
        
        # Randomly decide which modality to drop (if any)
        dropout_rand = torch.rand(1).item()
        
        if dropout_rand < self.modality_dropout_prob:
            # Decide which modality to drop (50/50 split)
            drop_video_prob = 0.5
            
            if torch.rand(1).item() < drop_video_prob and video_feat is not None:
                # Drop video: zero out video features
                video_feat = torch.zeros_like(video_feat)
                logger.debug("Applied modality dropout: zeroed VIDEO features")
            elif audio_feat is not None:
                # Drop audio: zero out audio features
                audio_feat = torch.zeros_like(audio_feat)
                logger.debug("Applied modality dropout: zeroed AUDIO features")
        
        return video_feat, audio_feat
    
    def compute_temporal_consistency_loss(self, video: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute temporal consistency loss from video frames.
        
        This loss penalizes high variance across adjacent frame embeddings,
        encouraging smooth temporal consistency in the learned representations.
        
        Args:
            video: [B, T, 3, H, W] video frames
        
        Returns:
            loss: scalar tensor or None if loss is disabled or video not provided
        
        Example:
            >>> video = torch.randn(4, 16, 3, 224, 224)
            >>> loss = model.compute_temporal_consistency_loss(video)
            >>> if loss is not None:
            ...     total_loss = criterion_loss + 0.1 * loss
        """
        if not self.temporal_consistency_enabled or self.temporal_consistency_loss_fn is None:
            return None
        
        if video is None:
            return None
        
        B, T, C, H, W = video.shape
        
        # Process all frames through backbone to get per-frame embeddings
        video_reshaped = video.reshape(B * T, C, H, W)
        frame_embeddings = self.video_backbone(video_reshaped)  # [B*T, D]
        
        # Reshape back to [B, T, D]
        frame_embeddings = frame_embeddings.reshape(B, T, -1)
        
        # Compute temporal consistency loss
        temporal_loss = self.temporal_consistency_loss_fn(frame_embeddings)
        
        return temporal_loss
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def load_for_inference(
        cls,
        checkpoint_path: str,
        config=None,
        device: str = 'cuda',
    ) -> 'MultimodalModel':
        """
        Load model from checkpoint for inference.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object (optional)
            device: Device to load model on
        
        Returns:
            model: Loaded model in eval mode
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract config if available
        if config is None and 'config' in checkpoint:
            config = checkpoint['config']
        
        # Instantiate model
        model = cls(config=config)
        model.to(device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}. Attempting non-strict load.")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return model
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        epoch: int = 0,
        metrics: Optional[dict] = None,
    ):
        """
        Save model checkpoint with metadata.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch number
            metrics: Optional metrics dict
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'model_name': self.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
