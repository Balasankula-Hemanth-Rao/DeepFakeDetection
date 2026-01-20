"""
Models package for multimodal deepfake detection.

This package contains all model architectures and components:
- Video encoders (EfficientNet, ResNet)
- Audio encoders (Wav2Vec2, CNN)
- Temporal encoders (Transformer, TCN)
- Fusion modules (Attention, Gated, Transformer)
- Multimodal model (combined architecture)
- Loss functions
- Explainability tools
"""

from .frame_model import FrameModel
from .multimodal_model import MultimodalModel
from .losses import TemporalConsistencyLoss
from .audio_encoder import Wav2Vec2AudioEncoder, SimpleAudioEncoder
from .fusion import (
    CrossModalAttentionFusion,
    GatedFusion,
    BimodalTransformerFusion,
    ConcatenationFusion,
)
from .temporal_transformer import TemporalTransformer, TemporalConvNet
from .explainability import GradCAM, SaliencyMapGenerator, AttentionVisualizer

__all__ = [
    # Core models
    'FrameModel',
    'MultimodalModel',
    
    # Audio encoders
    'Wav2Vec2AudioEncoder',
    'SimpleAudioEncoder',
    
    # Fusion modules
    'CrossModalAttentionFusion',
    'GatedFusion',
    'BimodalTransformerFusion',
    'ConcatenationFusion',
    
    # Temporal encoders
    'TemporalTransformer',
    'TemporalConvNet',
    
    # Loss functions
    'TemporalConsistencyLoss',
    
    # Explainability
    'GradCAM',
    'SaliencyMapGenerator',
    'AttentionVisualizer',
]
