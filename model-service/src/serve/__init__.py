"""
Serving package for model inference APIs.

This package contains serving and inference components:
- Frame-level inference API
- Video-level async inference API
- Inference engine with aggregation
- Ensemble model serving
"""

# Lazy imports - only load when needed to avoid torch dependency in simulation mode
def __getattr__(name):
    if name == 'VideoInferenceEngine':
        from .inference import VideoInferenceEngine
        return VideoInferenceEngine
    elif name == 'EnsembleModel':
        from .ensemble import EnsembleModel
        return EnsembleModel
    elif name == 'EnsembleInferenceEngine':
        from .ensemble import EnsembleInferenceEngine
        return EnsembleInferenceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'VideoInferenceEngine',
    'EnsembleModel',
    'EnsembleInferenceEngine',
]
