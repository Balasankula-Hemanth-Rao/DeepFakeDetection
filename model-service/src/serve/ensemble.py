"""
Ensemble Model Serving

This module provides ensemble inference by combining predictions
from multiple trained models for improved robustness and accuracy.

Expected Performance Gain: +2-4% AUC improvement

Usage:
    ensemble = EnsembleModel(checkpoint_paths=['model1.pth', 'model2.pth'])
    prediction = ensemble.predict(video, audio)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble of multiple deepfake detection models.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        device: Device to load models on
        ensemble_strategy: How to combine predictions ('mean', 'weighted', 'voting')
        weights: Optional weights for weighted ensemble
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        device: str = 'cuda',
        ensemble_strategy: str = 'mean',
        weights: Optional[List[float]] = None,
    ):
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.device = device
        self.ensemble_strategy = ensemble_strategy
        
        # Validate weights
        if weights is not None:
            if len(weights) != len(checkpoint_paths):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
        else:
            # Equal weights
            self.weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
        
        # Load models
        self.models = self._load_models()
        
        logger.info(
            f"Initialized EnsembleModel with {len(self.models)} models, "
            f"strategy={ensemble_strategy}"
        )
    
    def _load_models(self) -> List:
        """Load all model checkpoints."""
        from ..models.multimodal_model import MultimodalModel
        
        models = []
        
        for i, checkpoint_path in enumerate(self.checkpoint_paths):
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
                continue
            
            try:
                model = MultimodalModel.load_for_inference(
                    str(checkpoint_path),
                    device=self.device,
                )
                model.eval()
                models.append(model)
                logger.info(f"Loaded model {i+1}/{len(self.checkpoint_paths)}: {checkpoint_path.name}")
            
            except Exception as e:
                logger.error(f"Failed to load {checkpoint_path}: {e}")
                continue
        
        if not models:
            raise RuntimeError("No models loaded successfully")
        
        return models
    
    def predict(
        self,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run ensemble prediction.
        
        Args:
            video: Video tensor [B, T, 3, H, W]
            audio: Audio tensor [B, samples]
        
        Returns:
            prediction: Ensemble prediction dictionary
        """
        # Collect predictions from all models
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(video=video, audio=audio)
                probs = F.softmax(logits, dim=1)
                
                all_logits.append(logits)
                all_probs.append(probs)
        
        # Combine predictions
        if self.ensemble_strategy == 'mean':
            ensemble_probs = self._mean_ensemble(all_probs)
        
        elif self.ensemble_strategy == 'weighted':
            ensemble_probs = self._weighted_ensemble(all_probs)
        
        elif self.ensemble_strategy == 'voting':
            ensemble_probs = self._voting_ensemble(all_probs)
        
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")
        
        # Extract prediction
        pred_class = ensemble_probs.argmax(dim=1).item()
        confidence = ensemble_probs[0, pred_class].item()
        
        result = {
            'class': pred_class,
            'prediction': 'fake' if pred_class == 1 else 'real',
            'confidence': confidence,
            'probabilities': ensemble_probs[0].cpu().numpy(),
            'individual_predictions': [
                {
                    'class': probs.argmax(dim=1).item(),
                    'confidence': probs.max(dim=1).values.item(),
                }
                for probs in all_probs
            ],
        }
        
        return result
    
    def _mean_ensemble(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Mean ensemble: average probabilities.
        
        Args:
            all_probs: List of probability tensors [B, num_classes]
        
        Returns:
            ensemble_probs: Averaged probabilities [B, num_classes]
        """
        stacked_probs = torch.stack(all_probs, dim=0)  # [num_models, B, num_classes]
        ensemble_probs = stacked_probs.mean(dim=0)  # [B, num_classes]
        return ensemble_probs
    
    def _weighted_ensemble(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Weighted ensemble: weighted average of probabilities.
        
        Args:
            all_probs: List of probability tensors [B, num_classes]
        
        Returns:
            ensemble_probs: Weighted averaged probabilities [B, num_classes]
        """
        ensemble_probs = torch.zeros_like(all_probs[0])
        
        for weight, probs in zip(self.weights, all_probs):
            ensemble_probs += weight * probs
        
        return ensemble_probs
    
    def _voting_ensemble(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Voting ensemble: majority vote.
        
        Args:
            all_probs: List of probability tensors [B, num_classes]
        
        Returns:
            ensemble_probs: Voting-based probabilities [B, num_classes]
        """
        # Get predicted classes
        pred_classes = [probs.argmax(dim=1) for probs in all_probs]
        
        # Count votes for each class
        batch_size = all_probs[0].size(0)
        num_classes = all_probs[0].size(1)
        
        ensemble_probs = torch.zeros(batch_size, num_classes, device=self.device)
        
        for b in range(batch_size):
            votes = [pred[b].item() for pred in pred_classes]
            
            # Count votes
            for cls in range(num_classes):
                ensemble_probs[b, cls] = votes.count(cls) / len(votes)
        
        return ensemble_probs
    
    def get_model_agreement(
        self,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute agreement between models.
        
        Args:
            video: Video tensor
            audio: Audio tensor
        
        Returns:
            agreement: Agreement score [0, 1]
        """
        # Get predictions from all models
        pred_classes = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(video=video, audio=audio)
                pred_class = logits.argmax(dim=1).item()
                pred_classes.append(pred_class)
        
        # Compute agreement (fraction of models agreeing with majority)
        majority_class = max(set(pred_classes), key=pred_classes.count)
        agreement = pred_classes.count(majority_class) / len(pred_classes)
        
        return agreement
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters for each model.
        
        Returns:
            param_counts: Dictionary of parameter counts
        """
        param_counts = {}
        
        for i, model in enumerate(self.models):
            count = sum(p.numel() for p in model.parameters())
            param_counts[f'model_{i}'] = count
        
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts


class EnsembleInferenceEngine:
    """
    Inference engine using ensemble model.
    
    Args:
        ensemble: EnsembleModel instance
        config: Configuration dictionary
    """
    
    def __init__(self, ensemble: EnsembleModel, config: dict):
        self.ensemble = ensemble
        self.config = config
        
        logger.info("Initialized EnsembleInferenceEngine")
    
    def analyze_video(
        self,
        frames: List,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """
        Analyze video using ensemble.
        
        Args:
            frames: List of video frames
            audio: Audio waveform
            sample_rate: Sample rate
        
        Returns:
            result: Analysis result
        """
        # Preprocess inputs
        from .inference import VideoInferenceEngine
        
        # Use first model for preprocessing
        temp_engine = VideoInferenceEngine(self.ensemble.models[0], self.config)
        
        frame_tensors = temp_engine._preprocess_frames(frames)
        audio_tensor = temp_engine._preprocess_audio(audio, sample_rate)
        
        # Run ensemble prediction
        prediction = self.ensemble.predict(video=frame_tensors, audio=audio_tensor)
        
        # Get model agreement
        agreement = self.ensemble.get_model_agreement(video=frame_tensors, audio=audio_tensor)
        
        result = {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'model_agreement': agreement,
            'individual_predictions': prediction['individual_predictions'],
            'metadata': {
                'num_models': len(self.ensemble.models),
                'ensemble_strategy': self.ensemble.ensemble_strategy,
                'num_frames': len(frames),
                'audio_duration': len(audio) / sample_rate,
            }
        }
        
        return result
