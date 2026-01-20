"""
Video Inference Engine

This module provides the core inference logic for video-level deepfake detection,
including frame aggregation, saliency generation, and result formatting.

Usage:
    engine = VideoInferenceEngine(model, config)
    result = await engine.analyze_video(frames, audio, sample_rate)
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VideoInferenceEngine:
    """
    Video-level inference engine for deepfake detection.
    
    Args:
        model: Loaded multimodal model
        config: Configuration dictionary
    """
    
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        logger.info(f"Initialized VideoInferenceEngine on device: {self.device}")
    
    async def analyze_video(
        self,
        frames: List[np.ndarray],
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze video for deepfake detection.
        
        Args:
            frames: List of video frames [H, W, 3]
            audio: Audio waveform [samples]
            sample_rate: Audio sample rate
            progress_callback: Optional callback for progress updates
        
        Returns:
            result: Analysis result dictionary
        """
        logger.info(f"Analyzing video: {len(frames)} frames, {len(audio)/sample_rate:.2f}s audio")
        
        # Preprocess frames
        if progress_callback:
            progress_callback(0.1)
        
        frame_tensors = self._preprocess_frames(frames)
        
        # Preprocess audio
        if progress_callback:
            progress_callback(0.2)
        
        audio_tensor = self._preprocess_audio(audio, sample_rate)
        
        # Run frame-level inference
        if progress_callback:
            progress_callback(0.3)
        
        frame_predictions = self._predict_frames(frame_tensors, audio_tensor)
        
        # Aggregate predictions
        if progress_callback:
            progress_callback(0.7)
        
        video_prediction = self._aggregate_predictions(frame_predictions)
        
        # Identify anomalous frames
        if progress_callback:
            progress_callback(0.8)
        
        anomalous_frames = self._identify_anomalous_frames(frame_predictions)
        
        # Format result
        result = {
            'prediction': 'fake' if video_prediction['class'] == 1 else 'real',
            'confidence': float(video_prediction['confidence']),
            'frame_predictions': [
                {
                    'frame_idx': i,
                    'prediction': 'fake' if pred['class'] == 1 else 'real',
                    'confidence': float(pred['confidence']),
                }
                for i, pred in enumerate(frame_predictions)
            ],
            'anomalous_frames': anomalous_frames,
            'metadata': {
                'num_frames': len(frames),
                'audio_duration': len(audio) / sample_rate,
                'sample_rate': sample_rate,
            }
        }
        
        if progress_callback:
            progress_callback(1.0)
        
        return result
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess frames for model input.
        
        Args:
            frames: List of frames [H, W, 3]
        
        Returns:
            frame_tensors: Preprocessed frames [B, T, 3, H, W]
        """
        from torchvision import transforms
        
        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Process frames
        processed_frames = []
        for frame in frames:
            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            tensor = preprocess(frame)
            processed_frames.append(tensor)
        
        # Stack into batch [T, 3, H, W]
        frame_tensors = torch.stack(processed_frames)
        
        # Add batch dimension [1, T, 3, H, W]
        frame_tensors = frame_tensors.unsqueeze(0)
        
        return frame_tensors.to(self.device)
    
    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio waveform [samples]
            sample_rate: Sample rate
        
        Returns:
            audio_tensor: Preprocessed audio
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Normalize
        audio_tensor = audio_tensor / (torch.abs(audio_tensor).max() + 1e-8)
        
        # Add batch dimension [1, samples]
        audio_tensor = audio_tensor.unsqueeze(0)
        
        return audio_tensor.to(self.device)
    
    def _predict_frames(
        self,
        frame_tensors: torch.Tensor,
        audio_tensor: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on frames.
        
        Args:
            frame_tensors: Preprocessed frames [1, T, 3, H, W]
            audio_tensor: Preprocessed audio [1, samples]
        
        Returns:
            predictions: List of per-frame predictions
        """
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            # Process in temporal windows
            window_size = self.config.get('temporal_window', 16)
            num_frames = frame_tensors.size(1)
            
            for i in range(0, num_frames, window_size):
                end_idx = min(i + window_size, num_frames)
                window_frames = frame_tensors[:, i:end_idx, :, :, :]
                
                # Compute corresponding audio segment
                # TODO: Implement proper audio-video alignment
                
                # Run model
                logits = self.model(video=window_frames, audio=audio_tensor)
                probs = F.softmax(logits, dim=1)
                
                # Extract predictions for each frame in window
                for j in range(window_frames.size(1)):
                    pred_class = probs[0].argmax().item()
                    confidence = probs[0, pred_class].item()
                    
                    predictions.append({
                        'class': pred_class,
                        'confidence': confidence,
                        'logits': logits[0].cpu().numpy(),
                    })
        
        return predictions
    
    def _aggregate_predictions(
        self,
        frame_predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate frame-level predictions to video-level.
        
        Args:
            frame_predictions: List of frame predictions
        
        Returns:
            video_prediction: Aggregated video-level prediction
        """
        aggregation_strategy = self.config.get('aggregation', 'mean')
        
        if aggregation_strategy == 'mean':
            # Mean of confidences
            confidences = [pred['confidence'] for pred in frame_predictions]
            classes = [pred['class'] for pred in frame_predictions]
            
            # Majority vote for class
            video_class = max(set(classes), key=classes.count)
            
            # Mean confidence for that class
            class_confidences = [
                conf for conf, cls in zip(confidences, classes)
                if cls == video_class
            ]
            video_confidence = np.mean(class_confidences)
        
        elif aggregation_strategy == 'max':
            # Max confidence
            max_pred = max(frame_predictions, key=lambda x: x['confidence'])
            video_class = max_pred['class']
            video_confidence = max_pred['confidence']
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
        
        return {
            'class': video_class,
            'confidence': video_confidence,
        }
    
    def _identify_anomalous_frames(
        self,
        frame_predictions: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[int]:
        """
        Identify most anomalous frames.
        
        Args:
            frame_predictions: List of frame predictions
            top_k: Number of top anomalous frames to return
        
        Returns:
            anomalous_indices: Indices of anomalous frames
        """
        # Sort by confidence in 'fake' class
        fake_confidences = [
            (i, pred['confidence'] if pred['class'] == 1 else 1 - pred['confidence'])
            for i, pred in enumerate(frame_predictions)
        ]
        
        # Sort descending
        fake_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k indices
        anomalous_indices = [idx for idx, _ in fake_confidences[:top_k]]
        
        return anomalous_indices
    
    async def generate_saliency_maps(
        self,
        frames: List[np.ndarray],
        anomalous_indices: List[int],
    ) -> List[str]:
        """
        Generate saliency maps for anomalous frames.
        
        Args:
            frames: List of video frames
            anomalous_indices: Indices of anomalous frames
        
        Returns:
            saliency_urls: List of URLs to saliency map images
        """
        from ..models.explainability import SaliencyMapGenerator
        
        # Initialize saliency generator
        saliency_gen = SaliencyMapGenerator(
            model=self.model,
            target_layer='video_backbone.blocks[-1]',
        )
        
        saliency_urls = []
        
        for idx in anomalous_indices:
            if idx >= len(frames):
                continue
            
            frame = frames[idx]
            
            # Preprocess frame
            frame_tensor = self._preprocess_frames([frame])
            
            # Generate saliency
            saliency_map = saliency_gen.generate_saliency(
                frame_tensor,
                method='gradcam',
                target_class=1,  # Fake class
            )
            
            # Save saliency overlay
            output_path = Path(f"saliency/frame_{idx}_saliency.png")
            output_path.parent.mkdir(exist_ok=True)
            
            saliency_gen.save_saliency_overlay(
                frame,
                saliency_map,
                str(output_path),
            )
            
            # In production, upload to cloud storage and return URL
            saliency_url = f"/saliency/frame_{idx}_saliency.png"
            saliency_urls.append(saliency_url)
        
        logger.info(f"Generated {len(saliency_urls)} saliency maps")
        
        return saliency_urls
