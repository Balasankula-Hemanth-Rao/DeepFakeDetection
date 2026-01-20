"""
Model Explainability and Interpretability Tools

This module provides explainability methods for understanding model predictions,
including Grad-CAM for visual explanations and attention visualization.

Usage:
    explainer = GradCAM(model, target_layer='video_backbone.blocks[-1]')
    saliency_map = explainer.generate_cam(image, target_class=1)
"""

import logging
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations by highlighting regions that contribute
    most to the model's prediction.
    
    Args:
        model: PyTorch model
        target_layer: Name of the target layer for CAM generation
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        
        # Hooks for gradient and activation
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Initialized GradCAM for layer: {target_layer}")
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target_module = self._find_layer(self.model, self.target_layer)
        
        if target_module is None:
            raise ValueError(f"Layer {self.target_layer} not found in model")
        
        # Register hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
    
    def _find_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find layer by name in model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor [B, C, H, W] or [B, T, C, H, W]
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Class activation map [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Handle temporal input
        if input_tensor.dim() == 5:
            # [B, T, C, H, W] -> process first frame
            input_tensor = input_tensor[:, 0, :, :, :]
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[:, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = None,
    ) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image [H, W, 3] in RGB, values [0, 255]
            cam: Class activation map [H, W] in range [0, 1]
            alpha: Blending factor (default: 0.5)
            colormap: OpenCV colormap (default: cv2.COLORMAP_JET)
        
        Returns:
            overlay: Blended image [H, W, 3]
        """
        if cv2 is None:
            raise ImportError("opencv-python required for CAM overlay")
        
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


class AttentionVisualizer:
    """
    Visualize attention weights from transformer models.
    
    Args:
        model: PyTorch model with attention mechanisms
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        
        # Register hooks for attention layers
        self._register_attention_hooks()
        
        logger.info("Initialized AttentionVisualizer")
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        
        def attention_hook(module, input, output):
            # Store attention weights
            if isinstance(output, tuple) and len(output) > 1:
                # MultiheadAttention returns (output, attention_weights)
                self.attention_weights[module] = output[1].detach()
        
        # Find all MultiheadAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(attention_hook)
                logger.debug(f"Registered attention hook for: {name}")
    
    def visualize_temporal_attention(
        self,
        num_frames: int,
    ) -> np.ndarray:
        """
        Visualize temporal attention weights.
        
        Args:
            num_frames: Number of frames in sequence
        
        Returns:
            attention_matrix: Attention weights [num_frames, num_frames]
        """
        # Get attention weights from last layer
        if not self.attention_weights:
            logger.warning("No attention weights captured")
            return np.zeros((num_frames, num_frames))
        
        # Get last attention layer
        last_attention = list(self.attention_weights.values())[-1]
        
        # Average over heads and batch
        attention_matrix = last_attention.mean(dim=0).mean(dim=0).cpu().numpy()
        
        return attention_matrix


class FeatureImportance:
    """
    Compute feature importance using gradient-based methods.
    
    Args:
        model: PyTorch model
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        logger.info("Initialized FeatureImportance")
    
    def compute_input_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute gradients with respect to input.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class (if None, use predicted class)
        
        Returns:
            gradients: Input gradients
        """
        self.model.eval()
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()
        
        # Get gradients
        gradients = input_tensor.grad.detach()
        
        return gradients
    
    def compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute integrated gradients for attribution.
        
        Args:
            input_tensor: Input tensor
            baseline: Baseline input (if None, use zeros)
            target_class: Target class
            steps: Number of integration steps
        
        Returns:
            attributions: Feature attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # Compute gradient
            grad = self.compute_input_gradients(interpolated, target_class)
            gradients.append(grad)
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Compute attributions
        attributions = (input_tensor - baseline) * avg_gradients
        
        return attributions


class SaliencyMapGenerator:
    """
    Generate saliency maps for visual explanations.
    
    Args:
        model: PyTorch model
        target_layer: Target layer for Grad-CAM
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.gradcam = GradCAM(model, target_layer)
        self.feature_importance = FeatureImportance(model)
        
        logger.info("Initialized SaliencyMapGenerator")
    
    def generate_saliency(
        self,
        input_tensor: torch.Tensor,
        method: str = "gradcam",
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate saliency map using specified method.
        
        Args:
            input_tensor: Input tensor
            method: Saliency method ('gradcam', 'gradient', 'integrated_gradient')
            target_class: Target class
        
        Returns:
            saliency_map: Saliency map [H, W]
        """
        if method == "gradcam":
            return self.gradcam.generate_cam(input_tensor, target_class)
        
        elif method == "gradient":
            gradients = self.feature_importance.compute_input_gradients(input_tensor, target_class)
            saliency = gradients.abs().sum(dim=1).squeeze().cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency
        
        elif method == "integrated_gradient":
            attributions = self.feature_importance.compute_integrated_gradients(input_tensor, target_class=target_class)
            saliency = attributions.abs().sum(dim=1).squeeze().cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            return saliency
        
        else:
            raise ValueError(f"Unknown saliency method: {method}")
    
    def save_saliency_overlay(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        output_path: str,
        alpha: float = 0.5,
    ):
        """
        Save saliency map overlay to file.
        
        Args:
            image: Original image [H, W, 3]
            saliency_map: Saliency map [H, W]
            output_path: Output file path
            alpha: Blending factor
        """
        overlay = self.gradcam.overlay_cam(image, saliency_map, alpha)
        
        if cv2 is None:
            raise ImportError("opencv-python required for saving images")
        
        # Convert RGB to BGR for OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)
        
        logger.info(f"Saved saliency overlay to: {output_path}")
