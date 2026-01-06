"""
Frame Classification Model

A PyTorch model for binary frame classification (real vs. fake) using
EfficientNet-B3 as backbone with a custom classifier head.

The model is designed for video deepfake detection, processing individual
frames to classify them as authentic or synthetic.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional


class FrameModel(nn.Module):
    """
    Binary frame classifier using EfficientNet-B3 backbone.

    This model takes video frames as input and classifies them as either
    real (authentic) or fake (synthetic). It uses a pretrained EfficientNet-B3
    as the feature extractor followed by a simple classifier head.

    Architecture:
        - Backbone: EfficientNet-B3 (pretrained on ImageNet)
        - Feature extraction: 1536-dimensional embeddings
        - Classifier head:
            * Linear(1536 -> 512)
            * ReLU activation
            * Dropout(0.4) for regularization
            * Linear(512 -> 2) for binary classification logits

    Input shape: (batch_size, 3, 224, 224)
    Output shape: (batch_size, 2) - logits for [fake, real]

    Attributes:
        backbone: EfficientNet-B3 feature extractor
        classifier: Sequential classifier head
        device: Device for model computation (cuda or cpu)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        """
        Initialize the FrameModel.

        Args:
            num_classes: Number of output classes (default: 2 for binary).
            dropout: Dropout probability for regularization (default: 0.4).
        """
        super().__init__()

        # Determine device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained EfficientNet-B3 without classification head
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0,  # No classification head, extract features only
        )

        # Get output dimension from backbone
        # EfficientNet-B3 outputs 1536-dimensional features
        backbone_output_dim = self.backbone.num_features

        # Define classifier head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Move model to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
               Expected to be normalized to range [0, 1] or [-1, 1].

        Returns:
            torch.Tensor: Logits of shape (batch_size, 2).
                Index 0: logit for fake/synthetic class
                Index 1: logit for real/authentic class
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def load_for_inference(
        self, checkpoint_path: str, strict: bool = True
    ) -> None:
        """
        Load model weights from a checkpoint file for inference.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth or .pt).
            strict: If True, strictly match checkpoint keys with model keys
                   (default: True). Set to False for partial loading.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If checkpoint loading fails.

        Example:
            model = FrameModel()
            model.load_for_inference('checkpoints/model.pth')
            model.eval()  # Set to evaluation mode
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle both direct state_dict and wrapped checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    # Assume it's already a state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict, strict=strict)
            print(f"✓ Model loaded from: {checkpoint_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {e}"
            ) from e

    @property
    def is_cuda(self) -> bool:
        """Check if model is on CUDA device."""
        return next(self.parameters()).is_cuda

    def to_eval_mode(self) -> None:
        """
        Set model to evaluation mode and disable gradients.

        Use this before inference to ensure proper behavior of dropout,
        batch normalization, and to reduce memory usage.
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def get_device_info(self) -> dict:
        """
        Get information about model device placement.

        Returns:
            dict: Device information including device type and availability.
        """
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "model_on_cuda": self.is_cuda,
        }


def create_model(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.4,
) -> FrameModel:
    """
    Factory function to create a FrameModel instance.

    Args:
        pretrained: Whether to use pretrained weights (default: True).
        num_classes: Number of output classes (default: 2).
        dropout: Dropout probability (default: 0.4).

    Returns:
        FrameModel: Initialized model instance.

    Example:
        model = create_model(pretrained=True, num_classes=2)
        model.to_eval_mode()
    """
    model = FrameModel(num_classes=num_classes, dropout=dropout)
    return model
