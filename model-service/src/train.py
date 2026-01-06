"""
Training Script for Frame Classification Model

This script trains the FrameModel on a dataset of labeled images structured as:
    data/
    ├── train/
    │   ├── real/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── fake/
    │       ├── img1.jpg
    │       └── img2.jpg
    └── val/ (optional)
        ├── real/
        └── fake/

Features:
    - Deterministic training with fixed seeds
    - Debug mode for quick validation (16 samples, 1 epoch)
    - Checkpoint saving with metadata and git commit hash
    - Comprehensive logging (configured via config.yaml)
    - GPU/CPU device management (via config.inference.device)
    - Learning rate scheduling

Configuration:
    - Loads settings from config/config.yaml
    - Logging level from config.logging.level
    - Device from config.inference.device
    - Override via environment variables (MODEL_INFERENCE_DEVICE, etc.)

Usage:
    python train.py --data-dir data/train --epochs 10 --batch-size 32 --output checkpoints/
    python train.py --data-dir data/sample --debug --output checkpoints/
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.frame_model import FrameModel
from config import get_config
from logging_config import setup_logging, get_logger


# Load configuration
config = get_config()

# Set up structured logging at startup
setup_logging()
logger = get_logger(__name__)


class FrameDataset(Dataset):
    """
    PyTorch Dataset for loading images from folder structure.

    Expected structure:
        data/
        ├── real/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── fake/
            ├── img1.jpg
            └── img2.jpg

    Labels: 0 = fake, 1 = real
    """

    CLASS_NAMES = ["fake", "real"]
    CLASS_TO_IDX = {"fake": 0, "real": 1}

    def __init__(
        self,
        root_dir: Path,
        max_samples: Optional[int] = None,
        transform=None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing class folders (fake/, real/).
            max_samples: Maximum samples to load (useful for debug mode).
            transform: Torchvision transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Load samples from each class folder
        for class_name, class_idx in self.CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Find all image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            image_files = sorted(image_files)

            for img_path in image_files:
                self.samples.append((img_path, class_idx))

        # Limit samples if requested (for debug)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        for class_name, class_idx in self.CLASS_TO_IDX.items():
            count = sum(1 for _, idx in self.samples if idx == class_idx)
            logger.info(f"  - {class_name}: {count} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label).
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return black image as fallback
            image = Image.new("RGB", (224, 224), color="black")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms() -> transforms.Compose:
    """
    Get image transforms (resize, center crop, normalize).

    Returns:
        Composed transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to larger size first
            transforms.CenterCrop((224, 224)),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def set_deterministic_mode(seed: int = 42):
    """
    Set deterministic mode for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set deterministic mode with seed: {seed}")


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Commit hash or None if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.debug(f"Could not get git commit: {e}")
    return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # Log progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_loss = total_loss / total_samples
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}"
            )

    avg_loss = total_loss / total_samples
    logger.info(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: Path,
    args: argparse.Namespace,
):
    """
    Save training checkpoint with metadata.

    Args:
        model: Trained model.
        optimizer: Optimizer state.
        epoch: Current epoch.
        loss: Current loss.
        checkpoint_path: Path to save checkpoint.
        args: Training arguments.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Get git commit hash
    git_hash = get_git_commit_hash()

    # Prepare metadata
    metadata = {
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_hash,
        "seed": args.seed,
        "args": {
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "debug": args.debug,
            "seed": args.seed,
        },
    }

    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(
        "Checkpoint saved",
        checkpoint_path=str(checkpoint_path),
        epoch=epoch,
        loss=round(loss, 4),
        git_commit=git_hash,
        file_size_mb=round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
    )


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train FrameModel for binary image classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training
  python train.py --data-dir data/train --epochs 10 --batch-size 32 --output checkpoints/

  # Debug mode (16 samples, 1 epoch)
  python train.py --data-dir data/sample --debug --output checkpoints/

  # Custom learning rate
  python train.py --data-dir data/train --epochs 5 --lr 5e-5 --output checkpoints/
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory with train/real and train/fake subdirs",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file path",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: load 16 samples max, run 1 epoch only",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Apply debug mode overrides
    if args.debug:
        args.epochs = 1
        max_samples = 16
        logger.info("Debug mode enabled", epochs=1, max_samples=16)
    else:
        max_samples = None

    try:
        # Setup
        logger.info("Training started", debug=args.debug, epochs=args.epochs, batch_size=args.batch_size)

        set_deterministic_mode(args.seed)

        # Get device from config
        device_config = config.inference.device
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)
        logger.info("Device selected", device=str(device), cuda_available=torch.cuda.is_available())

        # Check data directory
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            sys.exit(1)

        # Load dataset
        logger.info(f"Loading dataset from: {data_dir}")
        train_transform = get_transforms()
        dataset = FrameDataset(
            root_dir=data_dir,
            max_samples=max_samples,
            transform=train_transform,
        )

        if len(dataset) == 0:
            logger.error("No samples found in dataset")
            sys.exit(1)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if str(device) == "cuda" else False,
        )

        logger.info(
            "Dataset loaded",
            total_samples=len(dataset),
            batch_size=args.batch_size,
            batches_per_epoch=len(dataloader),
        )

        # Create model
        logger.info("Loading FrameModel", model_type=config.model.model_type)
        model = FrameModel()
        model.to(device)
        logger.info("Model initialized on device", device=str(device))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        logger.info(
            "Optimizer configured",
            optimizer="AdamW",
            learning_rate=args.lr,
            weight_decay=1e-5,
        )
        logger.info(
            "Scheduler configured",
            scheduler="CosineAnnealingLR",
            t_max=args.epochs,
        )

        # Training loop
        logger.info("Starting training loop", epochs=args.epochs)

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )

            scheduler.step()

            # Save checkpoint after each epoch
            checkpoint_name = f"epoch_{epoch:03d}.pth"
            checkpoint_path = Path(args.output) / checkpoint_name
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss,
                checkpoint_path=checkpoint_path,
                args=args,
            )

        # Save final checkpoint (and debug checkpoint if debug mode)
        final_checkpoint_path = Path(args.output) / "final.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            loss=loss,
            checkpoint_path=final_checkpoint_path,
            args=args,
        )

        # In debug mode, also save as debug.pth
        if args.debug:
            debug_checkpoint_path = Path(args.output) / "debug.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=args.epochs,
                loss=loss,
                checkpoint_path=debug_checkpoint_path,
                args=args,
            )

        logger.info(
            "Training completed successfully",
            total_epochs=args.epochs,
            final_loss=round(loss, 4),
            output_dir=str(Path(args.output).absolute()),
        )

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
