"""
LOMO Training Script

Train multimodal deepfake detection model using Leave-One-Method-Out (LOMO) protocol.

Usage:
    python train_lomo.py \
        --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
        --output checkpoints/lomo_split_1/ \
        --epochs 10 \
        --batch-size 32
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.multimodal_lomo_dataset import create_lomo_dataloaders
from models.multimodal_model import MultimodalDeepfakeDetector
from config import get_config
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)
config = get_config()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        frames = batch['frames'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(video=frames, audio=audio)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Accumulate loss
        total_loss += loss.item() * labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(video=frames, audio=audio)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accumulate loss
            total_loss += loss.item() * labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict,
    args: argparse.Namespace,
    split_config: Dict,
    checkpoint_path: Path
):
    """
    Save training checkpoint with LOMO metadata.
    
    Args:
        model: Trained model
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training/validation metrics
        args: Training arguments
        split_config: LOMO split configuration
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'lomo_config': {
            'split_name': split_config['split_name'],
            'test_method': split_config['test_method'],
            'train_methods': split_config['train_methods']
        },
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'frames_per_video': args.frames_per_video,
            'audio_feature': args.audio_feature
        },
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal deepfake detector with LOMO protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LOMO Split 1 (test on DeepFakes)
  python train_lomo.py \\
      --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \\
      --output checkpoints/lomo_split_1/ \\
      --epochs 10 \\
      --batch-size 32
  
  # Quick test (1 epoch, small dataset)
  python train_lomo.py \\
      --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \\
      --output checkpoints/test/ \\
      --epochs 1 \\
      --batch-size 16 \\
      --max-samples 100
        """
    )
    
    parser.add_argument(
        '--split-config',
        type=Path,
        required=True,
        help='Path to LOMO split configuration JSON'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for checkpoints'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--frames-per-video',
        type=int,
        default=10,
        help='Number of frames to sample per video (default: 10)'
    )
    
    parser.add_argument(
        '--audio-feature',
        type=str,
        default='spectrogram',
        choices=['spectrogram', 'mfcc', 'waveform'],
        help='Type of audio feature (default: spectrogram)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per split (for debugging)'
    )
    
    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load LOMO split configuration
    if not args.split_config.exists():
        logger.error(f"Split config not found: {args.split_config}")
        return 1
    
    with open(args.split_config) as f:
        split_config = json.load(f)
    
    logger.info(f"Training: {split_config['split_name']}")
    logger.info(f"  - Train methods: {', '.join(split_config['train_methods'])}")
    logger.info(f"  - Test method (excluded): {split_config['test_method']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_lomo_dataloaders(
        split_config_path=str(args.split_config),
        batch_size=args.batch_size,
        frames_per_video=args.frames_per_video,
        audio_feature=args.audio_feature,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    
    # Create model
    logger.info("Initializing model...")
    model = MultimodalDeepfakeDetector(
        enable_video=True,
        enable_audio=True
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device
        )
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save checkpoint
        metrics = {
            'train': train_metrics,
            'val': val_metrics
        }
        
        checkpoint_path = args.output / f"epoch_{epoch:03d}.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            args=args,
            split_config=split_config,
            checkpoint_path=checkpoint_path
        )
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_checkpoint_path = args.output / "best.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                args=args,
                split_config=split_config,
                checkpoint_path=best_checkpoint_path
            )
            logger.info(f"✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")
    
    # Save final checkpoint
    final_checkpoint_path = args.output / "final.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        metrics=metrics,
        args=args,
        split_config=split_config,
        checkpoint_path=final_checkpoint_path
    )
    
    # Save training history
    history_path = args.output / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\n✓ Training completed!")
    logger.info(f"  - Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  - Checkpoints saved to: {args.output}")
    logger.info(f"  - Training history: {history_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
