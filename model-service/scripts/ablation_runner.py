"""
Modality Ablation Runner

Automates training and evaluation of video-only, audio-only, and multimodal
configurations to prove the benefit of multi-modal fusion.

Usage:
    python abl

ation_runner.py \
        --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \
        --output results/ablation_split_1/ \
        --epochs 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.multimodal_lomo_dataset import create_lomo_dataloaders
from models.multimodal_model import MultimodalModel
from config import get_config
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)
config = get_config()


class AblationRunner:
    """
    Runs ablation studies comparing different modality configurations.
    """
    
    def __init__(
        self,
        split_config_path: Path,
        output_dir: Path,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-4
    ):
        """
        Initialize ablation runner.
        
        Args:
            split_config_path: Path to LOMO split configuration
            output_dir: Output directory for results
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.split_config_path = split_config_path
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # Load split configuration
        with open(split_config_path) as f:
            self.split_config = json.load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_ablations(self) -> Dict:
        """
        Run all three ablation experiments.
        
        Returns:
            Dictionary with results from all ablations
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"MODALITY ABLATION STUDY")
        logger.info(f"{'='*70}")
        logger.info(f"Split: {self.split_config['split_name']}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"{'='*70}\n")
        
        results = {}
        
        # Ablation 1: Video-only
        logger.info("\n[1/3] Running VIDEO-ONLY ablation...")
        results['video_only'] = self.run_single_ablation(
            mode='video_only',
            enable_video=True,
            enable_audio=False
        )
        
        # Ablation 2: Audio-only
        logger.info("\n[2/3] Running AUDIO-ONLY ablation...")
        results['audio_only'] = self.run_single_ablation(
            mode='audio_only',
            enable_video=False,
            enable_audio=True
        )
        
        # Ablation 3: Multimodal
        logger.info("\n[3/3] Running MULTIMODAL ablation...")
        results['multimodal'] = self.run_single_ablation(
            mode='multimodal',
            enable_video=True,
            enable_audio=True
        )
        
        # Save aggregated results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def run_single_ablation(
        self,
        mode: str,
        enable_video: bool,
        enable_audio: bool
    ) -> Dict:
        """
        Run a single ablation experiment.
        
        Args:
            mode: Mode name ('video_only', 'audio_only', 'multimodal')
            enable_video: Whether to enable video branch
            enable_audio: Whether to enable audio branch
            
        Returns:
            Dictionary with training and validation metrics
        """
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Video enabled: {enable_video}")
        logger.info(f"  Audio enabled: {enable_audio}")
        
        # Create dataloaders
        dataloaders = create_lomo_dataloaders(
            split_config_path=str(self.split_config_path),
            batch_size=self.batch_size,
            frames_per_video=10,
            audio_feature='spectrogram',
            num_workers=4
        )
        
        # Create model with specific modality configuration
        model = MultimodalModel(
            config=config,
            num_classes=2,
            enable_video=enable_video,
            enable_audio=enable_audio
        )
        model.to(self.device)
        
        logger.info(f"  Model parameters: {model.count_parameters():,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self._train_epoch(
                model, dataloaders['train'], criterion, optimizer,
                epoch, enable_video, enable_audio
            )
            
            # Validate
            val_metrics = self._validate(
                model, dataloaders[' val'], criterion,
                enable_video, enable_audio
            )
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Track best
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
            
            logger.info(f"  Epoch {epoch}/{self.epochs} - "
                       f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}% | "
                       f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        
        # Test on excluded method
        test_metrics = self._validate(
            model, dataloaders['test'], criterion,
            enable_video, enable_audio
        )
        
        logger.info(f"  Test (unseen method): Acc={test_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"{mode}_best.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'mode': mode,
            'enable_video': enable_video,
            'enable_audio': enable_audio,
            'test_accuracy': test_metrics['accuracy'],
            'split_config': self.split_config
        }, checkpoint_path)
        
        return {
            'mode': mode,
            'enable_video': enable_video,
            'enable_audio': enable_audio,
            'best_val_acc': best_val_acc,
            'test_acc': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'history': history,
            'num_parameters': model.count_parameters()
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        enable_video: bool,
        enable_audio: bool
    ) -> Dict:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Prepare inputs based on enabled modalities
            video = batch['frames'].to(self.device) if enable_video else None
            audio = batch['audio'].to(self.device) if enable_audio else None
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = model(video=video, audio=audio)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        enable_video: bool,
        enable_audio: bool
    ) -> Dict:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                video = batch['frames'].to(self.device) if enable_video else None
                audio = batch['audio'].to(self.device) if enable_audio else None
                labels = batch['label'].to(self.device)
                
                outputs = model(video=video, audio=audio)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _save_results(self, results: Dict):
        """Save ablation results."""
        results_file = self.output_dir / "ablation_results.json"
        
        # Add metadata
        results_with_metadata = {
            'split_config': self.split_config,
            'timestamp': datetime.now().isoformat(),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {results_file}")
    
    def _print_summary(self, results: Dict):
        """Print ablation summary table."""
        print("\n" + "="*80)
        print("MODALITY ABLATION SUMMARY")
        print("="*80)
        print(f"Split: {self.split_config['split_name']}")
        print("")
        print(f"{'Mode':<15} {'Video':<8} {'Audio':<8} {'Params':<12} {'Val Acc':<10} {'Test Acc':<10}")
        print("-"*80)
        
        for mode_name in ['video_only', 'audio_only', 'multimodal']:
            r = results[mode_name]
            print(f"{r['mode']:<15} "
                  f"{'✓' if r['enable_video'] else '✗':<8} "
                  f"{'✓' if r['enable_audio'] else '✗':<8} "
                  f"{r['num_parameters']:>11,} "
                  f"{r['best_val_acc']:>9.2f}% "
                  f"{r['test_acc']:>9.2f}%")
        
        print("-"*80)
        
        # Calculate improvements
        video_acc = results['video_only']['test_acc']
        audio_acc = results['audio_only']['test_acc']
        multi_acc = results['multimodal']['test_acc']
        
        improvement_over_video = multi_acc - video_acc
        improvement_over_audio = multi_acc - audio_acc
        
        print(f"\nMultimodal Improvement:")
        print(f"  vs Video-only: +{improvement_over_video:.2f}%")
        print(f"  vs Audio-only: +{improvement_over_audio:.2f}%")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run modality ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ablation on LOMO Split 1
  python ablation_runner.py \\
      --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \\
      --output results/ablation_split_1/ \\
      --epochs 5
  
  # Quick test (1 epoch)
  python ablation_runner.py \\
      --split-config configs/lomo_splits/lomo_split_1_test_deepfakes.json \\
      --output results/ablation_test/ \\
      --epochs 1 \\
      --batch-size 16
        """
    )
    
    parser.add_argument(
        '--split-config',
        type=Path,
        required=True,
        help='Path to LOMO split configuration'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for ablation results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs per ablation (default: 5)'
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
    
    args = parser.parse_args()
    
    # Validate split config
    if not args.split_config.exists():
        logger.error(f"Split config not found: {args.split_config}")
        return 1
    
    # Run ablation study
    runner = AblationRunner(
        split_config_path=args.split_config,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    try:
        results = runner.run_all_ablations()
        logger.info("\n✓ Ablation study completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
