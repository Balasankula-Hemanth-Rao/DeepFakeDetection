"""
Cross-Dataset Validation

Evaluate FaceForensics++-trained model on FakeAVCeleb dataset.
Tests generalization to completely different data distribution.

Usage:
    python cross_dataset_evaluator.py \
        --checkpoint checkpoints/lomo_split_1/best.pth \
        --dataset-dir data/fakeavceleb \
        --output results/cross_dataset_fakeavceleb.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.multimodal_model import MultimodalModel


class CrossDatasetDataset(Dataset):
    """
    Dataset for cross-dataset validation (e.g., FakeAVCeleb, DFDC).
    Loads video frames and audio from a different dataset than training.
    """
    
    def __init__(
        self,
        dataset_dir: Path,
        frames_per_video: int = 10,
        audio_feature: str = 'spectrogram',
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize cross-dataset dataset.
        
        Args:
            dataset_dir: Root directory with real/ and fake/ subdirectories
            frames_per_video: Number of frames to sample per video
            audio_feature: Type of audio feature
            transform: Image transforms
            max_samples: Maximum samples (for quick testing)
        """
        self.dataset_dir = Path(dataset_dir)
        self.frames_per_video = frames_per_video
        self.audio_feature = audio_feature
        self.transform = transform or self._get_default_transform()
        
        # Load metadata if available
        metadata_path = self.dataset_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            print(f"Loaded {self.metadata['dataset_name']} dataset")
        else:
            self.metadata = {'dataset_name': 'Unknown'}
        
        # Build samples
        self.samples = self._build_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {self.metadata['dataset_name']}")
        print(f"  - Real: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"  - Fake: {sum(1 for s in self.samples if s['label'] == 1)}")
    
    def _get_default_transform(self):
        """Get default image transforms (same as training)."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_samples(self) -> List[Dict]:
        """Build list of samples from dataset directory."""
        samples = []
        
        # Real samples
        real_dir = self.dataset_dir / "real"
        if real_dir.exists():
            for video_path in real_dir.glob("*.mp4"):
                audio_path = self._get_audio_path(video_path, "real")
                samples.append({
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'label': 0,  # real
                    'video_id': video_path.stem
                })
        
        # Fake samples
        fake_dir = self.dataset_dir / "fake"
        if fake_dir.exists():
            for video_path in fake_dir.glob("*.mp4"):
                audio_path = self._get_audio_path(video_path, "fake")
                samples.append({
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'label': 1,  # fake
                    'video_id': video_path.stem
                })
        
        return samples
    
    def _get_audio_path(self, video_path: Path, label: str) -> Path:
        """Get corresponding audio file path."""
        audio_dir = self.dataset_dir.parent / f"{self.dataset_dir.name}_audio" / label
        audio_filename = video_path.stem + ".wav"
        return audio_dir / audio_filename
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load frames (placeholder, same as LOMO dataset)
        frames = self._load_frames(sample['video_path'])
        
        # Load audio
        audio = self._load_audio(sample['audio_path'])
        
        return {
            'frames': frames,
            'audio': audio,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id']
        }
    
    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """
        Load video frames (implemented using OpenCV).
        Samples frames uniformly from the video.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise IOError(f"Could not open video {video_path}")
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                 # Fallback for some headers
                 # Just force read all? Too slow. 
                 # Assume 30fps * 10 seconds = 300 frames as fallback
                 frame_count = 300
            
            # Sample indices uniformly
            indices = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=int)
            
            frames = []
            
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                
                if ret:
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                else:
                    # Failed to read, use black frame
                    img = Image.new('RGB', (224, 224), color='black')
                
                if self.transform:
                    img_tensor = self.transform(img)
                else:
                    img_tensor = transforms.ToTensor()(img)
                    
                frames.append(img_tensor)
            
            cap.release()
            
            # Ensure we have enough frames
            while len(frames) < self.frames_per_video:
                 frames.append(torch.zeros_like(frames[0]))
                 
            return torch.stack(frames)
            
        except Exception as e:
            print(f"Error loading frames from {video_path}: {e}")
            # Fallback: return black frames
            frames = []
            for _ in range(self.frames_per_video):
                frame = torch.zeros(3, 224, 224)
                frames.append(frame)
            return torch.stack(frames)
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and process audio."""
        if not audio_path.exists():
            # Return silent audio if missing
            if self.audio_feature == 'spectrogram':
                return torch.zeros((80, 300))
            elif self.audio_feature == 'mfcc':
                return torch.zeros((13, 300))
            else:
                return torch.zeros((48000,))
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            if self.audio_feature == 'spectrogram':
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=512,
                    hop_length=160,
                    n_mels=80
                )
                feature = mel_transform(waveform).squeeze(0)
                
                if feature.shape[1] < 300:
                    feature = torch.nn.functional.pad(feature, (0, 300 - feature.shape[1]))
                else:
                    feature = feature[:, :300]
            
            elif self.audio_feature == 'mfcc':
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=16000,
                    n_mfcc=13
                )
                feature = mfcc_transform(waveform).squeeze(0)
                
                if feature.shape[1] < 300:
                    feature = torch.nn.functional.pad(feature, (0, 300 - feature.shape[1]))
                else:
                    feature = feature[:, :300]
            
            else:  # waveform
                feature = waveform.squeeze(0)
                
                if feature.shape[0] < 48000:
                    feature = torch.nn.functional.pad(feature, (0, 48000 - feature.shape[0]))
                else:
                    feature = feature[:48000]
            
            return feature
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            if self.audio_feature == 'spectrogram':
                return torch.zeros((80, 300))
            elif self.audio_feature == 'mfcc':
                return torch.zeros((13, 300))
            else:
                return torch.zeros((48000,))


def evaluate_cross_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str
) -> Dict:
    """
    Evaluate model on cross-dataset.
    
    Args:
        model: Trained model
        dataloader: Cross-dataset dataloader
        device: Device to evaluate on
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_video_ids = []
    
    print(f"\nEvaluating on {dataset_name} (cross-dataset validation)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label']
            
            # Forward pass
            outputs = model(video=frames, audio=audio)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_video_ids.extend(batch['video_id'])
    
    # Convert to numpy
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)
    
    # Compute metrics
    metrics = {
        'dataset_name': dataset_name,
        'validation_type': 'cross_dataset',
        'num_samples': len(labels),
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, average='binary')),
        'recall': float(recall_score(labels, predictions, average='binary')),
        'f1_score': float(f1_score(labels, predictions, average='binary')),
        'auc_roc': float(roc_auc_score(labels, probabilities)),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }
    
    # Add specificity/sensitivity
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Store predictions
    metrics['predictions'] = {
        'video_ids': all_video_ids,
        'true_labels': labels.tolist(),
        'predicted_labels': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    return metrics


def print_cross_dataset_results(metrics: Dict):
    """Print cross-dataset evaluation results."""
    print("\n" + "="*70)
    print(f"CROSS-DATASET VALIDATION: {metrics['dataset_name']}")
    print("="*70)
    print(f"\nNumber of Samples: {metrics['num_samples']}")
    print("\nPerformance Metrics:")
    print("-"*70)
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    print("-"*70)
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                Real    Fake")
    print(f"  Actual Real  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
    print(f"        Fake   [{cm[1,0]:5d}  {cm[1,1]:5d}]")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation on FakeAVCeleb or DFDC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on FakeAVCeleb
  python cross_dataset_evaluator.py \\
      --checkpoint checkpoints/lomo_split_1/best.pth \\
      --dataset-dir data/fakeavceleb \\
      --output results/cross_dataset_fakeavceleb.json
  
  # Quick test
  python cross_dataset_evaluator.py \\
      --checkpoint checkpoints/lomo_split_1/best.pth \\
      --dataset-dir data/fakeavceleb \\
      --output results/test.json \\
      --max-samples 100
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Path to cross-dataset directory (with real/ and fake/ subdirs)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for results JSON'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--frames-per-video',
        type=int,
        default=10,
        help='Frames per video (default: 10)'
    )
    
    parser.add_argument(
        '--audio-feature',
        type=str,
        default='spectrogram',
        choices=['spectrogram', 'mfcc', 'waveform'],
        help='Audio feature type (default: spectrogram)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = MultimodalModel(
        enable_video=True,
        enable_audio=True
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load checkpoint metadata
    if 'lomo_config' in checkpoint:
        print(f"\nModel trained on: {checkpoint['lomo_config']['split_name']}")
        print(f"  - Train methods: {', '.join(checkpoint['lomo_config']['train_methods'])}")
    
    # Create dataset
    print("\nLoading cross-dataset...")
    dataset = CrossDatasetDataset(
        dataset_dir=args.dataset_dir,
        frames_per_video=args.frames_per_video,
        audio_feature=args.audio_feature,
        max_samples=args.max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    metrics = evaluate_cross_dataset(
        model=model,
        dataloader=dataloader,
        device=device,
        dataset_name=dataset.metadata['dataset_name']
    )
    
    # Add training info to results
    if 'lomo_config' in checkpoint:
        metrics['training_info'] = checkpoint['lomo_config']
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print results
    print_cross_dataset_results(metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
