"""
LOMO Multimodal Dataset for PyTorch

PyTorch dataset for Leave-One-Method-Out (LOMO) evaluation.
Loads video frames and audio features for training/testing on specific manipulation methods.

Usage:
    from datasets.multimodal_lomo_dataset import create_lomo_dataloaders
    
    loaders = create_lomo_dataloaders(
        split_config_path='configs/lomo_splits/lomo_split_1_test_deepfakes.json',
        batch_size=32
    )
    
    for batch in loaders['train']:
        frames, audio, labels = batch['frames'], batch['audio'], batch['label']
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class LomoMultimodalDataset(Dataset):
    """
    PyTorch Dataset for LOMO evaluation with video and audio.
    
    Loads synchronized video frames and audio features for deepfake detection,
    with support for excluding specific manipulation methods (LOMO protocol).
    """
    
    def __init__(
        self,
        split_config_path: Path,
        mode: str = 'train',
        frames_per_video: int = 10,
        audio_feature: str = 'spectrogram',
        transform = None,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize LOMO dataset.
        
        Args:
            split_config_path: Path to LOMO split JSON configuration
            mode: 'train', 'val', or 'test'
            frames_per_video: Number of frames to sample per video
            audio_feature: Type of audio feature ('spectrogram', 'mfcc', 'waveform')
            transform: Torchvision transforms for video frames
            max_samples: Maximum samples to load (for debugging)
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.frames_per_video = frames_per_video
        self.audio_feature = audio_feature
        self.transform = transform or self._get_default_transform()
        self.max_samples = max_samples
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load LOMO split configuration
        with open(split_config_path) as f:
            self.config = json.load(f)
        
        self.split_name = self.config['split_name']
        self.test_method = self.config['test_method']
        self.train_methods = self.config['train_methods']
        self.data_dir = Path(self.config['data_dir'])
        
        # Build dataset samples
        self.samples = self._build_samples()
        
        print(f"Loaded {len(self.samples)} samples for {self.split_name} ({mode} mode)")
        print(f"  - Test method (excluded from train): {self.test_method}")
        if mode == 'train':
            print(f"  - Train methods: {', '.join(self.train_methods)}")
    
    def _get_default_transform(self):
        """Get default image transforms."""
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
        """
        Build list of samples based on LOMO configuration and mode.
        
        Returns:
            List of sample dictionaries with paths and labels
        """
        samples = []
        
        if self.mode == 'train' or self.mode == 'val':
            # Training/validation: use train_methods + original
            methods_to_include = self.train_methods
        else:  # test mode
            # Testing: use test_method (excluded during training)
            methods_to_include = [self.test_method]
        
        # Add fake samples from included methods
        for method in methods_to_include:
            method_dir = self.data_dir / method
            if not method_dir.exists():
                print(f"Warning: Method directory not found: {method_dir}")
                continue
            
            video_files = list(method_dir.glob("*.mp4")) + list(method_dir.glob("*.avi"))
            
            for video_path in video_files:
                # Construct audio path
                audio_path = self._get_audio_path(video_path, method)
                
                samples.append({
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'label': 1,  # fake
                    'method': method,
                    'video_id': video_path.stem
                })
        
        # Add real samples from original
        original_dir = self.data_dir / "original"
        if original_dir.exists():
            video_files = list(original_dir.glob("*.mp4")) + list(original_dir.glob("*.avi"))
            
            for video_path in video_files:
                audio_path = self._get_audio_path(video_path, "original")
                
                samples.append({
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'label': 0,  # real
                    'method': 'original',
                    'video_id': video_path.stem
                })
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Split train/val if in train mode (80/20 split)
        if self.mode == 'train':
            split_idx = int(0.8 * len(samples))
            samples = samples[:split_idx]
        elif self.mode == 'val':
            split_idx = int(0.8 * len(samples))
            samples = samples[split_idx:]
        
        # Limit samples if specified
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _get_audio_path(self, video_path: Path, method: str) -> Path:
        """
        Get corresponding audio file path for a video.
        
        Args:
            video_path: Path to video file
            method: Manipulation method name
            
        Returns:
            Path to audio file (may not exist yet)
        """
        # Assume audio files are in data_dir/../audio/<method>/
        audio_dir = self.data_dir.parent / "lomo_audio" / method
        audio_filename = video_path.stem + ".wav"
        return audio_dir / audio_filename
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with frames, audio, label, and metadata
        """
        sample = self.samples[idx]
        
        # Load video frames
        frames = self._load_frames(sample['video_path'])
        
        # Load audio
        audio = self._load_audio(sample['audio_path'])
        
        return {
            'frames': frames,
            'audio': audio,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id'],
            'method': sample['method']
        }
    
    def _load_frames(self, video_path: Path) -> torch.Tensor:
        """
        Load video frames from a video file.
        
        For now, this is a placeholder. In practice, you'd:
        1. Extract frames from video
        2. Sample frames_per_video uniformly
        3. Apply transforms
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tensor of shape [frames_per_video, 3, 224, 224]
        """
        # TODO: Implement actual frame extraction
        # For now, return dummy frames
        frames = []
        for _ in range(self.frames_per_video):
            # Create dummy black frame
            frame = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        return torch.stack(frames)
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load and process audio features.
        
        Args:
            audio_path: Path to audio file (.wav)
            
        Returns:
            Audio feature tensor
        """
        if not audio_path.exists():
            # Return silent audio if file doesn't exist
            if self.audio_feature == 'spectrogram':
                return torch.zeros((80, 300))  # Mel-spectrogram shape
            elif self.audio_feature == 'mfcc':
                return torch.zeros((13, 300))  # MFCC shape
            else:  # waveform
                return torch.zeros((48000,))  # 3 seconds at 16kHz
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Extract feature
            if self.audio_feature == 'spectrogram':
                # Mel-spectrogram
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=512,
                    hop_length=160,
                    n_mels=80
                )
                feature = mel_transform(waveform).squeeze(0)
                
                # Pad or trim to fixed length
                if feature.shape[1] < 300:
                    feature = torch.nn.functional.pad(feature, (0, 300 - feature.shape[1]))
                else:
                    feature = feature[:, :300]
                
            elif self.audio_feature == 'mfcc':
                # MFCC
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=16000,
                    n_mfcc=13
                )
                feature = mfcc_transform(waveform).squeeze(0)
                
                # Pad or trim to fixed length
                if feature.shape[1] < 300:
                    feature = torch.nn.functional.pad(feature, (0, 300 - feature.shape[1]))
                else:
                    feature = feature[:, :300]
            
            else:  # waveform
                # Raw waveform
                feature = waveform.squeeze(0)
                
                # Pad or trim to 3 seconds (48000 samples at 16kHz)
                if feature.shape[0] < 48000:
                    feature = torch.nn.functional.pad(feature, (0, 48000 - feature.shape[0]))
                else:
                    feature = feature[:48000]
            
            return feature
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silent audio on error
            if self.audio_feature == 'spectrogram':
                return torch.zeros((80, 300))
            elif self.audio_feature == 'mfcc':
                return torch.zeros((13, 300))
            else:
                return torch.zeros((48000,))


def create_lomo_dataloaders(
    split_config_path: str,
    batch_size: int = 32,
    frames_per_video: int = 10,
    audio_feature: str = 'spectrogram',
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders for LOMO evaluation.
    
    Args:
        split_config_path: Path to LOMO split configuration JSON
        batch_size: Batch size for dataloaders
        frames_per_video: Number of frames to sample per video
        audio_feature: Type of audio feature ('spectrogram', 'mfcc', 'waveform')
        num_workers: Number of dataloader workers
        max_samples: Maximum samples per split (for debugging)
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    dataloaders = {}
    
    for mode in ['train', 'val', 'test']:
        dataset = LomoMultimodalDataset(
            split_config_path=Path(split_config_path),
            mode=mode,
            frames_per_video=frames_per_video,
            audio_feature=audio_feature,
            max_samples=max_samples
        )
        
        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python multimodal_lomo_dataset.py <split_config_path>")
        sys.exit(1)
    
    split_config_path = sys.argv[1]
    
    print("Testing LOMO dataset...")
    loaders = create_lomo_dataloaders(
        split_config_path=split_config_path,
        batch_size=4,
        max_samples=16
    )
    
    for mode in ['train', 'val', 'test']:
        print(f"\n{mode.upper()} DataLoader:")
        loader = loaders[mode]
        print(f"  - Batches: {len(loader)}")
        
        # Get first batch
        batch = next(iter(loader))
        print(f"  - Frames shape: {batch['frames'].shape}")
        print(f"  - Audio shape: {batch['audio'].shape}")
        print(f"  - Labels shape: {batch['label'].shape}")
        print(f"  - Methods: {batch['method']}")
    
    print("\n✓ Dataset test completed successfully!")
