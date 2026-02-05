"""
Multimodal Dataset and DataLoader for Video + Audio Deepfake Detection

Loads paired video frames and audio features for training multimodal models.
Supports different audio feature types (spectrogram, MFCC, raw waveform).

Usage:
    from src.datasets.multimodal_dataset import MultimodalDeepfakeDataset
    from torch.utils.data import DataLoader
    
    dataset = MultimodalDeepfakeDataset(
        frame_dir='data/processed/train/frames',
        audio_dir='data/processed/train/audio',
        audio_feature='spectrogram'
    )
    
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    
    for frames, audio, labels in loader:
        # frames: [batch, num_frames, 3, 224, 224]
        # audio: [batch, n_mels, time_steps]
        # labels: [batch]
        pass
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.preprocessing.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class MultimodalDeepfakeDataset(Dataset):
    """
    Dataset for multimodal deepfake detection (video frames + audio).
    
    Directory structure expected:
    ```
    root/
    ├── fake/
    │   ├── video_000_frame_00001.jpg
    │   ├── video_000_frame_00002.jpg
    │   └── ...
    └── real/
        ├── video_001_frame_00001.jpg
        └── ...
    ```
    """
    
    def __init__(
        self,
        frame_dir: Path,
        audio_dir: Optional[Path] = None,
        audio_feature: str = 'spectrogram',
        frames_per_video: int = 10,
        transform: Optional[transforms.Compose] = None,
        audio_duration: float = 3.0,
        sample_rate: int = 16000,
        augment_audio: bool = False,
        debug: bool = False
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            frame_dir: Directory containing video frames organized in fake/real subdirs
            audio_dir: Directory containing audio files (optional)
            audio_feature: Type of audio feature ('spectrogram', 'mfcc', 'waveform')
            frames_per_video: Number of frames to sample from each video
            transform: Image transforms to apply
            audio_duration: Duration of audio to process (seconds)
            sample_rate: Audio sample rate (Hz)
            augment_audio: Apply audio augmentation
            debug: Enable debug logging
        """
        self.frame_dir = Path(frame_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.audio_feature = audio_feature
        self.frames_per_video = frames_per_video
        self.audio_duration = audio_duration
        self.augment_audio = augment_audio
        self.debug = debug
        
        # Initialize audio processor if audio_dir provided
        if self.audio_dir:
            self.audio_processor = AudioProcessor(
                sample_rate=sample_rate,
                audio_duration=audio_duration
            )
        else:
            self.audio_processor = None
        
        # Default image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Build dataset
        self.samples = self._build_dataset()
        logger.info(f"Dataset initialized with {len(self.samples)} video samples")
    
    def _build_dataset(self) -> List[Dict]:
        """
        Build list of video samples.
        
        Returns:
            List of dicts with video metadata
        """
        samples = []
        
        for label_dir in ['fake', 'real']:
            label_path = self.frame_dir / label_dir
            
            if not label_path.exists():
                logger.warning(f"Label directory not found: {label_path}")
                continue
            
            label = 1 if label_dir == 'fake' else 0
            
            # Group frames by video
            video_frames = {}
            for frame_file in sorted(label_path.glob('*.jpg')):
                # Extract video ID from filename (e.g., video_0000_frame_00001.jpg)
                video_id = frame_file.stem.rsplit('_frame', 1)[0]
                
                if video_id not in video_frames:
                    video_frames[video_id] = []
                
                video_frames[video_id].append(frame_file)
            
            # Create samples
            for video_id, frames in video_frames.items():
                sample = {
                    'video_id': video_id,
                    'frames': sorted(frames),
                    'label': label,
                    'label_str': label_dir,
                    'n_frames': len(frames)
                }
                
                # Add audio path if available
                if self.audio_dir:
                    audio_path = self.audio_dir / label_dir / f"{video_id}.wav"
                    if audio_path.exists():
                        sample['audio_path'] = audio_path
                    else:
                        logger.warning(f"Audio not found for {video_id}")
                
                samples.append(sample)
        
        if self.debug:
            logger.info(f"Fake samples: {sum(1 for s in samples if s['label'] == 1)}")
            logger.info(f"Real samples: {sum(1 for s in samples if s['label'] == 0)}")
        
        return samples
    
    def _load_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        """
        Load and process video frames.
        
        Args:
            frame_paths: List of frame file paths
            
        Returns:
            Tensor of shape [num_frames, 3, 224, 224]
        """
        # Sample frames uniformly
        if len(frame_paths) > self.frames_per_video:
            indices = np.linspace(0, len(frame_paths) - 1, self.frames_per_video, dtype=int)
            sampled_frames = [frame_paths[i] for i in indices]
        else:
            sampled_frames = frame_paths
        
        frames = []
        for frame_path in sampled_frames:
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            except Exception as e:
                logger.warning(f"Error loading frame {frame_path}: {e}")
                # Use black frame as fallback
                frames.append(torch.zeros(3, 224, 224))
        
        # Pad or trim to exact number of frames
        while len(frames) < self.frames_per_video:
            frames.append(torch.zeros(3, 224, 224))
        
        frames = frames[:self.frames_per_video]
        return torch.stack(frames, dim=0)  # [num_frames, 3, 224, 224]
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load and process audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio feature tensor
        """
        try:
            if self.audio_feature == 'spectrogram':
                features = self.audio_processor.audio_to_spectrogram(audio_path)
            elif self.audio_feature == 'mfcc':
                features = self.audio_processor.audio_to_mfcc(audio_path)
            elif self.audio_feature == 'waveform':
                features = self.audio_processor.audio_to_waveform(audio_path)
            else:
                raise ValueError(f"Unknown audio feature: {self.audio_feature}")
            
            return torch.from_numpy(features).float()
            
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            # Return zeros
            if self.audio_feature == 'waveform':
                return torch.zeros(self.audio_processor.n_samples)
            else:
                return torch.zeros(80, 100)  # Dummy spectrogram
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dict with 'frames', 'audio' (optional), and 'label'
        """
        sample = self.samples[idx]
        
        # Load frames
        frames = self._load_frames(sample['frames'])
        
        # Load audio if available
        audio = None
        if 'audio_path' in sample and self.audio_dir:
            audio = self._load_audio(sample['audio_path'])
        
        # Prepare output
        output = {
            'frames': frames,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id']
        }
        
        if audio is not None:
            output['audio'] = audio
        
        return output


def create_multimodal_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    frames_per_video: int = 10,
    audio_feature: str = 'spectrogram',
    splits: Optional[Dict[str, float]] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        data_dir: Root data directory with processed splits
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        frames_per_video: Frames to sample per video
        audio_feature: Type of audio features to extract
        splits: Dict with split names and proportions
        
    Returns:
        Dict of DataLoaders {'train': ..., 'val': ..., 'test': ...}
    """
    if splits is None:
        splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    data_dir = Path(data_dir)
    loaders = {}
    
    for split_name in ['train', 'val', 'test']:
        split_dir = data_dir / split_name
        audio_dir = data_dir / 'audio' / split_name if (data_dir / 'audio').exists() else None
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        dataset = MultimodalDeepfakeDataset(
            frame_dir=split_dir,
            audio_dir=audio_dir,
            audio_feature=audio_feature,
            frames_per_video=frames_per_video
        )
        
        # Determine shuffle for train set
        shuffle = (split_name == 'train')
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size if split_name == 'train' else batch_size * 2,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=(split_name == 'train')
        )
        
        loaders[split_name] = loader
        logger.info(f"{split_name.upper()}: {len(dataset)} videos, "
                   f"{len(loader)} batches")
    
    return loaders


def verify_multimodal_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """
    Verify a batch from multimodal dataloader.
    
    Args:
        batch: Batch dictionary from DataLoader
        
    Returns:
        True if batch is valid
    """
    frames = batch.get('frames')
    audio = batch.get('audio')
    labels = batch.get('label')
    
    if frames is None:
        logger.error("Missing 'frames' in batch")
        return False
    
    if frames.shape[0] == 0:
        logger.error("Empty batch")
        return False
    
    logger.info(f"✓ Batch shapes:")
    logger.info(f"  frames: {frames.shape}")  # [batch, frames, 3, 224, 224]
    
    if audio is not None:
        logger.info(f"  audio: {audio.shape}")
    
    if labels is not None:
        logger.info(f"  labels: {labels.shape}")
        logger.info(f"  Fake: {(labels == 1).sum()}, Real: {(labels == 0).sum()}")
    
    return True
