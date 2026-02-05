"""
Audio Preprocessing for Multimodal Deepfake Detection

Converts raw audio files to spectrograms and MFCC features
for use with audio encoders in multimodal models.

Features:
- Mel-Spectrogram extraction (80 bins, standard for speech)
- MFCC (Mel-Frequency Cepstral Coefficients) extraction
- Audio augmentation (pitch shift, time stretch, noise)
- Normalization and padding to fixed length
- GPU support for fast processing

Usage:
    from src.preprocessing.audio_processor import AudioProcessor
    
    processor = AudioProcessor(sample_rate=16000, n_mels=80)
    spectrogram = processor.audio_to_spectrogram('audio.wav')
    mfcc = processor.audio_to_mfcc('audio.wav')
"""

import logging
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio files for multimodal deepfake detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mfcc: int = 13,
        audio_duration: float = 3.0
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            n_mels: Number of mel-frequency bins (default: 80)
            n_fft: FFT window size (default: 512)
            hop_length: Number of samples per hop (default: 160)
            n_mfcc: Number of MFCC coefficients (default: 13)
            audio_duration: Fixed audio duration in seconds (default: 3.0)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.audio_duration = audio_duration
        self.n_samples = int(sample_rate * audio_duration)
        
        # Initialize torchaudio transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        
        logger.info(f"Audio Processor initialized")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Duration: {audio_duration} s ({self.n_samples} samples)")
        logger.info(f"  Mel bins: {n_mels}, MFCC: {n_mfcc}")
    
    def load_audio(self, audio_path: Path, offset: float = 0.0) -> Tuple[torch.Tensor, int]:
        """
        Load audio file with torchaudio.
        
        Args:
            audio_path: Path to audio file
            offset: Start time in seconds
            
        Returns:
            Tuple of (waveform tensor, sample rate)
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Apply offset
            if offset > 0:
                offset_samples = int(offset * self.sample_rate)
                waveform = waveform[:, offset_samples:]
            
            return waveform.squeeze(0), self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Pad or trim waveform to fixed length.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Fixed-length waveform tensor
        """
        if len(waveform) >= self.n_samples:
            # Trim from center
            start = (len(waveform) - self.n_samples) // 2
            return waveform[start:start + self.n_samples]
        else:
            # Pad with zeros
            pad_amount = self.n_samples - len(waveform)
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            return torch.nn.functional.pad(waveform, (pad_left, pad_right))
    
    def audio_to_spectrogram(
        self,
        audio_path: Path,
        to_db: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio_path: Path to audio file
            to_db: Convert to dB scale (default: True)
            normalize: Normalize to [0, 1] (default: True)
            
        Returns:
            Mel-spectrogram as numpy array (n_mels, time_steps)
        """
        # Load and process audio
        waveform, sr = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        
        # Compute mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to dB scale
        if to_db:
            mel_spec = T.AmplitudeToDB()(mel_spec)
        
        # Normalize
        if normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec.numpy()
    
    def audio_to_mfcc(
        self,
        audio_path: Path,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert audio to MFCC features.
        
        Args:
            audio_path: Path to audio file
            normalize: Normalize features (default: True)
            
        Returns:
            MFCC features as numpy array (n_mfcc, time_steps)
        """
        # Load and process audio
        waveform, sr = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        
        # Compute MFCC
        mfcc = self.mfcc_transform(waveform)
        
        # Normalize
        if normalize:
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        return mfcc.numpy()
    
    def audio_to_waveform(self, audio_path: Path) -> torch.Tensor:
        """
        Load and process audio to waveform tensor (for Wav2Vec2).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Fixed-length waveform tensor [n_samples]
        """
        waveform, sr = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        return waveform
    
    def augment_audio(
        self,
        waveform: torch.Tensor,
        pitch_shift: float = 2,
        time_stretch: float = 0.1,
        noise_level: float = 0.01
    ) -> torch.Tensor:
        """
        Apply data augmentation to audio.
        
        Args:
            waveform: Input waveform
            pitch_shift: Semitone shift amount
            time_stretch: Time stretch factor variation
            noise_level: Gaussian noise std dev
            
        Returns:
            Augmented waveform
        """
        audio = waveform.numpy()
        
        # Pitch shift
        if pitch_shift > 0 and np.random.rand() > 0.5:
            shift = np.random.uniform(-pitch_shift, pitch_shift)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=shift)
        
        # Time stretch
        if time_stretch > 0 and np.random.rand() > 0.5:
            factor = np.random.uniform(1 - time_stretch, 1 + time_stretch)
            audio = librosa.effects.time_stretch(audio, rate=factor)
        
        # Add noise
        if noise_level > 0 and np.random.rand() > 0.5:
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise
        
        # Convert back to tensor and ensure correct length
        waveform_aug = torch.from_numpy(audio).float()
        waveform_aug = self.pad_or_trim(waveform_aug)
        
        return waveform_aug
    
    def batch_process(
        self,
        audio_paths: list,
        feature_type: str = 'spectrogram',
        augment: bool = False
    ) -> torch.Tensor:
        """
        Process batch of audio files.
        
        Args:
            audio_paths: List of paths to audio files
            feature_type: 'spectrogram', 'mfcc', or 'waveform'
            augment: Apply augmentation (default: False)
            
        Returns:
            Batch tensor (batch_size, features, time_steps)
        """
        batch = []
        
        for audio_path in audio_paths:
            try:
                if feature_type == 'spectrogram':
                    features = self.audio_to_spectrogram(audio_path)
                elif feature_type == 'mfcc':
                    features = self.audio_to_mfcc(audio_path)
                elif feature_type == 'waveform':
                    features = self.audio_to_waveform(audio_path)
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
                # Add to batch
                batch.append(torch.from_numpy(features).float())
                
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue
        
        # Stack batch
        if batch:
            return torch.stack(batch, dim=0)
        else:
            return None


def get_audio_info(audio_path: Path) -> dict:
    """Get information about audio file."""
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        duration = waveform.shape[1] / sr
        
        return {
            'path': str(audio_path),
            'sample_rate': sr,
            'channels': waveform.shape[0],
            'duration': duration,
            'n_samples': waveform.shape[1]
        }
    except Exception as e:
        logger.error(f"Error reading {audio_path}: {e}")
        return None
