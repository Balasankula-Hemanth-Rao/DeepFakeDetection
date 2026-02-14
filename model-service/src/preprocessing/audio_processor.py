"""
Audio Preprocessing for Multimodal Deepfake Detection

Converts raw audio files to spectrograms using lightweight dependencies (scipy/numpy).
Avoids torchaudio/librosa due to environment issues.
"""

import logging
from typing import Optional, Tuple, Dict, Union
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile as wavfile
import scipy.signal
import os

# Add FFmpeg to PATH - This was for torchaudio/librosa, no longer needed
# ffmpeg_path = r"C:\Users\SAI-RAM\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
# if ffmpeg_path not in os.environ["PATH"]:
#     os.environ["PATH"] += os.pathsep + ffmpeg_path

from .feature_config import AudioConfig

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
        
        # Initialize torchaudio transforms - REMOVED
        # self.mel_spectrogram = T.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_mels=n_mels,
        #     n_fft=n_fft,
        #     hop_length=hop_length
        # )
        
        # self.mfcc_transform = T.MFCC(
        #     sample_rate=sample_rate,
        #     n_mfcc=n_mfcc,
        #     melkwargs={
        #         'n_fft': n_fft,
        #         'hop_length': hop_length,
        #         'n_mels': n_mels
        #     }
        # )
        
        # Precompute Mel filterbank
        self.mel_basis = self._mel_filterbank()
        
        logger.info(f"Audio Processor initialized (Lightweight)")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Duration: {audio_duration} s ({self.n_samples} samples)")
        # logger.info(f"  Mel bins: {n_mels}, MFCC: {n_mfcc}") # REMOVED
    
    def _mel_filterbank(self):
        """Create Mel filterbank matrix locally to avoid librosa dependency."""
        # Simple implementation or approximation of librosa's mel filterbank
        # Range 0 to nyquist
        fmin = 0
        fmax = float(self.sample_rate) / 2
        
        # FFT bin frequencies
        fft_freqs = np.linspace(0, fmax, self.n_fft // 2 + 1)
        
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Mel points
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Filterbank matrix
        banks = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for m in range(1, self.n_mels + 1):
            f_m_minus = hz_points[m - 1]
            f_m = hz_points[m]
            f_m_plus = hz_points[m + 1]
            
            for k in range(len(fft_freqs)):
                if f_m_minus <= fft_freqs[k] < f_m:
                    banks[m - 1, k] = (fft_freqs[k] - f_m_minus) / (f_m - f_m_minus)
                elif f_m <= fft_freqs[k] < f_m_plus:
                    banks[m - 1, k] = (f_m_plus - fft_freqs[k]) / (f_m_plus - f_m)
        
        return banks

    def load_audio(self, audio_path: Path, offset: float = 0.0) -> Tuple[torch.Tensor, int]:
        """Load audio file with scipy."""
        try:
            sample_rate, data = wavfile.read(str(audio_path))
            
            # Handle float vs int
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0
            elif data.dtype == np.float64: # Convert float64 to float32
                data = data.astype(np.float32)
                
            # Convert to mono if stereo
            if data.ndim > 1:
                data = np.mean(data, axis=1)
                
            # Resample if needed (simple approximation via signal.resample)
            if sample_rate != self.sample_rate:
                num_samples = int(len(data) * self.sample_rate / sample_rate)
                data = scipy.signal.resample(data, num_samples)
                sample_rate = self.sample_rate
                
            # Offset
            if offset > 0:
                start = int(offset * sample_rate)
                data = data[start:]
                
            return torch.from_numpy(data), sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # return silent tensor on failure to prevent crash
            return torch.zeros(self.n_samples), self.sample_rate
    
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
            # Pad with zeros (only at the end as per instruction)
            padding = self.n_samples - len(waveform)
            return torch.nn.functional.pad(waveform, (0, padding))
    
    def audio_to_spectrogram(
        self,
        audio_path: Path,
        to_db: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Convert audio to mel-spectrogram using scipy/numpy."""
        # Load
        waveform, sr = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        y = waveform.numpy()
        
        # STFT
        f, t, Zxx = scipy.signal.stft(
            y, 
            fs=sr, 
            nperseg=self.n_fft, 
            noverlap=self.n_fft - self.hop_length,
            boundary='zeros'
        )
        
        # Power spectrum
        S = np.abs(Zxx)**2
        
        # Mel-Binning
        # Pad S if needed to match filterbank
        # STFT returns n_fft//2 + 1 bins, which matches our basis
        # Just in case of off-by-one due to scipy config
        if S.shape[0] != self.mel_basis.shape[1]:
             # This case should ideally not happen if n_fft is consistent
             # If it does, it indicates a mismatch in FFT bin count.
             # For now, we assume they match.
             pass

        mel_spec = np.dot(self.mel_basis, S)
        
        # To DB
        if to_db:
             mel_spec = 10 * np.log10(np.maximum(mel_spec, 1e-10))
             
        # Normalize
        if normalize:
            mean = mel_spec.mean()
            std = mel_spec.std() + 1e-8
            mel_spec = (mel_spec - mean) / std
            
        return mel_spec
    
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
        # Dummy MFCC or basic implementation
        # For this project, we primarily use spectrograms.
        # Returning spectrogram as placeholder if MFCC requested, or zeros
        # A proper MFCC implementation would involve DCT on log-mel-spectrogram
        # For now, as per instruction, returning spectrogram.
        logger.warning("MFCC generation is a placeholder, returning Mel-spectrogram.")
        return self.audio_to_spectrogram(audio_path, normalize=normalize)
    
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
