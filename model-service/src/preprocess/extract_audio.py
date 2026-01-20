"""
Audio Extraction Utilities

This module provides utilities for extracting and preprocessing audio
from video files for deepfake detection.

Usage:
    extractor = AudioExtractor(sample_rate=16000)
    waveform, sr = extractor.extract_from_video('video.mp4')
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import librosa
except ImportError:
    librosa = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    Extract and preprocess audio from video files.
    
    Args:
        sample_rate: Target sample rate in Hz (default: 16000)
        mono: Convert to mono (default: True)
        backend: Audio loading backend ('librosa' or 'torchaudio')
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        backend: str = "librosa",
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.backend = backend
        
        if backend == "librosa" and librosa is None:
            raise ImportError("librosa required. Install: pip install librosa")
        elif backend == "torchaudio" and torchaudio is None:
            raise ImportError("torchaudio required. Install: pip install torchaudio")
        
        logger.info(f"Initialized AudioExtractor: sr={sample_rate}, mono={mono}, backend={backend}")
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds (optional)
            duration: Duration in seconds (optional)
        
        Returns:
            waveform: Audio waveform [samples] or [channels, samples]
            sample_rate: Sample rate in Hz
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio to temporary WAV file using FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_audio_path = tmp_file.name
        
        try:
            self._extract_audio_ffmpeg(
                video_path,
                tmp_audio_path,
                start_time,
                duration,
            )
            
            # Load extracted audio
            waveform, sr = self.load_audio(tmp_audio_path)
            
        finally:
            # Clean up temporary file
            Path(tmp_audio_path).unlink(missing_ok=True)
        
        return waveform, sr
    
    def _extract_audio_ffmpeg(
        self,
        video_path: Path,
        output_path: str,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """
        Extract audio using FFmpeg.
        
        Args:
            video_path: Input video path
            output_path: Output audio path
            start_time: Start time in seconds
            duration: Duration in seconds
        """
        cmd = ['ffmpeg', '-y', '-i', str(video_path)]
        
        # Add time range if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        
        # Audio extraction parameters
        cmd.extend([
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(self.sample_rate),  # Sample rate
            '-ac', '1' if self.mono else '2',  # Channels
            output_path
        ])
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            logger.debug(f"Extracted audio from {video_path} to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio from {video_path}")
    
    def load_audio(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            waveform: Audio waveform
            sample_rate: Sample rate in Hz
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.backend == "librosa":
            waveform, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=self.mono,
            )
        
        elif self.backend == "torchaudio":
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate
            
            # Convert to mono if needed
            if self.mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Convert to numpy
            waveform = waveform.squeeze().numpy()
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return waveform, sr
    
    def extract_mel_spectrogram(
        self,
        waveform: np.ndarray,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract mel-spectrogram from waveform.
        
        Args:
            waveform: Audio waveform [samples]
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency (default: sr/2)
        
        Returns:
            mel_spec: Mel-spectrogram [n_mels, time_steps]
        """
        if librosa is None:
            raise ImportError("librosa required for mel-spectrogram extraction")
        
        if fmax is None:
            fmax = self.sample_rate / 2
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def segment_audio(
        self,
        waveform: np.ndarray,
        segment_duration: float = 1.0,
        overlap: float = 0.5,
    ) -> list:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            waveform: Audio waveform [samples]
            segment_duration: Segment duration in seconds
            overlap: Overlap ratio (0.0 to 1.0)
        
        Returns:
            segments: List of audio segments
        """
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        start = 0
        
        while start + segment_samples <= len(waveform):
            segment = waveform[start:start + segment_samples]
            segments.append(segment)
            start += hop_samples
        
        # Handle last segment if needed
        if start < len(waveform):
            # Pad last segment
            last_segment = waveform[start:]
            padding = segment_samples - len(last_segment)
            last_segment = np.pad(last_segment, (0, padding), mode='constant')
            segments.append(last_segment)
        
        logger.debug(f"Segmented audio into {len(segments)} segments")
        return segments
    
    def normalize_audio(
        self,
        waveform: np.ndarray,
        method: str = "peak",
    ) -> np.ndarray:
        """
        Normalize audio waveform.
        
        Args:
            waveform: Audio waveform
            method: Normalization method ('peak' or 'rms')
        
        Returns:
            normalized: Normalized waveform
        """
        if method == "peak":
            # Peak normalization
            peak = np.abs(waveform).max()
            if peak > 0:
                normalized = waveform / peak
            else:
                normalized = waveform
        
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms > 0:
                normalized = waveform / rms
            else:
                normalized = waveform
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
