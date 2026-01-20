"""
Voice Activity Detection (VAD) for Audio Preprocessing

This module provides voice activity detection to identify speech regions
in audio and filter out silence/background noise. This improves audio
encoder focus on relevant speech content.

Supports two VAD strategies:
1. Energy-based VAD (librosa) - Simple, fast, no dependencies
2. pyannote-audio VAD - More accurate, requires additional setup

Expected Performance Gain: +1-2% AUC improvement

Usage:
    vad = VoiceActivityDetector(strategy='energy')
    waveform, sr = librosa.load('audio.wav', sr=16000)
    speech_segments = vad.detect(waveform, sr)
    masked_waveform = vad.apply_mask(waveform, speech_segments)
"""

import logging
from typing import List, Tuple, Optional, Union

import numpy as np
import torch

try:
    import librosa
except ImportError:
    librosa = None

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Voice Activity Detection for filtering silence and background noise.
    
    Args:
        strategy: VAD strategy ('energy' or 'pyannote')
        energy_threshold: Energy threshold for energy-based VAD (default: 0.02)
        frame_length: Frame length in samples for energy computation (default: 2048)
        hop_length: Hop length in samples (default: 512)
        min_speech_duration: Minimum speech segment duration in seconds (default: 0.1)
        min_silence_duration: Minimum silence duration to split segments (default: 0.3)
    """
    
    def __init__(
        self,
        strategy: str = "energy",
        energy_threshold: float = 0.02,
        frame_length: int = 2048,
        hop_length: int = 512,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.3,
    ):
        self.strategy = strategy
        self.energy_threshold = energy_threshold
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        
        if strategy == "energy":
            if librosa is None:
                raise ImportError("librosa required for energy-based VAD. Install: pip install librosa")
            logger.info("Initialized energy-based VAD")
        
        elif strategy == "pyannote":
            if Pipeline is None:
                raise ImportError(
                    "pyannote-audio required for pyannote VAD. "
                    "Install: pip install pyannote-audio"
                )
            # Load pre-trained VAD pipeline
            try:
                self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
                logger.info("Initialized pyannote VAD pipeline")
            except Exception as e:
                logger.error(f"Failed to load pyannote pipeline: {e}")
                raise
        
        else:
            raise ValueError(f"Unknown VAD strategy: {strategy}. Choose 'energy' or 'pyannote'")
    
    def detect(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio.
        
        Args:
            waveform: Audio waveform [T] or [1, T]
            sample_rate: Sample rate in Hz
        
        Returns:
            speech_segments: List of (start_time, end_time) tuples in seconds
        
        Example:
            >>> vad = VoiceActivityDetector(strategy='energy')
            >>> waveform = np.random.randn(16000)  # 1 second
            >>> segments = vad.detect(waveform, sample_rate=16000)
            >>> print(segments)
            [(0.0, 0.5), (0.7, 1.0)]
        """
        # Convert to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Ensure 1D
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        if self.strategy == "energy":
            return self._detect_energy_based(waveform, sample_rate)
        elif self.strategy == "pyannote":
            return self._detect_pyannote(waveform, sample_rate)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _detect_energy_based(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        """
        Energy-based VAD using short-time energy.
        
        Args:
            waveform: Audio waveform [T]
            sample_rate: Sample rate in Hz
        
        Returns:
            speech_segments: List of (start_time, end_time) tuples
        """
        # Compute short-time energy
        energy = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]
        
        # Normalize energy
        energy = energy / (np.max(energy) + 1e-8)
        
        # Threshold to get speech frames
        speech_frames = energy > self.energy_threshold
        
        # Convert frame indices to time
        times = librosa.frames_to_time(
            np.arange(len(speech_frames)),
            sr=sample_rate,
            hop_length=self.hop_length,
        )
        
        # Extract continuous segments
        segments = []
        in_speech = False
        start_time = 0.0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech segment
                start_time = times[i]
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                end_time = times[i]
                duration = end_time - start_time
                
                # Only keep segments longer than minimum duration
                if duration >= self.min_speech_duration:
                    segments.append((start_time, end_time))
                
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            end_time = len(waveform) / sample_rate
            duration = end_time - start_time
            if duration >= self.min_speech_duration:
                segments.append((start_time, end_time))
        
        # Merge segments that are too close together
        segments = self._merge_close_segments(segments, self.min_silence_duration)
        
        logger.debug(f"Detected {len(segments)} speech segments (energy-based)")
        return segments
    
    def _detect_pyannote(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        """
        Pyannote-based VAD using pre-trained model.
        
        Args:
            waveform: Audio waveform [T]
            sample_rate: Sample rate in Hz
        
        Returns:
            speech_segments: List of (start_time, end_time) tuples
        """
        # Pyannote expects dict with 'waveform' and 'sample_rate'
        audio = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": sample_rate,
        }
        
        # Run VAD pipeline
        vad_output = self.pipeline(audio)
        
        # Extract segments
        segments = [(segment.start, segment.end) for segment in vad_output.get_timeline()]
        
        logger.debug(f"Detected {len(segments)} speech segments (pyannote)")
        return segments
    
    def _merge_close_segments(
        self,
        segments: List[Tuple[float, float]],
        min_gap: float,
    ) -> List[Tuple[float, float]]:
        """
        Merge segments that are separated by less than min_gap.
        
        Args:
            segments: List of (start, end) tuples
            min_gap: Minimum gap duration in seconds
        
        Returns:
            merged_segments: Merged list of segments
        """
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            
            # Check if gap is small enough to merge
            if start - prev_end < min_gap:
                # Merge with previous segment
                merged[-1] = (prev_start, end)
            else:
                # Keep as separate segment
                merged.append((start, end))
        
        return merged
    
    def apply_mask(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        segments: List[Tuple[float, float]],
        sample_rate: int = 16000,
        mask_value: float = 0.0,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply VAD mask to waveform (zero out non-speech regions).
        
        Args:
            waveform: Audio waveform [T]
            segments: List of (start_time, end_time) speech segments
            sample_rate: Sample rate in Hz
            mask_value: Value to use for masked regions (default: 0.0)
        
        Returns:
            masked_waveform: Waveform with non-speech regions masked
        
        Example:
            >>> vad = VoiceActivityDetector()
            >>> waveform = np.random.randn(16000)
            >>> segments = vad.detect(waveform, 16000)
            >>> masked = vad.apply_mask(waveform, segments, 16000)
        """
        is_torch = isinstance(waveform, torch.Tensor)
        
        # Convert to numpy for processing
        if is_torch:
            device = waveform.device
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform.copy()
        
        # Ensure 1D
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.squeeze()
        
        # Create mask (all zeros initially)
        mask = np.zeros_like(waveform_np)
        
        # Set speech regions to 1
        for start_time, end_time in segments:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            start_idx = max(0, start_idx)
            end_idx = min(len(waveform_np), end_idx)
            mask[start_idx:end_idx] = 1.0
        
        # Apply mask
        masked_waveform = waveform_np * mask + mask_value * (1 - mask)
        
        # Convert back to torch if needed
        if is_torch:
            masked_waveform = torch.from_numpy(masked_waveform).to(device)
        
        # Compute percentage of speech
        speech_ratio = mask.sum() / len(mask)
        logger.debug(f"Speech ratio: {speech_ratio:.2%}")
        
        return masked_waveform
    
    def get_attention_mask(
        self,
        segments: List[Tuple[float, float]],
        total_duration: float,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Create attention mask for transformer models.
        
        Args:
            segments: List of (start_time, end_time) speech segments
            total_duration: Total audio duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            attention_mask: Binary mask [T] where 1 = speech, 0 = silence
        """
        total_samples = int(total_duration * sample_rate)
        mask = torch.zeros(total_samples, dtype=torch.long)
        
        for start_time, end_time in segments:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            start_idx = max(0, start_idx)
            end_idx = min(total_samples, end_idx)
            mask[start_idx:end_idx] = 1
        
        return mask
