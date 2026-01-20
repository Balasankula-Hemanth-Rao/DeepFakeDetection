"""
Preprocessing package for video and audio data.

This package contains preprocessing utilities:
- Frame extraction from videos
- Audio extraction from videos
- Voice activity detection
- Optical flow computation
- Face detection and alignment
"""

from .extract_frames import extract_frames, FrameExtractor
from .extract_audio import AudioExtractor
from .voice_activity_detection import VoiceActivityDetector
from .optical_flow import OpticalFlowExtractor

__all__ = [
    'extract_frames',
    'FrameExtractor',
    'AudioExtractor',
    'VoiceActivityDetector',
    'OpticalFlowExtractor',
]
