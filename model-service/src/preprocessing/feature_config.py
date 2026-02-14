from dataclasses import dataclass

@dataclass
class AudioConfig:
    """Configuration for audio feature extraction."""
    sample_rate: int = 16000
    n_bit: int = 16
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    n_mels: int = 80
    n_mfcc: int = 13
    f_min: float = 50.0
    f_max: float = 8000.0
    power: float = 2.0
    normalized: bool = True
