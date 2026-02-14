
import os
import glob
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_silent_audio(
    processed_root: str,
    fps: int = 24,
    sample_rate: int = 16000,
    min_seconds: float = 4.0  # Ensure at least 4 seconds to cover 3.0s crop
):
    processed_root = Path(processed_root)
    audio_root = processed_root / 'audio'
    
    splits = ['train', 'val', 'test']
    labels = ['real', 'fake']
    
    total_generated = 0
    total_skipped = 0
    
    for split in splits:
        for label in labels:
            frame_dir = processed_root / split / label
            audio_dir = audio_root / split / label
            
            if not frame_dir.exists():
                logger.warning(f"Frame directory not found: {frame_dir}")
                continue
                
            logger.info(f"Processing {split}/{label}...")
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Group frames by video_id
            # Pattern: video_ID_frame_XXXX.jpg
            # We scan the directory efficiently
            
            video_frame_counts = {}
            
            # Using simple glob is safer but might be slow for many files.
            # Using scandir is faster.
            with os.scandir(frame_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.jpg'):
                        # Parse video_id
                        # Expecting: video_xxxx_frame_yyyy.jpg
                        # Split by '_frame'
                        if '_frame_' in entry.name:
                            vid_id = entry.name.split('_frame_')[0]
                            video_frame_counts[vid_id] = video_frame_counts.get(vid_id, 0) + 1
                            
            logger.info(f"Found {len(video_frame_counts)} videos in {split}/{label}")
            
            for vid_id, count in tqdm(video_frame_counts.items(), desc=f"Gen Audio {split}/{label}"):
                output_path = audio_dir / f"{vid_id}.wav"
                
                if output_path.exists():
                    total_skipped += 1
                    continue
                
                # Calculate duration
                duration = count / float(fps)
                # Enforce minimum duration for training stability
                if duration < min_seconds:
                    duration = min_seconds
                
                # Generate silence
                # 16-bit PCM WAV
                num_samples = int(duration * sample_rate)
                # Create numpy array of zeros
                data = np.zeros(num_samples, dtype=np.int16)
                
                try:
                    wavfile.write(str(output_path), sample_rate, data)
                    total_generated += 1
                except Exception as e:
                    logger.error(f"Failed to write {output_path}: {e}")
                    
    logger.info("Silent Audio Generation Complete.")
    logger.info(f"Generated: {total_generated}")
    logger.info(f"Skipped: {total_skipped}")

if __name__ == "__main__":
    PROCESSED_ROOT = r"d:\project\DeepFakeDetection\model-service\data\processed"
    generate_silent_audio(PROCESSED_ROOT)
