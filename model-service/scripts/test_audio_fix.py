
import sys
import os
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.getcwd())

def test_audio_loading():
    logger.info("Importing AudioProcessor...")
    try:
        from src.preprocessing.audio_processor import AudioProcessor
        logger.info("AudioProcessor imported successfully.")
    except Exception as e:
        logger.error(f"Failed to import AudioProcessor: {e}")
        return

    # Check for available audio files
    processed_root = Path("data/processed")
    audio_file = None
    
    # search for a wav file
    for p in processed_root.rglob("*.wav"):
        audio_file = p
        break
        
    if not audio_file:
        logger.error("No audio files found in data/processed")
        return
        
    logger.info(f"Testing audio loading with: {audio_file}")
    
    try:
        processor = AudioProcessor()
        waveform, sr = processor.load_audio(audio_file)
        logger.info(f"Loaded audio successfully. Shape: {waveform.shape}, SR: {sr}")
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_loading()
