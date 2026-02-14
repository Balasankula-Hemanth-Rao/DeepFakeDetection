import torch
from pathlib import Path
from src.eval.cross_dataset_evaluator import CrossDatasetDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loading():
    dataset_dir = Path("data/fakeavceleb")
    
    print(f"Testing CrossDatasetDataset loading from {dataset_dir}...")
    
    try:
        dataset = CrossDatasetDataset(
            dataset_dir=dataset_dir,
            frames_per_video=5,
            audio_feature='spectrogram',
            max_samples=10  # Test 10 samples
        )
        
        if len(dataset) == 0:
            print("[FAIL] Dataset is empty!")
            return
            
        print(f"[OK] Dataset loaded {len(dataset)} samples.")
        
        # Test getting a sample
        sample = dataset[0]
        frames = sample['frames']
        audio = sample['audio']
        label = sample['label']
        
        print(f"[OK] Loaded sample:")
        print(f"  - Frames shape: {frames.shape} (Expected: [5, 3, 224, 224])")
        print(f"  - Audio shape: {audio.shape}")
        print(f"  - Label: {label}")
        print(f"  - Video ID: {sample['video_id']}")
        
        # Check for non-black frames (if working)
        if frames.sum() == 0:
            print("[WARN] Frames are all zeros! (Using fallback black frames?)")
        else:
            print("[OK] Frames contain data.")
            
        print("\nSUCCESS: FakeAVCeleb loading verified!")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
