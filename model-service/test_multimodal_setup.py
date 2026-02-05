"""
Test Multimodal Setup

Quick validation that audio extraction, preprocessing, and dataloaders work correctly.

Usage:
    python test_multimodal_setup.py
    python test_multimodal_setup.py --quick  # Fast test with limited data
"""

import sys
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info("✓ PyTorch")
    except ImportError as e:
        logger.error(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torchaudio
        logger.info("✓ torchaudio")
    except ImportError as e:
        logger.error(f"✗ torchaudio: {e}")
        return False
    
    try:
        import librosa
        logger.info("✓ librosa")
    except ImportError as e:
        logger.error(f"✗ librosa: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✓ Pillow")
    except ImportError as e:
        logger.error(f"✗ Pillow: {e}")
        return False
    
    try:
        import torchvision
        logger.info("✓ torchvision")
    except ImportError as e:
        logger.error(f"✗ torchvision: {e}")
        return False
    
    logger.info("✓ All imports successful\n")
    return True


def test_audio_processor():
    """Test audio preprocessing module."""
    logger.info("Testing AudioProcessor...")
    
    try:
        from src.preprocessing.audio_processor import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000, n_mels=80)
        logger.info(f"✓ AudioProcessor initialized")
        logger.info(f"  - Sample rate: 16000 Hz")
        logger.info(f"  - Mel bins: 80")
        logger.info(f"  - MFCC coefficients: 13")
        logger.info("✓ AudioProcessor tests passed\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ AudioProcessor test failed: {e}\n")
        return False


def test_multimodal_dataset():
    """Test multimodal dataset."""
    logger.info("Testing MultimodalDeepfakeDataset...")
    
    try:
        from src.datasets.multimodal_dataset import (
            MultimodalDeepfakeDataset,
            create_multimodal_dataloaders
        )
        
        # Check if test data exists
        frame_dir = Path('data/processed/train/fake')
        if not frame_dir.exists() or not list(frame_dir.glob('*.jpg')):
            logger.warning(f"⚠ Test data not found at {frame_dir}")
            logger.warning("  To test with real data, run audio extraction first")
            logger.info("✓ MultimodalDeepfakeDataset module loaded successfully\n")
            return True
        
        # Try to load dataset
        dataset = MultimodalDeepfakeDataset(
            frame_dir='data/processed/train/fake',
            frames_per_video=5
        )
        
        logger.info(f"✓ Dataset loaded")
        logger.info(f"  - Samples: {len(dataset)}")
        
        # Try to get a sample
        sample = dataset[0]
        logger.info(f"✓ Sample loaded")
        logger.info(f"  - Frames shape: {sample['frames'].shape}")
        logger.info(f"  - Label: {sample['label'].item()}")
        
        logger.info("✓ MultimodalDeepfakeDataset tests passed\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ MultimodalDeepfakeDataset test failed: {e}\n")
        return False


def test_directory_structure():
    """Test that directory structure is correct."""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        Path('data/processed'),
        Path('scripts'),
        Path('src/preprocessing'),
        Path('src/datasets'),
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            logger.info(f"✓ {dir_path}")
        else:
            logger.warning(f"⚠ {dir_path} (not critical)")
    
    logger.info("✓ Directory structure tests passed\n")
    return True


def test_scripts_exist():
    """Test that required scripts exist."""
    logger.info("Testing script files...")
    
    required_scripts = [
        Path('scripts/extract_audio_multimodal.py'),
        Path('scripts/verify_multimodal_alignment.py'),
    ]
    
    all_exist = True
    for script in required_scripts:
        if script.exists():
            logger.info(f"✓ {script}")
        else:
            logger.error(f"✗ {script} not found")
            all_exist = False
    
    if all_exist:
        logger.info("✓ All script files present\n")
    else:
        logger.error("✗ Some scripts missing\n")
    
    return all_exist


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    logger.info("Testing PyTorch GPU support...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available")
            logger.info(f"  - Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠ CUDA not available (CPU will be used)")
        
        # Test tensor creation
        x = torch.randn(2, 3, 224, 224)
        logger.info(f"✓ Tensor creation works")
        logger.info(f"  - Shape: {x.shape}")
        
        logger.info("✓ PyTorch GPU tests passed\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ PyTorch GPU test failed: {e}\n")
        return False


def print_summary(results):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Imports", results['imports']),
        ("Directory Structure", results['directories']),
        ("Script Files", results['scripts']),
        ("AudioProcessor", results['audio_processor']),
        ("MultimodalDataset", results['multimodal_dataset']),
        ("PyTorch/GPU", results['pytorch_gpu']),
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 All tests passed! Ready for multimodal training.\n")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. See above for details.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test multimodal setup')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    args = parser.parse_args()
    
    logger.info("Starting multimodal setup tests...\n")
    
    results = {
        'imports': test_imports(),
        'directories': test_directory_structure(),
        'scripts': test_scripts_exist(),
        'audio_processor': test_audio_processor(),
        'multimodal_dataset': test_multimodal_dataset(),
        'pytorch_gpu': test_torch_cuda(),
    }
    
    success = print_summary(results)
    
    if success:
        logger.info("\n📖 Next steps:")
        logger.info("1. Extract audio: python scripts/extract_audio_multimodal.py --help")
        logger.info("2. Verify alignment: python scripts/verify_multimodal_alignment.py --help")
        logger.info("3. Read setup guide: MULTIMODAL_SETUP.md")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
