"""
FakeAVCeleb Dataset Processor

Extract and organize FakeAVCeleb dataset for cross-dataset validation.
Tests model generalization to a completely different dataset.

Usage:
    python fakeavceleb_processor.py \
        --input downloads/FakeAVCeleb_v1.2.zip \
        --output data/fakeavceleb \
        --extract-audio
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FakeAVCelebProcessor:
    """
    Process FakeAVCeleb dataset for cross-dataset validation.
    
    FakeAVCeleb contains:
    - Real videos from VoxCeleb2
    - Fake videos with various manipulations (audio, video, audio-video)
    """
    
    def __init__(self, input_path: Path, output_dir: Path):
        """
        Initialize processor.
        
        Args:
            input_path: Path to FakeAVCeleb zip file
            output_dir: Output directory for processed dataset
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.temp_dir = output_dir / "temp"
        
    def extract_zip(self):
        """Extract FakeAVCeleb zip file."""
        logger.info(f"Extracting {self.input_path.name}...")
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        logger.info(f"✓ Extracted to {self.temp_dir}")
    
    def organize_dataset(self):
        """
        Organize FakeAVCeleb into real/fake directories.
        
        FakeAVCeleb structure:
        - RealVideo-RealAudio/ (real)
        - FakeVideo-RealAudio/ (fake - video manipulation)
        - RealVideo-FakeAudio/ (fake - audio manipulation)
        - FakeVideo-FakeAudio/ (fake - both manipulated)
        """
        logger.info("Organizing FakeAVCeleb dataset...")
        
        # Define category mappings
        categories = {
            'real': ['RealVideo-RealAudio'],
            'fake': ['FakeVideo-RealAudio', 'RealVideo-FakeAudio', 'FakeVideo-FakeAudio']
        }
        
        stats = {'real': 0, 'fake': 0}
        
        for label, category_names in categories.items():
            label_dir = self.output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            for category in category_names:
                category_path = self.temp_dir / "FakeAVCeleb_v1.2" / category
                
                if not category_path.exists():
                    # Try alternative paths
                    category_path = self.temp_dir / category
                
                if category_path.exists():
                    video_files = list(category_path.glob("*.mp4")) + list(category_path.glob("*.avi"))
                    
                    logger.info(f"  Processing {category}: {len(video_files)} videos")
                    
                    for video_file in video_files:
                        # Create unique filename with category prefix
                        new_name = f"{category}_{video_file.name}"
                        dest_path = label_dir / new_name
                        
                        # Copy or symlink
                        if not dest_path.exists():
                            try:
                                dest_path.symlink_to(video_file)
                            except (OSError, NotImplementedError):
                                shutil.copy2(video_file, dest_path)
                        
                        stats[label] += 1
                else:
                    logger.warning(f"  Category not found: {category}")
        
        logger.info(f"\n✓ Dataset organized:")
        logger.info(f"  - Real videos: {stats['real']}")
        logger.info(f"  - Fake videos: {stats['fake']}")
        logger.info(f"  - Total: {stats['real'] + stats['fake']}")
        
        return stats
    
    def create_metadata(self, stats: Dict[str, int]):
        """Create metadata file for cross-dataset validation."""
        metadata = {
            'dataset_name': 'FakeAVCeleb',
            'version': 'v1.2',
            'description': 'Celebrity deepfake dataset with audio-video manipulations',
            'num_real': stats['real'],
            'num_fake': stats['fake'],
            'total_videos': stats['real'] + stats['fake'],
            'manipulation_types': [
                'FakeVideo-RealAudio',
                'RealVideo-FakeAudio',
                'FakeVideo-FakeAudio'
            ],
            'usage': 'cross_dataset_validation',
            'data_dir': str(self.output_dir.absolute())
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to {metadata_path}")
        
        return metadata
    
    def cleanup_temp(self):
        """Remove temporary extraction directory."""
        if self.temp_dir.exists():
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Process FakeAVCeleb dataset for cross-dataset validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and organize FakeAVCeleb
  python fakeavceleb_processor.py \\
      --input downloads/FakeAVCeleb_v1.2.zip \\
      --output data/fakeavceleb
  
  # Keep temporary files (for debugging)
  python fakeavceleb_processor.py \\
      --input downloads/FakeAVCeleb_v1.2.zip \\
      --output data/fakeavceleb \\
      --keep-temp
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to FakeAVCeleb zip file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for processed dataset'
    )
    
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary extraction files'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Process dataset
    processor = FakeAVCelebProcessor(args.input, args.output)
    
    try:
        # Extract
        processor.extract_zip()
        
        # Organize
        stats = processor.organize_dataset()
        
        # Create metadata
        metadata = processor.create_metadata(stats)
        
        # Cleanup
        if not args.keep_temp:
            processor.cleanup_temp()
        
        # Print summary
        print("\n" + "="*60)
        print("FAKEAVCELEB PROCESSING COMPLETE")
        print("="*60)
        print(f"\nDataset: {metadata['dataset_name']} {metadata['version']}")
        print(f"Real videos: {metadata['num_real']}")
        print(f"Fake videos: {metadata['num_fake']}")
        print(f"Total: {metadata['total_videos']}")
        print(f"\nOutput directory: {args.output}")
        print(f"  - {args.output / 'real'}/ (real videos)")
        print(f"  - {args.output / 'fake'}/ (fake videos)")
        print(f"  - {args.output / 'dataset_metadata.json'} (metadata)")
        print("\nNext steps:")
        print("  1. Extract audio: python scripts/extract_audio_multimodal.py ...")
        print("  2. Run cross-dataset eval: python src/eval/cross_dataset_evaluator.py ...")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
