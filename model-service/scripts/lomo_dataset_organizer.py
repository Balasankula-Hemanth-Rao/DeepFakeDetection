"""
LOMO Dataset Organizer for FaceForensics++

This script organizes the FaceForensics++ dataset by manipulation method
and creates LOMO (Leave-One-Method-Out) split configurations.

Usage:
    python lomo_dataset_organizer.py \
        --input ../FaceForensics-master \
        --output data/lomo \
        --verbose
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# FaceForensics++ manipulation methods
MANIPULATION_METHODS = ["DeepFakes", "FaceSwap", "Face2Face", "NeuralTextures"]
METHOD_ABBREVIATIONS = {
    "DeepFakes": "DF",
    "FaceSwap": "FS",
    "Face2Face": "F2F",
    "NeuralTextures": "NT"
}


def find_faceforensics_videos(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan FaceForensics++ directory and find videos by manipulation method.
    
    Args:
        input_dir: Root directory of FaceForensics++ dataset
        
    Returns:
        Dictionary mapping method names to list of video paths
    """
    logger.info(f"Scanning FaceForensics++ dataset at: {input_dir}")
    
    videos_by_method = {method: [] for method in MANIPULATION_METHODS}
    videos_by_method["original"] = []
    
    # Check for manipulated sequences
    manipulated_dir = input_dir / "manipulated_sequences"
    if manipulated_dir.exists():
        for method in MANIPULATION_METHODS:
            method_dir = manipulated_dir / method / "c23" / "videos"
            if not method_dir.exists():
                # Try alternative structure
                method_dir = manipulated_dir / method / "raw" / "videos"
            
            if method_dir.exists():
                video_files = list(method_dir.glob("*.mp4")) + list(method_dir.glob("*.avi"))
                videos_by_method[method] = sorted(video_files)
                logger.info(f"  - {method}: {len(video_files)} videos")
            else:
                logger.warning(f"  - {method}: directory not found")
    
    # Check for original sequences
    original_dir = input_dir / "original_sequences" / "youtube" / "c23" / "videos"
    if not original_dir.exists():
        original_dir = input_dir / "original_sequences" / "youtube" / "raw" / "videos"
    
    if original_dir.exists():
        video_files = list(original_dir.glob("*.mp4")) + list(original_dir.glob("*.avi"))
        videos_by_method["original"] = sorted(video_files)
        logger.info(f"  - original: {len(video_files)} videos")
    else:
        logger.warning(f"  - original: directory not found")
    
    return videos_by_method


def organize_by_method(videos_by_method: Dict[str, List[Path]], output_dir: Path, copy_files: bool = False):
    """
    Organize videos into method-specific directories.
    
    Args:
        videos_by_method: Dictionary of method -> video paths
        output_dir: Output directory for organized dataset
        copy_files: If True, copy files. If False, create symlinks (Unix) or copy (Windows)
    """
    logger.info(f"Organizing videos into: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method, video_paths in videos_by_method.items():
        if not video_paths:
            continue
            
        method_output_dir = output_dir / method
        method_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {method}: {len(video_paths)} videos")
        
        for video_path in video_paths:
            dest_path = method_output_dir / video_path.name
            
            if dest_path.exists():
                continue
            
            if copy_files:
                shutil.copy2(video_path, dest_path)
            else:
                # Try symlink, fallback to copy on Windows
                try:
                    dest_path.symlink_to(video_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(video_path, dest_path)
        
        logger.info(f"  ✓ {method}: {len(list(method_output_dir.iterdir()))} videos organized")


def create_lomo_split_config(
    split_num: int,
    test_method: str,
    train_methods: List[str],
    data_dir: Path,
    output_dir: Path
) -> Path:
    """
    Create a LOMO split configuration file.
    
    Args:
        split_num: Split number (1-4)
        test_method: Method to exclude for testing
        train_methods: Methods to include for training
        data_dir: Directory containing organized data
        output_dir: Directory to save config files
        
    Returns:
        Path to created config file
    """
    config = {
        "split_name": f"LOMO Split {split_num}: Test on {test_method}",
        "split_number": split_num,
        "test_method": test_method,
        "test_method_abbrev": METHOD_ABBREVIATIONS[test_method],
        "train_methods": train_methods,
        "train_methods_abbrev": [METHOD_ABBREVIATIONS[m] for m in train_methods],
        "data_dir": str(data_dir.absolute()),
        "test_method_dir": str((data_dir / test_method).absolute()),
        "original_dir": str((data_dir / "original").absolute()),
        "description": f"Train on {', '.join(train_methods)} and test generalization on unseen {test_method} method"
    }
    
    # Create output directory for configs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config file
    config_filename = f"lomo_split_{split_num}_test_{test_method.lower()}.json"
    config_path = output_dir / config_filename
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created config: {config_path.name}")
    return config_path


def create_all_lomo_splits(data_dir: Path, output_dir: Path) -> List[Path]:
    """
    Create all 4 LOMO split configurations.
    
    Args:
        data_dir: Directory containing organized data by method
        output_dir: Directory to save config files
        
    Returns:
        List of paths to created config files
    """
    logger.info("Creating LOMO split configurations...")
    
    config_paths = []
    
    for i, test_method in enumerate(MANIPULATION_METHODS, start=1):
        train_methods = [m for m in MANIPULATION_METHODS if m != test_method]
        
        config_path = create_lomo_split_config(
            split_num=i,
            test_method=test_method,
            train_methods=train_methods,
            data_dir=data_dir,
            output_dir=output_dir
        )
        config_paths.append(config_path)
    
    logger.info(f"✓ Created {len(config_paths)} LOMO split configurations")
    return config_paths


def print_summary(videos_by_method: Dict[str, List[Path]], config_paths: List[Path]):
    """Print summary of dataset organization."""
    print("\n" + "="*60)
    print("LOMO DATASET ORGANIZATION SUMMARY")
    print("="*60)
    
    print("\nDataset Statistics:")
    total_fake = 0
    for method in MANIPULATION_METHODS:
        count = len(videos_by_method.get(method, []))
        total_fake += count
        print(f"  - {method:20s}: {count:4d} videos")
    
    original_count = len(videos_by_method.get("original", []))
    print(f"  - {'original':20s}: {original_count:4d} videos")
    print(f"  - {'TOTAL FAKE':20s}: {total_fake:4d} videos")
    print(f"  - {'TOTAL DATASET':20s}: {total_fake + original_count:4d} videos")
    
    print("\nLOMO Splits Created:")
    for i, config_path in enumerate(config_paths, start=1):
        with open(config_path) as f:
            config = json.load(f)
        print(f"  Split {i}: Train on {', '.join(config['train_methods'])[:40]:40s} → Test on {config['test_method']}")
    
    print("\nConfiguration Files:")
    for config_path in config_paths:
        print(f"  - {config_path}")
    
    print("\n" + "="*60)
    print("✓ Dataset organization complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Organize FaceForensics++ dataset for LOMO evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize dataset and create LOMO splits
  python lomo_dataset_organizer.py \\
      --input ../FaceForensics-master \\
      --output data/lomo
  
  # Copy files instead of symlinks
  python lomo_dataset_organizer.py \\
      --input ../FaceForensics-master \\
      --output data/lomo \\
      --copy
        """
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to FaceForensics++ root directory"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for organized dataset"
    )
    
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Output directory for LOMO config files (default: <output>/configs/lomo_splits)"
    )
    
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate input directory
    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return 1
    
    # Set config output directory
    if args.config_output is None:
        args.config_output = args.output.parent / "configs" / "lomo_splits"
    
    # Step 1: Find all videos by method
    videos_by_method = find_faceforensics_videos(args.input)
    
    # Check if we found any videos
    total_videos = sum(len(videos) for videos in videos_by_method.values())
    if total_videos == 0:
        logger.error("No videos found in input directory")
        return 1
    
    # Step 2: Organize videos by method
    organize_by_method(videos_by_method, args.output, copy_files=args.copy)
    
    # Step 3: Create LOMO split configurations
    config_paths = create_all_lomo_splits(args.output, args.config_output)
    
    # Step 4: Print summary
    print_summary(videos_by_method, config_paths)
    
    return 0


if __name__ == "__main__":
    exit(main())
