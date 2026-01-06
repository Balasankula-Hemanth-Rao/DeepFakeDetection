#!/usr/bin/env python3
"""
Dataset Integrity and Leakage Audit Module

Validates train/val/test splits for data leakage by checking:
- Identity overlap across splits (same person in multiple splits)
- Video file hash overlap (same video in multiple splits)
- Audio track hash overlap (same audio in multiple splits)
- Encoding/compression metadata similarity

Usage:
    python audit_dataset.py --config config/config.yaml
    python audit_dataset.py --config config/config.yaml --output reports/audit.json
    python audit_dataset.py --config config/config.yaml --splits train val test
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

# Optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False

try:
    from src.config import get_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


logger = logging.getLogger(__name__)


class DatasetAuditor:
    """
    Audits dataset for integrity and data leakage risks.
    
    Checks for:
    - Identity overlap across splits
    - Video hash collisions (same file content)
    - Audio hash collisions
    - Encoding metadata similarities (suspicious compression)
    """
    
    def __init__(self, data_root: Path, config=None, splits: Optional[List[str]] = None):
        """
        Initialize auditor.
        
        Args:
            data_root: Root directory of dataset
            config: Configuration object (optional)
            splits: List of splits to audit (default: ['train', 'val', 'test'])
        """
        self.data_root = Path(data_root)
        self.config = config
        self.splits = splits or ['train', 'val', 'test']
        
        # Storage for audit results
        self.samples_by_split: Dict[str, List[Dict]] = defaultdict(list)
        self.hashes_by_split: Dict[str, Dict[str, List[str]]] = {
            'video': defaultdict(list),
            'audio': defaultdict(list),
        }
        self.metadata_by_split: Dict[str, List[Dict]] = defaultdict(list)
        
        # Leakage findings
        self.findings = {
            'identity_overlap': defaultdict(list),
            'video_hash_collision': defaultdict(list),
            'audio_hash_collision': defaultdict(list),
            'metadata_similarity': defaultdict(list),
            'errors': [],
        }
        
    def audit(self) -> Dict[str, Any]:
        """
        Run full audit and return results.
        
        Returns:
            Dictionary with audit results and leakage warnings
        """
        logger.info(f"Starting dataset audit for splits: {self.splits}")
        start_time = datetime.now()
        
        try:
            # Load all samples
            self._load_samples()
            
            if not any(self.samples_by_split.values()):
                raise ValueError(f"No samples found in {self.data_root}")
            
            # Extract metadata and compute hashes
            self._extract_metadata()
            
            # Check for leakage
            self._check_identity_overlap()
            self._check_video_hash_collisions()
            self._check_audio_hash_collisions()
            self._check_metadata_similarity()
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            self.findings['errors'].append({
                'type': type(e).__name__,
                'message': str(e),
            })
        
        # Compile report
        elapsed = (datetime.now() - start_time).total_seconds()
        report = self._compile_report(elapsed)
        
        logger.info(f"Audit completed in {elapsed:.2f}s")
        return report
    
    def _load_samples(self) -> None:
        """Load all samples from directory structure."""
        for split in self.splits:
            split_dir = self.data_root / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            # Find all video files (MP4, AVI, MOV, etc.)
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            sample_dirs = []
            
            for item in split_dir.iterdir():
                if item.is_dir():
                    # Look for video files in subdirectories
                    videos = [f for f in item.iterdir() 
                             if f.suffix.lower() in video_extensions]
                    if videos:
                        sample_dirs.append(item)
                elif item.suffix.lower() in video_extensions:
                    # Direct video file in split dir
                    sample_dirs.append(item.parent)
            
            logger.info(f"Found {len(sample_dirs)} samples in split '{split}'")
            
            # Extract metadata for each sample
            for sample_dir in sample_dirs:
                try:
                    sample_info = self._extract_sample_info(sample_dir, split)
                    if sample_info:
                        self.samples_by_split[split].append(sample_info)
                except Exception as e:
                    self.findings['errors'].append({
                        'split': split,
                        'sample': str(sample_dir),
                        'error': str(e),
                    })
    
    def _extract_sample_info(self, sample_dir: Path, split: str) -> Optional[Dict]:
        """
        Extract metadata for a single sample.
        
        Args:
            sample_dir: Directory containing sample
            split: Split name (train/val/test)
            
        Returns:
            Dictionary with sample metadata or None if extraction fails
        """
        sample_info = {
            'split': split,
            'sample_id': sample_dir.name,
            'path': str(sample_dir),
        }
        
        # Find video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = [f for f in sample_dir.iterdir() 
                      if f.suffix.lower() in video_extensions]
        
        if not video_files:
            return None
        
        video_path = video_files[0]
        sample_info['video_path'] = str(video_path)
        
        # Compute video hash
        try:
            sample_info['video_hash'] = self._compute_file_hash(video_path)
        except Exception as e:
            logger.debug(f"Failed to hash video {video_path}: {e}")
        
        # Extract video metadata
        try:
            video_meta = self._extract_video_metadata(video_path)
            sample_info['video_metadata'] = video_meta
        except Exception as e:
            logger.debug(f"Failed to extract video metadata: {e}")
        
        # Find and hash audio
        audio_files = [f for f in sample_dir.iterdir() 
                      if f.suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.npy'}]
        
        if audio_files:
            audio_path = audio_files[0]
            sample_info['audio_path'] = str(audio_path)
            try:
                sample_info['audio_hash'] = self._compute_file_hash(audio_path)
            except Exception as e:
                logger.debug(f"Failed to hash audio {audio_path}: {e}")
            
            # Extract audio metadata
            try:
                audio_meta = self._extract_audio_metadata(audio_path)
                sample_info['audio_metadata'] = audio_meta
            except Exception as e:
                logger.debug(f"Failed to extract audio metadata: {e}")
        
        # Try to load metadata JSON
        meta_file = sample_dir / 'meta.json'
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    sample_info['json_metadata'] = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load metadata JSON: {e}")
        
        return sample_info
    
    def _compute_file_hash(self, filepath: Path, chunk_size: int = 65536) -> str:
        """
        Compute SHA256 hash of file.
        
        Args:
            filepath: Path to file
            chunk_size: Size of chunks to read (for efficiency)
            
        Returns:
            Hex digest of file hash
        """
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _extract_video_metadata(self, video_path: Path) -> Optional[Dict]:
        """
        Extract video metadata (codec, resolution, fps, etc.).
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with metadata or None if cv2 not available
        """
        if not HAS_CV2:
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            metadata = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            cap.release()
            
            return metadata
        except Exception as e:
            logger.debug(f"Error extracting video metadata: {e}")
            return None
    
    def _extract_audio_metadata(self, audio_path: Path) -> Optional[Dict]:
        """
        Extract audio metadata (sample rate, duration, etc.).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with metadata or None if torchaudio not available
        """
        if audio_path.suffix.lower() == '.npy':
            # Handle preextracted features
            try:
                arr = np.load(audio_path)
                return {
                    'shape': list(arr.shape),
                    'dtype': str(arr.dtype),
                }
            except Exception as e:
                logger.debug(f"Error loading NPY audio: {e}")
                return None
        
        if not HAS_TORCHAUDIO:
            return None
        
        try:
            info = torchaudio.info(str(audio_path))
            return {
                'sample_rate': info.sample_rate,
                'num_frames': info.num_frames,
                'num_channels': info.num_channels,
                'duration_seconds': info.num_frames / info.sample_rate,
            }
        except Exception as e:
            logger.debug(f"Error extracting audio metadata: {e}")
            return None
    
    def _extract_metadata(self) -> None:
        """Build hash indices and metadata storage."""
        for split in self.splits:
            for sample in self.samples_by_split[split]:
                self.metadata_by_split[split].append(sample)
                
                # Index hashes
                if 'video_hash' in sample:
                    hash_val = sample['video_hash']
                    self.hashes_by_split['video'][hash_val].append(sample['sample_id'])
                
                if 'audio_hash' in sample:
                    hash_val = sample['audio_hash']
                    self.hashes_by_split['audio'][hash_val].append(sample['sample_id'])
    
    def _check_identity_overlap(self) -> None:
        """
        Check for identity overlap across splits.
        
        Looks for sample IDs that suggest same person/identity in multiple splits.
        This is heuristic-based (same prefix, etc.) unless metadata provides identity info.
        """
        # Extract identities from sample IDs if available
        identities_by_split = defaultdict(set)
        
        for split in self.splits:
            for sample in self.samples_by_split[split]:
                sample_id = sample['sample_id']
                
                # Try to extract identity from sample ID
                # Common patterns: "person_001_video_01", "id_123_variation_a", etc.
                identity = self._extract_identity(sample_id)
                
                if identity:
                    identities_by_split[split].add(identity)
                
                # Also check JSON metadata for identity/person/speaker info
                if 'json_metadata' in sample:
                    meta = sample['json_metadata']
                    for key in ['identity', 'person', 'speaker', 'actor', 'subject_id']:
                        if key in meta:
                            identities_by_split[split].add(str(meta[key]))
        
        # Check for overlaps
        if len(self.splits) >= 2:
            all_splits = list(self.splits)
            for i, split1 in enumerate(all_splits):
                for split2 in all_splits[i+1:]:
                    overlap = identities_by_split[split1] & identities_by_split[split2]
                    if overlap:
                        self.findings['identity_overlap'][(split1, split2)] = list(overlap)
    
    def _check_video_hash_collisions(self) -> None:
        """Check for identical videos across splits."""
        video_hashes = self.hashes_by_split['video']
        
        for hash_val, sample_ids in video_hashes.items():
            if len(sample_ids) < 2:
                continue
            
            # Find which splits contain this hash
            splits_with_hash = defaultdict(list)
            for sample_id in sample_ids:
                for split in self.splits:
                    if any(s['sample_id'] == sample_id for s in self.samples_by_split[split]):
                        splits_with_hash[split].append(sample_id)
            
            # Check for cross-split collisions
            split_list = list(splits_with_hash.keys())
            if len(split_list) >= 2:
                for i, split1 in enumerate(split_list):
                    for split2 in split_list[i+1:]:
                        self.findings['video_hash_collision'][(split1, split2)].append({
                            'hash': hash_val[:16] + '...',
                            'split1_samples': splits_with_hash[split1][:3],  # Limit output
                            'split2_samples': splits_with_hash[split2][:3],
                        })
    
    def _check_audio_hash_collisions(self) -> None:
        """Check for identical audio tracks across splits."""
        audio_hashes = self.hashes_by_split['audio']
        
        for hash_val, sample_ids in audio_hashes.items():
            if len(sample_ids) < 2:
                continue
            
            # Find which splits contain this hash
            splits_with_hash = defaultdict(list)
            for sample_id in sample_ids:
                for split in self.splits:
                    if any(s['sample_id'] == sample_id for s in self.samples_by_split[split]):
                        splits_with_hash[split].append(sample_id)
            
            # Check for cross-split collisions
            split_list = list(splits_with_hash.keys())
            if len(split_list) >= 2:
                for i, split1 in enumerate(split_list):
                    for split2 in split_list[i+1:]:
                        self.findings['audio_hash_collision'][(split1, split2)].append({
                            'hash': hash_val[:16] + '...',
                            'split1_samples': splits_with_hash[split1][:3],
                            'split2_samples': splits_with_hash[split2][:3],
                        })
    
    def _check_metadata_similarity(self) -> None:
        """
        Check for suspicious metadata similarities.
        
        Detects potential encoding/compression artifacts that suggest same source.
        """
        for split in self.splits:
            metadata_groups = defaultdict(list)
            
            for sample in self.samples_by_split[split]:
                if 'video_metadata' in sample:
                    meta = sample['video_metadata']
                    
                    # Group by resolution and codec
                    key = (meta.get('width'), meta.get('height'), meta.get('fourcc'))
                    metadata_groups[key].append(sample['sample_id'])
            
            # Identify suspicious clusters (same exact resolution + codec)
            for (width, height, fourcc), sample_ids in metadata_groups.items():
                if len(sample_ids) > 10:  # Threshold for suspicion
                    # Check if audio metadata is also identical
                    audio_meta_groups = defaultdict(list)
                    
                    for sample in self.samples_by_split[split]:
                        if sample['sample_id'] in sample_ids and 'audio_metadata' in sample:
                            audio_meta = sample['audio_metadata']
                            key = (audio_meta.get('sample_rate'), 
                                   audio_meta.get('num_channels'))
                            audio_meta_groups[key].append(sample['sample_id'])
                    
                    if audio_meta_groups:
                        for audio_key, audio_samples in audio_meta_groups.items():
                            if len(audio_samples) > 5:
                                self.findings['metadata_similarity'][split].append({
                                    'video_resolution': f"{width}x{height}",
                                    'codec': fourcc,
                                    'audio_config': audio_key,
                                    'sample_count': len(audio_samples),
                                    'samples': audio_samples[:5],
                                })
    
    def _extract_identity(self, sample_id: str) -> Optional[str]:
        """
        Extract identity from sample ID using heuristics.
        
        Args:
            sample_id: Sample directory name
            
        Returns:
            Identity string or None
        """
        # Common patterns: person_001, speaker_05, actor_123_var_a, etc.
        parts = sample_id.lower().split('_')
        
        for i, part in enumerate(parts):
            if part in {'person', 'speaker', 'actor', 'subject', 'id'}:
                if i + 1 < len(parts):
                    # Extract the ID number/name
                    return f"{part}_{parts[i+1]}"
        
        # Fallback: use first part if it looks like an ID
        if parts and (parts[0].startswith('p') or parts[0].startswith('s') 
                     or parts[0].startswith('id')):
            return parts[0]
        
        return None
    
    def _compile_report(self, elapsed_seconds: float) -> Dict[str, Any]:
        """
        Compile audit results into a structured report.
        
        Args:
            elapsed_seconds: Time taken for audit
            
        Returns:
            Dictionary with complete audit report
        """
        # Assess risk level
        risk_level = self._assess_risk()
        
        # Count issues
        issue_counts = {
            'identity_overlap': sum(len(v) for v in self.findings['identity_overlap'].values()),
            'video_hash_collision': sum(len(v) for v in self.findings['video_hash_collision'].values()),
            'audio_hash_collision': sum(len(v) for v in self.findings['audio_hash_collision'].values()),
            'metadata_similarity': sum(len(v) for v in self.findings['metadata_similarity'].values()),
        }
        
        # Sample statistics
        sample_counts = {split: len(samples) 
                        for split, samples in self.samples_by_split.items()}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed_seconds, 2),
            'audit_config': {
                'splits': self.splits,
                'data_root': str(self.data_root),
            },
            'sample_statistics': sample_counts,
            'total_samples': sum(sample_counts.values()),
            'risk_assessment': {
                'level': risk_level,
                'issues_found': sum(issue_counts.values()),
                'issue_breakdown': issue_counts,
            },
            'findings': {
                'identity_overlap': dict(self.findings['identity_overlap']),
                'video_hash_collisions': dict(self.findings['video_hash_collision']),
                'audio_hash_collisions': dict(self.findings['audio_hash_collision']),
                'suspicious_metadata_clusters': dict(self.findings['metadata_similarity']),
                'errors': self.findings['errors'],
            },
            'recommendations': self._generate_recommendations(),
        }
        
        return report
    
    def _assess_risk(self) -> str:
        """
        Assess overall data leakage risk.
        
        Returns:
            Risk level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', or 'NONE'
        """
        issue_count = (
            sum(len(v) for v in self.findings['identity_overlap'].values()) +
            sum(len(v) for v in self.findings['video_hash_collision'].values()) +
            sum(len(v) for v in self.findings['audio_hash_collision'].values())
        )
        
        if issue_count >= 10:
            return 'CRITICAL'
        elif issue_count >= 5:
            return 'HIGH'
        elif issue_count >= 2:
            return 'MEDIUM'
        elif issue_count >= 1:
            return 'LOW'
        else:
            return 'NONE'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if self.findings['identity_overlap']:
            recommendations.append(
                "Identity overlap detected: Remove or reallocate samples to ensure "
                "no person appears in multiple splits"
            )
        
        if self.findings['video_hash_collision']:
            recommendations.append(
                "Identical videos found across splits: This may indicate data duplication "
                "or copy-paste errors during split creation"
            )
        
        if self.findings['audio_hash_collision']:
            recommendations.append(
                "Identical audio tracks found across splits: Extract audio separately "
                "or verify split assignment for these samples"
            )
        
        if self.findings['metadata_similarity']:
            recommendations.append(
                "Suspicious metadata clusters detected: Verify samples were independently "
                "collected and not batch-processed identically"
            )
        
        if not recommendations:
            recommendations.append("No data leakage detected. Dataset appears clean.")
        
        return recommendations


def main():
    """Command-line interface for dataset audit."""
    parser = argparse.ArgumentParser(
        description='Audit dataset for integrity and leakage risks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audit_dataset.py --config config/config.yaml
  python audit_dataset.py --config config/config.yaml --output reports/audit.json
  python audit_dataset.py --data-root data/deepfake --splits train val test
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Override data root from config'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to audit (default: train val test)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='audit_report.json',
        help='Output JSON report path (default: audit_report.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = None
    if HAS_CONFIG:
        try:
            config = get_config()
            data_root = args.data_root or config.dataset.data_root
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            data_root = args.data_root or 'data/deepfake'
    else:
        data_root = args.data_root or 'data/deepfake'
    
    # Run audit
    auditor = DatasetAuditor(
        data_root=data_root,
        config=config,
        splits=args.splits
    )
    report = auditor.audit()
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary
    risk = report['risk_assessment']['level']
    issues = report['risk_assessment']['issues_found']
    print(f"\n{'='*60}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Risk Level: {risk}")
    print(f"Issues Found: {issues}")
    print(f"Total Samples: {report['total_samples']}")
    print(f"Samples by Split:")
    for split, count in report['sample_statistics'].items():
        print(f"  {split:10s}: {count:5d}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    print(f"{'='*60}\n")
    
    # Return exit code based on risk
    if risk == 'CRITICAL':
        return 2
    elif risk in {'HIGH', 'MEDIUM'}:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())
