"""
Unit tests for dataset audit module.

Tests overlap detection, hash collision detection, and report generation.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from audit_dataset import DatasetAuditor


@pytest.fixture
def temp_dataset():
    """Create temporary dataset with structured splits for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = tmpdir / split
            split_dir.mkdir()
            
            # Create sample directories with videos and metadata
            for i in range(5):
                sample_dir = split_dir / f"{split}_sample_{i:03d}"
                sample_dir.mkdir()
                
                # Create dummy video file
                video_file = sample_dir / "video.mp4"
                video_file.write_bytes(b"dummy_video_" + str(i).encode() + b"_" + split.encode())
                
                # Create metadata JSON
                meta = {
                    'label': i % 2,  # Alternate fake/real
                    'duration': 10.0 + i,
                    'source': f"{split}_source",
                }
                
                with open(sample_dir / "meta.json", 'w') as f:
                    json.dump(meta, f)
        
        yield tmpdir


@pytest.fixture
def dataset_with_overlap():
    """Create dataset with intentional identity overlap for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create train split
        train_dir = tmpdir / 'train'
        train_dir.mkdir()
        for i in range(3):
            sample_dir = train_dir / f"person_001_video_{i}"
            sample_dir.mkdir()
            video_file = sample_dir / "video.mp4"
            video_file.write_bytes(b"train_person_001_" + str(i).encode())
            meta = {'identity': 'person_001', 'label': 0}
            with open(sample_dir / "meta.json", 'w') as f:
                json.dump(meta, f)
        
        # Create test split with SAME identity
        test_dir = tmpdir / 'test'
        test_dir.mkdir()
        for i in range(3):
            sample_dir = test_dir / f"person_001_video_{10+i}"
            sample_dir.mkdir()
            video_file = sample_dir / "video.mp4"
            video_file.write_bytes(b"test_person_001_" + str(i).encode())
            meta = {'identity': 'person_001', 'label': 1}  # Different label but SAME person
            with open(sample_dir / "meta.json", 'w') as f:
                json.dump(meta, f)
        
        yield tmpdir


@pytest.fixture
def dataset_with_hash_collision():
    """Create dataset with duplicate video across splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Identical video content
        identical_content = b"IDENTICAL_VIDEO_CONTENT_" * 100
        
        # Create train split
        train_dir = tmpdir / 'train'
        train_dir.mkdir()
        
        sample_dir = train_dir / "video_001"
        sample_dir.mkdir()
        video_file = sample_dir / "video.mp4"
        video_file.write_bytes(identical_content)
        with open(sample_dir / "meta.json", 'w') as f:
            json.dump({'label': 0}, f)
        
        # Create test split with IDENTICAL video
        test_dir = tmpdir / 'test'
        test_dir.mkdir()
        
        sample_dir = test_dir / "video_002"
        sample_dir.mkdir()
        video_file = sample_dir / "video.mp4"
        video_file.write_bytes(identical_content)  # Same content
        with open(sample_dir / "meta.json", 'w') as f:
            json.dump({'label': 1}, f)
        
        yield tmpdir


class TestDatasetAuditor:
    """Test suite for DatasetAuditor class."""
    
    def test_auditor_initialization(self, temp_dataset):
        """Test auditor can be initialized."""
        auditor = DatasetAuditor(
            data_root=temp_dataset,
            splits=['train', 'val', 'test']
        )
        assert auditor.data_root == temp_dataset
        assert auditor.splits == ['train', 'val', 'test']
    
    def test_load_samples(self, temp_dataset):
        """Test that auditor loads samples from splits."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        auditor._load_samples()
        
        # Should load 5 samples per split
        assert len(auditor.samples_by_split['train']) == 5
        assert len(auditor.samples_by_split['val']) == 5
        assert len(auditor.samples_by_split['test']) == 5
    
    def test_extract_sample_info(self, temp_dataset):
        """Test sample info extraction."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        auditor._load_samples()
        
        # Check first sample
        sample = auditor.samples_by_split['train'][0]
        
        assert 'sample_id' in sample
        assert 'path' in sample
        assert 'video_path' in sample
        assert 'video_hash' in sample
        assert 'json_metadata' in sample
        assert sample['json_metadata']['label'] in {0, 1}
    
    def test_compute_file_hash(self, temp_dataset):
        """Test file hashing is deterministic."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        
        video_file = temp_dataset / 'train' / 'train_sample_000' / 'video.mp4'
        
        hash1 = auditor._compute_file_hash(video_file)
        hash2 = auditor._compute_file_hash(video_file)
        
        # Hashes should be identical
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_full_audit_no_issues(self, temp_dataset):
        """Test full audit on clean dataset."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        report = auditor.audit()
        
        # Should have no critical issues
        assert 'risk_assessment' in report
        assert report['risk_assessment']['level'] in {'LOW', 'NONE'}
        assert report['total_samples'] == 15  # 5 per split
    
    def test_identity_overlap_detection(self, dataset_with_overlap):
        """Test detection of identity overlap across splits."""
        auditor = DatasetAuditor(
            data_root=dataset_with_overlap,
            splits=['train', 'test']
        )
        auditor._load_samples()
        auditor._extract_metadata()
        auditor._check_identity_overlap()
        
        # Should detect overlap
        assert len(auditor.findings['identity_overlap']) > 0
        
        # Check specific overlap
        for (split1, split2), identities in auditor.findings['identity_overlap'].items():
            assert 'person_001' in identities or any('person_001' in str(i) for i in identities)
    
    def test_video_hash_collision_detection(self, dataset_with_hash_collision):
        """Test detection of identical videos across splits."""
        auditor = DatasetAuditor(
            data_root=dataset_with_hash_collision,
            splits=['train', 'test']
        )
        auditor._load_samples()
        auditor._extract_metadata()
        auditor._check_video_hash_collisions()
        
        # Should detect collision
        assert len(auditor.findings['video_hash_collision']) > 0
    
    def test_report_structure(self, temp_dataset):
        """Test audit report has required structure."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        report = auditor.audit()
        
        # Check required top-level keys
        required_keys = {
            'timestamp', 'elapsed_seconds', 'audit_config',
            'sample_statistics', 'total_samples', 'risk_assessment',
            'findings', 'recommendations'
        }
        assert required_keys.issubset(report.keys())
        
        # Check risk assessment
        assert 'level' in report['risk_assessment']
        assert 'issues_found' in report['risk_assessment']
        assert 'issue_breakdown' in report['risk_assessment']
        
        # Check findings structure
        findings = report['findings']
        assert 'identity_overlap' in findings
        assert 'video_hash_collisions' in findings
        assert 'audio_hash_collisions' in findings
        assert 'suspicious_metadata_clusters' in findings
    
    def test_risk_assessment(self, temp_dataset):
        """Test risk assessment logic."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        
        # Test with no issues
        auditor.findings['identity_overlap'] = {}
        auditor.findings['video_hash_collision'] = {}
        auditor.findings['audio_hash_collision'] = {}
        
        risk = auditor._assess_risk()
        assert risk == 'NONE'
        
        # Test with some issues
        auditor.findings['identity_overlap'][('train', 'test')] = ['person_001']
        risk = auditor._assess_risk()
        assert risk == 'LOW'
        
        # Test with many issues
        for i in range(15):
            auditor.findings['video_hash_collision'][(f'split_{i}', f'split_{i+1}')] = [
                {'hash': f'hash_{i}', 'split1_samples': ['a'], 'split2_samples': ['b']}
            ]
        risk = auditor._assess_risk()
        assert risk == 'CRITICAL'
    
    def test_extract_identity_from_sample_id(self, temp_dataset):
        """Test identity extraction from sample IDs."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        
        # Test various patterns
        assert auditor._extract_identity('person_001_video_01') == 'person_001'
        assert auditor._extract_identity('speaker_05_clip_a') == 'speaker_05'
        assert auditor._extract_identity('actor_123') == 'actor_123'
        assert auditor._extract_identity('subject_99_var') == 'subject_99'
        
        # Test fallback
        identity = auditor._extract_identity('p001_something')
        assert identity is not None  # Should match prefix heuristic
    
    def test_error_handling(self, temp_dataset):
        """Test graceful error handling."""
        # Create auditor with non-existent directory
        auditor = DatasetAuditor(
            data_root=temp_dataset / 'nonexistent',
            splits=['train']
        )
        report = auditor.audit()
        
        # Should complete without crashing
        assert 'findings' in report
        assert isinstance(report['findings']['errors'], list)
    
    def test_recommendations_generation(self, temp_dataset):
        """Test recommendation generation."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        auditor._load_samples()
        
        # Test with no issues
        auditor.findings = {
            'identity_overlap': {},
            'video_hash_collision': {},
            'audio_hash_collision': {},
            'metadata_similarity': {},
            'errors': [],
        }
        recs = auditor._generate_recommendations()
        assert any('clean' in r.lower() for r in recs)
        
        # Test with identity overlap
        auditor.findings['identity_overlap'][('train', 'test')] = ['person_001']
        recs = auditor._generate_recommendations()
        assert any('identity' in r.lower() for r in recs)
    
    def test_metadata_extraction(self, temp_dataset):
        """Test metadata extraction from samples."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        auditor._load_samples()
        auditor._extract_metadata()
        
        # Check metadata was indexed
        assert len(auditor.metadata_by_split['train']) > 0
        assert len(auditor.hashes_by_split['video']) > 0
    
    def test_sample_limit_in_collision_report(self, dataset_with_hash_collision):
        """Test that collision reports limit sample output."""
        auditor = DatasetAuditor(
            data_root=dataset_with_hash_collision,
            splits=['train', 'test']
        )
        auditor._load_samples()
        auditor._extract_metadata()
        auditor._check_video_hash_collisions()
        
        # Check that sample lists are limited
        for collisions in auditor.findings['video_hash_collision'].values():
            for collision in collisions:
                assert len(collision['split1_samples']) <= 3
                assert len(collision['split2_samples']) <= 3


class TestAuditIntegration:
    """Integration tests for audit workflow."""
    
    def test_end_to_end_audit(self, temp_dataset):
        """Test complete audit workflow."""
        auditor = DatasetAuditor(
            data_root=temp_dataset,
            splits=['train', 'val', 'test']
        )
        
        report = auditor.audit()
        
        # Verify all components executed
        assert report['total_samples'] == 15
        assert all(split in report['sample_statistics'] for split in ['train', 'val', 'test'])
        assert len(report['recommendations']) > 0
        assert 'timestamp' in report
        assert report['elapsed_seconds'] > 0
    
    def test_report_serializable(self, temp_dataset):
        """Test that report is JSON serializable."""
        auditor = DatasetAuditor(data_root=temp_dataset)
        report = auditor.audit()
        
        # Should be JSON serializable without errors
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert parsed['total_samples'] == 15
    
    def test_partial_dataset(self, temp_dataset):
        """Test audit on dataset with only some splits."""
        auditor = DatasetAuditor(
            data_root=temp_dataset,
            splits=['train']  # Only train split
        )
        
        report = auditor.audit()
        
        assert report['total_samples'] == 5
        assert 'train' in report['sample_statistics']


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_split_directory(self):
        """Test handling of empty split directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / 'train').mkdir()
            
            auditor = DatasetAuditor(data_root=tmpdir, splits=['train'])
            report = auditor.audit()
            
            # Should handle gracefully
            assert report['total_samples'] == 0
    
    def test_missing_all_splits(self):
        """Test handling when all splits are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = DatasetAuditor(
                data_root=tmpdir,
                splits=['nonexistent1', 'nonexistent2']
            )
            
            report = auditor.audit()
            
            # Should complete without crashing
            assert 'findings' in report
    
    def test_sample_without_metadata(self):
        """Test handling of samples without metadata JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create sample without meta.json
            sample_dir = tmpdir / 'train' / 'sample_001'
            sample_dir.mkdir(parents=True)
            
            video_file = sample_dir / "video.mp4"
            video_file.write_bytes(b"dummy_video")
            
            auditor = DatasetAuditor(data_root=tmpdir, splits=['train'])
            auditor._load_samples()
            
            # Should load sample despite missing metadata
            assert len(auditor.samples_by_split['train']) > 0
            sample = auditor.samples_by_split['train'][0]
            assert 'json_metadata' not in sample


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
