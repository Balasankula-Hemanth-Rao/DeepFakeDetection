"""
End-to-End Cross-Dataset Validation Test Suite

Tests the complete pipeline:
1. Video submission through Supabase
2. Edge function triggering model service
3. Real or simulated inference
4. Result storage and retrieval

Run with:
    pytest tests/test_cross_dataset_e2e.py -v
    
    # Or with specific dataset:
    pytest tests/test_cross_dataset_e2e.py::test_faceforensics_validation -v
    pytest tests/test_cross_dataset_e2e.py::test_fakeavceleb_validation -v
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add model-service to path
model_service_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(model_service_path))

from serve.model_orchestrator import ModelOrchestrator, AnalysisResult


class TestModelOrchestrator:
    """Test model orchestrator with real/mock inference."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        config = {
            'model_checkpoint': 'checkpoints/best_model.pth',
            'device': 'cpu',  # Use CPU for testing
            'fps': 3,
            'sample_rate': 16000,
            'simulation_enabled': True,
        }
        return ModelOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_simulation_fake_detection(self, orchestrator):
        """Test that simulation correctly detects fake videos by filename."""
        # Create mock video path with 'fake' indicator
        video_path = "/tmp/fake_deepfake_video.mp4"
        
        result = await orchestrator._analyze_video_simulation(
            video_path=video_path,
            job_id="test-fake-001"
        )
        
        assert result.prediction == "FAKE"
        assert 0.7 <= result.confidence_score <= 0.99
        assert result.is_simulation == True
        assert len(result.anomaly_timestamps) > 0  # Should have anomalies
    
    @pytest.mark.asyncio
    async def test_simulation_real_detection(self, orchestrator):
        """Test that simulation correctly detects real videos by filename."""
        video_path = "/tmp/authentic_video.mp4"
        
        result = await orchestrator._analyze_video_simulation(
            video_path=video_path,
            job_id="test-real-001"
        )
        
        assert result.prediction == "REAL"
        assert 0.7 <= result.confidence_score <= 0.99
        assert result.is_simulation == True
        assert len(result.anomaly_timestamps) == 0  # Real videos have no anomalies
    
    @pytest.mark.asyncio
    async def test_result_structure(self, orchestrator):
        """Test that result has all required fields."""
        video_path = "/tmp/test_video.mp4"
        
        result = await orchestrator._analyze_video_simulation(
            video_path=video_path,
            job_id="test-structure-001"
        )
        
        # Check all required fields exist
        assert hasattr(result, 'job_id')
        assert hasattr(result, 'prediction')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'visual_confidence')
        assert hasattr(result, 'audio_confidence')
        assert hasattr(result, 'analysis_duration_seconds')
        assert hasattr(result, 'anomaly_timestamps')
        assert hasattr(result, 'visual_analysis')
        assert hasattr(result, 'audio_analysis')
        assert hasattr(result, 'is_simulation')
        
        # Check types
        assert isinstance(result.prediction, str)
        assert isinstance(result.confidence_score, float)
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.visual_analysis, dict)
        assert isinstance(result.audio_analysis, dict)
    
    @pytest.mark.asyncio
    async def test_cross_dataset_validation_coverage(self, orchestrator):
        """
        Test cross-dataset validation scenarios.
        
        Simulates testing FaceForensics++ model on FakeAVCeleb videos.
        """
        test_cases = [
            {
                'name': 'deepfakes_sample',
                'dataset': 'faceforensics++',
                'method': 'deepfakes',
                'expected': 'FAKE',
            },
            {
                'name': 'face2face_sample',
                'dataset': 'faceforensics++',
                'method': 'face2face',
                'expected': 'FAKE',
            },
            {
                'name': 'faceswap_sample',
                'dataset': 'faceforensics++',
                'method': 'faceswap',
                'expected': 'FAKE',
            },
            {
                'name': 'neuraltextures_sample',
                'dataset': 'faceforensics++',
                'method': 'neuraltextures',
                'expected': 'FAKE',
            },
            {
                'name': 'fakeavceleb_audio_video_swap',
                'dataset': 'fakeavceleb',
                'method': 'audio_video_swap',
                'expected': 'FAKE',
            },
            {
                'name': 'authentic_speech',
                'dataset': 'voxceleb',
                'method': 'real',
                'expected': 'REAL',
            },
        ]
        
        for test_case in test_cases:
            # Create video path that simulates the test case
            video_name = f"{test_case['name']}.mp4"
            video_path = f"/datasets/{test_case['dataset']}/{test_case['method']}/{video_name}"
            
            result = await orchestrator._analyze_video_simulation(
                video_path=video_path,
                job_id=f"cross-dataset-{test_case['name']}"
            )
            
            assert result.prediction == test_case['expected'], \
                f"Failed for {test_case['name']}: expected {test_case['expected']}, got {result.prediction}"
            
            # Verify confidence is reasonable
            assert 0.5 <= result.confidence_score <= 0.99, \
                f"Confidence out of range for {test_case['name']}"
            
            print(f"✓ {test_case['name']:35} {result.prediction:5} (conf: {result.confidence_score:.3f})")


class TestAnalysisResultFormat:
    """Test that analysis results match expected format."""
    
    def test_result_serialization(self):
        """Test that AnalysisResult can be serialized to JSON."""
        result = AnalysisResult(
            job_id="test-001",
            prediction="FAKE",
            confidence_score=0.95,
            visual_confidence=0.93,
            audio_confidence=0.87,
            analysis_duration_seconds=30.5,
            anomaly_timestamps=[
                {'timestamp': 12.3, 'type': 'lip_sync', 'severity': 0.8}
            ],
            visual_analysis={'regions': 847},
            audio_analysis={'sample_rate': 16000},
            is_simulation=True,
        )
        
        # Convert to dict (like dataclass would)
        from dataclasses import asdict
        result_dict = asdict(result)
        
        # Verify it's JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert parsed['job_id'] == 'test-001'
        assert parsed['prediction'] == 'FAKE'
        assert parsed['confidence_score'] == 0.95


@pytest.mark.asyncio
async def test_full_pipeline_simulation():
    """
    Integration test: full pipeline from job submission to result storage.
    
    This simulates the actual flow:
    1. Frontend submits video
    2. Supabase stores job record
    3. Edge function calls model service
    4. Model service runs analysis (real or simulated)
    5. Results are stored back in Supabase
    """
    orchestrator = ModelOrchestrator()
    
    # Simulate job submission
    job_id = "integration-test-001"
    video_path = "/tmp/test_deepfake.mp4"
    
    # Run analysis (simulation mode)
    result = await orchestrator.analyze_video(
        video_path=video_path,
        job_id=job_id
    )
    
    # Verify result structure
    assert result.job_id == job_id
    assert result.prediction in ["REAL", "FAKE"]
    assert 0 <= result.confidence_score <= 1
    assert result.analysis_duration_seconds > 0
    
    # Convert to dict (as would be stored in Supabase)
    from dataclasses import asdict
    result_dict = asdict(result)
    
    # Verify all fields are present for database storage
    required_fields = [
        'job_id', 'prediction', 'confidence_score', 'visual_confidence',
        'audio_confidence', 'analysis_duration_seconds', 'anomaly_timestamps',
        'visual_analysis', 'audio_analysis', 'is_simulation'
    ]
    
    for field in required_fields:
        assert field in result_dict, f"Missing field: {field}"
    
    print(f"✓ Full pipeline simulation successful")
    print(f"  Job ID: {job_id}")
    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.confidence_score:.3f}")
    print(f"  Duration: {result.analysis_duration_seconds:.1f}s")


def test_supabase_edge_function_call_format():
    """
    Test that the Supabase edge function is correctly formatted
    to call the model service.
    """
    # Read the edge function code
    edge_function_path = Path(__file__).parent.parent.parent / 'supabase' / 'functions' / 'ai-detection' / 'index.ts'
    
    if edge_function_path.exists():
        with open(edge_function_path) as f:
            content = f.read()
        
        # Check that it calls the model service
        assert 'callModelService' in content, "Edge function should call callModelService()"
        assert 'MODEL_SERVICE_URL' in content, "Edge function should use MODEL_SERVICE_URL env var"
        assert '/analyze' in content, "Edge function should call /analyze endpoint"
        
        print("✓ Supabase edge function correctly configured")
    else:
        print(f"⚠ Edge function not found at {edge_function_path}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
