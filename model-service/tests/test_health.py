"""
Health check endpoint tests for Kubernetes readiness and liveness probes.

Tests verify:
- /health/live always returns 200 (liveness probe)
- /health/ready returns 200 only when model is loaded (readiness probe)
- /health/ready returns 503 when model is not ready
- Startup event correctly sets READY flag
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Import after path is set
    from serve.api import app
    return TestClient(app)


def test_health_live_always_200(client):
    """Test that /health/live always returns 200 OK."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_health_live_returns_json(client):
    """Test that /health/live returns proper JSON format."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "alive"


def test_health_ready_503_when_not_ready(client, monkeypatch):
    """Test that /health/ready returns 503 when READY flag is False."""
    # Import the api module to set READY flag
    from serve import api
    
    # Set READY to False
    api.READY = False
    api._model = None
    
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_health_ready_200_when_ready(client, monkeypatch):
    """Test that /health/ready returns 200 only when READY flag is True and model is loaded."""
    from serve import api
    
    # Mock the model and set READY flag
    api.READY = True
    api._model = MagicMock()  # Mock model
    
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_loaded"] is True


def test_health_ready_includes_checkpoint_path(client, monkeypatch):
    """Test that /health/ready includes checkpoint path in response."""
    from serve import api
    
    api.READY = True
    api._model = MagicMock()
    
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert "checkpoint_path" in response.json()


def test_health_ready_returns_correct_format(client):
    """Test that /health/ready returns proper JSON structure."""
    from serve import api
    
    api.READY = False
    api._model = None
    
    response = client.get("/health/ready")
    data = response.json()
    
    # Check required fields
    assert "status" in data
    assert "model_loaded" in data
    assert "checkpoint_path" in data


def test_health_check_legacy_endpoint_alive(client):
    """Test that legacy /health endpoint returns healthy when READY is True."""
    from serve import api
    
    api.READY = True
    api._model = MagicMock()
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["status"] == "healthy"


def test_health_check_legacy_endpoint_unhealthy(client):
    """Test that legacy /health endpoint returns unhealthy when READY is False."""
    from serve import api
    
    api.READY = False
    api._model = None
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is False
    assert data["status"] == "unhealthy"


def test_ready_flag_without_model(client):
    """Test that /health/ready returns 503 even if READY is True but model is None."""
    from serve import api
    
    api.READY = True
    api._model = None  # Model is not loaded
    
    response = client.get("/health/ready")
    assert response.status_code == 503


def test_ready_flag_with_model(client):
    """Test that /health/ready returns 200 when both READY is True and model is loaded."""
    from serve import api
    
    api.READY = True
    api._model = MagicMock()  # Model is loaded
    
    response = client.get("/health/ready")
    assert response.status_code == 200


def test_startup_event_sets_ready_flag(monkeypatch, tmp_path):
    """Test that startup event sets READY flag to True on success."""
    from serve import api
    
    # Reset READY flag
    api.READY = False
    
    # Mock the get_model function to succeed
    mock_get_model = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(api, "get_model", mock_get_model)
    
    # Manually call startup event
    import asyncio
    asyncio.run(api.startup_event())
    
    # READY should be True after successful startup
    assert api.READY is True


def test_startup_event_sets_ready_false_on_error(monkeypatch):
    """Test that startup event sets READY flag to False on model load failure."""
    from serve import api
    
    # Reset READY flag
    api.READY = True
    
    # Mock the get_model function to fail
    def mock_get_model_fails():
        raise RuntimeError("Model loading failed")
    
    monkeypatch.setattr(api, "get_model", mock_get_model_fails)
    
    # Manually call startup event
    import asyncio
    asyncio.run(api.startup_event())
    
    # READY should be False after failed startup
    assert api.READY is False


def test_shutdown_event_sets_ready_false(monkeypatch):
    """Test that shutdown event sets READY flag to False."""
    from serve import api
    
    # Set READY to True
    api.READY = True
    
    # Manually call shutdown event
    import asyncio
    asyncio.run(api.shutdown_event())
    
    # READY should be False after shutdown
    assert api.READY is False


def test_health_live_no_dependencies(client):
    """Test that /health/live doesn't depend on model state."""
    from serve import api
    
    # Set model to None
    api.READY = False
    api._model = None
    
    # Should still return 200
    response = client.get("/health/live")
    assert response.status_code == 200


def test_health_ready_with_model_none_ready_true(client):
    """Test /health/ready returns 503 if model is None but READY is True."""
    from serve import api
    
    api.READY = True
    api._model = None
    
    response = client.get("/health/ready")
    # Should return 503 because model is None
    assert response.status_code == 503


def test_health_endpoints_consistency(client):
    """Test that health endpoints are consistent with each other."""
    from serve import api
    
    api.READY = True
    api._model = MagicMock()
    
    live_response = client.get("/health/live")
    ready_response = client.get("/health/ready")
    health_response = client.get("/health")
    
    # Live should always be 200
    assert live_response.status_code == 200
    
    # Ready should be 200 when READY and model are set
    assert ready_response.status_code == 200
    
    # Health should show ready=True
    assert health_response.json()["ready"] is True


@pytest.mark.integration
def test_kubernetes_probe_workflow(client):
    """Integration test: simulate Kubernetes probe workflow."""
    from serve import api
    
    # Before startup: ready should fail
    api.READY = False
    api._model = None
    ready_response = client.get("/health/ready")
    assert ready_response.status_code == 503
    
    # Liveness should pass at all times
    live_response = client.get("/health/live")
    assert live_response.status_code == 200
    
    # Simulate startup completion
    api.READY = True
    api._model = MagicMock()
    
    # After startup: ready should pass
    ready_response = client.get("/health/ready")
    assert ready_response.status_code == 200
    
    # Liveness should still pass
    live_response = client.get("/health/live")
    assert live_response.status_code == 200
    
    # Simulate shutdown
    api.READY = False
    
    # After shutdown: ready should fail again
    ready_response = client.get("/health/ready")
    assert ready_response.status_code == 503
    
    # Liveness should still pass (container is still alive)
    live_response = client.get("/health/live")
    assert live_response.status_code == 200
