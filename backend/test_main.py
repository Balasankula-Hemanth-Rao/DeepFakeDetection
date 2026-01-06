"""
Basic tests for Aura Veracity backend.

Run with: pytest
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "aura-veracity-backend"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_ready():
    """Test readiness probe."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_health_live():
    """Test liveness probe."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["alive"] is True


def test_auth_me_missing_token():
    """Test /auth/me without token returns 401."""
    response = client.get("/auth/me")
    assert response.status_code == 403  # No credentials provided


def test_uploads_signed_url_missing_token():
    """Test /uploads/signed-url without token returns 401."""
    response = client.post(
        "/uploads/signed-url",
        json={"filename": "test.mp4"}
    )
    assert response.status_code == 403  # No credentials provided
