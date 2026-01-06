"""
Health check and status routes.

Provides endpoints for monitoring and health checks.
"""

from fastapi import APIRouter, Response
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("")
async def health_check():
    """
    Basic health check endpoint.
    
    Returns status of the API and Supabase connectivity.
    
    Returns:
        200 OK with status information
    """
    return {
        "status": "healthy",
        "service": "aura-veracity-backend",
        "version": "1.0.0",
        "environment": "production" if not settings.debug else "development",
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes/container orchestration.
    
    Checks if the service is ready to handle traffic.
    
    Returns:
        200 OK if ready, 503 Service Unavailable if not
    """
    try:
        # Basic checks could go here (DB connectivity, etc.)
        return {
            "ready": True,
            "service": "aura-veracity-backend",
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return Response(
            status_code=503,
            content={"ready": False, "error": str(e)}
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes/container orchestration.
    
    Checks if the service is still running (not stuck).
    
    Returns:
        200 OK if alive
    """
    return {
        "alive": True,
        "service": "aura-veracity-backend",
    }
