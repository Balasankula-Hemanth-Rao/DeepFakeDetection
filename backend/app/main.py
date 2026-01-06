"""
Main FastAPI application factory and middleware setup.

Initializes the FastAPI app with CORS, logging, and route registration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.routes import health, auth, uploads
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Includes:
    - CORS middleware for cross-origin requests
    - Route registration
    - Exception handlers
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Aura Veracity Backend",
        description="Backend API for AI-powered deepfake detection",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Configure CORS
    cors_origins = settings.get_cors_origins()
    logger.info(f"Configuring CORS with origins: {cors_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    logger.info("Registering API routes")
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(uploads.router)
    
    @app.on_event("startup")
    async def startup_event():
        """Run on application startup."""
        logger.info("Aura Veracity backend starting up")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Supabase URL: {settings.supabase_url}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run on application shutdown."""
        logger.info("Aura Veracity backend shutting down")
    
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "service": "aura-veracity-backend",
            "version": "1.0.0",
            "docs": "/docs" if settings.debug else "Not available",
        }
    
    return app


# Create app instance
app = create_app()
