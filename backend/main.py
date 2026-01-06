"""
ASGI entry point for Aura Veracity backend.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
    
Or with auto-reload during development:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    
    # Import settings to get configuration
    from app.config.settings import settings
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
