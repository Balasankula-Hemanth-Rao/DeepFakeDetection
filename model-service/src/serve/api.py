"""
FastAPI Model Serving Application

Serves the FrameModel for inference via REST API.

This module provides a simple FastAPI application that:
- Accepts image files via multipart/form-data
- Validates API key from headers
- Runs inference using FrameModel
- Returns binary classification probabilities (real/fake)

Setup:
    1. Create .env file with MODEL_API_KEY=your-secret-key
    2. Ensure checkpoints/debug.pth exists in model-service/ directory
    3. Install dependencies: pip install -r requirements.txt

Run:
    uvicorn model_service.src.serve.api:app --host 0.0.0.0 --port 8000

    Or from model-service directory:
    uvicorn src.serve.api:app --host 0.0.0.0 --port 8001 --reload

Access:
    - API: http://localhost:8000/infer (POST)
    - Docs: http://localhost:8000/docs (interactive Swagger UI)
    - ReDoc: http://localhost:8000/redoc (alternative docs)
"""

import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from config import get_config
from logging_config import setup_logging, get_logger
from models.frame_model import FrameModel


# Configure application
_config = get_config()

# Set up structured logging at startup
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Frame Classification API",
    description="Binary image classification (real vs. fake) using FrameModel",
    version="1.0.0",
)

# Load configuration values
MAX_FILE_SIZE = _config.server.max_file_size_mb * 1024 * 1024
MODEL_API_KEY = _config.security.api_key
CHECKPOINT_PATH = Path(_config.model.checkpoint_path)

# Global model cache and readiness flag
_model = None
_device = None
READY = False  # Set to True only after successful model initialization


def get_device():
    """Get or initialize device (CUDA/CPU)."""
    global _device
    if _device is None:
        device_config = _config.inference.device
        if device_config == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device_config)
        logger.debug(f"Device resolved", device=str(_device), cuda_available=torch.cuda.is_available())
    return _device


def get_model():
    """Get or initialize model (lazy loading)."""
    global _model
    if _model is None:
        logger.info("Initializing model", model_type=_config.model.model_type)
        device = get_device()

        try:
            _model = FrameModel()
            _model.to(device)

            # Load checkpoint if it exists
            if CHECKPOINT_PATH.exists():
                _model.load_for_inference(str(CHECKPOINT_PATH), strict=True)
                logger.info(
                    "Model checkpoint loaded",
                    checkpoint_path=str(CHECKPOINT_PATH),
                    file_size_mb=round(CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 2),
                )
            else:
                logger.warning(
                    "Checkpoint not found, using untrained model",
                    checkpoint_path=str(CHECKPOINT_PATH),
                )

            _model.to_eval_mode()
            logger.info("Model ready for inference", device=str(device))

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}", exc_info=True)
            raise

    return _model


def get_transforms():
    """Get image preprocessing transforms."""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup and set readiness flag."""
    global READY
    logger.info("Server startup sequence initiated")
    try:
        get_model()
        READY = True
        logger.info("Model loaded successfully on startup", component="api")
    except Exception as e:
        READY = False
        logger.error(f"Failed to load model on startup: {e}", component="api")
        # Don't raise - allow server to start even if model loading fails
        # Inference endpoint and readiness probe will handle the error


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown event."""
    global READY
    READY = False
    logger.info("Server shutdown sequence initiated", component="api")


@app.get("/health/live")
async def health_live():
    """
    Liveness probe endpoint for Kubernetes.
    
    Returns 200 OK if the server process is alive.
    This is called by Kubernetes/Docker to verify the container is running.
    
    Returns:
        200 OK with alive status
    """
    health_logger = get_logger("health").bind(component="health")
    health_logger.debug("Liveness probe check")
    return JSONResponse(
        status_code=200,
        content={"status": "alive"},
    )


@app.get("/health/ready")
async def health_ready():
    """
    Readiness probe endpoint for Kubernetes.
    
    Returns 200 OK only if:
    - The model weights are successfully loaded
    - Logging is initialized
    
    Returns 503 Service Unavailable if the service is not ready to handle traffic.
    This is called by Kubernetes/load balancers to determine if traffic should be routed.
    
    Returns:
        200 OK with ready status if model is loaded
        503 Service Unavailable if model is not ready
    """
    global READY
    health_logger = get_logger("health").bind(component="health")
    
    if READY and _model is not None:
        health_logger.debug("Readiness probe check", status="ready")
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "model_loaded": True,
                "checkpoint_path": str(CHECKPOINT_PATH),
            },
        )
    else:
        health_logger.warning("Readiness probe check failed", status="not_ready", ready_flag=READY, model_loaded=_model is not None)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_loaded": _model is not None,
                "checkpoint_path": str(CHECKPOINT_PATH),
            },
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint (legacy, use /health/live and /health/ready instead).
    
    Returns comprehensive health information.
    
    Returns:
        200 OK with detailed health status
    """
    return {
        "status": "healthy" if READY else "unhealthy",
        "model_loaded": _model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "ready": READY,
    }


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
) -> JSONResponse:
    """
    Run inference on uploaded image.

    Args:
        file: Image file (JPEG, PNG, etc.)
        x_api_key: API key from X-API-KEY header

    Returns:
        JSON with fake_prob (probability of fake/synthetic class)

    Raises:
        401: Invalid or missing API key
        413: File too large (> 10 MB)
        400: Invalid image format
    """
    request_id = str(uuid.uuid4())[:8]
    inference_logger = get_logger("inference").bind(request_id=request_id, component="inference")
    start_time = time.time()
    
    inference_logger.info("Inference request received", filename=file.filename)

    try:
        # Validate API key
        if x_api_key != MODEL_API_KEY:
            inference_logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing X-API-KEY header",
            )

        # Check file size
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            inference_logger.warning(
                "File too large",
                size_bytes=file_size,
                max_bytes=MAX_FILE_SIZE,
            )
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB",
            )

        inference_logger.debug("File size valid", size_bytes=file_size, size_mb=round(file_size / (1024 * 1024), 2))

        # Load and validate image
        try:
            image = Image.open(BytesIO(file_content)).convert("RGB")
            image_width, image_height = image.size
            inference_logger.debug(
                "Image loaded",
                width=image_width,
                height=image_height,
            )
        except Exception as e:
            inference_logger.error(f"Failed to load image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {e}",
            )

        # Preprocess image
        transform = get_transforms()
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Move to device
        device = get_device()
        image_tensor = image_tensor.to(device)

        # Get model
        model = get_model()

        # Run inference
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            fake_prob = probs[0, 0].item()  # Class 0 = fake

        latency_ms = (time.time() - start_time) * 1000
        
        # Log inference event with structured metadata
        from logging_config import log_inference_event
        log_inference_event(
            logger=inference_logger,
            request_id=request_id,
            filename=file.filename or "unknown",
            size_bytes=file_size,
            model_version=_config.model.model_type,
            fake_prob=fake_prob,
            latency_ms=latency_ms,
            status="success",
        )

        return JSONResponse(
            {
                "request_id": request_id,
                "fake_prob": float(fake_prob),
                "real_prob": float(1.0 - fake_prob),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        inference_logger.error(f"Unexpected error during inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference",
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Frame Classification API",
        "version": "1.0.0",
        "description": "Binary image classification (real vs. fake)",
        "endpoints": {
            "infer": "POST /infer - Run inference on image file",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API docs (Swagger UI)",
            "redoc": "GET /redoc - Alternative API docs (ReDoc)",
        },
        "auth": "X-API-KEY header required",
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
    }


@app.get("/config")
async def get_config_endpoint():
    """Get server configuration (debug endpoint)."""
    return {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "model_loaded": _model is not None,
        "max_file_size_mb": _config.server.max_file_size_mb,
        "api_key_required": _config.security.require_api_key,
        "logging_level": _config.logging.level,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
    )
