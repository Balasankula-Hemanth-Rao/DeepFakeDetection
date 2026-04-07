"""
Real-Time Model Analysis Service

FastAPI service for video deepfake detection analysis.
Provides HTTP endpoints for job submission, status tracking, and result retrieval.

Designed as a lightweight service that can be called from Supabase Edge Functions
or standalone for cross-dataset validation.

Start with:
    uvicorn src.serve.service:app --host 0.0.0.0 --port 8001 --reload
"""

import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import aiofiles.os
from pathlib import Path

from .model_orchestrator import get_orchestrator, AnalysisResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection Analysis Service",
    description="Real-time video analysis for cross-dataset deepfake validation",
    version="1.0.0",
)

# Request/Response models
class AnalysisRequest(BaseModel):
    """Request to analyze video from file path or URL."""
    video_path: str
    job_id: str


class AnalysisResponse(BaseModel):
    """Response from video analysis."""
    job_id: str
    prediction: str
    confidence_score: float
    visual_confidence: float
    audio_confidence: float
    analysis_duration_seconds: float
    anomaly_timestamps: list
    visual_analysis: Dict[str, Any]
    audio_analysis: Dict[str, Any]
    is_simulation: bool


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str
    progress: float
    result: Optional[AnalysisResponse] = None


# Global job tracking (in production, use Redis or database)
_analysis_jobs: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 60)
    logger.info("Deepfake Detection Analysis Service Starting")
    logger.info("=" * 60)
    
    # Initialize orchestrator
    orchestrator = await get_orchestrator()
    logger.info(f"Model orchestrator initialized (simulation: {orchestrator.use_simulation})")
    logger.info("Ready to accept analysis requests")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'service': 'deepfake-analysis',
        'version': '1.0.0',
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Analyze video for deepfake detection.
    
    Can accept:
    - Local file paths: "/path/to/video.mp4"
    - HTTP URLs: "https://example.com/video.mp4"
    - Supabase signed URLs
    
    Args:
        request: AnalysisRequest with video_path and job_id
    
    Returns:
        AnalysisResponse with predictions and analysis data
    """
    logger.info(f"Received analysis request for job {request.job_id}: {request.video_path}")
    
    # Validate job_id
    if not request.job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    
    if request.job_id in _analysis_jobs:
        raise HTTPException(
            status_code=409,
            detail=f"Job {request.job_id} already exists"
        )
    
    # Validate video path
    if not request.video_path:
        raise HTTPException(status_code=400, detail="video_path is required")
    
    # Check if it's a URL or local path
    try:
        urlparse(request.video_path)
        is_url = request.video_path.startswith('http')
    except:
        is_url = False
    
    # For now, only support local paths
    # (URL support would require downloading the file first)
    if is_url:
        raise HTTPException(
            status_code=400,
            detail="URL-based videos not yet supported. Use local file paths."
        )
    
    # Verify file exists
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_path}")
    
    # Register job
    _analysis_jobs[request.job_id] = {
        'status': 'processing',
        'progress': 0.0,
        'result': None,
        'error': None,
    }
    
    # Run analysis in background
    background_tasks.add_task(
        _run_analysis_task,
        request.job_id,
        request.video_path
    )
    
    logger.info(f"Job {request.job_id} queued for processing")
    
    return {
        'job_id': request.job_id,
        'message': 'Analysis queued',
        'status_url': f'/jobs/{request.job_id}',
    }


@app.post("/analyze-upload", response_model=Dict[str, Any])
async def analyze_upload(
    file: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload and analyze a video file.
    
    Args:
        file: Video file upload
        job_id: Optional job identifier (generated if not provided)
        background_tasks: Background task runner
    
    Returns:
        Job info with job_id and status URL
    """
    import uuid
    
    # Generate job_id if not provided
    job_id = job_id or str(uuid.uuid4())
    
    logger.info(f"Received file upload for job {job_id}: {file.filename}")
    
    # Create temp directory for uploads
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    video_path = upload_dir / f"{job_id}_{file.filename}"
    
    try:
        # Save file
        contents = await file.read()
        async with aiofiles.open(video_path, 'wb') as f:
            await f.write(contents)
        
        logger.info(f"Saved upload to {video_path}")
        
        # Register job
        _analysis_jobs[job_id] = {
            'status': 'processing',
            'progress': 0.0,
            'result': None,
            'error': None,
        }
        
        # Run analysis in background
        if background_tasks:
            background_tasks.add_task(
                _run_analysis_task,
                job_id,
                str(video_path)
            )
        
        logger.info(f"Job {job_id} queued for processing")
        
        return {
            'job_id': job_id,
            'message': 'Video uploaded and queued for analysis',
            'status_url': f'/jobs/{job_id}',
        }
    
    except Exception as e:
        logger.error(f"File upload failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of analysis job.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Job status and results (if completed)
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _analysis_jobs[job_id]
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'result': job.get('result'),
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a job (if still processing).
    
    Args:
        job_id: Job identifier
    
    Returns:
        Cancellation confirmation
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = _analysis_jobs[job_id]
    
    if job['status'] in ['completed', 'failed']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status: {job['status']}"
        )
    
    job['status'] = 'cancelled'
    logger.info(f"Cancelled job {job_id}")
    
    return {'message': f'Job {job_id} cancelled'}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        'service': 'deepfake-analysis',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'analyze': 'POST /analyze',
            'analyze_upload': 'POST /analyze-upload',
            'job_status': 'GET /jobs/{job_id}',
            'cancel_job': 'DELETE /jobs/{job_id}',
            'docs': '/docs',
        },
    }


async def _run_analysis_task(job_id: str, video_path: str):
    """
    Background task to run video analysis.
    
    Args:
        job_id: Job identifier
        video_path: Path to video file
    """
    try:
        logger.info(f"Starting analysis task for job {job_id}")
        
        # Get orchestrator
        orchestrator = await get_orchestrator()
        
        # Run analysis
        result: AnalysisResult = await orchestrator.analyze_video(
            video_path=video_path,
            job_id=job_id,
        )
        
        # Store result
        _analysis_jobs[job_id]['status'] = 'completed'
        _analysis_jobs[job_id]['progress'] = 1.0
        _analysis_jobs[job_id]['result'] = {
            'job_id': result.job_id,
            'prediction': result.prediction,
            'confidence_score': result.confidence_score,
            'visual_confidence': result.visual_confidence,
            'audio_confidence': result.audio_confidence,
            'analysis_duration_seconds': result.analysis_duration_seconds,
            'anomaly_timestamps': result.anomaly_timestamps,
            'visual_analysis': result.visual_analysis,
            'audio_analysis': result.audio_analysis,
            'is_simulation': result.is_simulation,
        }
        
        logger.info(f"✓ Analysis complete for job {job_id}: {result.prediction}")
        
    except Exception as e:
        logger.error(f"✗ Analysis failed for job {job_id}: {e}", exc_info=True)
        _analysis_jobs[job_id]['status'] = 'failed'
        _analysis_jobs[job_id]['error'] = str(e)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 8001))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
