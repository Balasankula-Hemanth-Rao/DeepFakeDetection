"""
File upload routes.

Provides endpoints for:
- POST /uploads/signed-url: Generate signed URL for direct uploads to Supabase Storage
- Upload tracking and job creation
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.middleware.auth import verify_auth_token
from app.services.supabase_service import supabase_service
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/uploads",
    tags=["uploads"],
)


class SignedUrlRequest(BaseModel):
    """Request model for signed URL generation."""
    filename: str
    expires_in: int = 3600  # 1 hour default


class SignedUrlResponse(BaseModel):
    """Response model for signed URL."""
    signed_url: str
    bucket: str
    expires_in: int


class UploadInitRequest(BaseModel):
    """Request model to initiate an upload and create a detection job."""
    original_filename: str
    file_path: str  # Path in storage, e.g., "user123/video.mp4"


class DetectionJobResponse(BaseModel):
    """Response model for detection job creation."""
    job_id: str
    status: str
    original_filename: str
    upload_timestamp: str


@router.post("/signed-url", response_model=SignedUrlResponse)
async def generate_signed_url(
    request: SignedUrlRequest,
    user: Dict[str, Any] = Depends(verify_auth_token)
) -> Dict[str, Any]:
    """
    Generate a signed URL for uploading a file directly to Supabase Storage.
    
    The frontend can use this URL to upload large files without
    sending them through the backend, reducing bandwidth and improving performance.
    
    Args:
        request: Signed URL request with filename
        user: Authenticated user info from dependency
        
    Returns:
        Signed URL and metadata
        
    Raises:
        400 Bad Request: If filename is invalid
        401 Unauthorized: If token is invalid
        500 Internal Server Error: If signed URL generation fails
    """
    if not request.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    # Construct file path: user_id/timestamp/filename
    import time
    timestamp = int(time.time() * 1000)
    file_path = f"{user['id']}/{timestamp}/{request.filename}"
    
    logger.info(f"Generating signed URL for user {user['email']}: {file_path}")
    
    # Generate signed URL from Supabase
    signed_url = supabase_service.generate_signed_upload_url(
        user_id=user['id'],
        file_path=file_path,
        expires_in=request.expires_in
    )
    
    if not signed_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate signed URL"
        )
    
    return {
        "signed_url": signed_url,
        "bucket": supabase_service.storage_bucket,
        "expires_in": request.expires_in,
    }


@router.post("/init-job", response_model=DetectionJobResponse)
async def init_detection_job(
    request: UploadInitRequest,
    user: Dict[str, Any] = Depends(verify_auth_token)
) -> Dict[str, Any]:
    """
    Create a detection job record after file upload.
    
    This endpoint is called after the frontend has successfully uploaded
    a video to Supabase Storage using the signed URL. It creates a record
    in the detection_jobs table to track the analysis.
    
    Args:
        request: Upload init request with filename and file path
        user: Authenticated user info from dependency
        
    Returns:
        Created detection job info
        
    Raises:
        400 Bad Request: If required fields are missing
        401 Unauthorized: If token is invalid
        500 Internal Server Error: If job creation fails
    """
    if not request.original_filename or not request.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="original_filename and file_path are required"
        )
    
    logger.info(f"Creating detection job for user {user['email']}: {request.original_filename}")
    
    # Create detection job in database
    job = supabase_service.create_detection_job(
        user_id=user['id'],
        original_filename=request.original_filename,
        file_path=request.file_path
    )
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create detection job"
        )
    
    return {
        "job_id": job['id'],
        "status": job['status'],
        "original_filename": job['original_filename'],
        "upload_timestamp": job['upload_timestamp'],
    }
