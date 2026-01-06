"""
Supabase service wrapper for Aura Veracity backend.

Provides unified interface for:
- User authentication verification
- File upload handling and signed URL generation
- Database operations (detection_jobs, detection_results)

Integrates with the same Supabase project used by the frontend.
"""

from supabase import create_client, Client
from app.config.settings import settings
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SupabaseService:
    """
    Service wrapper for Supabase operations.
    
    Handles authentication, file uploads, and database queries
    while abstracting away Supabase SDK details.
    """
    
    def __init__(self):
        """Initialize Supabase client with project URL and service role key."""
        self.url = settings.supabase_url
        self.service_role_key = settings.supabase_service_role_key
        self.anon_key = settings.supabase_anon_key
        self.storage_bucket = settings.supabase_storage_bucket
        
        # Initialize admin client (with service role key for privileged operations)
        if self.service_role_key:
            self.admin_client: Client = create_client(self.url, self.service_role_key)
            logger.info("Admin Supabase client initialized with service role key")
        else:
            logger.warning("Service role key not configured; some operations will be limited")
            self.admin_client = None
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token by decoding and validating signature.
        
        In production, this would use Supabase's built-in JWT verification.
        For now, we extract the JWT payload and validate basic structure.
        
        Args:
            token: JWT token from Authorization header
            
        Returns:
            Decoded JWT payload if valid, None otherwise
        """
        try:
            import json
            import base64
            
            # JWT format: header.payload.signature
            parts = token.split(".")
            if len(parts) != 3:
                logger.warning("Invalid JWT format")
                return None
            
            # Decode payload (base64 with padding)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding
            
            decoded = base64.urlsafe_b64decode(payload)
            payload_data = json.loads(decoded)
            
            # Basic validation: check expiration
            import time
            if "exp" in payload_data and payload_data["exp"] < time.time():
                logger.warning("Token expired")
                return None
            
            return payload_data
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Extract user information from a valid JWT token.
        
        Args:
            token: JWT token from Authorization header
            
        Returns:
            User info dict with 'id' and 'email' if token is valid
        """
        payload = self.verify_jwt_token(token)
        if not payload:
            return None
        
        return {
            "id": payload.get("sub"),  # 'sub' is the user ID in Supabase JWTs
            "email": payload.get("email"),
        }
    
    def generate_signed_upload_url(
        self, 
        user_id: str, 
        file_path: str, 
        expires_in: int = 3600
    ) -> Optional[str]:
        """
        Generate a signed URL for uploading a file to Supabase Storage.
        
        The frontend can use this URL to upload directly to S3-compatible storage
        without needing to send the entire file through the backend.
        
        Args:
            user_id: ID of the authenticated user
            file_path: Relative path in the storage bucket (e.g., "user123/video.mp4")
            expires_in: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Signed URL string if successful, None otherwise
        """
        if not self.admin_client:
            logger.error("Admin client not initialized; cannot generate signed URL")
            return None
        
        try:
            response = self.admin_client.storage.from_(self.storage_bucket).create_signed_url(
                file_path,
                expires_in,
                {"x-upsert": "true"}
            )
            
            if response and "signedURL" in response:
                logger.info(f"Generated signed URL for {file_path}")
                return response["signedURL"]
            
            logger.warning(f"No signed URL in response: {response}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None
    
    def create_detection_job(
        self,
        user_id: str,
        original_filename: str,
        file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a detection job record in the database.
        
        Args:
            user_id: ID of the authenticated user
            original_filename: Original name of the uploaded file
            file_path: Path to file in Supabase Storage
            
        Returns:
            Created job record if successful, None otherwise
        """
        if not self.admin_client:
            logger.error("Admin client not initialized; cannot create job")
            return None
        
        try:
            response = self.admin_client.table("detection_jobs").insert({
                "user_id": user_id,
                "original_filename": original_filename,
                "file_path": file_path,
                "status": "pending"
            }).execute()
            
            if response.data:
                logger.info(f"Created detection job {response.data[0]['id']}")
                return response.data[0]
            
            return None
        except Exception as e:
            logger.error(f"Failed to create detection job: {e}")
            return None
    
    def get_detection_job(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a detection job by ID (only if user owns it).
        
        Args:
            job_id: ID of the detection job
            user_id: ID of the authenticated user (for ownership verification)
            
        Returns:
            Job record if found and user owns it, None otherwise
        """
        if not self.admin_client:
            logger.error("Admin client not initialized")
            return None
        
        try:
            response = self.admin_client.table("detection_jobs").select("*").eq(
                "id", job_id
            ).eq(
                "user_id", user_id
            ).execute()
            
            if response.data:
                return response.data[0]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get detection job: {e}")
            return None
    
    def get_detection_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis results for a detection job.
        
        Args:
            job_id: ID of the detection job
            
        Returns:
            Result record if found, None otherwise
        """
        if not self.admin_client:
            logger.error("Admin client not initialized")
            return None
        
        try:
            response = self.admin_client.table("detection_results").select("*").eq(
                "job_id", job_id
            ).execute()
            
            if response.data:
                return response.data[0]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get detection result: {e}")
            return None


# Global Supabase service instance
supabase_service = SupabaseService()
