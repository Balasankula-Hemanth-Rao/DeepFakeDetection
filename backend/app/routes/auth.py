"""
Authentication routes.

Provides endpoints for:
- GET /auth/me: Get current authenticated user info
- Token verification and validation
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.auth import verify_auth_token
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)


@router.get("/me")
async def get_current_user(
    user: Dict[str, Any] = Depends(verify_auth_token)
) -> Dict[str, Any]:
    """
    Get current authenticated user information.
    
    Requires valid JWT token in Authorization header.
    
    Args:
        user: Authenticated user info from dependency
        
    Returns:
        User information including id and email
        
    Raises:
        401 Unauthorized: If token is missing or invalid
    """
    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "authenticated": True,
    }
