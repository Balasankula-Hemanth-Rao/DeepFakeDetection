"""
Authentication middleware for Aura Veracity backend.

Provides FastAPI dependency for verifying JWT tokens from the Authorization header
and extracting authenticated user information.
"""

from fastapi import HTTPException, Depends, status, Header
from app.services.supabase_service import supabase_service
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


async def verify_auth_token(
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Dependency to verify and extract user from JWT token.
    
    Expects Authorization header in format: "Bearer <jwt_token>"
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User info dict with 'id' and 'email'
        
    Raises:
        HTTPException: 401 Unauthorized if token is invalid or missing
    """
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Missing or invalid Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    # Verify the token
    user = supabase_service.get_user_from_token(token)
    
    if not user or not user.get("id"):
        logger.warning("Invalid or expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Authenticated user: {user.get('email')}")
    return user


async def optional_auth_token(
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any] | None:
    """
    Optional authentication dependency.
    
    Returns user info if token is valid, None otherwise.
    Does not raise exception on invalid token.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        User info dict if token is valid, None otherwise
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization[7:]  # Remove "Bearer " prefix
    user = supabase_service.get_user_from_token(token)
    
    return user if user and user.get("id") else None
