"""
Configuration module for Aura Veracity backend.

Loads and validates environment variables for Supabase integration,
server settings, and other runtime configuration.

Extracted from frontend at: src/integrations/supabase/client.ts
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables:
    - SUPABASE_URL: Supabase project URL (required)
    - SUPABASE_ANON_KEY: Supabase public anon key (required)
    - SUPABASE_SERVICE_ROLE_KEY: Supabase service role key (required for admin operations)
    - SUPABASE_STORAGE_BUCKET: Storage bucket name (default: "videos")
    - JWT_SECRET: JWT secret for token verification (optional, falls back to anon key)
    - DEBUG: Enable debug mode (default: False)
    - ALLOW_ORIGINS: CORS allowed origins (comma-separated, default: "*")
    """
    
    # Supabase Configuration (extracted from frontend)
    supabase_url: str = os.getenv("SUPABASE_URL", "https://ppwatjhahicuwnvlpzqf.supabase.co")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", 
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBwd2F0amhhaGljdXdudmxwenFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NzAxMzEsImV4cCI6MjA2OTQ0NjEzMX0.9TnpvZQGZwYuA-yiLxqi29XgqFmaehgqEh4udTTMnUo")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Storage Configuration
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "videos")
    
    # Server Configuration
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # CORS Configuration
    allow_origins: str = os.getenv("ALLOW_ORIGINS", "*")
    
    # JWT Configuration (for token verification)
    jwt_secret: Optional[str] = os.getenv("JWT_SECRET", None)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars from .env file

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.allow_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allow_origins.split(",")]


# Global settings instance
settings = Settings()
