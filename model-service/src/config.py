"""
Configuration Management for Model Service

This module loads configuration from YAML file and environment variables.
It uses Pydantic for validation and provides a singleton get_config() function.

Configuration hierarchy (highest to lowest priority):
1. Environment variables (e.g., MODEL_SERVER_HOST, MODEL_MODEL_CHECKPOINT_PATH)
2. .env file (loaded via python-dotenv)
3. config/config.yaml file
4. Default values in class definitions

Usage:
    from src.config import get_config
    
    config = get_config()
    checkpoint_path = config.model.checkpoint_path
    server_host = config.server.host
    log_level = config.logging.level
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv


# Load .env file if it exists
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class ModelConfig(BaseModel):
    """Model configuration."""

    checkpoint_path: str = Field(
        default="checkpoints/debug.pth",
        description="Path to model checkpoint file",
    )
    image_size: int = Field(
        default=224,
        ge=32,
        le=1024,
        description="Input image size in pixels",
    )
    model_type: str = Field(
        default="efficientnet_b3",
        description="Model architecture type",
    )

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def resolve_checkpoint_path(cls, v):
        """Resolve checkpoint path relative to model-service root."""
        if isinstance(v, str):
            path = Path(v)
            if not path.is_absolute():
                # Make path relative to model-service root
                root = Path(__file__).parent.parent
                path = root / v
            return str(path)
        return v


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum upload file size in MB",
    )
    workers: int = Field(default=1, ge=1, description="Number of workers")
    reload: bool = Field(
        default=False,
        description="Auto-reload on code changes (dev only)",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    json_output: bool = Field(
        default=True,
        alias="json",
        description="Output logs as JSON",
    )
    log_file: str = Field(
        default="",
        description="Log file path (empty = stdout only)",
    )
    
    model_config = ConfigDict(populate_by_name=True)


class SecurityConfig(BaseModel):
    """Security configuration."""

    api_key: str = Field(
        default="change-in-production",
        description="API key for authentication",
    )
    require_api_key: bool = Field(
        default=True,
        description="Require API key validation",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def override_from_env(cls, v):
        """Override api_key from MODEL_API_KEY env var if set."""
        env_key = os.getenv("MODEL_API_KEY")
        return env_key if env_key else v


class InferenceConfig(BaseModel):
    """Inference configuration."""

    device: str = Field(
        default="auto",
        pattern="^(auto|cuda|cpu)$",
        description="Device to use: auto, cuda, or cpu",
    )
    cache_model: bool = Field(
        default=True,
        description="Cache model in memory",
    )
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Max concurrent inference requests",
    )


class Config(BaseModel):
    """Complete application configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


# Global config instance (singleton)
_config: Optional[Config] = None
_logger = logging.getLogger(__name__)


def _find_config_file() -> Path:
    """Find config.yaml file, searching common locations."""
    search_paths = [
        Path(__file__).parent.parent / "config" / "config.yaml",  # src/../config/config.yaml
        Path(__file__).parent.parent.parent / "config" / "config.yaml",  # root/config/config.yaml
        Path("config") / "config.yaml",  # ./config/config.yaml
    ]

    for path in search_paths:
        if path.exists():
            return path

    # No config file found, return default location (will be created if needed)
    return Path(__file__).parent.parent / "config" / "config.yaml"


def _load_yaml_config() -> dict:
    """Load configuration from YAML file."""
    config_path = _find_config_file()

    if not config_path.exists():
        _logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
            _logger.info(f"Loaded config from {config_path}")
            return config_data
    except Exception as e:
        _logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
        return {}


def _merge_env_config(yaml_config: dict) -> dict:
    """Merge environment variable overrides into YAML config."""
    env_config = {}

    # Model config overrides
    if os.getenv("MODEL_MODEL_CHECKPOINT_PATH"):
        env_config.setdefault("model", {})["checkpoint_path"] = os.getenv(
            "MODEL_MODEL_CHECKPOINT_PATH"
        )
    if os.getenv("MODEL_MODEL_IMAGE_SIZE"):
        env_config.setdefault("model", {})["image_size"] = int(
            os.getenv("MODEL_MODEL_IMAGE_SIZE")
        )

    # Server config overrides
    if os.getenv("MODEL_SERVER_HOST"):
        env_config.setdefault("server", {})["host"] = os.getenv("MODEL_SERVER_HOST")
    if os.getenv("MODEL_SERVER_PORT"):
        env_config.setdefault("server", {})["port"] = int(
            os.getenv("MODEL_SERVER_PORT")
        )
    if os.getenv("MODEL_SERVER_MAX_FILE_SIZE_MB"):
        env_config.setdefault("server", {})["max_file_size_mb"] = int(
            os.getenv("MODEL_SERVER_MAX_FILE_SIZE_MB")
        )

    # Logging config overrides
    if os.getenv("MODEL_LOGGING_LEVEL"):
        env_config.setdefault("logging", {})["level"] = os.getenv(
            "MODEL_LOGGING_LEVEL"
        )
    if os.getenv("MODEL_LOGGING_JSON"):
        env_config.setdefault("logging", {})["json"] = os.getenv(
            "MODEL_LOGGING_JSON"
        ).lower() in ("true", "1", "yes")

    # Inference config overrides
    if os.getenv("MODEL_INFERENCE_DEVICE"):
        env_config.setdefault("inference", {})["device"] = os.getenv(
            "MODEL_INFERENCE_DEVICE"
        )

    # Deep merge: env config overrides yaml config
    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(yaml_config, env_config)


def load_config() -> Config:
    """
    Load and validate configuration.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValueError: If configuration is invalid.
    """
    yaml_config = _load_yaml_config()
    merged_config = _merge_env_config(yaml_config)

    try:
        config = Config(**merged_config)
        _logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        _logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def get_config() -> Config:
    """
    Get the global configuration object (singleton).

    Returns:
        Config: Application configuration.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    """
    Reload configuration from file (useful for testing).

    Returns:
        Config: Reloaded configuration.
    """
    global _config
    _config = load_config()
    return _config


def print_config() -> str:
    """
    Print configuration as formatted JSON (for debugging).

    Returns:
        str: Formatted JSON configuration.
    """
    config = get_config()
    return json.dumps(config.model_dump(), indent=2)
