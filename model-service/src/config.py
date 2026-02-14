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


class VideoConfig(BaseModel):
    """Video modality configuration."""
    backbone: str = Field(default="efficientnet_b3", description="Video backbone model")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    embed_dim: int = Field(default=1536, description="Video embedding dimension")
    temporal_strategy: str = Field(default="avg_pool", description="Temporal aggregation strategy")


class AudioConfig(BaseModel):
    """Audio modality configuration."""
    encoder_type: str = Field(default="simple", description="Audio encoder type (simple/wav2vec2)")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    n_mels: int = Field(default=64, description="Number of mel bins")
    embed_dim: int = Field(default=256, description="Audio embedding dimension")


class FusionConfig(BaseModel):
    """Multimodal fusion configuration."""
    strategy: str = Field(default="concat", description="Fusion strategy (concat/attention/gated)")
    hidden_dim: int = Field(default=512, description="Fusion hidden dimension")
    dropout: float = Field(default=0.3, description="Dropout rate")
    modality_dropout_prob: float = Field(default=0.0, description="Probability of dropping a modality during training")
    temporal_consistency_loss: dict = Field(default_factory=dict, description="Temporal consistency loss config")


class TrainingConfig(BaseModel):
    """Training configuration."""
    seed: int = Field(default=42, description="Random seed")
    learning_rate: float = Field(default=1.0e-4, description="Learning rate")
    weight_decay: float = Field(default=1.0e-5, description="Weight decay")
    optimizer: str = Field(default="adamw", description="Optimizer name")
    scheduler: str = Field(default="cosine", description="Scheduler name")
    epochs: int = Field(default=30, description="Number of epochs")
    batch_size: int = Field(default=16, description="Batch size")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")
    checkpoint_interval: int = Field(default=1, description="Checkpoint save interval")
    loss_function: str = Field(default="crossentropy", description="Loss function name")


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
    
    # Multimodal specific
    enable_video: bool = Field(default=True, description="Enable video modality")
    enable_video_config: VideoConfig = Field(default_factory=VideoConfig, alias="video")
    
    enable_audio: bool = Field(default=True, description="Enable audio modality")
    # Audio config is at root level or here? 
    # Based on MultimodalModel, audio config is separate, but fusion is inside model. 
    # Let's keep structure consistent with usage or update usage. 
    # MultimodalModel expects: video_cfg = model_cfg.get('video'), fusion_cfg = model_cfg.get('fusion').
    
    fusion: FusionConfig = Field(default_factory=FusionConfig)

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
    
    model_config = ConfigDict(populate_by_name=True)


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
    
    # New sections
    audio: AudioConfig = Field(default_factory=AudioConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# Global config instance (singleton)
_config: Optional[Config] = None
_logger = logging.getLogger(__name__)


def _find_config_file(custom_path: Optional[str] = None) -> Path:
    """Find config.yaml file, searching common locations or using custom path."""
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            return path
        _logger.warning(f"Custom config path {custom_path} not found, falling back to defaults")

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


def _load_yaml_config(custom_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    config_path = _find_config_file(custom_path)

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
        
    # Audio config overrides
    if os.getenv("MODEL_AUDIO_ENCODER_TYPE"):
        env_config.setdefault("audio", {})["encoder_type"] = os.getenv("MODEL_AUDIO_ENCODER_TYPE")

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


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load and validate configuration.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValueError: If configuration is invalid.
    """
    yaml_config = _load_yaml_config(config_path)
    merged_config = _merge_env_config(yaml_config)

    try:
        config = Config(**merged_config)
        _logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        _logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration object (singleton).
    
    Args:
        config_path: Optional path to config file (only used on first load or force reload)

    Returns:
        Config: Application configuration.
    """
    global _config
    if _config is None or config_path is not None:
        _config = load_config(config_path)
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
