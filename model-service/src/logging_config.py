"""
Structured Logging Configuration using Loguru

This module provides structured logging with support for:
- JSON output format for log aggregation systems
- Rotating file sinks (10MB rotation, keep 3 files)
- Configurable log level via config.yaml
- Context binding (request_id, component, etc.)
- Inference event logging with metadata

The logging configuration is loaded from config/config.yaml:
    logging:
      level: "INFO"          # DEBUG, INFO, WARNING, ERROR, CRITICAL
      json: true            # If true, output JSON-friendly logs
      log_file: ""          # Empty string = stdout only, else relative path

Usage:
    from src.logging_config import setup_logging, get_logger, log_inference_event
    
    # At application startup
    setup_logging()
    
    # In your code
    logger = get_logger(__name__)
    logger.info("Application started")
    
    # With request context
    logger = get_logger("inference").bind(request_id="req-123", component="api")
    logger.info("Inference started", filename="image.jpg")
    
    # Log inference event
    log_inference_event(
        logger=logger,
        request_id="req-123",
        filename="image.jpg",
        size_bytes=1024000,
        model_version="v1.0",
        fake_prob=0.8234,
        latency_ms=234.5,
        status="success"
    )
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Any

from loguru import logger as loguru_logger


# Configuration singleton
_config = None
_setup_done = False


def _get_config():
    """Lazy load config to avoid circular imports."""
    global _config
    if _config is None:
        from src.config import get_config
        _config = get_config()
    return _config


def _get_log_dir() -> Path:
    """
    Get the log directory, creating it if needed.
    
    Priority:
    1. LOG_DIR environment variable
    2. Default: model-service/logs
    """
    log_dir_env = os.environ.get("LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env)
    else:
        # Default to model-service/logs
        # Current file is at src/logging_config.py, so parent.parent is model-service root
        root = Path(__file__).parent.parent
        log_dir = root / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _format_json_record(record: dict) -> str:
    """
    Custom formatter for JSON log output.
    
    Filters sensitive information and includes structured fields.
    """
    try:
        # Extract only safe fields for JSON output
        log_entry = {
            "timestamp": str(record.get("time", "")),
            "level": str(record.get("level", {}).get("name", "UNKNOWN")),
            "message": record.get("message", ""),
            "name": record.get("name", ""),
        }
        
        # Add extra fields if present (from logger.bind())
        if record.get("extra"):
            for key, value in record["extra"].items():
                # Skip sensitive fields
                if key not in ["file_content", "image_bytes", "raw_data", "password", "token", "secret"]:
                    log_entry[key] = value
        
        # Add exception info if present
        if record.get("exception"):
            exc_info = record["exception"]
            log_entry["exception"] = {
                "type": exc_info[0].__name__ if exc_info[0] else "Unknown",
                "message": str(exc_info[1]) if exc_info[1] else "",
            }
        
        return json.dumps(log_entry)
    except Exception:
        # Fallback if formatting fails
        return json.dumps({
            "level": "ERROR",
            "message": "Failed to format log record",
            "original_message": str(record.get("message", ""))
        })


def setup_logging() -> None:
    """
    Initialize structured logging based on configuration.
    
    This function:
    1. Removes default handlers
    2. Adds stdout sink
    3. Adds rotating file sink (if log_file configured)
    4. Sets log level from config
    5. Configures JSON format if enabled
    
    Safe to call multiple times (idempotent).
    """
    global _setup_done
    
    if _setup_done:
        return
    
    config = _get_config()
    
    # Remove default handler
    loguru_logger.remove()
    
    # Get log configuration
    log_level = config.logging.level
    use_json = config.logging.json_output
    log_file = config.logging.log_file
    
    # Choose formatter - use loguru's serialize option for JSON
    if use_json:
        # Use loguru's built-in JSON serialization
        formatter_str = "{message}"
    else:
        # Plain text format
        formatter_str = (
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - <level>{message}</level>"
        )
    
    # Add stdout sink
    loguru_logger.add(
        sys.stdout,
        level=log_level,
        format=formatter_str,
        colorize=(not use_json),  # Disable colors in JSON mode
        serialize=use_json,  # Enable JSON serialization
    )
    
    # Add file sink if configured
    if log_file and log_file.strip():
        log_dir = _get_log_dir()
        file_path = log_dir / log_file
        
        loguru_logger.add(
            str(file_path),
            level=log_level,
            format=formatter_str,
            rotation="10 MB",  # Rotate at 10MB
            retention=3,  # Keep 3 rotated files
            compression=None,  # No compression
            serialize=use_json,  # Enable JSON serialization
        )
    
    _setup_done = True
    loguru_logger.debug(f"Logging configured: level={log_level}, json={use_json}")


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get a logger instance with optional context binding.
    
    Args:
        name: Logger name (typically __name__), optional
    
    Returns:
        Loguru logger instance
    
    Usage:
        logger = get_logger(__name__)
        logger = get_logger("inference").bind(request_id="123")
    """
    # Ensure logging is set up
    if not _setup_done:
        setup_logging()
    
    if name:
        return loguru_logger.bind(module=name)
    return loguru_logger


def log_inference_event(
    logger: Any,
    request_id: str,
    filename: str,
    size_bytes: int,
    model_version: str,
    fake_prob: float,
    latency_ms: float,
    status: str,
) -> None:
    """
    Log an inference event with structured metadata.
    
    This logs inference events in a structured way without including
    raw image data or other sensitive information.
    
    Args:
        logger: Loguru logger instance (should have request_id bound)
        request_id: Unique request identifier
        filename: Original filename (not path)
        size_bytes: File size in bytes
        model_version: Model version identifier
        fake_prob: Probability of fake class (0-1)
        latency_ms: Inference latency in milliseconds
        status: Status ("success", "error", etc.)
    
    Example:
        logger = get_logger("inference").bind(request_id="req-123")
        log_inference_event(
            logger=logger,
            request_id="req-123",
            filename="image.jpg",
            size_bytes=1024000,
            model_version="efficientnet_b3_v1",
            fake_prob=0.8234,
            latency_ms=234.5,
            status="success"
        )
    """
    logger.info(
        "Inference completed",
        request_id=request_id,
        component="inference",
        filename=filename,
        size_mb=round(size_bytes / (1024 * 1024), 2),
        model_version=model_version,
        fake_probability=round(fake_prob, 4),
        real_probability=round(1.0 - fake_prob, 4),
        latency_ms=round(latency_ms, 2),
        status=status,
    )


# Compatibility layer: wrap loguru logger to work like standard logging
class LoggerAdapter:
    """Adapter to make loguru logger work like standard logging module."""
    
    def __init__(self, loguru_instance):
        self._logger = loguru_instance
    
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def bind(self, **kwargs):
        return LoggerAdapter(self._logger.bind(**kwargs))


# Setup compatibility: allow replacing standard logging with loguru
def setup_standard_logging_replacement():
    """
    Replace standard logging module with loguru.
    
    This allows existing code using logging.getLogger() to use loguru.
    Call this early in your application startup.
    """
    setup_logging()
    
    # Replace logging.getLogger with our function
    original_getLogger = logging.getLogger
    
    def patched_getLogger(name=None):
        return LoggerAdapter(get_logger(name))
    
    logging.getLogger = patched_getLogger
