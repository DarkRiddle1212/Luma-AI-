"""
Structured logging configuration for Luma Memory Module.

This module provides structured logging with JSON formatting, contextual information,
and proper log levels for production and development environments.
"""

import logging
import sys
import json
from datetime import datetime, UTC
from typing import Any, Dict, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in structured JSON format.
    
    Each log record is formatted as a JSON object with:
    - timestamp: ISO 8601 formatted timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name (module path)
    - message: Log message
    - context: Additional contextual information
    - exception: Exception details if present
    
    Example output:
    {
        "timestamp": "2024-02-14T10:30:45.123456",
        "level": "INFO",
        "logger": "luma_memory.memory_manager",
        "message": "Created MemoryEntry with id=abc-123",
        "context": {
            "operation": "create_memory",
            "entry_id": "abc-123",
            "elapsed_ms": 45.2
        }
    }
    """
    
    def __init__(self, include_context: bool = True):
        """
        Initialize the structured formatter.
        
        Args:
            include_context: Whether to include contextual information in logs
        """
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as structured JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON-formatted log string
        """
        # Build base log structure
        log_data = {
            "timestamp": datetime.now(UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add contextual information if available
        if self.include_context:
            context = {}
            
            # Add custom context from extra fields
            for key, value in record.__dict__.items():
                if key not in [
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'message', 'pathname', 'process', 'processName',
                    'relativeCreated', 'thread', 'threadName', 'exc_info',
                    'exc_text', 'stack_info', 'getMessage', 'taskName'
                ]:
                    # Only include serializable values
                    try:
                        json.dumps(value)
                        context[key] = value
                    except (TypeError, ValueError):
                        context[key] = str(value)
            
            if context:
                log_data["context"] = context
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add source location for debugging
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }
        
        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development and console output.
    
    Formats logs in a readable format with colors (if terminal supports it):
    2024-02-14 10:30:45 - luma_memory.memory_manager - INFO - Created MemoryEntry with id=abc-123
    """
    
    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the human-readable formatter.
        
        Args:
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record in human-readable format.
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted log string
        """
        if self.use_colors:
            # Add color to level name
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
        
        formatted = super().format(record)
        
        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


def setup_structured_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    include_context: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB default
    backup_count: int = 5
) -> None:
    """
    Configure structured logging for the application.
    
    Sets up logging with:
    - Structured JSON format or human-readable format
    - Console output to stdout
    - Optional file output with rotation
    - Configurable log level
    - Proper formatting for exceptions and stack traces
    - Context propagation for request tracing
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" for structured JSON, "human" for readable)
        log_file: Optional path to log file for persistent logging
        include_context: Whether to include contextual information in logs
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    
    Example:
        >>> setup_structured_logging(log_level="INFO", log_format="json")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Server started", extra={"port": 8000, "workers": 4})
    """
    # Clear any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Choose formatter based on format type
    if log_format.lower() == "json":
        formatter = StructuredFormatter(include_context=include_context)
    else:
        formatter = HumanReadableFormatter(use_colors=True)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create rotating file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler for automatic log rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        # Always use JSON format for file logs
        file_handler.setFormatter(StructuredFormatter(include_context=include_context))
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for third-party libraries to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Structured logging configured",
        extra={
            "log_level": log_level.upper(),
            "log_format": log_format,
            "log_file": log_file,
            "include_context": include_context,
            "max_bytes": max_bytes if log_file else None,
            "backup_count": backup_count if log_file else None
        }
    )


class LogContext:
    """
    Context manager for adding contextual information to log records.
    
    Allows adding request-specific or operation-specific context that will be
    included in all log messages within the context.
    
    Example:
        >>> with LogContext(operation="create_memory", entry_id="abc-123"):
        ...     logger.info("Processing entry")
        ...     # Log will include operation and entry_id in context
    """
    
    def __init__(self, **context: Any):
        """
        Initialize log context with key-value pairs.
        
        Args:
            **context: Contextual information to add to logs
        """
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter the context and set up log record factory."""
        old_factory = logging.getLogRecordFactory()
        self.old_factory = old_factory
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original log record factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


def get_logger(name: str, **default_context: Any) -> logging.Logger:
    """
    Get a logger with optional default context.
    
    Args:
        name: Logger name (typically __name__)
        **default_context: Default contextual information for all logs
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = get_logger(__name__, service="memory_module", version="1.0.0")
        >>> logger.info("Service started")
    """
    logger = logging.getLogger(name)
    
    # Add default context to logger if provided
    if default_context:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in default_context.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger
