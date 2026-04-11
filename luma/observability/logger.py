"""
Logger Facade

This module provides a simplified interface for common logging operations,
building on top of the core StructuredLogger implementation.

The facade provides:
- Re-export of StructuredLogger for direct use
- Convenience methods for common log events
- Factory functions for logger creation
- Simplified interfaces for standard logging patterns

All operations maintain the deterministic JSON schema with event, timestamp,
and payload fields using consistent snake_case key naming.
"""

from typing import Any, Dict, Optional
import logging

from luma.core.structured_logger import StructuredLogger


# Re-export StructuredLogger for convenience
__all__ = ["StructuredLogger", "create_logger", "log_event"]


def create_logger(name: str = "structured_logger", level: int = logging.INFO) -> StructuredLogger:
    """
    Factory function to create a new StructuredLogger instance.
    
    This provides a convenient way to instantiate structured loggers
    without directly importing from luma.core.
    
    Args:
        name: The name of the logger (default: "structured_logger")
        level: The logging level (default: logging.INFO)
    
    Returns:
        A new StructuredLogger instance
    """
    return StructuredLogger(name=name, level=level)


def log_event(
    logger: Optional[StructuredLogger],
    event: str,
    payload: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to log an event with optional logger.
    
    This function safely handles None loggers, making it easy to use
    optional logging in instrumented components. If logger is None,
    the function returns immediately without any overhead.
    
    Usage:
        log_event(logger, "operation_completed", {"count": 5})
        log_event(None, "operation_completed", {"count": 5})  # No-op
    
    Args:
        logger: Optional StructuredLogger instance
        event: The event name/type being logged
        payload: Optional dictionary containing event-specific data
    """
    if logger is not None:
        logger.log(event, payload)
