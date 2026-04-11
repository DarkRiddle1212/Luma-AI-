"""
StructuredLogger: JSON-formatted structured logging.

This module provides a structured logging system that wraps the Python logging
module to produce JSON-formatted log entries with a deterministic schema.
All log entries include event, timestamp, and payload fields with consistent
snake_case key naming.
"""

import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Structured logger that produces JSON-formatted log entries.
    
    This class wraps the Python logging module to ensure all log entries
    follow a consistent JSON schema with event, timestamp, and payload fields.
    
    Attributes:
        _logger: The underlying Python logger instance
    """
    
    def __init__(self, name: str = "structured_logger", level: int = logging.INFO):
        """
        Initialize the StructuredLogger.
        
        Args:
            name: The name of the logger (default: "structured_logger")
            level: The logging level (default: logging.INFO)
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # Only add handler if none exists to avoid duplicate logs
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            self._logger.addHandler(handler)
    
    def log(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a structured event with JSON formatting.
        
        Produces a JSON log entry with the following deterministic schema:
        {
            "event": <event_name>,
            "timestamp": <ISO8601_timestamp>,
            "payload": <payload_dict>
        }
        
        All keys use snake_case naming for consistency.
        
        Args:
            event: The event name/type being logged
            payload: Optional dictionary containing event-specific data
        """
        if payload is None:
            payload = {}
        
        log_entry = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            "payload": payload
        }
        
        # Use json.dumps to ensure proper JSON formatting
        json_output = json.dumps(log_entry, separators=(',', ':'))
        self._logger.info(json_output)
