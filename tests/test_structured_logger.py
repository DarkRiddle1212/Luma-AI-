"""
Unit tests for StructuredLogger.

Tests verify:
- JSON output format
- Presence of required fields (event, timestamp, payload)
- Key naming consistency (snake_case)
- Logging module integration (not print)
"""

import json
import logging
from io import StringIO
from datetime import datetime

import pytest

from luma.core.structured_logger import StructuredLogger


class TestStructuredLoggerJSONFormat:
    """Test JSON output format requirements."""
    
    def test_log_produces_valid_json(self):
        """Test that log output is valid JSON."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"key": "value"})
        
        output = stream.getvalue().strip()
        
        # Should be valid JSON
        log_data = json.loads(output)
        assert isinstance(log_data, dict)
    
    def test_json_uses_compact_format(self):
        """Test that JSON uses compact format (no spaces after separators)."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"key": "value"})
        
        output = stream.getvalue().strip()
        
        # Compact JSON should not have spaces after colons or commas
        assert '": "' not in output or '":"' in output
        assert ', "' not in output or ',"' in output


class TestStructuredLoggerRequiredFields:
    """Test presence of required fields in log output."""
    
    def test_log_contains_event_field(self):
        """Test that every log entry contains an 'event' field."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("user_login", {"user_id": "123"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert "event" in log_data
        assert log_data["event"] == "user_login"
    
    def test_log_contains_timestamp_field(self):
        """Test that every log entry contains a 'timestamp' field."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event")
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert "timestamp" in log_data
        assert isinstance(log_data["timestamp"], str)
    
    def test_log_contains_payload_field(self):
        """Test that every log entry contains a 'payload' field."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"data": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert "payload" in log_data
        assert log_data["payload"] == {"data": "value"}
    
    def test_log_with_empty_payload(self):
        """Test that payload defaults to empty dict when not provided."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event")
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert "payload" in log_data
        assert log_data["payload"] == {}
    
    def test_timestamp_format_is_iso8601(self):
        """Test that timestamp follows ISO8601 format with Z suffix."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event")
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        timestamp = log_data["timestamp"]
        
        # Should end with Z for UTC
        assert timestamp.endswith("Z")
        
        # Should be parseable as ISO8601 (remove Z for parsing)
        datetime.fromisoformat(timestamp[:-1])


class TestStructuredLoggerKeyNaming:
    """Test key naming consistency (snake_case)."""
    
    def test_top_level_keys_use_snake_case(self):
        """Test that top-level keys use snake_case."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"key": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        # All top-level keys should be snake_case
        for key in log_data.keys():
            assert key.islower() or "_" in key
            assert key == key.lower()
    
    def test_payload_preserves_user_key_naming(self):
        """Test that payload preserves user-provided key naming."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        # User can use any naming in payload
        logger.log("test_event", {"camelCase": "value", "PascalCase": "value2"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        # Payload should preserve user naming
        assert "camelCase" in log_data["payload"]
        assert "PascalCase" in log_data["payload"]


class TestStructuredLoggerLoggingIntegration:
    """Test integration with Python logging module."""
    
    def test_uses_logging_module_not_print(self):
        """Test that StructuredLogger uses logging module, not print."""
        logger = StructuredLogger(name="test_logger")
        
        # Should have a _logger attribute that is a Logger instance
        assert hasattr(logger, "_logger")
        assert isinstance(logger._logger, logging.Logger)
    
    def test_logger_has_correct_name(self):
        """Test that logger is created with the specified name."""
        logger = StructuredLogger(name="custom_logger")
        
        assert logger._logger.name == "custom_logger"
    
    def test_logger_respects_log_level(self):
        """Test that logger respects the configured log level."""
        logger = StructuredLogger(name="test_logger", level=logging.WARNING)
        
        assert logger._logger.level == logging.WARNING
    
    def test_logger_uses_info_level_for_log_method(self):
        """Test that log() method uses INFO level."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger", level=logging.INFO)
        logger._logger.handlers = [handler]
        
        logger.log("test_event")
        
        # Should produce output at INFO level
        output = stream.getvalue()
        assert len(output) > 0
    
    def test_logger_respects_level_filtering(self):
        """Test that logger filters messages below configured level."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        # Set level to WARNING (higher than INFO)
        logger = StructuredLogger(name="test_logger", level=logging.WARNING)
        logger._logger.handlers = [handler]
        handler.setLevel(logging.WARNING)
        
        logger.log("test_event")
        
        # Should not produce output since INFO < WARNING
        output = stream.getvalue()
        assert len(output) == 0
    
    def test_avoids_duplicate_handlers(self):
        """Test that creating multiple instances doesn't add duplicate handlers."""
        # Create first instance
        logger1 = StructuredLogger(name="shared_logger")
        initial_handler_count = len(logger1._logger.handlers)
        
        # Create second instance with same name
        logger2 = StructuredLogger(name="shared_logger")
        
        # Should not add duplicate handlers
        assert len(logger2._logger.handlers) == initial_handler_count


class TestStructuredLoggerPayloadHandling:
    """Test payload data handling."""
    
    def test_log_with_complex_payload(self):
        """Test logging with complex nested payload."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        complex_payload = {
            "user_id": "123",
            "metadata": {
                "ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0"
            },
            "tags": ["login", "success"]
        }
        
        logger.log("user_login", complex_payload)
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["payload"] == complex_payload
    
    def test_log_with_numeric_payload_values(self):
        """Test logging with numeric values in payload."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("metric_recorded", {"count": 42, "duration_ms": 123.45})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["payload"]["count"] == 42
        assert log_data["payload"]["duration_ms"] == 123.45
    
    def test_log_with_boolean_payload_values(self):
        """Test logging with boolean values in payload."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("operation_complete", {"success": True, "cached": False})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["payload"]["success"] is True
        assert log_data["payload"]["cached"] is False


class TestStructuredLoggerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_log_with_none_payload(self):
        """Test that None payload is converted to empty dict."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", None)
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["payload"] == {}
    
    def test_log_with_empty_event_name(self):
        """Test logging with empty event name."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("", {"data": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["event"] == ""
    
    def test_multiple_log_calls(self):
        """Test multiple sequential log calls."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("event1", {"id": 1})
        logger.log("event2", {"id": 2})
        logger.log("event3", {"id": 3})
        
        output = stream.getvalue().strip()
        lines = output.split("\n")
        
        assert len(lines) == 3
        
        # Verify each line is valid JSON with correct event
        log1 = json.loads(lines[0])
        log2 = json.loads(lines[1])
        log3 = json.loads(lines[2])
        
        assert log1["event"] == "event1"
        assert log2["event"] == "event2"
        assert log3["event"] == "event3"
