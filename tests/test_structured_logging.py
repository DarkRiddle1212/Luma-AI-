"""
Tests for structured logging configuration.

This module tests the structured logging functionality including:
- JSON and human-readable formatters
- Log context management
- Configuration validation
- Log output format
"""

import pytest
import logging
import json
import sys
from io import StringIO
from pathlib import Path
import tempfile

from luma_memory.utils.logging_config import (
    StructuredFormatter,
    HumanReadableFormatter,
    setup_structured_logging,
    LogContext,
    get_logger
)


class TestStructuredFormatter:
    """Test the StructuredFormatter class."""
    
    def test_basic_log_formatting(self):
        """Test that basic log messages are formatted as JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data
        assert log_data["source"]["line"] == 42
        assert log_data["source"]["function"] == "test_function"
    
    def test_log_with_context(self):
        """Test that contextual information is included in logs."""
        formatter = StructuredFormatter(include_context=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        
        # Add custom context
        record.operation = "create_memory"
        record.entry_id = "abc-123"
        record.elapsed_ms = 45.2
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "context" in log_data
        assert log_data["context"]["operation"] == "create_memory"
        assert log_data["context"]["entry_id"] == "abc-123"
        assert log_data["context"]["elapsed_ms"] == 45.2
    
    def test_log_without_context(self):
        """Test that context can be disabled."""
        formatter = StructuredFormatter(include_context=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        
        # Add custom context
        record.operation = "create_memory"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # Context should not be included
        assert "context" not in log_data or not log_data.get("context")
    
    def test_log_with_exception(self):
        """Test that exceptions are properly formatted."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
            func="test_function"
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
        assert "traceback" in log_data["exception"]


class TestHumanReadableFormatter:
    """Test the HumanReadableFormatter class."""
    
    def test_basic_formatting(self):
        """Test that logs are formatted in human-readable format."""
        formatter = HumanReadableFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_function"
        )
        
        formatted = formatter.format(record)
        
        assert "test.logger" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
    
    def test_exception_formatting(self):
        """Test that exceptions are included in output."""
        formatter = HumanReadableFormatter(use_colors=False)
        
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
            func="test_function"
        )
        
        formatted = formatter.format(record)
        
        assert "Error occurred" in formatted
        assert "ValueError" in formatted
        assert "Test error" in formatted


class TestSetupStructuredLogging:
    """Test the setup_structured_logging function."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_setup_with_json_format(self):
        """Test setting up logging with JSON format."""
        setup_structured_logging(log_level="INFO", log_format="json")
        
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        
        # Check that at least one handler has StructuredFormatter
        # (pytest may add its own handlers)
        has_structured_formatter = any(
            isinstance(h.formatter, StructuredFormatter)
            for h in root_logger.handlers
        )
        assert has_structured_formatter, "No StructuredFormatter found in handlers"
    
    def test_setup_with_human_format(self):
        """Test setting up logging with human-readable format."""
        setup_structured_logging(log_level="INFO", log_format="human")
        
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        
        # Check that at least one handler has HumanReadableFormatter
        # (pytest may add its own handlers)
        has_human_formatter = any(
            isinstance(h.formatter, HumanReadableFormatter)
            for h in root_logger.handlers
        )
        assert has_human_formatter, "No HumanReadableFormatter found in handlers"
    
    def test_setup_with_log_file(self):
        """Test setting up logging with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            setup_structured_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file)
            )
            
            # Log a message
            logger = logging.getLogger("test")
            logger.info("Test message")
            
            # Close all file handlers to release the file on Windows
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    root_logger.removeHandler(handler)
            
            # Check that log file was created and contains the message
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
            
            # Verify it's JSON format (check first line only)
            lines = content.strip().split('\n')
            log_data = json.loads(lines[-1])  # Get last line which should be our test message
            assert log_data["message"] == "Test message"
    
    def test_log_level_configuration(self):
        """Test that log level is properly configured."""
        setup_structured_logging(log_level="WARNING", log_format="json")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
    
    def test_third_party_logger_levels(self):
        """Test that third-party loggers have appropriate levels."""
        setup_structured_logging(log_level="DEBUG", log_format="json")
        
        # Check that third-party loggers are set to reduce noise
        uvicorn_logger = logging.getLogger("uvicorn.access")
        assert uvicorn_logger.level == logging.WARNING


class TestLogContext:
    """Test the LogContext context manager."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_context_adds_fields(self):
        """Test that context adds fields to log records."""
        setup_structured_logging(log_level="INFO", log_format="json")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        
        with LogContext(operation="test_op", request_id="req-123"):
            logger.info("Test message")
        
        # Parse log output
        output = stream.getvalue()
        log_data = json.loads(output.strip())
        
        assert log_data["context"]["operation"] == "test_op"
        assert log_data["context"]["request_id"] == "req-123"
    
    def test_context_is_removed_after_exit(self):
        """Test that context is removed after exiting the context manager."""
        setup_structured_logging(log_level="INFO", log_format="json")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        
        with LogContext(operation="test_op"):
            pass
        
        # Clear stream
        stream.truncate(0)
        stream.seek(0)
        
        # Log after context
        logger.info("Test message")
        
        # Parse log output
        output = stream.getvalue()
        log_data = json.loads(output.strip())
        
        # Context should not include operation
        assert "context" not in log_data or "operation" not in log_data.get("context", {})


class TestGetLogger:
    """Test the get_logger function."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_get_logger_with_default_context(self):
        """Test that get_logger adds default context to all logs."""
        setup_structured_logging(log_level="INFO", log_format="json")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = get_logger("test", service="memory_module", version="1.0.0")
        logger.addHandler(handler)
        logger.info("Test message")
        
        # Parse log output
        output = stream.getvalue()
        log_data = json.loads(output.strip())
        
        assert log_data["context"]["service"] == "memory_module"
        assert log_data["context"]["version"] == "1.0.0"


class TestIntegration:
    """Integration tests for structured logging."""
    
    def teardown_method(self):
        """Clean up logging handlers after each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_full_logging_workflow(self):
        """Test a complete logging workflow with context and exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            setup_structured_logging(
                log_level="INFO",
                log_format="json",
                log_file=str(log_file)
            )
            
            logger = logging.getLogger("test.workflow")
            
            # Log with context
            with LogContext(operation="create_memory", entry_id="abc-123"):
                logger.info("Starting operation")
                
                try:
                    raise ValueError("Test error")
                except ValueError:
                    logger.error("Operation failed", exc_info=True)
            
            # Close all file handlers to release the file on Windows
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    root_logger.removeHandler(handler)
            
            # Read and verify log file
            content = log_file.read_text()
            lines = content.strip().split('\n')
            
            # Should have at least 2 log entries (may have more from setup)
            assert len(lines) >= 2
            
            # Find our test logs (skip the setup log)
            test_logs = [line for line in lines if '"test.workflow"' in line]
            assert len(test_logs) == 2
            
            # Verify first log
            log1 = json.loads(test_logs[0])
            assert log1["message"] == "Starting operation"
            assert log1["context"]["operation"] == "create_memory"
            assert log1["context"]["entry_id"] == "abc-123"
            
            # Verify second log with exception
            log2 = json.loads(test_logs[1])
            assert log2["message"] == "Operation failed"
            assert "exception" in log2
            assert log2["exception"]["type"] == "ValueError"
