"""
Unit tests for the logger facade.

Tests verify:
- Factory function creates valid StructuredLogger instances
- log_event convenience function handles None logger gracefully
- log_event properly delegates to StructuredLogger
- Facade properly re-exports StructuredLogger
- Integration with main observability module
"""

import json
import logging
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from luma.observability.logger import (
    create_logger,
    log_event,
    StructuredLogger
)


class TestCreateLogger:
    """Test the create_logger factory function."""
    
    def test_create_logger_with_defaults(self):
        """Test that factory function creates a valid StructuredLogger with defaults."""
        logger = create_logger()
        
        assert isinstance(logger, StructuredLogger)
        assert logger is not None
        
        # Verify default parameters
        assert logger._logger.name == "structured_logger"
        assert logger._logger.level == logging.INFO
    
    def test_create_logger_with_custom_name(self):
        """Test that factory function accepts custom logger name."""
        logger = create_logger(name="custom_logger")
        
        assert isinstance(logger, StructuredLogger)
        assert logger._logger.name == "custom_logger"
    
    def test_create_logger_with_custom_level(self):
        """Test that factory function accepts custom logging level."""
        logger = create_logger(level=logging.DEBUG)
        
        assert isinstance(logger, StructuredLogger)
        assert logger._logger.level == logging.DEBUG
    
    def test_create_logger_with_both_parameters(self):
        """Test that factory function accepts both name and level."""
        logger = create_logger(name="test_logger", level=logging.WARNING)
        
        assert isinstance(logger, StructuredLogger)
        assert logger._logger.name == "test_logger"
        assert logger._logger.level == logging.WARNING
    
    def test_create_logger_creates_functional_logger(self):
        """Test that created logger is functional."""
        logger = create_logger()
        
        # Should be able to log without exceptions
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"key": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["event"] == "test_event"
        assert log_data["payload"] == {"key": "value"}


class TestLogEvent:
    """Test the log_event convenience function."""
    
    def test_log_event_with_valid_logger(self):
        """Test that log_event properly delegates to StructuredLogger."""
        logger = create_logger()
        
        # Mock the logger's log method to verify it's called
        with patch.object(logger, 'log') as mock_log:
            log_event(logger, "test_event", {"key": "value"})
            
            mock_log.assert_called_once_with("test_event", {"key": "value"})
    
    def test_log_event_with_none_logger(self):
        """Test that log_event handles None logger gracefully."""
        # Should not raise any exceptions
        log_event(None, "test_event", {"key": "value"})
        log_event(None, "test_event")  # No payload
        
        # No assertions needed - just verify no exceptions
    
    def test_log_event_without_payload(self):
        """Test that log_event works without payload."""
        logger = create_logger()
        
        with patch.object(logger, 'log') as mock_log:
            log_event(logger, "test_event")
            
            mock_log.assert_called_once_with("test_event", None)
    
    def test_log_event_with_empty_payload(self):
        """Test that log_event works with empty payload."""
        logger = create_logger()
        
        with patch.object(logger, 'log') as mock_log:
            log_event(logger, "test_event", {})
            
            mock_log.assert_called_once_with("test_event", {})
    
    def test_log_event_with_complex_payload(self):
        """Test that log_event works with complex payload data."""
        logger = create_logger()
        
        complex_payload = {
            "count": 42,
            "duration_ms": 123.45,
            "metadata": {
                "nested": "value",
                "list": [1, 2, 3]
            },
            "success": True
        }
        
        with patch.object(logger, 'log') as mock_log:
            log_event(logger, "complex_event", complex_payload)
            
            mock_log.assert_called_once_with("complex_event", complex_payload)
    
    def test_log_event_functional_test(self):
        """Test that log_event produces actual log output."""
        logger = create_logger()
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger._logger.handlers = [handler]
        
        log_event(logger, "functional_test", {"result": "success"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["event"] == "functional_test"
        assert log_data["payload"]["result"] == "success"
        assert "timestamp" in log_data


class TestFacadeExports:
    """Test that the facade properly exports components."""
    
    def test_facade_exports_structured_logger(self):
        """Test that StructuredLogger is properly re-exported."""
        from luma.observability.logger import StructuredLogger as FacadeLogger
        from luma.core.structured_logger import StructuredLogger as CoreLogger
        
        # Should be the same class
        assert FacadeLogger is CoreLogger
    
    def test_facade_functions_available_from_main_module(self):
        """Test that facade functions are available from main observability module."""
        from luma.observability import create_logger, log_event
        
        # Should be importable
        assert callable(create_logger)
        assert callable(log_event)
        
        # Should be functional
        logger = create_logger()
        assert isinstance(logger, StructuredLogger)
        
        # log_event should work
        log_event(logger, "test_from_main_module")


class TestIntegrationWithInstrumentedComponents:
    """Test integration patterns with instrumented components."""
    
    def test_logger_injection_pattern(self):
        """Test the typical logger injection pattern used by instrumented components."""
        logger = create_logger(name="component_logger")
        
        # Simulate how an instrumented component would use the logger
        def instrumented_operation(logger_instance=None):
            log_event(logger_instance, "operation_started")
            
            # Simulate some work
            result = {"processed": 10, "duration_ms": 50.0}
            
            log_event(logger_instance, "operation_completed", result)
            return result
        
        # Test with logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger._logger.handlers = [handler]
        
        result = instrumented_operation(logger_instance=logger)
        
        assert result["processed"] == 10
        
        # Verify logs were generated
        output = stream.getvalue()
        lines = output.strip().split('\n')
        assert len(lines) == 2
        
        start_log = json.loads(lines[0])
        complete_log = json.loads(lines[1])
        
        assert start_log["event"] == "operation_started"
        assert complete_log["event"] == "operation_completed"
        assert complete_log["payload"]["processed"] == 10
    
    def test_optional_logger_pattern(self):
        """Test that components work correctly when logger is None."""
        def instrumented_operation(logger_instance=None):
            log_event(logger_instance, "operation_started")
            
            # Business logic should work regardless of logger
            result = {"processed": 5}
            
            log_event(logger_instance, "operation_completed", result)
            return result
        
        # Test without logger (None)
        result = instrumented_operation(logger_instance=None)
        
        # Business logic should still work
        assert result["processed"] == 5
    
    def test_multiple_loggers_isolation(self):
        """Test that multiple loggers work independently."""
        logger1 = create_logger(name="component_1")
        logger2 = create_logger(name="component_2")
        
        stream1 = StringIO()
        stream2 = StringIO()
        
        handler1 = logging.StreamHandler(stream1)
        handler2 = logging.StreamHandler(stream2)
        
        logger1._logger.handlers = [handler1]
        logger2._logger.handlers = [handler2]
        
        # Log to different loggers
        log_event(logger1, "component_1_event", {"component": 1})
        log_event(logger2, "component_2_event", {"component": 2})
        
        # Verify isolation
        output1 = stream1.getvalue().strip()
        output2 = stream2.getvalue().strip()
        
        log1_data = json.loads(output1)
        log2_data = json.loads(output2)
        
        assert log1_data["event"] == "component_1_event"
        assert log1_data["payload"]["component"] == 1
        
        assert log2_data["event"] == "component_2_event"
        assert log2_data["payload"]["component"] == 2


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_log_event_with_invalid_logger_type(self):
        """Test that log_event handles invalid logger types gracefully."""
        # These should raise AttributeError since log_event expects a logger with .log() method
        # or None. This is the expected behavior - invalid types should fail fast.
        with pytest.raises(AttributeError):
            log_event("not_a_logger", "test_event")
        
        with pytest.raises(AttributeError):
            log_event(123, "test_event")
        
        with pytest.raises(AttributeError):
            log_event([], "test_event")
    
    def test_create_logger_with_invalid_level(self):
        """Test create_logger with invalid logging level."""
        # Should still create logger, Python logging will handle invalid levels
        logger = create_logger(level=999)  # Invalid level
        assert isinstance(logger, StructuredLogger)
    
    def test_log_event_with_non_serializable_payload(self):
        """Test log_event behavior with non-JSON-serializable payload."""
        logger = create_logger()
        
        # This should raise TypeError since the underlying StructuredLogger
        # tries to serialize to JSON. This is expected behavior.
        class NonSerializable:
            pass
        
        with pytest.raises(TypeError, match="not JSON serializable"):
            log_event(logger, "test_event", {"obj": NonSerializable()})


class TestPerformance:
    """Test performance characteristics of facade functions."""
    
    def test_log_event_with_none_is_fast(self):
        """Test that log_event with None logger has minimal overhead."""
        import time
        
        # Time multiple calls with None logger
        start_time = time.perf_counter()
        
        for _ in range(1000):
            log_event(None, "test_event", {"key": "value"})
        
        duration = time.perf_counter() - start_time
        
        # Should complete very quickly (less than 10ms for 1000 calls)
        assert duration < 0.01, f"log_event(None, ...) took {duration:.4f}s for 1000 calls"
    
    def test_create_logger_creates_unique_instances(self):
        """Test that create_logger creates unique instances."""
        logger1 = create_logger()
        logger2 = create_logger()
        
        # Should be different instances
        assert logger1 is not logger2
        
        # But same type
        assert type(logger1) is type(logger2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])