"""
Comprehensive unit tests for the observability module.

This test suite provides comprehensive coverage of all observability module components:
- metrics.py facade functionality
- logger.py facade functionality  
- schemas.py model serialization
- tracing.py trace context management
- observability_service.py orchestration

The tests verify that all components work correctly both individually and together,
ensuring the observability module provides a cohesive and reliable interface.

**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7**
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Any
from unittest.mock import Mock, patch

import pytest

# Import all observability module components
from luma.observability import (
    MetricsCollector,
    StructuredLogger,
    timed_operation,
    create_metrics_collector,
    create_logger,
    log_event,
    TraceEvent,
    MetricRecord,
    LogEvent,
    create_trace_event,
    create_log_event,
    TraceContext,
    set_current_trace_context,
    get_current_trace_context,
    trace_context,
    ObservabilityService,
)


class TestMetricsFacade:
    """Test metrics.py facade functionality."""
    
    def test_create_metrics_collector_factory(self):
        """Test that create_metrics_collector creates functional MetricsCollector."""
        metrics = create_metrics_collector()
        
        assert isinstance(metrics, MetricsCollector)
        
        # Verify functionality
        metrics.increment("test_counter", 5)
        metrics.record_duration("test_timer", 100.0)
        
        snapshot = metrics.get_snapshot()
        assert snapshot["counters"]["test_counter"] == 5
        assert snapshot["timers"]["test_timer"]["count"] == 1
        assert snapshot["timers"]["test_timer"]["sum"] == 100.0
    
    def test_timed_operation_context_manager(self):
        """Test timed_operation context manager functionality."""
        metrics = create_metrics_collector()
        
        with timed_operation(metrics, "operation_latency_ms"):
            time.sleep(0.01)  # 10ms delay
        
        snapshot = metrics.get_snapshot()
        timer_stats = snapshot["timers"]["operation_latency_ms"]
        
        assert timer_stats["count"] == 1
        assert timer_stats["min"] >= 10  # At least 10ms
        assert timer_stats["mean"] >= 10
    
    def test_timed_operation_with_none_collector(self):
        """Test timed_operation handles None metrics_collector gracefully."""
        # Should not raise exceptions
        with timed_operation(None, "operation_latency_ms"):
            time.sleep(0.01)
        
        # No assertions needed - just verify no exceptions
    
    def test_timed_operation_on_exception(self):
        """Test timed_operation records duration even on exceptions."""
        metrics = create_metrics_collector()
        
        with pytest.raises(ValueError):
            with timed_operation(metrics, "failing_operation_ms"):
                raise ValueError("Test error")
        
        # Duration should still be recorded
        snapshot = metrics.get_snapshot()
        assert "failing_operation_ms" in snapshot["timers"]
        assert snapshot["timers"]["failing_operation_ms"]["count"] == 1
    
    def test_metrics_facade_exports(self):
        """Test that metrics facade properly exports MetricsCollector."""
        from luma.observability.metrics import MetricsCollector as FacadeCollector
        from luma.core.metrics_collector import MetricsCollector as CoreCollector
        
        assert FacadeCollector is CoreCollector


class TestLoggerFacade:
    """Test logger.py facade functionality."""
    
    def test_create_logger_factory(self):
        """Test that create_logger creates functional StructuredLogger."""
        logger = create_logger(name="test_logger", level=logging.DEBUG)
        
        assert isinstance(logger, StructuredLogger)
        assert logger._logger.name == "test_logger"
        assert logger._logger.level == logging.DEBUG
    
    def test_log_event_convenience_function(self):
        """Test log_event convenience function."""
        logger = create_logger()
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger._logger.handlers = [handler]
        
        log_event(logger, "test_event", {"key": "value", "count": 42})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert log_data["event"] == "test_event"
        assert log_data["payload"]["key"] == "value"
        assert log_data["payload"]["count"] == 42
        assert "timestamp" in log_data
    
    def test_log_event_with_none_logger(self):
        """Test log_event handles None logger gracefully."""
        # Should not raise exceptions
        log_event(None, "test_event", {"key": "value"})
        log_event(None, "test_event")  # No payload
    
    def test_logger_facade_exports(self):
        """Test that logger facade properly exports StructuredLogger."""
        from luma.observability.logger import StructuredLogger as FacadeLogger
        from luma.core.structured_logger import StructuredLogger as CoreLogger
        
        assert FacadeLogger is CoreLogger


class TestSchemasModelSerialization:
    """Test schemas.py model serialization functionality."""
    
    def test_trace_event_serialization(self):
        """Test TraceEvent serialization and deserialization."""
        trace = create_trace_event(
            trace_id="trace-123",
            span_id="span-456",
            operation_name="test_operation",
            status="success",
            metadata={"key": "value"}
        )
        
        # Test to_dict
        data = trace.to_dict()
        assert isinstance(data, dict)
        assert data["trace_id"] == "trace-123"
        assert data["operation_name"] == "test_operation"
        assert data["metadata"]["key"] == "value"
        
        # Test to_json
        json_str = trace.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["trace_id"] == "trace-123"
        
        # Test from_dict
        restored = TraceEvent.from_dict(data)
        assert restored.trace_id == trace.trace_id
        assert restored.operation_name == trace.operation_name
        assert restored.metadata == trace.metadata
    
    def test_metric_record_serialization(self):
        """Test MetricRecord serialization and deserialization."""
        snapshot = {
            "counters": {"total_memories": 100, "cleanup_runs": 5},
            "timers": {
                "retrieval_latency_ms": {
                    "count": 10,
                    "sum": 500.0,
                    "min": 10.0,
                    "max": 100.0,
                    "mean": 50.0
                }
            }
        }
        
        record = MetricRecord.from_snapshot(snapshot, metadata={"source": "test"})
        
        # Test serialization
        data = record.to_dict()
        assert data["counters"] == snapshot["counters"]
        assert data["timers"] == snapshot["timers"]
        assert data["metadata"]["source"] == "test"
        
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["counters"]["total_memories"] == 100
        
        # Test deserialization
        restored = MetricRecord.from_dict(data)
        assert restored.counters == record.counters
        assert restored.timers == record.timers
        assert restored.metadata == record.metadata
    
    def test_log_event_serialization(self):
        """Test LogEvent serialization and deserialization."""
        log = create_log_event(
            event="test_event",
            payload={"count": 42, "success": True},
            level="INFO",
            source="TestComponent"
        )
        
        # Test serialization
        data = log.to_dict()
        assert data["event"] == "test_event"
        assert data["payload"]["count"] == 42
        assert data["level"] == "INFO"
        assert data["source"] == "TestComponent"
        
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["event"] == "test_event"
        
        # Test deserialization
        restored = LogEvent.from_dict(data)
        assert restored.event == log.event
        assert restored.payload == log.payload
        assert restored.level == log.level
        assert restored.source == log.source
    
    def test_factory_functions_generate_timestamps(self):
        """Test that factory functions generate proper timestamps."""
        trace = create_trace_event("trace-1", "span-1", "op1")
        log = create_log_event("event1")
        
        # Both should have ISO format timestamps with Z suffix
        assert trace.start_time.endswith("Z")
        assert "T" in trace.start_time
        assert log.timestamp.endswith("Z")
        assert "T" in log.timestamp
        
        # Should be parseable as datetime
        datetime.fromisoformat(trace.start_time.replace('Z', '+00:00'))
        datetime.fromisoformat(log.timestamp.replace('Z', '+00:00'))


class TestTracingContextManagement:
    """Test tracing.py trace context management functionality."""
    
    def test_trace_context_lifecycle(self):
        """Test complete trace context lifecycle."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        # Start nested spans
        span1 = ctx.start_span("operation_1")
        span2 = ctx.start_span("operation_2")
        
        # End spans
        ctx.end_span(span2, status="success")
        ctx.end_span(span1, status="success")
        
        # End trace
        ctx.end_trace(status="success")
        
        events = ctx.get_events()
        assert len(events) == 3  # root + 2 spans
        
        # All should be completed
        for event in events:
            assert event.status == "success"
            assert event.end_time is not None
    
    def test_trace_context_manager(self):
        """Test trace_context context manager."""
        with trace_context("trace-123", "root_operation") as ctx:
            assert ctx.trace_id == "trace-123"
            assert get_current_trace_context() is ctx
            
            with ctx.span("child_operation"):
                pass
        
        # Context should be cleared after exit
        assert get_current_trace_context() is None
        
        # Events should be recorded
        events = ctx.get_events()
        assert len(events) == 2  # root + child
        for event in events:
            assert event.status == "success"
    
    def test_trace_context_thread_safety(self):
        """Test that trace contexts are thread-safe."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span_ids = []
        errors = []
        
        def create_spans():
            try:
                for i in range(5):
                    span_id = ctx.start_span(f"operation_{threading.current_thread().name}_{i}")
                    span_ids.append(span_id)
                    time.sleep(0.001)
                    ctx.end_span(span_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_spans) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        events = ctx.get_events()
        assert len(events) == 16  # root + 15 spans (3 threads * 5 each)
    
    def test_thread_local_context_isolation(self):
        """Test that thread-local contexts are properly isolated."""
        ctx1 = TraceContext.start_trace("trace-1", "op1")
        ctx2 = TraceContext.start_trace("trace-2", "op2")
        
        results = {}
        
        def thread_func(ctx, thread_id):
            set_current_trace_context(ctx)
            time.sleep(0.01)
            retrieved = get_current_trace_context()
            results[thread_id] = retrieved
        
        t1 = threading.Thread(target=thread_func, args=(ctx1, "t1"))
        t2 = threading.Thread(target=thread_func, args=(ctx2, "t2"))
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        assert results["t1"] is ctx1
        assert results["t2"] is ctx2


class TestObservabilityServiceOrchestration:
    """Test observability_service.py orchestration functionality."""
    
    def test_service_lifecycle_management(self):
        """Test service start, stop, reset lifecycle."""
        service = ObservabilityService()
        
        # Initially not started
        assert not service.is_started
        assert service.metrics is None
        assert service.logger is None
        
        # Start service
        service.start()
        assert service.is_started
        assert service.metrics is not None
        assert service.logger is not None
        
        # Record some data
        service.increment_counter("test_counter", 5)
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["test_counter"] == 5
        
        # Reset clears data but keeps service running
        service.reset()
        snapshot = service.get_metrics_snapshot()
        assert len(snapshot["counters"]) == 0
        assert service.is_started
        
        # Stop service
        service.stop()
        assert not service.is_started
    
    def test_service_dependency_injection(self):
        """Test service with injected dependencies."""
        metrics = MetricsCollector()
        logger = StructuredLogger(name="injected_logger")
        
        metrics.increment("pre_existing", 10)
        
        service = ObservabilityService(
            metrics_collector=metrics,
            logger=logger
        )
        service.start()
        
        # Should use injected instances
        assert service.metrics is metrics
        assert service.logger is logger
        
        # Pre-existing data should be preserved
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["pre_existing"] == 10
    
    def test_service_unified_interface(self):
        """Test that service provides unified interface for all operations."""
        service = ObservabilityService()
        service.start()
        
        # Metrics operations
        service.increment_counter("unified_counter", 3)
        service.record_duration("unified_timer", 150.0)
        
        # Logging operations
        service.log_event("unified_event", {"test": True})
        
        # Tracing operations
        trace_ctx = service.start_trace("trace-unified", "unified_operation")
        with trace_ctx.span("sub_operation"):
            pass
        service.end_trace("trace-unified")
        
        # Verify all operations worked
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["unified_counter"] == 3
        assert snapshot["timers"]["unified_timer"]["count"] == 1
        
        # Trace should be completed
        events = trace_ctx.get_events()
        assert len(events) == 2  # root + sub
        for event in events:
            assert event.status == "success"
    
    def test_service_thread_safety(self):
        """Test that service operations are thread-safe."""
        service = ObservabilityService()
        service.start()
        
        def concurrent_operations():
            for i in range(10):
                service.increment_counter("concurrent_counter")
                service.record_duration("concurrent_timer", 10.0)
                service.log_event("concurrent_event", {"iteration": i})
        
        threads = [threading.Thread(target=concurrent_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["concurrent_counter"] == 50  # 5 threads * 10 each
        assert snapshot["timers"]["concurrent_timer"]["count"] == 50


class TestModuleIntegration:
    """Test integration between all observability module components."""
    
    def test_complete_observability_workflow(self):
        """Test complete workflow using all observability components together."""
        # Create service with all components
        service = ObservabilityService()
        service.start()
        
        # Start a trace for the workflow
        with trace_context("workflow-123", "complete_workflow") as trace_ctx:
            # Step 1: Metrics collection
            with timed_operation(service.metrics, "step1_duration_ms"):
                service.increment_counter("workflow_steps")
                log_event(service.logger, "step1_completed", {"step": 1})
            
            # Step 2: Nested operations with tracing
            with trace_ctx.span("step2_processing") as span:
                service.increment_counter("workflow_steps")
                service.record_duration("step2_duration_ms", 75.0)
                log_event(service.logger, "step2_completed", {"step": 2, "span_id": span})
            
            # Step 3: Error handling
            try:
                with trace_ctx.span("step3_error_prone"):
                    service.increment_counter("workflow_steps")
                    raise ValueError("Simulated error")
            except ValueError:
                service.increment_counter("workflow_errors")
                log_event(service.logger, "step3_error_handled", {"error": "ValueError"})
        
        # Verify complete workflow was recorded
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["workflow_steps"] == 3
        assert snapshot["counters"]["workflow_errors"] == 1
        assert "step1_duration_ms" in snapshot["timers"]
        assert "step2_duration_ms" in snapshot["timers"]
        
        # Verify trace was recorded
        events = trace_ctx.get_events()
        assert len(events) == 4  # root + step2 + step3 + root completion
        
        # Root and step2 should succeed, step3 should error
        root_event = events[0]
        step2_event = next(e for e in events if "step2" in e.operation_name)
        step3_event = next(e for e in events if "step3" in e.operation_name)
        
        assert root_event.status == "success"
        assert step2_event.status == "success"
        assert step3_event.status == "error"
    
    def test_module_exports_all_components(self):
        """Test that the main observability module exports all expected components."""
        # Verify all expected exports are available
        expected_exports = [
            "MetricsCollector", "StructuredLogger",
            "timed_operation", "create_metrics_collector",
            "create_logger", "log_event",
            "TraceEvent", "MetricRecord", "LogEvent",
            "create_trace_event", "create_log_event",
            "TraceContext", "set_current_trace_context", "get_current_trace_context", "trace_context",
            "ObservabilityService"
        ]
        
        import luma.observability as obs_module
        
        for export in expected_exports:
            assert hasattr(obs_module, export), f"Missing export: {export}"
            assert callable(getattr(obs_module, export)) or hasattr(getattr(obs_module, export), '__class__')
    
    def test_no_circular_imports(self):
        """Test that all observability components can be imported without circular dependencies."""
        # If we can import everything without errors, there are no circular imports
        from luma.observability import (
            MetricsCollector, StructuredLogger, timed_operation, create_metrics_collector,
            create_logger, log_event, TraceEvent, MetricRecord, LogEvent,
            create_trace_event, create_log_event, TraceContext,
            set_current_trace_context, get_current_trace_context, trace_context,
            ObservabilityService
        )
        
        # Also test importing from core modules
        from luma.core.metrics_collector import MetricsCollector as CoreMetrics
        from luma.core.structured_logger import StructuredLogger as CoreLogger
        
        # Verify facade exports match core implementations
        assert MetricsCollector is CoreMetrics
        assert StructuredLogger is CoreLogger
    
    def test_observability_with_instrumented_components(self):
        """Test observability module integration with instrumented components."""
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        from datetime import datetime, UTC
        
        service = ObservabilityService()
        service.start()
        
        # Create instrumented component
        config = RankingConfig(
            alpha=0.5, beta=0.5, gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        engine = RankingEngine(
            config,
            metrics_collector=service.metrics,
            logger=service.logger
        )
        
        # Use the component
        memories = [
            RankedMemory(
                memory_id="mem_1",
                timestamp=datetime.now(UTC),
                content="test memory",
                namespace="test",
                similarity_score=0.8,
                importance_score=0.5,
                recency_score=0.3,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        result = engine.rank(memories)
        
        # Verify observability was captured
        snapshot = service.get_metrics_snapshot()
        assert "ranking_latency_ms" in snapshot["timers"]
        assert snapshot["timers"]["ranking_latency_ms"]["count"] == 1
        
        # Component should still function correctly
        assert len(result) == 1
        assert result[0].final_score > 0


class TestErrorHandling:
    """Test error handling across all observability components."""
    
    def test_graceful_degradation_with_none_dependencies(self):
        """Test that all components handle None dependencies gracefully."""
        # Metrics facade
        with timed_operation(None, "test_timer"):
            pass  # Should not raise
        
        # Logger facade
        log_event(None, "test_event", {"key": "value"})  # Should not raise
        
        # Service operations should fail gracefully when not started
        service = ObservabilityService()
        
        with pytest.raises(RuntimeError):
            service.increment_counter("test")
        
        with pytest.raises(RuntimeError):
            service.log_event("test")
    
    def test_exception_handling_in_context_managers(self):
        """Test that context managers handle exceptions properly."""
        metrics = create_metrics_collector()
        
        # timed_operation should record duration even on exception
        with pytest.raises(ValueError):
            with timed_operation(metrics, "failing_operation_ms"):
                raise ValueError("Test error")
        
        snapshot = metrics.get_snapshot()
        assert "failing_operation_ms" in snapshot["timers"]
        
        # trace_context should mark trace as error
        with pytest.raises(RuntimeError):
            with trace_context("error-trace", "failing_operation") as ctx:
                raise RuntimeError("Trace error")
        
        events = ctx.get_events()
        assert events[0].status == "error"
        assert "Trace error" in events[0].metadata["error"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])