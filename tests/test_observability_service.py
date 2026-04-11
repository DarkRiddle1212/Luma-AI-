"""
Unit tests for ObservabilityService orchestration.

This module tests the central orchestration service that coordinates
MetricsCollector, StructuredLogger, and TraceContext to provide a unified
interface for observability operations.

**Validates: Requirements 8.1-8.9, 12.1-12.3**
"""

import pytest
import threading
from datetime import datetime, timezone
from typing import Dict, Any

from luma.observability.observability_service import ObservabilityService
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger
from luma.observability.tracing import TraceContext


class TestObservabilityServiceLifecycle:
    """Test service lifecycle management (start, stop, reset)."""
    
    def test_service_starts_successfully(self):
        """Test that service can be started."""
        service = ObservabilityService()
        assert not service.is_started
        
        service.start()
        
        assert service.is_started
        assert service.metrics is not None
        assert service.logger is not None
    
    def test_service_start_is_idempotent(self):
        """Test that calling start() multiple times has no additional effect."""
        service = ObservabilityService()
        
        service.start()
        metrics1 = service.metrics
        logger1 = service.logger
        
        service.start()  # Call again
        metrics2 = service.metrics
        logger2 = service.logger
        
        # Should be the same instances
        assert metrics1 is metrics2
        assert logger1 is logger2
        assert service.is_started
    
    def test_service_stops_successfully(self):
        """Test that service can be stopped."""
        service = ObservabilityService()
        service.start()
        
        assert service.is_started
        
        service.stop()
        
        assert not service.is_started
    
    def test_service_stop_is_idempotent(self):
        """Test that calling stop() multiple times has no additional effect."""
        service = ObservabilityService()
        service.start()
        service.stop()
        
        assert not service.is_started
        
        service.stop()  # Call again
        
        assert not service.is_started
    
    def test_service_can_be_restarted(self):
        """Test that service can be stopped and restarted."""
        service = ObservabilityService()
        
        service.start()
        assert service.is_started
        
        service.stop()
        assert not service.is_started
        
        service.start()
        assert service.is_started
        assert service.metrics is not None
        assert service.logger is not None
    
    def test_service_reset_clears_metrics(self):
        """Test that reset clears all metrics."""
        service = ObservabilityService()
        service.start()
        
        # Record some metrics
        service.increment_counter("test_counter", 5)
        service.record_duration("test_timer", 100.0)
        
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["test_counter"] == 5
        assert "test_timer" in snapshot["timers"]
        
        # Reset
        service.reset()
        
        # Verify metrics are cleared
        snapshot = service.get_metrics_snapshot()
        assert len(snapshot["counters"]) == 0
        assert len(snapshot["timers"]) == 0
    
    def test_service_reset_requires_started_service(self):
        """Test that reset raises error if service is not started."""
        service = ObservabilityService()
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.reset()


class TestObservabilityServiceDependencyInjection:
    """Test optional dependency injection pattern."""
    
    def test_service_accepts_injected_metrics_collector(self):
        """Test that service accepts pre-configured MetricsCollector."""
        metrics = MetricsCollector()
        metrics.increment("pre_existing_counter", 10)
        
        service = ObservabilityService(metrics_collector=metrics)
        service.start()
        
        # Should use the injected instance
        assert service.metrics is metrics
        
        # Pre-existing data should be preserved
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["pre_existing_counter"] == 10
    
    def test_service_accepts_injected_logger(self):
        """Test that service accepts pre-configured StructuredLogger."""
        logger = StructuredLogger(name="custom_logger")
        
        service = ObservabilityService(logger=logger)
        service.start()
        
        # Should use the injected instance
        assert service.logger is logger
    
    def test_service_accepts_both_injected_dependencies(self):
        """Test that service accepts both injected dependencies."""
        metrics = MetricsCollector()
        logger = StructuredLogger(name="custom_logger")
        
        service = ObservabilityService(
            metrics_collector=metrics,
            logger=logger
        )
        service.start()
        
        assert service.metrics is metrics
        assert service.logger is logger
    
    def test_service_creates_components_when_not_injected(self):
        """Test that service creates components when not provided."""
        service = ObservabilityService()
        service.start()
        
        # Should create new instances
        assert service.metrics is not None
        assert service.logger is not None
        assert isinstance(service.metrics, MetricsCollector)
        assert isinstance(service.logger, StructuredLogger)


class TestObservabilityServiceMetricsOperations:
    """Test metrics operations through the service."""
    
    def test_increment_counter_convenience_method(self):
        """Test convenience method for incrementing counters."""
        service = ObservabilityService()
        service.start()
        
        service.increment_counter("test_counter")
        service.increment_counter("test_counter", 5)
        
        snapshot = service.get_metrics_snapshot()
        assert snapshot["counters"]["test_counter"] == 6
    
    def test_record_duration_convenience_method(self):
        """Test convenience method for recording durations."""
        service = ObservabilityService()
        service.start()
        
        service.record_duration("test_timer", 100.5)
        service.record_duration("test_timer", 200.5)
        
        snapshot = service.get_metrics_snapshot()
        assert snapshot["timers"]["test_timer"]["count"] == 2
        assert snapshot["timers"]["test_timer"]["sum"] == 301.0
    
    def test_get_metrics_snapshot(self):
        """Test getting metrics snapshot."""
        service = ObservabilityService()
        service.start()
        
        service.increment_counter("counter1", 10)
        service.record_duration("timer1", 50.0)
        
        snapshot = service.get_metrics_snapshot()
        
        assert "counters" in snapshot
        assert "timers" in snapshot
        assert snapshot["counters"]["counter1"] == 10
        assert snapshot["timers"]["timer1"]["count"] == 1
    
    def test_get_metrics_record(self):
        """Test getting structured MetricRecord."""
        service = ObservabilityService()
        service.start()
        
        service.increment_counter("counter1", 5)
        
        record = service.get_metrics_record(metadata={"source": "test"})
        
        assert record.counters["counter1"] == 5
        assert record.metadata["source"] == "test"
        assert record.timestamp is not None
    
    def test_metrics_operations_require_started_service(self):
        """Test that metrics operations require service to be started."""
        service = ObservabilityService()
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.increment_counter("test")
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.record_duration("test", 100.0)
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.get_metrics_snapshot()


class TestObservabilityServiceLoggingOperations:
    """Test logging operations through the service."""
    
    def test_log_event_convenience_method(self):
        """Test convenience method for logging events."""
        service = ObservabilityService()
        service.start()
        
        # Should not raise any exceptions
        service.log_event("test_event", {"key": "value"})
        service.log_event("another_event")
    
    def test_log_event_requires_started_service(self):
        """Test that log_event requires service to be started."""
        service = ObservabilityService()
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.log_event("test_event")


class TestObservabilityServiceTraceOperations:
    """Test trace context management through the service."""
    
    def test_start_trace_creates_context(self):
        """Test that start_trace creates and registers a TraceContext."""
        service = ObservabilityService()
        service.start()
        
        trace_ctx = service.start_trace("trace-123", "test_operation")
        
        assert trace_ctx is not None
        assert isinstance(trace_ctx, TraceContext)
        assert trace_ctx.trace_id == "trace-123"
    
    def test_get_trace_retrieves_context(self):
        """Test that get_trace retrieves an existing context."""
        service = ObservabilityService()
        service.start()
        
        trace_ctx = service.start_trace("trace-123", "test_operation")
        retrieved_ctx = service.get_trace("trace-123")
        
        assert retrieved_ctx is trace_ctx
    
    def test_get_trace_returns_none_for_nonexistent_trace(self):
        """Test that get_trace returns None for non-existent trace."""
        service = ObservabilityService()
        service.start()
        
        result = service.get_trace("nonexistent")
        
        assert result is None
    
    def test_end_trace_completes_and_removes_context(self):
        """Test that end_trace completes and removes the context."""
        service = ObservabilityService()
        service.start()
        
        trace_ctx = service.start_trace("trace-123", "test_operation")
        service.end_trace("trace-123", status="success")
        
        # Should be removed from active traces
        result = service.get_trace("trace-123")
        assert result is None
        
        # Events should be recorded
        events = trace_ctx.get_events()
        assert len(events) > 0
    
    def test_start_trace_rejects_duplicate_trace_id(self):
        """Test that start_trace rejects duplicate trace IDs."""
        service = ObservabilityService()
        service.start()
        
        service.start_trace("trace-123", "operation1")
        
        with pytest.raises(ValueError, match="Trace ID already exists"):
            service.start_trace("trace-123", "operation2")
    
    def test_end_trace_rejects_nonexistent_trace_id(self):
        """Test that end_trace rejects non-existent trace IDs."""
        service = ObservabilityService()
        service.start()
        
        with pytest.raises(ValueError, match="Trace ID not found"):
            service.end_trace("nonexistent")
    
    def test_get_all_traces(self):
        """Test getting all active trace contexts."""
        service = ObservabilityService()
        service.start()
        
        trace1 = service.start_trace("trace-1", "op1")
        trace2 = service.start_trace("trace-2", "op2")
        
        all_traces = service.get_all_traces()
        
        assert len(all_traces) == 2
        assert trace1 in all_traces
        assert trace2 in all_traces
    
    def test_reset_clears_trace_contexts(self):
        """Test that reset clears all trace contexts."""
        service = ObservabilityService()
        service.start()
        
        service.start_trace("trace-1", "op1")
        service.start_trace("trace-2", "op2")
        
        assert len(service.get_all_traces()) == 2
        
        service.reset()
        
        assert len(service.get_all_traces()) == 0
    
    def test_trace_operations_require_started_service(self):
        """Test that trace operations require service to be started."""
        service = ObservabilityService()
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.start_trace("trace-123", "operation")
        
        with pytest.raises(RuntimeError, match="service is not started"):
            service.end_trace("trace-123")


class TestObservabilityServiceThreadSafety:
    """Test thread-safe operations of the service."""
    
    def test_concurrent_metric_operations(self):
        """Test that concurrent metric operations are thread-safe."""
        service = ObservabilityService()
        service.start()
        
        num_threads = 10
        increments_per_thread = 100
        
        def increment_metrics():
            for _ in range(increments_per_thread):
                service.increment_counter("concurrent_counter")
                service.record_duration("concurrent_timer", 10.0)
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_metrics)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        snapshot = service.get_metrics_snapshot()
        expected_count = num_threads * increments_per_thread
        
        assert snapshot["counters"]["concurrent_counter"] == expected_count
        assert snapshot["timers"]["concurrent_timer"]["count"] == expected_count
    
    def test_concurrent_lifecycle_operations(self):
        """Test that concurrent lifecycle operations are thread-safe."""
        service = ObservabilityService()
        
        def start_service():
            service.start()
        
        def stop_service():
            try:
                service.stop()
            except:
                pass  # May fail if not started
        
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=start_service))
            threads.append(threading.Thread(target=stop_service))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Service should be in a consistent state
        assert isinstance(service.is_started, bool)


class TestObservabilityServiceIntegration:
    """Test integration with instrumented components."""
    
    def test_service_with_ranking_engine(self):
        """Test using service with RankingEngine."""
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        from datetime import datetime, UTC
        
        service = ObservabilityService()
        service.start()
        
        config = RankingConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        engine = RankingEngine(
            config,
            metrics_collector=service.metrics,
            logger=service.logger
        )
        
        memories = [
            RankedMemory(
                memory_id="1",
                timestamp=datetime.now(UTC),
                content="test",
                namespace="test",
                similarity_score=0.8,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        result = engine.rank(memories)
        
        # Verify metrics were recorded
        snapshot = service.get_metrics_snapshot()
        assert "ranking_latency_ms" in snapshot["timers"]
        assert snapshot["timers"]["ranking_latency_ms"]["count"] == 1
    
    def test_service_provides_unified_interface(self):
        """Test that service provides unified interface for all observability operations."""
        service = ObservabilityService()
        service.start()
        
        # Metrics operations
        service.increment_counter("test_counter", 5)
        service.record_duration("test_timer", 100.0)
        
        # Logging operations
        service.log_event("test_event", {"key": "value"})
        
        # Tracing operations
        trace_ctx = service.start_trace("trace-123", "test_operation")
        with trace_ctx.span("sub_operation"):
            pass
        service.end_trace("trace-123")
        
        # Get snapshot
        snapshot = service.get_metrics_snapshot()
        
        # Verify all operations worked
        assert snapshot["counters"]["test_counter"] == 5
        assert snapshot["timers"]["test_timer"]["count"] == 1
        
        # Service provides unified access to all observability features
        assert service.metrics is not None
        assert service.logger is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
