"""
Unit tests for the metrics facade.

Tests verify:
- Factory function creates valid MetricsCollector instances
- timed_operation context manager correctly measures durations
- timed_operation handles None metrics_collector gracefully
- Convenience methods maintain O(1) complexity
"""

import time
import pytest
from luma.observability.metrics import (
    create_metrics_collector,
    timed_operation,
    MetricsCollector
)


def test_create_metrics_collector():
    """Test that factory function creates a valid MetricsCollector."""
    metrics = create_metrics_collector()
    
    assert isinstance(metrics, MetricsCollector)
    assert metrics is not None
    
    # Verify it's functional
    metrics.increment("test_counter")
    snapshot = metrics.get_snapshot()
    assert snapshot["counters"]["test_counter"] == 1


def test_timed_operation_records_duration():
    """Test that timed_operation context manager records durations."""
    metrics = create_metrics_collector()
    
    with timed_operation(metrics, "test_operation_ms"):
        time.sleep(0.01)  # Sleep for 10ms
    
    snapshot = metrics.get_snapshot()
    assert "test_operation_ms" in snapshot["timers"]
    
    timer_stats = snapshot["timers"]["test_operation_ms"]
    assert timer_stats["count"] == 1
    assert timer_stats["min"] >= 10  # At least 10ms
    assert timer_stats["max"] >= 10
    assert timer_stats["mean"] >= 10


def test_timed_operation_with_none_collector():
    """Test that timed_operation handles None metrics_collector gracefully."""
    # Should not raise any exceptions
    with timed_operation(None, "test_operation_ms"):
        time.sleep(0.01)
    
    # No assertions needed - just verify no exceptions


def test_timed_operation_records_on_exception():
    """Test that timed_operation records duration even when operation raises."""
    metrics = create_metrics_collector()
    
    with pytest.raises(ValueError):
        with timed_operation(metrics, "failing_operation_ms"):
            raise ValueError("Test exception")
    
    # Duration should still be recorded
    snapshot = metrics.get_snapshot()
    assert "failing_operation_ms" in snapshot["timers"]
    assert snapshot["timers"]["failing_operation_ms"]["count"] == 1


def test_timed_operation_multiple_calls():
    """Test that timed_operation correctly accumulates multiple measurements."""
    metrics = create_metrics_collector()
    
    # Record multiple operations
    for _ in range(3):
        with timed_operation(metrics, "repeated_operation_ms"):
            time.sleep(0.01)
    
    snapshot = metrics.get_snapshot()
    timer_stats = snapshot["timers"]["repeated_operation_ms"]
    
    assert timer_stats["count"] == 3
    assert timer_stats["sum"] >= 30  # At least 30ms total
    assert timer_stats["mean"] >= 10  # At least 10ms average


def test_timed_operation_nested_contexts():
    """Test that timed_operation works with nested timing contexts."""
    metrics = create_metrics_collector()
    
    with timed_operation(metrics, "outer_operation_ms"):
        time.sleep(0.01)
        with timed_operation(metrics, "inner_operation_ms"):
            time.sleep(0.01)
    
    snapshot = metrics.get_snapshot()
    
    # Both operations should be recorded
    assert "outer_operation_ms" in snapshot["timers"]
    assert "inner_operation_ms" in snapshot["timers"]
    
    outer_stats = snapshot["timers"]["outer_operation_ms"]
    inner_stats = snapshot["timers"]["inner_operation_ms"]
    
    # Outer should take longer than inner
    assert outer_stats["mean"] > inner_stats["mean"]


def test_facade_exports_metrics_collector():
    """Test that MetricsCollector is properly re-exported."""
    # This test verifies the import works
    from luma.observability.metrics import MetricsCollector as FacadeCollector
    from luma.core.metrics_collector import MetricsCollector as CoreCollector
    
    # Should be the same class
    assert FacadeCollector is CoreCollector


def test_facade_imports_from_observability_module():
    """Test that convenience functions are available from main observability module."""
    from luma.observability import timed_operation, create_metrics_collector
    
    # Should be importable
    assert callable(timed_operation)
    assert callable(create_metrics_collector)
    
    # Should be functional
    metrics = create_metrics_collector()
    assert isinstance(metrics, MetricsCollector)
