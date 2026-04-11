"""
Metrics Facade

This module provides a simplified interface for common metric operations,
building on top of the core MetricsCollector implementation.

The facade provides:
- Re-export of MetricsCollector for direct use
- Convenience methods for common metric patterns
- Context managers for automatic timing
- Simplified interfaces for standard metrics

All operations maintain O(1) complexity and thread-safety guarantees.
"""

from contextlib import contextmanager
from typing import Optional, Generator
import time

from luma.core.metrics_collector import MetricsCollector


# Re-export MetricsCollector for convenience
__all__ = ["MetricsCollector", "timed_operation", "create_metrics_collector"]


def create_metrics_collector() -> MetricsCollector:
    """
    Factory function to create a new MetricsCollector instance.
    
    This provides a convenient way to instantiate metrics collectors
    without directly importing from luma.core.
    
    Returns:
        A new MetricsCollector instance
    """
    return MetricsCollector()


@contextmanager
def timed_operation(
    metrics_collector: Optional[MetricsCollector],
    metric_name: str
) -> Generator[None, None, None]:
    """
    Context manager for automatically timing operations.
    
    This convenience method wraps an operation and automatically records
    its duration to the specified metric. If metrics_collector is None,
    the operation executes without timing overhead.
    
    Usage:
        with timed_operation(metrics, "operation_latency_ms"):
            # Your operation here
            perform_work()
    
    Args:
        metrics_collector: Optional MetricsCollector instance
        metric_name: Name of the timer metric to record to
    
    Yields:
        None
    """
    if metrics_collector is None:
        # No metrics collection - just execute the operation
        yield
        return
    
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        metrics_collector.record_duration(metric_name, duration_ms)
