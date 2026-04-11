"""
Observability & Metrics Layer

This module provides higher-level abstractions for observability in the Luma system.
It re-exports core observability components from luma.core for convenient access.

Core Components:
- MetricsCollector: Thread-safe metrics collection (counters and timers)
- StructuredLogger: JSON-formatted structured logging

Usage:
    from luma.observability import MetricsCollector, StructuredLogger
    
    # Create observability components
    metrics = MetricsCollector()
    logger = StructuredLogger()
    
    # Use with instrumented components
    from luma.core.ranking_engine import RankingEngine
    engine = RankingEngine(config, metrics_collector=metrics, logger=logger)
"""

from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger
from luma.observability.metrics import timed_operation, create_metrics_collector
from luma.observability.logger import create_logger, log_event
from luma.observability.schemas import (
    TraceEvent,
    MetricRecord,
    LogEvent,
    create_trace_event,
    create_log_event,
)
from luma.observability.tracing import (
    TraceContext,
    set_current_trace_context,
    get_current_trace_context,
    trace_context,
)
from luma.observability.observability_service import ObservabilityService

__all__ = [
    "MetricsCollector",
    "StructuredLogger",
    "timed_operation",
    "create_metrics_collector",
    "create_logger",
    "log_event",
    "TraceEvent",
    "MetricRecord",
    "LogEvent",
    "create_trace_event",
    "create_log_event",
    "TraceContext",
    "set_current_trace_context",
    "get_current_trace_context",
    "trace_context",
    "ObservabilityService",
]
