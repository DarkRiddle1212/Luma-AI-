"""
Observability Service Orchestration

This module provides a central orchestration service that coordinates
MetricsCollector, StructuredLogger, and TraceContext to provide a unified
interface for observability operations across the Luma system.

The ObservabilityService:
- Manages lifecycle of observability components (start, stop, reset)
- Provides unified interface for metrics, logging, and tracing
- Supports optional dependency injection pattern
- Ensures thread-safe operations across all components
- Enables easy integration with instrumented components

Usage:
    # Create and start service
    service = ObservabilityService()
    service.start()
    
    # Use with instrumented components
    engine = RankingEngine(
        config,
        metrics_collector=service.metrics,
        logger=service.logger
    )
    
    # Get metrics snapshot
    snapshot = service.get_metrics_snapshot()
    
    # Reset and stop
    service.reset()
    service.stop()
"""

import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger
from luma.observability.tracing import TraceContext
from luma.observability.schemas import MetricRecord, LogEvent


class ObservabilityService:
    """
    Central orchestration service for observability operations.
    
    This service coordinates MetricsCollector, StructuredLogger, and TraceContext
    to provide a unified interface for observability across the system. It manages
    component lifecycle and provides convenient access to all observability features.
    
    Attributes:
        metrics: MetricsCollector instance for tracking counters and timers
        logger: StructuredLogger instance for structured JSON logging
        _started: Flag indicating whether the service has been started
        _lock: Thread lock for ensuring thread-safe lifecycle operations
        _trace_contexts: Dictionary mapping trace IDs to TraceContext instances
    
    Example:
        service = ObservabilityService()
        service.start()
        
        # Use with components
        engine = RankingEngine(
            config,
            metrics_collector=service.metrics,
            logger=service.logger
        )
        
        # Get snapshot
        snapshot = service.get_metrics_snapshot()
        
        service.stop()
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the ObservabilityService.
        
        Supports optional dependency injection - if components are not provided,
        new instances will be created when the service is started.
        
        Args:
            metrics_collector: Optional MetricsCollector instance to use
            logger: Optional StructuredLogger instance to use
        """
        self._metrics = metrics_collector
        self._logger = logger
        self._started = False
        self._lock = threading.Lock()
        self._trace_contexts: Dict[str, TraceContext] = {}
    
    @property
    def metrics(self) -> Optional[MetricsCollector]:
        """
        Get the MetricsCollector instance.
        
        Returns:
            MetricsCollector instance if service is started, None otherwise
        """
        return self._metrics
    
    @property
    def logger(self) -> Optional[StructuredLogger]:
        """
        Get the StructuredLogger instance.
        
        Returns:
            StructuredLogger instance if service is started, None otherwise
        """
        return self._logger
    
    @property
    def is_started(self) -> bool:
        """
        Check if the service has been started.
        
        Returns:
            True if service is started, False otherwise
        """
        with self._lock:
            return self._started
    
    def start(self) -> None:
        """
        Start the observability service.
        
        This initializes all observability components if they haven't been
        provided via dependency injection. The service must be started before
        its components can be used.
        
        This operation is idempotent - calling start() multiple times has no
        additional effect after the first call.
        """
        with self._lock:
            if self._started:
                return
            
            # Create components if not provided via dependency injection
            if self._metrics is None:
                self._metrics = MetricsCollector()
            
            if self._logger is None:
                self._logger = StructuredLogger(name="observability_service")
            
            self._started = True
            
            # Log service start
            if self._logger:
                self._logger.log(
                    "observability_service_started",
                    {"timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}
                )
    
    def stop(self) -> None:
        """
        Stop the observability service.
        
        This performs cleanup operations and logs the service shutdown.
        After stopping, the service can be restarted with start().
        
        This operation is idempotent - calling stop() multiple times has no
        additional effect after the first call.
        """
        with self._lock:
            if not self._started:
                return
            
            # Log service stop
            if self._logger:
                self._logger.log(
                    "observability_service_stopped",
                    {"timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}
                )
            
            # Clear trace contexts
            self._trace_contexts.clear()
            
            self._started = False
    
    def reset(self) -> None:
        """
        Reset all metrics and clear trace contexts.
        
        This clears all collected metrics and removes all trace contexts,
        providing a fresh start for measurement. The service remains started
        after reset.
        
        Raises:
            RuntimeError: If service is not started
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot reset: service is not started")
            
            # Reset metrics
            if self._metrics:
                self._metrics.reset()
            
            # Clear trace contexts
            self._trace_contexts.clear()
            
            # Log reset
            if self._logger:
                self._logger.log(
                    "observability_service_reset",
                    {"timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}
                )
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of all current metrics.
        
        Returns:
            Dictionary containing counters and timers with statistics
        
        Raises:
            RuntimeError: If service is not started
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot get snapshot: service is not started")
            
            if self._metrics:
                return self._metrics.get_snapshot()
            else:
                return {"counters": {}, "timers": {}}
    
    def get_metrics_record(self, metadata: Optional[Dict[str, Any]] = None) -> MetricRecord:
        """
        Get a MetricRecord with current metrics and timestamp.
        
        This is a convenience method that wraps get_metrics_snapshot() and
        returns a structured MetricRecord instance.
        
        Args:
            metadata: Optional metadata to include in the record
        
        Returns:
            MetricRecord instance with current metrics
        
        Raises:
            RuntimeError: If service is not started
        """
        snapshot = self.get_metrics_snapshot()
        return MetricRecord.from_snapshot(snapshot, metadata)
    
    def start_trace(
        self,
        trace_id: str,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceContext:
        """
        Start a new trace context.
        
        Creates and registers a new TraceContext for tracking request-level
        operations. The trace context can be used to create nested spans.
        
        Args:
            trace_id: Unique identifier for this trace
            operation_name: Name of the root operation
            metadata: Optional metadata for the root span
        
        Returns:
            TraceContext instance for this trace
        
        Raises:
            RuntimeError: If service is not started
            ValueError: If trace_id already exists
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot start trace: service is not started")
            
            if trace_id in self._trace_contexts:
                raise ValueError(f"Trace ID already exists: {trace_id}")
            
            trace_ctx = TraceContext.start_trace(trace_id, operation_name, metadata)
            self._trace_contexts[trace_id] = trace_ctx
            
            # Log trace start
            if self._logger:
                self._logger.log(
                    "trace_started",
                    {
                        "trace_id": trace_id,
                        "operation_name": operation_name,
                        "metadata": metadata or {}
                    }
                )
            
            return trace_ctx
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """
        Get an existing trace context by ID.
        
        Args:
            trace_id: ID of the trace to retrieve
        
        Returns:
            TraceContext instance if found, None otherwise
        """
        with self._lock:
            return self._trace_contexts.get(trace_id)
    
    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End a trace and remove it from active traces.
        
        Args:
            trace_id: ID of the trace to end
            status: Final status of the trace (default: "success")
            metadata: Optional additional metadata for the root span
        
        Raises:
            RuntimeError: If service is not started
            ValueError: If trace_id does not exist
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot end trace: service is not started")
            
            trace_ctx = self._trace_contexts.get(trace_id)
            if trace_ctx is None:
                raise ValueError(f"Trace ID not found: {trace_id}")
            
            trace_ctx.end_trace(status, metadata)
            
            # Log trace end
            if self._logger:
                events = trace_ctx.get_events()
                self._logger.log(
                    "trace_ended",
                    {
                        "trace_id": trace_id,
                        "status": status,
                        "event_count": len(events),
                        "metadata": metadata or {}
                    }
                )
            
            # Remove from active traces
            del self._trace_contexts[trace_id]
    
    def get_all_traces(self) -> List[TraceContext]:
        """
        Get all active trace contexts.
        
        Returns:
            List of all active TraceContext instances
        """
        with self._lock:
            return list(self._trace_contexts.values())
    
    def increment_counter(self, name: str, value: float = 1) -> None:
        """
        Convenience method to increment a counter.
        
        Args:
            name: Name of the counter to increment
            value: Amount to increment by (default: 1)
        
        Raises:
            RuntimeError: If service is not started
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot increment counter: service is not started")
            
            if self._metrics:
                self._metrics.increment(name, value)
    
    def record_duration(self, name: str, duration_ms: float) -> None:
        """
        Convenience method to record a duration.
        
        Args:
            name: Name of the timer
            duration_ms: Duration value in milliseconds
        
        Raises:
            RuntimeError: If service is not started
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot record duration: service is not started")
            
            if self._metrics:
                self._metrics.record_duration(name, duration_ms)
    
    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Convenience method to log an event.
        
        Args:
            event: Event name/type
            payload: Optional event-specific data
        
        Raises:
            RuntimeError: If service is not started
        """
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot log event: service is not started")
            
            if self._logger:
                self._logger.log(event, payload)


__all__ = ["ObservabilityService"]
