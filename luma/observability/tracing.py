"""
Request-Level Tracing Module

This module provides request-level tracing capabilities for tracking operation
lifecycles across the Luma system. It supports nested trace spans for operation
hierarchy and ensures thread-safe trace context management.

Key Features:
- TraceContext for tracking request lifecycle
- Nested span support for operation hierarchy
- Thread-safe context management using threading.local
- Automatic timestamp recording
- Integration with TraceEvent data model

Usage:
    # Start a trace
    trace_ctx = TraceContext.start_trace("request-123", "retrieve_memories")
    
    # Create nested spans
    with trace_ctx.span("rank_memories") as span:
        # Perform ranking operation
        pass
    
    # Complete the trace
    trace_ctx.end_trace()
    
    # Get all trace events
    events = trace_ctx.get_events()
"""

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from luma.observability.schemas import TraceEvent, create_trace_event


class TraceContext:
    """
    Thread-safe context for tracking request-level traces and nested spans.
    
    TraceContext manages the lifecycle of a distributed trace, including
    the root trace and any nested spans. It uses threading.local to ensure
    thread-safe context management in concurrent environments.
    
    Attributes:
        trace_id: Unique identifier for the entire trace
        events: List of TraceEvent instances recorded during the trace
        _active_spans: Stack of currently active span IDs for nesting
        _lock: Thread lock for ensuring thread-safe operations
    
    Example:
        trace_ctx = TraceContext.start_trace("req-123", "process_request")
        
        with trace_ctx.span("retrieve_memories") as span:
            # Retrieval logic
            pass
        
        with trace_ctx.span("rank_memories") as span:
            # Ranking logic
            pass
        
        trace_ctx.end_trace()
        events = trace_ctx.get_events()
    """
    
    def __init__(self, trace_id: str):
        """
        Initialize a new TraceContext.
        
        Args:
            trace_id: Unique identifier for this trace
        """
        self.trace_id = trace_id
        self.events: List[TraceEvent] = []
        self._active_spans: List[str] = []
        self._lock = threading.Lock()
        self._span_start_times: Dict[str, float] = {}
    
    @classmethod
    def start_trace(
        cls,
        trace_id: str,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TraceContext":
        """
        Start a new trace with a root span.
        
        Args:
            trace_id: Unique identifier for this trace
            operation_name: Name of the root operation
            metadata: Optional metadata for the root span
        
        Returns:
            TraceContext instance with root span started
        """
        context = cls(trace_id)
        root_span_id = f"{trace_id}-root"
        
        with context._lock:
            event = create_trace_event(
                trace_id=trace_id,
                span_id=root_span_id,
                operation_name=operation_name,
                status="pending",
                metadata=metadata
            )
            context.events.append(event)
            context._active_spans.append(root_span_id)
            context._span_start_times[root_span_id] = time.perf_counter()
        
        return context
    
    def start_span(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new nested span within the current trace.
        
        Args:
            operation_name: Name of the operation for this span
            metadata: Optional metadata for the span
        
        Returns:
            Span ID of the newly created span
        """
        span_id = f"{self.trace_id}-{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            parent_span_id = self._active_spans[-1] if self._active_spans else None
            
            event = create_trace_event(
                trace_id=self.trace_id,
                span_id=span_id,
                operation_name=operation_name,
                status="pending",
                parent_span_id=parent_span_id,
                metadata=metadata
            )
            self.events.append(event)
            self._active_spans.append(span_id)
            self._span_start_times[span_id] = time.perf_counter()
        
        return span_id
    
    def _end_span_unlocked(
        self,
        span_id: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End an active span and record its completion (internal method without locking).

        This method assumes the caller already holds the lock.

        Args:
            span_id: ID of the span to end
            status: Final status of the span (default: "success")
            metadata: Optional additional metadata to merge with existing
        """
        # Find the span event
        span_event = None
        for event in self.events:
            if event.span_id == span_id:
                span_event = event
                break

        if span_event is None:
            return

        # Calculate duration
        start_time = self._span_start_times.get(span_id)
        if start_time is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span_event.duration_ms = duration_ms
            del self._span_start_times[span_id]

        # Update span status and metadata
        span_event.status = status
        span_event.end_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        if metadata:
            span_event.metadata.update(metadata)

        # Remove from active spans
        if span_id in self._active_spans:
            self._active_spans.remove(span_id)

    def end_span(
        self,
        span_id: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End an active span and record its completion.

        Args:
            span_id: ID of the span to end
            status: Final status of the span (default: "success")
            metadata: Optional additional metadata to merge with existing
        """
        with self._lock:
            self._end_span_unlocked(span_id, status, metadata)

    
    @contextmanager
    def span(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for creating and automatically ending a span.
        
        This is the recommended way to create spans as it ensures proper
        cleanup even if exceptions occur.
        
        Args:
            operation_name: Name of the operation for this span
            metadata: Optional metadata for the span
        
        Yields:
            Span ID of the created span
        
        Example:
            with trace_ctx.span("retrieve_memories") as span_id:
                # Perform retrieval
                pass
        """
        span_id = self.start_span(operation_name, metadata)
        try:
            yield span_id
            self.end_span(span_id, status="success")
        except Exception as e:
            self.end_span(
                span_id,
                status="error",
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
            raise
    
    def end_trace(
        self,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End the root trace and all active spans.
        
        Args:
            status: Final status of the trace (default: "success")
            metadata: Optional additional metadata for the root span
        """
        with self._lock:
            # End all active spans in reverse order (most nested first)
            for span_id in reversed(self._active_spans[:]):
                self._end_span_unlocked(span_id, status=status, metadata=metadata)
    
    def get_events(self) -> List[TraceEvent]:
        """
        Get all trace events recorded in this context.
        
        Returns:
            List of TraceEvent instances in chronological order
        """
        with self._lock:
            return self.events.copy()
    
    def get_event_by_span_id(self, span_id: str) -> Optional[TraceEvent]:
        """
        Get a specific trace event by span ID.
        
        Args:
            span_id: ID of the span to retrieve
        
        Returns:
            TraceEvent instance if found, None otherwise
        """
        with self._lock:
            for event in self.events:
                if event.span_id == span_id:
                    return event
            return None
    
    def get_active_span_id(self) -> Optional[str]:
        """
        Get the ID of the currently active span.
        
        Returns:
            Span ID of the most recently started span, or None if no spans are active
        """
        with self._lock:
            return self._active_spans[-1] if self._active_spans else None


# Thread-local storage for managing trace contexts per thread
_thread_local = threading.local()


def set_current_trace_context(context: Optional[TraceContext]) -> None:
    """
    Set the current trace context for this thread.
    
    Args:
        context: TraceContext to set as current, or None to clear
    """
    _thread_local.context = context


def get_current_trace_context() -> Optional[TraceContext]:
    """
    Get the current trace context for this thread.
    
    Returns:
        Current TraceContext if set, None otherwise
    """
    return getattr(_thread_local, 'context', None)


@contextmanager
def trace_context(
    trace_id: str,
    operation_name: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for creating and managing a trace context.
    
    This automatically sets the trace context as current for the thread
    and ensures proper cleanup.
    
    Args:
        trace_id: Unique identifier for this trace
        operation_name: Name of the root operation
        metadata: Optional metadata for the root span
    
    Yields:
        TraceContext instance
    
    Example:
        with trace_context("req-123", "process_request") as ctx:
            with ctx.span("retrieve_memories"):
                # Retrieval logic
                pass
    """
    ctx = TraceContext.start_trace(trace_id, operation_name, metadata)
    set_current_trace_context(ctx)
    try:
        yield ctx
        ctx.end_trace(status="success")
    except Exception as e:
        ctx.end_trace(
            status="error",
            metadata={"error": str(e), "error_type": type(e).__name__}
        )
        raise
    finally:
        set_current_trace_context(None)


__all__ = [
    "TraceContext",
    "set_current_trace_context",
    "get_current_trace_context",
    "trace_context",
]
