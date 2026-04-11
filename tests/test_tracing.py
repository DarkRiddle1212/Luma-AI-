"""
Unit Tests for Request-Level Tracing Module

This test suite verifies the functionality of the tracing module including:
- TraceContext lifecycle management
- Nested span support
- Thread-safe context management
- Automatic timestamp and duration recording
- Context manager integration
"""

import pytest
import threading
import time
from typing import List

from luma.observability.tracing import (
    TraceContext,
    set_current_trace_context,
    get_current_trace_context,
    trace_context,
)
from luma.observability.schemas import TraceEvent


class TestTraceContext:
    """Test suite for TraceContext class."""
    
    def test_start_trace_creates_root_span(self):
        """Test that starting a trace creates a root span event."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        events = ctx.get_events()
        assert len(events) == 1
        
        root_event = events[0]
        assert root_event.trace_id == "trace-123"
        assert root_event.operation_name == "root_operation"
        assert root_event.status == "pending"
        assert root_event.parent_span_id is None
        assert root_event.span_id == "trace-123-root"
    
    def test_start_trace_with_metadata(self):
        """Test that trace metadata is properly stored."""
        metadata = {"user_id": "user-456", "request_type": "query"}
        ctx = TraceContext.start_trace("trace-123", "root_operation", metadata=metadata)
        
        events = ctx.get_events()
        root_event = events[0]
        assert root_event.metadata == metadata
    
    def test_start_span_creates_nested_span(self):
        """Test that starting a span creates a nested span with parent reference."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span_id = ctx.start_span("child_operation")
        
        events = ctx.get_events()
        assert len(events) == 2
        
        child_event = events[1]
        assert child_event.trace_id == "trace-123"
        assert child_event.operation_name == "child_operation"
        assert child_event.status == "pending"
        assert child_event.parent_span_id == "trace-123-root"
        assert span_id in child_event.span_id
    
    def test_end_span_records_duration_and_status(self):
        """Test that ending a span records duration and updates status."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span_id = ctx.start_span("timed_operation")
        
        time.sleep(0.01)  # Small delay to ensure measurable duration
        ctx.end_span(span_id, status="success")
        
        event = ctx.get_event_by_span_id(span_id)
        assert event is not None
        assert event.status == "success"
        assert event.end_time is not None
        assert event.duration_ms is not None
        assert event.duration_ms > 0

    def test_end_span_with_metadata(self):
        """Test that ending a span can add additional metadata."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span_id = ctx.start_span("operation", metadata={"initial": "value"})
        
        ctx.end_span(span_id, status="success", metadata={"result": "completed"})
        
        event = ctx.get_event_by_span_id(span_id)
        assert event.metadata["initial"] == "value"
        assert event.metadata["result"] == "completed"
    
    def test_span_context_manager_success(self):
        """Test that span context manager properly creates and ends spans."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        with ctx.span("managed_operation") as span_id:
            assert span_id is not None
            event = ctx.get_event_by_span_id(span_id)
            assert event.status == "pending"
        
        # After context manager exits, span should be completed
        event = ctx.get_event_by_span_id(span_id)
        assert event.status == "success"
        assert event.duration_ms is not None
    
    def test_span_context_manager_with_exception(self):
        """Test that span context manager handles exceptions properly."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        with pytest.raises(ValueError):
            with ctx.span("failing_operation") as span_id:
                raise ValueError("Test error")
        
        # Span should be marked as error with exception details
        event = ctx.get_event_by_span_id(span_id)
        assert event.status == "error"
        assert event.metadata["error"] == "Test error"
        assert event.metadata["error_type"] == "ValueError"
    
    def test_nested_spans_hierarchy(self):
        """Test that nested spans maintain proper parent-child relationships."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        with ctx.span("level_1") as span1:
            with ctx.span("level_2") as span2:
                with ctx.span("level_3") as span3:
                    pass
        
        events = ctx.get_events()
        assert len(events) == 4  # root + 3 nested spans
        
        # Verify hierarchy
        event1 = ctx.get_event_by_span_id(span1)
        event2 = ctx.get_event_by_span_id(span2)
        event3 = ctx.get_event_by_span_id(span3)
        
        assert event1.parent_span_id == "trace-123-root"
        assert event2.parent_span_id == span1
        assert event3.parent_span_id == span2
    
    def test_end_trace_completes_all_spans(self):
        """Test that ending a trace completes all active spans."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span1 = ctx.start_span("operation_1")
        span2 = ctx.start_span("operation_2")
        
        ctx.end_trace(status="success")
        
        # All spans should be completed
        for span_id in [span1, span2, "trace-123-root"]:
            event = ctx.get_event_by_span_id(span_id)
            assert event.status == "success"
            assert event.end_time is not None
    
    def test_get_active_span_id(self):
        """Test that get_active_span_id returns the most recent span."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        assert ctx.get_active_span_id() == "trace-123-root"
        
        span1 = ctx.start_span("operation_1")
        assert ctx.get_active_span_id() == span1
        
        span2 = ctx.start_span("operation_2")
        assert ctx.get_active_span_id() == span2
        
        ctx.end_span(span2)
        assert ctx.get_active_span_id() == span1
    
    def test_get_events_returns_copy(self):
        """Test that get_events returns a copy to prevent external modification."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        events1 = ctx.get_events()
        events2 = ctx.get_events()
        
        # Should be equal but not the same object
        assert events1 == events2
        assert events1 is not events2



class TestThreadSafety:
    """Test suite for thread-safe trace context management."""
    
    def test_concurrent_span_creation(self):
        """Test that concurrent span creation is thread-safe."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        span_ids: List[str] = []
        errors: List[Exception] = []
        
        def create_spans():
            try:
                for i in range(10):
                    span_id = ctx.start_span(f"operation_{threading.current_thread().name}_{i}")
                    span_ids.append(span_id)
                    time.sleep(0.001)
                    ctx.end_span(span_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_spans) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
        
        # All spans should be recorded (root + 50 spans from 5 threads * 10 each)
        events = ctx.get_events()
        assert len(events) == 51
        
        # All spans should have unique IDs
        span_ids_from_events = [e.span_id for e in events]
        assert len(span_ids_from_events) == len(set(span_ids_from_events))
    
    def test_concurrent_end_span(self):
        """Test that concurrent end_span calls are thread-safe."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        # Create multiple spans
        span_ids = [ctx.start_span(f"operation_{i}") for i in range(20)]
        errors: List[Exception] = []
        
        def end_spans(ids):
            try:
                for span_id in ids:
                    ctx.end_span(span_id, status="success")
            except Exception as e:
                errors.append(e)
        
        # Split spans across threads
        mid = len(span_ids) // 2
        threads = [
            threading.Thread(target=end_spans, args=(span_ids[:mid],)),
            threading.Thread(target=end_spans, args=(span_ids[mid:],))
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
        
        # All spans should be completed
        for span_id in span_ids:
            event = ctx.get_event_by_span_id(span_id)
            assert event.status == "success"



class TestThreadLocalContext:
    """Test suite for thread-local trace context management."""
    
    def test_set_and_get_current_context(self):
        """Test that current context can be set and retrieved."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        
        set_current_trace_context(ctx)
        retrieved = get_current_trace_context()
        
        assert retrieved is ctx
        assert retrieved.trace_id == "trace-123"
    
    def test_clear_current_context(self):
        """Test that current context can be cleared."""
        ctx = TraceContext.start_trace("trace-123", "root_operation")
        set_current_trace_context(ctx)
        
        set_current_trace_context(None)
        retrieved = get_current_trace_context()
        
        assert retrieved is None
    
    def test_thread_local_isolation(self):
        """Test that trace contexts are isolated per thread."""
        ctx1 = TraceContext.start_trace("trace-1", "operation_1")
        ctx2 = TraceContext.start_trace("trace-2", "operation_2")
        
        results = {}
        
        def thread_func(ctx, thread_id):
            set_current_trace_context(ctx)
            time.sleep(0.01)  # Simulate work
            retrieved = get_current_trace_context()
            results[thread_id] = retrieved
        
        t1 = threading.Thread(target=thread_func, args=(ctx1, "thread1"))
        t2 = threading.Thread(target=thread_func, args=(ctx2, "thread2"))
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Each thread should have its own context
        assert results["thread1"] is ctx1
        assert results["thread2"] is ctx2
        assert results["thread1"].trace_id == "trace-1"
        assert results["thread2"].trace_id == "trace-2"


class TestTraceContextManager:
    """Test suite for trace_context context manager."""
    
    def test_trace_context_manager_success(self):
        """Test that trace_context manager properly creates and completes trace."""
        with trace_context("trace-123", "root_operation") as ctx:
            assert ctx.trace_id == "trace-123"
            assert get_current_trace_context() is ctx
            
            with ctx.span("child_operation"):
                pass
        
        # After context manager exits, trace should be completed
        events = ctx.get_events()
        assert len(events) == 2  # root + child
        
        for event in events:
            assert event.status == "success"
            assert event.end_time is not None
        
        # Context should be cleared
        assert get_current_trace_context() is None
    
    def test_trace_context_manager_with_exception(self):
        """Test that trace_context manager handles exceptions properly."""
        with pytest.raises(ValueError):
            with trace_context("trace-123", "root_operation") as ctx:
                with ctx.span("child_operation"):
                    raise ValueError("Test error")
        
        # Trace should be marked as error
        events = ctx.get_events()
        for event in events:
            assert event.status == "error"
        
        # Context should be cleared even after exception
        assert get_current_trace_context() is None
    
    def test_trace_context_manager_with_metadata(self):
        """Test that trace_context manager supports metadata."""
        metadata = {"request_id": "req-456", "user": "test_user"}
        
        with trace_context("trace-123", "root_operation", metadata=metadata) as ctx:
            pass
        
        events = ctx.get_events()
        root_event = events[0]
        assert root_event.metadata["request_id"] == "req-456"
        assert root_event.metadata["user"] == "test_user"


class TestIntegration:
    """Integration tests for complete tracing workflows."""
    
    def test_complete_request_trace(self):
        """Test a complete request trace with multiple operations."""
        with trace_context("req-123", "process_request") as ctx:
            # Simulate retrieval
            with ctx.span("retrieve_memories", metadata={"query": "test"}) as span1:
                time.sleep(0.01)
            
            # Simulate ranking
            with ctx.span("rank_memories", metadata={"count": 10}) as span2:
                time.sleep(0.01)
            
            # Simulate response generation
            with ctx.span("generate_response") as span3:
                time.sleep(0.01)
        
        events = ctx.get_events()
        assert len(events) == 4  # root + 3 operations
        
        # Verify all operations completed successfully
        for event in events:
            assert event.status == "success"
            assert event.duration_ms is not None
            assert event.duration_ms > 0
        
        # Verify hierarchy
        root_event = events[0]
        assert root_event.operation_name == "process_request"
        assert root_event.parent_span_id is None
        
        for i in range(1, 4):
            assert events[i].parent_span_id == root_event.span_id
    
    def test_nested_operation_hierarchy(self):
        """Test deeply nested operation hierarchy."""
        with trace_context("req-123", "main") as ctx:
            with ctx.span("level_1") as s1:
                with ctx.span("level_2") as s2:
                    with ctx.span("level_3") as s3:
                        with ctx.span("level_4") as s4:
                            pass
        
        events = ctx.get_events()
        assert len(events) == 5
        
        # Verify each level has correct parent
        root_id = events[0].span_id
        assert events[1].parent_span_id == root_id
        assert events[2].parent_span_id == s1
        assert events[3].parent_span_id == s2
        assert events[4].parent_span_id == s3
    
    def test_partial_failure_trace(self):
        """Test trace with partial failures."""
        with trace_context("req-123", "process_request") as ctx:
            with ctx.span("operation_1"):
                pass  # Success
            
            try:
                with ctx.span("operation_2"):
                    raise RuntimeError("Operation failed")
            except RuntimeError:
                pass  # Handle error
            
            with ctx.span("operation_3"):
                pass  # Continue after error
        
        events = ctx.get_events()
        assert len(events) == 4
        
        # First and third operations should succeed
        assert events[1].status == "success"
        assert events[3].status == "success"
        
        # Second operation should fail
        assert events[2].status == "error"
        assert "Operation failed" in events[2].metadata["error"]
