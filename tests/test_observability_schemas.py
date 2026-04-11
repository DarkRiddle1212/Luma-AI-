"""
Unit tests for observability data models (schemas.py).

Tests verify that TraceEvent, MetricRecord, and LogEvent models:
- Are properly instantiated with required fields
- Support JSON serialization/deserialization
- Provide convenience factory methods
- Follow consistent naming conventions
"""

import json
from datetime import datetime

from luma.observability.schemas import (
    TraceEvent,
    MetricRecord,
    LogEvent,
    create_trace_event,
    create_log_event,
)


def test_trace_event_creation():
    """Test TraceEvent can be created with required fields."""
    trace = TraceEvent(
        trace_id="trace-123",
        span_id="span-456",
        operation_name="test_operation",
        start_time="2024-01-15T10:30:00.000Z",
        status="pending"
    )
    
    assert trace.trace_id == "trace-123"
    assert trace.span_id == "span-456"
    assert trace.operation_name == "test_operation"
    assert trace.start_time == "2024-01-15T10:30:00.000Z"
    assert trace.status == "pending"
    assert trace.parent_span_id is None
    assert trace.end_time is None
    assert trace.duration_ms is None
    assert trace.metadata == {}


def test_trace_event_with_optional_fields():
    """Test TraceEvent with all optional fields."""
    trace = TraceEvent(
        trace_id="trace-123",
        span_id="span-456",
        operation_name="test_operation",
        start_time="2024-01-15T10:30:00.000Z",
        status="success",
        parent_span_id="span-789",
        end_time="2024-01-15T10:30:01.000Z",
        duration_ms=1000.0,
        metadata={"key": "value"}
    )
    
    assert trace.parent_span_id == "span-789"
    assert trace.end_time == "2024-01-15T10:30:01.000Z"
    assert trace.duration_ms == 1000.0
    assert trace.metadata == {"key": "value"}


def test_trace_event_to_dict():
    """Test TraceEvent serialization to dictionary."""
    trace = TraceEvent(
        trace_id="trace-123",
        span_id="span-456",
        operation_name="test_operation",
        start_time="2024-01-15T10:30:00.000Z",
        status="pending",
        metadata={"count": 5}
    )
    
    data = trace.to_dict()
    
    assert isinstance(data, dict)
    assert data["trace_id"] == "trace-123"
    assert data["span_id"] == "span-456"
    assert data["operation_name"] == "test_operation"
    assert data["start_time"] == "2024-01-15T10:30:00.000Z"
    assert data["status"] == "pending"
    assert data["metadata"] == {"count": 5}


def test_trace_event_to_json():
    """Test TraceEvent serialization to JSON string."""
    trace = TraceEvent(
        trace_id="trace-123",
        span_id="span-456",
        operation_name="test_operation",
        start_time="2024-01-15T10:30:00.000Z",
        status="pending"
    )
    
    json_str = trace.to_json()
    
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["trace_id"] == "trace-123"
    assert data["operation_name"] == "test_operation"


def test_trace_event_from_dict():
    """Test TraceEvent deserialization from dictionary."""
    data = {
        "trace_id": "trace-123",
        "span_id": "span-456",
        "operation_name": "test_operation",
        "start_time": "2024-01-15T10:30:00.000Z",
        "status": "pending",
        "parent_span_id": None,
        "end_time": None,
        "duration_ms": None,
        "metadata": {}
    }
    
    trace = TraceEvent.from_dict(data)
    
    assert trace.trace_id == "trace-123"
    assert trace.span_id == "span-456"
    assert trace.operation_name == "test_operation"


def test_create_trace_event_factory():
    """Test create_trace_event factory function with current timestamp."""
    trace = create_trace_event(
        trace_id="trace-123",
        span_id="span-456",
        operation_name="test_operation",
        status="pending",
        metadata={"key": "value"}
    )
    
    assert trace.trace_id == "trace-123"
    assert trace.span_id == "span-456"
    assert trace.operation_name == "test_operation"
    assert trace.status == "pending"
    assert trace.metadata == {"key": "value"}
    # Verify timestamp is in ISO format with Z suffix
    assert trace.start_time.endswith("Z")
    assert "T" in trace.start_time


def test_metric_record_creation():
    """Test MetricRecord can be created with required fields."""
    record = MetricRecord(
        timestamp="2024-01-15T10:30:00.000Z",
        counters={"total_memories": 100},
        timers={"retrieval_latency_ms": {"count": 10, "sum": 500.0, "min": 10.0, "max": 100.0, "mean": 50.0}}
    )
    
    assert record.timestamp == "2024-01-15T10:30:00.000Z"
    assert record.counters == {"total_memories": 100}
    assert record.timers["retrieval_latency_ms"]["count"] == 10
    assert record.timers["retrieval_latency_ms"]["mean"] == 50.0


def test_metric_record_defaults():
    """Test MetricRecord with default empty collections."""
    record = MetricRecord(timestamp="2024-01-15T10:30:00.000Z")
    
    assert record.counters == {}
    assert record.timers == {}
    assert record.metadata == {}


def test_metric_record_to_dict():
    """Test MetricRecord serialization to dictionary."""
    record = MetricRecord(
        timestamp="2024-01-15T10:30:00.000Z",
        counters={"total_memories": 100},
        timers={"retrieval_latency_ms": {"count": 10, "mean": 50.0}}
    )
    
    data = record.to_dict()
    
    assert isinstance(data, dict)
    assert data["timestamp"] == "2024-01-15T10:30:00.000Z"
    assert data["counters"]["total_memories"] == 100
    assert data["timers"]["retrieval_latency_ms"]["count"] == 10


def test_metric_record_to_json():
    """Test MetricRecord serialization to JSON string."""
    record = MetricRecord(
        timestamp="2024-01-15T10:30:00.000Z",
        counters={"total_memories": 100}
    )
    
    json_str = record.to_json()
    
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["counters"]["total_memories"] == 100


def test_metric_record_from_dict():
    """Test MetricRecord deserialization from dictionary."""
    data = {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "counters": {"total_memories": 100},
        "timers": {},
        "metadata": {}
    }
    
    record = MetricRecord.from_dict(data)
    
    assert record.timestamp == "2024-01-15T10:30:00.000Z"
    assert record.counters["total_memories"] == 100


def test_metric_record_from_snapshot():
    """Test MetricRecord.from_snapshot factory method."""
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
    
    assert record.counters == snapshot["counters"]
    assert record.timers == snapshot["timers"]
    assert record.metadata == {"source": "test"}
    # Verify timestamp is generated
    assert record.timestamp.endswith("Z")
    assert "T" in record.timestamp


def test_log_event_creation():
    """Test LogEvent can be created with required fields."""
    log = LogEvent(
        event="test_event",
        timestamp="2024-01-15T10:30:00.000Z",
        payload={"count": 5}
    )
    
    assert log.event == "test_event"
    assert log.timestamp == "2024-01-15T10:30:00.000Z"
    assert log.payload == {"count": 5}
    assert log.level is None
    assert log.source is None


def test_log_event_with_optional_fields():
    """Test LogEvent with all optional fields."""
    log = LogEvent(
        event="test_event",
        timestamp="2024-01-15T10:30:00.000Z",
        payload={"count": 5},
        level="INFO",
        source="TestComponent"
    )
    
    assert log.level == "INFO"
    assert log.source == "TestComponent"


def test_log_event_to_dict():
    """Test LogEvent serialization to dictionary."""
    log = LogEvent(
        event="test_event",
        timestamp="2024-01-15T10:30:00.000Z",
        payload={"count": 5},
        level="INFO"
    )
    
    data = log.to_dict()
    
    assert isinstance(data, dict)
    assert data["event"] == "test_event"
    assert data["timestamp"] == "2024-01-15T10:30:00.000Z"
    assert data["payload"] == {"count": 5}
    assert data["level"] == "INFO"


def test_log_event_to_json():
    """Test LogEvent serialization to JSON string."""
    log = LogEvent(
        event="test_event",
        timestamp="2024-01-15T10:30:00.000Z",
        payload={"count": 5}
    )
    
    json_str = log.to_json()
    
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["event"] == "test_event"
    assert data["payload"]["count"] == 5


def test_log_event_from_dict():
    """Test LogEvent deserialization from dictionary."""
    data = {
        "event": "test_event",
        "timestamp": "2024-01-15T10:30:00.000Z",
        "payload": {"count": 5},
        "level": "INFO",
        "source": None
    }
    
    log = LogEvent.from_dict(data)
    
    assert log.event == "test_event"
    assert log.timestamp == "2024-01-15T10:30:00.000Z"
    assert log.payload == {"count": 5}
    assert log.level == "INFO"


def test_log_event_from_log_output():
    """Test LogEvent.from_log_output parsing from JSON string."""
    log_output = json.dumps({
        "event": "test_event",
        "timestamp": "2024-01-15T10:30:00.000Z",
        "payload": {"count": 5}
    })
    
    log = LogEvent.from_log_output(log_output)
    
    assert log.event == "test_event"
    assert log.timestamp == "2024-01-15T10:30:00.000Z"
    assert log.payload == {"count": 5}


def test_create_log_event_factory():
    """Test create_log_event factory function with current timestamp."""
    log = create_log_event(
        event="test_event",
        payload={"count": 5},
        level="INFO",
        source="TestComponent"
    )
    
    assert log.event == "test_event"
    assert log.payload == {"count": 5}
    assert log.level == "INFO"
    assert log.source == "TestComponent"
    # Verify timestamp is in ISO format with Z suffix
    assert log.timestamp.endswith("Z")
    assert "T" in log.timestamp


def test_all_models_json_serializable():
    """Test that all models can be serialized to JSON and back."""
    # TraceEvent
    trace = create_trace_event("trace-1", "span-1", "op1")
    trace_json = trace.to_json()
    trace_dict = json.loads(trace_json)
    trace_restored = TraceEvent.from_dict(trace_dict)
    assert trace_restored.trace_id == trace.trace_id
    
    # MetricRecord
    record = MetricRecord.from_snapshot({
        "counters": {"test": 1},
        "timers": {}
    })
    record_json = record.to_json()
    record_dict = json.loads(record_json)
    record_restored = MetricRecord.from_dict(record_dict)
    assert record_restored.counters == record.counters
    
    # LogEvent
    log = create_log_event("test", {"key": "value"})
    log_json = log.to_json()
    log_dict = json.loads(log_json)
    log_restored = LogEvent.from_dict(log_dict)
    assert log_restored.event == log.event
