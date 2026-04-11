"""
Data Models for Observability

This module defines structured data models for observability data including
trace events, metric records, and log events. All models use Python dataclasses
to avoid external dependencies and are fully JSON-serializable.

Models:
- TraceEvent: Request-level tracing with timestamps and context
- MetricRecord: Snapshot of metric values with statistics
- LogEvent: Structured log entry with event, timestamp, and payload

All models follow consistent naming conventions (snake_case) and provide
deterministic JSON serialization for integration with external systems.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import json


@dataclass
class TraceEvent:
    """
    Represents a request-level trace event for tracking operation lifecycle.
    
    TraceEvents capture the start and end of operations, allowing for
    distributed tracing and performance analysis across the system.
    
    Attributes:
        trace_id: Unique identifier for the entire trace
        span_id: Unique identifier for this specific span/operation
        parent_span_id: Optional identifier of the parent span for nested operations
        operation_name: Name of the operation being traced
        start_time: ISO 8601 timestamp when the operation started
        end_time: Optional ISO 8601 timestamp when the operation completed
        duration_ms: Optional duration in milliseconds (calculated from start/end)
        status: Status of the operation (e.g., "success", "error", "pending")
        metadata: Optional dictionary containing operation-specific context
    
    Example:
        trace = TraceEvent(
            trace_id="req-123",
            span_id="span-456",
            operation_name="retrieve_memories",
            start_time="2024-01-15T10:30:00.000Z",
            status="pending"
        )
    """
    trace_id: str
    span_id: str
    operation_name: str
    start_time: str
    status: str
    parent_span_id: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TraceEvent to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation with all fields
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert TraceEvent to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        """
        Create TraceEvent from dictionary.
        
        Args:
            data: Dictionary containing trace event fields
        
        Returns:
            TraceEvent instance
        """
        return cls(**data)


@dataclass
class MetricRecord:
    """
    Represents a snapshot of metric values at a point in time.
    
    MetricRecords capture both counter values and timer statistics,
    providing a complete view of system metrics for monitoring and analysis.
    
    Attributes:
        timestamp: ISO 8601 timestamp when the snapshot was taken
        counters: Dictionary mapping counter names to their current values
        timers: Dictionary mapping timer names to their statistics
                (count, sum, min, max, mean)
        metadata: Optional dictionary containing snapshot-specific context
    
    Example:
        record = MetricRecord(
            timestamp="2024-01-15T10:30:00.000Z",
            counters={"total_memories": 100, "cleanup_runs": 5},
            timers={
                "retrieval_latency_ms": {
                    "count": 50,
                    "sum": 2500.0,
                    "min": 10.0,
                    "max": 100.0,
                    "mean": 50.0
                }
            }
        )
    """
    timestamp: str
    counters: Dict[str, float] = field(default_factory=dict)
    timers: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MetricRecord to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation with all fields
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert MetricRecord to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricRecord":
        """
        Create MetricRecord from dictionary.
        
        Args:
            data: Dictionary containing metric record fields
        
        Returns:
            MetricRecord instance
        """
        return cls(**data)
    
    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> "MetricRecord":
        """
        Create MetricRecord from a MetricsCollector snapshot.
        
        This convenience method converts the snapshot format from
        MetricsCollector.get_snapshot() into a MetricRecord with timestamp.
        
        Args:
            snapshot: Snapshot dictionary from MetricsCollector.get_snapshot()
            metadata: Optional metadata to include in the record
        
        Returns:
            MetricRecord instance with current timestamp
        """
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            counters=snapshot.get("counters", {}),
            timers=snapshot.get("timers", {}),
            metadata=metadata or {}
        )


@dataclass
class LogEvent:
    """
    Represents a structured log entry with deterministic schema.
    
    LogEvents provide a consistent format for all log entries, enabling
    programmatic parsing and analysis. The schema matches the output
    format of StructuredLogger.
    
    Attributes:
        event: The event name/type being logged
        timestamp: ISO 8601 timestamp when the event occurred
        payload: Dictionary containing event-specific data
        level: Optional log level (e.g., "INFO", "WARNING", "ERROR")
        source: Optional source component that generated the log
    
    Example:
        log = LogEvent(
            event="retrieval_completed",
            timestamp="2024-01-15T10:30:00.000Z",
            payload={"count": 10, "duration_ms": 50.0},
            level="INFO",
            source="RetrievalLayer"
        )
    """
    event: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    level: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LogEvent to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation with all fields
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert LogEvent to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEvent":
        """
        Create LogEvent from dictionary.
        
        Args:
            data: Dictionary containing log event fields
        
        Returns:
            LogEvent instance
        """
        return cls(**data)
    
    @classmethod
    def from_log_output(cls, log_output: str) -> "LogEvent":
        """
        Parse a LogEvent from StructuredLogger JSON output.
        
        This convenience method parses the JSON output from StructuredLogger
        into a LogEvent instance for programmatic access.
        
        Args:
            log_output: JSON string from StructuredLogger
        
        Returns:
            LogEvent instance
        
        Raises:
            json.JSONDecodeError: If log_output is not valid JSON
            KeyError: If required fields are missing
        """
        data = json.loads(log_output)
        return cls(
            event=data["event"],
            timestamp=data["timestamp"],
            payload=data.get("payload", {}),
            level=data.get("level"),
            source=data.get("source")
        )


# Convenience functions for creating instances with current timestamp

def create_trace_event(
    trace_id: str,
    span_id: str,
    operation_name: str,
    status: str = "pending",
    parent_span_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TraceEvent:
    """
    Create a TraceEvent with current timestamp.
    
    Args:
        trace_id: Unique identifier for the entire trace
        span_id: Unique identifier for this specific span
        operation_name: Name of the operation being traced
        status: Status of the operation (default: "pending")
        parent_span_id: Optional parent span identifier
        metadata: Optional operation-specific context
    
    Returns:
        TraceEvent instance with current timestamp
    """
    return TraceEvent(
        trace_id=trace_id,
        span_id=span_id,
        operation_name=operation_name,
        start_time=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        status=status,
        parent_span_id=parent_span_id,
        metadata=metadata or {}
    )


def create_log_event(
    event: str,
    payload: Optional[Dict[str, Any]] = None,
    level: Optional[str] = None,
    source: Optional[str] = None
) -> LogEvent:
    """
    Create a LogEvent with current timestamp.
    
    Args:
        event: The event name/type being logged
        payload: Optional event-specific data
        level: Optional log level
        source: Optional source component
    
    Returns:
        LogEvent instance with current timestamp
    """
    return LogEvent(
        event=event,
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        payload=payload or {},
        level=level,
        source=source
    )


__all__ = [
    "TraceEvent",
    "MetricRecord",
    "LogEvent",
    "create_trace_event",
    "create_log_event",
]
