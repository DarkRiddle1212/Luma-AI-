"""
Test instrumentation of SQLiteMemoryAdapter.retrieve method.

This test verifies that the retrieve method correctly records metrics
and logs events when metrics_collector and logger are provided.
"""

import pytest
from unittest.mock import Mock, MagicMock
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


def test_retrieve_records_metrics_when_collector_provided():
    """Test that retrieve records metrics when metrics_collector is provided."""
    # Create mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create real metrics collector
    metrics_collector = MetricsCollector()
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval with metrics_collector
    result = adapter.retrieve(
        query="test query",
        limit=10,
        metrics_collector=metrics_collector
    )
    
    # Verify metrics were recorded
    snapshot = metrics_collector.get_snapshot()
    
    # Check that retrieval_latency_ms was recorded
    assert "retrieval_latency_ms" in snapshot["timers"]
    assert snapshot["timers"]["retrieval_latency_ms"]["count"] == 1
    assert snapshot["timers"]["retrieval_latency_ms"]["sum"] > 0
    
    # Check that retrieval_count was incremented
    assert "retrieval_count" in snapshot["counters"]
    assert snapshot["counters"]["retrieval_count"] == 1


def test_retrieve_logs_event_when_logger_provided():
    """Test that retrieve logs events when logger is provided."""
    # Create mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create mock logger
    mock_logger = Mock(spec=StructuredLogger)
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval with logger
    result = adapter.retrieve(
        query="test query",
        limit=10,
        logger_instance=mock_logger
    )
    
    # Verify logger.log was called
    mock_logger.log.assert_called_once()
    
    # Verify the log call had correct event name
    call_args = mock_logger.log.call_args
    assert call_args[0][0] == "memory_retrieval"
    
    # Verify payload contains expected fields
    payload = call_args[0][1]
    assert "total_count" in payload
    assert "duration_ms" in payload
    assert "filters" in payload


def test_retrieve_works_without_metrics_collector():
    """Test that retrieve works correctly when metrics_collector is None."""
    # Create mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval without metrics_collector (should not raise exception)
    result = adapter.retrieve(query="test query", limit=10)
    
    # Verify result is valid
    assert "memories" in result
    assert "total_count" in result
    assert "query_metadata" in result


def test_retrieve_works_without_logger():
    """Test that retrieve works correctly when logger is None."""
    # Create mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval without logger (should not raise exception)
    result = adapter.retrieve(query="test query", limit=10)
    
    # Verify result is valid
    assert "memories" in result
    assert "total_count" in result
    assert "query_metadata" in result


def test_retrieve_behavior_identical_with_and_without_instrumentation():
    """Test that retrieval results are identical with and without instrumentation."""
    # Create mock memory manager with test data
    mock_memory_manager = Mock()
    
    # Create mock entry
    mock_entry = Mock()
    mock_entry.id = "mem_123"
    mock_entry.action = "test content"
    mock_entry.context = {"category": "test"}
    mock_entry.created_at = None
    mock_entry.timestamp = Mock()
    mock_entry.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
    mock_entry.tags = ["tag1", "tag2"]
    
    mock_memory_manager.query_memories.return_value = [mock_entry]
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval without instrumentation
    result_without = adapter.retrieve(query="test", limit=5)
    
    # Reset mock
    mock_memory_manager.reset_mock()
    mock_memory_manager.query_memories.return_value = [mock_entry]
    
    # Execute retrieval with instrumentation
    metrics_collector = MetricsCollector()
    mock_logger = Mock(spec=StructuredLogger)
    result_with = adapter.retrieve(
        query="test",
        limit=5,
        metrics_collector=metrics_collector,
        logger_instance=mock_logger
    )
    
    # Verify results are identical (excluding query_metadata.execution_time_ms which may vary)
    assert result_without["memories"] == result_with["memories"]
    assert result_without["total_count"] == result_with["total_count"]
    assert result_without["query_metadata"]["limit"] == result_with["query_metadata"]["limit"]
    assert result_without["query_metadata"]["has_more"] == result_with["query_metadata"]["has_more"]
    assert result_without["query_metadata"]["filters_applied"] == result_with["query_metadata"]["filters_applied"]


def test_retrieve_increments_counter_multiple_times():
    """Test that multiple retrievals increment the counter correctly."""
    # Create mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create real metrics collector
    metrics_collector = MetricsCollector()
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute multiple retrievals
    for i in range(5):
        adapter.retrieve(
            query=f"test query {i}",
            limit=10,
            metrics_collector=metrics_collector
        )
    
    # Verify counter was incremented 5 times
    snapshot = metrics_collector.get_snapshot()
    assert snapshot["counters"]["retrieval_count"] == 5
    
    # Verify 5 latency measurements were recorded
    assert snapshot["timers"]["retrieval_latency_ms"]["count"] == 5
