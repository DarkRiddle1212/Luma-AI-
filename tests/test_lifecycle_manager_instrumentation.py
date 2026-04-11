"""
Integration tests for MemoryLifecycleManager instrumentation.

This test verifies that the cleanup method correctly records metrics
and logs events when metrics_collector and logger are provided.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, UTC, timedelta
from luma.core.lifecycle_manager import MemoryLifecycleManager
from luma.core.lifecycle_config import LifecycleConfig
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


def test_cleanup_increments_cleanup_runs_counter():
    """Test that cleanup increments cleanup_runs counter when metrics_collector is provided."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create lifecycle manager with metrics collector
    config = LifecycleConfig(max_total_memories=1000)
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        metrics_collector=metrics_collector
    )
    
    # Run cleanup
    manager.cleanup()
    
    # Verify cleanup_runs was incremented
    snapshot = metrics_collector.get_snapshot()
    assert snapshot["counters"]["cleanup_runs"] == 1


def test_cleanup_increments_memories_deleted_total():
    """Test that cleanup increments memories_deleted_total by deletion count."""
    # Create mock memory interface with memories to delete
    old_timestamp = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {
        "memories": [
            {
                "id": "mem1",
                "timestamp": old_timestamp,
                "metadata": {"importance": 0.3, "final_score": 0.5}
            },
            {
                "id": "mem2",
                "timestamp": old_timestamp,
                "metadata": {"importance": 0.4, "final_score": 0.6}
            }
        ]
    }
    mock_memory.delete.return_value = True
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create lifecycle manager with age pruning enabled
    config = LifecycleConfig(
        max_total_memories=1000,
        max_age_days=30,
        min_importance_protected=0.8
    )
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        metrics_collector=metrics_collector
    )
    
    # Run cleanup
    result = manager.cleanup()
    
    # Verify memories_deleted_total was incremented
    snapshot = metrics_collector.get_snapshot()
    assert snapshot["counters"]["memories_deleted_total"] == result.total_deleted
    assert result.total_deleted == 2


def test_cleanup_increments_protected_memories_skipped():
    """Test that cleanup increments protected_memories_skipped by skip count."""
    # Create mock memory interface with protected and unprotected memories
    old_timestamp = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {
        "memories": [
            {
                "id": "mem1",
                "timestamp": old_timestamp,
                "metadata": {"importance": 0.9, "final_score": 0.5}  # Protected
            },
            {
                "id": "mem2",
                "timestamp": old_timestamp,
                "metadata": {"importance": 0.85, "final_score": 0.6}  # Protected
            },
            {
                "id": "mem3",
                "timestamp": old_timestamp,
                "metadata": {"importance": 0.3, "final_score": 0.4}  # Not protected
            }
        ]
    }
    mock_memory.delete.return_value = True
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create lifecycle manager with age pruning enabled
    config = LifecycleConfig(
        max_total_memories=1000,
        max_age_days=30,
        min_importance_protected=0.8
    )
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        metrics_collector=metrics_collector
    )
    
    # Run cleanup
    result = manager.cleanup()
    
    # Verify protected_memories_skipped was incremented (2 protected memories)
    snapshot = metrics_collector.get_snapshot()
    assert snapshot["counters"]["protected_memories_skipped"] == 2
    assert result.total_deleted == 1  # Only 1 unprotected memory deleted


def test_cleanup_records_duration_metric():
    """Test that cleanup records cleanup_duration_ms when metrics_collector is provided."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create lifecycle manager
    config = LifecycleConfig(max_total_memories=1000)
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        metrics_collector=metrics_collector
    )
    
    # Run cleanup
    manager.cleanup()
    
    # Verify cleanup_duration_ms was recorded
    snapshot = metrics_collector.get_snapshot()
    assert "cleanup_duration_ms" in snapshot["timers"]
    assert snapshot["timers"]["cleanup_duration_ms"]["count"] == 1
    assert snapshot["timers"]["cleanup_duration_ms"]["sum"] > 0


def test_cleanup_increments_cleanup_failures_on_exception():
    """Test that cleanup increments cleanup_failures on exceptions in main cleanup block."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    
    # Create lifecycle manager with age pruning enabled
    config = LifecycleConfig(
        max_total_memories=1000,
        max_age_days=30  # Enable age pruning so it gets called
    )
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        metrics_collector=metrics_collector
    )
    
    # Mock the age_pruner to raise an exception that escapes to main cleanup block
    # This simulates an unexpected error in the cleanup logic itself
    manager.age_pruner = Mock()
    manager.age_pruner.prune.side_effect = Exception("Unexpected error")
    
    # Run cleanup (should not raise exception)
    result = manager.cleanup()
    
    # Verify cleanup_failures was incremented
    snapshot = metrics_collector.get_snapshot()
    assert snapshot["counters"]["cleanup_failures"] == 1


def test_cleanup_works_without_metrics_collector():
    """Test that cleanup works correctly when metrics_collector is None."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create lifecycle manager without metrics collector
    config = LifecycleConfig(max_total_memories=1000)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Run cleanup (should not raise exception)
    result = manager.cleanup()
    
    # Verify cleanup completed successfully
    assert result.status.value == "success"


def test_cleanup_works_without_logger():
    """Test that cleanup works correctly when logger is None."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create lifecycle manager without logger
    config = LifecycleConfig(max_total_memories=1000)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Run cleanup (should not raise exception)
    result = manager.cleanup()
    
    # Verify cleanup completed successfully
    assert result.status.value == "success"


def test_cleanup_logs_events_when_logger_provided():
    """Test that cleanup logs events when logger is provided."""
    # Create mock memory interface
    mock_memory = Mock()
    mock_memory.retrieve.return_value = {"memories": []}
    
    # Create mock logger
    mock_logger = Mock(spec=StructuredLogger)
    
    # Create lifecycle manager with logger
    config = LifecycleConfig(max_total_memories=1000)
    manager = MemoryLifecycleManager(
        config, 
        mock_memory, 
        logger=mock_logger
    )
    
    # Run cleanup
    manager.cleanup()
    
    # Verify logger was called
    assert mock_logger.log.call_count >= 2  # At least cleanup_started and cleanup_completed
    
    # Verify cleanup_started event
    first_call = mock_logger.log.call_args_list[0]
    assert first_call[0][0] == "cleanup_started"
    
    # Verify cleanup_completed event
    last_call = mock_logger.log.call_args_list[-1]
    assert last_call[0][0] == "cleanup_completed"


def test_cleanup_behavior_identical_with_and_without_instrumentation():
    """Test that cleanup produces identical results with and without instrumentation."""
    # Create mock memory interface
    old_timestamp = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    memories = [
        {
            "id": "mem1",
            "timestamp": old_timestamp,
            "metadata": {"importance": 0.3, "final_score": 0.5}
        }
    ]
    
    # Test without instrumentation
    mock_memory1 = Mock()
    mock_memory1.retrieve.return_value = {"memories": memories.copy()}
    mock_memory1.delete.return_value = True
    
    config1 = LifecycleConfig(
        max_total_memories=1000,
        max_age_days=30,
        min_importance_protected=0.8
    )
    manager1 = MemoryLifecycleManager(config1, mock_memory1)
    result1 = manager1.cleanup()
    
    # Test with instrumentation
    mock_memory2 = Mock()
    mock_memory2.retrieve.return_value = {"memories": memories.copy()}
    mock_memory2.delete.return_value = True
    
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    
    config2 = LifecycleConfig(
        max_total_memories=1000,
        max_age_days=30,
        min_importance_protected=0.8
    )
    manager2 = MemoryLifecycleManager(
        config2, 
        mock_memory2, 
        metrics_collector=metrics_collector,
        logger=logger
    )
    result2 = manager2.cleanup()
    
    # Verify identical results
    assert result1.total_deleted == result2.total_deleted
    assert result1.failed_deletions == result2.failed_deletions
    assert result1.status == result2.status
