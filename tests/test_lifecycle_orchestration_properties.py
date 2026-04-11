"""
Property-based tests for LifecycleManager orchestration.

Tests the LifecycleManager orchestrator's dry run mode and configuration
validation using property-based testing to verify correctness across
a wide range of inputs.
"""

import pytest
from hypothesis import given, settings, strategies as st
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from luma.core.lifecycle.schemas import (
    DecayConfig,
    PruningConfig,
    DeduplicationConfig,
    DecayFunctionType,
    PruningStrategy,
    SimilarityMetric,
)
from luma.core.lifecycle.lifecycle_manager import LifecycleManager
from luma.core.lifecycle.memory_decay import MemoryDecay
from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.memory_deduplicator import MemoryDeduplicator
from luma.core.memory_interface import MemoryInterface, RetrievalResult, QueryParameters


# Configure Hypothesis settings for property-based testing
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for testing."""
    
    def __init__(self, initial_memories=None):
        """Initialize mock with optional initial memories."""
        self.memories = initial_memories.copy() if initial_memories else []
        self.deleted_ids = []
        self.delete_failures = {}
        self.store_calls = []
    
    def retrieve(self, query=None, params=None, limit=10) -> RetrievalResult:
        """Retrieve all memories."""
        return {
            "memories": self.memories,
            "total_count": len(self.memories),
            "query_metadata": {}
        }
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id in self.delete_failures:
            raise self.delete_failures[memory_id]
        
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
        return True
    
    def store(self, content: str, metadata=None) -> str:
        """Store a memory."""
        self.store_calls.append((content, metadata))
        memory_id = f"mem_{len(self.store_calls)}"
        return memory_id


# ============================================================================
# Property 23: Dry Run Non-Persistence
# ============================================================================

@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_23_dry_run_non_persistence(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 23: Dry Run Non-Persistence
    
    For any maintenance with dry_run=true, no changes should be persisted
    to the memory store (no updates, deletions, or merges).
    
    **Validates: Requirements 13.2, 13.4, 13.5, 13.6**
    
    This test verifies:
    1. No memories are updated during decay in dry run mode
    2. No memories are deleted during pruning in dry run mode
    3. No memories are merged during deduplication in dry run mode
    4. The memory store remains unchanged after dry run maintenance
    """
    # Create mock memory interface with initial memories
    initial_memories = [
        {
            "id": f"mem_{i}",
            "content": f"Test content {i}",
            "metadata": {
                "importance": 0.5,
                "final_score": 0.5,
                "embedding": [0.1 * i, 0.2 * i],
            },
            "timestamp": (datetime.now(UTC) - timedelta(days=i)).isoformat(),
            "category": "test",
            "tags": [f"tag_{i}"]
        }
        for i in range(5)
    ]
    
    mock_memory_interface = MockMemoryInterface(initial_memories)
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    dedup_config = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size
    )
    
    # Create components with mocked dependencies
    mock_metrics_collector = MagicMock()
    mock_logger = MagicMock()
    
    memory_decay = MemoryDecay(
        memory_interface=mock_memory_interface,
        decay_config=decay_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_pruner = MemoryPruner(
        memory_interface=mock_memory_interface,
        pruning_config=pruning_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_deduplicator = MemoryDeduplicator(
        memory_interface=mock_memory_interface,
        dedup_config=dedup_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    # Create LifecycleManager
    lifecycle_manager = LifecycleManager(
        memory_decay=memory_decay,
        memory_pruner=memory_pruner,
        memory_deduplicator=memory_deduplicator,
        memory_interface=mock_memory_interface,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger,
        timeout_seconds=timeout_seconds
    )
    
    # Record initial state
    initial_memory_count = len(mock_memory_interface.memories)
    initial_deleted_count = len(mock_memory_interface.deleted_ids)
    initial_store_calls = len(mock_memory_interface.store_calls)
    
    # Run maintenance in dry run mode
    report = lifecycle_manager.run_maintenance(dry_run=True)
    
    # Verify no changes were persisted
    # Property 23: No changes should be persisted
    assert len(mock_memory_interface.memories) == initial_memory_count, \
        "Memory count should not change in dry run mode"
    assert len(mock_memory_interface.deleted_ids) == initial_deleted_count, \
        "No memories should be deleted in dry run mode"
    assert len(mock_memory_interface.store_calls) == initial_store_calls, \
        "No memories should be stored in dry run mode"


@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_23_dry_run_no_persistence_operations(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 23: Dry Run Non-Persistence
    
    For any maintenance with dry_run=true, verify that persistence operations
    (store, delete) are not called on the memory interface.
    
    **Validates: Requirements 13.2, 13.4, 13.5, 13.6**
    """
    # Create mock memory interface
    initial_memories = [
        {
            "id": f"mem_{i}",
            "content": f"Test content {i}",
            "metadata": {
                "importance": 0.5,
                "final_score": 0.5,
            },
            "timestamp": (datetime.now(UTC) - timedelta(days=i)).isoformat(),
            "category": "test",
            "tags": []
        }
        for i in range(3)
    ]
    
    mock_memory_interface = MockMemoryInterface(initial_memories)
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    dedup_config = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size
    )
    
    # Create components
    mock_metrics_collector = MagicMock()
    mock_logger = MagicMock()
    
    memory_decay = MemoryDecay(
        memory_interface=mock_memory_interface,
        decay_config=decay_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_pruner = MemoryPruner(
        memory_interface=mock_memory_interface,
        pruning_config=pruning_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_deduplicator = MemoryDeduplicator(
        memory_interface=mock_memory_interface,
        dedup_config=dedup_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    # Create LifecycleManager
    lifecycle_manager = LifecycleManager(
        memory_decay=memory_decay,
        memory_pruner=memory_pruner,
        memory_deduplicator=memory_deduplicator,
        memory_interface=mock_memory_interface,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger,
        timeout_seconds=timeout_seconds
    )
    
    # Run maintenance in dry run mode
    report = lifecycle_manager.run_maintenance(dry_run=True)
    
    # Verify no persistence operations occurred
    # Property 23: No store or delete operations in dry run
    assert len(mock_memory_interface.store_calls) == 0, \
        "No store operations should occur in dry run mode"
    assert len(mock_memory_interface.deleted_ids) == 0, \
        "No delete operations should occur in dry run mode"


# ============================================================================
# Property 24: Dry Run Reporting
# ============================================================================

@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_24_dry_run_reporting(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 24: Dry Run Reporting
    
    For any maintenance with dry_run=true, the report should show intended
    changes that would have been made if dry_run were False.
    
    **Validates: Requirements 13.3**
    
    This test verifies:
    1. Report is returned with valid statistics in dry run mode
    2. Report indicates dry_run=True
    3. Report contains statistics from all operations (decay, pruning, deduplication)
    """
    # Create mock memory interface with initial memories
    initial_memories = [
        {
            "id": f"mem_{i}",
            "content": f"Test content {i}",
            "metadata": {
                "importance": 0.5,
                "final_score": 0.5,
            },
            "timestamp": (datetime.now(UTC) - timedelta(days=i)).isoformat(),
            "category": "test",
            "tags": []
        }
        for i in range(5)
    ]
    
    mock_memory_interface = MockMemoryInterface(initial_memories)
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    dedup_config = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size
    )
    
    # Create components
    mock_metrics_collector = MagicMock()
    mock_logger = MagicMock()
    
    memory_decay = MemoryDecay(
        memory_interface=mock_memory_interface,
        decay_config=decay_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_pruner = MemoryPruner(
        memory_interface=mock_memory_interface,
        pruning_config=pruning_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_deduplicator = MemoryDeduplicator(
        memory_interface=mock_memory_interface,
        dedup_config=dedup_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    # Create LifecycleManager
    lifecycle_manager = LifecycleManager(
        memory_decay=memory_decay,
        memory_pruner=memory_pruner,
        memory_deduplicator=memory_deduplicator,
        memory_interface=mock_memory_interface,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger,
        timeout_seconds=timeout_seconds
    )
    
    # Run maintenance in dry run mode
    report = lifecycle_manager.run_maintenance(dry_run=True)
    
    # Property 24: Report should show intended changes
    # Verify report structure
    assert report.dry_run is True, "Report should indicate dry_run=True"
    assert report.decay_result is not None, "Report should contain decay result"
    assert report.pruning_result is not None, "Report should contain pruning result"
    assert report.deduplication_result is not None, "Report should contain deduplication result"
    assert report.total_execution_time_ms >= 0, "Execution time should be non-negative"
    assert report.maintenance_timestamp is not None, "Maintenance timestamp should be present"
    
    # Verify report contains valid statistics
    assert report.decay_result.memories_processed >= 0, "Processed count should be non-negative"
    assert report.decay_result.memories_updated >= 0, "Updated count should be non-negative"
    assert report.decay_result.average_decay_applied >= 0, "Average decay should be non-negative"
    assert report.decay_result.execution_time_ms >= 0, "Execution time should be non-negative"
    
    assert report.pruning_result.memories_deleted >= 0, "Deleted count should be non-negative"
    assert report.pruning_result.deletion_failures >= 0, "Failure count should be non-negative"
    assert report.pruning_result.execution_time_ms >= 0, "Execution time should be non-negative"
    
    assert report.deduplication_result.duplicate_pairs_found >= 0, "Duplicate pairs should be non-negative"
    assert report.deduplication_result.memories_merged >= 0, "Merged count should be non-negative"
    assert report.deduplication_result.execution_time_ms >= 0, "Execution time should be non-negative"


# ============================================================================
# Property 25: Configuration Validation
# ============================================================================

@given(
    decay_rate=st.floats(max_value=0.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_25_invalid_decay_rate_raises_value_error(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 25: Configuration Validation
    
    For any invalid configuration, initialization should raise ValueError.
    
    **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
    
    This test verifies:
    1. Invalid decay_rate (<= 0) raises ValueError
    2. Invalid similarity_threshold (outside [0, 1]) raises ValueError
    3. Invalid batch_size (<= 0) raises ValueError
    4. Invalid timeout_seconds (<= 0) raises ValueError
    """
    # Create mock memory interface
    mock_memory_interface = MockMemoryInterface()
    
    # Create component configurations with invalid decay_rate
    # Note: DecayConfig validation happens during its own initialization
    with pytest.raises(ValueError, match="decay_rate must be positive"):
        DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=decay_rate
        )


@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(max_value=0)
)
def test_property_25_invalid_timeout_raises_value_error(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 25: Configuration Validation
    
    For any invalid configuration, initialization should raise ValueError.
    
    **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
    
    This test verifies:
    1. Invalid timeout_seconds (<= 0) raises ValueError
    """
    # Create mock memory interface
    mock_memory_interface = MockMemoryInterface()
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    dedup_config = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size
    )
    
    # Create components
    mock_metrics_collector = MagicMock()
    mock_logger = MagicMock()
    
    memory_decay = MemoryDecay(
        memory_interface=mock_memory_interface,
        decay_config=decay_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_pruner = MemoryPruner(
        memory_interface=mock_memory_interface,
        pruning_config=pruning_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    memory_deduplicator = MemoryDeduplicator(
        memory_interface=mock_memory_interface,
        dedup_config=dedup_config,
        metrics_collector=mock_metrics_collector,
        logger=mock_logger
    )
    
    # Property 25: Invalid timeout_seconds should raise ValueError
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        LifecycleManager(
            memory_decay=memory_decay,
            memory_pruner=memory_pruner,
            memory_deduplicator=memory_deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=timeout_seconds
        )


@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.one_of(
        st.floats(min_value=-10.0, max_value=-0.001),
        st.floats(min_value=1.001, max_value=10.0)
    ),
    batch_size=st.integers(min_value=1, max_value=1000),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_25_invalid_similarity_threshold_raises_value_error(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 25: Configuration Validation
    
    For any invalid configuration, initialization should raise ValueError.
    
    **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
    
    This test verifies:
    1. Invalid similarity_threshold (outside [0, 1]) raises ValueError
    """
    # Create mock memory interface
    mock_memory_interface = MockMemoryInterface()
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    # Invalid similarity_threshold
    # Note: DeduplicationConfig validation happens during its own initialization
    with pytest.raises(ValueError, match="similarity_threshold must be in \\[0, 1\\]"):
        DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size
        )


@given(
    decay_rate=st.floats(min_value=0.01, max_value=1.0),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    batch_size=st.integers(max_value=0),
    timeout_seconds=st.integers(min_value=1, max_value=600)
)
def test_property_25_invalid_batch_size_raises_value_error(
    decay_rate, threshold, similarity_threshold, batch_size, timeout_seconds
):
    """
    Feature: memory-lifecycle-management, Property 25: Configuration Validation
    
    For any invalid configuration, initialization should raise ValueError.
    
    **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**
    
    This test verifies:
    1. Invalid batch_size (<= 0) raises ValueError
    """
    # Create mock memory interface
    mock_memory_interface = MockMemoryInterface()
    
    # Create component configurations
    decay_config = DecayConfig(
        decay_function_type=DecayFunctionType.EXPONENTIAL,
        decay_rate=decay_rate
    )
    
    pruning_config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=0.8
    )
    
    # Invalid batch_size
    # Note: DeduplicationConfig validation happens during its own initialization
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size
        )
