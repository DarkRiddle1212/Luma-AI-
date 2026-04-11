"""
Property-based tests for MemoryPruner component using Hypothesis.

Tests universal correctness properties for pruning strategies including
protected memory filtering, threshold/percentile/capacity logic, and
deterministic ordering.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional

from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.schemas import (
    PruningConfig,
    PruningStrategy,
    PruningResult,
)
from luma.core.memory_interface import MemoryInterface, RetrievalResult, QueryParameters


# Configure Hypothesis settings for property tests
settings.register_profile("default", max_examples=100, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for property-based testing."""
    
    def __init__(self, initial_memories: Optional[List[Dict[str, Any]]] = None):
        """Initialize mock with optional initial memories."""
        self.memories: List[Dict[str, Any]] = initial_memories.copy() if initial_memories else []
        self.deleted_ids: List[str] = []
        self.delete_failures: Dict[str, Exception] = {}
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
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
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store method (not used in pruner tests)."""
        raise NotImplementedError("Store not used in pruner tests")


# ============================================================================
# Hypothesis Strategies
# ============================================================================

@st.composite
def memory_entry(draw):
    """Generate a single memory entry."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    importance = draw(st.floats(min_value=0.0, max_value=1.0))
    protected = draw(st.booleans())
    
    age_days = draw(st.integers(min_value=0, max_value=365))
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    
    return {
        "id": memory_id,
        "content": "test content",
        "metadata": {
            "importance": importance,
            "protected": protected
        },
        "timestamp": timestamp.isoformat() + "Z",
        "category": "test",
        "tags": []
    }


@st.composite
def memory_collection(draw, min_size=0, max_size=20):
    """Generate a collection of memory entries with unique IDs."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    memories = []
    used_ids = set()
    
    for i in range(count):
        memory = draw(memory_entry())
        # Ensure unique IDs
        while memory["id"] in used_ids:
            memory["id"] = f"{memory['id']}_{i}"
        used_ids.add(memory["id"])
        memories.append(memory)
    
    return memories


# ============================================================================
# Property 6: Protected Memory Filtering
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=20),
    config=st.one_of(
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.THRESHOLD),
            threshold=st.floats(min_value=0.0, max_value=1.0),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        ),
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.PERCENTILE),
            percentile=st.floats(min_value=0.01, max_value=99.99),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        ),
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.CAPACITY),
            capacity_limit=st.integers(min_value=1, max_value=100),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        )
    )
)
def test_property_protected_memory_filtering(memories, config):
    """
    Feature: memory-lifecycle-management, Property 6: Protected Memory Filtering
    
    For any memory set, all memories with protected=true should be excluded
    from deletion regardless of importance or score.
    
    **Validates: Requirements 3.3, 11.2**
    """
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Identify protected memories before pruning
    protected_ids = {
        m["id"] for m in memories
        if m.get("metadata", {}).get("protected", False)
    }
    
    # Run pruning
    result = pruner.prune()
    
    # Verify all protected memories still exist after pruning
    remaining_ids = {m["id"] for m in mock_memory.memories}
    for memory_id in protected_ids:
        assert memory_id in remaining_ids, \
            f"Protected memory {memory_id} should never be deleted"


# ============================================================================
# Property 7: Threshold Pruning Logic
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=20),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_threshold_pruning_logic(memories, threshold, min_importance_protected):
    """
    Feature: memory-lifecycle-management, Property 7: Threshold Pruning Logic
    
    For any memory set and threshold, only memories with importance < threshold
    and protected=false should be deleted.
    
    **Validates: Requirements 3.2, 3.4, 8.1**
    """
    config = PruningConfig(
        strategy=PruningStrategy.THRESHOLD,
        threshold=threshold,
        min_importance_protected=min_importance_protected
    )
    
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Identify memories that should be deleted
    should_be_deleted = set()
    should_be_preserved = set()
    
    for memory in memories:
        importance = memory.get("metadata", {}).get("importance", 1.0)
        protected = memory.get("metadata", {}).get("protected", False)
        
        if importance < threshold and not protected:
            should_be_deleted.add(memory["id"])
        else:
            should_be_preserved.add(memory["id"])
    
    # Run pruning
    result = pruner.prune()
    
    # Verify correct memories were deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    for memory_id in should_be_deleted:
        assert memory_id not in remaining_ids, \
            f"Memory {memory_id} with importance < threshold should be deleted"
    
    for memory_id in should_be_preserved:
        assert memory_id in remaining_ids, \
            f"Memory {memory_id} should be preserved"


# ============================================================================
# Property 8: Percentile Pruning Logic
# ============================================================================

@given(
    memories=memory_collection(min_size=1, max_size=20),  # Need at least 1 memory
    percentile=st.floats(min_value=0.01, max_value=99.99),
    min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_percentile_pruning_logic(memories, percentile, min_importance_protected):
    """
    Feature: memory-lifecycle-management, Property 8: Percentile Pruning Logic
    
    For any memory set and percentile N, the bottom N% of memories by importance
    (excluding protected) should be deleted.
    
    **Validates: Requirements 8.2, 8.5**
    """
    config = PruningConfig(
        strategy=PruningStrategy.PERCENTILE,
        percentile=percentile,
        min_importance_protected=min_importance_protected
    )
    
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Filter out protected memories
    unprotected_memories = [
        m for m in memories
        if not m.get("metadata", {}).get("protected", False)
    ]
    
    # Sort unprotected memories by importance (ascending)
    sorted_unprotected = sorted(
        unprotected_memories,
        key=lambda m: m.get("metadata", {}).get("importance", 1.0)
    )
    
    # Calculate how many to delete
    delete_count = int(len(sorted_unprotected) * percentile / 100.0)
    delete_count = max(1, delete_count)  # At least one memory
    
    # Identify memories that should be deleted
    # Sort by (importance, timestamp, id) for deterministic ordering
    sorted_unprotected = sorted(
        unprotected_memories,
        key=lambda m: (
            m.get("metadata", {}).get("importance", 1.0),
            m.get("timestamp", ""),
            m.get("id", "")
        )
    )
    should_be_deleted = {m["id"] for m in sorted_unprotected[:delete_count]}
    should_be_preserved = {
        m["id"] for m in sorted_unprotected[delete_count:]
    } | {
        m["id"] for m in memories
        if m.get("metadata", {}).get("protected", False)
    }
    
    # Run pruning
    result = pruner.prune()
    
    # Verify correct memories were deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    for memory_id in should_be_deleted:
        assert memory_id not in remaining_ids, \
            f"Memory {memory_id} in bottom {percentile}% should be deleted"
    
    for memory_id in should_be_preserved:
        assert memory_id in remaining_ids, \
            f"Memory {memory_id} should be preserved"


# ============================================================================
# Property 9: Capacity Pruning Logic
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=20),
    capacity_limit=st.integers(min_value=1, max_value=15),
    min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_capacity_pruning_logic(memories, capacity_limit, min_importance_protected):
    """
    Feature: memory-lifecycle-management, Property 9: Capacity Pruning Logic
    
    For any memory set exceeding capacity, the lowest-importance unprotected
    memories should be deleted until count <= capacity, with deterministic
    ordering by (importance, timestamp, id).
    
    **Validates: Requirements 8.3, 8.6**
    """
    config = PruningConfig(
        strategy=PruningStrategy.CAPACITY,
        capacity_limit=capacity_limit,
        min_importance_protected=min_importance_protected
    )
    
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Filter out protected memories
    unprotected_memories = [
        m for m in memories
        if not m.get("metadata", {}).get("protected", False)
    ]
    
    # Sort unprotected memories by (importance, timestamp, id)
    def sort_key(memory):
        importance = memory.get("metadata", {}).get("importance", 1.0)
        timestamp = memory.get("timestamp", "")
        memory_id = memory.get("id", "")
        return (importance, timestamp, memory_id)
    
    sorted_unprotected = sorted(unprotected_memories, key=sort_key)
    
    # Calculate how many to delete
    total_to_delete = max(0, len(memories) - capacity_limit)
    unprotected_to_delete = max(0, len(unprotected_memories) - capacity_limit)
    
    # Identify memories that should be deleted
    # Sort by (importance, timestamp, id) for deterministic ordering
    sorted_unprotected = sorted(
        unprotected_memories,
        key=lambda m: (
            m.get("metadata", {}).get("importance", 1.0),
            m.get("timestamp", ""),
            m.get("id", "")
        )
    )
    should_be_deleted = {m["id"] for m in sorted_unprotected[:unprotected_to_delete]}
    should_be_preserved = {
        m["id"] for m in sorted_unprotected[unprotected_to_delete:]
    } | {
        m["id"] for m in memories
        if m.get("metadata", {}).get("protected", False)
    }
    
    # Run pruning
    result = pruner.prune()
    
    # Verify total count <= capacity OR all remaining are protected
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Check if we need to delete memories
    if len(unprotected_memories) > capacity_limit:
        # If unprotected count exceeds capacity, delete enough to get under limit
        # Verify that all remaining unprotected memories have higher importance than deleted ones
        remaining_unprotected = [
            m for m in mock_memory.memories
            if not m.get("metadata", {}).get("protected", False)
        ]
        assert len(remaining_unprotected) <= capacity_limit, \
            f"Remaining unprotected count {len(remaining_unprotected)} exceeds capacity {capacity_limit}"
    else:
        # Unprotected count is within capacity, no deletion needed
        assert len(mock_memory.memories) == len(memories), \
            f"No memories should be deleted when unprotected count {len(unprotected_memories)} <= capacity {capacity_limit}"
    
    # Verify correct memories were deleted
    for memory_id in should_be_deleted:
        assert memory_id not in remaining_ids, \
            f"Memory {memory_id} should have been deleted to enforce capacity"
    
    for memory_id in should_be_preserved:
        assert memory_id in remaining_ids, \
            f"Memory {memory_id} should be preserved"


# ============================================================================
# Property 10: Deterministic Ordering
# ============================================================================

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    memories=memory_collection(min_size=10, max_size=30),
    capacity_limit=st.integers(min_value=5, max_value=20),
    min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_deterministic_ordering(memories, capacity_limit, min_importance_protected):
    """
    Feature: memory-lifecycle-management, Property 10: Deterministic Ordering
    
    For any memory set requiring capacity pruning, the deletion order should
    be deterministic and stable using importance, timestamp, and id as sort keys.
    
    **Validates: Requirements 8.6**
    """
    config = PruningConfig(
        strategy=PruningStrategy.CAPACITY,
        capacity_limit=capacity_limit,
        min_importance_protected=min_importance_protected
    )
    
    # Run pruning twice with identical input
    mock1 = MockMemoryInterface(memories.copy())
    pruner1 = MemoryPruner(
        memory_interface=mock1,
        pruning_config=config
    )
    result1 = pruner1.prune()
    
    mock2 = MockMemoryInterface(memories.copy())
    pruner2 = MemoryPruner(
        memory_interface=mock2,
        pruning_config=config
    )
    result2 = pruner2.prune()
    
    # Verify identical deletion order
    assert mock1.deleted_ids == mock2.deleted_ids, \
        "Deletion order should be deterministic"
    
    # Verify identical final state
    remaining_ids_1 = {m["id"] for m in mock1.memories}
    remaining_ids_2 = {m["id"] for m in mock2.memories}
    assert remaining_ids_1 == remaining_ids_2, \
        "Final state should be identical"


# ============================================================================
# Property 11: Dry Run Non-Persistence
# ============================================================================

@given(
    memories=memory_collection(min_size=5, max_size=20),
    config=st.one_of(
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.THRESHOLD),
            threshold=st.floats(min_value=0.0, max_value=1.0),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        ),
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.CAPACITY),
            capacity_limit=st.integers(min_value=1, max_value=15),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        )
    )
)
def test_property_dry_run_non_persistence(memories, config):
    """
    Feature: memory-lifecycle-management, Property 11: Dry Run Non-Persistence
    
    For any pruning operation run with dry_run=true, no changes should be
    persisted to the memory store.
    
    **Validates: Requirements 13.5**
    """
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Get initial state
    initial_count = len(mock_memory.memories)
    initial_ids = {m["id"] for m in mock_memory.memories}
    
    # Run pruning in dry_run mode
    result = pruner.prune(dry_run=True)
    
    # Verify no changes were made
    assert len(mock_memory.memories) == initial_count, \
        "Memory count should not change in dry_run mode"
    
    remaining_ids = {m["id"] for m in mock_memory.memories}
    assert remaining_ids == initial_ids, \
        "Memory IDs should not change in dry_run mode"
    
    # Verify delete was never called
    assert len(mock_memory.deleted_ids) == 0, \
        "Delete should not be called in dry_run mode"


# ============================================================================
# Property 12: Pruning Statistics Accuracy
# ============================================================================

@given(
    memories=memory_collection(min_size=5, max_size=20),
    config=st.one_of(
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.THRESHOLD),
            threshold=st.floats(min_value=0.0, max_value=1.0),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        ),
        st.builds(
            PruningConfig,
            strategy=st.just(PruningStrategy.CAPACITY),
            capacity_limit=st.integers(min_value=1, max_value=15),
            min_importance_protected=st.floats(min_value=0.0, max_value=1.0)
        )
    )
)
def test_property_pruning_statistics_accuracy(memories, config):
    """
    Feature: memory-lifecycle-management, Property 12: Pruning Statistics Accuracy
    
    For any pruning operation, the returned statistics should accurately reflect
    the number of memories deleted and failures encountered.
    
    **Validates: Requirements 3.5**
    """
    mock_memory = MockMemoryInterface(memories)
    pruner = MemoryPruner(
        memory_interface=mock_memory,
        pruning_config=config
    )
    
    # Run pruning
    result = pruner.prune()
    
    # Verify statistics are accurate
    assert isinstance(result, PruningResult)
    assert result.memories_deleted >= 0
    assert result.deletion_failures >= 0
    assert len(result.pruned_memories) == result.memories_deleted
    
    # Verify total matches
    assert result.memories_deleted + result.deletion_failures == len(mock_memory.deleted_ids)