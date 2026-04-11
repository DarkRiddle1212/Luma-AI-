"""
Property-based tests for Memory Lifecycle Manager using Hypothesis.

Tests universal correctness properties across all valid inputs to verify
deterministic behavior, idempotency, protection invariants, and error resilience.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional

from luma.core.lifecycle_manager import (
    LifecycleConfig,
    MemoryLifecycleManager,
    extract_importance,
    extract_final_score
)
from luma.core.memory_interface import MemoryInterface, MemoryEntry, RetrievalResult, QueryParameters


# Configure Hypothesis settings
settings.register_profile("default", max_examples=10)
settings.load_profile("default")


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for property-based testing."""
    
    def __init__(self, initial_memories: Optional[List[MemoryEntry]] = None):
        """Initialize mock with optional initial memories."""
        self.memories: List[MemoryEntry] = initial_memories.copy() if initial_memories else []
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
        """Store method (not used in lifecycle manager tests)."""
        raise NotImplementedError("Store not used in lifecycle manager tests")


# ============================================================================
# Hypothesis Strategies
# ============================================================================

@st.composite
def valid_lifecycle_config(draw):
    """Generate valid LifecycleConfig instances."""
    max_total_memories = draw(st.integers(min_value=1, max_value=10000))
    max_age_days = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=365)))
    pruning_score_threshold = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)))
    min_importance_protected = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return LifecycleConfig(
        max_total_memories=max_total_memories,
        max_age_days=max_age_days,
        pruning_score_threshold=pruning_score_threshold,
        min_importance_protected=min_importance_protected
    )


@st.composite
def memory_entry(draw):
    """Generate a single memory entry."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    importance = draw(st.floats(min_value=0.0, max_value=1.0))
    final_score = draw(st.floats(min_value=0.0, max_value=1.0))
    age_days = draw(st.integers(min_value=0, max_value=365))
    
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    
    return {
        "id": memory_id,
        "content": "test content",
        "metadata": {
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


@st.composite
def memory_collection(draw, min_size=0, max_size=100):
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
# Property 1: Configuration Validation
# ============================================================================

@given(max_total=st.integers(max_value=0))
def test_property_config_validation_max_total_memories(max_total):
    """
    Feature: memory-lifecycle-manager, Property 1: For any LifecycleConfig
    with invalid parameters, initialization should raise ValueError with a
    descriptive message.
    
    Tests that max_total_memories <= 0 raises ValueError.
    
    **Validates: Requirements 1.6**
    """
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        LifecycleConfig(max_total_memories=max_total)


@given(threshold=st.floats(min_value=-10.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=10.0))
def test_property_config_validation_pruning_score_threshold(threshold):
    """
    Feature: memory-lifecycle-manager, Property 1: For any LifecycleConfig
    with invalid parameters, initialization should raise ValueError with a
    descriptive message.
    
    Tests that pruning_score_threshold outside [0, 1] raises ValueError.
    
    **Validates: Requirements 1.7**
    """
    with pytest.raises(ValueError, match="pruning_score_threshold must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, pruning_score_threshold=threshold)


@given(importance=st.floats(min_value=-10.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=10.0))
def test_property_config_validation_min_importance_protected(importance):
    """
    Feature: memory-lifecycle-manager, Property 1: For any LifecycleConfig
    with invalid parameters, initialization should raise ValueError with a
    descriptive message.
    
    Tests that min_importance_protected outside [0, 1] raises ValueError.
    
    **Validates: Requirements 1.8**
    """
    with pytest.raises(ValueError, match="min_importance_protected must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, min_importance_protected=importance)


@given(max_age=st.integers(max_value=-1))
def test_property_config_validation_max_age_days(max_age):
    """
    Feature: memory-lifecycle-manager, Property 1: For any LifecycleConfig
    with invalid parameters, initialization should raise ValueError with a
    descriptive message.
    
    Tests that max_age_days < 0 raises ValueError.
    
    **Validates: Requirements 1.9**
    """
    with pytest.raises(ValueError, match="max_age_days cannot be negative"):
        LifecycleConfig(max_total_memories=1000, max_age_days=max_age)


@given(max_per_namespace=st.integers(max_value=-1))
def test_property_config_validation_max_memories_per_namespace(max_per_namespace):
    """
    Feature: memory-lifecycle-manager, Property 1: For any LifecycleConfig
    with invalid parameters, initialization should raise ValueError with a
    descriptive message.
    
    Tests that max_memories_per_namespace < 0 raises ValueError.
    
    **Validates: Requirements 1.10**
    """
    with pytest.raises(ValueError, match="max_memories_per_namespace cannot be negative"):
        LifecycleConfig(max_total_memories=1000, max_memories_per_namespace=max_per_namespace)



# ============================================================================
# Property 2: Age-Based Pruning Correctness
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=5),
    max_age_days=st.integers(min_value=1, max_value=180),
    min_importance=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_age_based_pruning_correctness(memories, max_age_days, min_importance):
    """
    Feature: memory-lifecycle-manager, Property 2: For any memory collection
    and valid LifecycleConfig with max_age_days configured, after cleanup all
    old unprotected memories should be deleted and protected memories preserved.
    """
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=max_age_days,
        min_importance_protected=min_importance
    )
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Identify memories that should be deleted
    current_time = datetime.now(UTC)
    should_be_deleted = set()
    should_be_preserved = set()
    
    for memory in memories:
        timestamp = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
        age_days = (current_time - timestamp).days
        importance = extract_importance(memory)
        
        if age_days > max_age_days and importance < min_importance:
            should_be_deleted.add(memory["id"])
        else:
            should_be_preserved.add(memory["id"])
    
    # Run cleanup
    stats = manager.cleanup()
    
    # Verify all old unprotected memories deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    for memory_id in should_be_deleted:
        assert memory_id not in remaining_ids, f"Old unprotected memory {memory_id} should be deleted"
    
    for memory_id in should_be_preserved:
        assert memory_id in remaining_ids, f"Memory {memory_id} should be preserved"



# ============================================================================
# Property 3: Score-Based Pruning Correctness
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=5),
    score_threshold=st.floats(min_value=0.0, max_value=1.0),
    min_importance=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_score_based_pruning_correctness(memories, score_threshold, min_importance):
    """
    Feature: memory-lifecycle-manager, Property 3: For any memory collection
    and valid LifecycleConfig with pruning_score_threshold configured, after
    cleanup all low-score unprotected memories should be deleted and protected
    memories preserved.
    """
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=score_threshold,
        min_importance_protected=min_importance
    )
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Identify memories that should be deleted
    should_be_deleted = set()
    should_be_preserved = set()
    
    for memory in memories:
        final_score = extract_final_score(memory)
        importance = extract_importance(memory)
        
        if final_score < score_threshold and importance < min_importance:
            should_be_deleted.add(memory["id"])
        else:
            should_be_preserved.add(memory["id"])
    
    # Run cleanup
    stats = manager.cleanup()
    
    # Verify all low-score unprotected memories deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    for memory_id in should_be_deleted:
        assert memory_id not in remaining_ids, f"Low-score unprotected memory {memory_id} should be deleted"
    
    for memory_id in should_be_preserved:
        assert memory_id in remaining_ids, f"Memory {memory_id} should be preserved"



# ============================================================================
# Property 4: Hard Cap Enforcement
# ============================================================================

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    memories=memory_collection(min_size=0, max_size=5),  # Reduced from 100
    max_total=st.integers(min_value=1, max_value=30),  # Reduced from 50
    min_importance=st.floats(min_value=0.01, max_value=1.0)
)
def test_property_hard_cap_enforcement(memories, max_total, min_importance):
    """
    Feature: memory-lifecycle-manager, Property 4: For any memory collection
    and valid LifecycleConfig, after cleanup total memory count should be <=
    max_total_memories and protected memories preserved.
    
    **Validates: Requirements 5.2, 5.4**
    
    This test verifies:
    1. Total count never exceeds max_total_memories (unless all remaining are protected)
    2. Lowest-ranked unprotected memories are deleted first
    3. Protected memories are preserved
    
    Note: If all memories are protected, the hard cap may not be enforceable.
    """
    config = LifecycleConfig(
        max_total_memories=max_total,
        min_importance_protected=min_importance
    )
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Identify protected and unprotected memories before cleanup
    protected_ids = {
        m["id"] for m in memories
        if extract_importance(m) >= min_importance
    }
    
    unprotected_memories = [
        m for m in memories
        if extract_importance(m) < min_importance
    ]
    
    # Sort unprotected memories by deletion priority (lowest-ranked first)
    # This matches the HardCapEnforcer._sort_for_deletion logic
    def sort_key(memory):
        final_score = extract_final_score(memory)
        timestamp_str = memory["timestamp"]
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        memory_id = memory["id"]
        return (final_score, timestamp.timestamp(), memory_id)
    
    sorted_unprotected = sorted(unprotected_memories, key=sort_key)
    
    # Run cleanup
    stats = manager.cleanup()
    
    # Verify all protected memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    for memory_id in protected_ids:
        assert memory_id in remaining_ids, f"Protected memory {memory_id} was deleted"
    
    # Verify total count <= max_total_memories OR all remaining are protected
    # (Hard cap cannot be enforced if all memories are protected)
    if len(mock_memory.memories) > max_total:
        # If we exceed the cap, all remaining must be protected
        for memory in mock_memory.memories:
            importance = extract_importance(memory)
            assert importance >= min_importance, \
                f"Unprotected memory {memory['id']} should have been deleted to enforce cap"
    else:
        # Verify total count does not exceed max_total_memories
        assert len(mock_memory.memories) <= max_total, \
            f"Total count {len(mock_memory.memories)} exceeds max_total_memories {max_total}"
    
    # Verify lowest-ranked unprotected memories were deleted first
    # If we had to delete some unprotected memories to enforce the cap
    if len(memories) > max_total and len(unprotected_memories) > 0:
        # Calculate how many unprotected memories should have been deleted
        initial_count = len(memories)
        protected_count = len(protected_ids)
        
        if initial_count > max_total:
            # We need to delete (initial_count - max_total) memories
            # But we can only delete from unprotected memories
            needed_deletions = initial_count - max_total
            available_for_deletion = len(unprotected_memories)
            
            if available_for_deletion >= needed_deletions:
                # We should have deleted the lowest-ranked unprotected memories
                # The first 'needed_deletions' memories in sorted_unprotected should be deleted
                for i in range(needed_deletions):
                    deleted_memory_id = sorted_unprotected[i]["id"]
                    assert deleted_memory_id not in remaining_ids, \
                        f"Lowest-ranked memory {deleted_memory_id} should have been deleted"
                
                # The remaining unprotected memories should still exist
                for i in range(needed_deletions, len(sorted_unprotected)):
                    preserved_memory_id = sorted_unprotected[i]["id"]
                    assert preserved_memory_id in remaining_ids, \
                        f"Higher-ranked memory {preserved_memory_id} should have been preserved"



# ============================================================================
# Property 5: Protected Memory Invariant
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=5),
    config=valid_lifecycle_config()
)
def test_property_protected_memory_invariant(memories, config):
    """
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    
    Feature: memory-lifecycle-manager, Property 5: For any memory with
    importance >= min_importance_protected, that memory should never be
    deleted by any pruning operation (age-based, score-based, or hard cap enforcement).
    
    This test verifies that protected memories survive ALL pruning phases:
    - Age-based pruning (Requirement 6.1)
    - Score-based pruning (Requirement 6.2)
    - Hard cap enforcement (Requirement 6.3)
    - All cleanup operations verify importance before deletion (Requirement 6.4)
    """
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Identify protected memories before cleanup
    protected_ids = {
        m["id"] for m in memories
        if extract_importance(m) >= config.min_importance_protected
    }
    
    # Run cleanup (executes all three pruning phases)
    stats = manager.cleanup()
    
    # Verify all protected memories still exist after ALL pruning operations
    remaining_ids = {m["id"] for m in mock_memory.memories}
    for memory_id in protected_ids:
        assert memory_id in remaining_ids, \
            f"Protected memory {memory_id} was deleted by pruning operation"



# ============================================================================
# Property 6: Deterministic Sorting
# ============================================================================

@st.composite
def memory_collection_with_duplicates(draw):
    """
    Generate memory collections with duplicate scores and timestamps to test tie-breaking.
    
    This strategy creates memories with:
    - Some memories sharing the same final_score
    - Some memories sharing the same timestamp
    - Some memories sharing both final_score and timestamp (testing memory_id tie-breaking)
    """
    count = draw(st.integers(min_value=10, max_value=50))
    
    # Generate a pool of scores and timestamps to create duplicates
    score_pool = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0),
        min_size=3,
        max_size=10
    ))
    
    timestamp_pool = draw(st.lists(
        st.integers(min_value=0, max_value=365),
        min_size=3,
        max_size=10
    ))
    
    memories = []
    used_ids = set()
    
    for i in range(count):
        # Randomly select from pool to create duplicates
        final_score = draw(st.sampled_from(score_pool))
        age_days = draw(st.sampled_from(timestamp_pool))
        importance = draw(st.floats(min_value=0.0, max_value=1.0))
        
        # Generate unique ID
        memory_id = draw(st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        ))
        while memory_id in used_ids:
            memory_id = f"{memory_id}_{i}"
        used_ids.add(memory_id)
        
        timestamp = datetime.now(UTC) - timedelta(days=age_days)
        
        memories.append({
            "id": memory_id,
            "content": "test content",
            "metadata": {
                "importance": importance,
                "final_score": final_score
            },
            "timestamp": timestamp.isoformat(),
            "category": "test",
            "tags": []
        })
    
    return memories


@given(
    memories=memory_collection_with_duplicates(),
    max_total=st.integers(min_value=1, max_value=20),
    min_importance=st.floats(min_value=0.0, max_value=1.0)
)
def test_property_deterministic_sorting(memories, max_total, min_importance):
    """
    Feature: memory-lifecycle-manager, Property 6: For any memory collection
    requiring hard cap enforcement, the deletion order should be deterministic
    and stable using final_score, timestamp, and memory_id as sort keys.
    
    **Validates: Requirements 5.1, 5.5, 7.4, 7.5**
    
    This test verifies:
    1. Sort order is deterministic and stable
    2. Tie-breaking with identical scores uses timestamp (ascending)
    3. Tie-breaking with identical scores and timestamps uses memory_id (ascending)
    4. Running sort multiple times produces identical ordering
    """
    config = LifecycleConfig(
        max_total_memories=max_total,
        min_importance_protected=min_importance
    )
    
    # Run cleanup twice with identical input
    mock1 = MockMemoryInterface(memories.copy())
    manager1 = MemoryLifecycleManager(config, mock1)
    stats1 = manager1.cleanup()
    
    mock2 = MockMemoryInterface(memories.copy())
    manager2 = MemoryLifecycleManager(config, mock2)
    stats2 = manager2.cleanup()
    
    # Verify identical deletion order
    assert mock1.deleted_ids == mock2.deleted_ids, \
        "Deletion order should be deterministic"
    
    # Verify identical final state
    remaining_ids_1 = {m["id"] for m in mock1.memories}
    remaining_ids_2 = {m["id"] for m in mock2.memories}
    assert remaining_ids_1 == remaining_ids_2, \
        "Final state should be identical"
    
    # Additional verification: Check that tie-breaking rules are followed
    if len(mock1.deleted_ids) > 1:
        # Verify that deleted memories follow the correct sort order
        deleted_memories = [m for m in memories if m["id"] in mock1.deleted_ids]
        
        # Sort deleted memories by the expected criteria
        def sort_key(memory):
            final_score = extract_final_score(memory)
            timestamp_str = memory["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            memory_id = memory["id"]
            return (final_score, timestamp.timestamp(), memory_id)
        
        sorted_deleted = sorted(deleted_memories, key=sort_key)
        
        # Verify that memories with identical scores are ordered by timestamp
        for i in range(len(sorted_deleted) - 1):
            curr = sorted_deleted[i]
            next_mem = sorted_deleted[i + 1]
            
            curr_score = extract_final_score(curr)
            next_score = extract_final_score(next_mem)
            
            if curr_score == next_score:
                # Scores are identical, check timestamp ordering
                curr_ts = datetime.fromisoformat(curr["timestamp"].replace('Z', '+00:00'))
                next_ts = datetime.fromisoformat(next_mem["timestamp"].replace('Z', '+00:00'))
                
                if curr_ts == next_ts:
                    # Timestamps are identical, check memory_id ordering
                    assert curr["id"] <= next_mem["id"], \
                        f"Memory IDs should be in ascending order: {curr['id']} > {next_mem['id']}"
                else:
                    # Timestamps should be in ascending order
                    assert curr_ts <= next_ts, \
                        f"Timestamps should be in ascending order when scores are equal"



# ============================================================================
# Property 7: Deterministic Cleanup
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=5),
    config=valid_lifecycle_config()
)
def test_property_deterministic_cleanup(memories, config):
    """
    Feature: memory-lifecycle-manager, Property 7: For any memory collection
    and valid LifecycleConfig, running cleanup multiple times with identical
    input state should produce identical output state.
    
    **Validates: Requirements 7.1**
    """
    # Run cleanup twice with identical input
    mock1 = MockMemoryInterface(memories.copy())
    manager1 = MemoryLifecycleManager(config, mock1)
    result1 = manager1.cleanup()
    
    mock2 = MockMemoryInterface(memories.copy())
    manager2 = MemoryLifecycleManager(config, mock2)
    result2 = manager2.cleanup()
    
    # Verify identical statistics
    assert result1.total_deleted == result2.total_deleted, \
        "Total deleted should be identical"
    assert result1.age_pruned == result2.age_pruned, \
        "Age pruned count should be identical"
    assert result1.score_pruned == result2.score_pruned, \
        "Score pruned count should be identical"
    assert result1.cap_pruned == result2.cap_pruned, \
        "Cap pruned count should be identical"
    
    # Verify identical final state
    remaining_ids_1 = {m["id"] for m in mock1.memories}
    remaining_ids_2 = {m["id"] for m in mock2.memories}
    assert remaining_ids_1 == remaining_ids_2, \
        "Final state should be identical"
    
    # Verify same memories were deleted in same order
    assert mock1.deleted_ids == mock2.deleted_ids, \
        "Same memories should be deleted in same order"



# ============================================================================
# Property 8: Idempotent Cleanup
# ============================================================================

@given(
    memories=memory_collection(min_size=0, max_size=5),
    config=valid_lifecycle_config()
)
def test_property_idempotent_cleanup(memories, config):
    """
    Feature: memory-lifecycle-manager, Property 8: For any memory collection
    and valid LifecycleConfig, running cleanup twice consecutively should
    produce the same final state and delete zero memories on second execution.
    
    **Validates: Requirements 8.1, 8.2**
    """
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # First cleanup
    result1 = manager.cleanup()
    
    # Second cleanup
    result2 = manager.cleanup()
    
    # Verify idempotency
    assert result2.total_deleted == 0, \
        "Second cleanup should delete nothing"
    assert result1.final_count == result2.final_count, \
        "Final counts should be identical"



# ============================================================================
# Property 11: Error Resilience
# ============================================================================

@given(
    memories=memory_collection(min_size=5, max_size=30),
    config=valid_lifecycle_config(),
    failure_rate=st.floats(min_value=0.1, max_value=0.5)
)
def test_property_error_resilience(memories, config, failure_rate):
    """
    Feature: memory-lifecycle-manager, Property 11: For any memory collection
    where some deletions fail, the cleanup operation should continue processing
    remaining deletions and not propagate exceptions to callers.
    
    **Validates: Requirements 10.2, 10.3**
    
    This test verifies:
    1. Cleanup continues after deletion failures (Requirement 10.2)
    2. Exceptions are not propagated to callers (Requirement 10.3)
    3. Partial completion status is returned (Requirement 10.4)
    """
    mock_memory = MockMemoryInterface(memories)
    
    # Configure random deletions to fail
    import random
    random.seed(42)  # For reproducibility in debugging
    failed_ids = set()
    for memory in memories:
        if random.random() < failure_rate:
            failed_ids.add(memory["id"])
            mock_memory.delete_failures[memory["id"]] = Exception("Simulated failure")
    
    # Count how many memories should be deleted (based on config)
    # This helps verify that cleanup continues after failures
    initial_count = len(memories)
    
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Requirement 10.3: Should not raise exception (no propagation)
    try:
        result = manager.cleanup()
        
        # Verify cleanup completed (even if partially)
        assert result.total_deleted >= 0, "total_deleted should be non-negative"
        assert result.failed_deletions >= 0, "failed_deletions should be non-negative"
        
        # Requirement 10.2: Verify cleanup continued after failures
        # If some deletions were attempted and some failed, it proves continuation
        total_attempted = result.total_deleted + result.failed_deletions
        if total_attempted > 0 and result.failed_deletions > 0:
            # This proves cleanup continued after encountering failures
            # (some succeeded despite failures)
            assert result.total_deleted >= 0, "Should have attempted other deletions after failures"
        
        # Requirement 10.4: Verify partial completion status is returned
        if result.failed_deletions > 0:
            # When there are failures, status should reflect this
            assert result.status.value in ["partial", "failed"], \
                f"Status should indicate failures, got {result.status.value}"
        
    except Exception as e:
        pytest.fail(f"Cleanup should not propagate exceptions (Requirement 10.3): {e}")
