"""
Unit tests for Memory Lifecycle Manager namespace isolation.

Tests namespace isolation, independent cap enforcement, and aggregation
according to Requirements 9.1, 9.2, 9.3, 9.4, and 13.1.
"""

import pytest
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Any

from luma.core.lifecycle_config import LifecycleConfig
from luma.core.lifecycle_manager import MemoryLifecycleManager
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for testing."""
    
    def __init__(self, initial_memories: List[MemoryEntry]):
        """Initialize with a list of memory entries."""
        self.memories = initial_memories.copy()
        self.deleted_ids: List[str] = []
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Retrieve memories from the mock storage."""
        return {
            "memories": self.memories,
            "total_count": len(self.memories),
            "query_metadata": {}
        }
    
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a new memory (not used in lifecycle tests)."""
        raise NotImplementedError("Store not needed for lifecycle tests")



def create_memory(
    memory_id: str,
    namespace: str,
    age_days: int = 30,
    importance: float = 0.0,
    final_score: float = 0.5
) -> MemoryEntry:
    """
    Helper function to create a memory entry with specified namespace.
    
    Args:
        memory_id: Unique identifier for the memory
        namespace: Namespace for the memory
        age_days: Age of the memory in days (0 = today, positive = past)
        importance: Importance score [0, 1]
        final_score: Final relevance score [0, 1]
    
    Returns:
        MemoryEntry with specified properties
    """
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    return {
        "id": memory_id,
        "content": f"Memory content {memory_id}",
        "metadata": {
            "namespace": namespace,
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


# ============================================================================
# Namespace Cap Enforcement Tests - Requirements 9.1, 9.2, 9.4, 13.1
# ============================================================================

def test_namespace_caps_enforced_independently():
    """
    Test that namespace caps are enforced independently per namespace.
    
    Validates: Requirement 9.1
    
    Setup:
    - Create multiple namespaces, each exceeding max_memories_per_namespace
    - Each namespace has different memory counts
    
    Expected:
    - Each namespace should be pruned to max_memories_per_namespace independently
    - Pruning in one namespace should not affect other namespaces
    """
    # Setup: Create memories in multiple namespaces
    memories = [
        # Namespace "work" - 5 memories (should prune to 3)
        create_memory("work_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_3", "work", importance=0.5, final_score=0.3),
        create_memory("work_4", "work", importance=0.5, final_score=0.4),
        create_memory("work_5", "work", importance=0.5, final_score=0.5),
        
        # Namespace "personal" - 4 memories (should prune to 3)
        create_memory("personal_1", "personal", importance=0.5, final_score=0.1),
        create_memory("personal_2", "personal", importance=0.5, final_score=0.2),
        create_memory("personal_3", "personal", importance=0.5, final_score=0.3),
        create_memory("personal_4", "personal", importance=0.5, final_score=0.4),
        
        # Namespace "system" - 2 memories (should not prune)
        create_memory("system_1", "system", importance=0.5, final_score=0.1),
        create_memory("system_2", "system", importance=0.5, final_score=0.2),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Each namespace pruned independently
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Work namespace: should have 3 memories (highest scores: 0.3, 0.4, 0.5)
    work_remaining = {id for id in remaining_ids if id.startswith("work_")}
    assert len(work_remaining) == 3, f"Work namespace should have 3 memories, got {len(work_remaining)}"
    assert work_remaining == {"work_3", "work_4", "work_5"}, "Work namespace should keep highest scored memories"
    
    # Personal namespace: should have 3 memories (highest scores: 0.2, 0.3, 0.4)
    personal_remaining = {id for id in remaining_ids if id.startswith("personal_")}
    assert len(personal_remaining) == 3, f"Personal namespace should have 3 memories, got {len(personal_remaining)}"
    assert personal_remaining == {"personal_2", "personal_3", "personal_4"}, "Personal namespace should keep highest scored memories"
    
    # System namespace: should have 2 memories (no pruning needed)
    system_remaining = {id for id in remaining_ids if id.startswith("system_")}
    assert len(system_remaining) == 2, f"System namespace should have 2 memories, got {len(system_remaining)}"
    assert system_remaining == {"system_1", "system_2"}, "System namespace should keep all memories"
    
    # Verify total
    assert len(remaining_ids) == 8, f"Total should be 8 memories, got {len(remaining_ids)}"
    
    # Verify statistics
    # Note: namespace cap deletions are tracked separately in the implementation
    # but included in total_deleted
    assert result.total_deleted == 3, "Should have deleted 3 memories (2 from work, 1 from personal)"



def test_pruning_in_one_namespace_does_not_affect_others():
    """
    Test that pruning in one namespace doesn't affect memories in other namespaces.
    
    Validates: Requirement 9.2
    
    Setup:
    - Create memories in multiple namespaces
    - Only one namespace exceeds the cap
    
    Expected:
    - Only the exceeding namespace should be pruned
    - Other namespaces should remain untouched
    - Exact memory_id verification
    """
    # Setup: Create memories where only "work" namespace exceeds cap
    memories = [
        # Namespace "work" - 5 memories (exceeds cap of 3)
        create_memory("work_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_3", "work", importance=0.5, final_score=0.3),
        create_memory("work_4", "work", importance=0.5, final_score=0.4),
        create_memory("work_5", "work", importance=0.5, final_score=0.5),
        
        # Namespace "personal" - 2 memories (under cap)
        create_memory("personal_1", "personal", importance=0.5, final_score=0.1),
        create_memory("personal_2", "personal", importance=0.5, final_score=0.2),
        
        # Namespace "system" - 1 memory (under cap)
        create_memory("system_1", "system", importance=0.5, final_score=0.1),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only work namespace affected
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Work namespace: should be pruned to 3
    work_remaining = {id for id in remaining_ids if id.startswith("work_")}
    assert len(work_remaining) == 3, "Work namespace should be pruned to 3"
    assert work_remaining == {"work_3", "work_4", "work_5"}, "Work namespace should keep highest scored memories"
    
    # Personal namespace: should be completely untouched
    personal_remaining = {id for id in remaining_ids if id.startswith("personal_")}
    assert personal_remaining == {"personal_1", "personal_2"}, "Personal namespace should be untouched"
    
    # System namespace: should be completely untouched
    system_remaining = {id for id in remaining_ids if id.startswith("system_")}
    assert system_remaining == {"system_1"}, "System namespace should be untouched"
    
    # Verify deleted memories are only from work namespace
    deleted_ids = set(mock_memory.deleted_ids)
    assert deleted_ids == {"work_1", "work_2"}, "Only work namespace memories should be deleted"
    
    # Verify statistics
    assert result.total_deleted == 2, "Should have deleted 2 memories from work namespace only"



def test_total_count_aggregates_across_all_namespaces():
    """
    Test that total memory count aggregates across all namespaces.
    
    Validates: Requirement 9.3
    
    Setup:
    - Create memories in multiple namespaces
    - Total exceeds max_total_memories but each namespace is under per-namespace cap
    
    Expected:
    - Total count should be sum of all namespace counts
    - Hard cap enforcement should consider all namespaces
    """
    # Setup: Create memories across namespaces
    memories = [
        # Namespace "work" - 3 memories
        create_memory("work_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_3", "work", importance=0.5, final_score=0.3),
        
        # Namespace "personal" - 3 memories
        create_memory("personal_1", "personal", importance=0.5, final_score=0.1),
        create_memory("personal_2", "personal", importance=0.5, final_score=0.2),
        create_memory("personal_3", "personal", importance=0.5, final_score=0.3),
        
        # Namespace "system" - 3 memories
        create_memory("system_1", "system", importance=0.5, final_score=0.1),
        create_memory("system_2", "system", importance=0.5, final_score=0.2),
        create_memory("system_3", "system", importance=0.5, final_score=0.3),
    ]
    
    # Total: 9 memories, max_total: 7, per_namespace: 5
    # Each namespace is under per-namespace cap (3 < 5)
    # But total exceeds hard cap (9 > 7), so 2 should be deleted
    config = LifecycleConfig(
        max_total_memories=7,
        max_memories_per_namespace=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Total count is enforced across all namespaces
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Total should be 7 (hard cap)
    assert len(remaining_ids) == 7, f"Total should be 7 memories, got {len(remaining_ids)}"
    
    # Verify that lowest scored memories were deleted (work_1 and personal_1 both have score 0.1)
    # Due to deterministic sorting, we need to check which were deleted
    deleted_ids = set(mock_memory.deleted_ids)
    assert len(deleted_ids) == 2, "Should have deleted 2 memories"
    
    # All deleted memories should have the lowest scores
    for deleted_id in deleted_ids:
        assert deleted_id in ["work_1", "personal_1", "system_1"], \
            f"Deleted memory {deleted_id} should be one of the lowest scored"
    
    # Verify statistics
    assert result.cap_pruned == 2, "Should have deleted 2 memories by hard cap"
    assert result.final_count == 7, "Final count should be 7"



def test_memories_do_not_leak_between_namespaces():
    """
    Test that memories don't leak between namespaces during pruning.
    
    Validates: Requirement 9.4
    
    Setup:
    - Create memories in multiple namespaces with identical IDs (different namespaces)
    - Trigger pruning in one namespace
    
    Expected:
    - Only memories in the pruned namespace should be affected
    - Memories with same ID pattern in other namespaces should remain
    - Exact memory_id set verification per namespace
    """
    # Setup: Create memories with similar patterns across namespaces
    memories = [
        # Namespace "work" - 5 memories (will be pruned)
        create_memory("mem_1", "work", importance=0.5, final_score=0.1),
        create_memory("mem_2", "work", importance=0.5, final_score=0.2),
        create_memory("mem_3", "work", importance=0.5, final_score=0.3),
        create_memory("mem_4", "work", importance=0.5, final_score=0.4),
        create_memory("mem_5", "work", importance=0.5, final_score=0.5),
        
        # Namespace "personal" - 5 memories (will be pruned)
        create_memory("mem_6", "personal", importance=0.5, final_score=0.1),
        create_memory("mem_7", "personal", importance=0.5, final_score=0.2),
        create_memory("mem_8", "personal", importance=0.5, final_score=0.3),
        create_memory("mem_9", "personal", importance=0.5, final_score=0.4),
        create_memory("mem_10", "personal", importance=0.5, final_score=0.5),
        
        # Namespace "system" - 2 memories (will not be pruned)
        create_memory("mem_11", "system", importance=0.5, final_score=0.1),
        create_memory("mem_12", "system", importance=0.5, final_score=0.2),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact memory_id sets per namespace
    remaining_memories = mock_memory.memories
    
    # Group remaining memories by namespace
    work_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "work"}
    personal_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "personal"}
    system_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "system"}
    
    # Work namespace: should have exactly 3 highest scored memories
    assert work_ids == {"mem_3", "mem_4", "mem_5"}, \
        f"Work namespace should have mem_3, mem_4, mem_5, got {work_ids}"
    
    # Personal namespace: should have exactly 3 highest scored memories
    assert personal_ids == {"mem_8", "mem_9", "mem_10"}, \
        f"Personal namespace should have mem_8, mem_9, mem_10, got {personal_ids}"
    
    # System namespace: should have all 2 memories (no pruning)
    assert system_ids == {"mem_11", "mem_12"}, \
        f"System namespace should have mem_11, mem_12, got {system_ids}"
    
    # Verify no cross-namespace contamination
    all_remaining_ids = work_ids | personal_ids | system_ids
    assert len(all_remaining_ids) == 8, "Total should be 8 memories"
    
    # Verify deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {"mem_1", "mem_2", "mem_6", "mem_7"}
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.total_deleted == 4, "Should have deleted 4 memories (2 from work, 2 from personal)"



def test_exact_memory_id_sets_per_namespace():
    """
    Test exact memory_id sets per namespace after pruning.
    
    Validates: Requirements 9.1, 9.2, 9.4
    
    Setup:
    - Create a comprehensive multi-namespace scenario
    - Mix of protected and unprotected memories
    - Different cap violations per namespace
    
    Expected:
    - Exact verification of remaining memory IDs per namespace
    - No weak assertions (count-only checks)
    - Protected memories preserved in all namespaces
    """
    # Setup: Comprehensive multi-namespace scenario
    memories = [
        # Namespace "work" - 6 memories (cap: 3)
        create_memory("work_low_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_low_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_mid", "work", importance=0.5, final_score=0.3),
        create_memory("work_high_1", "work", importance=0.5, final_score=0.4),
        create_memory("work_high_2", "work", importance=0.5, final_score=0.5),
        create_memory("work_protected", "work", importance=0.9, final_score=0.05),  # Protected despite low score
        
        # Namespace "personal" - 4 memories (cap: 3)
        create_memory("personal_low", "personal", importance=0.5, final_score=0.1),
        create_memory("personal_mid", "personal", importance=0.5, final_score=0.3),
        create_memory("personal_high", "personal", importance=0.5, final_score=0.5),
        create_memory("personal_protected", "personal", importance=0.8, final_score=0.05),  # Protected
        
        # Namespace "system" - 2 memories (cap: 3, no pruning needed)
        create_memory("system_1", "system", importance=0.5, final_score=0.1),
        create_memory("system_2", "system", importance=0.5, final_score=0.2),
        
        # Namespace "default" - 1 memory (cap: 3, no pruning needed)
        create_memory("default_1", "default", importance=0.5, final_score=0.1),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact memory_id sets per namespace
    remaining_memories = mock_memory.memories
    
    # Group by namespace
    work_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "work"}
    personal_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "personal"}
    system_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "system"}
    default_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "default"}
    
    # Work namespace: 6 total (5 unprotected + 1 protected), cap is 3
    # Need to delete 3 memories. Protected memory is excluded from deletion.
    # So we delete 3 lowest unprotected: work_low_1, work_low_2, work_mid
    # Remaining: work_high_1, work_high_2, work_protected (3 total)
    expected_work = {"work_protected", "work_high_2", "work_high_1"}
    assert work_ids == expected_work, \
        f"Work namespace expected {expected_work}, got {work_ids}"
    
    # Personal namespace: 4 total (3 unprotected + 1 protected), cap is 3
    # Need to delete 1 memory. Protected memory is excluded from deletion.
    # So we delete 1 lowest unprotected: personal_low
    # Remaining: personal_mid, personal_high, personal_protected (3 total)
    expected_personal = {"personal_protected", "personal_high", "personal_mid"}
    assert personal_ids == expected_personal, \
        f"Personal namespace expected {expected_personal}, got {personal_ids}"
    
    # System namespace: all memories (under cap)
    expected_system = {"system_1", "system_2"}
    assert system_ids == expected_system, \
        f"System namespace expected {expected_system}, got {system_ids}"
    
    # Default namespace: all memories (under cap)
    expected_default = {"default_1"}
    assert default_ids == expected_default, \
        f"Default namespace expected {expected_default}, got {default_ids}"
    
    # Verify total
    all_remaining = work_ids | personal_ids | system_ids | default_ids
    assert len(all_remaining) == 9, f"Total should be 9 memories, got {len(all_remaining)}"
    
    # Verify deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {"work_low_1", "work_low_2", "work_mid", "personal_low"}
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"



def test_namespace_isolation_with_default_namespace():
    """
    Test namespace isolation when some memories have no explicit namespace.
    
    Validates: Requirements 9.1, 9.2, 9.4
    
    Setup:
    - Create memories with explicit namespaces
    - Create memories without namespace (should default to "default")
    
    Expected:
    - Memories without namespace should be grouped in "default" namespace
    - Default namespace should be pruned independently
    """
    # Setup: Mix of explicit and default namespaces
    memories = [
        # Explicit namespace "work" - 4 memories
        create_memory("work_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_3", "work", importance=0.5, final_score=0.3),
        create_memory("work_4", "work", importance=0.5, final_score=0.4),
    ]
    
    # Add memories without explicit namespace (will default to "default")
    for i in range(1, 5):
        timestamp = datetime.now(UTC) - timedelta(days=30)
        memories.append({
            "id": f"default_{i}",
            "content": f"Memory content default_{i}",
            "metadata": {
                "importance": 0.5,
                "final_score": 0.1 * i
            },
            "timestamp": timestamp.isoformat(),
            "category": "test",
            "tags": []
        })
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Both namespaces pruned independently
    remaining_memories = mock_memory.memories
    
    # Group by namespace
    work_ids = {m["id"] for m in remaining_memories if m["metadata"].get("namespace") == "work"}
    default_ids = {m["id"] for m in remaining_memories if m["metadata"].get("namespace") != "work"}
    
    # Work namespace: should have 3 highest scored
    assert work_ids == {"work_2", "work_3", "work_4"}, \
        f"Work namespace should have 3 highest scored, got {work_ids}"
    
    # Default namespace: should have 3 highest scored
    assert default_ids == {"default_2", "default_3", "default_4"}, \
        f"Default namespace should have 3 highest scored, got {default_ids}"
    
    # Verify total
    assert len(remaining_memories) == 6, "Total should be 6 memories"
    
    # Verify deleted
    deleted_ids = set(mock_memory.deleted_ids)
    assert deleted_ids == {"work_1", "default_1"}, \
        f"Should delete lowest from each namespace, got {deleted_ids}"



def test_namespace_cap_with_protected_memories():
    """
    Test that protected memories are preserved during namespace cap enforcement.
    
    Validates: Requirements 9.1, 6.3 (importance protection during namespace cap)
    
    Setup:
    - Create namespace exceeding cap with mix of protected and unprotected
    
    Expected:
    - Protected memories should not count against namespace cap
    - Only unprotected memories should be pruned
    """
    # Setup: Namespace with protected and unprotected memories
    memories = [
        # Namespace "work" - 7 memories (cap: 3)
        # 3 protected + 4 unprotected
        create_memory("work_protected_1", "work", importance=0.9, final_score=0.1),
        create_memory("work_protected_2", "work", importance=0.8, final_score=0.2),
        create_memory("work_protected_3", "work", importance=1.0, final_score=0.05),
        create_memory("work_unprotected_1", "work", importance=0.5, final_score=0.3),
        create_memory("work_unprotected_2", "work", importance=0.5, final_score=0.4),
        create_memory("work_unprotected_3", "work", importance=0.5, final_score=0.5),
        create_memory("work_unprotected_4", "work", importance=0.5, final_score=0.6),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories preserved, cap enforced on total
    # With 7 memories (3 protected + 4 unprotected) and cap of 3:
    # Need to delete 4 memories, but can only delete unprotected
    # So delete all 4 unprotected, leaving only 3 protected
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # All protected memories should remain
    assert "work_protected_1" in remaining_ids, "Protected memory 1 should be preserved"
    assert "work_protected_2" in remaining_ids, "Protected memory 2 should be preserved"
    assert "work_protected_3" in remaining_ids, "Protected memory 3 should be preserved"
    
    # All unprotected should be deleted (to meet the cap of 3)
    assert "work_unprotected_1" not in remaining_ids, "Unprotected 1 should be deleted"
    assert "work_unprotected_2" not in remaining_ids, "Unprotected 2 should be deleted"
    assert "work_unprotected_3" not in remaining_ids, "Unprotected 3 should be deleted"
    assert "work_unprotected_4" not in remaining_ids, "Unprotected 4 should be deleted"
    
    # Verify exact set - only protected memories remain
    expected_remaining = {
        "work_protected_1", "work_protected_2", "work_protected_3"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify statistics
    assert result.total_deleted == 4, "Should have deleted 4 unprotected memories"



def test_namespace_cap_disabled_when_none():
    """
    Test that namespace cap enforcement is skipped when max_memories_per_namespace is None.
    
    Validates: Requirement 9.1 (implicit - namespace cap is optional)
    
    Setup:
    - Create namespaces with many memories
    - Configure with max_memories_per_namespace=None
    
    Expected:
    - No namespace-based pruning should occur
    - All memories should be preserved (assuming no other pruning rules)
    """
    # Setup: Multiple namespaces with many memories
    memories = [
        create_memory("work_1", "work", importance=0.5, final_score=0.1),
        create_memory("work_2", "work", importance=0.5, final_score=0.2),
        create_memory("work_3", "work", importance=0.5, final_score=0.3),
        create_memory("work_4", "work", importance=0.5, final_score=0.4),
        create_memory("work_5", "work", importance=0.5, final_score=0.5),
        create_memory("personal_1", "personal", importance=0.5, final_score=0.1),
        create_memory("personal_2", "personal", importance=0.5, final_score=0.2),
        create_memory("personal_3", "personal", importance=0.5, final_score=0.3),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=None,  # Namespace cap disabled
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    expected_all = {
        "work_1", "work_2", "work_3", "work_4", "work_5",
        "personal_1", "personal_2", "personal_3"
    }
    assert remaining_ids == expected_all, \
        f"All memories should be preserved when namespace cap is None"
    
    # Verify statistics
    # When namespace cap is None, no namespace-based deletions occur
    # The cap_pruned field only tracks global hard cap deletions
    assert result.total_deleted == 0, "Total deleted should be 0"


def test_namespace_isolation_with_combined_pruning():
    """
    Test namespace isolation when combined with age and score pruning.
    
    Validates: Requirements 9.1, 9.2, 9.4 (namespace isolation with other pruning rules)
    
    Setup:
    - Create memories in multiple namespaces
    - Some memories are old (age pruning)
    - Some memories have low scores (score pruning)
    - Some namespaces exceed cap (namespace pruning)
    
    Expected:
    - All pruning rules should respect namespace boundaries
    - Age/score pruning should not affect namespace isolation
    """
    # Setup: Complex scenario with multiple pruning triggers
    memories = [
        # Namespace "work" - mix of old, low-score, and normal
        create_memory("work_old", "work", age_days=100, importance=0.5, final_score=0.5),  # Old, will be deleted
        create_memory("work_low_score", "work", age_days=30, importance=0.5, final_score=0.1),  # Low score, will be deleted
        create_memory("work_good_1", "work", age_days=30, importance=0.5, final_score=0.5),
        create_memory("work_good_2", "work", age_days=30, importance=0.5, final_score=0.6),
        create_memory("work_good_3", "work", age_days=30, importance=0.5, final_score=0.7),
        
        # Namespace "personal" - similar mix
        create_memory("personal_old", "personal", age_days=100, importance=0.5, final_score=0.5),  # Old
        create_memory("personal_low_score", "personal", age_days=30, importance=0.5, final_score=0.1),  # Low score
        create_memory("personal_good_1", "personal", age_days=30, importance=0.5, final_score=0.5),
        create_memory("personal_good_2", "personal", age_days=30, importance=0.5, final_score=0.6),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        max_memories_per_namespace=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Namespace isolation maintained
    remaining_memories = mock_memory.memories
    work_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "work"}
    personal_ids = {m["id"] for m in remaining_memories if m["metadata"]["namespace"] == "personal"}
    
    # Work namespace: should have 3 good memories (old and low-score deleted)
    assert work_ids == {"work_good_1", "work_good_2", "work_good_3"}, \
        f"Work namespace should have 3 good memories, got {work_ids}"
    
    # Personal namespace: should have 2 good memories (old and low-score deleted)
    assert personal_ids == {"personal_good_1", "personal_good_2"}, \
        f"Personal namespace should have 2 good memories, got {personal_ids}"
    
    # Verify deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {"work_old", "work_low_score", "personal_old", "personal_low_score"}
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 old memories"
    assert result.score_pruned == 2, "Should have deleted 2 low-score memories"
    assert result.total_deleted == 4, "Total deleted should be 4"


# ============================================================================
# Property-Based Tests for Namespace Isolation - Requirements 9.2, 9.3, 9.4
# ============================================================================

from hypothesis import given, settings, strategies as st


# Configure Hypothesis settings
settings.register_profile("default", max_examples=10)
settings.load_profile("default")


@st.composite
def memory_entry_with_namespace(draw):
    """Generate a single memory entry with namespace."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    namespace = draw(st.sampled_from(["work", "personal", "system", "default", "test"]))
    importance = draw(st.floats(min_value=0.0, max_value=1.0))
    final_score = draw(st.floats(min_value=0.0, max_value=1.0))
    age_days = draw(st.integers(min_value=0, max_value=365))
    
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    
    return {
        "id": memory_id,
        "content": f"test content {memory_id}",
        "metadata": {
            "namespace": namespace,
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


@st.composite
def multi_namespace_memory_collection(draw, min_size=0, max_size=100):
    """Generate a collection of memory entries with multiple namespaces and unique IDs."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    memories = []
    used_ids = set()
    
    for i in range(count):
        memory = draw(memory_entry_with_namespace())
        # Ensure unique IDs
        while memory["id"] in used_ids:
            memory["id"] = f"{memory['id']}_{i}"
        used_ids.add(memory["id"])
        memories.append(memory)
    
    return memories


@given(
    memories=multi_namespace_memory_collection(min_size=10, max_size=5),
    max_per_namespace=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10, deadline=None)
def test_property_namespace_isolation(memories, max_per_namespace):
    """
    Feature: memory-lifecycle-manager, Property 9: For any memory collection
    with multiple namespaces, pruning operations in one namespace should never
    affect memories in other namespaces (no cross-namespace deletion or modification).
    
    **Validates: Requirements 9.2, 9.4**
    
    This property test verifies that:
    1. Pruning in one namespace doesn't delete memories from other namespaces
    2. Each namespace is pruned independently
    3. No memory leakage between namespaces
    """
    # Group memories by namespace before cleanup
    namespace_groups_before: Dict[str, set] = {}
    for memory in memories:
        namespace = memory["metadata"]["namespace"]
        if namespace not in namespace_groups_before:
            namespace_groups_before[namespace] = set()
        namespace_groups_before[namespace].add(memory["id"])
    
    # Setup config with namespace cap
    config = LifecycleConfig(
        max_total_memories=10000,  # High enough to not trigger hard cap
        max_memories_per_namespace=max_per_namespace,
        min_importance_protected=0.9  # High threshold so most memories are unprotected
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute cleanup
    result = manager.cleanup()
    
    # Group memories by namespace after cleanup
    namespace_groups_after: Dict[str, set] = {}
    for memory in mock_memory.memories:
        namespace = memory["metadata"]["namespace"]
        if namespace not in namespace_groups_after:
            namespace_groups_after[namespace] = set()
        namespace_groups_after[namespace].add(memory["id"])
    
    # Verify namespace isolation: memories deleted from one namespace should not affect others
    for namespace in namespace_groups_before:
        before_ids = namespace_groups_before[namespace]
        after_ids = namespace_groups_after.get(namespace, set())
        
        # All remaining memories in this namespace should have been in this namespace before
        assert after_ids.issubset(before_ids), \
            f"Namespace {namespace}: remaining memories should be subset of original"
        
        # Count should be <= max_per_namespace (unless protected)
        # We allow more if there are protected memories
        if len(after_ids) > max_per_namespace:
            # Check that excess is due to protected memories
            protected_count = sum(
                1 for m in mock_memory.memories
                if m["metadata"]["namespace"] == namespace
                and m["metadata"]["importance"] >= config.min_importance_protected
            )
            assert protected_count > 0, \
                f"Namespace {namespace} exceeds cap without protected memories"
    
    # Verify no cross-namespace contamination
    deleted_ids = set(mock_memory.deleted_ids)
    for deleted_id in deleted_ids:
        # Find which namespace this memory belonged to
        original_namespace = None
        for memory in memories:
            if memory["id"] == deleted_id:
                original_namespace = memory["metadata"]["namespace"]
                break
        
        assert original_namespace is not None, f"Deleted memory {deleted_id} not found in original collection"
        
        # Verify this memory is not in any other namespace's remaining memories
        for namespace, remaining_ids in namespace_groups_after.items():
            if namespace != original_namespace:
                assert deleted_id not in remaining_ids, \
                    f"Deleted memory {deleted_id} from {original_namespace} found in {namespace}"


@given(
    memories=multi_namespace_memory_collection(min_size=10, max_size=5)
)
@settings(max_examples=10, deadline=None)
def test_property_namespace_aggregation(memories):
    """
    Feature: memory-lifecycle-manager, Property 10: For any memory collection
    with multiple namespaces, the total memory count used for hard cap enforcement
    should equal the sum of memories across all namespaces.
    
    **Validates: Requirements 9.3**
    
    This property test verifies that:
    1. Total count equals sum across all namespaces
    2. Hard cap enforcement considers all namespaces
    3. Aggregation is correct regardless of namespace distribution
    """
    # Count memories per namespace before cleanup
    namespace_counts_before: Dict[str, int] = {}
    for memory in memories:
        namespace = memory["metadata"]["namespace"]
        namespace_counts_before[namespace] = namespace_counts_before.get(namespace, 0) + 1
    
    total_before = len(memories)
    
    # Setup config with hard cap that will trigger deletion
    # Set hard cap to 70% of total to ensure some deletion
    hard_cap = max(1, int(total_before * 0.7))
    
    config = LifecycleConfig(
        max_total_memories=hard_cap,
        max_memories_per_namespace=None,  # No per-namespace cap
        min_importance_protected=0.9  # High threshold so most memories are unprotected
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute cleanup
    result = manager.cleanup()
    
    # Count memories per namespace after cleanup
    namespace_counts_after: Dict[str, int] = {}
    for memory in mock_memory.memories:
        namespace = memory["metadata"]["namespace"]
        namespace_counts_after[namespace] = namespace_counts_after.get(namespace, 0) + 1
    
    total_after = len(mock_memory.memories)
    
    # Verify aggregation: total should equal sum across namespaces
    sum_across_namespaces = sum(namespace_counts_after.values())
    assert total_after == sum_across_namespaces, \
        f"Total count {total_after} should equal sum across namespaces {sum_across_namespaces}"
    
    # Verify hard cap is enforced (total <= max_total_memories)
    assert total_after <= config.max_total_memories, \
        f"Total count {total_after} should not exceed hard cap {config.max_total_memories}"
    
    # Verify that deletion happened across namespaces (if needed)
    if total_before > hard_cap:
        # Some memories should have been deleted
        assert total_after < total_before, \
            "Memories should have been deleted when exceeding hard cap"
        
        # Verify final count matches expected
        assert result.final_count == total_after, \
            f"Result final_count {result.final_count} should match actual count {total_after}"
    
    # Verify namespace counts are consistent
    for namespace in namespace_counts_before:
        before_count = namespace_counts_before[namespace]
        after_count = namespace_counts_after.get(namespace, 0)
        
        # After count should be <= before count (memories can only be deleted, not added)
        assert after_count <= before_count, \
            f"Namespace {namespace}: after count {after_count} should be <= before count {before_count}"


@given(
    memories=multi_namespace_memory_collection(min_size=5, max_size=5),
    max_per_namespace=st.integers(min_value=2, max_value=8)
)
@settings(max_examples=10, deadline=None)
def test_property_namespace_isolation_with_protection(memories, max_per_namespace):
    """
    Feature: memory-lifecycle-manager, Property 9: For any memory collection
    with multiple namespaces, pruning operations should respect importance
    protection within each namespace independently.
    
    **Validates: Requirements 9.1, 9.2, 6.3**
    
    This property test verifies that:
    1. Protected memories are preserved in all namespaces
    2. Namespace caps don't delete protected memories
    3. Protection works independently per namespace
    """
    # Identify protected memories per namespace
    protected_by_namespace: Dict[str, set] = {}
    for memory in memories:
        namespace = memory["metadata"]["namespace"]
        importance = memory["metadata"]["importance"]
        
        if importance >= 0.8:  # Will use 0.8 as protection threshold
            if namespace not in protected_by_namespace:
                protected_by_namespace[namespace] = set()
            protected_by_namespace[namespace].add(memory["id"])
    
    # Setup config
    config = LifecycleConfig(
        max_total_memories=10000,  # High enough to not trigger hard cap
        max_memories_per_namespace=max_per_namespace,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute cleanup
    result = manager.cleanup()
    
    # Verify all protected memories are preserved in each namespace
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    for namespace, protected_ids in protected_by_namespace.items():
        # All protected memories in this namespace should remain
        remaining_protected = protected_ids & remaining_ids
        assert remaining_protected == protected_ids, \
            f"Namespace {namespace}: all protected memories should be preserved. " \
            f"Expected {protected_ids}, got {remaining_protected}"
    
    # Verify namespace caps are enforced for unprotected memories only
    for memory in mock_memory.memories:
        namespace = memory["metadata"]["namespace"]
        
        # Count unprotected memories in this namespace
        unprotected_count = sum(
            1 for m in mock_memory.memories
            if m["metadata"]["namespace"] == namespace
            and m["metadata"]["importance"] < config.min_importance_protected
        )
        
        # Unprotected count should be <= max_per_namespace
        assert unprotected_count <= max_per_namespace, \
            f"Namespace {namespace}: unprotected count {unprotected_count} " \
            f"should be <= {max_per_namespace}"
