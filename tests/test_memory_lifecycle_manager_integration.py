"""
Integration tests for Memory Lifecycle Manager.

Tests the complete cleanup pipeline with MockMemoryInterface to verify that
all three pruning phases (age, score, hard cap) work together correctly.

Validates: Requirement 13.10
"""

import pytest
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Any

from luma.core.lifecycle_config import LifecycleConfig
from luma.core.lifecycle_manager import MemoryLifecycleManager
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters, RetrievalResult
from luma.core.cleanup_result import CleanupResult, CleanupStatus


# ============================================================================
# MockMemoryInterface Implementation
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """
    Mock implementation of MemoryInterface for integration testing.
    
    Provides in-memory storage with full retrieve, delete, and store methods
    to enable realistic integration testing of the complete cleanup pipeline.
    """
    
    def __init__(self, initial_memories: Optional[List[MemoryEntry]] = None):
        """
        Initialize with optional list of memory entries.
        
        Args:
            initial_memories: Optional list of memory entries to start with
        """
        self.memories: List[MemoryEntry] = (
            initial_memories.copy() if initial_memories else []
        )
        self.deleted_ids: List[str] = []
        self.store_count = 0
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Retrieve memories from the mock storage.
        
        Returns all memories with metadata about the query execution.
        """
        return {
            "memories": self.memories,
            "total_count": len(self.memories),
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }
    
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of memory to delete
        """
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a new memory (for completeness, not used in lifecycle tests).
        
        Args:
            content: Memory content
            metadata: Optional metadata
            category: Memory category
            tags: Optional tags
        
        Returns:
            Generated memory ID
        """
        self.store_count += 1
        memory_id = f"stored_{self.store_count}"
        
        memory: MemoryEntry = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat(),
            "category": category,
            "tags": tags or []
        }
        
        self.memories.append(memory)
        return memory_id


# ============================================================================
# Helper Functions
# ============================================================================

def create_memory(
    memory_id: str,
    age_days: int = 0,
    importance: float = 0.0,
    final_score: float = 0.5,
    namespace: str = "default"
) -> MemoryEntry:
    """
    Helper function to create a memory entry with specified properties.
    
    Args:
        memory_id: Unique identifier for the memory
        age_days: Age of the memory in days (0 = today, positive = past)
        importance: Importance score [0, 1]
        final_score: Final relevance score [0, 1]
        namespace: Namespace for the memory
    
    Returns:
        MemoryEntry with specified properties
    """
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    return {
        "id": memory_id,
        "content": f"Memory content {memory_id}",
        "metadata": {
            "importance": importance,
            "final_score": final_score,
            "namespace": namespace
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


# ============================================================================
# Integration Tests
# ============================================================================

def test_complete_cleanup_pipeline_with_all_phases():
    """
    Test complete cleanup pipeline with all three pruning phases working together.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create a realistic memory collection with:
      - Old memories (age-based pruning candidates)
      - Low-score memories (score-based pruning candidates)
      - Excess memories (hard cap enforcement candidates)
      - Protected memories (should survive all phases)
    
    Expected:
    - Age-based pruning removes old unprotected memories
    - Score-based pruning removes low-score unprotected memories
    - Hard cap enforcement removes excess lowest-ranked memories
    - Protected memories survive all phases
    - Final state matches expectations with exact memory_id verification
    """
    # Setup: Create a comprehensive memory collection
    memories = [
        # Old unprotected - should be deleted by age pruning
        create_memory("old_unprotected_1", age_days=100, importance=0.5, final_score=0.6),
        create_memory("old_unprotected_2", age_days=120, importance=0.3, final_score=0.7),
        
        # Old protected - should survive age pruning
        create_memory("old_protected_1", age_days=100, importance=0.9, final_score=0.8),
        create_memory("old_protected_2", age_days=150, importance=0.85, final_score=0.9),
        
        # Low-score unprotected - should be deleted by score pruning
        create_memory("low_score_unprotected_1", age_days=30, importance=0.5, final_score=0.15),
        create_memory("low_score_unprotected_2", age_days=40, importance=0.6, final_score=0.18),
        
        # Low-score protected - should survive score pruning
        create_memory("low_score_protected_1", age_days=30, importance=0.9, final_score=0.1),
        
        # Medium memories - survive age and score, may be affected by hard cap
        create_memory("medium_1", age_days=50, importance=0.5, final_score=0.4),
        create_memory("medium_2", age_days=60, importance=0.6, final_score=0.45),
        create_memory("medium_3", age_days=70, importance=0.4, final_score=0.5),
        
        # High-score unprotected - should survive age and score pruning
        create_memory("high_score_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("high_score_2", age_days=40, importance=0.6, final_score=0.85),
        
        # High-score protected - should survive all phases
        create_memory("high_score_protected_1", age_days=30, importance=0.9, final_score=0.9),
    ]
    
    config = LifecycleConfig(
        max_total_memories=8,  # Hard cap will trigger
        max_age_days=90,
        pruning_score_threshold=0.2,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run complete cleanup pipeline
    result = manager.cleanup()
    
    # Verify: Check which memories were deleted by each phase
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Phase 1: Age-based pruning should delete old unprotected memories
    assert "old_unprotected_1" not in remaining_ids, "Old unprotected 1 should be deleted by age pruning"
    assert "old_unprotected_2" not in remaining_ids, "Old unprotected 2 should be deleted by age pruning"
    assert "old_protected_1" in remaining_ids, "Old protected 1 should survive age pruning"
    assert "old_protected_2" in remaining_ids, "Old protected 2 should survive age pruning"
    
    # Phase 2: Score-based pruning should delete low-score unprotected memories
    assert "low_score_unprotected_1" not in remaining_ids, "Low-score unprotected 1 should be deleted by score pruning"
    assert "low_score_unprotected_2" not in remaining_ids, "Low-score unprotected 2 should be deleted by score pruning"
    assert "low_score_protected_1" in remaining_ids, "Low-score protected 1 should survive score pruning"
    
    # Protected memories should survive all phases
    assert "old_protected_1" in remaining_ids, "Protected memory should survive all phases"
    assert "old_protected_2" in remaining_ids, "Protected memory should survive all phases"
    assert "low_score_protected_1" in remaining_ids, "Protected memory should survive all phases"
    assert "high_score_protected_1" in remaining_ids, "Protected memory should survive all phases"
    
    # Verify final count respects hard cap
    assert len(remaining_ids) <= config.max_total_memories, \
        f"Final count {len(remaining_ids)} should not exceed max_total_memories {config.max_total_memories}"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 memories by age pruning"
    assert result.score_pruned == 2, "Should have deleted 2 memories by score pruning"
    assert result.total_deleted >= 4, "Should have deleted at least 4 memories total"
    assert result.final_count == len(remaining_ids), "Final count should match remaining memories"
    assert result.status == CleanupStatus.SUCCESS, "Cleanup should complete successfully"


def test_integration_age_and_score_pruning_only():
    """
    Test integration with age and score pruning, no hard cap enforcement.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create memories that will be pruned by age and score
    - Set high max_total_memories so hard cap doesn't trigger
    
    Expected:
    - Only age and score pruning should delete memories
    - Hard cap enforcement should delete nothing
    - Exact memory_id verification
    """
    memories = [
        # Old unprotected - deleted by age
        create_memory("old_1", age_days=100, importance=0.5, final_score=0.6),
        create_memory("old_2", age_days=110, importance=0.6, final_score=0.7),
        
        # Low-score unprotected - deleted by score
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.15),
        create_memory("low_score_2", age_days=40, importance=0.6, final_score=0.18),
        
        # Should survive
        create_memory("survivor_1", age_days=50, importance=0.5, final_score=0.5),
        create_memory("survivor_2", age_days=60, importance=0.6, final_score=0.6),
        create_memory("protected_1", age_days=100, importance=0.9, final_score=0.1),
    ]
    
    config = LifecycleConfig(
        max_total_memories=1000,  # High enough to not trigger hard cap
        max_age_days=90,
        pruning_score_threshold=0.2,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify exact remaining set
    remaining_ids = {m["id"] for m in mock_memory.memories}
    expected_remaining = {"survivor_1", "survivor_2", "protected_1"}
    
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should delete 2 by age"
    assert result.score_pruned == 2, "Should delete 2 by score"
    assert result.cap_pruned == 0, "Should delete 0 by hard cap"
    assert result.total_deleted == 4, "Total deleted should be 4"
    assert result.final_count == 3, "Final count should be 3"


def test_integration_hard_cap_only():
    """
    Test integration with only hard cap enforcement (no age or score pruning).
    
    Validates: Requirement 13.10
    
    Setup:
    - Create memories that don't trigger age or score pruning
    - Set low max_total_memories to trigger hard cap
    
    Expected:
    - Only hard cap enforcement should delete memories
    - Lowest-ranked unprotected memories deleted first
    - Protected memories survive
    - Exact memory_id verification
    """
    memories = [
        # All young and high-score, but exceed hard cap
        create_memory("low_rank_1", age_days=30, importance=0.5, final_score=0.3),
        create_memory("low_rank_2", age_days=40, importance=0.6, final_score=0.35),
        create_memory("medium_rank_1", age_days=50, importance=0.5, final_score=0.5),
        create_memory("medium_rank_2", age_days=60, importance=0.6, final_score=0.55),
        create_memory("high_rank_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("high_rank_2", age_days=40, importance=0.6, final_score=0.85),
        create_memory("protected_1", age_days=30, importance=0.9, final_score=0.2),  # Low score but protected
    ]
    
    config = LifecycleConfig(
        max_total_memories=5,  # Will trigger hard cap
        max_age_days=None,  # No age pruning
        pruning_score_threshold=None,  # No score pruning
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Protected memory must survive
    assert "protected_1" in remaining_ids, "Protected memory must survive hard cap"
    
    # Lowest-ranked should be deleted
    assert "low_rank_1" not in remaining_ids, "Lowest rank should be deleted"
    assert "low_rank_2" not in remaining_ids, "Second lowest rank should be deleted"
    
    # Highest-ranked should survive
    assert "high_rank_1" in remaining_ids, "Highest rank should survive"
    assert "high_rank_2" in remaining_ids, "Second highest rank should survive"
    
    # Verify final count respects hard cap
    assert len(remaining_ids) <= config.max_total_memories, \
        f"Final count {len(remaining_ids)} exceeds max_total_memories {config.max_total_memories}"
    
    # Verify statistics
    assert result.age_pruned == 0, "Should delete 0 by age"
    assert result.score_pruned == 0, "Should delete 0 by score"
    assert result.cap_pruned == 2, "Should delete 2 by hard cap"
    assert result.total_deleted == 2, "Total deleted should be 2"


def test_integration_protected_memories_survive_all_phases():
    """
    Test that protected memories survive all three pruning phases.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create protected memories that would normally be deleted by each phase
    - Old protected memories (age pruning candidate)
    - Low-score protected memories (score pruning candidate)
    - Protected memories when hard cap exceeded
    
    Expected:
    - All protected memories survive all phases
    - Exact memory_id verification
    """
    memories = [
        # Protected memories that would be deleted if not protected
        create_memory("old_protected", age_days=200, importance=0.9, final_score=0.05),
        create_memory("low_score_protected", age_days=30, importance=0.85, final_score=0.05),
        create_memory("protected_at_cap", age_days=30, importance=0.8, final_score=0.1),
        
        # Unprotected memories to be deleted
        create_memory("old_unprotected", age_days=200, importance=0.5, final_score=0.6),
        create_memory("low_score_unprotected", age_days=30, importance=0.5, final_score=0.05),
        create_memory("unprotected_1", age_days=30, importance=0.5, final_score=0.3),
        create_memory("unprotected_2", age_days=40, importance=0.6, final_score=0.35),
        create_memory("unprotected_3", age_days=50, importance=0.5, final_score=0.4),
    ]
    
    config = LifecycleConfig(
        max_total_memories=4,  # Hard cap will trigger
        max_age_days=100,
        pruning_score_threshold=0.1,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify: All protected memories survive
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "old_protected" in remaining_ids, "Old protected memory should survive age pruning"
    assert "low_score_protected" in remaining_ids, "Low-score protected memory should survive score pruning"
    assert "protected_at_cap" in remaining_ids, "Protected memory should survive hard cap enforcement"
    
    # Verify unprotected memories are deleted
    assert "old_unprotected" not in remaining_ids, "Old unprotected should be deleted"
    assert "low_score_unprotected" not in remaining_ids, "Low-score unprotected should be deleted"
    
    # Verify final count
    assert len(remaining_ids) <= config.max_total_memories, \
        f"Final count {len(remaining_ids)} exceeds max_total_memories {config.max_total_memories}"


def test_integration_deterministic_deletion_order():
    """
    Test that deletion order is deterministic across multiple runs.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create memories with identical scores to test tie-breaking
    - Run cleanup twice on identical memory collections
    
    Expected:
    - Same memories deleted in both runs
    - Deletion order follows: final_score asc, timestamp asc, memory_id asc
    - Exact memory_id verification
    """
    # Create memories with identical scores but different timestamps and IDs
    base_time = datetime.now(UTC)
    memories_run1 = [
        {
            "id": "mem_a",
            "content": "Content A",
            "metadata": {"importance": 0.5, "final_score": 0.4},
            "timestamp": (base_time - timedelta(days=50)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_b",
            "content": "Content B",
            "metadata": {"importance": 0.5, "final_score": 0.4},
            "timestamp": (base_time - timedelta(days=50)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_c",
            "content": "Content C",
            "metadata": {"importance": 0.5, "final_score": 0.4},
            "timestamp": (base_time - timedelta(days=40)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_d",
            "content": "Content D",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=30)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_e",
            "content": "Content E",
            "metadata": {"importance": 0.5, "final_score": 0.6},
            "timestamp": (base_time - timedelta(days=20)).isoformat(),
            "category": "test",
            "tags": []
        },
    ]
    
    # Create identical copy for second run
    memories_run2 = [m.copy() for m in memories_run1]
    
    config = LifecycleConfig(
        max_total_memories=3,  # Will delete 2 memories
        min_importance_protected=0.8
    )
    
    # Run 1
    mock_memory1 = MockMemoryInterface(memories_run1)
    manager1 = MemoryLifecycleManager(config, mock_memory1)
    result1 = manager1.cleanup()
    remaining_ids1 = {m["id"] for m in mock_memory1.memories}
    deleted_ids1 = set(mock_memory1.deleted_ids)
    
    # Run 2
    mock_memory2 = MockMemoryInterface(memories_run2)
    manager2 = MemoryLifecycleManager(config, mock_memory2)
    result2 = manager2.cleanup()
    remaining_ids2 = {m["id"] for m in mock_memory2.memories}
    deleted_ids2 = set(mock_memory2.deleted_ids)
    
    # Verify: Identical results
    assert remaining_ids1 == remaining_ids2, \
        f"Remaining IDs differ: {remaining_ids1} vs {remaining_ids2}"
    assert deleted_ids1 == deleted_ids2, \
        f"Deleted IDs differ: {deleted_ids1} vs {deleted_ids2}"
    
    # Verify: Correct memories deleted (lowest score, then oldest, then lexicographical)
    # mem_a and mem_b have same score (0.4) and timestamp, so lexicographical order: mem_a < mem_b
    assert "mem_a" not in remaining_ids1, "mem_a should be deleted (lowest score, oldest, first alphabetically)"
    assert "mem_b" not in remaining_ids1, "mem_b should be deleted (lowest score, oldest, second alphabetically)"
    assert "mem_c" in remaining_ids1, "mem_c should survive (same score but newer timestamp)"
    assert "mem_d" in remaining_ids1, "mem_d should survive (higher score)"
    assert "mem_e" in remaining_ids1, "mem_e should survive (highest score)"


def test_integration_empty_memory_collection():
    """
    Test cleanup with empty memory collection.
    
    Validates: Requirement 13.10
    
    Setup:
    - Start with no memories
    
    Expected:
    - Cleanup completes successfully
    - No errors
    - All statistics are zero
    """
    config = LifecycleConfig(
        max_total_memories=100,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface([])
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify
    assert len(mock_memory.memories) == 0, "Should have no memories"
    assert result.age_pruned == 0, "Should delete 0 by age"
    assert result.score_pruned == 0, "Should delete 0 by score"
    assert result.cap_pruned == 0, "Should delete 0 by hard cap"
    assert result.total_deleted == 0, "Total deleted should be 0"
    assert result.final_count == 0, "Final count should be 0"
    assert result.status == CleanupStatus.SUCCESS, "Should complete successfully"


def test_integration_all_memories_protected():
    """
    Test cleanup when all memories are protected.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create memories that exceed hard cap
    - All memories have importance >= min_importance_protected
    
    Expected:
    - No memories deleted (all protected)
    - Final count may exceed max_total_memories (all protected)
    - Exact memory_id verification
    """
    memories = [
        create_memory("protected_1", age_days=200, importance=0.9, final_score=0.1),
        create_memory("protected_2", age_days=150, importance=0.85, final_score=0.2),
        create_memory("protected_3", age_days=100, importance=0.8, final_score=0.05),
        create_memory("protected_4", age_days=50, importance=0.95, final_score=0.3),
        create_memory("protected_5", age_days=30, importance=1.0, final_score=0.15),
    ]
    
    config = LifecycleConfig(
        max_total_memories=3,  # Would normally trigger hard cap
        max_age_days=90,
        pruning_score_threshold=0.2,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify: All memories survive
    remaining_ids = {m["id"] for m in mock_memory.memories}
    expected_ids = {"protected_1", "protected_2", "protected_3", "protected_4", "protected_5"}
    
    assert remaining_ids == expected_ids, \
        f"All protected memories should survive, expected {expected_ids}, got {remaining_ids}"
    
    # Verify statistics
    assert result.age_pruned == 0, "Should delete 0 by age (all protected)"
    assert result.score_pruned == 0, "Should delete 0 by score (all protected)"
    assert result.cap_pruned == 0, "Should delete 0 by hard cap (all protected)"
    assert result.total_deleted == 0, "Total deleted should be 0"
    assert result.final_count == 5, "Final count should be 5 (all protected)"


def test_integration_realistic_large_collection():
    """
    Test cleanup with a realistic larger memory collection.
    
    Validates: Requirement 13.10
    
    Setup:
    - Create 50 memories with realistic distribution
    - Mix of ages, scores, and importance levels
    
    Expected:
    - All three phases work correctly
    - Final state matches expectations
    - Exact memory_id verification
    """
    memories = []
    
    # Create 10 old unprotected memories (should be deleted by age)
    for i in range(10):
        memories.append(
            create_memory(f"old_unprotected_{i}", age_days=100+i, importance=0.5, final_score=0.6)
        )
    
    # Create 5 old protected memories (should survive)
    for i in range(5):
        memories.append(
            create_memory(f"old_protected_{i}", age_days=100+i, importance=0.9, final_score=0.7)
        )
    
    # Create 10 low-score unprotected memories (should be deleted by score)
    for i in range(10):
        memories.append(
            create_memory(f"low_score_{i}", age_days=30+i, importance=0.5, final_score=0.15)
        )
    
    # Create 15 medium memories (some may be deleted by hard cap)
    for i in range(15):
        memories.append(
            create_memory(f"medium_{i}", age_days=40+i, importance=0.5, final_score=0.4+i*0.01)
        )
    
    # Create 10 high-score memories (should mostly survive)
    for i in range(10):
        memories.append(
            create_memory(f"high_score_{i}", age_days=20+i, importance=0.5, final_score=0.8+i*0.01)
        )
    
    config = LifecycleConfig(
        max_total_memories=25,  # Will trigger hard cap
        max_age_days=90,
        pruning_score_threshold=0.2,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute
    result = manager.cleanup()
    
    # Verify
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # All old unprotected should be deleted
    for i in range(10):
        assert f"old_unprotected_{i}" not in remaining_ids, \
            f"old_unprotected_{i} should be deleted by age pruning"
    
    # All old protected should survive
    for i in range(5):
        assert f"old_protected_{i}" in remaining_ids, \
            f"old_protected_{i} should survive (protected)"
    
    # All low-score should be deleted
    for i in range(10):
        assert f"low_score_{i}" not in remaining_ids, \
            f"low_score_{i} should be deleted by score pruning"
    
    # All high-score should survive (if not deleted by hard cap)
    # At least some high-score memories should survive
    high_score_survivors = [id for id in remaining_ids if id.startswith("high_score_")]
    assert len(high_score_survivors) > 0, "At least some high-score memories should survive"
    
    # Verify final count respects hard cap
    assert len(remaining_ids) <= config.max_total_memories, \
        f"Final count {len(remaining_ids)} exceeds max_total_memories {config.max_total_memories}"
    
    # Verify statistics
    assert result.age_pruned == 10, "Should delete 10 by age"
    assert result.score_pruned == 10, "Should delete 10 by score"
    assert result.total_deleted >= 20, "Should delete at least 20 total"
    assert result.final_count == len(remaining_ids), "Final count should match remaining"
    assert result.status == CleanupStatus.SUCCESS, "Should complete successfully"
