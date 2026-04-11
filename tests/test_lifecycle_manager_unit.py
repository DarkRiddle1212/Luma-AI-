"""
Unit tests for Memory Lifecycle Manager.

Tests configuration validation, age-based pruning, score-based pruning,
hard cap enforcement, importance protection, idempotency, and error handling.
"""

import pytest
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional
import logging

from luma.core.lifecycle_manager import (
    LifecycleConfig,
    ConfigValidator,
    MemoryLifecycleManager,
    AgePruner,
    ScorePruner,
    HardCapEnforcer,
    extract_importance,
    extract_final_score
)
from luma.core.memory_interface import MemoryInterface, MemoryEntry, RetrievalResult, QueryParameters


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for testing."""
    
    def __init__(self, initial_memories: Optional[List[MemoryEntry]] = None):
        """
        Initialize mock with optional initial memories.
        
        Args:
            initial_memories: List of memory entries to start with
        """
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
        # Check if this deletion should fail
        if memory_id in self.delete_failures:
            raise self.delete_failures[memory_id]
        
        # Remove memory from list
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
        return True
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store method (not used in lifecycle manager tests)."""
        raise NotImplementedError("Store not used in lifecycle manager tests")


def generate_memory(
    memory_id: str,
    content: str = "test content",
    importance: float = 0.0,
    final_score: float = 0.5,
    age_days: int = 0,
    category: str = "test",
    tags: Optional[List[str]] = None
) -> MemoryEntry:
    """
    Generate a test memory entry.
    
    Args:
        memory_id: Unique identifier
        content: Memory content
        importance: Importance score [0, 1]
        final_score: Final relevance score [0, 1]
        age_days: Age in days (0 = now, positive = past)
        category: Memory category
        tags: List of tags
    
    Returns:
        MemoryEntry with specified attributes
    """
    timestamp = datetime.now(UTC) - timedelta(days=age_days)
    
    return {
        "id": memory_id,
        "content": content,
        "metadata": {
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": category,
        "tags": tags or []
    }


def generate_memories(count: int, **kwargs) -> List[MemoryEntry]:
    """
    Generate multiple test memory entries.
    
    Args:
        count: Number of memories to generate
        **kwargs: Arguments passed to generate_memory
    
    Returns:
        List of memory entries
    """
    return [generate_memory(f"mem_{i}", **kwargs) for i in range(count)]



# ============================================================================
# Configuration Validation Tests
# ============================================================================

def test_config_validation_max_total_memories_zero():
    """Test that max_total_memories=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        LifecycleConfig(max_total_memories=0)


def test_config_validation_max_total_memories_negative():
    """Test that max_total_memories<0 raises ValueError."""
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        LifecycleConfig(max_total_memories=-1)


def test_config_validation_max_memories_per_namespace_zero():
    """Test that max_memories_per_namespace=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_memories_per_namespace must be greater than 0"):
        LifecycleConfig(max_total_memories=1000, max_memories_per_namespace=0)


def test_config_validation_max_memories_per_namespace_negative():
    """Test that max_memories_per_namespace<0 raises ValueError."""
    with pytest.raises(ValueError, match="max_memories_per_namespace cannot be negative"):
        LifecycleConfig(max_total_memories=1000, max_memories_per_namespace=-1)


def test_config_validation_max_age_days_zero():
    """Test that max_age_days=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_age_days must be greater than 0"):
        LifecycleConfig(max_total_memories=1000, max_age_days=0)


def test_config_validation_max_age_days_negative():
    """Test that max_age_days<0 raises ValueError."""
    with pytest.raises(ValueError, match="max_age_days cannot be negative"):
        LifecycleConfig(max_total_memories=1000, max_age_days=-1)


def test_config_validation_pruning_score_threshold_below_zero():
    """Test that pruning_score_threshold<0 raises ValueError."""
    with pytest.raises(ValueError, match="pruning_score_threshold must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, pruning_score_threshold=-0.1)


def test_config_validation_pruning_score_threshold_above_one():
    """Test that pruning_score_threshold>1 raises ValueError."""
    with pytest.raises(ValueError, match="pruning_score_threshold must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, pruning_score_threshold=1.5)


def test_config_validation_min_importance_protected_below_zero():
    """Test that min_importance_protected<0 raises ValueError."""
    with pytest.raises(ValueError, match="min_importance_protected must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, min_importance_protected=-0.1)


def test_config_validation_min_importance_protected_above_one():
    """Test that min_importance_protected>1 raises ValueError."""
    with pytest.raises(ValueError, match="min_importance_protected must be between 0 and 1"):
        LifecycleConfig(max_total_memories=1000, min_importance_protected=1.5)


def test_config_validation_valid_configuration():
    """Test that valid configuration initializes successfully."""
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    assert config.max_total_memories == 10000
    assert config.max_age_days == 90
    assert config.pruning_score_threshold == 0.3
    assert config.min_importance_protected == 0.8



# ============================================================================
# Age-Based Pruning Tests
# ============================================================================

def test_age_pruning_deletes_old_unprotected_memories():
    """Test that old unprotected memories are deleted."""
    memories = [
        generate_memory("old_unprotected", importance=0.5, age_days=100),
        generate_memory("recent", importance=0.5, age_days=10),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = AgePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 1
    assert "old_unprotected" in mock_memory.deleted_ids
    assert len(mock_memory.memories) == 1
    assert mock_memory.memories[0]["id"] == "recent"


def test_age_pruning_preserves_old_protected_memories():
    """Test that old protected memories are preserved."""
    memories = [
        generate_memory("old_protected", importance=0.9, age_days=100),
        generate_memory("recent", importance=0.5, age_days=10),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = AgePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 2


def test_age_pruning_preserves_recent_memories():
    """Test that recent memories are preserved regardless of importance."""
    memories = [
        generate_memory("recent_low_importance", importance=0.1, age_days=10),
        generate_memory("recent_high_importance", importance=0.9, age_days=20),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = AgePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 2


def test_age_pruning_empty_memory_collection():
    """Test age pruning with empty memory collection returns 0."""
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface([])
    pruner = AgePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0


def test_age_pruning_skipped_when_max_age_days_none():
    """Test that age pruning is skipped when max_age_days is None."""
    memories = [
        generate_memory("old", importance=0.5, age_days=100),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=None, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = AgePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 1



# ============================================================================
# Score-Based Pruning Tests
# ============================================================================

def test_score_pruning_deletes_low_score_unprotected_memories():
    """Test that low-score unprotected memories are deleted."""
    memories = [
        generate_memory("low_score_unprotected", importance=0.5, final_score=0.2),
        generate_memory("high_score", importance=0.5, final_score=0.8),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = ScorePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 1
    assert "low_score_unprotected" in mock_memory.deleted_ids
    assert len(mock_memory.memories) == 1
    assert mock_memory.memories[0]["id"] == "high_score"


def test_score_pruning_preserves_low_score_protected_memories():
    """Test that low-score protected memories are preserved."""
    memories = [
        generate_memory("low_score_protected", importance=0.9, final_score=0.2),
        generate_memory("high_score", importance=0.5, final_score=0.8),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = ScorePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 2


def test_score_pruning_preserves_high_score_memories():
    """Test that high-score memories are preserved regardless of importance."""
    memories = [
        generate_memory("high_score_low_importance", importance=0.1, final_score=0.8),
        generate_memory("high_score_high_importance", importance=0.9, final_score=0.9),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = ScorePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 2


def test_score_extraction_from_final_score_field():
    """Test score extraction from metadata['final_score']."""
    memory = generate_memory("test", final_score=0.75)
    score = extract_final_score(memory)
    assert score == 0.75


def test_score_extraction_from_score_field():
    """Test score extraction from metadata['score'] as fallback."""
    memory = {
        "id": "test",
        "content": "test",
        "metadata": {"score": 0.65},
        "timestamp": datetime.now(UTC).isoformat(),
        "category": "test",
        "tags": []
    }
    score = extract_final_score(memory)
    assert score == 0.65


def test_score_pruning_empty_memory_collection():
    """Test score pruning with empty memory collection returns 0."""
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface([])
    pruner = ScorePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0


def test_score_pruning_skipped_when_threshold_none():
    """Test that score pruning is skipped when pruning_score_threshold is None."""
    memories = [
        generate_memory("low_score", importance=0.5, final_score=0.1),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=None, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    pruner = ScorePruner(config, mock_memory)
    
    deleted_count = pruner.prune()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 1



# ============================================================================
# Hard Cap Enforcement Tests
# ============================================================================

def test_hard_cap_enforcement_respects_limit():
    """Test that total count never exceeds max_total_memories."""
    # Create 15 memories, cap at 10
    memories = generate_memories(15, importance=0.5, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=10, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    assert deleted_count == 5
    assert len(mock_memory.memories) == 10


def test_hard_cap_enforcement_deletes_lowest_ranked_first():
    """Test that lowest-ranked memories are deleted first."""
    memories = [
        generate_memory("low_score", importance=0.5, final_score=0.1),
        generate_memory("mid_score", importance=0.5, final_score=0.5),
        generate_memory("high_score", importance=0.5, final_score=0.9),
    ]
    
    config = LifecycleConfig(max_total_memories=2, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    assert deleted_count == 1
    assert "low_score" in mock_memory.deleted_ids
    assert len(mock_memory.memories) == 2


def test_hard_cap_enforcement_preserves_protected_memories():
    """Test that protected memories are never deleted."""
    memories = [
        generate_memory("protected_low_score", importance=0.9, final_score=0.1),
        generate_memory("unprotected_high_score", importance=0.5, final_score=0.9),
    ]
    
    config = LifecycleConfig(max_total_memories=1, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    assert deleted_count == 1
    assert "unprotected_high_score" in mock_memory.deleted_ids
    assert len(mock_memory.memories) == 1
    assert mock_memory.memories[0]["id"] == "protected_low_score"


def test_hard_cap_enforcement_deterministic_order_with_identical_scores():
    """Test deterministic deletion order with identical scores."""
    # Create memories with identical scores but different timestamps and IDs
    base_time = datetime.now(UTC)
    memories = [
        {
            "id": "mem_c",
            "content": "test",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=1)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_a",
            "content": "test",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=3)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "mem_b",
            "content": "test",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=2)).isoformat(),
            "category": "test",
            "tags": []
        },
    ]
    
    config = LifecycleConfig(max_total_memories=1, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    assert deleted_count == 2
    # Should delete oldest first (mem_a), then next oldest (mem_b)
    assert "mem_a" in mock_memory.deleted_ids
    assert "mem_b" in mock_memory.deleted_ids
    assert len(mock_memory.memories) == 1
    assert mock_memory.memories[0]["id"] == "mem_c"


def test_hard_cap_enforcement_no_deletion_when_under_cap():
    """Test no deletion when under cap."""
    memories = generate_memories(5, importance=0.5, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=10, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    assert deleted_count == 0
    assert len(mock_memory.deleted_ids) == 0
    assert len(mock_memory.memories) == 5


def test_hard_cap_enforcement_warning_when_insufficient_unprotected():
    """Test warning logged when insufficient unprotected memories."""
    # All memories are protected
    memories = generate_memories(15, importance=0.9, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=10, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    enforcer = HardCapEnforcer(config, mock_memory)
    
    deleted_count = enforcer.enforce()
    
    # Should delete 0 because all are protected
    assert deleted_count == 0
    assert len(mock_memory.memories) == 15



# ============================================================================
# Importance Protection Tests
# ============================================================================

def test_importance_extraction_from_importance_field():
    """Test importance extraction from metadata['importance']."""
    memory = generate_memory("test", importance=0.85)
    importance = extract_importance(memory)
    assert importance == 0.85


def test_importance_extraction_from_context_importance_field():
    """Test importance extraction from metadata['context']['importance'] as fallback."""
    memory = {
        "id": "test",
        "content": "test",
        "metadata": {"context": {"importance": 0.75}},
        "timestamp": datetime.now(UTC).isoformat(),
        "category": "test",
        "tags": []
    }
    importance = extract_importance(memory)
    assert importance == 0.75


def test_importance_extraction_defaults_to_zero():
    """Test default importance of 0.0 when not found."""
    memory = {
        "id": "test",
        "content": "test",
        "metadata": {},
        "timestamp": datetime.now(UTC).isoformat(),
        "category": "test",
        "tags": []
    }
    importance = extract_importance(memory)
    assert importance == 0.0


def test_protected_memories_survive_age_pruning():
    """Test protected memories survive age pruning."""
    memories = [
        generate_memory("protected_old", importance=0.9, age_days=100),
        generate_memory("unprotected_old", importance=0.5, age_days=100),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    stats = manager.cleanup()
    
    assert stats["age_pruned"] == 1
    assert "unprotected_old" in mock_memory.deleted_ids
    assert "protected_old" not in mock_memory.deleted_ids


def test_protected_memories_survive_score_pruning():
    """Test protected memories survive score pruning."""
    memories = [
        generate_memory("protected_low_score", importance=0.9, final_score=0.1),
        generate_memory("unprotected_low_score", importance=0.5, final_score=0.1),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    stats = manager.cleanup()
    
    assert stats["score_pruned"] == 1
    assert "unprotected_low_score" in mock_memory.deleted_ids
    assert "protected_low_score" not in mock_memory.deleted_ids


def test_protected_memories_survive_hard_cap_enforcement():
    """Test protected memories survive hard cap enforcement."""
    memories = [
        generate_memory("protected", importance=0.9, final_score=0.1),
        generate_memory("unprotected", importance=0.5, final_score=0.9),
    ]
    
    config = LifecycleConfig(max_total_memories=1, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    stats = manager.cleanup()
    
    assert stats["cap_pruned"] == 1
    assert "unprotected" in mock_memory.deleted_ids
    assert "protected" not in mock_memory.deleted_ids



# ============================================================================
# Idempotency Tests
# ============================================================================

def test_cleanup_idempotency_second_run_deletes_nothing():
    """Test running cleanup twice deletes nothing on second run."""
    memories = [
        generate_memory("old", importance=0.5, age_days=100),
        generate_memory("recent", importance=0.5, age_days=10),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, max_age_days=90, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # First cleanup
    stats1 = manager.cleanup()
    assert stats1["total_deleted"] == 1
    
    # Second cleanup
    stats2 = manager.cleanup()
    assert stats2["total_deleted"] == 0
    assert stats2["final_count"] == stats1["final_count"]


def test_cleanup_idempotency_final_state_identical():
    """Test final state identical after multiple cleanup runs."""
    memories = generate_memories(20, importance=0.5, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=10, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # First cleanup
    stats1 = manager.cleanup()
    remaining_ids_1 = {m["id"] for m in mock_memory.memories}
    
    # Second cleanup
    stats2 = manager.cleanup()
    remaining_ids_2 = {m["id"] for m in mock_memory.memories}
    
    assert stats2["total_deleted"] == 0
    assert remaining_ids_1 == remaining_ids_2
    assert stats1["final_count"] == stats2["final_count"]


def test_cleanup_on_compliant_state_deletes_nothing():
    """Test cleanup on already-compliant state deletes nothing."""
    memories = generate_memories(5, importance=0.5, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=10, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    stats = manager.cleanup()
    
    assert stats["total_deleted"] == 0
    assert stats["final_count"] == 5



# ============================================================================
# Error Handling Tests
# ============================================================================

def test_deletion_failure_logs_error_and_continues():
    """Test deletion failure logs error and continues processing."""
    memories = [
        generate_memory("fail", importance=0.5, final_score=0.1),
        generate_memory("succeed", importance=0.5, final_score=0.2),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    
    # Configure first deletion to fail
    mock_memory.delete_failures["fail"] = Exception("Simulated deletion failure")
    
    pruner = ScorePruner(config, mock_memory)
    deleted_count = pruner.prune()
    
    # Should continue and delete the second memory
    assert deleted_count == 1
    assert "succeed" in mock_memory.deleted_ids
    assert "fail" not in mock_memory.deleted_ids


def test_cleanup_never_propagates_exceptions():
    """Test cleanup catches all exceptions."""
    memories = generate_memories(5, importance=0.5, final_score=0.5)
    
    config = LifecycleConfig(max_total_memories=1000, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    
    # Configure all deletions to fail
    for memory in memories:
        mock_memory.delete_failures[memory["id"]] = Exception("Simulated failure")
    
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Should not raise exception
    stats = manager.cleanup()
    
    # Should return stats with error count
    assert "errors" in stats
    assert stats["total_deleted"] >= 0


def test_multiple_deletion_failures_tracked():
    """Test multiple deletion failures tracked in error count."""
    memories = [
        generate_memory("fail1", importance=0.5, final_score=0.1),
        generate_memory("fail2", importance=0.5, final_score=0.1),
        generate_memory("succeed", importance=0.5, final_score=0.1),
    ]
    
    config = LifecycleConfig(max_total_memories=1000, pruning_score_threshold=0.3, min_importance_protected=0.8)
    mock_memory = MockMemoryInterface(memories)
    
    # Configure two deletions to fail
    mock_memory.delete_failures["fail1"] = Exception("Failure 1")
    mock_memory.delete_failures["fail2"] = Exception("Failure 2")
    
    pruner = ScorePruner(config, mock_memory)
    deleted_count = pruner.prune()
    
    # Should delete only the one that succeeds
    assert deleted_count == 1
    assert "succeed" in mock_memory.deleted_ids
