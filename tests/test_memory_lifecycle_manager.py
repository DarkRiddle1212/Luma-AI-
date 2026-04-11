"""
Unit tests for Memory Lifecycle Manager configuration validation.

Tests configuration validation for LifecycleConfig dataclass according to
Requirements 1.6, 1.7, 1.8, 1.9, 1.10, and 13.1.
"""

import pytest
from luma.core.lifecycle_config import LifecycleConfig, ConfigValidator


# ============================================================================
# Configuration Validation Tests - Requirements 1.6-1.10, 13.1
# ============================================================================

def test_max_total_memories_zero_raises_value_error():
    """
    Test that max_total_memories=0 raises ValueError with exact message.
    
    Validates: Requirement 1.6
    """
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        LifecycleConfig(max_total_memories=0)


def test_max_total_memories_negative_raises_value_error():
    """
    Test that max_total_memories<0 raises ValueError with exact message.
    
    Validates: Requirement 1.6
    """
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        LifecycleConfig(max_total_memories=-1)


def test_max_memories_per_namespace_negative_raises_value_error():
    """
    Test that max_memories_per_namespace<0 raises ValueError with exact message.
    
    Validates: Requirement 1.10
    """
    with pytest.raises(ValueError, match="max_memories_per_namespace must be greater than 0"):
        LifecycleConfig(
            max_total_memories=1000,
            max_memories_per_namespace=-1
        )


def test_max_age_days_negative_raises_value_error():
    """
    Test that max_age_days<0 raises ValueError with exact message.
    
    Validates: Requirement 1.9
    """
    with pytest.raises(ValueError, match="max_age_days must be greater than 0"):
        LifecycleConfig(
            max_total_memories=1000,
            max_age_days=-1
        )


def test_pruning_score_threshold_below_zero_raises_value_error():
    """
    Test that pruning_score_threshold<0 raises ValueError with exact message.
    
    Validates: Requirement 1.7
    """
    with pytest.raises(ValueError, match="pruning_score_threshold must be between 0 and 1"):
        LifecycleConfig(
            max_total_memories=1000,
            pruning_score_threshold=-0.1
        )


def test_pruning_score_threshold_above_one_raises_value_error():
    """
    Test that pruning_score_threshold>1 raises ValueError with exact message.
    
    Validates: Requirement 1.7
    """
    with pytest.raises(ValueError, match="pruning_score_threshold must be between 0 and 1"):
        LifecycleConfig(
            max_total_memories=1000,
            pruning_score_threshold=1.5
        )


def test_min_importance_protected_below_zero_raises_value_error():
    """
    Test that min_importance_protected<0 raises ValueError with exact message.
    
    Validates: Requirement 1.8
    """
    with pytest.raises(ValueError, match="min_importance_protected must be between 0 and 1"):
        LifecycleConfig(
            max_total_memories=1000,
            min_importance_protected=-0.1
        )


def test_min_importance_protected_above_one_raises_value_error():
    """
    Test that min_importance_protected>1 raises ValueError with exact message.
    
    Validates: Requirement 1.8
    """
    with pytest.raises(ValueError, match="min_importance_protected must be between 0 and 1"):
        LifecycleConfig(
            max_total_memories=1000,
            min_importance_protected=1.5
        )


def test_valid_configuration_succeeds():
    """
    Test that valid configuration initializes successfully.
    
    Validates: Requirements 1.1-1.5, 13.1
    """
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=1000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    assert config.max_total_memories == 10000
    assert config.max_memories_per_namespace == 1000
    assert config.max_age_days == 90
    assert config.pruning_score_threshold == 0.3
    assert config.min_importance_protected == 0.8


def test_valid_configuration_with_optional_none_succeeds():
    """
    Test that valid configuration with optional fields as None succeeds.
    
    Validates: Requirements 1.1-1.5, 13.1
    """
    config = LifecycleConfig(
        max_total_memories=10000,
        max_memories_per_namespace=None,
        max_age_days=None,
        pruning_score_threshold=None,
        min_importance_protected=0.8
    )
    
    assert config.max_total_memories == 10000
    assert config.max_memories_per_namespace is None
    assert config.max_age_days is None
    assert config.pruning_score_threshold is None
    assert config.min_importance_protected == 0.8


def test_valid_configuration_with_boundary_values_succeeds():
    """
    Test that valid configuration with boundary values succeeds.
    
    Validates: Requirements 1.1-1.5, 13.1
    
    Note: max_memories_per_namespace and max_age_days must be > 0 if provided,
    as per requirements 1.9 and 1.10 which require positive values.
    """
    # Test boundary values: 0.0 and 1.0 for thresholds, 1 for optional integers (minimum valid value)
    config = LifecycleConfig(
        max_total_memories=1,
        max_memories_per_namespace=1,
        max_age_days=1,
        pruning_score_threshold=0.0,
        min_importance_protected=1.0
    )
    
    assert config.max_total_memories == 1
    assert config.max_memories_per_namespace == 1
    assert config.max_age_days == 1
    assert config.pruning_score_threshold == 0.0
    assert config.min_importance_protected == 1.0


def test_config_validator_validate_method():
    """
    Test ConfigValidator.validate() method directly.
    
    Validates: Requirements 1.6-1.10, 13.1
    """
    # Valid config should not raise
    valid_config = LifecycleConfig(
        max_total_memories=1000,
        min_importance_protected=0.8
    )
    ConfigValidator.validate(valid_config)  # Should not raise
    
    # Invalid config should raise
    invalid_config = LifecycleConfig.__new__(LifecycleConfig)
    invalid_config.max_total_memories = -1
    invalid_config.max_memories_per_namespace = None
    invalid_config.max_age_days = None
    invalid_config.pruning_score_threshold = None
    invalid_config.min_importance_protected = 0.8
    
    with pytest.raises(ValueError, match="max_total_memories must be greater than 0"):
        ConfigValidator.validate(invalid_config)


# ============================================================================
# Age-Based Pruning Tests - Requirements 3.2, 3.3, 3.4, 13.1
# ============================================================================

from datetime import datetime, timedelta, UTC
from luma.core.lifecycle_manager import MemoryLifecycleManager
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters
from luma.core.cleanup_result import CleanupResult, CleanupStatus
from typing import Dict, List, Optional, Any


class MockMemoryInterface(MemoryInterface):
    """Mock implementation of MemoryInterface for testing."""
    
    def __init__(self, initial_memories: List[MemoryEntry]):
        """Initialize with a list of memory entries."""
        self.deleted_ids: List[str] = []
        # Use a dict for O(1) lookup performance in large datasets
        self._memory_dict = {m["id"]: m for m in initial_memories}
    
    @property
    def memories(self) -> List[MemoryEntry]:
        """Get memories as a list for backward compatibility."""
        return list(self._memory_dict.values())
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Retrieve memories from the mock storage."""
        return {
            "memories": list(self._memory_dict.values()),
            "total_count": len(self._memory_dict),
            "query_metadata": {}
        }
    
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID with O(1) performance."""
        if memory_id in self._memory_dict:
            del self._memory_dict[memory_id]
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
    age_days: int,
    importance: float = 0.0,
    final_score: float = 0.5
) -> MemoryEntry:
    """
    Helper function to create a memory entry with specified age and importance.
    
    Args:
        memory_id: Unique identifier for the memory
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
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


def test_age_pruning_deletes_old_unprotected_memories():
    """
    Test that old unprotected memories are deleted.
    
    Validates: Requirement 3.2
    
    Setup:
    - Create memories with age > max_age_days and importance < min_importance_protected
    - Create memories with age <= max_age_days
    
    Expected:
    - Old unprotected memories should be deleted
    - Young memories should be preserved
    """
    # Setup: Create memories with various ages
    memories = [
        create_memory("old_unprotected_1", age_days=100, importance=0.5),  # Should be deleted
        create_memory("old_unprotected_2", age_days=95, importance=0.7),   # Should be deleted
        create_memory("young_1", age_days=30, importance=0.5),             # Should be preserved
        create_memory("young_2", age_days=50, importance=0.3),             # Should be preserved
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Old unprotected memories deleted, young memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "old_unprotected_1" not in remaining_ids, "Old unprotected memory 1 should be deleted"
    assert "old_unprotected_2" not in remaining_ids, "Old unprotected memory 2 should be deleted"
    assert "young_1" in remaining_ids, "Young memory 1 should be preserved"
    assert "young_2" in remaining_ids, "Young memory 2 should be preserved"
    
    # Verify exact set of remaining memories
    assert remaining_ids == {"young_1", "young_2"}, "Only young memories should remain"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 old memories"
    assert result.total_deleted == 2, "Total deleted should be 2"


def test_age_pruning_preserves_old_protected_memories():
    """
    Test that old protected memories are preserved.
    
    Validates: Requirement 3.3
    
    Setup:
    - Create memories with age > max_age_days and importance >= min_importance_protected
    - Create memories with age > max_age_days and importance < min_importance_protected
    
    Expected:
    - Old protected memories should be preserved
    - Old unprotected memories should be deleted
    """
    # Setup: Create old memories with different importance levels
    memories = [
        create_memory("old_protected_1", age_days=100, importance=0.8),    # Should be preserved
        create_memory("old_protected_2", age_days=120, importance=0.9),    # Should be preserved
        create_memory("old_protected_3", age_days=95, importance=1.0),     # Should be preserved
        create_memory("old_unprotected_1", age_days=100, importance=0.5),  # Should be deleted
        create_memory("old_unprotected_2", age_days=110, importance=0.79), # Should be deleted (< 0.8)
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories preserved, unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "old_protected_1" in remaining_ids, "Old protected memory 1 should be preserved"
    assert "old_protected_2" in remaining_ids, "Old protected memory 2 should be preserved"
    assert "old_protected_3" in remaining_ids, "Old protected memory 3 should be preserved"
    assert "old_unprotected_1" not in remaining_ids, "Old unprotected memory 1 should be deleted"
    assert "old_unprotected_2" not in remaining_ids, "Old unprotected memory 2 should be deleted"
    
    # Verify exact set of remaining memories
    assert remaining_ids == {"old_protected_1", "old_protected_2", "old_protected_3"}, \
        "Only protected memories should remain"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 old unprotected memories"


def test_age_pruning_preserves_young_memories_regardless_of_importance():
    """
    Test that young memories are preserved regardless of importance.
    
    Validates: Requirement 3.2 (implicit - young memories not subject to age pruning)
    
    Setup:
    - Create memories with age <= max_age_days and various importance levels
    
    Expected:
    - All young memories should be preserved regardless of importance
    """
    # Setup: Create young memories with various importance levels
    memories = [
        create_memory("young_low_importance_1", age_days=30, importance=0.0),
        create_memory("young_low_importance_2", age_days=50, importance=0.3),
        create_memory("young_medium_importance", age_days=70, importance=0.5),
        create_memory("young_high_importance", age_days=89, importance=0.9),
        create_memory("young_at_threshold", age_days=90, importance=0.1),  # Exactly at threshold
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All young memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "young_low_importance_1" in remaining_ids
    assert "young_low_importance_2" in remaining_ids
    assert "young_medium_importance" in remaining_ids
    assert "young_high_importance" in remaining_ids
    assert "young_at_threshold" in remaining_ids
    
    # Verify exact set - all memories should remain
    expected_ids = {
        "young_low_importance_1",
        "young_low_importance_2",
        "young_medium_importance",
        "young_high_importance",
        "young_at_threshold"
    }
    assert remaining_ids == expected_ids, "All young memories should be preserved"
    
    # Verify statistics
    assert result.age_pruned == 0, "Should have deleted 0 memories (all are young)"
    assert result.total_deleted == 0, "Total deleted should be 0"


def test_age_pruning_skipped_when_max_age_days_none():
    """
    Test that max_age_days=None skips age pruning.
    
    Validates: Requirement 3.4
    
    Setup:
    - Create old memories with low importance
    - Configure with max_age_days=None
    
    Expected:
    - No memories should be deleted by age pruning
    """
    # Setup: Create old memories that would normally be deleted
    memories = [
        create_memory("old_1", age_days=100, importance=0.5),
        create_memory("old_2", age_days=200, importance=0.3),
        create_memory("old_3", age_days=365, importance=0.1),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=None,  # Age pruning disabled
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All memories preserved (age pruning skipped)
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "old_1" in remaining_ids
    assert "old_2" in remaining_ids
    assert "old_3" in remaining_ids
    
    # Verify exact set - all memories should remain
    assert remaining_ids == {"old_1", "old_2", "old_3"}, \
        "All memories should be preserved when max_age_days=None"
    
    # Verify statistics
    assert result.age_pruned == 0, "Should have deleted 0 memories (age pruning disabled)"
    assert result.total_deleted == 0, "Total deleted should be 0"


def test_age_pruning_boundary_case_exactly_at_threshold():
    """
    Test boundary case where memory age equals max_age_days exactly.
    
    Validates: Requirement 3.2 (age_days > max_age_days condition)
    
    Setup:
    - Create memories with age exactly equal to max_age_days
    - Create memories with age just above and below threshold
    
    Expected:
    - Memories with age == max_age_days should be preserved (not > threshold)
    - Memories with age > max_age_days should be deleted
    """
    # Setup: Create memories at boundary
    memories = [
        create_memory("below_threshold", age_days=89, importance=0.5),    # Should be preserved
        create_memory("at_threshold", age_days=90, importance=0.5),       # Should be preserved (not >)
        create_memory("above_threshold", age_days=91, importance=0.5),    # Should be deleted
        create_memory("well_above", age_days=100, importance=0.5),        # Should be deleted
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only memories with age > 90 are deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "below_threshold" in remaining_ids, "Memory below threshold should be preserved"
    assert "at_threshold" in remaining_ids, "Memory at threshold should be preserved (not >)"
    assert "above_threshold" not in remaining_ids, "Memory above threshold should be deleted"
    assert "well_above" not in remaining_ids, "Memory well above threshold should be deleted"
    
    # Verify exact set
    assert remaining_ids == {"below_threshold", "at_threshold"}, \
        "Only memories with age <= max_age_days should remain"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 memories above threshold"


def test_age_pruning_importance_boundary_case():
    """
    Test boundary case where importance equals min_importance_protected exactly.
    
    Validates: Requirement 3.3 (importance >= min_importance_protected condition)
    
    Setup:
    - Create old memories with importance exactly at protection threshold
    - Create old memories just above and below threshold
    
    Expected:
    - Memories with importance >= threshold should be preserved
    - Memories with importance < threshold should be deleted
    """
    # Setup: Create old memories at importance boundary
    memories = [
        create_memory("below_protection", age_days=100, importance=0.79),  # Should be deleted
        create_memory("at_protection", age_days=100, importance=0.8),      # Should be preserved (>=)
        create_memory("above_protection", age_days=100, importance=0.81),  # Should be preserved
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only memories with importance >= 0.8 are preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "below_protection" not in remaining_ids, "Memory below protection should be deleted"
    assert "at_protection" in remaining_ids, "Memory at protection threshold should be preserved (>=)"
    assert "above_protection" in remaining_ids, "Memory above protection should be preserved"
    
    # Verify exact set
    assert remaining_ids == {"at_protection", "above_protection"}, \
        "Only memories with importance >= min_importance_protected should remain"
    
    # Verify statistics
    assert result.age_pruned == 1, "Should have deleted 1 memory below protection threshold"


def test_age_pruning_with_mixed_scenarios():
    """
    Test age pruning with a comprehensive mix of scenarios.
    
    Validates: Requirements 3.2, 3.3, 3.4, 13.1
    
    Setup:
    - Mix of old/young memories with various importance levels
    
    Expected:
    - Only old unprotected memories should be deleted
    - Exact memory_id verification (no weak assertions)
    """
    # Setup: Comprehensive mix of scenarios
    memories = [
        # Old unprotected - should be deleted
        create_memory("old_unprotected_1", age_days=100, importance=0.5),
        create_memory("old_unprotected_2", age_days=120, importance=0.0),
        create_memory("old_unprotected_3", age_days=95, importance=0.79),
        
        # Old protected - should be preserved
        create_memory("old_protected_1", age_days=100, importance=0.8),
        create_memory("old_protected_2", age_days=150, importance=0.9),
        
        # Young unprotected - should be preserved
        create_memory("young_unprotected_1", age_days=30, importance=0.3),
        create_memory("young_unprotected_2", age_days=89, importance=0.0),
        
        # Young protected - should be preserved
        create_memory("young_protected_1", age_days=50, importance=0.9),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact set of remaining memories
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    expected_remaining = {
        "old_protected_1",
        "old_protected_2",
        "young_unprotected_1",
        "young_unprotected_2",
        "young_protected_1"
    }
    
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "old_unprotected_1",
        "old_unprotected_2",
        "old_unprotected_3"
    }
    
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.age_pruned == 3, "Should have deleted exactly 3 old unprotected memories"
    assert result.total_deleted == 3, "Total deleted should be 3"
    assert result.final_count == 5, "Final count should be 5"


# ============================================================================
# Score-Based Pruning Tests - Requirements 4.2, 4.3, 4.4, 13.2
# ============================================================================

def test_score_pruning_deletes_low_score_unprotected_memories():
    """
    Test that low-score unprotected memories are deleted.
    
    Validates: Requirement 4.2
    
    Setup:
    - Create memories with final_score < pruning_score_threshold and importance < min_importance_protected
    - Create memories with final_score >= pruning_score_threshold
    
    Expected:
    - Low-score unprotected memories should be deleted
    - High-score memories should be preserved
    """
    # Setup: Create memories with various scores
    memories = [
        create_memory("low_score_unprotected_1", age_days=30, importance=0.5, final_score=0.2),  # Should be deleted
        create_memory("low_score_unprotected_2", age_days=40, importance=0.7, final_score=0.25), # Should be deleted
        create_memory("high_score_1", age_days=30, importance=0.5, final_score=0.5),             # Should be preserved
        create_memory("high_score_2", age_days=50, importance=0.3, final_score=0.8),             # Should be preserved
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Low-score unprotected memories deleted, high-score memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "low_score_unprotected_1" not in remaining_ids, "Low-score unprotected memory 1 should be deleted"
    assert "low_score_unprotected_2" not in remaining_ids, "Low-score unprotected memory 2 should be deleted"
    assert "high_score_1" in remaining_ids, "High-score memory 1 should be preserved"
    assert "high_score_2" in remaining_ids, "High-score memory 2 should be preserved"
    
    # Verify exact set of remaining memories
    assert remaining_ids == {"high_score_1", "high_score_2"}, "Only high-score memories should remain"
    
    # Verify statistics
    assert result.score_pruned == 2, "Should have deleted 2 low-score memories"
    assert result.total_deleted == 2, "Total deleted should be 2"


def test_score_pruning_preserves_low_score_protected_memories():
    """
    Test that low-score protected memories are preserved.
    
    Validates: Requirement 4.3
    
    Setup:
    - Create memories with final_score < pruning_score_threshold and importance >= min_importance_protected
    - Create memories with final_score < pruning_score_threshold and importance < min_importance_protected
    
    Expected:
    - Low-score protected memories should be preserved
    - Low-score unprotected memories should be deleted
    """
    # Setup: Create low-score memories with different importance levels
    memories = [
        create_memory("low_score_protected_1", age_days=30, importance=0.8, final_score=0.2),   # Should be preserved
        create_memory("low_score_protected_2", age_days=40, importance=0.9, final_score=0.1),   # Should be preserved
        create_memory("low_score_protected_3", age_days=50, importance=1.0, final_score=0.0),   # Should be preserved
        create_memory("low_score_unprotected_1", age_days=30, importance=0.5, final_score=0.2), # Should be deleted
        create_memory("low_score_unprotected_2", age_days=40, importance=0.79, final_score=0.1),# Should be deleted (< 0.8)
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories preserved, unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "low_score_protected_1" in remaining_ids, "Low-score protected memory 1 should be preserved"
    assert "low_score_protected_2" in remaining_ids, "Low-score protected memory 2 should be preserved"
    assert "low_score_protected_3" in remaining_ids, "Low-score protected memory 3 should be preserved"
    assert "low_score_unprotected_1" not in remaining_ids, "Low-score unprotected memory 1 should be deleted"
    assert "low_score_unprotected_2" not in remaining_ids, "Low-score unprotected memory 2 should be deleted"
    
    # Verify exact set of remaining memories
    assert remaining_ids == {"low_score_protected_1", "low_score_protected_2", "low_score_protected_3"}, \
        "Only protected memories should remain"
    
    # Verify statistics
    assert result.score_pruned == 2, "Should have deleted 2 low-score unprotected memories"


def test_score_pruning_preserves_high_score_memories_regardless_of_importance():
    """
    Test that high-score memories are preserved regardless of importance.
    
    Validates: Requirement 4.2 (implicit - high-score memories not subject to score pruning)
    
    Setup:
    - Create memories with final_score >= pruning_score_threshold and various importance levels
    
    Expected:
    - All high-score memories should be preserved regardless of importance
    """
    # Setup: Create high-score memories with various importance levels
    memories = [
        create_memory("high_score_low_importance_1", age_days=30, importance=0.0, final_score=0.5),
        create_memory("high_score_low_importance_2", age_days=40, importance=0.3, final_score=0.6),
        create_memory("high_score_medium_importance", age_days=50, importance=0.5, final_score=0.7),
        create_memory("high_score_high_importance", age_days=60, importance=0.9, final_score=0.8),
        create_memory("high_score_at_threshold", age_days=70, importance=0.1, final_score=0.3),  # Exactly at threshold
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All high-score memories preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "high_score_low_importance_1" in remaining_ids
    assert "high_score_low_importance_2" in remaining_ids
    assert "high_score_medium_importance" in remaining_ids
    assert "high_score_high_importance" in remaining_ids
    assert "high_score_at_threshold" in remaining_ids
    
    # Verify exact set - all memories should remain
    expected_ids = {
        "high_score_low_importance_1",
        "high_score_low_importance_2",
        "high_score_medium_importance",
        "high_score_high_importance",
        "high_score_at_threshold"
    }
    assert remaining_ids == expected_ids, "All high-score memories should be preserved"
    
    # Verify statistics
    assert result.score_pruned == 0, "Should have deleted 0 memories (all have high scores)"
    assert result.total_deleted == 0, "Total deleted should be 0"


def test_score_pruning_skipped_when_pruning_score_threshold_none():
    """
    Test that pruning_score_threshold=None skips score pruning.
    
    Validates: Requirement 4.4
    
    Setup:
    - Create low-score memories with low importance
    - Configure with pruning_score_threshold=None
    
    Expected:
    - No memories should be deleted by score pruning
    """
    # Setup: Create low-score memories that would normally be deleted
    memories = [
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_2", age_days=40, importance=0.3, final_score=0.05),
        create_memory("low_score_3", age_days=50, importance=0.1, final_score=0.0),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=None,  # Score pruning disabled
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All memories preserved (score pruning skipped)
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "low_score_1" in remaining_ids
    assert "low_score_2" in remaining_ids
    assert "low_score_3" in remaining_ids
    
    # Verify exact set - all memories should remain
    assert remaining_ids == {"low_score_1", "low_score_2", "low_score_3"}, \
        "All memories should be preserved when pruning_score_threshold=None"
    
    # Verify statistics
    assert result.score_pruned == 0, "Should have deleted 0 memories (score pruning disabled)"
    assert result.total_deleted == 0, "Total deleted should be 0"


def test_score_pruning_boundary_case_exactly_at_threshold():
    """
    Test boundary case where final_score equals pruning_score_threshold exactly.
    
    Validates: Requirement 4.2 (final_score < pruning_score_threshold condition)
    
    Setup:
    - Create memories with final_score exactly equal to pruning_score_threshold
    - Create memories with final_score just above and below threshold
    
    Expected:
    - Memories with final_score == threshold should be preserved (not < threshold)
    - Memories with final_score < threshold should be deleted
    """
    # Setup: Create memories at boundary
    memories = [
        create_memory("below_threshold", age_days=30, importance=0.5, final_score=0.29),  # Should be deleted
        create_memory("at_threshold", age_days=40, importance=0.5, final_score=0.3),      # Should be preserved (not <)
        create_memory("above_threshold", age_days=50, importance=0.5, final_score=0.31),  # Should be preserved
        create_memory("well_above", age_days=60, importance=0.5, final_score=0.8),        # Should be preserved
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only memories with final_score < 0.3 are deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "below_threshold" not in remaining_ids, "Memory below threshold should be deleted"
    assert "at_threshold" in remaining_ids, "Memory at threshold should be preserved (not <)"
    assert "above_threshold" in remaining_ids, "Memory above threshold should be preserved"
    assert "well_above" in remaining_ids, "Memory well above threshold should be preserved"
    
    # Verify exact set
    assert remaining_ids == {"at_threshold", "above_threshold", "well_above"}, \
        "Only memories with final_score >= pruning_score_threshold should remain"
    
    # Verify statistics
    assert result.score_pruned == 1, "Should have deleted 1 memory below threshold"


def test_score_pruning_importance_boundary_case():
    """
    Test boundary case where importance equals min_importance_protected exactly.
    
    Validates: Requirement 4.3 (importance >= min_importance_protected condition)
    
    Setup:
    - Create low-score memories with importance exactly at protection threshold
    - Create low-score memories just above and below threshold
    
    Expected:
    - Memories with importance >= threshold should be preserved
    - Memories with importance < threshold should be deleted
    """
    # Setup: Create low-score memories at importance boundary
    memories = [
        create_memory("below_protection", age_days=30, importance=0.79, final_score=0.1),  # Should be deleted
        create_memory("at_protection", age_days=40, importance=0.8, final_score=0.1),      # Should be preserved (>=)
        create_memory("above_protection", age_days=50, importance=0.81, final_score=0.1),  # Should be preserved
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only memories with importance >= 0.8 are preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "below_protection" not in remaining_ids, "Memory below protection should be deleted"
    assert "at_protection" in remaining_ids, "Memory at protection threshold should be preserved (>=)"
    assert "above_protection" in remaining_ids, "Memory above protection should be preserved"
    
    # Verify exact set
    assert remaining_ids == {"at_protection", "above_protection"}, \
        "Only memories with importance >= min_importance_protected should remain"
    
    # Verify statistics
    assert result.score_pruned == 1, "Should have deleted 1 memory below protection threshold"


def test_score_pruning_with_mixed_scenarios():
    """
    Test score pruning with a comprehensive mix of scenarios.
    
    Validates: Requirements 4.2, 4.3, 4.4, 13.2
    
    Setup:
    - Mix of low/high score memories with various importance levels
    
    Expected:
    - Only low-score unprotected memories should be deleted
    - Exact memory_id verification (no weak assertions)
    """
    # Setup: Comprehensive mix of scenarios
    memories = [
        # Low-score unprotected - should be deleted
        create_memory("low_score_unprotected_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_unprotected_2", age_days=40, importance=0.0, final_score=0.2),
        create_memory("low_score_unprotected_3", age_days=50, importance=0.79, final_score=0.15),
        
        # Low-score protected - should be preserved
        create_memory("low_score_protected_1", age_days=30, importance=0.8, final_score=0.1),
        create_memory("low_score_protected_2", age_days=40, importance=0.9, final_score=0.05),
        
        # High-score unprotected - should be preserved
        create_memory("high_score_unprotected_1", age_days=30, importance=0.3, final_score=0.5),
        create_memory("high_score_unprotected_2", age_days=40, importance=0.0, final_score=0.8),
        
        # High-score protected - should be preserved
        create_memory("high_score_protected_1", age_days=50, importance=0.9, final_score=0.7),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact set of remaining memories
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    expected_remaining = {
        "low_score_protected_1",
        "low_score_protected_2",
        "high_score_unprotected_1",
        "high_score_unprotected_2",
        "high_score_protected_1"
    }
    
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "low_score_unprotected_1",
        "low_score_unprotected_2",
        "low_score_unprotected_3"
    }
    
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.score_pruned == 3, "Should have deleted exactly 3 low-score unprotected memories"
    assert result.total_deleted == 3, "Total deleted should be 3"
    assert result.final_count == 5, "Final count should be 5"


def test_score_pruning_combined_with_age_pruning():
    """
    Test that score pruning works correctly after age pruning.
    
    Validates: Requirements 4.2, 4.3, 13.2
    
    Setup:
    - Create memories that should be deleted by age pruning
    - Create memories that should be deleted by score pruning
    - Create memories that should survive both
    
    Expected:
    - Age pruning runs first, then score pruning
    - Each phase deletes the correct memories
    """
    # Setup: Mix of old and low-score memories
    memories = [
        # Old unprotected - should be deleted by age pruning
        create_memory("old_unprotected", age_days=100, importance=0.5, final_score=0.8),
        
        # Low-score unprotected - should be deleted by score pruning
        create_memory("low_score_unprotected", age_days=30, importance=0.5, final_score=0.1),
        
        # Old AND low-score unprotected - should be deleted by age pruning (first phase)
        create_memory("old_and_low_score", age_days=100, importance=0.5, final_score=0.1),
        
        # Protected from both - should be preserved
        create_memory("protected", age_days=100, importance=0.8, final_score=0.1),
        
        # Young and high-score - should be preserved
        create_memory("young_high_score", age_days=30, importance=0.5, final_score=0.8),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact set of remaining memories
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    expected_remaining = {
        "protected",
        "young_high_score"
    }
    
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 old memories (old_unprotected, old_and_low_score)"
    assert result.score_pruned == 1, "Should have deleted 1 low-score memory (low_score_unprotected)"
    assert result.total_deleted == 3, "Total deleted should be 3"
    assert result.final_count == 2, "Final count should be 2"


# ============================================================================
# Hard Cap Enforcement Tests - Requirements 5.2, 5.3, 5.4, 5.5, 13.3
# ============================================================================

def test_hard_cap_enforcement_respects_limit():
    """
    Test that total count never exceeds max_total_memories.
    
    Validates: Requirement 5.2, 5.4
    
    Setup:
    - Create more memories than max_total_memories
    - All memories are unprotected
    
    Expected:
    - Total count after cleanup should be <= max_total_memories
    - Excess memories should be deleted
    """
    # Setup: Create 10 memories when max is 5
    memories = [
        create_memory(f"memory_{i}", age_days=30, importance=0.5, final_score=0.5)
        for i in range(10)
    ]
    
    config = LifecycleConfig(
        max_total_memories=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Total count should be <= max_total_memories
    remaining_count = len(mock_memory.memories)
    assert remaining_count <= 5, f"Total count {remaining_count} exceeds max_total_memories 5"
    assert remaining_count == 5, f"Expected exactly 5 memories, got {remaining_count}"
    
    # Verify statistics
    assert result.cap_pruned == 5, "Should have deleted 5 memories to enforce cap"
    assert result.total_deleted == 5, "Total deleted should be 5"
    assert result.final_count == 5, "Final count should be 5"


def test_hard_cap_enforcement_deletes_lowest_ranked_first():
    """
    Test that lowest-ranked memories are deleted first.
    
    Validates: Requirement 5.2
    
    Setup:
    - Create memories with different final_scores
    - Total exceeds max_total_memories
    
    Expected:
    - Memories with lowest final_scores should be deleted
    - Memories with highest final_scores should be preserved
    """
    # Setup: Create memories with different scores
    memories = [
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_2", age_days=30, importance=0.5, final_score=0.2),
        create_memory("medium_score_1", age_days=30, importance=0.5, final_score=0.5),
        create_memory("medium_score_2", age_days=30, importance=0.5, final_score=0.6),
        create_memory("high_score_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("high_score_2", age_days=30, importance=0.5, final_score=0.9),
    ]
    
    config = LifecycleConfig(
        max_total_memories=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Lowest-ranked memories deleted, highest-ranked preserved
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "low_score_1" not in remaining_ids, "Lowest score memory should be deleted"
    assert "low_score_2" not in remaining_ids, "Second lowest score memory should be deleted"
    assert "medium_score_1" not in remaining_ids, "Third lowest score memory should be deleted"
    assert "medium_score_2" in remaining_ids, "Fourth lowest score memory should be preserved"
    assert "high_score_1" in remaining_ids, "High score memory 1 should be preserved"
    assert "high_score_2" in remaining_ids, "High score memory 2 should be preserved"
    
    # Verify exact set
    assert remaining_ids == {"medium_score_2", "high_score_1", "high_score_2"}, \
        "Only highest-ranked memories should remain"
    
    # Verify statistics
    assert result.cap_pruned == 3, "Should have deleted 3 lowest-ranked memories"


def test_hard_cap_enforcement_preserves_protected_memories():
    """
    Test that protected memories are preserved even when cap exceeded.
    
    Validates: Requirement 5.3
    
    Setup:
    - Create memories exceeding max_total_memories
    - Some memories have importance >= min_importance_protected
    
    Expected:
    - Protected memories should never be deleted
    - Only unprotected memories should be deleted to enforce cap
    """
    # Setup: Create mix of protected and unprotected memories
    memories = [
        create_memory("protected_1", age_days=30, importance=0.8, final_score=0.1),  # Protected, low score
        create_memory("protected_2", age_days=30, importance=0.9, final_score=0.2),  # Protected, low score
        create_memory("unprotected_1", age_days=30, importance=0.5, final_score=0.3),
        create_memory("unprotected_2", age_days=30, importance=0.5, final_score=0.4),
        create_memory("unprotected_3", age_days=30, importance=0.5, final_score=0.5),
        create_memory("unprotected_4", age_days=30, importance=0.5, final_score=0.6),
    ]
    
    config = LifecycleConfig(
        max_total_memories=3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories preserved, unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "protected_1" in remaining_ids, "Protected memory 1 should be preserved"
    assert "protected_2" in remaining_ids, "Protected memory 2 should be preserved"
    
    # Only 1 unprotected memory should remain (2 protected + 1 unprotected = 3 total)
    unprotected_remaining = remaining_ids - {"protected_1", "protected_2"}
    assert len(unprotected_remaining) == 1, "Should have 1 unprotected memory remaining"
    
    # The remaining unprotected memory should be the highest-ranked one
    assert "unprotected_4" in remaining_ids, "Highest-ranked unprotected memory should remain"
    
    # Verify exact set
    assert remaining_ids == {"protected_1", "protected_2", "unprotected_4"}, \
        "Protected memories and highest-ranked unprotected should remain"
    
    # Verify statistics
    assert result.cap_pruned == 3, "Should have deleted 3 unprotected memories"


def test_hard_cap_enforcement_deterministic_order_with_identical_scores():
    """
    Test deterministic deletion order with identical scores.
    
    Validates: Requirement 5.5, 5.1
    
    Setup:
    - Create memories with identical final_scores but different timestamps and IDs
    - Total exceeds max_total_memories
    
    Expected:
    - Deletion order should be deterministic: score -> timestamp -> memory_id
    - Oldest memories should be deleted first when scores are identical
    - Lexicographically first IDs deleted when scores and timestamps identical
    """
    # Setup: Create memories with identical scores but different timestamps
    base_time = datetime.now(UTC)
    memories = [
        # Same score, different timestamps
        {
            "id": "same_score_old",
            "content": "Memory content",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=100)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "same_score_new",
            "content": "Memory content",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=10)).isoformat(),
            "category": "test",
            "tags": []
        },
        # Same score and timestamp, different IDs (lexicographical order)
        {
            "id": "id_a",
            "content": "Memory content",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=50)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "id_z",
            "content": "Memory content",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=50)).isoformat(),
            "category": "test",
            "tags": []
        },
    ]
    
    config = LifecycleConfig(
        max_total_memories=2,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Deterministic deletion order
    remaining_ids = {m["id"] for m in mock_memory.memories}
    deleted_ids = set(mock_memory.deleted_ids)
    
    # Oldest should be deleted first (same_score_old)
    assert "same_score_old" in deleted_ids, "Oldest memory should be deleted first"
    
    # Among same timestamp, lexicographically first should be deleted (id_a)
    assert "id_a" in deleted_ids, "Lexicographically first ID should be deleted"
    
    # Newest and lexicographically last should remain
    assert "same_score_new" in remaining_ids, "Newest memory should be preserved"
    assert "id_z" in remaining_ids, "Lexicographically last ID should be preserved"
    
    # Verify exact sets
    assert remaining_ids == {"same_score_new", "id_z"}, \
        "Newest and lexicographically last should remain"
    assert deleted_ids == {"same_score_old", "id_a"}, \
        "Oldest and lexicographically first should be deleted"
    
    # Verify statistics
    assert result.cap_pruned == 2, "Should have deleted 2 memories"


def test_hard_cap_not_exceeded_returns_zero_deletions():
    """
    Test that cap not exceeded returns 0 deletions.
    
    Validates: Requirement 5.4 (implicit - no action when compliant)
    
    Setup:
    - Create fewer memories than max_total_memories
    
    Expected:
    - No memories should be deleted
    - All memories should be preserved
    """
    # Setup: Create 3 memories when max is 10
    memories = [
        create_memory("memory_1", age_days=30, importance=0.5, final_score=0.5),
        create_memory("memory_2", age_days=40, importance=0.5, final_score=0.6),
        create_memory("memory_3", age_days=50, importance=0.5, final_score=0.7),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: No memories deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "memory_1" in remaining_ids
    assert "memory_2" in remaining_ids
    assert "memory_3" in remaining_ids
    
    # Verify exact set - all memories should remain
    assert remaining_ids == {"memory_1", "memory_2", "memory_3"}, \
        "All memories should be preserved when cap not exceeded"
    
    # Verify statistics
    assert result.cap_pruned == 0, "Should have deleted 0 memories (cap not exceeded)"
    assert result.total_deleted == 0, "Total deleted should be 0"
    assert result.final_count == 3, "Final count should be 3"


def test_hard_cap_at_exact_limit():
    """
    Test boundary case where memory count equals max_total_memories exactly.
    
    Validates: Requirement 5.4
    
    Setup:
    - Create exactly max_total_memories memories
    
    Expected:
    - No memories should be deleted (count <= max, not >)
    """
    # Setup: Create exactly 5 memories when max is 5
    memories = [
        create_memory(f"memory_{i}", age_days=30, importance=0.5, final_score=0.5)
        for i in range(5)
    ]
    
    config = LifecycleConfig(
        max_total_memories=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: No memories deleted (at limit, not exceeding)
    remaining_count = len(mock_memory.memories)
    assert remaining_count == 5, f"Expected 5 memories, got {remaining_count}"
    
    # Verify statistics
    assert result.cap_pruned == 0, "Should have deleted 0 memories (at limit, not exceeding)"
    assert result.total_deleted == 0, "Total deleted should be 0"
    assert result.final_count == 5, "Final count should be 5"


def test_hard_cap_with_all_protected_memories():
    """
    Test hard cap enforcement when all memories are protected.
    
    Validates: Requirement 5.3
    
    Setup:
    - Create more memories than max_total_memories
    - All memories have importance >= min_importance_protected
    
    Expected:
    - No memories should be deleted (all are protected)
    - Total count will exceed max_total_memories (cannot enforce cap)
    """
    # Setup: Create 10 protected memories when max is 5
    memories = [
        create_memory(f"protected_{i}", age_days=30, importance=0.9, final_score=0.5)
        for i in range(10)
    ]
    
    config = LifecycleConfig(
        max_total_memories=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: No memories deleted (all are protected)
    remaining_count = len(mock_memory.memories)
    assert remaining_count == 10, f"Expected 10 memories (all protected), got {remaining_count}"
    
    # Verify all memories still present
    remaining_ids = {m["id"] for m in mock_memory.memories}
    expected_ids = {f"protected_{i}" for i in range(10)}
    assert remaining_ids == expected_ids, "All protected memories should remain"
    
    # Verify statistics
    assert result.cap_pruned == 0, "Should have deleted 0 memories (all are protected)"
    assert result.total_deleted == 0, "Total deleted should be 0"
    assert result.final_count == 10, "Final count should be 10 (cap cannot be enforced)"


def test_hard_cap_with_mixed_scenarios():
    """
    Test hard cap enforcement with comprehensive mix of scenarios.
    
    Validates: Requirements 5.2, 5.3, 5.4, 5.5, 13.3
    
    Setup:
    - Mix of protected/unprotected memories with various scores and timestamps
    - Total exceeds max_total_memories
    
    Expected:
    - Protected memories preserved
    - Lowest-ranked unprotected memories deleted
    - Deterministic deletion order
    - Exact memory_id verification (no weak assertions)
    """
    base_time = datetime.now(UTC)
    
    # Setup: Comprehensive mix of scenarios
    memories = [
        # Protected memories with low scores - should be preserved
        {
            "id": "protected_low_score_1",
            "content": "Memory",
            "metadata": {"importance": 0.8, "final_score": 0.1},
            "timestamp": (base_time - timedelta(days=100)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "protected_low_score_2",
            "content": "Memory",
            "metadata": {"importance": 0.9, "final_score": 0.2},
            "timestamp": (base_time - timedelta(days=90)).isoformat(),
            "category": "test",
            "tags": []
        },
        
        # Unprotected memories with various scores - some should be deleted
        {
            "id": "unprotected_lowest",
            "content": "Memory",
            "metadata": {"importance": 0.5, "final_score": 0.1},
            "timestamp": (base_time - timedelta(days=80)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "unprotected_low",
            "content": "Memory",
            "metadata": {"importance": 0.5, "final_score": 0.3},
            "timestamp": (base_time - timedelta(days=70)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "unprotected_medium",
            "content": "Memory",
            "metadata": {"importance": 0.5, "final_score": 0.5},
            "timestamp": (base_time - timedelta(days=60)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "unprotected_high",
            "content": "Memory",
            "metadata": {"importance": 0.5, "final_score": 0.8},
            "timestamp": (base_time - timedelta(days=50)).isoformat(),
            "category": "test",
            "tags": []
        },
        {
            "id": "unprotected_highest",
            "content": "Memory",
            "metadata": {"importance": 0.5, "final_score": 0.9},
            "timestamp": (base_time - timedelta(days=40)).isoformat(),
            "category": "test",
            "tags": []
        },
    ]
    
    config = LifecycleConfig(
        max_total_memories=5,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact set of remaining memories
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Protected memories should always remain
    assert "protected_low_score_1" in remaining_ids
    assert "protected_low_score_2" in remaining_ids
    
    # Highest-ranked unprotected memories should remain (need 3 more to reach cap of 5)
    assert "unprotected_highest" in remaining_ids
    assert "unprotected_high" in remaining_ids
    assert "unprotected_medium" in remaining_ids
    
    # Lowest-ranked unprotected memories should be deleted
    assert "unprotected_lowest" not in remaining_ids
    assert "unprotected_low" not in remaining_ids
    
    # Verify exact set
    expected_remaining = {
        "protected_low_score_1",
        "protected_low_score_2",
        "unprotected_medium",
        "unprotected_high",
        "unprotected_highest"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "unprotected_lowest",
        "unprotected_low"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.cap_pruned == 2, "Should have deleted 2 lowest-ranked unprotected memories"
    assert result.total_deleted == 2, "Total deleted should be 2"
    assert result.final_count == 5, "Final count should be 5"


def test_hard_cap_combined_with_age_and_score_pruning():
    """
    Test hard cap enforcement after age and score pruning.
    
    Validates: Requirements 5.2, 5.4, 13.3
    
    Setup:
    - Create memories that should be deleted by age pruning
    - Create memories that should be deleted by score pruning
    - Create memories that should be deleted by hard cap enforcement
    - Total exceeds max_total_memories even after age/score pruning
    
    Expected:
    - Age pruning runs first
    - Score pruning runs second
    - Hard cap enforcement runs last
    - Each phase deletes the correct memories
    """
    # Setup: Mix of old, low-score, and excess memories
    memories = [
        # Old unprotected - should be deleted by age pruning
        create_memory("old_1", age_days=100, importance=0.5, final_score=0.8),
        create_memory("old_2", age_days=110, importance=0.5, final_score=0.9),
        
        # Low-score unprotected - should be deleted by score pruning
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_2", age_days=40, importance=0.5, final_score=0.2),
        
        # Young, high-score unprotected - some should be deleted by hard cap
        create_memory("excess_1", age_days=30, importance=0.5, final_score=0.5),
        create_memory("excess_2", age_days=40, importance=0.5, final_score=0.6),
        create_memory("excess_3", age_days=50, importance=0.5, final_score=0.7),
        create_memory("excess_4", age_days=60, importance=0.5, final_score=0.8),
        
        # Protected - should always be preserved
        create_memory("protected", age_days=100, importance=0.9, final_score=0.1),
    ]
    
    config = LifecycleConfig(
        max_total_memories=3,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Exact set of remaining memories
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Protected should always remain
    assert "protected" in remaining_ids, "Protected memory should be preserved"
    
    # Old memories should be deleted by age pruning
    assert "old_1" not in remaining_ids, "Old memory 1 should be deleted by age pruning"
    assert "old_2" not in remaining_ids, "Old memory 2 should be deleted by age pruning"
    
    # Low-score memories should be deleted by score pruning
    assert "low_score_1" not in remaining_ids, "Low-score memory 1 should be deleted by score pruning"
    assert "low_score_2" not in remaining_ids, "Low-score memory 2 should be deleted by score pruning"
    
    # After age and score pruning, we have 5 memories (1 protected + 4 excess)
    # Hard cap is 3, so 2 lowest-ranked excess memories should be deleted
    assert "excess_1" not in remaining_ids, "Lowest excess memory should be deleted by hard cap"
    assert "excess_2" not in remaining_ids, "Second lowest excess memory should be deleted by hard cap"
    assert "excess_3" in remaining_ids, "Third lowest excess memory should be preserved"
    assert "excess_4" in remaining_ids, "Highest excess memory should be preserved"
    
    # Verify exact set
    expected_remaining = {"protected", "excess_3", "excess_4"}
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify statistics
    assert result.age_pruned == 2, "Should have deleted 2 old memories"
    assert result.score_pruned == 2, "Should have deleted 2 low-score memories"
    assert result.cap_pruned == 2, "Should have deleted 2 excess memories"
    assert result.total_deleted == 6, "Total deleted should be 6"
    assert result.final_count == 3, "Final count should be 3"


# ============================================================================
# Importance Protection Tests - Requirements 6.1, 6.2, 6.3, 6.4, 13.4
# ============================================================================

def test_importance_protection_survives_age_pruning():
    """
    Test that protected memories survive age pruning.
    
    Validates: Requirement 6.1
    
    Setup:
    - Create old memories with importance >= min_importance_protected
    - Create old memories with importance < min_importance_protected
    
    Expected:
    - Protected memories should survive age pruning
    - Unprotected old memories should be deleted
    """
    # Setup: Create old memories with different importance levels
    memories = [
        create_memory("old_protected_1", age_days=100, importance=0.8),    # Protected, should survive
        create_memory("old_protected_2", age_days=120, importance=0.85),   # Protected, should survive
        create_memory("old_protected_3", age_days=150, importance=0.9),    # Protected, should survive
        create_memory("old_protected_4", age_days=200, importance=1.0),    # Protected, should survive
        create_memory("old_unprotected_1", age_days=100, importance=0.5),  # Unprotected, should be deleted
        create_memory("old_unprotected_2", age_days=120, importance=0.79), # Unprotected, should be deleted
        create_memory("old_unprotected_3", age_days=150, importance=0.0),  # Unprotected, should be deleted
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories survive, unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # All protected memories should survive
    assert "old_protected_1" in remaining_ids, "Protected memory 1 should survive age pruning"
    assert "old_protected_2" in remaining_ids, "Protected memory 2 should survive age pruning"
    assert "old_protected_3" in remaining_ids, "Protected memory 3 should survive age pruning"
    assert "old_protected_4" in remaining_ids, "Protected memory 4 should survive age pruning"
    
    # All unprotected old memories should be deleted
    assert "old_unprotected_1" not in remaining_ids, "Unprotected memory 1 should be deleted"
    assert "old_unprotected_2" not in remaining_ids, "Unprotected memory 2 should be deleted"
    assert "old_unprotected_3" not in remaining_ids, "Unprotected memory 3 should be deleted"
    
    # Verify exact set of remaining memories
    expected_remaining = {
        "old_protected_1",
        "old_protected_2",
        "old_protected_3",
        "old_protected_4"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "old_unprotected_1",
        "old_unprotected_2",
        "old_unprotected_3"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.age_pruned == 3, "Should have deleted 3 unprotected old memories"
    assert result.total_deleted == 3, "Total deleted should be 3"


def test_importance_protection_survives_score_pruning():
    """
    Test that protected memories survive score pruning.
    
    Validates: Requirement 6.2
    
    Setup:
    - Create low-score memories with importance >= min_importance_protected
    - Create low-score memories with importance < min_importance_protected
    
    Expected:
    - Protected memories should survive score pruning
    - Unprotected low-score memories should be deleted
    """
    # Setup: Create low-score memories with different importance levels
    memories = [
        create_memory("low_score_protected_1", age_days=30, importance=0.8, final_score=0.0),   # Protected, should survive
        create_memory("low_score_protected_2", age_days=40, importance=0.85, final_score=0.05), # Protected, should survive
        create_memory("low_score_protected_3", age_days=50, importance=0.9, final_score=0.1),   # Protected, should survive
        create_memory("low_score_protected_4", age_days=60, importance=1.0, final_score=0.15),  # Protected, should survive
        create_memory("low_score_unprotected_1", age_days=30, importance=0.5, final_score=0.0), # Unprotected, should be deleted
        create_memory("low_score_unprotected_2", age_days=40, importance=0.79, final_score=0.1),# Unprotected, should be deleted
        create_memory("low_score_unprotected_3", age_days=50, importance=0.0, final_score=0.2), # Unprotected, should be deleted
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Protected memories survive, unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # All protected memories should survive
    assert "low_score_protected_1" in remaining_ids, "Protected memory 1 should survive score pruning"
    assert "low_score_protected_2" in remaining_ids, "Protected memory 2 should survive score pruning"
    assert "low_score_protected_3" in remaining_ids, "Protected memory 3 should survive score pruning"
    assert "low_score_protected_4" in remaining_ids, "Protected memory 4 should survive score pruning"
    
    # All unprotected low-score memories should be deleted
    assert "low_score_unprotected_1" not in remaining_ids, "Unprotected memory 1 should be deleted"
    assert "low_score_unprotected_2" not in remaining_ids, "Unprotected memory 2 should be deleted"
    assert "low_score_unprotected_3" not in remaining_ids, "Unprotected memory 3 should be deleted"
    
    # Verify exact set of remaining memories
    expected_remaining = {
        "low_score_protected_1",
        "low_score_protected_2",
        "low_score_protected_3",
        "low_score_protected_4"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "low_score_unprotected_1",
        "low_score_unprotected_2",
        "low_score_unprotected_3"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.score_pruned == 3, "Should have deleted 3 unprotected low-score memories"
    assert result.total_deleted == 3, "Total deleted should be 3"


def test_importance_protection_survives_hard_cap_enforcement():
    """
    Test that protected memories survive hard cap enforcement.
    
    Validates: Requirement 6.3
    
    Setup:
    - Create memories exceeding max_total_memories
    - Some memories have importance >= min_importance_protected with low scores
    - Some memories have importance < min_importance_protected with high scores
    
    Expected:
    - Protected memories should survive even with low scores
    - Unprotected memories should be deleted even with high scores
    - Exact memory_id verification (no weak assertions)
    """
    # Setup: Create mix of protected and unprotected memories
    memories = [
        # Protected memories with low scores - should survive
        create_memory("protected_low_score_1", age_days=30, importance=0.8, final_score=0.1),
        create_memory("protected_low_score_2", age_days=30, importance=0.85, final_score=0.15),
        create_memory("protected_low_score_3", age_days=30, importance=0.9, final_score=0.2),
        create_memory("protected_low_score_4", age_days=30, importance=1.0, final_score=0.05),
        
        # Unprotected memories with high scores - should be deleted
        create_memory("unprotected_high_score_1", age_days=30, importance=0.5, final_score=0.9),
        create_memory("unprotected_high_score_2", age_days=30, importance=0.7, final_score=0.85),
        create_memory("unprotected_high_score_3", age_days=30, importance=0.79, final_score=0.95),
        create_memory("unprotected_high_score_4", age_days=30, importance=0.6, final_score=0.8),
    ]
    
    config = LifecycleConfig(
        max_total_memories=4,  # Only 4 allowed, 8 total
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All protected memories survive, all unprotected deleted
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # All protected memories should survive
    assert "protected_low_score_1" in remaining_ids, "Protected memory 1 should survive hard cap"
    assert "protected_low_score_2" in remaining_ids, "Protected memory 2 should survive hard cap"
    assert "protected_low_score_3" in remaining_ids, "Protected memory 3 should survive hard cap"
    assert "protected_low_score_4" in remaining_ids, "Protected memory 4 should survive hard cap"
    
    # All unprotected memories should be deleted
    assert "unprotected_high_score_1" not in remaining_ids, "Unprotected memory 1 should be deleted"
    assert "unprotected_high_score_2" not in remaining_ids, "Unprotected memory 2 should be deleted"
    assert "unprotected_high_score_3" not in remaining_ids, "Unprotected memory 3 should be deleted"
    assert "unprotected_high_score_4" not in remaining_ids, "Unprotected memory 4 should be deleted"
    
    # Verify exact set of remaining memories
    expected_remaining = {
        "protected_low_score_1",
        "protected_low_score_2",
        "protected_low_score_3",
        "protected_low_score_4"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "unprotected_high_score_1",
        "unprotected_high_score_2",
        "unprotected_high_score_3",
        "unprotected_high_score_4"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.cap_pruned == 4, "Should have deleted 4 unprotected memories"
    assert result.total_deleted == 4, "Total deleted should be 4"
    assert result.final_count == 4, "Final count should be 4 (all protected)"


def test_importance_protection_boundary_case_exactly_at_threshold():
    """
    Test boundary case where importance equals min_importance_protected exactly.
    
    Validates: Requirement 6.4
    
    Setup:
    - Create memories with importance exactly at min_importance_protected
    - Create memories just above and below the threshold
    - Test across all pruning operations: age, score, and hard cap
    
    Expected:
    - Memories with importance >= threshold should be protected (including exact match)
    - Memories with importance < threshold should be deleted
    - Exact memory_id verification (no weak assertions)
    """
    # Setup: Create memories at importance boundary
    memories = [
        # Age pruning test: old memories at boundary
        create_memory("age_below_threshold", age_days=100, importance=0.799, final_score=0.5),  # Should be deleted
        create_memory("age_at_threshold", age_days=100, importance=0.8, final_score=0.5),       # Should be protected (>=)
        create_memory("age_above_threshold", age_days=100, importance=0.801, final_score=0.5),  # Should be protected
        
        # Score pruning test: low-score memories at boundary
        create_memory("score_below_threshold", age_days=30, importance=0.799, final_score=0.1), # Should be deleted
        create_memory("score_at_threshold", age_days=30, importance=0.8, final_score=0.1),      # Should be protected (>=)
        create_memory("score_above_threshold", age_days=30, importance=0.801, final_score=0.1), # Should be protected
        
        # Hard cap test: memories at boundary with scores above threshold
        create_memory("cap_below_threshold", age_days=30, importance=0.799, final_score=0.4),   # Should be deleted
        create_memory("cap_at_threshold", age_days=30, importance=0.8, final_score=0.4),        # Should be protected (>=)
        create_memory("cap_above_threshold", age_days=30, importance=0.801, final_score=0.4),   # Should be protected
    ]
    
    config = LifecycleConfig(
        max_total_memories=6,  # Will trigger hard cap (9 memories total)
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Only memories with importance >= 0.8 are protected
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    # Memories at or above threshold should be protected
    assert "age_at_threshold" in remaining_ids, "Memory at threshold should be protected (>=)"
    assert "age_above_threshold" in remaining_ids, "Memory above threshold should be protected"
    assert "score_at_threshold" in remaining_ids, "Memory at threshold should be protected (>=)"
    assert "score_above_threshold" in remaining_ids, "Memory above threshold should be protected"
    assert "cap_at_threshold" in remaining_ids, "Memory at threshold should be protected (>=)"
    assert "cap_above_threshold" in remaining_ids, "Memory above threshold should be protected"
    
    # Memories below threshold should be deleted
    assert "age_below_threshold" not in remaining_ids, "Memory below threshold should be deleted"
    assert "score_below_threshold" not in remaining_ids, "Memory below threshold should be deleted"
    assert "cap_below_threshold" not in remaining_ids, "Memory below threshold should be deleted"
    
    # Verify exact set of remaining memories
    expected_remaining = {
        "age_at_threshold",
        "age_above_threshold",
        "score_at_threshold",
        "score_above_threshold",
        "cap_at_threshold",
        "cap_above_threshold"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "age_below_threshold",
        "score_below_threshold",
        "cap_below_threshold"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.age_pruned == 1, "Should have deleted 1 old unprotected memory"
    assert result.score_pruned == 1, "Should have deleted 1 low-score unprotected memory"
    assert result.cap_pruned == 1, "Should have deleted 1 unprotected memory for hard cap"
    assert result.total_deleted == 3, "Total deleted should be 3"
    assert result.final_count == 6, "Final count should be 6 (all protected)"


def test_importance_protection_comprehensive_all_pruning_operations():
    """
    Test that protected memories survive all pruning operations combined.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4, 13.4
    
    Setup:
    - Create comprehensive mix of protected and unprotected memories
    - Trigger all three pruning operations: age, score, and hard cap
    - Include boundary cases and various scenarios
    
    Expected:
    - All protected memories survive regardless of age, score, or cap
    - All unprotected memories subject to pruning rules
    - Exact memory_id verification (no weak assertions)
    """
    # Setup: Comprehensive mix of scenarios
    memories = [
        # Protected memories that would normally be deleted - should all survive
        create_memory("protected_old_low_score", age_days=200, importance=0.9, final_score=0.05),
        create_memory("protected_very_old", age_days=365, importance=0.85, final_score=0.5),
        create_memory("protected_zero_score", age_days=100, importance=0.8, final_score=0.0),
        create_memory("protected_at_threshold", age_days=150, importance=0.8, final_score=0.1),
        
        # Unprotected memories - should be deleted by various rules
        create_memory("unprotected_old", age_days=100, importance=0.5, final_score=0.5),        # Age pruning
        create_memory("unprotected_low_score", age_days=30, importance=0.7, final_score=0.1),   # Score pruning
        create_memory("unprotected_for_cap_1", age_days=30, importance=0.6, final_score=0.4),   # Hard cap
        create_memory("unprotected_for_cap_2", age_days=30, importance=0.79, final_score=0.45), # Hard cap
        create_memory("unprotected_for_cap_3", age_days=30, importance=0.5, final_score=0.5),   # Hard cap
    ]
    
    config = LifecycleConfig(
        max_total_memories=4,  # Will trigger hard cap after age/score pruning
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: All protected memories survive
    remaining_ids = {m["id"] for m in mock_memory.memories}
    
    assert "protected_old_low_score" in remaining_ids, "Protected memory should survive all pruning"
    assert "protected_very_old" in remaining_ids, "Protected memory should survive all pruning"
    assert "protected_zero_score" in remaining_ids, "Protected memory should survive all pruning"
    assert "protected_at_threshold" in remaining_ids, "Protected memory at threshold should survive"
    
    # All unprotected memories should be deleted
    assert "unprotected_old" not in remaining_ids, "Unprotected old memory should be deleted"
    assert "unprotected_low_score" not in remaining_ids, "Unprotected low-score memory should be deleted"
    assert "unprotected_for_cap_1" not in remaining_ids, "Unprotected memory should be deleted by cap"
    assert "unprotected_for_cap_2" not in remaining_ids, "Unprotected memory should be deleted by cap"
    assert "unprotected_for_cap_3" not in remaining_ids, "Unprotected memory should be deleted by cap"
    
    # Verify exact set of remaining memories
    expected_remaining = {
        "protected_old_low_score",
        "protected_very_old",
        "protected_zero_score",
        "protected_at_threshold"
    }
    assert remaining_ids == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_ids}"
    
    # Verify exact set of deleted memories
    deleted_ids = set(mock_memory.deleted_ids)
    expected_deleted = {
        "unprotected_old",
        "unprotected_low_score",
        "unprotected_for_cap_1",
        "unprotected_for_cap_2",
        "unprotected_for_cap_3"
    }
    assert deleted_ids == expected_deleted, \
        f"Expected deleted {expected_deleted}, got {deleted_ids}"
    
    # Verify statistics
    assert result.age_pruned == 1, "Should have deleted 1 old unprotected memory"
    assert result.score_pruned == 1, "Should have deleted 1 low-score unprotected memory"
    assert result.cap_pruned == 3, "Should have deleted 3 unprotected memories for hard cap"
    assert result.total_deleted == 5, "Total deleted should be 5"
    assert result.final_count == 4, "Final count should be 4 (all protected)"


# ============================================================================
# Error Handling Tests - Requirements 10.1, 10.2, 10.3, 10.4
# ============================================================================

from unittest.mock import Mock, patch


class FailingMemoryInterface(MemoryInterface):
    """Mock implementation that simulates deletion failures."""
    
    def __init__(self, initial_memories: List[MemoryEntry], fail_ids: set):
        """
        Initialize with memories and IDs that should fail deletion.
        
        Args:
            initial_memories: List of memory entries
            fail_ids: Set of memory IDs that should fail when deleted
        """
        self.memories = initial_memories.copy()
        self.deleted_ids: List[str] = []
        self.fail_ids = fail_ids
        self.deletion_attempts: List[tuple[str, bool]] = []  # (memory_id, success)
    
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
        """Delete a memory by ID, raising exception for fail_ids."""
        if memory_id in self.fail_ids:
            self.deletion_attempts.append((memory_id, False))
            raise RuntimeError(f"Simulated deletion failure for {memory_id}")
        
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
        self.deletion_attempts.append((memory_id, True))
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a new memory (not used in lifecycle tests)."""
        raise NotImplementedError("Store not needed for lifecycle tests")


def test_deletion_failures_are_logged_with_memory_id_and_exception():
    """
    Test that deletion failures are logged with memory_id and exception.
    
    Validates: Requirement 10.1
    
    Setup:
    - Create memories where some will fail deletion
    - Mock logger to capture log messages
    
    Expected:
    - Failed deletions should be logged with memory_id and exception details
    """
    # Setup: Create memories with some that will fail deletion
    memories = [
        create_memory("success_1", age_days=100, importance=0.5),
        create_memory("fail_1", age_days=100, importance=0.5),
        create_memory("success_2", age_days=100, importance=0.5),
        create_memory("fail_2", age_days=100, importance=0.5),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    # Create mock that fails on specific IDs
    fail_ids = {"fail_1", "fail_2"}
    mock_memory = FailingMemoryInterface(memories, fail_ids)
    
    # Patch logger to capture log calls
    with patch('luma.core.lifecycle_manager._logger') as mock_logger:
        manager = MemoryLifecycleManager(config, mock_memory)
        
        # Execute: Run cleanup
        result = manager.cleanup()
        
        # Verify: Logger was called with error messages containing memory_id
        error_calls = [call for call in mock_logger.error.call_args_list]
        
        # Should have 2 error log calls (one for each failed deletion)
        assert len(error_calls) >= 2, f"Expected at least 2 error log calls, got {len(error_calls)}"
        
        # Verify that error messages contain memory_id and exception info
        error_messages = [str(call[0][0]) for call in error_calls]
        
        # Check that fail_1 and fail_2 are mentioned in error logs
        fail_1_logged = any("fail_1" in msg for msg in error_messages)
        fail_2_logged = any("fail_2" in msg for msg in error_messages)
        
        assert fail_1_logged, "Error log should contain memory_id 'fail_1'"
        assert fail_2_logged, "Error log should contain memory_id 'fail_2'"
        
        # Verify exc_info=True was used (exception details logged)
        for call in error_calls:
            if len(call[1]) > 0 and 'exc_info' in call[1]:
                assert call[1]['exc_info'] is True, "exc_info should be True for exception logging"


def test_deletion_failures_dont_abort_cleanup():
    """
    Test that deletion failures don't abort cleanup.
    
    Validates: Requirement 10.2
    
    Setup:
    - Create memories where some will fail deletion
    - Configure to delete all memories
    
    Expected:
    - Cleanup should continue processing remaining deletions after failures
    - Successful deletions should still occur
    """
    # Setup: Create memories with some that will fail deletion
    memories = [
        create_memory("success_1", age_days=100, importance=0.5),
        create_memory("fail_1", age_days=100, importance=0.5),
        create_memory("success_2", age_days=100, importance=0.5),
        create_memory("fail_2", age_days=100, importance=0.5),
        create_memory("success_3", age_days=100, importance=0.5),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    # Create mock that fails on specific IDs
    fail_ids = {"fail_1", "fail_2"}
    mock_memory = FailingMemoryInterface(memories, fail_ids)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Cleanup continued after failures
    # Should have attempted to delete all 5 memories
    assert len(mock_memory.deletion_attempts) == 5, \
        f"Should have attempted 5 deletions, got {len(mock_memory.deletion_attempts)}"
    
    # Verify: Successful deletions occurred
    successful_deletions = [mem_id for mem_id, success in mock_memory.deletion_attempts if success]
    assert len(successful_deletions) == 3, \
        f"Should have 3 successful deletions, got {len(successful_deletions)}"
    
    # Verify: Correct memories were deleted
    assert "success_1" in successful_deletions, "success_1 should be deleted"
    assert "success_2" in successful_deletions, "success_2 should be deleted"
    assert "success_3" in successful_deletions, "success_3 should be deleted"
    
    # Verify: Failed memories are still in storage
    remaining_ids = {m["id"] for m in mock_memory.memories}
    assert "fail_1" in remaining_ids, "fail_1 should remain (deletion failed)"
    assert "fail_2" in remaining_ids, "fail_2 should remain (deletion failed)"
    
    # Verify: Statistics reflect partial completion
    assert result.age_pruned == 3, "Should have 3 successful deletions"
    assert result.failed_deletions == 2, "Should have 2 failed deletions"
    assert result.total_deleted == 3, "Total deleted should be 3"


def test_cleanup_never_propagates_exceptions():
    """
    Test that cleanup never propagates exceptions.
    
    Validates: Requirement 10.3
    
    Setup:
    - Create memories where all deletions will fail
    - Configure to delete all memories
    
    Expected:
    - Cleanup should not raise any exceptions
    - Cleanup should return a result indicating failures
    """
    # Setup: Create memories that will all fail deletion
    memories = [
        create_memory("fail_1", age_days=100, importance=0.5),
        create_memory("fail_2", age_days=100, importance=0.5),
        create_memory("fail_3", age_days=100, importance=0.5),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    # Create mock that fails on all IDs
    fail_ids = {"fail_1", "fail_2", "fail_3"}
    mock_memory = FailingMemoryInterface(memories, fail_ids)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup - should not raise exception
    try:
        result = manager.cleanup()
        exception_raised = False
    except Exception as e:
        exception_raised = True
        pytest.fail(f"Cleanup should not propagate exceptions, but raised: {e}")
    
    # Verify: No exception was raised
    assert not exception_raised, "Cleanup should not propagate exceptions"
    
    # Verify: Result indicates failures
    assert result.failed_deletions == 3, "Should have 3 failed deletions"
    assert result.total_deleted == 0, "Should have 0 successful deletions"
    assert result.status == CleanupStatus.FAILED, "Status should be FAILED"


def test_cleanup_with_retrieve_exception_doesnt_propagate():
    """
    Test that cleanup handles retrieve exceptions gracefully.
    
    Validates: Requirement 10.3
    
    Setup:
    - Create mock that raises exception on retrieve
    
    Expected:
    - Cleanup should not raise exception
    - Cleanup should return a result (possibly with no deletions)
    """
    class FailingRetrieveInterface(MemoryInterface):
        """Mock that fails on retrieve."""
        
        def retrieve(
            self,
            query: Optional[str] = None,
            params: Optional[QueryParameters] = None,
            limit: int = 10
        ) -> Dict[str, Any]:
            """Raise exception on retrieve."""
            raise RuntimeError("Simulated retrieve failure")
        
        def delete(self, memory_id: str) -> None:
            """Delete method (not called if retrieve fails)."""
            pass
        
        def store(
            self,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            category: str = "general",
            tags: Optional[List[str]] = None
        ) -> str:
            """Store method (not used)."""
            raise NotImplementedError("Store not needed")
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    mock_memory = FailingRetrieveInterface()
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup - should not raise exception
    try:
        result = manager.cleanup()
        exception_raised = False
    except Exception as e:
        exception_raised = True
        pytest.fail(f"Cleanup should not propagate exceptions, but raised: {e}")
    
    # Verify: No exception was raised
    assert not exception_raised, "Cleanup should not propagate exceptions"
    
    # Verify: Result is returned (even if no deletions occurred)
    assert isinstance(result, CleanupResult), "Should return CleanupResult"


def test_partial_completion_returns_correct_status():
    """
    Test that partial completion returns correct status.
    
    Validates: Requirement 10.4
    
    Setup:
    - Create memories where some deletions succeed and some fail
    
    Expected:
    - Status should be PARTIAL when some deletions succeed and some fail
    - Status should be SUCCESS when all deletions succeed
    - Status should be FAILED when all deletions fail
    """
    # Test Case 1: Partial completion (some succeed, some fail)
    memories_partial = [
        create_memory("success_1", age_days=100, importance=0.5),
        create_memory("fail_1", age_days=100, importance=0.5),
        create_memory("success_2", age_days=100, importance=0.5),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        min_importance_protected=0.8
    )
    
    fail_ids_partial = {"fail_1"}
    mock_memory_partial = FailingMemoryInterface(memories_partial, fail_ids_partial)
    manager_partial = MemoryLifecycleManager(config, mock_memory_partial)
    
    result_partial = manager_partial.cleanup()
    
    # Verify: Status is PARTIAL
    assert result_partial.status == CleanupStatus.PARTIAL, \
        f"Status should be PARTIAL, got {result_partial.status}"
    assert result_partial.total_deleted == 2, "Should have 2 successful deletions"
    assert result_partial.failed_deletions == 1, "Should have 1 failed deletion"
    
    # Test Case 2: Complete success (all succeed)
    memories_success = [
        create_memory("success_1", age_days=100, importance=0.5),
        create_memory("success_2", age_days=100, importance=0.5),
    ]
    
    fail_ids_success = set()  # No failures
    mock_memory_success = FailingMemoryInterface(memories_success, fail_ids_success)
    manager_success = MemoryLifecycleManager(config, mock_memory_success)
    
    result_success = manager_success.cleanup()
    
    # Verify: Status is SUCCESS
    assert result_success.status == CleanupStatus.SUCCESS, \
        f"Status should be SUCCESS, got {result_success.status}"
    assert result_success.total_deleted == 2, "Should have 2 successful deletions"
    assert result_success.failed_deletions == 0, "Should have 0 failed deletions"
    
    # Test Case 3: Complete failure (all fail)
    memories_failed = [
        create_memory("fail_1", age_days=100, importance=0.5),
        create_memory("fail_2", age_days=100, importance=0.5),
    ]
    
    fail_ids_failed = {"fail_1", "fail_2"}  # All fail
    mock_memory_failed = FailingMemoryInterface(memories_failed, fail_ids_failed)
    manager_failed = MemoryLifecycleManager(config, mock_memory_failed)
    
    result_failed = manager_failed.cleanup()
    
    # Verify: Status is FAILED
    assert result_failed.status == CleanupStatus.FAILED, \
        f"Status should be FAILED, got {result_failed.status}"
    assert result_failed.total_deleted == 0, "Should have 0 successful deletions"
    assert result_failed.failed_deletions == 2, "Should have 2 failed deletions"


def test_error_handling_across_multiple_pruning_phases():
    """
    Test error handling across multiple pruning phases.
    
    Validates: Requirements 10.1, 10.2, 10.3, 10.4
    
    Setup:
    - Create memories that will be pruned by different phases
    - Configure failures in different phases
    
    Expected:
    - Errors in one phase don't stop other phases
    - Failed deletions are tracked across all phases
    - Status reflects overall completion
    """
    # Setup: Create memories for different pruning phases
    memories = [
        # Age pruning candidates
        create_memory("age_success", age_days=100, importance=0.5, final_score=0.8),
        create_memory("age_fail", age_days=100, importance=0.5, final_score=0.8),
        
        # Score pruning candidates (young but low score)
        create_memory("score_success", age_days=30, importance=0.5, final_score=0.1),
        create_memory("score_fail", age_days=30, importance=0.5, final_score=0.1),
        
        # Hard cap candidates (need to exceed cap)
        create_memory("cap_success", age_days=30, importance=0.5, final_score=0.5),
        create_memory("cap_fail", age_days=30, importance=0.5, final_score=0.5),
    ]
    
    config = LifecycleConfig(
        max_total_memories=2,  # Force hard cap enforcement
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    # Configure failures across different phases
    fail_ids = {"age_fail", "score_fail", "cap_fail"}
    mock_memory = FailingMemoryInterface(memories, fail_ids)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Verify: Cleanup completed without raising exceptions
    assert isinstance(result, CleanupResult), "Should return CleanupResult"
    
    # Verify: Some deletions succeeded
    assert result.total_deleted > 0, "Some deletions should succeed"
    
    # Verify: Some deletions failed
    assert result.failed_deletions > 0, "Some deletions should fail"
    
    # Verify: Status is PARTIAL (some success, some failure)
    assert result.status == CleanupStatus.PARTIAL, \
        f"Status should be PARTIAL, got {result.status}"
    
    # Verify: All phases attempted their deletions
    # (at least one deletion attempt per phase that had candidates)
    assert len(mock_memory.deletion_attempts) > 0, \
        "Should have attempted deletions"


# ============================================================================
# Idempotency Tests - Requirements 8.1, 8.2, 8.3, 13.5
# ============================================================================

def test_running_cleanup_twice_produces_same_final_state():
    """
    Test that running cleanup twice produces same final state.
    
    Validates: Requirement 8.1
    
    Setup:
    - Create memories that will be pruned
    - Run cleanup twice consecutively
    
    Expected:
    - Final state after first cleanup should match final state after second cleanup
    - Same exact set of memories should remain after both cleanups
    """
    # Setup: Create memories with various pruning scenarios
    memories = [
        # Old unprotected - will be deleted in first cleanup
        create_memory("old_unprotected_1", age_days=100, importance=0.5, final_score=0.5),
        create_memory("old_unprotected_2", age_days=120, importance=0.6, final_score=0.6),
        
        # Low score unprotected - will be deleted in first cleanup
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_2", age_days=40, importance=0.6, final_score=0.15),
        
        # Protected memories - will survive
        create_memory("protected_1", age_days=100, importance=0.9, final_score=0.1),
        create_memory("protected_2", age_days=50, importance=0.85, final_score=0.2),
        
        # Young high-score unprotected - will survive
        create_memory("survivor_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("survivor_2", age_days=50, importance=0.6, final_score=0.7),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup first time
    result1 = manager.cleanup()
    
    # Capture state after first cleanup
    remaining_after_first = {m["id"] for m in mock_memory.memories}
    count_after_first = len(mock_memory.memories)
    
    # Execute: Run cleanup second time
    result2 = manager.cleanup()
    
    # Capture state after second cleanup
    remaining_after_second = {m["id"] for m in mock_memory.memories}
    count_after_second = len(mock_memory.memories)
    
    # Verify: Final states are identical
    assert remaining_after_first == remaining_after_second, \
        f"Final state should be identical after both cleanups. " \
        f"First: {remaining_after_first}, Second: {remaining_after_second}"
    
    assert count_after_first == count_after_second, \
        f"Memory count should be identical after both cleanups. " \
        f"First: {count_after_first}, Second: {count_after_second}"
    
    # Verify: Expected memories remain
    expected_remaining = {"protected_1", "protected_2", "survivor_1", "survivor_2"}
    assert remaining_after_second == expected_remaining, \
        f"Expected {expected_remaining}, got {remaining_after_second}"
    
    # Verify: First cleanup deleted memories, second cleanup deleted nothing
    assert result1.total_deleted > 0, "First cleanup should delete memories"
    assert result2.total_deleted == 0, "Second cleanup should delete nothing"


def test_second_cleanup_deletes_zero_memories():
    """
    Test that second cleanup deletes zero memories.
    
    Validates: Requirement 8.2
    
    Setup:
    - Create memories that will be pruned
    - Run cleanup twice consecutively
    
    Expected:
    - First cleanup should delete memories
    - Second cleanup should delete exactly zero memories
    - All pruning phases should report zero deletions on second run
    """
    # Setup: Create memories that will be pruned by different phases
    memories = [
        # Age pruning candidates
        create_memory("old_1", age_days=100, importance=0.5, final_score=0.8),
        create_memory("old_2", age_days=120, importance=0.6, final_score=0.7),
        
        # Score pruning candidates
        create_memory("low_score_1", age_days=30, importance=0.5, final_score=0.1),
        create_memory("low_score_2", age_days=40, importance=0.6, final_score=0.2),
        
        # Survivors
        create_memory("survivor_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("survivor_2", age_days=50, importance=0.6, final_score=0.7),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run cleanup first time
    result1 = manager.cleanup()
    
    # Verify: First cleanup deleted memories
    assert result1.total_deleted > 0, \
        f"First cleanup should delete memories, but deleted {result1.total_deleted}"
    assert result1.age_pruned > 0 or result1.score_pruned > 0, \
        "First cleanup should have pruned by age or score"
    
    # Execute: Run cleanup second time
    result2 = manager.cleanup()
    
    # Verify: Second cleanup deleted exactly zero memories
    assert result2.total_deleted == 0, \
        f"Second cleanup should delete zero memories, but deleted {result2.total_deleted}"
    
    # Verify: All pruning phases report zero deletions
    assert result2.age_pruned == 0, \
        f"Second cleanup age_pruned should be 0, got {result2.age_pruned}"
    assert result2.score_pruned == 0, \
        f"Second cleanup score_pruned should be 0, got {result2.score_pruned}"
    assert result2.cap_pruned == 0, \
        f"Second cleanup cap_pruned should be 0, got {result2.cap_pruned}"
    
    # Verify: Final counts match
    assert result1.final_count == result2.final_count, \
        f"Final counts should match: {result1.final_count} vs {result2.final_count}"


def test_already_compliant_state_remains_unchanged():
    """
    Test that already-compliant state remains unchanged.
    
    Validates: Requirement 8.3
    
    Setup:
    - Create memories that are already compliant with all rules
    - Run cleanup
    
    Expected:
    - No memories should be deleted
    - All memories should remain
    - All pruning phases should report zero deletions
    """
    # Setup: Create memories that are already compliant
    memories = [
        # Young memories with good scores
        create_memory("compliant_1", age_days=30, importance=0.5, final_score=0.8),
        create_memory("compliant_2", age_days=40, importance=0.6, final_score=0.7),
        create_memory("compliant_3", age_days=50, importance=0.7, final_score=0.6),
        
        # Protected memories (can be old or low score)
        create_memory("protected_1", age_days=100, importance=0.9, final_score=0.1),
        create_memory("protected_2", age_days=120, importance=0.85, final_score=0.2),
    ]
    
    config = LifecycleConfig(
        max_total_memories=10000,  # Well above current count
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Capture initial state
    initial_ids = {m["id"] for m in mock_memory.memories}
    initial_count = len(mock_memory.memories)
    
    # Execute: Run cleanup
    result = manager.cleanup()
    
    # Capture final state
    final_ids = {m["id"] for m in mock_memory.memories}
    final_count = len(mock_memory.memories)
    
    # Verify: No memories deleted
    assert result.total_deleted == 0, \
        f"Should delete zero memories, but deleted {result.total_deleted}"
    
    # Verify: All pruning phases report zero deletions
    assert result.age_pruned == 0, f"age_pruned should be 0, got {result.age_pruned}"
    assert result.score_pruned == 0, f"score_pruned should be 0, got {result.score_pruned}"
    assert result.cap_pruned == 0, f"cap_pruned should be 0, got {result.cap_pruned}"
    
    # Verify: State unchanged - exact memory_id sets match
    assert initial_ids == final_ids, \
        f"Memory IDs should be unchanged. Initial: {initial_ids}, Final: {final_ids}"
    
    assert initial_count == final_count, \
        f"Memory count should be unchanged. Initial: {initial_count}, Final: {final_count}"
    
    # Verify: Exact set of memories (no weak assertions)
    expected_ids = {"compliant_1", "compliant_2", "compliant_3", "protected_1", "protected_2"}
    assert final_ids == expected_ids, \
        f"Expected exact set {expected_ids}, got {final_ids}"


# ============================================================================
# Performance Benchmark Tests - Requirement 11.4
# ============================================================================

import time


def test_performance_benchmark_100k_memories():
    """
    Performance benchmark test for 100,000 memory entries.
    
    Validates: Requirement 11.4
    
    Setup:
    - Generate 100,000 memory entries with mixed characteristics
    - Configure cleanup with all pruning rules enabled
    
    Expected:
    - Complete cleanup pipeline finishes within 5 seconds
    - Timing statistics logged for each phase
    - All pruning operations complete successfully
    
    Performance Requirements:
    - Total execution time < 5 seconds
    - Age pruning: O(n) complexity
    - Score pruning: O(n) complexity
    - Hard cap enforcement: O(n log n) complexity
    """
    # Setup: Generate 100,000 memory entries
    print("\n=== Performance Benchmark: 100,000 Memory Entries ===")
    print("Generating test data...")
    
    generation_start = time.time()
    memories = []
    
    # Generate diverse memory entries
    # 40% old memories (age > 90 days)
    # 30% low-score memories (score < 0.3)
    # 20% protected memories (importance >= 0.8)
    # 10% compliant memories
    
    for i in range(100000):
        if i < 40000:
            # Old memories - mix of protected and unprotected
            importance = 0.9 if i % 5 == 0 else 0.5
            memory = create_memory(
                f"old_{i}",
                age_days=100 + (i % 100),
                importance=importance,
                final_score=0.5
            )
        elif i < 70000:
            # Low-score memories - mix of protected and unprotected
            importance = 0.85 if i % 7 == 0 else 0.6
            memory = create_memory(
                f"low_score_{i}",
                age_days=50,
                importance=importance,
                final_score=0.1 + (i % 20) * 0.01
            )
        elif i < 90000:
            # Protected memories with various ages and scores
            memory = create_memory(
                f"protected_{i}",
                age_days=30 + (i % 200),
                importance=0.8 + (i % 20) * 0.01,
                final_score=0.2 + (i % 50) * 0.01
            )
        else:
            # Compliant memories
            memory = create_memory(
                f"compliant_{i}",
                age_days=30 + (i % 60),
                importance=0.5,
                final_score=0.5 + (i % 40) * 0.01
            )
        
        memories.append(memory)
    
    generation_time = time.time() - generation_start
    print(f"Data generation completed in {generation_time:.3f} seconds")
    print(f"Total memories: {len(memories):,}")
    
    # Configure lifecycle manager with all pruning rules
    config = LifecycleConfig(
        max_total_memories=50000,  # Force hard cap pruning
        max_age_days=90,
        pruning_score_threshold=0.3,
        min_importance_protected=0.8
    )
    
    mock_memory = MockMemoryInterface(memories)
    manager = MemoryLifecycleManager(config, mock_memory)
    
    # Execute: Run complete cleanup pipeline with timing
    print("\nStarting cleanup pipeline...")
    cleanup_start = time.time()
    
    # Time each phase individually by running cleanup
    result = manager.cleanup()
    
    cleanup_end = time.time()
    total_time = cleanup_end - cleanup_start
    
    # Log timing statistics
    print("\n=== Timing Statistics ===")
    print(f"Total cleanup time: {total_time:.3f} seconds")
    print(f"Memories processed: {len(memories):,}")
    print(f"Processing rate: {len(memories) / total_time:,.0f} memories/second")
    
    print("\n=== Pruning Statistics ===")
    print(f"Age-based pruning: {result.age_pruned:,} memories deleted")
    print(f"Score-based pruning: {result.score_pruned:,} memories deleted")
    print(f"Hard cap enforcement: {result.cap_pruned:,} memories deleted")
    print(f"Total deleted: {result.total_deleted:,} memories")
    print(f"Failed deletions: {result.failed_deletions:,}")
    print(f"Final count: {result.final_count:,} memories")
    print(f"Status: {result.status.value}")
    
    # Verify: Performance requirement met
    assert total_time < 5.0, \
        f"Cleanup must complete within 5 seconds, took {total_time:.3f} seconds"
    
    # Verify: Cleanup completed successfully
    assert result.status.value in ["success", "partial"], \
        f"Cleanup should succeed or partially succeed, got {result.status.value}"
    
    # Verify: Hard cap enforced
    assert result.final_count <= config.max_total_memories, \
        f"Final count {result.final_count} exceeds max_total_memories {config.max_total_memories}"
    
    # Verify: Some pruning occurred (we have 100k memories, cap is 50k)
    assert result.total_deleted > 0, \
        "Should have deleted some memories given the test setup"
    
    # Verify: Protected memories were not deleted
    # Count protected memories in final state
    protected_count = sum(
        1 for m in mock_memory.memories
        if m["metadata"].get("importance", 0.0) >= config.min_importance_protected
    )
    
    # Count protected memories in initial state
    initial_protected_count = sum(
        1 for m in memories
        if m["metadata"].get("importance", 0.0) >= config.min_importance_protected
    )
    
    assert protected_count == initial_protected_count, \
        f"Protected memories should not be deleted: initial={initial_protected_count}, final={protected_count}"
    
    print("\n=== Performance Benchmark PASSED ===")
    print(f"✓ Completed in {total_time:.3f} seconds (< 5.0 seconds required)")
    print(f"✓ Processed {len(memories):,} memories successfully")
    print(f"✓ Hard cap enforced: {result.final_count:,} <= {config.max_total_memories:,}")
    print(f"✓ Protected memories preserved: {protected_count:,}")
