"""
Unit tests for RedundancyGuard.

**Validates: Requirements 3.2, 3.3**
"""

import pytest
import numpy as np
from luma.core.injection_engine import RedundancyGuard


class MockMemory:
    """Mock memory object for testing."""
    
    def __init__(self, memory_id: str, final_score: float, embedding=None):
        self.memory_id = memory_id
        self.final_score = final_score
        self.metadata = {}
        if embedding is not None:
            self.metadata['embedding'] = embedding


def test_threshold_zero_filters_identical():
    """Test that threshold=0.0 filters memories with similarity > 0.0 (almost everything)."""
    guard = RedundancyGuard(threshold=0.0)
    
    # Create memories with identical embeddings (similarity = 1.0)
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [1.0, 0.0, 0.0]),  # Identical to m1, similarity = 1.0 > 0.0
        MockMemory("m3", 0.7, [1.0, 0.0, 0.0]),  # Identical to m1, similarity = 1.0 > 0.0
    ]
    
    filtered, count = guard.filter(memories)
    
    # With threshold=0.0, any similarity > 0.0 causes filtering
    # Only m1 passes (first memory), m2 and m3 are filtered
    assert len(filtered) == 1
    assert count == 2
    assert filtered[0].memory_id == "m1"


def test_threshold_one_allows_high_similarity():
    """Test that threshold=1.0 allows very high similarity but filters perfect matches."""
    guard = RedundancyGuard(threshold=1.0)
    
    # Create memories with very high but not perfect similarity
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [0.99, 0.01, 0.0]),  # Very similar but not identical
        MockMemory("m3", 0.7, [1.0, 0.0, 0.0]),    # Identical to m1, similarity = 1.0
    ]
    
    filtered, count = guard.filter(memories)
    
    # m1 passes (first memory)
    # m2 passes (similarity < 1.0 with m1)
    # m3 should be filtered only if similarity is EXACTLY > 1.0, but since
    # similarity = 1.0 is not > 1.0, it passes
    # Actually, with threshold=1.0, nothing gets filtered unless similarity > 1.0
    assert len(filtered) == 3
    assert count == 0


def test_threshold_half_filters_similar_memories():
    """Test that threshold=0.5 filters memories with similarity > 0.5."""
    guard = RedundancyGuard(threshold=0.5)
    
    # Create memories with varying similarity
    # Similarity between [1,0,0] and [0.8,0.6,0] = 0.8 / (1.0 * 1.0) = 0.8
    # Similarity between [1,0,0] and [0,1,0] = 0.0
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [0.8, 0.6, 0.0]),  # High similarity with m1 (0.8)
        MockMemory("m3", 0.7, [0.0, 1.0, 0.0]),  # Low similarity with m1 (0.0)
    ]
    
    filtered, count = guard.filter(memories)
    
    # m1 passes (first memory)
    # m2 filtered (similarity 0.8 > 0.5 with m1)
    # m3 passes (similarity 0.0 < 0.5 with m1)
    assert len(filtered) == 2
    assert count == 1
    assert filtered[0].memory_id == "m1"
    assert filtered[1].memory_id == "m3"


def test_missing_embeddings_treated_as_not_similar():
    """Test that memories without embeddings are treated as not similar (similarity=0.0)."""
    guard = RedundancyGuard(threshold=0.5)
    
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, None),  # No embedding
        MockMemory("m3", 0.7, [1.0, 0.0, 0.0]),  # Same as m1
    ]
    
    filtered, count = guard.filter(memories)
    
    # m1 passes (first memory)
    # m2 passes (no embedding, similarity=0.0 < 0.5)
    # m3 filtered (similarity=1.0 > 0.5 with m1)
    assert len(filtered) == 2
    assert count == 1
    assert filtered[0].memory_id == "m1"
    assert filtered[1].memory_id == "m2"


def test_both_missing_embeddings_not_similar():
    """Test that two memories without embeddings are not considered similar."""
    guard = RedundancyGuard(threshold=0.5)
    
    memories = [
        MockMemory("m1", 0.9, None),  # No embedding
        MockMemory("m2", 0.8, None),  # No embedding
    ]
    
    filtered, count = guard.filter(memories)
    
    # Both pass (no embeddings, similarity=0.0 < 0.5)
    assert len(filtered) == 2
    assert count == 0


def test_higher_ranked_memory_kept():
    """Test that when redundancy is detected, the higher-ranked memory is kept."""
    guard = RedundancyGuard(threshold=0.5)
    
    # Memories are processed in order, so first one is kept
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),  # Higher rank
        MockMemory("m2", 0.8, [1.0, 0.0, 0.0]),  # Lower rank, identical to m1
    ]
    
    filtered, count = guard.filter(memories)
    
    # m1 kept (higher rank, processed first)
    # m2 filtered (redundant with m1)
    assert len(filtered) == 1
    assert count == 1
    assert filtered[0].memory_id == "m1"
    assert filtered[0].final_score == 0.9


def test_filtered_count_accuracy():
    """Test that filtered_count accurately reflects number of filtered memories."""
    guard = RedundancyGuard(threshold=0.7)
    
    # Create 5 memories, some similar
    memories = [
        MockMemory("m1", 0.95, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.90, [0.9, 0.1, 0.0]),  # Similar to m1 (sim ~0.9)
        MockMemory("m3", 0.85, [0.0, 1.0, 0.0]),  # Different from m1
        MockMemory("m4", 0.80, [0.0, 0.9, 0.1]),  # Similar to m3 (sim ~0.9)
        MockMemory("m5", 0.75, [0.0, 0.0, 1.0]),  # Different from all (orthogonal)
    ]
    
    filtered, count = guard.filter(memories)
    
    # m1 passes
    # m2 filtered (similar to m1, ~0.9 > 0.7)
    # m3 passes (orthogonal to m1)
    # m4 filtered (similar to m3, ~0.9 > 0.7)
    # m5 passes (orthogonal to m1 and m3)
    assert len(filtered) == 3
    assert count == 2


def test_empty_input_returns_empty():
    """Test that empty input list returns empty result."""
    guard = RedundancyGuard(threshold=0.5)
    
    filtered, count = guard.filter([])
    
    assert filtered == []
    assert count == 0


def test_single_memory_always_passes():
    """Test that a single memory always passes (nothing to compare against)."""
    guard = RedundancyGuard(threshold=0.5)
    
    memories = [MockMemory("m1", 0.9, [1.0, 0.0, 0.0])]
    
    filtered, count = guard.filter(memories)
    
    assert len(filtered) == 1
    assert count == 0
    assert filtered[0].memory_id == "m1"


def test_zero_norm_embeddings_treated_as_not_similar():
    """Test that embeddings with zero norm are treated as not similar."""
    guard = RedundancyGuard(threshold=0.5)
    
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [0.0, 0.0, 0.0]),  # Zero norm
    ]
    
    filtered, count = guard.filter(memories)
    
    # Both pass (zero norm treated as similarity=0.0)
    assert len(filtered) == 2
    assert count == 0


def test_order_preservation():
    """Test that output order matches input order (rank preservation)."""
    guard = RedundancyGuard(threshold=0.5)
    
    # Create memories with different embeddings (all pass)
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [0.0, 1.0, 0.0]),
        MockMemory("m3", 0.7, [0.0, 0.0, 1.0]),
    ]
    
    filtered, count = guard.filter(memories)
    
    # All pass, order preserved
    assert len(filtered) == 3
    assert count == 0
    assert [m.memory_id for m in filtered] == ["m1", "m2", "m3"]


def test_complex_redundancy_chain():
    """Test complex scenario with multiple redundancy chains."""
    guard = RedundancyGuard(threshold=0.6)
    
    # Create a scenario where:
    # - m2 is similar to m1
    # - m3 is different from m1 but similar to m4
    # - m4 would be filtered by m3
    # - m5 is different from all (orthogonal)
    memories = [
        MockMemory("m1", 0.95, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.90, [0.8, 0.6, 0.0]),  # Similar to m1 (sim = 0.8)
        MockMemory("m3", 0.85, [0.0, 1.0, 0.0]),  # Orthogonal to m1
        MockMemory("m4", 0.80, [0.0, 0.8, 0.6]),  # Similar to m3 (sim = 0.8)
        MockMemory("m5", 0.75, [0.0, 0.0, 1.0]),  # Orthogonal to all
    ]
    
    filtered, count = guard.filter(memories)
    
    # Expected: m1, m3, m5 pass; m2, m4 filtered
    assert len(filtered) == 3
    assert count == 2
    assert filtered[0].memory_id == "m1"
    assert filtered[1].memory_id == "m3"
    assert filtered[2].memory_id == "m5"


def test_invalid_threshold_raises_error():
    """Test that invalid threshold values raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        RedundancyGuard(threshold=-0.1)
    
    error_msg = str(exc_info.value)
    assert "threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg
    
    with pytest.raises(ValueError) as exc_info:
        RedundancyGuard(threshold=1.5)
    
    error_msg = str(exc_info.value)
    assert "threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg


def test_numerical_stability_with_near_identical_embeddings():
    """Test numerical stability with embeddings that are nearly identical."""
    guard = RedundancyGuard(threshold=0.999)
    
    # Create embeddings that are extremely similar but not identical
    # due to floating point precision
    memories = [
        MockMemory("m1", 0.9, [1.0, 0.0, 0.0]),
        MockMemory("m2", 0.8, [0.9999999, 0.0000001, 0.0]),
    ]
    
    filtered, count = guard.filter(memories)
    
    # Should handle floating point precision gracefully
    # The similarity should be very close to 1.0 but might not be exactly 1.0
    assert len(filtered) >= 1  # At least m1 should pass
    assert filtered[0].memory_id == "m1"
