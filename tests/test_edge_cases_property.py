"""Property-based tests for edge case handling.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
"""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    final_score: float = 0.0,
    timestamp: datetime = None
) -> RankedMemory:
    """Helper to create test memory."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=0.0,
        recency_score=0.0,
        final_score=final_score,
        memory_entry=None
    )


def test_empty_input_handling():
    """
    Property 7: Edge case robustness - empty input
    
    **Validates: Requirement 7.5**
    
    Test that empty input returns empty output.
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    engine = RankingEngine(config)
    result = engine.rank([])
    
    assert len(result) == 0
    assert isinstance(result, list)


@given(
    num_memories=st.integers(min_value=1, max_value=20),
    threshold=st.floats(min_value=0.9, max_value=1.0)
)
def test_all_filtered_handling(num_memories: int, threshold: float):
    """
    Property 7: Edge case robustness - all filtered
    
    **Validates: Requirement 7.6**
    
    Test that when all memories are filtered, empty result is returned.
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=threshold,
        score_threshold=threshold
    )
    
    # Create memories with scores below threshold
    memories = [
        create_test_memory(f"mem_{i}", similarity_score=threshold - 0.1)
        for i in range(num_memories)
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories)
    
    assert len(result) == 0


def test_future_timestamps_handling():
    """
    Property 7: Edge case robustness - future timestamps
    
    **Validates: Requirement 7.3**
    
    Test that future timestamps are handled correctly (age = 0, recency = 1.0).
    """
    config = RankingConfig(
        alpha=0.0,
        beta=1.0,  # Only recency
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    current_time = datetime.now(timezone.utc)
    future_time = current_time + timedelta(hours=1)
    
    memories = [
        create_test_memory("future", 0.5, timestamp=future_time),
        create_test_memory("present", 0.5, timestamp=current_time),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=current_time)
    
    # Future memory should have recency_score = 1.0
    future_mem = next(m for m in result if m.memory_id == "future")
    assert future_mem.recency_score == 1.0
    
    # Future memory should rank first (highest recency)
    assert result[0].memory_id == "future"


def test_very_old_timestamps_handling():
    """
    Property 7: Edge case robustness - very old timestamps
    
    **Validates: Requirement 7.4**
    
    Test that very old timestamps don't cause numerical overflow.
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    current_time = datetime.now(timezone.utc)
    very_old_time = current_time - timedelta(days=365 * 10)  # 10 years ago
    
    memories = [
        create_test_memory("very_old", 0.5, timestamp=very_old_time),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=current_time)
    
    # Should not crash and should return valid result
    assert len(result) == 1
    assert 0 <= result[0].recency_score <= 1
    assert 0 <= result[0].final_score <= 1
    # Very old memory should have recency_score close to 0
    assert result[0].recency_score < 0.01


@given(
    num_memories=st.integers(min_value=2, max_value=20)
)
def test_identical_scores_handling(num_memories: int):
    """
    Property 7: Edge case robustness - identical scores
    
    **Validates: Requirement 7.1**
    
    Test that memories with identical scores trigger full tie-breaking.
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    
    # Create memories with identical scores but different IDs
    memories = [
        create_test_memory(f"mem_{i:03d}", 0.5, timestamp=base_time)
        for i in range(num_memories)
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # All memories should be included
    assert len(result) == num_memories
    
    # Should be ordered by memory_id (lexicographical)
    for i in range(len(result) - 1):
        assert result[i].memory_id <= result[i + 1].memory_id


def test_importance_zero_handling():
    """
    Property 7: Edge case robustness - importance = 0
    
    **Validates: Requirement 7.2**
    
    Test that when importance is 0 for all memories, final score uses only similarity and recency.
    """
    config = RankingConfig(
        alpha=0.6,
        beta=0.4,
        gamma=0.0,  # No importance
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, timestamp=base_time),
        create_test_memory("2", 0.6, timestamp=base_time - timedelta(hours=1)),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Both should be included
    assert len(result) == 2
    
    # Final scores should be computed without importance
    for memory in result:
        # final_score = alpha * similarity + beta * recency + 0 * importance
        expected = config.alpha * memory.similarity_score + config.beta * memory.recency_score
        assert abs(memory.final_score - expected) < 1e-9


def test_single_memory_handling():
    """Test handling of single memory input."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    memories = [create_test_memory("1", 0.5)]
    
    engine = RankingEngine(config)
    result = engine.rank(memories)
    
    assert len(result) == 1
    assert result[0].memory_id == "1"
