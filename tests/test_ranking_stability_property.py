"""Comprehensive property-based tests for ranking stability.

**Validates: Requirements 9.1, 9.2**
"""

import pytest
import random
from hypothesis import given, strategies as st
from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    importance_score: float,
    timestamp: datetime
) -> RankedMemory:
    """Helper to create test memory."""
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=0.0,
        final_score=0.0,
        memory_entry=None
    )


@given(
    memories=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),  # memory_id
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # importance_score
            st.integers(min_value=0, max_value=100000)  # timestamp offset in seconds
        ),
        min_size=0,
        max_size=30,
        unique_by=lambda x: x[0]  # Unique memory_ids
    ),
    seed=st.integers(min_value=0, max_value=1000000)
)
def test_input_order_independence(memories: list, seed: int):
    """
    Property 8: Input order independence
    
    **Validates: Requirements 9.2**
    
    Test that shuffling input produces same ranked output.
    """
    if not memories:
        return  # Skip empty case
    
    config = RankingConfig(
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    test_memories = [
        create_test_memory(
            mem_id,
            sim,
            imp,
            base_time - timedelta(seconds=offset)
        )
        for mem_id, sim, imp, offset in memories
    ]
    
    engine = RankingEngine(config)
    
    # Rank original order
    result1 = engine.rank(test_memories.copy(), current_time=base_time)
    
    # Shuffle and rank
    shuffled = test_memories.copy()
    random.Random(seed).shuffle(shuffled)
    result2 = engine.rank(shuffled, current_time=base_time)
    
    # Property: Same ordering regardless of input order
    assert len(result1) == len(result2)
    for i in range(len(result1)):
        assert result1[i].memory_id == result2[i].memory_id, \
            f"Position {i}: {result1[i].memory_id} != {result2[i].memory_id}"


@given(
    memories=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),  # memory_id
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # importance_score
            st.integers(min_value=0, max_value=100000)  # timestamp offset in seconds
        ),
        min_size=0,
        max_size=30,
        unique_by=lambda x: x[0]  # Unique memory_ids
    )
)
def test_ranking_consistency_round_trip(memories: list):
    """
    Property 9: Ranking consistency (round-trip stability)
    
    **Validates: Requirements 9.1**
    
    Test that ranking twice produces identical results.
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    test_memories = [
        create_test_memory(
            mem_id,
            sim,
            imp,
            base_time - timedelta(seconds=offset)
        )
        for mem_id, sim, imp, offset in memories
    ]
    
    engine = RankingEngine(config)
    
    # Rank twice
    result1 = engine.rank(test_memories.copy(), current_time=base_time)
    result2 = engine.rank(test_memories.copy(), current_time=base_time)
    
    # Property: Identical ordering
    assert len(result1) == len(result2)
    for i in range(len(result1)):
        assert result1[i].memory_id == result2[i].memory_id, \
            f"Position {i}: {result1[i].memory_id} != {result2[i].memory_id}"
        assert abs(result1[i].final_score - result2[i].final_score) < 1e-9, \
            f"Position {i}: final_score differs"


@given(
    alpha=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    beta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_ranking_stability_with_different_configs(alpha: float, beta: float):
    """
    Test ranking stability with different weight configurations.
    
    **Validates: Requirements 9.1**
    """
    # Normalize weights to sum to 1
    total = alpha + beta
    if total < 0.01:  # Avoid division by very small numbers
        return
    
    alpha_norm = alpha / total
    beta_norm = beta / total
    
    config = RankingConfig(
        alpha=alpha_norm,
        beta=beta_norm,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, 0.0, base_time),
        create_test_memory("2", 0.6, 0.0, base_time - timedelta(hours=1)),
        create_test_memory("3", 0.9, 0.0, base_time - timedelta(hours=2)),
    ]
    
    engine = RankingEngine(config)
    
    # Rank twice
    result1 = engine.rank(memories.copy(), current_time=base_time)
    result2 = engine.rank(memories.copy(), current_time=base_time)
    
    # Property: Identical ordering
    assert len(result1) == len(result2)
    for i in range(len(result1)):
        assert result1[i].memory_id == result2[i].memory_id


def test_ranking_stability_with_identical_inputs():
    """Test that identical inputs always produce identical outputs."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, 0.0, base_time),
        create_test_memory("2", 0.6, 0.0, base_time),
        create_test_memory("3", 0.9, 0.0, base_time),
    ]
    
    engine = RankingEngine(config)
    
    # Rank multiple times
    results = [
        engine.rank(memories.copy(), current_time=base_time)
        for _ in range(5)
    ]
    
    # All results should be identical
    for i in range(1, len(results)):
        assert len(results[0]) == len(results[i])
        for j in range(len(results[0])):
            assert results[0][j].memory_id == results[i][j].memory_id
            assert abs(results[0][j].final_score - results[i][j].final_score) < 1e-9
