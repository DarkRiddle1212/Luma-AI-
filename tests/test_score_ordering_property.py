"""Property-based tests for score ordering in results.

**Validates: Requirements 9.3, 9.4**
"""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import StableSorter, RankedMemory


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    final_score: float,
    timestamp: datetime
) -> RankedMemory:
    """Helper to create test memory."""
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


@given(
    memories=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),  # memory_id
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # final_score
            st.integers(min_value=0, max_value=1000000)  # timestamp offset
        ),
        min_size=1,
        max_size=50,
        unique_by=lambda x: x[0]  # Unique memory_ids
    )
)
def test_score_monotonicity_in_results(memories: list):
    """
    Property 6: Score monotonicity in results
    
    **Validates: Requirements 9.4**
    
    Test that each memory has final_score >= next memory's final_score.
    """
    base_time = datetime.now(timezone.utc)
    test_memories = [
        create_test_memory(
            mem_id,
            sim,
            score,
            base_time - timedelta(seconds=offset)
        )
        for mem_id, sim, score, offset in memories
    ]
    
    result = StableSorter.sort(test_memories)
    
    # Property: final_score is monotonically decreasing (or equal)
    for i in range(len(result) - 1):
        curr = result[i]
        next_mem = result[i + 1]
        
        assert curr.final_score >= next_mem.final_score, \
            f"Score monotonicity violated at position {i}: {curr.final_score} < {next_mem.final_score}"


@given(
    score1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    score2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    similarity1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    similarity2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_higher_scored_memories_rank_higher(
    score1: float,
    score2: float,
    similarity1: float,
    similarity2: float
):
    """
    Property 6: Higher scored memories always rank higher
    
    **Validates: Requirements 9.3**
    
    Test that memories with higher final_score always rank higher regardless of input order.
    """
    base_time = datetime.now(timezone.utc)
    
    # Create two memories
    mem1 = create_test_memory("mem1", similarity1, score1, base_time)
    mem2 = create_test_memory("mem2", similarity2, score2, base_time)
    
    # Sort in both orders
    result1 = StableSorter.sort([mem1, mem2])
    result2 = StableSorter.sort([mem2, mem1])
    
    # Property: Same ordering regardless of input order
    assert result1[0].memory_id == result2[0].memory_id
    assert result1[1].memory_id == result2[1].memory_id
    
    # Property: Higher final_score comes first
    if score1 > score2:
        assert result1[0].memory_id == "mem1"
        assert result1[1].memory_id == "mem2"
    elif score2 > score1:
        assert result1[0].memory_id == "mem2"
        assert result1[1].memory_id == "mem1"
    # If equal scores, order determined by secondary keys (similarity, timestamp, id)


def test_empty_list_handling():
    """Test that empty list returns empty result."""
    result = StableSorter.sort([])
    assert len(result) == 0


def test_single_memory_handling():
    """Test that single memory returns single result."""
    base_time = datetime.now(timezone.utc)
    memory = create_test_memory("mem1", 0.5, 0.5, base_time)
    
    result = StableSorter.sort([memory])
    
    assert len(result) == 1
    assert result[0].memory_id == "mem1"
