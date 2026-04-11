"""Property-based tests for StableSorter component.

**Validates: Requirements 3.6, 9.1, 9.2**
"""

import pytest
import random
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
            st.text(min_size=1, max_size=5),  # memory_id
            st.floats(min_value=0.0, max_value=1.0),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0),  # final_score
            st.integers(min_value=0, max_value=1000000)  # timestamp offset in seconds
        ),
        min_size=0,
        max_size=50,
        unique_by=lambda x: x[0]  # Unique memory_ids
    )
)
def test_sorting_determinism_idempotence(memories: list):
    """
    Property 4: Sorting determinism (idempotence)
    
    **Validates: Requirements 3.6, 9.1**
    
    Test that sorting the same list twice produces identical ordering.
    """
    # Create memories
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
    
    # Sort twice
    result1 = StableSorter.sort(test_memories)
    result2 = StableSorter.sort(test_memories)
    
    # Property: Identical ordering
    assert len(result1) == len(result2)
    for i in range(len(result1)):
        assert result1[i].memory_id == result2[i].memory_id, \
            f"Position {i}: {result1[i].memory_id} != {result2[i].memory_id}"


@given(
    memories=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=5),  # memory_id
            st.floats(min_value=0.0, max_value=1.0),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0),  # final_score
            st.integers(min_value=0, max_value=1000000)  # timestamp offset in seconds
        ),
        min_size=0,
        max_size=50,
        unique_by=lambda x: x[0]  # Unique memory_ids
    ),
    seed=st.integers(min_value=0, max_value=1000000)
)
def test_input_order_independence(memories: list, seed: int):
    """
    Property 4: Input order independence
    
    **Validates: Requirements 9.2**
    
    Test that shuffling input then sorting produces same result.
    """
    # Create memories
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
    
    # Sort original
    result1 = StableSorter.sort(test_memories)
    
    # Shuffle and sort
    shuffled = test_memories.copy()
    random.Random(seed).shuffle(shuffled)
    result2 = StableSorter.sort(shuffled)
    
    # Property: Same ordering regardless of input order
    assert len(result1) == len(result2)
    for i in range(len(result1)):
        assert result1[i].memory_id == result2[i].memory_id, \
            f"Position {i}: {result1[i].memory_id} != {result2[i].memory_id}"


def test_sorting_preserves_input():
    """Test that sorting creates a new list and doesn't modify input."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.5, 0.5, base_time),
        create_test_memory("2", 0.8, 0.8, base_time),
    ]
    
    original_ids = [m.memory_id for m in memories]
    result = StableSorter.sort(memories)
    
    # Input unchanged
    assert [m.memory_id for m in memories] == original_ids
    # Result is different list
    assert result is not memories
