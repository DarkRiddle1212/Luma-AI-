"""Property-based tests for tie-breaking correctness.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5**
"""

import pytest
from hypothesis import given, strategies as st, assume
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
    final_score1=st.floats(min_value=0.0, max_value=1.0),
    final_score2=st.floats(min_value=0.0, max_value=1.0),
    similarity_score1=st.floats(min_value=0.0, max_value=1.0),
    similarity_score2=st.floats(min_value=0.0, max_value=1.0),
)
def test_primary_sort_by_final_score(
    final_score1: float,
    final_score2: float,
    similarity_score1: float,
    similarity_score2: float
):
    """
    Property 5: Primary sort by final_score
    
    **Validates: Requirements 3.2**
    
    Test that memories with different final scores are ordered by final_score (descending).
    """
    # Only test when final scores are different
    assume(abs(final_score1 - final_score2) > 0.001)
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("mem1", similarity_score1, final_score1, base_time),
        create_test_memory("mem2", similarity_score2, final_score2, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    # Property: Higher final_score comes first
    if final_score1 > final_score2:
        assert result[0].memory_id == "mem1"
        assert result[1].memory_id == "mem2"
    else:
        assert result[0].memory_id == "mem2"
        assert result[1].memory_id == "mem1"


@given(
    final_score=st.floats(min_value=0.0, max_value=1.0),
    similarity_score1=st.floats(min_value=0.0, max_value=1.0),
    similarity_score2=st.floats(min_value=0.0, max_value=1.0),
)
def test_secondary_sort_by_similarity_score(
    final_score: float,
    similarity_score1: float,
    similarity_score2: float
):
    """
    Property 5: Secondary sort by similarity_score
    
    **Validates: Requirements 3.3**
    
    Test that when final scores are equal, memories are ordered by similarity_score (descending).
    """
    # Only test when similarity scores are different
    assume(abs(similarity_score1 - similarity_score2) > 0.001)
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("mem1", similarity_score1, final_score, base_time),
        create_test_memory("mem2", similarity_score2, final_score, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    # Property: Higher similarity_score comes first when final_score is equal
    if similarity_score1 > similarity_score2:
        assert result[0].memory_id == "mem1"
        assert result[1].memory_id == "mem2"
    else:
        assert result[0].memory_id == "mem2"
        assert result[1].memory_id == "mem1"


@given(
    final_score=st.floats(min_value=0.0, max_value=1.0),
    similarity_score=st.floats(min_value=0.0, max_value=1.0),
    timestamp_offset1=st.integers(min_value=0, max_value=1000000),
    timestamp_offset2=st.integers(min_value=0, max_value=1000000),
)
def test_tertiary_sort_by_timestamp(
    final_score: float,
    similarity_score: float,
    timestamp_offset1: int,
    timestamp_offset2: int
):
    """
    Property 5: Tertiary sort by timestamp
    
    **Validates: Requirements 3.4**
    
    Test that when final and similarity scores are equal, memories are ordered by timestamp (newer first).
    """
    # Only test when timestamps are different
    assume(abs(timestamp_offset1 - timestamp_offset2) > 1)
    
    base_time = datetime.now(timezone.utc)
    time1 = base_time - timedelta(seconds=timestamp_offset1)
    time2 = base_time - timedelta(seconds=timestamp_offset2)
    
    memories = [
        create_test_memory("mem1", similarity_score, final_score, time1),
        create_test_memory("mem2", similarity_score, final_score, time2),
    ]
    
    result = StableSorter.sort(memories)
    
    # Property: Newer timestamp comes first when scores are equal
    if time1 > time2:
        assert result[0].memory_id == "mem1"
        assert result[1].memory_id == "mem2"
    else:
        assert result[0].memory_id == "mem2"
        assert result[1].memory_id == "mem1"


def test_quaternary_sort_by_memory_id():
    """
    Property 5: Quaternary sort by memory_id
    
    **Validates: Requirements 3.5**
    
    Test that when all scores and timestamp are equal, memories are ordered by memory_id (lexicographical).
    """
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("mem_c", 0.5, 0.5, base_time),
        create_test_memory("mem_a", 0.5, 0.5, base_time),
        create_test_memory("mem_b", 0.5, 0.5, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    # Property: Lexicographical order by memory_id when everything else is equal
    assert result[0].memory_id == "mem_a"
    assert result[1].memory_id == "mem_b"
    assert result[2].memory_id == "mem_c"


@given(
    memories=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),  # memory_id (ASCII only)
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),  # final_score
            st.integers(min_value=0, max_value=1000000)  # timestamp offset
        ),
        min_size=2,
        max_size=50,
        unique_by=lambda x: x[0]  # Unique memory_ids
    )
)
def test_complete_tie_breaking_hierarchy(memories: list):
    """
    Property 5: Complete tie-breaking hierarchy
    
    **Validates: Requirements 3.2, 3.3, 3.4, 3.5**
    
    Test that the complete 4-level tie-breaking works correctly.
    Order: final_score (desc) > similarity_score (desc) > timestamp (desc) > memory_id (asc)
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
    
    # Property: Result is ordered according to the 4-level hierarchy
    # Order: final_score (primary) > similarity_score (secondary) > timestamp (tertiary) > memory_id (quaternary)
    for i in range(len(result) - 1):
        curr = result[i]
        next_mem = result[i + 1]
        
        # Check ordering is correct according to the AUTHORITATIVE order
        # We need to check the actual comparison key, not individual fields with tolerance
        curr_key = (-curr.final_score, -curr.similarity_score, -curr.timestamp.timestamp(), curr.memory_id)
        next_key = (-next_mem.final_score, -next_mem.similarity_score, -next_mem.timestamp.timestamp(), next_mem.memory_id)
        
        # The sorting key should be in ascending order (because we use tuple comparison)
        assert curr_key <= next_key, \
            f"Sorting order violated:\n  curr: {curr_key}\n  next: {next_key}\n  curr memory: {curr.memory_id}\n  next memory: {next_mem.memory_id}"
