"""Property-based tests for ThresholdFilter component.

**Validates: Requirements 4.3, 4.4**
"""

import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timezone
from luma.core.ranking_engine import ThresholdFilter, RankedMemory, RankingConfig


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    final_score: float
) -> RankedMemory:
    """Helper to create test memory."""
    return RankedMemory(
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc),
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=0.0,
        recency_score=0.0,
        final_score=final_score,
        memory_entry=None
    )


@given(
    similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
    score_threshold=st.floats(min_value=0.0, max_value=1.0),
    memories=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),  # similarity_score
            st.floats(min_value=0.0, max_value=1.0)   # final_score
        ),
        min_size=0,
        max_size=50
    )
)
def test_threshold_filtering_correctness(
    similarity_threshold: float,
    score_threshold: float,
    memories: list
):
    """
    Property 3: Threshold filtering correctness
    
    **Validates: Requirements 4.3, 4.4**
    
    Test that all returned memories meet both thresholds and that
    memories below thresholds are excluded.
    """
    # Create config
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold
    )
    
    # Create memories
    test_memories = [
        create_test_memory(f"mem_{i}", sim, score)
        for i, (sim, score) in enumerate(memories)
    ]
    
    # Apply filter
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(test_memories)
    
    # Property: All returned memories meet both thresholds
    for memory in result:
        assert memory.similarity_score >= similarity_threshold, \
            f"Memory {memory.memory_id} has similarity {memory.similarity_score} < threshold {similarity_threshold}"
        assert memory.final_score >= score_threshold, \
            f"Memory {memory.memory_id} has final_score {memory.final_score} < threshold {score_threshold}"
    
    # Property: Memories below thresholds are excluded
    for memory in test_memories:
        if memory.similarity_score < similarity_threshold or memory.final_score < score_threshold:
            assert memory not in result, \
                f"Memory {memory.memory_id} should be filtered out"
