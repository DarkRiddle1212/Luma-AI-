"""Property-based tests for Importance Scorer score range validation.

This module tests Property 1: Importance Score Range Validation
For any candidate memory processed by the Importance Scorer, the assigned
importance score must be between 0.0 and 1.0 inclusive.

**Validates: Requirements 3.1**
"""

import pytest
from hypothesis import given, strategies as st, settings
from luma.core.memory_write.importance_scorer import ImportanceScorer
from luma.core.memory_write.schemas import MemoryCandidate


# Strategy for generating valid memory types
memory_type_strategy = st.sampled_from([
    "project_goal",
    "user_preference", 
    "fact",
    "statement"
])

# Strategy for generating text content
text_strategy = st.text(min_size=1, max_size=5).filter(lambda x: x.strip())

# Strategy for generating valid memory candidates
memory_candidate_strategy = st.builds(
    MemoryCandidate,
    text=text_strategy,
    type=memory_type_strategy
)

# Strategy for generating threshold values
threshold_strategy = st.floats(min_value=0.0, max_value=1.0)


class TestImportanceScoreRangeProperty:
    """Property tests for importance score range validation.
    
    Feature: memory-write-engine
    Property 1: Importance Score Range Validation
    """
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold=threshold_strategy
    )
    def test_score_always_in_valid_range(self, candidate, threshold):
        """**Validates: Requirements 3.1**
        
        Property 1: Importance Score Range Validation
        
        For any candidate memory processed by the Importance Scorer,
        the assigned importance score must be between 0.0 and 1.0 inclusive.
        """
        scorer = ImportanceScorer(threshold=threshold)
        
        result = scorer.score_memory(candidate)
        
        # If result is not None (not filtered by threshold), verify score range
        if result is not None:
            assert 0.0 <= result.importance <= 1.0, (
                f"Score {result.importance} is outside valid range [0.0, 1.0] "
                f"for candidate: {candidate.text[:50]}..."
            )
    
    @settings(max_examples=10)
    @given(
        text=text_strategy,
        memory_type=memory_type_strategy
    )
    def test_score_range_with_zero_threshold(self, text, memory_type):
        """**Validates: Requirements 3.1**
        
        Property 1: Importance Score Range Validation (with zero threshold)
        
        With threshold=0.0, all candidates should be scored and all scores
        must be in the valid range [0.0, 1.0].
        """
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(text=text, type=memory_type)
        
        result = scorer.score_memory(candidate)
        
        # With threshold 0.0, all candidates should be scored
        assert result is not None, "Expected result with threshold=0.0"
        assert 0.0 <= result.importance <= 1.0, (
            f"Score {result.importance} is outside valid range [0.0, 1.0]"
        )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy
    )
    def test_score_never_exceeds_one(self, candidate):
        """**Validates: Requirements 3.1**
        
        Property 1: Importance Score Range Validation (upper bound)
        
        For any candidate memory, the importance score must never exceed 1.0,
        even with content adjustments and keyword boosts.
        """
        scorer = ImportanceScorer(threshold=0.0)
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance <= 1.0, (
            f"Score {result.importance} exceeds maximum value 1.0 "
            f"for candidate: {candidate.text[:50]}..."
        )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy
    )
    def test_score_never_below_zero(self, candidate):
        """**Validates: Requirements 3.1**
        
        Property 1: Importance Score Range Validation (lower bound)
        
        For any candidate memory, the importance score must never be below 0.0,
        even with content penalties and adjustments.
        """
        scorer = ImportanceScorer(threshold=0.0)
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance >= 0.0, (
            f"Score {result.importance} is below minimum value 0.0 "
            f"for candidate: {candidate.text[:50]}..."
        )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold1=threshold_strategy,
        threshold2=threshold_strategy
    )
    def test_score_consistent_across_thresholds(self, candidate, threshold1, threshold2):
        """**Validates: Requirements 3.1**
        
        Property 1: Importance Score Range Validation (consistency)
        
        The importance score assigned to a candidate should be the same
        regardless of the threshold value (threshold only affects filtering).
        """
        scorer1 = ImportanceScorer(threshold=threshold1)
        scorer2 = ImportanceScorer(threshold=threshold2)
        
        result1 = scorer1.score_memory(candidate)
        result2 = scorer2.score_memory(candidate)
        
        # If both returned results, scores should be identical
        if result1 is not None and result2 is not None:
            assert result1.importance == result2.importance, (
                f"Score inconsistent across thresholds: {result1.importance} vs {result2.importance}"
            )
            # Both scores must be in valid range
            assert 0.0 <= result1.importance <= 1.0
            assert 0.0 <= result2.importance <= 1.0
