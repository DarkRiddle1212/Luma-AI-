"""Property-based tests for Importance Scorer threshold filtering.

This module tests Property 2: Threshold Filtering Consistency
For any candidate memory with an importance score below the configured threshold,
the Importance Scorer must filter it out and not return it in the scored memories list.

**Validates: Requirements 3.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
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


class TestThresholdFilteringProperty:
    """Property tests for threshold filtering consistency.
    
    Feature: memory-write-engine
    Property 2: Threshold Filtering Consistency
    """
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold=threshold_strategy
    )
    def test_filtering_respects_threshold(self, candidate, threshold):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency
        
        For any candidate memory with an importance score below the configured
        threshold, the Importance Scorer must filter it out (return None).
        """
        scorer = ImportanceScorer(threshold=threshold)
        
        result = scorer.score_memory(candidate)
        
        # If result is returned, its score must be >= threshold
        if result is not None:
            assert result.importance >= threshold, (
                f"Returned memory with score {result.importance} below threshold {threshold}"
            )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold=threshold_strategy
    )
    def test_below_threshold_returns_none(self, candidate, threshold):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (explicit None check)
        
        When a candidate's score is below the threshold, the scorer must
        return None (not a ScoredMemory object).
        """
        # First, get the score with threshold=0.0 to see actual score
        scorer_zero = ImportanceScorer(threshold=0.0)
        result_zero = scorer_zero.score_memory(candidate)
        
        # If the actual score exists and is below our test threshold
        if result_zero is not None and result_zero.importance < threshold:
            scorer = ImportanceScorer(threshold=threshold)
            result = scorer.score_memory(candidate)
            
            # Must return None since score < threshold
            assert result is None, (
                f"Expected None for score {result_zero.importance} < threshold {threshold}, "
                f"but got {result}"
            )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold=threshold_strategy
    )
    def test_at_or_above_threshold_returns_scored_memory(self, candidate, threshold):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (acceptance check)
        
        When a candidate's score is at or above the threshold, the scorer must
        return a ScoredMemory object (not None).
        """
        # First, get the score with threshold=0.0 to see actual score
        scorer_zero = ImportanceScorer(threshold=0.0)
        result_zero = scorer_zero.score_memory(candidate)
        
        # If the actual score exists and is >= our test threshold
        if result_zero is not None and result_zero.importance >= threshold:
            scorer = ImportanceScorer(threshold=threshold)
            result = scorer.score_memory(candidate)
            
            # Must return a ScoredMemory since score >= threshold
            assert result is not None, (
                f"Expected ScoredMemory for score {result_zero.importance} >= threshold {threshold}, "
                f"but got None"
            )
            assert result.importance == result_zero.importance
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy
    )
    def test_threshold_zero_accepts_all(self, candidate):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (zero threshold)
        
        With threshold=0.0, all valid candidates should be accepted
        (none should be filtered out).
        """
        scorer = ImportanceScorer(threshold=0.0)
        
        result = scorer.score_memory(candidate)
        
        # With threshold 0.0, all candidates should pass
        assert result is not None, (
            f"Expected all candidates to pass with threshold=0.0, "
            f"but got None for: {candidate.text[:50]}..."
        )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy
    )
    def test_threshold_one_filters_most(self, candidate):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (maximum threshold)
        
        With threshold=1.0, only candidates with perfect score (1.0) should
        be accepted. Most candidates should be filtered out.
        """
        scorer = ImportanceScorer(threshold=1.0)
        
        result = scorer.score_memory(candidate)
        
        # If result is returned, score must be exactly 1.0
        if result is not None:
            assert result.importance == 1.0, (
                f"With threshold=1.0, only score 1.0 should pass, "
                f"but got score {result.importance}"
            )
    
    @settings(max_examples=10)
    @given(
        candidate=memory_candidate_strategy,
        threshold1=threshold_strategy,
        threshold2=threshold_strategy
    )
    def test_higher_threshold_filters_more(self, candidate, threshold1, threshold2):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (monotonicity)
        
        A higher threshold should filter out at least as many candidates
        as a lower threshold (monotonic filtering behavior).
        """
        assume(threshold1 < threshold2)  # Ensure threshold1 < threshold2
        
        scorer1 = ImportanceScorer(threshold=threshold1)
        scorer2 = ImportanceScorer(threshold=threshold2)
        
        result1 = scorer1.score_memory(candidate)
        result2 = scorer2.score_memory(candidate)
        
        # If lower threshold filters out, higher threshold must also filter out
        if result1 is None:
            assert result2 is None, (
                f"Lower threshold {threshold1} filtered out, but higher threshold "
                f"{threshold2} did not"
            )
        
        # If higher threshold accepts, lower threshold must also accept
        if result2 is not None:
            assert result1 is not None, (
                f"Higher threshold {threshold2} accepted, but lower threshold "
                f"{threshold1} did not"
            )
    
    @settings(max_examples=10)
    @given(
        text=text_strategy,
        memory_type=memory_type_strategy,
        threshold=threshold_strategy
    )
    def test_filtered_candidates_not_in_result(self, text, memory_type, threshold):
        """**Validates: Requirements 3.5**
        
        Property 2: Threshold Filtering Consistency (exclusion)
        
        Candidates that are filtered out must not appear in any form in the
        result (result must be None, not an empty or partial ScoredMemory).
        """
        scorer = ImportanceScorer(threshold=threshold)
        candidate = MemoryCandidate(text=text, type=memory_type)
        
        result = scorer.score_memory(candidate)
        
        # Result is either None or a complete ScoredMemory
        if result is None:
            # Verify it was actually filtered (score < threshold)
            scorer_zero = ImportanceScorer(threshold=0.0)
            result_zero = scorer_zero.score_memory(candidate)
            if result_zero is not None:
                assert result_zero.importance < threshold, (
                    f"Candidate was filtered but score {result_zero.importance} >= threshold {threshold}"
                )
        else:
            # If not None, must be a valid ScoredMemory with all fields
            assert hasattr(result, 'text')
            assert hasattr(result, 'type')
            assert hasattr(result, 'importance')
            assert result.text == text
            assert result.type == memory_type
            assert result.importance >= threshold
