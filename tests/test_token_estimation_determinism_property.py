"""
Property-Based Tests for Token Estimation Determinism

This module implements property-based tests using Hypothesis to verify
that TokenEstimator produces deterministic results - calling estimate_tokens()
multiple times on the same memory always returns the same token count.

Feature: context-injection-engine
Property 8: Token Estimation Determinism
Validates: Requirements 2.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from luma.core.injection_engine import TokenEstimator
from luma.core.ranking_engine import RankedMemory


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def metadata_strategy(draw):
    """Generate random metadata dictionaries.
    
    Metadata can optionally contain a precomputed token_count field.
    This strategy generates realistic metadata for testing both code paths:
    - With precomputed token_count (should use that value)
    - Without token_count (should use word-count approximation)
    """
    # 50% chance of including precomputed token_count
    include_token_count = draw(st.booleans())
    
    metadata = {}
    
    if include_token_count:
        # Generate a reasonable token count (0-10000)
        metadata['token_count'] = draw(st.integers(min_value=0, max_value=10000))
    
    # Add some other metadata fields
    num_fields = draw(st.integers(min_value=0, max_value=3))
    for i in range(num_fields):
        key = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_'
        )))
        if key != 'token_count':  # Don't overwrite token_count
            value = draw(st.one_of(
                st.text(max_size=5),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ))
            metadata[key] = value
    
    return metadata


@st.composite
def ranked_memory_strategy(draw):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing token estimation determinism.
    """
    memory_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    # Generate content with various characteristics
    content = draw(st.text(
        min_size=0,
        max_size=50,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')
        )
    ))
    
    metadata = draw(metadata_strategy())
    
    # Generate timestamp with timezone
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        timezones=st.just(timezone.utc)
    ))
    
    # Namespace is optional
    namespace = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        ))
    ))
    
    # Generate valid scores [0, 1]
    similarity_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    importance_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    recency_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    final_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content=content,
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=recency_score,
        final_score=final_score,
        memory_entry=None  # Not needed for token estimation
    )


# ============================================================================
# Property 8: Token Estimation Determinism
# ============================================================================

# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(memory=ranked_memory_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism(memory):
    """
    Property: For any RankedMemory, calling estimate_tokens() multiple times
    should always return the same token count value.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. TokenEstimator produces deterministic results
    2. Same input always produces same output
    3. No randomness or non-deterministic operations are used
    4. Both code paths (precomputed and approximation) are deterministic
    5. Multiple calls produce identical results
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Call estimate_tokens multiple times (5 iterations)
    results = []
    for _ in range(5):
        token_count = estimator.estimate_tokens(memory)
        results.append(token_count)
    
    # All results should be identical
    first_result = results[0]
    for i, result in enumerate(results[1:], start=1):
        assert result == first_result, (
            f"Token estimation is not deterministic: "
            f"call 0 returned {first_result}, call {i} returned {result}. "
            f"Memory content length: {len(memory.content)}, "
            f"Has precomputed token_count: {'token_count' in memory.metadata}"
        )
    
    # Verify result is an integer (deterministic type)
    assert isinstance(first_result, int), (
        f"Token count should be an integer, got {type(first_result)}"
    )
    
    # Verify result is non-negative
    assert first_result >= 0, (
        f"Token count should be non-negative, got {first_result}"
    )


# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(
    memory=ranked_memory_strategy(),
    estimation_factor=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism_with_custom_factor(memory, estimation_factor):
    """
    Property: Token estimation determinism holds for any valid estimation_factor.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Determinism holds regardless of estimation_factor value
    2. Custom estimation factors don't introduce non-determinism
    3. Results are consistent across multiple calls
    """
    estimator = TokenEstimator(estimation_factor=estimation_factor)
    
    # Call estimate_tokens multiple times
    results = []
    for _ in range(3):
        token_count = estimator.estimate_tokens(memory)
        results.append(token_count)
    
    # All results should be identical
    assert len(set(results)) == 1, (
        f"Token estimation with custom factor {estimation_factor} is not deterministic: "
        f"got different results {results}"
    )


# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(content=st.text(min_size=0, max_size=5))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism_without_precomputed(content):
    """
    Property: Token estimation is deterministic when using word-count approximation
    (no precomputed token_count in metadata).
    
    **Validates: Requirements 2.5**
    
    This test specifically verifies the determinism of the fallback
    word-count approximation code path.
    """
    # Create a simple memory object without precomputed token_count
    class SimpleMemory:
        def __init__(self, content):
            self.content = content
            self.metadata = {}  # No token_count
    
    memory = SimpleMemory(content)
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Call estimate_tokens multiple times
    results = []
    for _ in range(5):
        token_count = estimator.estimate_tokens(memory)
        results.append(token_count)
    
    # All results should be identical
    first_result = results[0]
    for i, result in enumerate(results[1:], start=1):
        assert result == first_result, (
            f"Word-count approximation is not deterministic: "
            f"call 0 returned {first_result}, call {i} returned {result}. "
            f"Content: {repr(content[:50])}"
        )


# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(token_count=st.integers(min_value=0, max_value=100000))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism_with_precomputed(token_count):
    """
    Property: Token estimation is deterministic when using precomputed token_count
    from metadata.
    
    **Validates: Requirements 2.5**
    
    This test specifically verifies the determinism of the precomputed
    token_count code path.
    """
    # Create a simple memory object with precomputed token_count
    class SimpleMemory:
        def __init__(self, token_count):
            self.content = "Some content"
            self.metadata = {"token_count": token_count}
    
    memory = SimpleMemory(token_count)
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Call estimate_tokens multiple times
    results = []
    for _ in range(5):
        result = estimator.estimate_tokens(memory)
        results.append(result)
    
    # All results should be identical and equal to precomputed value
    expected = int(token_count)
    for i, result in enumerate(results):
        assert result == expected, (
            f"Precomputed token_count not used deterministically: "
            f"call {i} returned {result}, expected {expected}"
        )


# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(memory=ranked_memory_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism_across_estimator_instances(memory):
    """
    Property: Token estimation is deterministic across different TokenEstimator
    instances with the same configuration.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Different estimator instances produce same results
    2. No hidden state affects estimation
    3. Estimation is purely functional
    """
    # Create two separate estimator instances with same config
    estimator1 = TokenEstimator(estimation_factor=1.3)
    estimator2 = TokenEstimator(estimation_factor=1.3)
    
    # Estimate tokens with both estimators
    result1 = estimator1.estimate_tokens(memory)
    result2 = estimator2.estimate_tokens(memory)
    
    # Results should be identical
    assert result1 == result2, (
        f"Token estimation differs across estimator instances: "
        f"estimator1 returned {result1}, estimator2 returned {result2}"
    )


# Feature: context-injection-engine, Property 8: Token Estimation Determinism
@given(memory=ranked_memory_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_estimation_determinism_with_reused_estimator(memory):
    """
    Property: Token estimation is deterministic when reusing the same
    TokenEstimator instance for multiple memories.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Estimator state doesn't affect subsequent estimations
    2. Estimator can be safely reused
    3. No side effects from previous estimations
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Estimate tokens for the same memory multiple times
    # with other estimations in between
    result1 = estimator.estimate_tokens(memory)
    
    # Create and estimate a different memory (to potentially corrupt state)
    other_memory = RankedMemory(
        memory_id="other",
        timestamp=datetime.now(timezone.utc),
        content="Different content",
        namespace=None,
        similarity_score=0.5,
        importance_score=0.5,
        recency_score=0.5,
        final_score=0.5,
        memory_entry=None
    )
    _ = estimator.estimate_tokens(other_memory)
    
    # Estimate original memory again
    result2 = estimator.estimate_tokens(memory)
    
    # Results should be identical
    assert result1 == result2, (
        f"Token estimation changed after reusing estimator: "
        f"first call returned {result1}, second call returned {result2}"
    )
