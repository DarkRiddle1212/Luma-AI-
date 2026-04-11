"""
Property-Based Tests for Monotonic Token Accumulation

This module implements property-based tests using Hypothesis to verify
that the cumulative token count is monotonically increasing as memories
are selected during injection.

Feature: context-injection-engine
Property 10: Monotonic Token Accumulation
Validates: Requirements 2.2
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timezone
from typing import List, Any

from luma.core.injection_engine import TokenBudgetEnforcer, TokenEstimator


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
        # Generate a reasonable token count (1-500)
        metadata['token_count'] = draw(st.integers(min_value=1, max_value=500))
    
    return metadata


@st.composite
def ranked_memory_strategy(draw):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing monotonic token accumulation.
    """
    memory_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    # Generate content with various lengths
    # Use word-based content for more realistic token estimation
    num_words = draw(st.integers(min_value=1, max_value=100))
    words = [draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll')
    ))) for _ in range(num_words)]
    content = ' '.join(words)
    
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
    
    # Create a mock memory object with metadata attribute
    class MockMemory:
        def __init__(self, memory_id, timestamp, content, namespace, 
                     similarity_score, importance_score, recency_score, 
                     final_score, metadata):
            self.memory_id = memory_id
            self.timestamp = timestamp
            self.content = content
            self.namespace = namespace
            self.similarity_score = similarity_score
            self.importance_score = importance_score
            self.recency_score = recency_score
            self.final_score = final_score
            self.metadata = metadata
            self.memory_entry = None
    
    return MockMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content=content,
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=recency_score,
        final_score=final_score,
        metadata=metadata
    )


@st.composite
def memory_list_strategy(draw, min_size=0, max_size=5):
    """Generate lists of memories.
    
    Creates lists of mock memory objects with unique memory_ids.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
    
    Returns:
        List of mock memory objects
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    memories = []
    used_ids = set()
    
    for i in range(size):
        memory = draw(ranked_memory_strategy())
        
        # Ensure unique memory_id
        counter = 0
        while memory.memory_id in used_ids:
            memory.memory_id = f"{memory.memory_id}_{counter}"
            counter += 1
        
        used_ids.add(memory.memory_id)
        memories.append(memory)
    
    return memories


# ============================================================================
# Property 10: Monotonic Token Accumulation
# ============================================================================

# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_token_budget=st.integers(min_value=100, max_value=10000),
    max_memory_count=st.integers(min_value=1, max_value=100),
    estimation_factor=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_monotonic_token_accumulation(memories, max_token_budget, max_memory_count, estimation_factor):
    """
    Property: For any sequence of memories selected during injection,
    the cumulative token count should be monotonically increasing
    (each memory adds non-negative tokens).
    
    **Validates: Requirements 2.2**
    
    This test verifies that:
    1. Token count never decreases as memories are selected
    2. Each memory adds a non-negative number of tokens
    3. The cumulative token count is monotonically increasing
    4. The property holds for all valid inputs
    """
    # Create token estimator and budget enforcer
    token_estimator = TokenEstimator(estimation_factor=estimation_factor)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    # Apply budget enforcement
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Track cumulative token count as we iterate through selected memories
    cumulative_tokens = 0
    token_sequence = [0]  # Start with 0 tokens
    
    for memory in selected_memories:
        memory_tokens = token_estimator.estimate_tokens(memory)
        
        # Each memory should add non-negative tokens
        assert memory_tokens >= 0, (
            f"Memory added negative tokens: {memory_tokens}. "
            f"Memory ID: {memory.memory_id}, Content length: {len(memory.content)}"
        )
        
        cumulative_tokens += memory_tokens
        token_sequence.append(cumulative_tokens)
    
    # Verify monotonic property: each element should be >= previous element
    for i in range(1, len(token_sequence)):
        assert token_sequence[i] >= token_sequence[i-1], (
            f"Token count decreased: token_sequence[{i-1}]={token_sequence[i-1]} > "
            f"token_sequence[{i}]={token_sequence[i]}. "
            f"This violates the monotonic accumulation property."
        )
    
    # Verify final cumulative matches reported total
    assert cumulative_tokens == total_tokens, (
        f"Final cumulative tokens ({cumulative_tokens}) does not match "
        f"reported total_tokens ({total_tokens})"
    )


# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    max_token_budget=st.integers(min_value=100, max_value=5000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_monotonic_accumulation_with_precomputed_tokens(memories, max_token_budget):
    """
    Property: Monotonic token accumulation holds when memories have
    precomputed token_count values in metadata.
    
    **Validates: Requirements 2.2**
    
    This test ensures the monotonic property holds when using the
    precomputed token_count code path.
    """
    # Ensure all memories have precomputed token_count
    for memory in memories:
        if 'token_count' not in memory.metadata:
            # Add a precomputed token count based on content length
            word_count = len(memory.content.split())
            memory.metadata['token_count'] = max(1, int(word_count * 1.3))
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=100,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Track cumulative token count
    cumulative_tokens = 0
    previous_cumulative = 0
    
    for memory in selected_memories:
        memory_tokens = token_estimator.estimate_tokens(memory)
        cumulative_tokens += memory_tokens
        
        # Verify monotonic property
        assert cumulative_tokens >= previous_cumulative, (
            f"Token count decreased with precomputed tokens: "
            f"previous={previous_cumulative}, current={cumulative_tokens}"
        )
        
        previous_cumulative = cumulative_tokens


# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    max_token_budget=st.integers(min_value=100, max_value=5000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_monotonic_accumulation_with_approximation(memories, max_token_budget):
    """
    Property: Monotonic token accumulation holds when using word-count
    approximation (no precomputed token_count in metadata).
    
    **Validates: Requirements 2.2**
    
    This test ensures the monotonic property holds when using the
    fallback word-count approximation code path.
    """
    # Ensure no memories have precomputed token_count
    for memory in memories:
        if 'token_count' in memory.metadata:
            del memory.metadata['token_count']
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=100,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Track cumulative token count
    cumulative_tokens = 0
    previous_cumulative = 0
    
    for memory in selected_memories:
        memory_tokens = token_estimator.estimate_tokens(memory)
        cumulative_tokens += memory_tokens
        
        # Verify monotonic property
        assert cumulative_tokens >= previous_cumulative, (
            f"Token count decreased with approximation: "
            f"previous={previous_cumulative}, current={cumulative_tokens}"
        )
        
        previous_cumulative = cumulative_tokens


# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(
    memories=memory_list_strategy(min_size=2, max_size=5),
    max_token_budget=st.integers(min_value=50, max_value=1000),
    max_memory_count=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_monotonic_accumulation_with_tight_constraints(memories, max_token_budget, max_memory_count):
    """
    Property: Monotonic token accumulation holds even with tight budget
    and memory count constraints.
    
    **Validates: Requirements 2.2**
    
    This test creates scenarios where constraints are tight, ensuring
    the monotonic property holds even when many memories are filtered.
    """
    # Assume we have at least 2 memories to make the test meaningful
    assume(len(memories) >= 2)
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Track cumulative token count
    token_sequence = []
    cumulative_tokens = 0
    
    for memory in selected_memories:
        memory_tokens = token_estimator.estimate_tokens(memory)
        cumulative_tokens += memory_tokens
        token_sequence.append(cumulative_tokens)
    
    # Verify monotonic property across the entire sequence
    for i in range(1, len(token_sequence)):
        assert token_sequence[i] >= token_sequence[i-1], (
            f"Token count decreased with tight constraints: "
            f"token_sequence[{i-1}]={token_sequence[i-1]} > "
            f"token_sequence[{i}]={token_sequence[i]}"
        )


# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(
    memories=memory_list_strategy(min_size=5, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_each_memory_adds_positive_tokens(memories):
    """
    Property: Each memory adds a positive (non-zero) number of tokens
    to the cumulative count.
    
    **Validates: Requirements 2.2**
    
    This test verifies that every memory contributes at least 1 token,
    ensuring the cumulative count strictly increases (not just non-decreasing).
    """
    # Assume we have at least 5 memories
    assume(len(memories) >= 5)
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100000,  # Large budget to select many memories
        max_memory_count=100,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Verify each selected memory adds positive tokens
    for memory in selected_memories:
        memory_tokens = token_estimator.estimate_tokens(memory)
        
        # Each memory should add at least 1 token
        assert memory_tokens > 0, (
            f"Memory added zero or negative tokens: {memory_tokens}. "
            f"Memory ID: {memory.memory_id}, "
            f"Content: '{memory.content[:50]}...', "
            f"Content length: {len(memory.content)}, "
            f"Word count: {len(memory.content.split())}"
        )


# Feature: context-injection-engine, Property 10: Monotonic Token Accumulation
@given(memories=memory_list_strategy(min_size=0, max_size=0))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_monotonic_accumulation_with_empty_input(memories):
    """
    Property: Monotonic token accumulation holds for empty input (edge case).
    
    **Validates: Requirements 2.2**
    
    This test verifies that with empty input, the token sequence is [0]
    (trivially monotonic).
    """
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=1000,
        max_memory_count=50,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # With empty input, should return empty result
    assert len(selected_memories) == 0, "Empty input should produce empty output"
    assert total_tokens == 0, "Empty input should produce zero tokens"
    
    # Token sequence is [0] (trivially monotonic)
    token_sequence = [0]
    
    # Verify monotonic property (trivially true for single element)
    for i in range(1, len(token_sequence)):
        assert token_sequence[i] >= token_sequence[i-1]
