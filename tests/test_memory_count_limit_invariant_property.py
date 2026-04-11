"""
Property-Based Tests for Memory Count Limit Invariant

This module implements property-based tests using Hypothesis to verify
that the TokenBudgetEnforcer ensures the number of memories in the result
never exceeds max_memory_count.

Feature: context-injection-engine
Property 2: Memory Count Limit Invariant
Validates: Requirements 8.2
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
    for testing memory count limit enforcement.
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
# Property 2: Memory Count Limit Invariant
# ============================================================================

# Feature: context-injection-engine, Property 2: Memory Count Limit Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=100),
    max_memory_count=st.integers(min_value=1, max_value=50),
    max_token_budget=st.integers(min_value=100, max_value=100000),
    estimation_factor=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memory_count_never_exceeded(memories, max_memory_count, max_token_budget, estimation_factor):
    """
    Property: For any list of ranked memories and any valid injection configuration,
    the number of memories in the result should never exceed max_memory_count.
    
    **Validates: Requirements 8.2**
    
    This test verifies that:
    1. TokenBudgetEnforcer respects the max_memory_count constraint
    2. The number of selected memories never exceeds max_memory_count
    3. The invariant holds for all valid inputs
    4. The invariant holds for all memory count limits
    5. The invariant holds regardless of token budget or estimation_factor
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
    
    # THE INVARIANT: number of selected memories must NEVER exceed max_memory_count
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count limit invariant violated: "
        f"selected {len(selected_memories)} memories but max_memory_count={max_memory_count}. "
        f"Input had {len(memories)} memories. "
        f"Token budget: {max_token_budget}, total_tokens: {total_tokens}"
    )
    
    # Additional verification: if we have enough memories and token budget,
    # we should select exactly max_memory_count memories
    if len(memories) >= max_memory_count:
        # Check if token budget allows for max_memory_count memories
        # by computing tokens for first max_memory_count memories
        tokens_for_max_count = sum(
            token_estimator.estimate_tokens(memory)
            for memory in memories[:max_memory_count]
        )
        
        # If token budget allows, we should have selected exactly max_memory_count
        if tokens_for_max_count <= max_token_budget:
            assert len(selected_memories) == max_memory_count, (
                f"Expected exactly {max_memory_count} memories when budget allows, "
                f"but got {len(selected_memories)}. "
                f"Token budget: {max_token_budget}, tokens needed: {tokens_for_max_count}"
            )


# Feature: context-injection-engine, Property 2: Memory Count Limit Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=100),
    max_memory_count=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memory_count_with_small_limit(memories, max_memory_count):
    """
    Property: Memory count limit invariant holds even with very small limits
    that may only allow a few memories.
    
    **Validates: Requirements 8.2**
    
    This test creates scenarios where the memory count limit is small relative
    to the input size, ensuring the enforcer handles edge cases correctly.
    """
    # Assume we have at least one memory to make the test meaningful
    assume(len(memories) >= 1)
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=1000000,  # Very high budget to focus on memory count
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: number of selected memories must NEVER exceed max_memory_count
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count limit invariant violated with small limit: "
        f"selected {len(selected_memories)} memories but max_memory_count={max_memory_count}. "
        f"Input had {len(memories)} memories."
    )
    
    # Verify filtered count is accurate
    assert filtered_count == len(memories) - len(selected_memories), (
        f"Filtered count mismatch: filtered_count={filtered_count} but "
        f"expected {len(memories) - len(selected_memories)}"
    )


# Feature: context-injection-engine, Property 2: Memory Count Limit Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_memory_count=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memory_count_with_large_limit(memories, max_memory_count):
    """
    Property: Memory count limit invariant holds when the limit is larger
    than the input size (all memories should be selected if budget allows).
    
    **Validates: Requirements 8.2**
    
    This test verifies that when max_memory_count is larger than the input,
    the enforcer correctly selects all memories (up to token budget).
    """
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=1000000,  # Very high budget to focus on memory count
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: number of selected memories must NEVER exceed max_memory_count
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count limit invariant violated with large limit: "
        f"selected {len(selected_memories)} memories but max_memory_count={max_memory_count}. "
        f"Input had {len(memories)} memories."
    )
    
    # When limit is larger than input, we should select all memories (if budget allows)
    if max_memory_count >= len(memories):
        assert len(selected_memories) == len(memories), (
            f"Expected all {len(memories)} memories to be selected when "
            f"max_memory_count={max_memory_count} >= input size, but got {len(selected_memories)}"
        )


# Feature: context-injection-engine, Property 2: Memory Count Limit Invariant
@given(
    memories=memory_list_strategy(min_size=5, max_size=5),
    max_memory_count=st.integers(min_value=1, max_value=20),
    max_token_budget=st.integers(min_value=10, max_value=1000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memory_count_with_dual_constraints(memories, max_memory_count, max_token_budget):
    """
    Property: Memory count limit invariant holds when both token budget
    and memory count constraints are active simultaneously.
    
    **Validates: Requirements 8.2, 8.3**
    
    This test verifies that the enforcer correctly handles the interaction
    between token budget and memory count constraints, respecting both limits.
    """
    # Assume we have at least one memory to make the test meaningful
    assume(len(memories) >= 1)
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: number of selected memories must NEVER exceed max_memory_count
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count limit invariant violated with dual constraints: "
        f"selected {len(selected_memories)} memories but max_memory_count={max_memory_count}. "
        f"Token budget: {max_token_budget}, total_tokens: {total_tokens}"
    )
    
    # Also verify token budget constraint
    assert total_tokens <= max_token_budget, (
        f"Token budget constraint violated: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}"
    )
    
    # Verify that we stopped at the first constraint that was hit
    # Either we hit memory count limit or token budget limit (or ran out of memories)
    if len(selected_memories) < len(memories):
        # We filtered some memories, so we must have hit a constraint
        hit_memory_count_limit = len(selected_memories) == max_memory_count
        
        # To check if we hit token budget, we need to find the first unselected memory
        # and see if it would exceed the budget
        would_exceed_token_budget = False
        
        # Find the first memory that wasn't selected
        # We need to iterate through memories and check which ones were selected
        selected_ids = {mem.memory_id for mem in selected_memories}
        for memory in memories:
            if memory.memory_id not in selected_ids:
                # This is the first unselected memory
                next_memory_tokens = token_estimator.estimate_tokens(memory)
                would_exceed_token_budget = (total_tokens + next_memory_tokens) > max_token_budget
                break
        
        # We should have stopped for one of these reasons
        assert hit_memory_count_limit or would_exceed_token_budget, (
            f"Stopped selecting memories but neither constraint was hit. "
            f"Selected: {len(selected_memories)}, max_memory_count: {max_memory_count}, "
            f"total_tokens: {total_tokens}, max_token_budget: {max_token_budget}"
        )


# Feature: context-injection-engine, Property 2: Memory Count Limit Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_memory_count=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memory_count_with_empty_input(memories, max_memory_count):
    """
    Property: Memory count limit invariant holds for empty input lists.
    
    **Validates: Requirements 8.2**
    
    This test verifies that the enforcer correctly handles empty input,
    returning an empty result regardless of max_memory_count.
    """
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10000,
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: number of selected memories must NEVER exceed max_memory_count
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count limit invariant violated: "
        f"selected {len(selected_memories)} memories but max_memory_count={max_memory_count}"
    )
    
    # For empty input, we should always get empty output
    if len(memories) == 0:
        assert len(selected_memories) == 0, (
            f"Expected empty output for empty input, but got {len(selected_memories)} memories"
        )
        assert total_tokens == 0, (
            f"Expected zero tokens for empty input, but got {total_tokens}"
        )
        assert filtered_count == 0, (
            f"Expected zero filtered count for empty input, but got {filtered_count}"
        )
