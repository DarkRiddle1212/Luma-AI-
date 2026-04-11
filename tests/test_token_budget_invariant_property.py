"""
Property-Based Tests for Token Budget Invariant

This module implements property-based tests using Hypothesis to verify
that the TokenBudgetEnforcer ensures the total_tokens in the result
never exceeds max_token_budget.

Feature: context-injection-engine
Property 1: Token Budget Invariant
Validates: Requirements 2.3, 2.4
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timezone
from typing import List, Any

from luma.core.injection_engine import TokenBudgetEnforcer, TokenEstimator
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
        # Generate a reasonable token count (1-500)
        metadata['token_count'] = draw(st.integers(min_value=1, max_value=500))
    
    return metadata


@st.composite
def ranked_memory_strategy(draw):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing token budget enforcement.
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
# Property 1: Token Budget Invariant
# ============================================================================

# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_token_budget=st.integers(min_value=10, max_value=10000),
    max_memory_count=st.integers(min_value=1, max_value=100),
    estimation_factor=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_never_exceeded(memories, max_token_budget, max_memory_count, estimation_factor):
    """
    Property: For any list of ranked memories and any valid injection configuration,
    the total_tokens in the result should never exceed max_token_budget.
    
    **Validates: Requirements 2.3, 2.4**
    
    This test verifies that:
    1. TokenBudgetEnforcer respects the max_token_budget constraint
    2. The total_tokens returned never exceeds max_token_budget
    3. The invariant holds for all valid inputs
    4. The invariant holds for all budget values
    5. The invariant holds regardless of estimation_factor
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
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= max_token_budget, (
        f"Token budget invariant violated: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}. "
        f"Selected {len(selected_memories)} memories from {len(memories)} input memories. "
        f"Estimation factor: {estimation_factor}"
    )
    
    # Additional verification: verify the total by re-computing
    recomputed_total = sum(
        token_estimator.estimate_tokens(memory)
        for memory in selected_memories
    )
    
    assert recomputed_total == total_tokens, (
        f"Reported total_tokens={total_tokens} does not match "
        f"recomputed total={recomputed_total}"
    )
    
    # Verify recomputed total also respects the budget
    assert recomputed_total <= max_token_budget, (
        f"Recomputed total_tokens={recomputed_total} exceeds "
        f"max_token_budget={max_token_budget}"
    )


# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    max_token_budget=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_with_tight_budget(memories, max_token_budget):
    """
    Property: Token budget invariant holds even with very tight budgets
    that may only allow a few or even zero memories.
    
    **Validates: Requirements 2.3, 2.4**
    
    This test creates scenarios where the budget is tight relative to
    memory sizes, ensuring the enforcer handles edge cases correctly.
    """
    # Assume we have at least one memory to make the test meaningful
    assume(len(memories) >= 1)
    
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=100,  # High count to focus on token budget
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= max_token_budget, (
        f"Token budget invariant violated with tight budget: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}. "
        f"Selected {len(selected_memories)} memories."
    )


# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_token_budget=st.integers(min_value=100, max_value=10000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_with_precomputed_token_counts(memories, max_token_budget):
    """
    Property: Token budget invariant holds when memories have precomputed
    token_count values in metadata.
    
    **Validates: Requirements 2.3, 2.4**
    
    This test ensures the invariant holds when using the precomputed
    token_count code path.
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
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= max_token_budget, (
        f"Token budget invariant violated with precomputed token counts: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}"
    )


# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_token_budget=st.integers(min_value=100, max_value=10000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_with_approximation_fallback(memories, max_token_budget):
    """
    Property: Token budget invariant holds when using word-count approximation
    (no precomputed token_count in metadata).
    
    **Validates: Requirements 2.3, 2.4**
    
    This test ensures the invariant holds when using the fallback
    word-count approximation code path.
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
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= max_token_budget, (
        f"Token budget invariant violated with approximation fallback: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}"
    )


# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    max_token_budget=st.integers(min_value=10, max_value=10000),
    max_memory_count=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_with_dual_constraints(memories, max_token_budget, max_memory_count):
    """
    Property: Token budget invariant holds when both token budget and
    memory count constraints are active.
    
    **Validates: Requirements 2.3, 2.4, 8.3**
    
    This test verifies that the token budget invariant is maintained
    even when the memory count limit is also enforced.
    """
    token_estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=max_token_budget,
        max_memory_count=max_memory_count,
        token_estimator=token_estimator
    )
    
    selected_memories, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= max_token_budget, (
        f"Token budget invariant violated with dual constraints: "
        f"total_tokens={total_tokens} exceeds max_token_budget={max_token_budget}. "
        f"max_memory_count={max_memory_count}, selected={len(selected_memories)}"
    )
    
    # Also verify memory count constraint
    assert len(selected_memories) <= max_memory_count, (
        f"Memory count constraint violated: "
        f"selected {len(selected_memories)} memories, max_memory_count={max_memory_count}"
    )


# Feature: context-injection-engine, Property 1: Token Budget Invariant
@given(memories=memory_list_strategy(min_size=0, max_size=0))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_token_budget_with_empty_input(memories):
    """
    Property: Token budget invariant holds for empty input (edge case).
    
    **Validates: Requirements 2.3, 2.4**
    
    This test verifies that the enforcer handles empty input correctly
    and returns zero tokens.
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
    assert filtered_count == 0, "Empty input should have zero filtered count"
    
    # THE INVARIANT: total_tokens must NEVER exceed max_token_budget
    assert total_tokens <= 1000, "Token budget invariant violated for empty input"