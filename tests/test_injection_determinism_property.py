"""
Property-Based Tests for Injection Determinism (Idempotence)

This module implements property-based tests using Hypothesis to verify
that the InjectionEngine produces identical results when called multiple times
with the same inputs.

Feature: context-injection-engine
Property 7: Injection Determinism (Idempotence)
Validates: Requirements 6.3, 6.5
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timezone
from typing import List, Any

from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def metadata_strategy(draw, embedding_dim=None):
    """Generate random metadata dictionaries.
    
    Metadata can optionally contain a precomputed token_count field
    and embeddings for redundancy testing.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    metadata = {}
    
    # 50% chance of including precomputed token_count
    include_token_count = draw(st.booleans())
    if include_token_count:
        metadata['token_count'] = draw(st.integers(min_value=1, max_value=500))
    
    # Add embedding with fixed dimension if specified
    # Use smaller embedding dimension to reduce data size
    if embedding_dim is not None:
        metadata['embedding'] = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=embedding_dim, max_size=embedding_dim
        ))
    
    # Reduce additional fields to minimize data size
    num_fields = draw(st.integers(min_value=0, max_value=2))
    for i in range(num_fields):
        field_name = f"field_{i}"
        field_type = draw(st.sampled_from(['string', 'int']))
        
        if field_type == 'string':
            metadata[field_name] = draw(st.text(min_size=0, max_size=10))
        elif field_type == 'int':
            metadata[field_name] = draw(st.integers(min_value=-100, max_value=100))
    
    return metadata


@st.composite
def ranked_memory_strategy(draw, embedding_dim=None):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing injection determinism.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    # Generate shorter content to reduce data size
    num_words = draw(st.integers(min_value=1, max_value=20))
    words = [draw(st.text(min_size=1, max_size=5, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll')
    ))) for _ in range(num_words)]
    content = ' '.join(words)
    
    metadata = draw(metadata_strategy(embedding_dim=embedding_dim))
    
    # Generate timestamp with timezone
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        timezones=st.just(timezone.utc)
    ))
    
    # Category is optional
    category = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=15, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        ))
    ))
    
    # Generate valid scores [0, 1]
    similarity_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    importance_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    recency_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    final_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Create a mock memory object
    class MockMemory:
        def __init__(self, memory_id, timestamp, content, category,
                     similarity_score, importance_score, recency_score,
                     final_score, metadata):
            self.memory_id = memory_id
            self.timestamp = timestamp
            self.content = content
            self.category = category
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
        category=category,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=recency_score,
        final_score=final_score,
        metadata=metadata
    )


@st.composite
def memory_list_strategy(draw, min_size=0, max_size=50, embedding_dim=None):
    """Generate lists of memories with unique IDs.
    
    Creates lists of mock memory objects with unique memory_ids.
    If embedding_dim is specified, all memories will have embeddings
    with the same dimension.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
        embedding_dim: Fixed embedding dimension for all memories (optional)
    
    Returns:
        List of mock memory objects
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    memories = []
    used_ids = set()
    
    for i in range(size):
        memory = draw(ranked_memory_strategy(embedding_dim=embedding_dim))
        
        # Ensure unique memory_id
        counter = 0
        while memory.memory_id in used_ids:
            memory.memory_id = f"{memory.memory_id}_{counter}"
            counter += 1
        
        used_ids.add(memory.memory_id)
        memories.append(memory)
    
    return memories


@st.composite
def injection_config_strategy(draw):
    """Generate valid InjectionConfig objects.
    
    Creates random but valid injection configurations for testing.
    """
    return InjectionConfig(
        max_token_budget=draw(st.integers(min_value=100, max_value=10000)),
        max_memory_count=draw(st.integers(min_value=1, max_value=100)),
        redundancy_similarity_threshold=draw(st.floats(
            min_value=0.0, max_value=1.0,
            allow_nan=False, allow_infinity=False
        )),
        enable_category_isolation=False,  # Disable for simplicity in basic tests
        allowed_categories=None,
        token_estimation_factor=draw(st.floats(
            min_value=0.5, max_value=3.0,
            allow_nan=False, allow_infinity=False
        ))
    )


# ============================================================================
# Property 7: Injection Determinism (Idempotence)
# ============================================================================

# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=0, max_size=20, embedding_dim=32),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@pytest.mark.property_test
def test_injection_determinism_idempotence(memories, config):
    """
    Property: For any list of ranked memories and injection configuration,
    calling inject() twice with the same inputs should produce InjectionResult
    objects with identical memory_id sequences, total_tokens, and all diagnostic counts.
    
    **Validates: Requirements 6.3, 6.5**
    
    This test verifies that:
    1. The same input produces the same output (determinism)
    2. Memory selection is consistent across runs
    3. Token counts are consistent across runs
    4. Diagnostic counts are consistent across runs
    5. The order of selected memories is consistent
    """
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection twice with the same inputs
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    
    # THE PROPERTY: Results should be identical
    
    # 1. Verify memory_id sequences are identical
    memory_ids_1 = [m.memory_id for m in result1.memories]
    memory_ids_2 = [m.memory_id for m in result2.memories]
    
    assert memory_ids_1 == memory_ids_2, (
        f"Injection determinism violated: memory_id sequences differ. "
        f"First run: {memory_ids_1}, "
        f"Second run: {memory_ids_2}"
    )
    
    # 2. Verify total_tokens is identical
    assert result1.total_tokens == result2.total_tokens, (
        f"Injection determinism violated: total_tokens differ. "
        f"First run: {result1.total_tokens}, "
        f"Second run: {result2.total_tokens}"
    )
    
    # 3. Verify all diagnostic counts are identical
    assert result1.input_count == result2.input_count, (
        f"Injection determinism violated: input_count differs. "
        f"First run: {result1.input_count}, "
        f"Second run: {result2.input_count}"
    )
    
    assert result1.filtered_by_category == result2.filtered_by_category, (
        f"Injection determinism violated: filtered_by_category differs. "
        f"First run: {result1.filtered_by_category}, "
        f"Second run: {result2.filtered_by_category}"
    )
    
    assert result1.filtered_by_redundancy == result2.filtered_by_redundancy, (
        f"Injection determinism violated: filtered_by_redundancy differs. "
        f"First run: {result1.filtered_by_redundancy}, "
        f"Second run: {result2.filtered_by_redundancy}"
    )
    
    assert result1.filtered_by_budget == result2.filtered_by_budget, (
        f"Injection determinism violated: filtered_by_budget differs. "
        f"First run: {result1.filtered_by_budget}, "
        f"Second run: {result2.filtered_by_budget}"
    )
    
    # 4. Verify the number of selected memories is identical
    assert len(result1.memories) == len(result2.memories), (
        f"Injection determinism violated: number of selected memories differs. "
        f"First run: {len(result1.memories)}, "
        f"Second run: {len(result2.memories)}"
    )


# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=1, max_size=15, embedding_dim=32),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@pytest.mark.property_test
def test_injection_determinism_with_redundancy_filtering(memories, config):
    """
    Property: Injection determinism holds even when redundancy filtering is active.
    
    **Validates: Requirements 6.3, 6.5**
    
    This test ensures that redundancy filtering produces consistent results
    across multiple runs with the same inputs.
    """
    # Ensure redundancy filtering is active by setting a reasonable threshold
    config.redundancy_similarity_threshold = 0.7
    
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection multiple times (3 runs to be thorough)
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    result3 = engine.inject(memories)
    
    # Extract memory_id sequences
    memory_ids_1 = [m.memory_id for m in result1.memories]
    memory_ids_2 = [m.memory_id for m in result2.memories]
    memory_ids_3 = [m.memory_id for m in result3.memories]
    
    # THE PROPERTY: All runs should produce identical sequences
    assert memory_ids_1 == memory_ids_2 == memory_ids_3, (
        f"Injection determinism violated with redundancy filtering. "
        f"Run 1: {memory_ids_1}, "
        f"Run 2: {memory_ids_2}, "
        f"Run 3: {memory_ids_3}"
    )
    
    # Verify total_tokens is consistent
    assert result1.total_tokens == result2.total_tokens == result3.total_tokens, (
        f"Total tokens differ across runs: "
        f"{result1.total_tokens}, {result2.total_tokens}, {result3.total_tokens}"
    )


# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=0, max_size=0),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_injection_determinism_with_empty_input(memories, config):
    """
    Property: Injection determinism holds for empty input (edge case).
    
    **Validates: Requirements 6.3, 6.5**
    
    This test verifies that the engine handles empty input deterministically.
    """
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection twice with empty input
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    
    # THE PROPERTY: Results should be identical (empty)
    assert len(result1.memories) == len(result2.memories) == 0, (
        "Empty input should produce empty output consistently"
    )
    
    assert result1.total_tokens == result2.total_tokens == 0, (
        "Empty input should produce zero tokens consistently"
    )
    
    assert result1.input_count == result2.input_count == 0, (
        "Empty input should have zero input_count consistently"
    )


# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=5, max_size=20, embedding_dim=32)
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
@pytest.mark.property_test
def test_injection_determinism_with_tight_budget(memories):
    """
    Property: Injection determinism holds even with tight token budgets
    that force early cutoff.
    
    **Validates: Requirements 6.3, 6.5**
    
    This test creates scenarios where the budget is tight, ensuring
    determinism is maintained even when budget constraints are active.
    """
    # Create a tight budget configuration
    config = InjectionConfig(
        max_token_budget=200,  # Tight budget
        max_memory_count=50,
        redundancy_similarity_threshold=0.8,
        enable_category_isolation=False,
        allowed_categories=None,
        token_estimation_factor=1.3
    )
    
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection twice
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    
    # THE PROPERTY: Results should be identical
    memory_ids_1 = [m.memory_id for m in result1.memories]
    memory_ids_2 = [m.memory_id for m in result2.memories]
    
    assert memory_ids_1 == memory_ids_2, (
        f"Injection determinism violated with tight budget. "
        f"First run: {memory_ids_1}, "
        f"Second run: {memory_ids_2}"
    )
    
    assert result1.total_tokens == result2.total_tokens, (
        f"Total tokens differ with tight budget: "
        f"{result1.total_tokens} vs {result2.total_tokens}"
    )


# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=5, max_size=20, embedding_dim=32)
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
@pytest.mark.property_test
def test_injection_determinism_with_memory_count_limit(memories):
    """
    Property: Injection determinism holds when memory count limit is active.
    
    **Validates: Requirements 6.3, 6.5**
    
    This test ensures determinism when the memory count limit is the
    primary constraint (rather than token budget).
    """
    # Create a configuration with tight memory count limit
    config = InjectionConfig(
        max_token_budget=100000,  # Large budget
        max_memory_count=5,  # Tight memory count limit
        redundancy_similarity_threshold=0.8,
        enable_category_isolation=False,
        allowed_categories=None,
        token_estimation_factor=1.3
    )
    
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection twice
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    
    # THE PROPERTY: Results should be identical
    memory_ids_1 = [m.memory_id for m in result1.memories]
    memory_ids_2 = [m.memory_id for m in result2.memories]
    
    assert memory_ids_1 == memory_ids_2, (
        f"Injection determinism violated with memory count limit. "
        f"First run: {memory_ids_1}, "
        f"Second run: {memory_ids_2}"
    )
    
    # Verify memory count limit is respected
    assert len(result1.memories) <= 5, (
        f"Memory count limit violated: {len(result1.memories)} > 5"
    )
    
    assert len(result2.memories) <= 5, (
        f"Memory count limit violated: {len(result2.memories)} > 5"
    )


# Feature: context-injection-engine, Property 7: Injection Determinism (Idempotence)
@given(
    memories=memory_list_strategy(min_size=1, max_size=20, embedding_dim=32),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@pytest.mark.property_test
def test_injection_determinism_content_preservation(memories, config):
    """
    Property: Injection determinism extends to memory content and metadata.
    Not only should the memory_id sequences match, but the actual content
    and metadata of each selected memory should be identical across runs.
    
    **Validates: Requirements 6.3, 6.5**
    
    This test verifies that determinism applies to all fields of the
    selected memories, not just their IDs.
    """
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection twice
    result1 = engine.inject(memories)
    result2 = engine.inject(memories)
    
    # THE PROPERTY: All fields should be identical for each memory
    assert len(result1.memories) == len(result2.memories), (
        "Number of selected memories differs across runs"
    )
    
    for i, (mem1, mem2) in enumerate(zip(result1.memories, result2.memories)):
        # Verify memory_id
        assert mem1.memory_id == mem2.memory_id, (
            f"Memory {i}: memory_id differs ({mem1.memory_id} vs {mem2.memory_id})"
        )
        
        # Verify content
        assert mem1.content == mem2.content, (
            f"Memory {i} ({mem1.memory_id}): content differs"
        )
        
        # Verify metadata
        assert mem1.metadata == mem2.metadata, (
            f"Memory {i} ({mem1.memory_id}): metadata differs"
        )
        
        # Verify similarity_score
        assert mem1.similarity_score == mem2.similarity_score, (
            f"Memory {i} ({mem1.memory_id}): similarity_score differs"
        )
        
        # Verify timestamp
        assert mem1.timestamp == mem2.timestamp, (
            f"Memory {i} ({mem1.memory_id}): timestamp differs"
        )
        
        # Verify category
        assert mem1.category == mem2.category, (
            f"Memory {i} ({mem1.memory_id}): category differs"
        )
