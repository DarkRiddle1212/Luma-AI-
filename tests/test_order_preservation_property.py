"""
Property-Based Tests for Order Preservation Property

This module implements property-based tests using Hypothesis to verify
that the InjectionEngine preserves the rank order of memories from the
input list in the output result.

Feature: context-injection-engine
Property 6: Order Preservation Property
Validates: Requirements 1.3
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone
from typing import List, Any, Dict

from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig,
    InjectedMemory
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def metadata_strategy(draw, embedding_dim=None):
    """Generate random metadata dictionaries with various field types.
    
    Creates realistic metadata with different data types.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    metadata = {}
    
    # Add token_count for deterministic token estimation
    metadata['token_count'] = draw(st.integers(min_value=10, max_value=200))
    
    # Add embedding with fixed dimension if specified
    if embedding_dim is not None:
        metadata['embedding'] = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=embedding_dim, max_size=embedding_dim
        ))
    
    return metadata


@st.composite
def ranked_memory_strategy(draw, embedding_dim=None):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing order preservation.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    memory_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    # Generate content with various lengths
    num_words = draw(st.integers(min_value=5, max_value=50))
    words = [draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll')
    ))) for _ in range(num_words)]
    content = ' '.join(words)
    
    # Generate metadata
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
def sorted_memory_list_strategy(draw, min_size=0, max_size=50, embedding_dim=None):
    """Generate lists of memories sorted by final_score in descending order.
    
    Creates lists of mock memory objects with unique memory_ids, sorted
    by final_score (highest first) as required by the injection engine.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
        embedding_dim: Fixed embedding dimension for all memories (optional)
    
    Returns:
        List of mock memory objects sorted by final_score descending
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
    
    # Sort by final_score descending (highest score first)
    memories.sort(key=lambda m: m.final_score, reverse=True)
    
    return memories


@st.composite
def injection_config_strategy(draw):
    """Generate valid InjectionConfig objects.
    
    Creates random but valid injection configurations for testing.
    """
    return InjectionConfig(
        max_token_budget=draw(st.integers(min_value=500, max_value=10000)),
        max_memory_count=draw(st.integers(min_value=5, max_value=100)),
        redundancy_similarity_threshold=draw(st.floats(
            min_value=0.0, max_value=1.0,
            allow_nan=False, allow_infinity=False
        )),
        enable_category_isolation=False,  # Disable for simplicity
        allowed_categories=None,
        token_estimation_factor=draw(st.floats(
            min_value=0.5, max_value=3.0,
            allow_nan=False, allow_infinity=False
        ))
    )


# ============================================================================
# Property 6: Order Preservation Property
# ============================================================================

# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(
    memories=sorted_memory_list_strategy(min_size=0, max_size=50, embedding_dim=16),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_basic(memories, config):
    """
    Property: For any two memories M1 and M2 in the InjectionResult,
    if M1 appears before M2 in the output, then M1 should have appeared
    before M2 in the input ranked_memories list (preserving rank order).
    
    **Validates: Requirements 1.3**
    
    This test verifies that:
    1. The output memory_id sequence is a subsequence of the input memory_id sequence
    2. The relative order of memories is preserved
    3. No memory appears in the output before a higher-ranked memory from the input
    4. The property holds across various configurations
    """
    # Create injection engine
    engine = InjectionEngine(config)
    
    # Run injection
    result = engine.inject(memories)
    
    # Extract memory_id sequences
    input_ids = [m.memory_id for m in memories]
    output_ids = [m.memory_id for m in result.memories]
    
    # Verify that output is a subsequence of input (order preserved)
    # For each pair of memories in the output, check that their relative
    # order matches the input
    for i in range(len(output_ids)):
        for j in range(i + 1, len(output_ids)):
            output_id_i = output_ids[i]
            output_id_j = output_ids[j]
            
            # Find positions in input
            input_pos_i = input_ids.index(output_id_i)
            input_pos_j = input_ids.index(output_id_j)
            
            # Assert that the order is preserved
            assert input_pos_i < input_pos_j, (
                f"Order not preserved: {output_id_i} appears before {output_id_j} "
                f"in output (positions {i}, {j}), but in input they are at "
                f"positions {input_pos_i}, {input_pos_j}"
            )


# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(
    memories=sorted_memory_list_strategy(min_size=10, max_size=30, embedding_dim=16)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_tight_budget(memories):
    """
    Property: Order preservation holds even with tight budgets that
    filter out most memories.
    
    **Validates: Requirements 1.3**
    
    This test uses a tight token budget to ensure that order preservation
    works correctly when only a few memories are selected.
    """
    # Use a tight budget that will only allow a few memories
    config = InjectionConfig(
        max_token_budget=500,  # Tight budget
        max_memory_count=5,    # Small count limit
        redundancy_similarity_threshold=0.5,
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Extract memory_id sequences
    input_ids = [m.memory_id for m in memories]
    output_ids = [m.memory_id for m in result.memories]
    
    # Verify order preservation
    for i in range(len(output_ids)):
        for j in range(i + 1, len(output_ids)):
            output_id_i = output_ids[i]
            output_id_j = output_ids[j]
            
            input_pos_i = input_ids.index(output_id_i)
            input_pos_j = input_ids.index(output_id_j)
            
            assert input_pos_i < input_pos_j, (
                f"Order not preserved with tight budget: {output_id_i} before {output_id_j} "
                f"in output, but at positions {input_pos_i}, {input_pos_j} in input"
            )


# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(
    memories=sorted_memory_list_strategy(min_size=10, max_size=30, embedding_dim=16)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_redundancy_filtering(memories):
    """
    Property: Order preservation holds even when redundancy filtering
    removes similar memories.
    
    **Validates: Requirements 1.3**
    
    This test uses a high redundancy threshold to ensure that order
    preservation works correctly when redundancy filtering is active.
    """
    # Use a high redundancy threshold to trigger filtering
    config = InjectionConfig(
        max_token_budget=10000,  # Large budget
        max_memory_count=50,     # Large count limit
        redundancy_similarity_threshold=0.3,  # High threshold (more filtering)
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Extract memory_id sequences
    input_ids = [m.memory_id for m in memories]
    output_ids = [m.memory_id for m in result.memories]
    
    # Verify order preservation
    for i in range(len(output_ids)):
        for j in range(i + 1, len(output_ids)):
            output_id_i = output_ids[i]
            output_id_j = output_ids[j]
            
            input_pos_i = input_ids.index(output_id_i)
            input_pos_j = input_ids.index(output_id_j)
            
            assert input_pos_i < input_pos_j, (
                f"Order not preserved with redundancy filtering: {output_id_i} before {output_id_j} "
                f"in output, but at positions {input_pos_i}, {input_pos_j} in input"
            )


# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(memories=sorted_memory_list_strategy(min_size=0, max_size=0))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_empty_input(memories):
    """
    Property: Order preservation holds for empty input (edge case).
    
    **Validates: Requirements 1.3**
    
    This test verifies that the engine handles empty input correctly
    and returns an empty result (trivially preserving order).
    """
    config = InjectionConfig(
        max_token_budget=1000,
        max_memory_count=10,
        redundancy_similarity_threshold=0.5,
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Empty input should produce empty output
    assert len(result.memories) == 0, "Empty input should produce empty output"
    assert result.total_tokens == 0, "Empty input should have zero tokens"


# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(
    memories=sorted_memory_list_strategy(min_size=1, max_size=50, embedding_dim=16)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_single_memory(memories):
    """
    Property: Order preservation holds when only one memory is selected.
    
    **Validates: Requirements 1.3**
    
    This test uses a very tight budget to ensure only one memory is selected,
    verifying that order preservation is trivially satisfied.
    """
    # Use a very tight budget to select only one memory
    config = InjectionConfig(
        max_token_budget=50,   # Very tight budget
        max_memory_count=1,    # Only one memory
        redundancy_similarity_threshold=0.5,
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Should have at most one memory
    assert len(result.memories) <= 1, "Should have at most one memory with tight constraints"
    
    # If one memory is selected, it should be from the input
    if len(result.memories) == 1:
        output_id = result.memories[0].memory_id
        input_ids = [m.memory_id for m in memories]
        assert output_id in input_ids, "Selected memory should be from input"


# Feature: context-injection-engine, Property 6: Order Preservation Property
@given(
    memories=sorted_memory_list_strategy(min_size=5, max_size=30, embedding_dim=16),
    max_memory_count=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_memory_count_limit(memories, max_memory_count):
    """
    Property: Order preservation holds when memory count limit is reached.
    
    **Validates: Requirements 1.3**
    
    This test verifies that order preservation works correctly when the
    memory count limit is the primary constraint.
    """
    config = InjectionConfig(
        max_token_budget=100000,  # Very large budget (not the limiting factor)
        max_memory_count=max_memory_count,
        redundancy_similarity_threshold=0.9,  # Low threshold (minimal filtering)
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Extract memory_id sequences
    input_ids = [m.memory_id for m in memories]
    output_ids = [m.memory_id for m in result.memories]
    
    # Verify order preservation
    for i in range(len(output_ids)):
        for j in range(i + 1, len(output_ids)):
            output_id_i = output_ids[i]
            output_id_j = output_ids[j]
            
            input_pos_i = input_ids.index(output_id_i)
            input_pos_j = input_ids.index(output_id_j)
            
            assert input_pos_i < input_pos_j, (
                f"Order not preserved with memory count limit: {output_id_i} before {output_id_j} "
                f"in output, but at positions {input_pos_i}, {input_pos_j} in input"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'property_test'])
