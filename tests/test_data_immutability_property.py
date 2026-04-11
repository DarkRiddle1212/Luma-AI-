"""
Property-Based Tests for Data Immutability Invariant

This module implements property-based tests using Hypothesis to verify
that the InjectionEngine preserves memory content, metadata, and similarity_score
exactly as they appear in the input RankedMemory objects.

Feature: context-injection-engine
Property 5: Data Immutability Invariant
Validates: Requirements 9.1, 10.1, 10.2, 10.4
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime, timezone
from typing import List, Any, Dict
import copy

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
    
    Creates realistic metadata with different data types to ensure
    immutability is preserved across all field types.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    metadata = {}
    
    # Add various field types to test immutability
    num_fields = draw(st.integers(min_value=1, max_value=10))
    
    for i in range(num_fields):
        field_name = f"field_{i}"
        
        # Choose a random field type (exclude embedding if not specified)
        field_types = ['string', 'int', 'float', 'bool', 'list', 'dict', 'token_count']
        field_type = draw(st.sampled_from(field_types))
        
        if field_type == 'string':
            metadata[field_name] = draw(st.text(min_size=0, max_size=5))
        elif field_type == 'int':
            metadata[field_name] = draw(st.integers(min_value=-1000, max_value=1000))
        elif field_type == 'float':
            metadata[field_name] = draw(st.floats(
                min_value=-1000.0, max_value=1000.0,
                allow_nan=False, allow_infinity=False
            ))
        elif field_type == 'bool':
            metadata[field_name] = draw(st.booleans())
        elif field_type == 'list':
            metadata[field_name] = draw(st.lists(
                st.integers(min_value=0, max_value=100),
                min_size=0, max_size=10
            ))
        elif field_type == 'dict':
            metadata[field_name] = draw(st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.integers(min_value=0, max_value=100),
                min_size=0, max_size=5
            ))
        elif field_type == 'token_count':
            metadata['token_count'] = draw(st.integers(min_value=1, max_value=500))
    
    # Add embedding with fixed dimension if specified
    if embedding_dim is not None:
        metadata['embedding'] = draw(st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=embedding_dim, max_size=embedding_dim
        ))
    
    return metadata


@st.composite
def ranked_memory_strategy(draw, embedding_dim=None):
    """Generate random RankedMemory objects with rich metadata.
    
    Creates valid RankedMemory instances with random but valid data
    for testing data immutability.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    """
    memory_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    # Generate content with various lengths
    num_words = draw(st.integers(min_value=1, max_value=100))
    words = [draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll')
    ))) for _ in range(num_words)]
    content = ' '.join(words)
    
    # Generate rich metadata to test immutability
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
        enable_category_isolation=False,  # Disable for simplicity
        allowed_categories=None,
        token_estimation_factor=draw(st.floats(
            min_value=0.5, max_value=3.0,
            allow_nan=False, allow_infinity=False
        ))
    )


# ============================================================================
# Property 5: Data Immutability Invariant
# ============================================================================

# Feature: context-injection-engine, Property 5: Data Immutability Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@pytest.mark.property_test
def test_data_immutability_content_metadata_scores(memories, config):
    """
    Property: For any memory in the InjectionResult, its content, metadata,
    and similarity_score should be identical to the corresponding fields
    in the input RankedMemory.
    
    **Validates: Requirements 9.1, 10.1, 10.2, 10.4**
    
    This test verifies that:
    1. Memory content is never modified during injection
    2. Metadata is preserved exactly (all fields match)
    3. Similarity scores are not modified
    4. The invariant holds for all selected memories
    5. The invariant holds across various configurations
    """
    # Create a deep copy of input memories to detect any modifications
    input_memories_copy = []
    for mem in memories:
        mem_copy = copy.deepcopy(mem)
        input_memories_copy.append({
            'memory_id': mem_copy.memory_id,
            'content': mem_copy.content,
            'metadata': mem_copy.metadata,
            'similarity_score': mem_copy.similarity_score,
            'timestamp': mem_copy.timestamp,
            'category': mem_copy.category
        })
    
    # Create injection engine and run injection
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # THE INVARIANT: For each memory in the output, verify immutability
    for injected_memory in result.memories:
        # Find the corresponding input memory
        input_memory = None
        for i, mem in enumerate(memories):
            if mem.memory_id == injected_memory.memory_id:
                input_memory = mem
                input_memory_copy = input_memories_copy[i]
                break
        
        assert input_memory is not None, (
            f"Output memory {injected_memory.memory_id} not found in input"
        )
        
        # Verify content immutability (Requirement 10.2)
        assert injected_memory.content == input_memory.content, (
            f"Content modified for memory {injected_memory.memory_id}: "
            f"expected '{input_memory.content}', got '{injected_memory.content}'"
        )
        
        # Verify metadata immutability (Requirements 10.1, 10.4)
        assert injected_memory.metadata == input_memory.metadata, (
            f"Metadata modified for memory {injected_memory.memory_id}: "
            f"expected {input_memory.metadata}, got {injected_memory.metadata}"
        )
        
        # Verify similarity_score immutability (Requirement 9.1)
        assert injected_memory.similarity_score == input_memory.similarity_score, (
            f"Similarity score modified for memory {injected_memory.memory_id}: "
            f"expected {input_memory.similarity_score}, got {injected_memory.similarity_score}"
        )
        
        # Verify timestamp immutability
        assert injected_memory.timestamp == input_memory.timestamp, (
            f"Timestamp modified for memory {injected_memory.memory_id}: "
            f"expected {input_memory.timestamp}, got {injected_memory.timestamp}"
        )
        
        # Verify category immutability
        assert injected_memory.category == input_memory.category, (
            f"Category modified for memory {injected_memory.memory_id}: "
            f"expected {input_memory.category}, got {injected_memory.category}"
        )


# Feature: context-injection-engine, Property 5: Data Immutability Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=5),
    config=injection_config_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_metadata_deep_immutability(memories, config):
    """
    Property: Metadata dictionaries should be deeply immutable - nested
    structures (lists, dicts) should not be modified.
    
    **Validates: Requirements 10.1, 10.4**
    
    This test verifies that:
    1. Nested metadata structures are preserved
    2. Lists in metadata are not modified
    3. Dictionaries in metadata are not modified
    4. Deep equality holds for all metadata fields
    """
    # Create deep copies of metadata for comparison
    metadata_copies = {}
    for mem in memories:
        metadata_copies[mem.memory_id] = copy.deepcopy(mem.metadata)
    
    # Create injection engine and run injection
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Verify deep immutability for each output memory
    for injected_memory in result.memories:
        original_metadata = metadata_copies[injected_memory.memory_id]
        output_metadata = injected_memory.metadata
        
        # Verify deep equality
        assert output_metadata == original_metadata, (
            f"Metadata deep equality violated for memory {injected_memory.memory_id}"
        )
        
        # Verify specific nested structures if present
        for key, value in original_metadata.items():
            assert key in output_metadata, (
                f"Metadata key '{key}' missing in output for memory {injected_memory.memory_id}"
            )
            
            if isinstance(value, list):
                assert output_metadata[key] == value, (
                    f"List metadata field '{key}' modified for memory {injected_memory.memory_id}"
                )
            elif isinstance(value, dict):
                assert output_metadata[key] == value, (
                    f"Dict metadata field '{key}' modified for memory {injected_memory.memory_id}"
                )


# Feature: context-injection-engine, Property 5: Data Immutability Invariant
@given(memories=memory_list_strategy(min_size=0, max_size=0))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_immutability_with_empty_input(memories):
    """
    Property: Data immutability invariant holds for empty input (edge case).
    
    **Validates: Requirements 9.1, 10.1, 10.2, 10.4**
    
    This test verifies that:
    1. Empty input produces empty output
    2. No memories are created or modified
    3. The invariant trivially holds (no memories to check)
    """
    config = InjectionConfig(
        max_token_budget=1000,
        max_memory_count=10,
        redundancy_similarity_threshold=0.8,
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Verify empty output
    assert len(result.memories) == 0, (
        f"Expected empty output for empty input, got {len(result.memories)} memories"
    )
    assert result.total_tokens == 0, (
        f"Expected zero tokens for empty input, got {result.total_tokens}"
    )


# Feature: context-injection-engine, Property 5: Data Immutability Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    max_token_budget=st.integers(min_value=10, max_value=100)
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@pytest.mark.property_test
def test_immutability_with_tight_budget(memories, max_token_budget):
    """
    Property: Data immutability invariant holds even when tight budget
    filters out most memories.
    
    **Validates: Requirements 9.1, 10.1, 10.2, 10.4**
    
    This test verifies that:
    1. Immutability holds regardless of how many memories are filtered
    2. Budget constraints don't affect data preservation
    3. The invariant holds for the subset of selected memories
    """
    config = InjectionConfig(
        max_token_budget=max_token_budget,
        max_memory_count=100,
        redundancy_similarity_threshold=0.0,  # No redundancy filtering
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    # Store original data
    original_data = {}
    for mem in memories:
        original_data[mem.memory_id] = {
            'content': mem.content,
            'metadata': copy.deepcopy(mem.metadata),
            'similarity_score': mem.similarity_score
        }
    
    # Run injection
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Verify immutability for selected memories
    for injected_memory in result.memories:
        original = original_data[injected_memory.memory_id]
        
        assert injected_memory.content == original['content'], (
            f"Content modified despite budget filtering for memory {injected_memory.memory_id}"
        )
        assert injected_memory.metadata == original['metadata'], (
            f"Metadata modified despite budget filtering for memory {injected_memory.memory_id}"
        )
        assert injected_memory.similarity_score == original['similarity_score'], (
            f"Similarity score modified despite budget filtering for memory {injected_memory.memory_id}"
        )


# Feature: context-injection-engine, Property 5: Data Immutability Invariant
@given(
    memories=memory_list_strategy(min_size=2, max_size=20, embedding_dim=16),
    threshold=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_immutability_with_redundancy_filtering(memories, threshold):
    """
    Property: Data immutability invariant holds even when redundancy
    filtering removes similar memories.
    
    **Validates: Requirements 9.1, 10.1, 10.2, 10.4**
    
    This test verifies that:
    1. Immutability holds regardless of redundancy filtering
    2. Similarity computation doesn't modify memory data
    3. The invariant holds for non-redundant memories
    """
    # Memories already have embeddings with fixed dimension from strategy
    
    config = InjectionConfig(
        max_token_budget=10000,
        max_memory_count=100,
        redundancy_similarity_threshold=threshold,
        enable_category_isolation=False,
        allowed_categories=None
    )
    
    # Store original data
    original_data = {}
    for mem in memories:
        original_data[mem.memory_id] = {
            'content': mem.content,
            'metadata': copy.deepcopy(mem.metadata),
            'similarity_score': mem.similarity_score
        }
    
    # Run injection
    engine = InjectionEngine(config)
    result = engine.inject(memories)
    
    # Verify immutability for selected memories
    for injected_memory in result.memories:
        original = original_data[injected_memory.memory_id]
        
        assert injected_memory.content == original['content'], (
            f"Content modified despite redundancy filtering for memory {injected_memory.memory_id}"
        )
        assert injected_memory.metadata == original['metadata'], (
            f"Metadata modified despite redundancy filtering for memory {injected_memory.memory_id}"
        )
        assert injected_memory.similarity_score == original['similarity_score'], (
            f"Similarity score modified despite redundancy filtering for memory {injected_memory.memory_id}"
        )
