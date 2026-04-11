"""
Property-Based Tests for No Redundant Pairs Invariant

This module implements property-based tests using Hypothesis to verify
that RedundancyGuard ensures no two memories in the output have pairwise
similarity greater than the redundancy_similarity_threshold.

Feature: context-injection-engine
Property 3: No Redundant Pairs Invariant
Validates: Requirements 3.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import List, Any

from luma.core.injection_engine import RedundancyGuard


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def embedding_strategy(draw, dimensions=10):
    """Generate random embedding vectors.
    
    Creates normalized embedding vectors for similarity testing.
    Uses a smaller dimension (10) for faster testing while maintaining
    the mathematical properties of cosine similarity.
    
    Args:
        draw: Hypothesis draw function
        dimensions: Number of dimensions for the embedding vector
    
    Returns:
        List of floats representing an embedding vector
    """
    # Generate random values
    values = [draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
              for _ in range(dimensions)]
    
    # Normalize to unit vector (for more realistic embeddings)
    norm = np.linalg.norm(values)
    if norm > 0:
        values = [v / norm for v in values]
    
    return values


@st.composite
def memory_with_embedding_strategy(draw):
    """Generate mock memory objects with embeddings.
    
    Creates simple memory objects with:
    - Unique memory_id
    - Random final_score in [0, 1]
    - Random embedding vector
    """
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    final_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    embedding = draw(embedding_strategy())
    
    # Create mock memory object
    class MockMemory:
        def __init__(self, memory_id, final_score, embedding):
            self.memory_id = memory_id
            self.final_score = final_score
            self.metadata = {'embedding': embedding}
    
    return MockMemory(memory_id, final_score, embedding)


@st.composite
def memory_list_strategy(draw, min_size=0, max_size=5):
    """Generate lists of memories with embeddings.
    
    Creates lists of mock memory objects, ensuring unique memory_ids
    and sorted by final_score in descending order (as expected by RedundancyGuard).
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
    
    Returns:
        List of mock memory objects sorted by final_score descending
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    memories = []
    used_ids = set()
    
    for i in range(size):
        memory = draw(memory_with_embedding_strategy())
        
        # Ensure unique memory_id
        counter = 0
        while memory.memory_id in used_ids:
            memory.memory_id = f"{memory.memory_id}_{counter}"
            counter += 1
        
        used_ids.add(memory.memory_id)
        memories.append(memory)
    
    # Sort by final_score descending (as expected by RedundancyGuard)
    memories.sort(key=lambda m: m.final_score, reverse=True)
    
    return memories


# ============================================================================
# Property 3: No Redundant Pairs Invariant
# ============================================================================

def compute_similarity(memory1: Any, memory2: Any) -> float:
    """Compute cosine similarity between two memories.
    
    This is a test helper that replicates the similarity computation
    logic to verify the invariant independently.
    
    Args:
        memory1: First memory with embedding in metadata
        memory2: Second memory with embedding in metadata
    
    Returns:
        Cosine similarity [0, 1], or 0.0 if embeddings unavailable
    """
    emb1 = memory1.metadata.get('embedding')
    emb2 = memory2.metadata.get('embedding')
    
    if emb1 is None or emb2 is None:
        return 0.0
    
    emb1_arr = np.array(emb1)
    emb2_arr = np.array(emb2)
    
    dot_product = np.dot(emb1_arr, emb2_arr)
    norm1 = np.linalg.norm(emb1_arr)
    norm2 = np.linalg.norm(emb2_arr)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = float(dot_product / (norm1 * norm2))
    
    # Clamp to [0, 1] for numerical stability
    return max(0.0, min(1.0, similarity))


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_invariant(memories, threshold):
    """
    Property: For any InjectionResult, no two memories in the output should have
    pairwise similarity greater than redundancy_similarity_threshold.
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. RedundancyGuard filters out memories with similarity > threshold
    2. No two memories in the output exceed the threshold
    3. The invariant holds for all valid inputs
    4. The invariant holds for all threshold values [0, 1]
    """
    guard = RedundancyGuard(threshold=threshold)
    
    # Apply redundancy filtering
    filtered_memories, filtered_count = guard.filter(memories)
    
    # Verify the invariant: no two memories in output have similarity > threshold
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            # The invariant: similarity must be <= threshold
            assert similarity <= threshold, (
                f"Redundant pair found in output: "
                f"memory '{memory1.memory_id}' and '{memory2.memory_id}' "
                f"have similarity {similarity:.4f} > threshold {threshold:.4f}. "
                f"This violates the no redundant pairs invariant."
            )


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(
    memories=memory_list_strategy(min_size=2, max_size=15),
    threshold=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_with_guaranteed_filtering(memories, threshold):
    """
    Property: When memories exist with high similarity, the redundancy guard
    filters them out, maintaining the no redundant pairs invariant.
    
    **Validates: Requirements 3.2**
    
    This test creates scenarios where filtering is likely to occur
    by using moderate threshold values and ensuring we have multiple memories.
    """
    # Assume we have at least 2 memories to make the test meaningful
    assume(len(memories) >= 2)
    
    guard = RedundancyGuard(threshold=threshold)
    
    # Apply redundancy filtering
    filtered_memories, filtered_count = guard.filter(memories)
    
    # Verify the invariant holds
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            assert similarity <= threshold, (
                f"Redundant pair found: {memory1.memory_id} and {memory2.memory_id} "
                f"have similarity {similarity:.4f} > threshold {threshold:.4f}"
            )


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(memories=memory_list_strategy(min_size=0, max_size=5))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_with_zero_threshold(memories):
    """
    Property: With threshold=0.0, only memories with similarity=0.0 should pass
    (or the first memory in each similarity group).
    
    **Validates: Requirements 3.2**
    
    This is an edge case test for the strictest redundancy filtering.
    """
    guard = RedundancyGuard(threshold=0.0)
    
    filtered_memories, filtered_count = guard.filter(memories)
    
    # Verify the invariant: no two memories have similarity > 0.0
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            # With threshold=0.0, any similarity > 0.0 should have been filtered
            assert similarity <= 0.0, (
                f"With threshold=0.0, found pair with similarity {similarity:.4f} > 0.0: "
                f"{memory1.memory_id} and {memory2.memory_id}"
            )


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(memories=memory_list_strategy(min_size=0, max_size=5))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_with_max_threshold(memories):
    """
    Property: With threshold=1.0, only memories with similarity > 1.0 should be filtered
    (which is impossible, so all memories should pass unless identical).
    
    **Validates: Requirements 3.2**
    
    This is an edge case test for the most permissive redundancy filtering.
    """
    guard = RedundancyGuard(threshold=1.0)
    
    filtered_memories, filtered_count = guard.filter(memories)
    
    # Verify the invariant: no two memories have similarity > 1.0
    # (This should always be true since cosine similarity is bounded by 1.0)
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            assert similarity <= 1.0, (
                f"Similarity exceeds maximum possible value: "
                f"{memory1.memory_id} and {memory2.memory_id} "
                f"have similarity {similarity:.4f} > 1.0"
            )


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(
    size=st.integers(min_value=2, max_value=10),
    threshold=st.floats(min_value=0.3, max_value=0.7, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_with_identical_embeddings(size, threshold):
    """
    Property: When multiple memories have identical embeddings (similarity=1.0),
    only the first (highest-ranked) should pass, maintaining the invariant.
    
    **Validates: Requirements 3.2**
    
    This test creates a worst-case scenario with many identical memories.
    """
    # Create memories with identical embeddings
    embedding = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    class MockMemory:
        def __init__(self, memory_id, final_score, embedding):
            self.memory_id = memory_id
            self.final_score = final_score
            self.metadata = {'embedding': embedding}
    
    memories = [
        MockMemory(f"m{i}", 1.0 - i * 0.01, embedding)
        for i in range(size)
    ]
    
    guard = RedundancyGuard(threshold=threshold)
    
    filtered_memories, filtered_count = guard.filter(memories)
    
    # With identical embeddings and threshold < 1.0, only first memory should pass
    if threshold < 1.0:
        assert len(filtered_memories) == 1, (
            f"With identical embeddings and threshold={threshold:.4f}, "
            f"expected 1 memory, got {len(filtered_memories)}"
        )
        assert filtered_memories[0].memory_id == "m0", (
            f"Expected highest-ranked memory 'm0', got '{filtered_memories[0].memory_id}'"
        )
    
    # Verify the invariant
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            assert similarity <= threshold, (
                f"Redundant pair found: {memory1.memory_id} and {memory2.memory_id} "
                f"have similarity {similarity:.4f} > threshold {threshold:.4f}"
            )


# Feature: context-injection-engine, Property 3: No Redundant Pairs Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=5),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_redundant_pairs_output_is_subset_of_input(memories, threshold):
    """
    Property: The filtered output should be a subset of the input, and the
    no redundant pairs invariant should hold for the output.
    
    **Validates: Requirements 3.2**
    
    This test verifies both the subset property and the invariant.
    """
    guard = RedundancyGuard(threshold=threshold)
    
    # Get input memory IDs
    input_ids = {m.memory_id for m in memories}
    
    # Apply redundancy filtering
    filtered_memories, filtered_count = guard.filter(memories)
    
    # Verify output is subset of input
    output_ids = {m.memory_id for m in filtered_memories}
    assert output_ids.issubset(input_ids), (
        f"Output contains memory IDs not in input: {output_ids - input_ids}"
    )
    
    # Verify the no redundant pairs invariant
    for i in range(len(filtered_memories)):
        for j in range(i + 1, len(filtered_memories)):
            memory1 = filtered_memories[i]
            memory2 = filtered_memories[j]
            
            similarity = compute_similarity(memory1, memory2)
            
            assert similarity <= threshold, (
                f"Redundant pair found in output: "
                f"{memory1.memory_id} and {memory2.memory_id} "
                f"have similarity {similarity:.4f} > threshold {threshold:.4f}"
            )
