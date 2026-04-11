"""
Shared Test Data Generators for Property-Based Tests

This module provides reusable Hypothesis strategies for generating test data
for the context injection engine and related components.

Feature: context-injection-engine
"""

from hypothesis import strategies as st
from datetime import datetime, timezone
from typing import Optional


# ============================================================================
# Metadata Strategy
# ============================================================================

@st.composite
def metadata_strategy(draw, embedding_dim: Optional[int] = None):
    """Generate random metadata dictionaries with various field types.
    
    Creates realistic metadata with different data types including optional
    embeddings for similarity testing.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension (if None, embeddings are not added)
    
    Returns:
        Dictionary with metadata fields
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
    
    # Add optional custom fields
    if draw(st.booleans()):
        metadata['source'] = draw(st.sampled_from(['user_input', 'system', 'api', 'import']))
    
    if draw(st.booleans()):
        metadata['priority'] = draw(st.sampled_from(['low', 'medium', 'high']))
    
    return metadata


# ============================================================================
# RankedMemory Strategy
# ============================================================================

@st.composite
def ranked_memory_strategy(draw, embedding_dim: Optional[int] = None):
    """Generate random RankedMemory objects.
    
    Creates valid RankedMemory instances with random but valid data
    for testing the injection engine. Includes embeddings in metadata
    for similarity testing when embedding_dim is specified.
    
    Args:
        draw: Hypothesis draw function
        embedding_dim: Fixed embedding dimension for embeddings in metadata
                      (if None, embeddings are not added)
    
    Returns:
        Mock RankedMemory object with all required fields
    """
    # Generate memory_id with valid characters
    memory_id = draw(st.text(
        min_size=1, 
        max_size=50, 
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )
    ))
    
    # Generate content with various lengths (5-50 words)
    num_words = draw(st.integers(min_value=5, max_value=50))
    words = [
        draw(st.text(
            min_size=1, 
            max_size=10, 
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))
        )) 
        for _ in range(num_words)
    ]
    content = ' '.join(words)
    
    # Generate metadata with optional embeddings
    metadata = draw(metadata_strategy(embedding_dim=embedding_dim))
    
    # Generate timestamp with timezone
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31),
        timezones=st.just(timezone.utc)
    ))
    
    # Generate optional namespace
    namespace = draw(st.one_of(
        st.none(),
        st.text(
            min_size=1, 
            max_size=30, 
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_-'
            )
        )
    ))
    
    # Generate optional category
    category = draw(st.one_of(
        st.none(),
        st.text(
            min_size=1, 
            max_size=30, 
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_-'
            )
        )
    ))
    
    # Generate valid scores in [0, 1] range
    similarity_score = draw(st.floats(
        min_value=0.0, 
        max_value=1.0, 
        allow_nan=False, 
        allow_infinity=False
    ))
    importance_score = draw(st.floats(
        min_value=0.0, 
        max_value=1.0, 
        allow_nan=False, 
        allow_infinity=False
    ))
    recency_score = draw(st.floats(
        min_value=0.0, 
        max_value=1.0, 
        allow_nan=False, 
        allow_infinity=False
    ))
    final_score = draw(st.floats(
        min_value=0.0, 
        max_value=1.0, 
        allow_nan=False, 
        allow_infinity=False
    ))
    
    # Create a mock RankedMemory object
    class MockRankedMemory:
        """Mock RankedMemory object for testing."""
        def __init__(self, memory_id, timestamp, content, namespace, category,
                     similarity_score, importance_score, recency_score,
                     final_score, metadata):
            self.memory_id = memory_id
            self.timestamp = timestamp
            self.content = content
            self.namespace = namespace
            self.category = category
            self.similarity_score = similarity_score
            self.importance_score = importance_score
            self.recency_score = recency_score
            self.final_score = final_score
            self.metadata = metadata
            self.memory_entry = None
    
    return MockRankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content=content,
        namespace=namespace,
        category=category,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=recency_score,
        final_score=final_score,
        metadata=metadata
    )


# ============================================================================
# Memory List Strategies
# ============================================================================

@st.composite
def memory_list_strategy(draw, min_size: int = 0, max_size: int = 50, 
                        embedding_dim: Optional[int] = None):
    """Generate lists of memories with unique memory_ids.
    
    Creates lists of mock memory objects with unique memory_ids for testing.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
        embedding_dim: Fixed embedding dimension for all memories (optional)
    
    Returns:
        List of mock RankedMemory objects with unique memory_ids
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
def sorted_memory_list_strategy(draw, min_size: int = 0, max_size: int = 50,
                                embedding_dim: Optional[int] = None):
    """Generate lists of memories sorted by final_score in descending order.
    
    Creates lists of mock memory objects with unique memory_ids, sorted
    by final_score (highest first) as required by the injection engine.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum list size
        max_size: Maximum list size
        embedding_dim: Fixed embedding dimension for all memories (optional)
    
    Returns:
        List of mock RankedMemory objects sorted by final_score descending
    """
    memories = draw(memory_list_strategy(
        min_size=min_size, 
        max_size=max_size, 
        embedding_dim=embedding_dim
    ))
    
    # Sort by final_score descending (highest first)
    memories.sort(key=lambda m: m.final_score, reverse=True)
    
    return memories


# ============================================================================
# InjectionConfig Strategy
# ============================================================================

@st.composite
def injection_config_strategy(draw):
    """Generate random valid InjectionConfig objects.
    
    Creates valid InjectionConfig instances that pass validation for testing
    the injection engine with various configurations.
    
    Returns:
        Valid InjectionConfig object
    """
    max_token_budget = draw(st.integers(min_value=100, max_value=10000))
    max_memory_count = draw(st.integers(min_value=1, max_value=100))
    redundancy_similarity_threshold = draw(st.floats(
        min_value=0.0, 
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    ))
    enable_category_isolation = draw(st.booleans())
    
    # Generate allowed_categories only if isolation is enabled
    if enable_category_isolation:
        allowed_categories = draw(st.lists(
            st.text(
                min_size=1, 
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd'),
                    whitelist_characters='_-'
                )
            ),
            min_size=1,
            max_size=10,
            unique=True
        ))
    else:
        allowed_categories = None
    
    # Create a mock InjectionConfig object
    class MockInjectionConfig:
        """Mock InjectionConfig object for testing."""
        def __init__(self, max_token_budget, max_memory_count,
                     redundancy_similarity_threshold, enable_category_isolation,
                     allowed_categories):
            self.max_token_budget = max_token_budget
            self.max_memory_count = max_memory_count
            self.redundancy_similarity_threshold = redundancy_similarity_threshold
            self.enable_category_isolation = enable_category_isolation
            self.allowed_categories = allowed_categories
            self.token_estimation_factor = 1.3  # Default value
    
    return MockInjectionConfig(
        max_token_budget=max_token_budget,
        max_memory_count=max_memory_count,
        redundancy_similarity_threshold=redundancy_similarity_threshold,
        enable_category_isolation=enable_category_isolation,
        allowed_categories=allowed_categories
    )
