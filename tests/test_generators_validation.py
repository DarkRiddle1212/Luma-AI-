"""
Validation Tests for Test Data Generators

This module validates that the test data generators produce valid data
that meets the requirements for the injection engine.
"""

import pytest
from hypothesis import given, settings, HealthCheck
from tests.test_data_generators import (
    ranked_memory_strategy,
    memory_list_strategy,
    sorted_memory_list_strategy,
    injection_config_strategy,
    metadata_strategy
)


@given(metadata=metadata_strategy())
@settings(max_examples=10)
def test_metadata_strategy_generates_valid_metadata(metadata):
    """Verify metadata_strategy generates valid metadata dictionaries."""
    assert isinstance(metadata, dict)
    assert 'token_count' in metadata
    assert isinstance(metadata['token_count'], int)
    assert 10 <= metadata['token_count'] <= 200


@given(metadata=metadata_strategy(embedding_dim=768))
@settings(max_examples=10, suppress_health_check=[HealthCheck.large_base_example])
def test_metadata_strategy_with_embeddings(metadata):
    """Verify metadata_strategy generates embeddings when requested."""
    assert isinstance(metadata, dict)
    assert 'embedding' in metadata
    assert isinstance(metadata['embedding'], list)
    assert len(metadata['embedding']) == 768
    for val in metadata['embedding']:
        assert -1.0 <= val <= 1.0


@given(memory=ranked_memory_strategy())
@settings(max_examples=10)
def test_ranked_memory_strategy_generates_valid_memory(memory):
    """Verify ranked_memory_strategy generates valid RankedMemory objects."""
    # Check required fields exist
    assert hasattr(memory, 'memory_id')
    assert hasattr(memory, 'content')
    assert hasattr(memory, 'timestamp')
    assert hasattr(memory, 'namespace')
    assert hasattr(memory, 'category')
    assert hasattr(memory, 'similarity_score')
    assert hasattr(memory, 'importance_score')
    assert hasattr(memory, 'recency_score')
    assert hasattr(memory, 'final_score')
    assert hasattr(memory, 'metadata')
    assert hasattr(memory, 'memory_entry')
    
    # Check field types
    assert isinstance(memory.memory_id, str)
    assert len(memory.memory_id) > 0
    assert isinstance(memory.content, str)
    assert len(memory.content) > 0
    assert isinstance(memory.metadata, dict)
    
    # Check score ranges [0, 1]
    assert 0.0 <= memory.similarity_score <= 1.0
    assert 0.0 <= memory.importance_score <= 1.0
    assert 0.0 <= memory.recency_score <= 1.0
    assert 0.0 <= memory.final_score <= 1.0


@given(memory=ranked_memory_strategy(embedding_dim=768))
@settings(max_examples=10, suppress_health_check=[HealthCheck.large_base_example])
def test_ranked_memory_strategy_with_embeddings(memory):
    """Verify ranked_memory_strategy includes embeddings when requested."""
    assert 'embedding' in memory.metadata
    assert isinstance(memory.metadata['embedding'], list)
    assert len(memory.metadata['embedding']) == 768


@given(memories=memory_list_strategy(min_size=5, max_size=5))
@settings(max_examples=10)
def test_memory_list_strategy_generates_unique_ids(memories):
    """Verify memory_list_strategy generates unique memory_ids."""
    assert isinstance(memories, list)
    assert 5 <= len(memories) <= 20
    
    # Check all memory_ids are unique
    memory_ids = [m.memory_id for m in memories]
    assert len(memory_ids) == len(set(memory_ids))


@given(memories=sorted_memory_list_strategy(min_size=5, max_size=5))
@settings(max_examples=10)
def test_sorted_memory_list_strategy_is_sorted(memories):
    """Verify sorted_memory_list_strategy returns sorted list."""
    assert isinstance(memories, list)
    assert 5 <= len(memories) <= 20
    
    # Check sorted by final_score descending
    for i in range(len(memories) - 1):
        assert memories[i].final_score >= memories[i + 1].final_score


@given(config=injection_config_strategy())
@settings(max_examples=10)
def test_injection_config_strategy_generates_valid_config(config):
    """Verify injection_config_strategy generates valid configs."""
    # Check required fields
    assert hasattr(config, 'max_token_budget')
    assert hasattr(config, 'max_memory_count')
    assert hasattr(config, 'redundancy_similarity_threshold')
    assert hasattr(config, 'enable_category_isolation')
    assert hasattr(config, 'allowed_categories')
    
    # Check value ranges
    assert 100 <= config.max_token_budget <= 10000
    assert 1 <= config.max_memory_count <= 100
    assert 0.0 <= config.redundancy_similarity_threshold <= 1.0
    
    # Check category isolation logic
    if config.enable_category_isolation:
        assert config.allowed_categories is not None
        assert isinstance(config.allowed_categories, list)
        assert len(config.allowed_categories) > 0
    else:
        assert config.allowed_categories is None
