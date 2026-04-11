"""
Property-Based Tests for JSON Serialization Round-Trip

This module implements property-based tests using Hypothesis to verify
that InjectionResult objects can be serialized to JSON and deserialized
back without loss of information.

Feature: context-injection-engine
Property 9: JSON Serialization Round-Trip
Validates: Requirements 5.5
"""

import json
import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from luma.core.injection_engine import InjectedMemory, InjectionResult


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def metadata_strategy(draw):
    """Generate random metadata dictionaries.
    
    Metadata can contain various types of values including nested structures.
    This strategy generates realistic metadata that should survive JSON
    serialization round-trip.
    """
    # Generate 0-5 metadata fields
    num_fields = draw(st.integers(min_value=0, max_value=5))
    metadata = {}
    
    for i in range(num_fields):
        key = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )))
        
        # Generate various value types that are JSON-serializable
        value = draw(st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
            st.lists(st.text(max_size=5), max_size=5),
            st.lists(st.integers(), max_size=5),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=5),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(st.text(max_size=5), st.integers(), st.booleans()),
                max_size=3
            )
        ))
        
        metadata[key] = value
    
    return metadata


@st.composite
def injected_memory_strategy(draw):
    """Generate random InjectedMemory objects.
    
    Creates valid InjectedMemory instances with random but valid data
    that should survive JSON serialization round-trip.
    """
    memory_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='_-'
    )))
    
    content = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')
    )))
    
    metadata = draw(metadata_strategy())
    
    similarity_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
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
    
    return InjectedMemory(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
        similarity_score=similarity_score,
        timestamp=timestamp,
        category=category
    )


@st.composite
def injection_result_strategy(draw):
    """Generate random InjectionResult objects.
    
    Creates valid InjectionResult instances with random but valid data
    including a list of InjectedMemory objects and diagnostic counts.
    """
    # Generate 0-10 memories
    num_memories = draw(st.integers(min_value=0, max_value=10))
    memories = [draw(injected_memory_strategy()) for _ in range(num_memories)]
    
    # Generate realistic token count (sum of memory tokens)
    total_tokens = draw(st.integers(min_value=0, max_value=10000))
    
    # Generate diagnostic counts
    input_count = draw(st.integers(min_value=num_memories, max_value=1000))
    filtered_by_category = draw(st.integers(min_value=0, max_value=input_count - num_memories))
    remaining_after_category = input_count - filtered_by_category
    filtered_by_redundancy = draw(st.integers(min_value=0, max_value=remaining_after_category - num_memories))
    remaining_after_redundancy = remaining_after_category - filtered_by_redundancy
    filtered_by_budget = remaining_after_redundancy - num_memories
    
    return InjectionResult(
        memories=memories,
        total_tokens=total_tokens,
        input_count=input_count,
        filtered_by_category=filtered_by_category,
        filtered_by_redundancy=filtered_by_redundancy,
        filtered_by_budget=filtered_by_budget
    )


# ============================================================================
# Property 9: JSON Serialization Round-Trip
# ============================================================================

# Feature: context-injection-engine, Property 9: JSON Serialization Round-Trip
@given(result=injection_result_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_injection_result_json_round_trip(result):
    """
    Property: For any InjectionResult, serializing to JSON and then deserializing
    should produce an object equivalent to the original (all fields match).
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. InjectionResult can be serialized to JSON without errors
    2. The JSON can be deserialized back to InjectionResult
    3. All fields in the deserialized object match the original
    4. No information is lost during the round-trip
    5. Datetime objects are properly handled (ISO format)
    6. Metadata is preserved exactly
    7. Optional fields (like category) are handled correctly
    """
    # Serialize to dict
    result_dict = result.to_dict()
    
    # Verify dict is JSON-serializable
    json_str = json.dumps(result_dict)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Deserialize from JSON
    parsed_dict = json.loads(json_str)
    restored = InjectionResult.from_dict(parsed_dict)
    
    # Verify top-level fields match
    assert restored.total_tokens == result.total_tokens, \
        f"total_tokens mismatch: {restored.total_tokens} != {result.total_tokens}"
    assert restored.input_count == result.input_count, \
        f"input_count mismatch: {restored.input_count} != {result.input_count}"
    assert restored.filtered_by_category == result.filtered_by_category, \
        f"filtered_by_category mismatch: {restored.filtered_by_category} != {result.filtered_by_category}"
    assert restored.filtered_by_redundancy == result.filtered_by_redundancy, \
        f"filtered_by_redundancy mismatch: {restored.filtered_by_redundancy} != {result.filtered_by_redundancy}"
    assert restored.filtered_by_budget == result.filtered_by_budget, \
        f"filtered_by_budget mismatch: {restored.filtered_by_budget} != {result.filtered_by_budget}"
    
    # Verify memories list length matches
    assert len(restored.memories) == len(result.memories), \
        f"memories count mismatch: {len(restored.memories)} != {len(result.memories)}"
    
    # Verify each memory matches
    for i, (restored_mem, original_mem) in enumerate(zip(restored.memories, result.memories)):
        assert restored_mem.memory_id == original_mem.memory_id, \
            f"Memory {i} memory_id mismatch: {restored_mem.memory_id} != {original_mem.memory_id}"
        assert restored_mem.content == original_mem.content, \
            f"Memory {i} content mismatch"
        assert restored_mem.similarity_score == original_mem.similarity_score, \
            f"Memory {i} similarity_score mismatch: {restored_mem.similarity_score} != {original_mem.similarity_score}"
        assert restored_mem.timestamp == original_mem.timestamp, \
            f"Memory {i} timestamp mismatch: {restored_mem.timestamp} != {original_mem.timestamp}"
        assert restored_mem.category == original_mem.category, \
            f"Memory {i} category mismatch: {restored_mem.category} != {original_mem.category}"
        assert restored_mem.metadata == original_mem.metadata, \
            f"Memory {i} metadata mismatch: {restored_mem.metadata} != {original_mem.metadata}"


# Feature: context-injection-engine, Property 9: JSON Serialization Round-Trip
@given(memory=injected_memory_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_injected_memory_json_round_trip(memory):
    """
    Property: For any InjectedMemory, serializing to JSON and then deserializing
    should preserve all field values.
    
    **Validates: Requirements 5.4, 5.5**
    
    This test verifies that:
    1. InjectedMemory can be serialized to dict without errors
    2. The dict is JSON-serializable
    3. Datetime is converted to ISO format string
    4. Metadata is preserved exactly
    5. Optional category field is handled correctly
    """
    # Serialize to dict
    memory_dict = memory.to_dict()
    
    # Verify dict structure
    assert 'memory_id' in memory_dict
    assert 'content' in memory_dict
    assert 'metadata' in memory_dict
    assert 'similarity_score' in memory_dict
    assert 'timestamp' in memory_dict
    assert 'category' in memory_dict
    
    # Verify dict is JSON-serializable
    json_str = json.dumps(memory_dict)
    assert isinstance(json_str, str)
    
    # Parse back from JSON
    parsed_dict = json.loads(json_str)
    
    # Verify all fields match
    assert parsed_dict['memory_id'] == memory.memory_id
    assert parsed_dict['content'] == memory.content
    assert parsed_dict['metadata'] == memory.metadata
    assert parsed_dict['similarity_score'] == memory.similarity_score
    assert parsed_dict['timestamp'] == memory.timestamp.isoformat()
    assert parsed_dict['category'] == memory.category


# Feature: context-injection-engine, Property 9: JSON Serialization Round-Trip
@given(result=injection_result_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_empty_metadata_preserved(result):
    """
    Property: Empty metadata dictionaries should be preserved through
    JSON serialization round-trip.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. Empty metadata {} is preserved (not converted to None)
    2. Round-trip maintains empty dict structure
    """
    # Serialize and deserialize
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)
    parsed_dict = json.loads(json_str)
    restored = InjectionResult.from_dict(parsed_dict)
    
    # Check each memory's metadata
    for i, (restored_mem, original_mem) in enumerate(zip(restored.memories, result.memories)):
        if original_mem.metadata == {}:
            assert restored_mem.metadata == {}, \
                f"Memory {i} empty metadata not preserved: got {restored_mem.metadata}"
        else:
            assert restored_mem.metadata == original_mem.metadata, \
                f"Memory {i} metadata mismatch"


# Feature: context-injection-engine, Property 9: JSON Serialization Round-Trip
@given(result=injection_result_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_none_category_preserved(result):
    """
    Property: None values for optional category field should be preserved
    through JSON serialization round-trip.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. None category is preserved (not converted to empty string)
    2. Round-trip maintains None vs string distinction
    """
    # Serialize and deserialize
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)
    parsed_dict = json.loads(json_str)
    restored = InjectionResult.from_dict(parsed_dict)
    
    # Check each memory's category
    for i, (restored_mem, original_mem) in enumerate(zip(restored.memories, result.memories)):
        if original_mem.category is None:
            assert restored_mem.category is None, \
                f"Memory {i} None category not preserved: got {restored_mem.category}"
        else:
            assert restored_mem.category == original_mem.category, \
                f"Memory {i} category mismatch: {restored_mem.category} != {original_mem.category}"


# Feature: context-injection-engine, Property 9: JSON Serialization Round-Trip
@given(result=injection_result_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_multiple_round_trips_stable(result):
    """
    Property: Multiple serialization/deserialization cycles should produce
    identical results (idempotence).
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. First round-trip produces equivalent object
    2. Second round-trip produces same result as first
    3. Serialization is stable and deterministic
    """
    # First round-trip
    json_str_1 = json.dumps(result.to_dict())
    restored_1 = InjectionResult.from_dict(json.loads(json_str_1))
    
    # Second round-trip
    json_str_2 = json.dumps(restored_1.to_dict())
    restored_2 = InjectionResult.from_dict(json.loads(json_str_2))
    
    # Verify both restored objects are equivalent
    assert restored_1.total_tokens == restored_2.total_tokens
    assert restored_1.input_count == restored_2.input_count
    assert restored_1.filtered_by_category == restored_2.filtered_by_category
    assert restored_1.filtered_by_redundancy == restored_2.filtered_by_redundancy
    assert restored_1.filtered_by_budget == restored_2.filtered_by_budget
    assert len(restored_1.memories) == len(restored_2.memories)
    
    for mem1, mem2 in zip(restored_1.memories, restored_2.memories):
        assert mem1.memory_id == mem2.memory_id
        assert mem1.content == mem2.content
        assert mem1.similarity_score == mem2.similarity_score
        assert mem1.timestamp == mem2.timestamp
        assert mem1.category == mem2.category
        assert mem1.metadata == mem2.metadata
