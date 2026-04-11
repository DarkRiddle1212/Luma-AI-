"""
Property-Based Tests for Retrieval Result Completeness

This module implements property-based tests using Hypothesis to verify
that retrieval results include complete metadata about query execution.

Feature: intent-based-memory-retrieval-enhancements
Property 10: Retrieval Result Completeness
Validates: Requirements 10.1, 10.2, 10.3, 10.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from unittest.mock import Mock
from typing import List, Dict, Any

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import QueryParameters


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def query_parameters(draw):
    """Generate valid QueryParameters for testing."""
    params = {}
    
    # Optional query string
    if draw(st.booleans()):
        params["query"] = draw(st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=1,
            max_size=100
        ))
    
    # Optional category
    if draw(st.booleans()):
        params["category"] = draw(st.sampled_from([
            "general", "work", "education", "personal", "project"
        ]))
    
    # Optional tags
    if draw(st.booleans()):
        params["tags"] = draw(st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
                min_size=1,
                max_size=20
            ),
            min_size=1,
            max_size=5
        ))
    
    # Optional timestamp range
    if draw(st.booleans()):
        start = draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        ))
        # Ensure end >= start
        end = draw(st.datetimes(
            min_value=start,
            max_value=datetime(2026, 12, 31)
        ))
        params["start_time"] = start
        params["end_time"] = end
    
    # Always include limit
    params["limit"] = draw(st.integers(min_value=1, max_value=100))
    
    return params


@st.composite
def mock_memory_entries(draw):
    """Generate a list of mock memory entries."""
    num_entries = draw(st.integers(min_value=0, max_value=20))
    entries = []
    
    for i in range(num_entries):
        mock_entry = Mock()
        mock_entry.id = f"mem_{i}_{draw(st.integers(min_value=1000, max_value=9999))}"
        mock_entry.action = draw(st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=5,
            max_size=100
        ))
        mock_entry.tags = draw(st.lists(
            st.text(min_size=1, max_size=5),
            min_size=0,
            max_size=5
        ))
        mock_entry.context = {
            "category": draw(st.sampled_from([
                "general", "work", "education", "personal"
            ]))
        }
        mock_entry.created_at = draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        ))
        mock_entry.timestamp = mock_entry.created_at
        entries.append(mock_entry)
    
    return entries


# ============================================================================
# Property 10: Retrieval Result Completeness
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_includes_total_count(params, mock_entries):
    """
    Property: For any successful memory retrieval, the RetrievalResult must include
    total_count matching the number of memories returned.
    
    **Validates: Requirement 10.1 (total_count accuracy)**
    
    This test verifies that:
    1. RetrievalResult contains a "total_count" field
    2. total_count is an integer
    3. total_count matches the actual number of memories in the result
    4. total_count is accurate for both empty and non-empty results
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify total_count field exists
    assert "total_count" in result, \
        "RetrievalResult must include 'total_count' field"
    
    # Verify total_count is an integer
    assert isinstance(result["total_count"], int), \
        f"total_count must be an integer, got {type(result['total_count'])}"
    
    # Verify total_count matches actual number of memories
    actual_count = len(result["memories"])
    assert result["total_count"] == actual_count, \
        f"total_count ({result['total_count']}) must match actual memories count ({actual_count})"
    
    # Verify total_count is non-negative
    assert result["total_count"] >= 0, \
        f"total_count must be non-negative, got {result['total_count']}"


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_includes_execution_time(params, mock_entries):
    """
    Property: For any successful memory retrieval, the RetrievalResult must include
    query execution time in milliseconds.
    
    **Validates: Requirement 10.2 (execution time tracking)**
    
    This test verifies that:
    1. query_metadata contains "execution_time_ms" field
    2. execution_time_ms is a numeric value (int or float)
    3. execution_time_ms is non-negative
    4. execution_time_ms is reasonable (< 60 seconds for test scenarios)
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify query_metadata exists
    assert "query_metadata" in result, \
        "RetrievalResult must include 'query_metadata' field"
    
    metadata = result["query_metadata"]
    
    # Verify execution_time_ms field exists
    assert "execution_time_ms" in metadata, \
        "query_metadata must include 'execution_time_ms' field"
    
    exec_time = metadata["execution_time_ms"]
    
    # Verify execution_time_ms is numeric
    assert isinstance(exec_time, (int, float)), \
        f"execution_time_ms must be numeric, got {type(exec_time)}"
    
    # Verify execution_time_ms is non-negative
    assert exec_time >= 0, \
        f"execution_time_ms must be non-negative, got {exec_time}"
    
    # Verify execution_time_ms is reasonable (< 60 seconds)
    assert exec_time < 60000, \
        f"execution_time_ms should be < 60000ms for test scenarios, got {exec_time}"


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_includes_filters_applied(params, mock_entries):
    """
    Property: For any successful memory retrieval, the RetrievalResult must include
    the filters that were applied to the query.
    
    **Validates: Requirement 10.3 (filters tracking)**
    
    This test verifies that:
    1. query_metadata contains "filters_applied" field
    2. filters_applied is a dictionary
    3. filters_applied includes all non-None filter parameters (except limit)
    4. filters_applied values match the input parameters
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify query_metadata exists
    assert "query_metadata" in result, \
        "RetrievalResult must include 'query_metadata' field"
    
    metadata = result["query_metadata"]
    
    # Verify filters_applied field exists
    assert "filters_applied" in metadata, \
        "query_metadata must include 'filters_applied' field"
    
    filters_applied = metadata["filters_applied"]
    
    # Verify filters_applied is a dictionary
    assert isinstance(filters_applied, dict), \
        f"filters_applied must be a dictionary, got {type(filters_applied)}"
    
    # Verify all non-None parameters (except limit) are in filters_applied
    for key, value in params.items():
        if key == "limit":
            # limit should NOT be in filters_applied
            assert key not in filters_applied, \
                "limit should not be in filters_applied"
        elif value is not None:
            # All other non-None parameters should be in filters_applied
            assert key in filters_applied, \
                f"Parameter '{key}' with value should be in filters_applied"
            assert filters_applied[key] == value, \
                f"filters_applied['{key}'] should match input parameter"


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_includes_limit_and_has_more(params, mock_entries):
    """
    Property: For any successful memory retrieval, the RetrievalResult must include
    the limit that was applied and a has_more pagination flag.
    
    **Validates: Requirement 10.4 (pagination support)**
    
    This test verifies that:
    1. query_metadata contains "limit" field
    2. limit matches the input parameter
    3. query_metadata contains "has_more" field
    4. has_more is a boolean value
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify query_metadata exists
    assert "query_metadata" in result, \
        "RetrievalResult must include 'query_metadata' field"
    
    metadata = result["query_metadata"]
    
    # Verify limit field exists
    assert "limit" in metadata, \
        "query_metadata must include 'limit' field"
    
    # Verify limit matches input parameter
    expected_limit = params.get("limit", 10)
    assert metadata["limit"] == expected_limit, \
        f"query_metadata['limit'] ({metadata['limit']}) must match input limit ({expected_limit})"
    
    # Verify has_more field exists
    assert "has_more" in metadata, \
        "query_metadata must include 'has_more' field"
    
    # Verify has_more is a boolean
    assert isinstance(metadata["has_more"], bool), \
        f"has_more must be a boolean, got {type(metadata['has_more'])}"


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_structure_is_complete(params, mock_entries):
    """
    Property: For any successful memory retrieval, the RetrievalResult must have
    all required top-level fields with correct types.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4 (overall completeness)**
    
    This test verifies that:
    1. RetrievalResult has all three required top-level fields
    2. "memories" is a list
    3. "total_count" is an integer
    4. "query_metadata" is a dictionary
    5. All metadata subfields are present
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), \
        f"RetrievalResult must be a dictionary, got {type(result)}"
    
    # Verify all required top-level fields exist
    required_fields = ["memories", "total_count", "query_metadata"]
    for field in required_fields:
        assert field in result, \
            f"RetrievalResult must include '{field}' field"
    
    # Verify field types
    assert isinstance(result["memories"], list), \
        f"memories must be a list, got {type(result['memories'])}"
    
    assert isinstance(result["total_count"], int), \
        f"total_count must be an integer, got {type(result['total_count'])}"
    
    assert isinstance(result["query_metadata"], dict), \
        f"query_metadata must be a dictionary, got {type(result['query_metadata'])}"
    
    # Verify all required metadata subfields exist
    metadata = result["query_metadata"]
    required_metadata_fields = ["execution_time_ms", "filters_applied", "limit", "has_more"]
    for field in required_metadata_fields:
        assert field in metadata, \
            f"query_metadata must include '{field}' field"


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    query=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    limit=st.integers(min_value=1, max_value=100),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_completeness_with_legacy_api(query, limit, mock_entries):
    """
    Property: For any retrieval using the legacy API (query string), the RetrievalResult
    must still include complete metadata.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4 (backward compatibility)**
    
    This test verifies that:
    1. Legacy API calls produce complete RetrievalResult
    2. All metadata fields are present even with simple query string
    3. Metadata completeness is independent of API style used
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval using legacy API
    result = adapter.retrieve(query=query, limit=limit)
    
    # Verify complete structure
    assert "memories" in result
    assert "total_count" in result
    assert "query_metadata" in result
    
    # Verify metadata completeness
    metadata = result["query_metadata"]
    assert "execution_time_ms" in metadata
    assert "filters_applied" in metadata
    assert "limit" in metadata
    assert "has_more" in metadata
    
    # Verify limit matches
    assert metadata["limit"] == limit
    
    # Verify total_count matches memories
    assert result["total_count"] == len(result["memories"])


# Feature: intent-based-memory-retrieval-enhancements, Property 10: Retrieval Result Completeness
@given(
    params=query_parameters(),
    mock_entries=mock_memory_entries()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_result_memory_entries_are_complete(params, mock_entries):
    """
    Property: For any successful memory retrieval, each MemoryEntry in the result
    must have all required fields.
    
    **Validates: Requirement 10.1 (result completeness)**
    
    This test verifies that:
    1. Each memory entry has all required fields
    2. Field types are correct
    3. No memory entry is missing required information
    """
    # Setup mock memory manager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Execute retrieval
    result = adapter.retrieve(params=params)
    
    # Verify each memory entry has all required fields
    required_memory_fields = ["id", "content", "metadata", "timestamp", "category", "tags"]
    
    for i, memory in enumerate(result["memories"]):
        # Verify memory is a dictionary
        assert isinstance(memory, dict), \
            f"Memory entry {i} must be a dictionary, got {type(memory)}"
        
        # Verify all required fields exist
        for field in required_memory_fields:
            assert field in memory, \
                f"Memory entry {i} must include '{field}' field"
        
        # Verify field types
        assert isinstance(memory["id"], str), \
            f"Memory {i} id must be a string"
        assert isinstance(memory["content"], str), \
            f"Memory {i} content must be a string"
        assert isinstance(memory["metadata"], dict), \
            f"Memory {i} metadata must be a dictionary"
        assert isinstance(memory["timestamp"], str), \
            f"Memory {i} timestamp must be a string"
        assert isinstance(memory["category"], str), \
            f"Memory {i} category must be a string"
        assert isinstance(memory["tags"], list), \
            f"Memory {i} tags must be a list"
