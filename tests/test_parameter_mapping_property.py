"""
Property-Based Test for Parameter Mapping Correctness

This module implements Property 8: Parameter Mapping Correctness using Hypothesis
to verify that QueryParameters are correctly mapped to MemoryManager.query_memories()
parameters.

Feature: intent-based-memory-retrieval-enhancements
Property 8: Parameter Mapping Correctness
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import QueryParameters


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def query_parameters(draw):
    """
    Generate random QueryParameters for testing parameter mapping.
    
    Generates valid combinations of query parameters including:
    - query string (optional)
    - category (optional)
    - start_time and end_time (optional, valid range)
    - tags (optional list of strings)
    - limit (positive integer)
    """
    params: Dict[str, Any] = {}
    
    # Optional query string
    if draw(st.booleans()):
        params["query"] = draw(st.text(min_size=1, max_size=100))
    
    # Optional category
    if draw(st.booleans()):
        params["category"] = draw(st.sampled_from([
            "general", "education", "work", "personal", "system"
        ]))
    
    # Optional timestamp range (ensure start <= end)
    if draw(st.booleans()):
        base_time = datetime.now()
        start_offset = draw(st.integers(min_value=0, max_value=365))
        end_offset = draw(st.integers(min_value=0, max_value=365))
        
        # Ensure start_time <= end_time by using min/max
        time1 = base_time - timedelta(days=start_offset)
        time2 = base_time - timedelta(days=end_offset)
        
        params["start_time"] = min(time1, time2)
        params["end_time"] = max(time1, time2)
    
    # Optional tags
    if draw(st.booleans()):
        params["tags"] = draw(st.lists(
            st.text(min_size=1, max_size=5),
            min_size=1,
            max_size=5
        ))
    
    # Always include limit
    params["limit"] = draw(st.integers(min_value=1, max_value=100))
    
    return params


# ============================================================================
# Property 8: Parameter Mapping Correctness
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 8: Parameter Mapping Correctness
@given(params=query_parameters())
@settings(max_examples=10)
@pytest.mark.property_test
def test_parameter_mapping_correctness(params: Dict[str, Any]):
    """
    Property 8: Parameter Mapping Correctness
    
    For any QueryParameters provided to SQLiteMemoryAdapter.retrieve(),
    the parameters must be correctly mapped to MemoryManager.query_memories()
    parameters:
    - query → action_type
    - start_time → start_time
    - end_time → end_time
    - tags → tags
    - limit → limit
    
    **Validates: Requirements 1.6**
    
    This test verifies that:
    1. All QueryParameters fields are correctly mapped to MemoryManager parameters
    2. The mapping preserves parameter values exactly
    3. Optional parameters are handled correctly (None when not provided)
    4. The adapter calls query_memories with the correct parameter names
    5. No parameters are lost or incorrectly transformed during mapping
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Configure mock to return empty list (we only care about the call)
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() with params
    result = adapter.retrieve(params=params)
    
    # Verify query_memories was called
    assert mock_memory_manager.query_memories.called, \
        "retrieve() must call memory_manager.query_memories()"
    
    # Get the actual call arguments
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Verify parameter mapping: query → action_type (with normalization)
    # Phase 5A normalizes whitespace-only queries to None
    if "query" in params:
        expected_query = params["query"]
        if isinstance(expected_query, str) and not expected_query.strip():
            expected_query = None
        
        assert "action_type" in call_kwargs, \
            "query parameter must be mapped to action_type"
        assert call_kwargs["action_type"] == expected_query, \
            f"query '{params['query']}' must map to action_type (normalized to {repr(expected_query)}), got '{call_kwargs.get('action_type')}'"
    else:
        # When no query provided, action_type should be None or not present
        action_type = call_kwargs.get("action_type")
        assert action_type is None, \
            f"When no query provided, action_type should be None, got '{action_type}'"
    
    # Verify parameter mapping: start_time → start_time
    if "start_time" in params:
        assert "start_time" in call_kwargs, \
            "start_time parameter must be passed through"
        assert call_kwargs["start_time"] == params["start_time"], \
            f"start_time must be preserved exactly, expected {params['start_time']}, got {call_kwargs['start_time']}"
    else:
        # When no start_time provided, it should be None or not present
        start_time = call_kwargs.get("start_time")
        assert start_time is None, \
            f"When no start_time provided, it should be None, got '{start_time}'"
    
    # Verify parameter mapping: end_time → end_time
    if "end_time" in params:
        assert "end_time" in call_kwargs, \
            "end_time parameter must be passed through"
        assert call_kwargs["end_time"] == params["end_time"], \
            f"end_time must be preserved exactly, expected {params['end_time']}, got {call_kwargs['end_time']}"
    else:
        # When no end_time provided, it should be None or not present
        end_time = call_kwargs.get("end_time")
        assert end_time is None, \
            f"When no end_time provided, it should be None, got '{end_time}'"
    
    # Verify parameter mapping: tags → tags
    if "tags" in params:
        assert "tags" in call_kwargs, \
            "tags parameter must be passed through"
        assert call_kwargs["tags"] == params["tags"], \
            f"tags must be preserved exactly, expected {params['tags']}, got {call_kwargs['tags']}"
    else:
        # When no tags provided, it should be None or not present
        tags = call_kwargs.get("tags")
        assert tags is None, \
            f"When no tags provided, it should be None, got '{tags}'"
    
    # Verify parameter mapping: limit → limit
    assert "limit" in call_kwargs, \
        "limit parameter must always be passed"
    assert call_kwargs["limit"] == params["limit"], \
        f"limit must be preserved exactly, expected {params['limit']}, got {call_kwargs['limit']}"
    
    # Verify result structure is correct (RetrievalResult)
    assert isinstance(result, dict), \
        "retrieve() must return a dictionary (RetrievalResult)"
    assert "memories" in result, \
        "RetrievalResult must have 'memories' field"
    assert "total_count" in result, \
        "RetrievalResult must have 'total_count' field"
    assert "query_metadata" in result, \
        "RetrievalResult must have 'query_metadata' field"


# Feature: intent-based-memory-retrieval-enhancements, Property 8: Parameter Mapping Correctness
@given(
    query=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    limit=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_legacy_api_parameter_mapping(query: Optional[str], limit: int):
    """
    Property 8: Parameter Mapping Correctness (Legacy API)
    
    For any query string passed using the legacy API (retrieve(query="text")),
    the query must be correctly mapped to action_type parameter in
    MemoryManager.query_memories().
    
    **Validates: Requirements 1.6 (backward compatibility)**
    
    This test verifies that:
    1. Legacy API (query string only) still works correctly
    2. Query string is mapped to action_type
    3. Limit parameter is passed through correctly
    4. The mapping is identical to the enhanced API
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() with legacy API
    result = adapter.retrieve(query=query, limit=limit)
    
    # Verify query_memories was called
    assert mock_memory_manager.query_memories.called, \
        "retrieve() must call memory_manager.query_memories()"
    
    # Get the actual call arguments
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Verify query → action_type mapping (with normalization)
    # Phase 5A normalizes whitespace-only queries to None
    expected_query = query
    if query is not None and isinstance(query, str) and not query.strip():
        expected_query = None
    
    if expected_query is not None:
        assert call_kwargs.get("action_type") == expected_query, \
            f"Legacy query '{query}' must map to action_type (normalized to {repr(expected_query)}), got '{call_kwargs.get('action_type')}'"
    else:
        # Empty or None query should map to None
        assert call_kwargs.get("action_type") is None, \
            f"Empty/None query should map to None action_type, got '{call_kwargs.get('action_type')}'"
    
    # Verify limit mapping
    assert call_kwargs.get("limit") == limit, \
        f"limit must be {limit}, got {call_kwargs.get('limit')}"


# Feature: intent-based-memory-retrieval-enhancements, Property 8: Parameter Mapping Correctness
@given(params=query_parameters())
@settings(max_examples=10)
@pytest.mark.property_test
def test_category_not_passed_to_memory_manager(params: Dict[str, Any]):
    """
    Property 8: Parameter Mapping Correctness (Category Handling)
    
    The category parameter should NOT be passed to MemoryManager.query_memories()
    because MemoryManager doesn't support category filtering directly.
    Category filtering is applied during result transformation.
    
    **Validates: Requirements 1.6**
    
    This test verifies that:
    1. Category parameter is not passed to query_memories()
    2. Category filtering is deferred to post-processing
    3. Other parameters are still mapped correctly
    """
    # Add category to params if not present
    if "category" not in params:
        params["category"] = "test-category"
    
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() with params including category
    result = adapter.retrieve(params=params)
    
    # Verify query_memories was called
    assert mock_memory_manager.query_memories.called, \
        "retrieve() must call memory_manager.query_memories()"
    
    # Get the actual call arguments
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Verify category is NOT passed to query_memories
    assert "category" not in call_kwargs, \
        "category parameter should NOT be passed to query_memories() " \
        "(it's filtered during result transformation)"
    
    # Verify other parameters are still mapped correctly (with normalization)
    # Phase 5A normalizes whitespace-only queries to None
    if "query" in params:
        expected_query = params["query"]
        if isinstance(expected_query, str) and not expected_query.strip():
            expected_query = None
        
        assert call_kwargs.get("action_type") == expected_query, \
            f"query parameter should still be mapped correctly (normalized to {repr(expected_query)})"
    
    if "tags" in params:
        assert call_kwargs.get("tags") == params["tags"], \
            "tags parameter should still be mapped correctly"
    
    assert call_kwargs.get("limit") == params["limit"], \
        "limit parameter should still be mapped correctly"


# Feature: intent-based-memory-retrieval-enhancements, Property 8: Parameter Mapping Correctness
@given(params=query_parameters())
@settings(max_examples=10)
@pytest.mark.property_test
def test_embedding_parameter_ignored(params: Dict[str, Any]):
    """
    Property 8: Parameter Mapping Correctness (Embedding Handling)
    
    The embedding parameter should be accepted but ignored (not passed to
    MemoryManager) since vector search is not yet implemented.
    
    **Validates: Requirements 1.6, 9.3**
    
    This test verifies that:
    1. Embedding parameter doesn't cause errors
    2. Embedding is not passed to query_memories()
    3. Other parameters are still mapped correctly
    4. System is prepared for future vector search
    """
    # Add embedding to params
    params["embedding"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() with params including embedding
    # Should not raise an error
    result = adapter.retrieve(params=params)
    
    # Verify query_memories was called
    assert mock_memory_manager.query_memories.called, \
        "retrieve() must call memory_manager.query_memories()"
    
    # Get the actual call arguments
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Verify embedding is NOT passed to query_memories
    assert "embedding" not in call_kwargs, \
        "embedding parameter should NOT be passed to query_memories() " \
        "(vector search not yet implemented)"
    
    # Verify other parameters are still mapped correctly (with normalization)
    # Phase 5A normalizes whitespace-only queries to None
    if "query" in params:
        expected_query = params["query"]
        if isinstance(expected_query, str) and not expected_query.strip():
            expected_query = None
        
        assert call_kwargs.get("action_type") == expected_query, \
            f"query parameter should still be mapped correctly (normalized to {repr(expected_query)})"
    
    assert call_kwargs.get("limit") == params["limit"], \
        "limit parameter should still be mapped correctly"
    
    # Verify result is valid
    assert isinstance(result, dict), \
        "retrieve() should return valid RetrievalResult even with embedding parameter"
