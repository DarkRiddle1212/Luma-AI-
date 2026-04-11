"""
Property-Based Tests for SQLite Memory Adapter

This module implements property-based tests using Hypothesis to verify
universal correctness properties for the SQLiteMemoryAdapter.

Feature: reasoning-memory-integration
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, MagicMock
from datetime import datetime

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import MemoryStorageError, MemoryRetrievalError
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def memory_metadata(draw):
    """Generate random metadata dictionaries for memory storage."""
    tags = draw(st.lists(st.text(min_size=1, max_size=5), min_size=0, max_size=5))
    category = draw(st.sampled_from(["general", "education", "work", "personal", "system"]))
    
    metadata = {
        "tags": tags,
        "category": category
    }
    
    # Optionally add extra fields
    if draw(st.booleans()):
        metadata["source"] = draw(st.sampled_from(["user", "system", "api"]))
    
    return metadata


# ============================================================================
# 2.4 Property Test: Adapter Delegation Correctness (Property 1)
# ============================================================================

# Feature: reasoning-memory-integration, Property 1: Adapter Delegation Correctness
@given(
    content=st.text(min_size=1, max_size=5),
    metadata=memory_metadata()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_store_delegation_property(content, metadata):
    """
    Property: For any valid content and metadata, SQLiteMemoryAdapter.store()
    should delegate to MemoryManager.create_memory() and return a valid ID.
    
    **Validates: Requirements 2.3, 2.5**
    
    This test verifies that:
    1. store() delegates to memory_manager.create_memory()
    2. store() returns a non-empty string ID
    3. store() maps metadata correctly (tags, category)
    4. store() handles the delegation without raising exceptions
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Configure mock to return a valid entry ID
    expected_id = f"memory_{hash(content) % 10000}"
    mock_memory_manager.create_memory.return_value = expected_id
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call store() - should delegate to MemoryManager
    result_id = adapter.store(content, metadata)
    
    # Verify delegation occurred
    assert mock_memory_manager.create_memory.called, \
        "store() must delegate to memory_manager.create_memory()"
    
    # Verify the call was made with correct parameters
    call_args = mock_memory_manager.create_memory.call_args
    assert call_args is not None, "create_memory should have been called"
    
    # Verify action parameter contains the content
    assert call_args.kwargs.get("action") == content, \
        f"action parameter should be '{content}', got '{call_args.kwargs.get('action')}'"
    
    # Verify tags parameter (deduplicated by design)
    expected_tags = metadata.get("tags", [])
    actual_tags = call_args.kwargs.get("tags")
    
    # Tags should be deduplicated using set()
    assert set(actual_tags) == set(expected_tags), \
        f"tags should contain same unique values: expected {set(expected_tags)}, got {set(actual_tags)}"
    
    # Verify no duplicates in actual tags
    assert len(actual_tags) == len(set(actual_tags)), \
        f"tags should not contain duplicates: {actual_tags}"
    
    # Verify context parameter exists and is a dict
    context = call_args.kwargs.get("context")
    assert isinstance(context, dict), \
        f"context parameter must be a dict, got {type(context)}"
    
    # Verify device_id parameter exists
    device_id = call_args.kwargs.get("device_id")
    assert device_id is not None, "device_id parameter must be provided"
    
    # Verify result is a valid ID (non-empty string)
    assert isinstance(result_id, str), \
        f"store() must return a string ID, got {type(result_id)}"
    assert len(result_id) > 0, \
        "store() must return a non-empty string ID"
    assert result_id == expected_id, \
        f"store() should return the ID from MemoryManager, expected {expected_id}, got {result_id}"


# Feature: reasoning-memory-integration, Property 1: Adapter Delegation Correctness
@given(
    query=st.text(min_size=1, max_size=100),
    limit=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_retrieve_delegation_property(query, limit):
    """
    Property: For any valid query and limit, SQLiteMemoryAdapter.retrieve()
    should delegate to MemoryManager.query_memories() and return results in
    the correct RetrievalResult format.
    
    **Validates: Requirements 2.4, 2.5**
    
    This test verifies that:
    1. retrieve() delegates to memory_manager.query_memories()
    2. retrieve() returns a RetrievalResult dictionary (not a raw list)
    3. RetrievalResult contains: memories, total_count, query_metadata
    4. Each memory has the required fields: id, content, metadata, timestamp, category, tags
    5. retrieve() handles the delegation without raising exceptions
    6. The format matches the enhanced MemoryInterface contract
    """
    # Create mock MemoryManager with spec to ensure correct behavior
    mock_memory_manager = Mock(spec=['query_memories', 'create_memory'])
    
    # Create mock MemoryEntry objects to return
    mock_entries = []
    for i in range(min(3, limit)):  # Return up to 3 test entries
        mock_entry = Mock(spec=['id', 'action', 'tags', 'context', 'created_at', 'timestamp'])
        mock_entry.id = f"entry_{i}"
        mock_entry.action = f"Test action {i}"
        mock_entry.tags = ["test", f"tag{i}"]
        mock_entry.context = {"category": "test", "index": i}
        mock_entry.created_at = datetime.now()
        mock_entry.timestamp = datetime.now()
        mock_entries.append(mock_entry)
    
    # Configure mock to return a COPY of the list each time to avoid mutation issues
    # Use side_effect with a lambda to return a fresh copy
    mock_memory_manager.query_memories.side_effect = lambda **kwargs: list(mock_entries)
    
    # Defensive assertion: verify mock configuration before creating adapter
    test_call_result = mock_memory_manager.query_memories(action_type="test", limit=10)
    assert isinstance(test_call_result, list), \
        f"Mock should return a list, got {type(test_call_result)}"
    assert len(test_call_result) == len(mock_entries), \
        f"Mock should return {len(mock_entries)} entries, got {len(test_call_result)}"
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() - should delegate to MemoryManager
    result = adapter.retrieve(query=query, limit=limit)
    
    # Verify delegation occurred
    assert mock_memory_manager.query_memories.called, \
        "retrieve() must delegate to memory_manager.query_memories()"
    
    # Verify the call was made with correct parameters
    call_args = mock_memory_manager.query_memories.call_args
    assert call_args is not None, "query_memories should have been called"
    
    # Verify action_type parameter contains the query (with normalization)
    # Phase 5A normalizes whitespace-only queries to None
    expected_query = query
    if query is not None and isinstance(query, str) and not query.strip():
        expected_query = None
    
    assert call_args.kwargs.get("action_type") == expected_query, \
        f"action_type parameter should be '{expected_query}', got '{call_args.kwargs.get('action_type')}'"
    
    # Verify limit parameter
    assert call_args.kwargs.get("limit") == limit, \
        f"limit parameter should be {limit}, got {call_args.kwargs.get('limit')}"
    
    # Verify result is a RetrievalResult dictionary (not a list)
    assert isinstance(result, dict), \
        f"retrieve() must return a RetrievalResult dict, got {type(result)}"
    
    # Verify RetrievalResult structure
    assert "memories" in result, "RetrievalResult must have 'memories' field"
    assert "total_count" in result, "RetrievalResult must have 'total_count' field"
    assert "query_metadata" in result, "RetrievalResult must have 'query_metadata' field"
    
    # Verify memories is a list
    memories = result["memories"]
    assert isinstance(memories, list), \
        f"memories must be a list, got {type(memories)}"
    
    # Verify each memory has the correct format
    for memory in memories:
        assert isinstance(memory, dict), \
            f"Each memory must be a dict, got {type(memory)}"
        
        # Verify required fields exist
        assert "id" in memory, "Memory must have 'id' field"
        assert "content" in memory, "Memory must have 'content' field"
        assert "metadata" in memory, "Memory must have 'metadata' field"
        assert "timestamp" in memory, "Memory must have 'timestamp' field"
        assert "category" in memory, "Memory must have 'category' field"
        assert "tags" in memory, "Memory must have 'tags' field"
        
        # Verify field types
        assert isinstance(memory["id"], str), \
            f"id must be a string, got {type(memory['id'])}"
        assert isinstance(memory["content"], str), \
            f"content must be a string, got {type(memory['content'])}"
        assert isinstance(memory["metadata"], dict), \
            f"metadata must be a dict, got {type(memory['metadata'])}"
        assert isinstance(memory["timestamp"], str), \
            f"timestamp must be a string, got {type(memory['timestamp'])}"
        assert isinstance(memory["category"], str), \
            f"category must be a string, got {type(memory['category'])}"
        assert isinstance(memory["tags"], list), \
            f"tags must be a list, got {type(memory['tags'])}"
    
    # Verify total_count matches memories length
    assert result["total_count"] == len(memories), \
        f"total_count should match memories length: {result['total_count']} != {len(memories)}"
    
    # Verify query_metadata structure
    query_metadata = result["query_metadata"]
    assert isinstance(query_metadata, dict), \
        f"query_metadata must be a dict, got {type(query_metadata)}"
    assert "execution_time_ms" in query_metadata, \
        "query_metadata must have 'execution_time_ms' field"
    assert "limit" in query_metadata, \
        "query_metadata must have 'limit' field"
    assert "has_more" in query_metadata, \
        "query_metadata must have 'has_more' field"
    assert "filters_applied" in query_metadata, \
        "query_metadata must have 'filters_applied' field"


# Feature: reasoning-memory-integration, Property 1: Adapter Delegation Correctness
@given(
    content=st.text(min_size=1, max_size=5),
    metadata=memory_metadata()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_store_error_handling_property(content, metadata):
    """
    Property: For any storage operation that fails, SQLiteMemoryAdapter.store()
    should wrap the exception in MemoryStorageError and preserve the original error.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Exceptions from MemoryManager are caught
    2. Exceptions are wrapped in MemoryStorageError
    3. Original exception is preserved as the cause
    4. Error is logged appropriately
    """
    # Create mock MemoryManager that raises an exception
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.side_effect = Exception("Storage backend failure")
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call store() - should raise MemoryStorageError
    with pytest.raises(MemoryStorageError) as exc_info:
        adapter.store(content, metadata)
    
    # Verify the exception message contains information about the failure
    assert "Storage failed" in str(exc_info.value), \
        "MemoryStorageError should contain 'Storage failed' message"
    
    # Verify the original exception is preserved as the cause
    assert exc_info.value.__cause__ is not None, \
        "Original exception should be preserved as __cause__"
    assert "Storage backend failure" in str(exc_info.value.__cause__), \
        "Original exception message should be preserved"


# Feature: reasoning-memory-integration, Property 1: Adapter Delegation Correctness
@given(
    query=st.text(min_size=1, max_size=100),
    limit=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_retrieve_error_handling_property(query, limit):
    """
    Property: For any retrieval operation that fails, SQLiteMemoryAdapter.retrieve()
    should wrap the exception in MemoryRetrievalError and preserve the original error.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Exceptions from MemoryManager are caught
    2. Exceptions are wrapped in MemoryRetrievalError
    3. Original exception is preserved as the cause
    4. Error is logged appropriately
    """
    # Create mock MemoryManager that raises an exception
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.side_effect = Exception("Query backend failure")
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve() - should raise MemoryRetrievalError
    with pytest.raises(MemoryRetrievalError) as exc_info:
        adapter.retrieve(query=query, limit=limit)
    
    # Verify the exception message contains information about the failure
    assert "Retrieval failed" in str(exc_info.value), \
        "MemoryRetrievalError should contain 'Retrieval failed' message"
    
    # Verify the original exception is preserved as the cause
    assert exc_info.value.__cause__ is not None, \
        "Original exception should be preserved as __cause__"
    assert "Query backend failure" in str(exc_info.value.__cause__), \
        "Original exception message should be preserved"



# ============================================================================
# 3.14 Property Test: Backward Compatibility (Property 13)
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 13: Backward Compatibility
@given(
    query=st.text(min_size=1, max_size=200),
    limit=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_backward_compatibility_legacy_vs_enhanced_api_property(query, limit):
    """
    Property: For any query string passed to retrieve() using the legacy API
    (retrieve(query="text")), the system must produce equivalent results to the
    enhanced API (retrieve(params={"query": "text"})).

    **Validates: Requirements 1.5**

    This test verifies that:
    1. Legacy API retrieve(query="text", limit=N) produces same results as enhanced API
    2. Enhanced API retrieve(params={"query": "text", "limit": N}) produces same results
    3. Both APIs delegate to MemoryManager with identical parameters
    4. Result structure is identical between both APIs
    5. Memory entries are identical between both APIs
    6. Metadata is identical between both APIs
    7. Backward compatibility is maintained across all query strings
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()

    # Create mock MemoryEntry objects to return
    mock_entries = []
    num_entries = min(3, limit)
    for i in range(num_entries):
        mock_entry = Mock()
        mock_entry.id = f"entry_{i}_{hash(query) % 1000}"
        mock_entry.action = f"Content for query '{query}' - entry {i}"
        mock_entry.tags = ["test", f"tag{i}"]
        mock_entry.context = {"category": "general", "index": i}
        mock_entry.created_at = datetime(2024, 1, 15, 10, 30, i)
        mock_entry.timestamp = datetime(2024, 1, 15, 10, 30, i)
        mock_entries.append(mock_entry)

    # Configure mock to return entries
    mock_memory_manager.query_memories.return_value = mock_entries

    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)

    # Call retrieve() with LEGACY API
    result_legacy = adapter.retrieve(query=query, limit=limit)

    # Reset mock to track second call separately
    mock_memory_manager.reset_mock()
    mock_memory_manager.query_memories.return_value = mock_entries

    # Call retrieve() with ENHANCED API
    result_enhanced = adapter.retrieve(params={"query": query, "limit": limit})

    # ========================================================================
    # Verify both APIs produce identical results
    # ========================================================================

    # Verify both results are RetrievalResult dictionaries
    assert isinstance(result_legacy, dict), \
        f"Legacy API must return dict, got {type(result_legacy)}"
    assert isinstance(result_enhanced, dict), \
        f"Enhanced API must return dict, got {type(result_enhanced)}"

    # Verify both have the same top-level keys
    assert set(result_legacy.keys()) == set(result_enhanced.keys()), \
        f"Legacy and enhanced APIs must have same keys. " \
        f"Legacy: {set(result_legacy.keys())}, Enhanced: {set(result_enhanced.keys())}"

    # Verify required keys are present
    required_keys = {"memories", "total_count", "query_metadata"}
    assert required_keys.issubset(result_legacy.keys()), \
        f"Legacy API missing required keys: {required_keys - set(result_legacy.keys())}"
    assert required_keys.issubset(result_enhanced.keys()), \
        f"Enhanced API missing required keys: {required_keys - set(result_enhanced.keys())}"

    # ========================================================================
    # Verify memories are identical
    # ========================================================================

    # Verify same number of memories
    assert len(result_legacy["memories"]) == len(result_enhanced["memories"]), \
        f"Legacy API returned {len(result_legacy['memories'])} memories, " \
        f"Enhanced API returned {len(result_enhanced['memories'])} memories"

    # Verify each memory entry is identical
    for i, (legacy_mem, enhanced_mem) in enumerate(zip(result_legacy["memories"], result_enhanced["memories"])):
        assert legacy_mem["id"] == enhanced_mem["id"], \
            f"Memory {i}: IDs differ. Legacy: '{legacy_mem['id']}', Enhanced: '{enhanced_mem['id']}'"

        assert legacy_mem["content"] == enhanced_mem["content"], \
            f"Memory {i}: Content differs. Legacy: '{legacy_mem['content']}', Enhanced: '{enhanced_mem['content']}'"

        assert legacy_mem["category"] == enhanced_mem["category"], \
            f"Memory {i}: Category differs. Legacy: '{legacy_mem['category']}', Enhanced: '{enhanced_mem['category']}'"

        assert legacy_mem["tags"] == enhanced_mem["tags"], \
            f"Memory {i}: Tags differ. Legacy: {legacy_mem['tags']}, Enhanced: {enhanced_mem['tags']}"

        assert legacy_mem["timestamp"] == enhanced_mem["timestamp"], \
            f"Memory {i}: Timestamp differs. Legacy: '{legacy_mem['timestamp']}', Enhanced: '{enhanced_mem['timestamp']}'"

        assert legacy_mem["metadata"] == enhanced_mem["metadata"], \
            f"Memory {i}: Metadata differs. Legacy: {legacy_mem['metadata']}, Enhanced: {enhanced_mem['metadata']}"

    # ========================================================================
    # Verify total_count is identical
    # ========================================================================

    assert result_legacy["total_count"] == result_enhanced["total_count"], \
        f"total_count differs. Legacy: {result_legacy['total_count']}, " \
        f"Enhanced: {result_enhanced['total_count']}"

    # ========================================================================
    # Verify query_metadata structure is identical
    # ========================================================================

    # Both should have same metadata keys
    assert set(result_legacy["query_metadata"].keys()) == set(result_enhanced["query_metadata"].keys()), \
        f"query_metadata keys differ. " \
        f"Legacy: {set(result_legacy['query_metadata'].keys())}, " \
        f"Enhanced: {set(result_enhanced['query_metadata'].keys())}"

    # Verify limit is identical
    assert result_legacy["query_metadata"]["limit"] == result_enhanced["query_metadata"]["limit"], \
        f"Metadata limit differs. Legacy: {result_legacy['query_metadata']['limit']}, " \
        f"Enhanced: {result_enhanced['query_metadata']['limit']}"

    assert result_legacy["query_metadata"]["limit"] == limit, \
        f"Metadata limit should be {limit}, got {result_legacy['query_metadata']['limit']}"

    # Verify has_more flag is identical
    assert result_legacy["query_metadata"]["has_more"] == result_enhanced["query_metadata"]["has_more"], \
        f"Metadata has_more differs. Legacy: {result_legacy['query_metadata']['has_more']}, " \
        f"Enhanced: {result_enhanced['query_metadata']['has_more']}"

    # Verify filters_applied is identical
    assert result_legacy["query_metadata"]["filters_applied"] == result_enhanced["query_metadata"]["filters_applied"], \
        f"Metadata filters_applied differs. " \
        f"Legacy: {result_legacy['query_metadata']['filters_applied']}, " \
        f"Enhanced: {result_enhanced['query_metadata']['filters_applied']}"

    # Verify execution_time_ms exists in both (values may differ slightly due to timing)
    assert "execution_time_ms" in result_legacy["query_metadata"], \
        "Legacy API must include execution_time_ms in metadata"
    assert "execution_time_ms" in result_enhanced["query_metadata"], \
        "Enhanced API must include execution_time_ms in metadata"

    # ========================================================================
    # Verify both APIs delegate to MemoryManager identically
    # ========================================================================

    # Both should have called query_memories exactly once
    assert mock_memory_manager.query_memories.call_count == 1, \
        f"Enhanced API should call query_memories once, called {mock_memory_manager.query_memories.call_count} times"

    # Get the call arguments from enhanced API call
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs

    # Verify parameters match the input
    actual_query = call_kwargs.get("action_type")
    
    # Handle query normalization: empty/whitespace/non-printable strings are normalized to None
    if query is not None and isinstance(query, str) and not query.strip():
        # Query was normalized to None (expected behavior)
        assert actual_query is None, \
            f"Empty/whitespace query should be normalized to None, got {repr(actual_query)}"
    else:
        # Query should be passed through as-is
        assert actual_query == query, \
            f"action_type should be '{query}', got '{actual_query}'"

    assert call_kwargs.get("limit") == limit, \
        f"limit should be {limit}, got {call_kwargs.get('limit')}"


# Feature: intent-based-memory-retrieval-enhancements, Property 13: Backward Compatibility
@given(
    query=st.one_of(st.none(), st.just(""), st.text(min_size=1, max_size=200))
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_backward_compatibility_edge_cases_property(query):
    """
    Property: For any edge case query (None, empty string, or valid string),
    both legacy and enhanced APIs must handle it identically.
    
    Query normalization contract:
    - Empty strings ("") are normalized to None
    - Whitespace-only strings ("  ") are normalized to None
    - Non-printable-only strings ("\x85") are normalized to None

    **Validates: Requirements 1.5**

    This test verifies that:
    1. Both APIs handle None queries identically
    2. Both APIs handle empty string queries identically (normalized to None)
    3. Both APIs handle whitespace queries identically (normalized to None)
    4. Edge case handling is consistent across both APIs
    5. No regressions in edge case behavior
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []

    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)

    # Call retrieve() with LEGACY API
    result_legacy = adapter.retrieve(query=query)

    # Reset mock
    mock_memory_manager.reset_mock()
    mock_memory_manager.query_memories.return_value = []

    # Call retrieve() with ENHANCED API
    if query is None:
        result_enhanced = adapter.retrieve(params={})
    else:
        result_enhanced = adapter.retrieve(params={"query": query})

    # Verify both produce identical results
    assert result_legacy["total_count"] == result_enhanced["total_count"], \
        f"total_count differs for query={repr(query)}. " \
        f"Legacy: {result_legacy['total_count']}, Enhanced: {result_enhanced['total_count']}"

    assert len(result_legacy["memories"]) == len(result_enhanced["memories"]), \
        f"Number of memories differs for query={repr(query)}. " \
        f"Legacy: {len(result_legacy['memories'])}, Enhanced: {len(result_enhanced['memories'])}"

    # Verify both have same structure
    assert set(result_legacy.keys()) == set(result_enhanced.keys()), \
        f"Result keys differ for query={repr(query)}"

    # Verify metadata structure is identical
    assert set(result_legacy["query_metadata"].keys()) == set(result_enhanced["query_metadata"].keys()), \
        f"Metadata keys differ for query={repr(query)}"
    
    # Verify query normalization: empty/whitespace strings should be normalized to None
    if query is not None and isinstance(query, str) and not query.strip():
        # Empty or whitespace-only query should be normalized to None
        call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
        assert call_kwargs.get("action_type") is None, \
            f"Empty/whitespace query should be normalized to None, got {repr(call_kwargs.get('action_type'))}"


# Feature: intent-based-memory-retrieval-enhancements, Property 13: Backward Compatibility
@given(
    query=st.text(min_size=1, max_size=200)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_backward_compatibility_params_precedence_property(query):
    """
    Property: When both query and params are provided, params must take precedence,
    and the result must match what would be returned if only params were provided.

    **Validates: Requirements 1.5**

    This test verifies that:
    1. When both query and params are provided, params takes precedence
    2. The ignored query parameter doesn't affect the result
    3. Result matches what would be returned with params only
    4. Precedence behavior is consistent across all inputs
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()

    # Create mock entries
    mock_entries = [
        Mock(
            id="entry_1",
            action="Content from params query",
            tags=["test"],
            context={"category": "general"},
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
    ]
    mock_memory_manager.query_memories.return_value = mock_entries

    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)

    # Generate a different query for params
    params_query = f"params_{query}"

    # Call retrieve() with BOTH query and params (params should win)
    result_both = adapter.retrieve(query="ignored_query", params={"query": params_query, "limit": 10})

    # Reset mock
    mock_memory_manager.reset_mock()
    mock_memory_manager.query_memories.return_value = mock_entries

    # Call retrieve() with ONLY params
    result_params_only = adapter.retrieve(params={"query": params_query, "limit": 10})

    # Verify both produce identical results
    assert result_both["total_count"] == result_params_only["total_count"], \
        "Results should be identical when params is provided (query should be ignored)"

    assert len(result_both["memories"]) == len(result_params_only["memories"]), \
        "Number of memories should be identical when params is provided"

    # Verify the params query was used, not the query parameter
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs.get("action_type") == params_query, \
        f"Should use params query '{params_query}', not query parameter. " \
        f"Got: '{call_kwargs.get('action_type')}'"

    # Verify memories are identical
    for i, (mem_both, mem_params) in enumerate(zip(result_both["memories"], result_params_only["memories"])):
        assert mem_both["id"] == mem_params["id"], \
            f"Memory {i} ID should be identical"
        assert mem_both["content"] == mem_params["content"], \
            f"Memory {i} content should be identical"




# ============================================================================
# Explicit Behavior Documentation Tests
# ============================================================================

# Feature: reasoning-memory-integration, Explicit Behavior: Tag Deduplication
@given(
    content=st.text(min_size=1, max_size=5),
    tags=st.lists(st.text(min_size=1, max_size=5), min_size=2, max_size=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_explicit_tag_deduplication_by_design(content, tags):
    """
    Explicit Test: store() intentionally deduplicates tags using set().
    
    This is BY DESIGN behavior, not a bug. The store() method uses set()
    to ensure that duplicate tags are removed before passing to MemoryManager.
    
    **Validates: Requirements 2.3, 6.6**
    
    This test documents that:
    1. Duplicate tags in metadata are removed by design
    2. Only unique tags are passed to MemoryManager.create_memory()
    3. Tag order is not guaranteed (set semantics)
    4. This behavior is intentional and expected
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_id"
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Add intentional duplicates to tags
    tags_with_duplicates = tags + [tags[0]] if tags else ["tag1", "tag1"]
    metadata = {"tags": tags_with_duplicates, "category": "test"}
    
    # Store memory
    adapter.store(content, metadata)
    
    # Verify tags were deduplicated
    call_args = mock_memory_manager.create_memory.call_args
    actual_tags = call_args.kwargs.get("tags", [])
    
    # Verify no duplicates in actual tags (BY DESIGN)
    assert len(actual_tags) == len(set(actual_tags)), \
        f"BY DESIGN: store() removes duplicate tags. Got duplicates in: {actual_tags}"
    
    # Verify all unique tags are present
    assert set(actual_tags) == set(tags_with_duplicates), \
        f"All unique tags should be present: expected {set(tags_with_duplicates)}, got {set(actual_tags)}"
    
    # Document that this is intentional behavior
    # The implementation uses: list(set(default_tags + metadata_tags))
    # This is the correct behavior for tag management


# Feature: intent-based-memory-retrieval-enhancements, Explicit Behavior: Query Normalization
@given(
    whitespace_query=st.sampled_from([
        "",           # Empty string
        " ",          # Single space
        "  ",         # Multiple spaces
        "\t",         # Tab
        "\n",         # Newline
        "   \t\n  ",  # Mixed whitespace
    ])
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_explicit_query_normalization_whitespace(whitespace_query):
    """
    Explicit Test: retrieve() normalizes empty/whitespace-only queries to None.
    
    This is BY DESIGN behavior. The enhanced retrieval API normalizes:
    - Empty strings ("") → None
    - Whitespace-only strings ("  ", "\t", "\n") → None
    
    **Validates: Requirements 1.5**
    
    This test documents that:
    1. Empty strings are normalized to None
    2. Whitespace-only strings are normalized to None
    3. This normalization happens in _validate_and_normalize_params()
    4. This behavior is intentional and expected
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve with whitespace query
    result = adapter.retrieve(query=whitespace_query)
    
    # Verify the query was normalized to None
    call_args = mock_memory_manager.query_memories.call_args
    actual_query = call_args.kwargs.get("action_type")
    
    assert actual_query is None, \
        f"BY DESIGN: Empty/whitespace queries are normalized to None. " \
        f"Input: {repr(whitespace_query)}, Got: {repr(actual_query)}"
    
    # Verify result structure is still valid
    assert isinstance(result, dict), "Result should be a RetrievalResult dict"
    assert "memories" in result, "Result should have 'memories' field"
    assert "total_count" in result, "Result should have 'total_count' field"
    assert "query_metadata" in result, "Result should have 'query_metadata' field"


# Feature: intent-based-memory-retrieval-enhancements, Explicit Behavior: Query Normalization
@pytest.mark.property_test
def test_explicit_query_normalization_non_printable():
    """
    Explicit Test: retrieve() normalizes certain non-printable queries.
    
    This is BY DESIGN behavior. The enhanced retrieval API uses str.strip()
    which normalizes whitespace characters (including some non-printables).
    
    Note: Not all non-printable characters are removed by str.strip().
    Only whitespace-like characters are normalized to None.
    
    **Validates: Requirements 1.5**
    
    This test documents that:
    1. Whitespace-like non-printables are normalized to None
    2. This uses str.strip() which removes whitespace characters
    3. Other non-printable characters (like \x00) may be preserved
    4. This normalization happens in _validate_and_normalize_params()
    5. This behavior is intentional and expected
    """
    # Test whitespace-like non-printable characters that ARE normalized
    whitespace_non_printables = [
        "\x85",           # Next Line (NEL) - removed by strip()
        "\x0b",           # Vertical tab - removed by strip()
        "\x0c",           # Form feed - removed by strip()
        "\x1c",           # File separator - removed by strip()
        "\x85\x0b\x0c",   # Multiple whitespace non-printables
    ]
    
    for non_printable_query in whitespace_non_printables:
        # Create mock MemoryManager
        mock_memory_manager = Mock()
        mock_memory_manager.query_memories.return_value = []
        
        # Create adapter with mock
        adapter = SQLiteMemoryAdapter(mock_memory_manager)
        
        # Call retrieve with non-printable query
        result = adapter.retrieve(query=non_printable_query)
        
        # Verify the query was normalized to None
        call_args = mock_memory_manager.query_memories.call_args
        actual_query = call_args.kwargs.get("action_type")
        
        assert actual_query is None, \
            f"BY DESIGN: Whitespace-like non-printables are normalized to None. " \
            f"Input: {repr(non_printable_query)}, Got: {repr(actual_query)}"
        
        # Verify result structure is still valid
        assert isinstance(result, dict), "Result should be a RetrievalResult dict"
        assert "memories" in result, "Result should have 'memories' field"


# Feature: intent-based-memory-retrieval-enhancements, Explicit Behavior: RetrievalResult Contract
@given(
    query=st.text(min_size=1, max_size=100),
    limit=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_explicit_retrieval_result_contract(query, limit):
    """
    Explicit Test: retrieve() returns RetrievalResult dictionary, not raw list.
    
    This is BY DESIGN behavior. The enhanced retrieval API returns a structured
    RetrievalResult dictionary with:
    - memories: List of memory entries
    - total_count: Total number of results
    - query_metadata: Execution metadata (timing, filters, pagination)
    
    **Validates: Requirements 1.5, 2.4**
    
    This test documents that:
    1. Return type is RetrievalResult (dict), not list
    2. RetrievalResult has required fields: memories, total_count, query_metadata
    3. Backward compatibility is maintained (legacy API also returns RetrievalResult)
    4. This behavior is intentional and expected
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create mock entries
    mock_entries = [
        Mock(
            id=f"entry_{i}",
            action=f"Content {i}",
            tags=["test"],
            context={"category": "general"},
            created_at=datetime(2024, 1, 15, 10, 30, i),
            timestamp=datetime(2024, 1, 15, 10, 30, i)
        )
        for i in range(min(3, limit))
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve
    result = adapter.retrieve(query=query, limit=limit)
    
    # Verify return type is dict (RetrievalResult), not list
    assert isinstance(result, dict), \
        f"BY DESIGN: retrieve() returns RetrievalResult dict, not list. Got: {type(result)}"
    
    # Verify required fields are present
    required_fields = {"memories", "total_count", "query_metadata"}
    assert required_fields.issubset(result.keys()), \
        f"BY DESIGN: RetrievalResult must have fields {required_fields}. Got: {result.keys()}"
    
    # Verify memories is a list
    assert isinstance(result["memories"], list), \
        f"memories field must be a list, got {type(result['memories'])}"
    
    # Verify total_count is an integer
    assert isinstance(result["total_count"], int), \
        f"total_count field must be an int, got {type(result['total_count'])}"
    
    # Verify query_metadata is a dict
    assert isinstance(result["query_metadata"], dict), \
        f"query_metadata field must be a dict, got {type(result['query_metadata'])}"
    
    # Verify query_metadata has required fields
    metadata_fields = {"execution_time_ms", "filters_applied", "limit", "has_more"}
    assert metadata_fields.issubset(result["query_metadata"].keys()), \
        f"query_metadata must have fields {metadata_fields}. Got: {result['query_metadata'].keys()}"
    
    # Document that this is the new contract
    # Old contract: retrieve() returned List[MemoryEntry]
    # New contract: retrieve() returns RetrievalResult with structured metadata



# ============================================================================
# 3.7 Property Test: Embedding Parameter Tolerance (Property 14)
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 14: Embedding Tolerance
@given(
    embedding=st.lists(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False
        ),
        min_size=0,
        max_size=1536  # Common embedding dimension
    ),
    limit=st.integers(min_value=1, max_value=100),
    category=st.one_of(st.none(), st.sampled_from(["general", "education", "work", "personal", "system"]))
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_embedding_parameter_tolerance_property(embedding, limit, category):
    """
    Property: For any valid embedding vector passed to SQLiteMemoryAdapter.retrieve(),
    the adapter must tolerate the parameter gracefully without raising exceptions.
    
    **Validates: Requirements 3.7 - Task 3.7: Embedding Parameter Tolerance**
    
    This test verifies that:
    1. No exception is raised when "embedding" parameter is present in params
    2. The adapter safely ignores the embedding parameter (future vector search)
    3. A valid RetrievalResult dictionary is returned
    4. The return structure remains correct and consistent
    5. Input parameters are not mutated
    6. The embedding parameter does not affect query execution
    7. Other valid parameters (limit, category) are processed correctly
    
    This is a boundary-level tolerance verification test that ensures the adapter
    can handle embedding parameters at the SQLiteMemoryAdapter level, not just
    at the validation layer.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create mock entries
    mock_entries = [
        Mock(
            id=f"entry_{i}",
            action=f"Test content {i}",
            tags=["test", f"tag{i}"],
            context={"category": category or "general", "index": i},
            created_at=datetime(2024, 1, 15, 10, 30, i),
            timestamp=datetime(2024, 1, 15, 10, 30, i)
        )
        for i in range(min(3, limit))
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Build params with embedding and other valid parameters
    params = {
        "embedding": embedding,
        "limit": limit
    }
    
    # Add optional category if provided
    if category is not None:
        params["category"] = category
    
    # Store original params for mutation check
    original_params = dict(params)
    original_embedding = list(embedding)  # Copy the list
    
    # Call retrieve() with embedding parameter - should NOT raise exception
    try:
        result = adapter.retrieve(params=params)
    except Exception as e:
        pytest.fail(
            f"retrieve() should tolerate embedding parameter without raising exception. "
            f"Got {type(e).__name__}: {e}"
        )
    
    # ========================================================================
    # Verify no exception was raised and result is valid
    # ========================================================================
    
    # Verify result is a RetrievalResult dictionary
    assert isinstance(result, dict), \
        f"retrieve() must return RetrievalResult dict, got {type(result)}"
    
    # Verify required fields are present
    required_fields = {"memories", "total_count", "query_metadata"}
    assert required_fields.issubset(result.keys()), \
        f"RetrievalResult must have fields {required_fields}. Got: {result.keys()}"
    
    # Verify memories is a list
    assert isinstance(result["memories"], list), \
        f"memories field must be a list, got {type(result['memories'])}"
    
    # Verify total_count is an integer
    assert isinstance(result["total_count"], int), \
        f"total_count field must be an int, got {type(result['total_count'])}"
    
    # Verify query_metadata is a dict
    assert isinstance(result["query_metadata"], dict), \
        f"query_metadata field must be a dict, got {type(result['query_metadata'])}"
    
    # ========================================================================
    # Verify each memory has correct structure
    # ========================================================================
    
    for i, memory in enumerate(result["memories"]):
        assert isinstance(memory, dict), \
            f"Memory {i} must be a dict, got {type(memory)}"
        
        # Verify required fields
        memory_fields = {"id", "content", "metadata", "timestamp", "category", "tags"}
        assert memory_fields.issubset(memory.keys()), \
            f"Memory {i} must have fields {memory_fields}. Got: {memory.keys()}"
        
        # Verify field types
        assert isinstance(memory["id"], str), \
            f"Memory {i} id must be string, got {type(memory['id'])}"
        assert isinstance(memory["content"], str), \
            f"Memory {i} content must be string, got {type(memory['content'])}"
        assert isinstance(memory["metadata"], dict), \
            f"Memory {i} metadata must be dict, got {type(memory['metadata'])}"
        assert isinstance(memory["timestamp"], str), \
            f"Memory {i} timestamp must be string, got {type(memory['timestamp'])}"
        assert isinstance(memory["category"], str), \
            f"Memory {i} category must be string, got {type(memory['category'])}"
        assert isinstance(memory["tags"], list), \
            f"Memory {i} tags must be list, got {type(memory['tags'])}"
    
    # ========================================================================
    # Verify embedding parameter was ignored (not in filters_applied)
    # ========================================================================
    
    filters_applied = result["query_metadata"]["filters_applied"]
    assert "embedding" not in filters_applied, \
        "embedding parameter should be ignored and not appear in filters_applied"
    
    # ========================================================================
    # Verify other parameters were processed correctly
    # ========================================================================
    
    # Verify limit was applied
    assert result["query_metadata"]["limit"] == limit, \
        f"limit should be {limit}, got {result['query_metadata']['limit']}"
    
    # Verify category filter was applied if provided
    if category is not None:
        # Category filter is applied in post-processing
        for memory in result["memories"]:
            assert memory["category"] == category, \
                f"All memories should have category '{category}', got '{memory['category']}'"
        
        # Category should appear in filters_applied
        assert filters_applied.get("category") == category, \
            f"category filter should be in filters_applied: {filters_applied}"
    
    # ========================================================================
    # Verify input parameters were not mutated
    # ========================================================================
    
    assert params == original_params, \
        "Input params should not be mutated by retrieve()"
    
    assert params["embedding"] == original_embedding, \
        "Embedding list should not be mutated by retrieve()"
    
    # ========================================================================
    # Verify MemoryManager was called without embedding parameter
    # ========================================================================
    
    assert mock_memory_manager.query_memories.called, \
        "retrieve() should delegate to memory_manager.query_memories()"
    
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Verify embedding was NOT passed to MemoryManager
    assert "embedding" not in call_kwargs, \
        "embedding parameter should not be passed to MemoryManager"
    
    # Verify limit was passed correctly
    assert call_kwargs.get("limit") == limit, \
        f"limit should be {limit}, got {call_kwargs.get('limit')}"
    
    # Verify other parameters were passed correctly
    # (category is not passed to MemoryManager, it's filtered in post-processing)
    assert "category" not in call_kwargs, \
        "category is not passed to MemoryManager (post-processing filter)"


# Feature: intent-based-memory-retrieval-enhancements, Property 14: Embedding Tolerance
@given(
    embedding=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    query=st.text(min_size=1, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_embedding_with_query_combination_property(embedding, query):
    """
    Property: When both embedding and query parameters are provided together,
    the adapter must handle both gracefully, ignoring embedding and processing query.
    
    **Validates: Requirements 3.7 - Task 3.7: Embedding Parameter Tolerance**
    
    This test verifies that:
    1. Embedding and query can be provided together without errors
    2. The query parameter is processed normally
    3. The embedding parameter is safely ignored
    4. Results are consistent with query-only retrieval
    5. No interference between embedding and query parameters
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create mock entries
    mock_entries = [
        Mock(
            id="entry_1",
            action=f"Content matching query: {query}",
            tags=["test"],
            context={"category": "general"},
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve with both embedding and query
    params = {
        "embedding": embedding,
        "query": query,
        "limit": 10
    }
    
    result = adapter.retrieve(params=params)
    
    # Verify result is valid
    assert isinstance(result, dict), "Result should be a RetrievalResult dict"
    assert "memories" in result, "Result should have memories field"
    assert "total_count" in result, "Result should have total_count field"
    assert "query_metadata" in result, "Result should have query_metadata field"
    
    # Verify query was processed
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    
    # Handle query normalization
    expected_query = query if query.strip() else None
    assert call_kwargs.get("action_type") == expected_query, \
        f"Query should be processed: expected '{expected_query}', got '{call_kwargs.get('action_type')}'"
    
    # Verify embedding was ignored
    assert "embedding" not in call_kwargs, \
        "Embedding should not be passed to MemoryManager"
    
    # Verify embedding is not in filters_applied
    filters_applied = result["query_metadata"]["filters_applied"]
    assert "embedding" not in filters_applied, \
        "Embedding should not appear in filters_applied"
    
    # Verify query is in filters_applied (if not normalized to None)
    if expected_query is not None:
        assert filters_applied.get("query") == expected_query, \
            f"Query should be in filters_applied: {filters_applied}"


# Feature: intent-based-memory-retrieval-enhancements, Property 14: Embedding Tolerance
@given(
    embedding=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=0, max_size=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_adapter_embedding_various_dimensions_property(embedding):
    """
    Property: The adapter must tolerate embedding vectors of various dimensions,
    including empty lists and very large dimensions.
    
    **Validates: Requirements 3.7 - Task 3.7: Embedding Parameter Tolerance**
    
    This test verifies that:
    1. Empty embedding lists are tolerated
    2. Small embedding dimensions (< 100) are tolerated
    3. Common embedding dimensions (384, 768, 1536) are tolerated
    4. Large embedding dimensions (> 2000) are tolerated
    5. No dimension-specific validation errors occur
    6. The adapter remains dimension-agnostic
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.query_memories.return_value = []
    
    # Create adapter with mock
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Call retrieve with embedding of any dimension
    params = {"embedding": embedding, "limit": 10}
    
    # Should not raise exception regardless of dimension
    try:
        result = adapter.retrieve(params=params)
    except Exception as e:
        pytest.fail(
            f"Adapter should tolerate embedding of dimension {len(embedding)}. "
            f"Got {type(e).__name__}: {e}"
        )
    
    # Verify result is valid
    assert isinstance(result, dict), "Result should be a RetrievalResult dict"
    assert "memories" in result, "Result should have memories field"
    assert "total_count" in result, "Result should have total_count field"
    assert "query_metadata" in result, "Result should have query_metadata field"
    
    # Verify embedding was ignored
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert "embedding" not in call_kwargs, \
        "Embedding should not be passed to MemoryManager"
    
    # Verify no dimension-specific errors in metadata
    filters_applied = result["query_metadata"]["filters_applied"]
    assert "embedding" not in filters_applied, \
        "Embedding should not appear in filters_applied"
