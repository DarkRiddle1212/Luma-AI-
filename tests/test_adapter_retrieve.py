"""
Unit Tests for SQLiteMemoryAdapter retrieve() Method

This module implements unit tests for the enhanced retrieve() method
with support for both legacy API (query string) and enhanced API (params dict).

Feature: intent-based-memory-retrieval-enhancements
Task: 3.10 Write unit tests for retrieve() method
Requirements: 1.5, 10.1, 10.2, 10.3, 10.4
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import MemoryRetrievalError


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing."""
    return Mock()


@pytest.fixture
def adapter(mock_memory_manager):
    """Create a SQLiteMemoryAdapter with mock MemoryManager."""
    return SQLiteMemoryAdapter(mock_memory_manager)


def create_mock_entry(entry_id: str, content: str, category: str = "general", tags: list = None):
    """Helper to create a mock MemoryEntry object."""
    mock_entry = Mock()
    mock_entry.id = entry_id
    mock_entry.action = content
    mock_entry.tags = tags or []
    mock_entry.context = {"category": category}
    mock_entry.created_at = datetime(2024, 1, 15, 10, 30, 0)
    mock_entry.timestamp = datetime(2024, 1, 15, 10, 30, 0)
    return mock_entry


# ============================================================================
# Legacy API Tests (query string only)
# ============================================================================

def test_retrieve_legacy_api_with_query_string(adapter, mock_memory_manager):
    """
    Test legacy API: retrieve(query="text")
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    # Setup mock to return test entries
    mock_entries = [
        create_mock_entry("1", "Python tutorial", "education", ["python", "tutorial"]),
        create_mock_entry("2", "Python basics", "education", ["python", "basics"])
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call retrieve with legacy API
    result = adapter.retrieve(query="Python")
    
    # Verify result structure (RetrievalResult)
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "memories" in result, "Result must have 'memories' field"
    assert "total_count" in result, "Result must have 'total_count' field"
    assert "query_metadata" in result, "Result must have 'query_metadata' field"
    
    # Verify memories
    assert len(result["memories"]) == 2
    assert result["total_count"] == 2
    
    # Verify first memory structure
    memory = result["memories"][0]
    assert memory["id"] == "1"
    assert memory["content"] == "Python tutorial"
    assert memory["category"] == "education"
    assert memory["tags"] == ["python", "tutorial"]
    assert "timestamp" in memory
    
    # Verify delegation to MemoryManager
    mock_memory_manager.query_memories.assert_called_once()
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] == "Python"
    assert call_kwargs["limit"] == 10  # default limit


def test_retrieve_legacy_api_with_custom_limit(adapter, mock_memory_manager):
    """
    Test legacy API with custom limit: retrieve(query="text", limit=5)
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_entries = [create_mock_entry("1", "Test", "general", [])]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call retrieve with custom limit
    result = adapter.retrieve(query="test", limit=5)
    
    # Verify limit was passed correctly
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["limit"] == 5
    assert result["query_metadata"]["limit"] == 5


# ============================================================================
# Enhanced API Tests (params dictionary)
# ============================================================================

def test_retrieve_enhanced_api_with_params_dict(adapter, mock_memory_manager):
    """
    Test enhanced API: retrieve(params={...})
    
    **Validates: Requirement 1.5 (enhanced API support)**
    """
    mock_entries = [create_mock_entry("1", "Test content", "work", ["important"])]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call retrieve with params dict
    params = {
        "query": "test",
        "category": "work",
        "tags": ["important"],
        "limit": 20
    }
    result = adapter.retrieve(params=params)
    
    # Verify result structure
    assert isinstance(result, dict)
    assert len(result["memories"]) == 1
    assert result["total_count"] == 1
    
    # Verify delegation with correct parameters
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] == "test"
    assert call_kwargs["tags"] == ["important"]
    assert call_kwargs["limit"] == 20


def test_retrieve_enhanced_api_with_timestamp_range(adapter, mock_memory_manager):
    """
    Test enhanced API with timestamp filters.
    
    **Validates: Requirement 1.5 (enhanced API support)**
    """
    mock_entries = [create_mock_entry("1", "Recent memory", "general", [])]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = datetime(2024, 1, 31, 23, 59, 59)
    
    params = {
        "query": "recent",
        "start_time": start_time,
        "end_time": end_time,
        "limit": 10
    }
    result = adapter.retrieve(params=params)
    
    # Verify timestamp filters were passed
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["start_time"] == start_time
    assert call_kwargs["end_time"] == end_time
    
    # Verify filters are recorded in metadata
    filters_applied = result["query_metadata"]["filters_applied"]
    assert "start_time" in filters_applied
    assert "end_time" in filters_applied


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

def test_retrieve_backward_compatibility_query_vs_params(adapter, mock_memory_manager):
    """
    Test that legacy API and enhanced API produce equivalent results.
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_entries = [create_mock_entry("1", "Test", "general", [])]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call with legacy API
    result_legacy = adapter.retrieve(query="test", limit=10)
    
    # Reset mock
    mock_memory_manager.reset_mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call with enhanced API
    result_enhanced = adapter.retrieve(params={"query": "test", "limit": 10})
    
    # Verify both produce same structure
    assert result_legacy["total_count"] == result_enhanced["total_count"]
    assert len(result_legacy["memories"]) == len(result_enhanced["memories"])
    assert result_legacy["memories"][0]["id"] == result_enhanced["memories"][0]["id"]


def test_retrieve_params_takes_precedence_over_query(adapter, mock_memory_manager):
    """
    Test that params takes precedence when both query and params are provided.
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_entries = [create_mock_entry("1", "Test", "general", [])]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Call with both query and params
    result = adapter.retrieve(query="ignored", params={"query": "used", "limit": 10})
    
    # Verify params was used
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] == "used"


# ============================================================================
# Empty Results Tests
# ============================================================================

def test_retrieve_empty_results_no_matches(adapter, mock_memory_manager):
    """
    Test retrieve when no memories match the query.
    
    **Validates: Requirement 10.1 (total_count accuracy)**
    """
    # Setup mock to return empty list
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query="nonexistent")
    
    # Verify empty result structure
    assert result["memories"] == []
    assert result["total_count"] == 0
    assert isinstance(result["query_metadata"], dict)
    assert result["query_metadata"]["limit"] == 10


def test_retrieve_empty_query_string(adapter, mock_memory_manager):
    """
    Test retrieve with empty query string.
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query="")
    
    # Verify empty query is normalized to None
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] is None
    assert result["memories"] == []


def test_retrieve_none_query(adapter, mock_memory_manager):
    """
    Test retrieve with None query.
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query=None)
    
    # Verify None query is handled
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] is None
    assert result["memories"] == []


def test_retrieve_whitespace_only_query(adapter, mock_memory_manager):
    """
    Test retrieve with whitespace-only query.
    
    **Validates: Requirement 1.5 (backward compatibility)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query="   ")
    
    # Verify whitespace query is normalized to None
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["action_type"] is None


# ============================================================================
# Result Structure and Metadata Tests
# ============================================================================

def test_retrieve_result_structure_complete(adapter, mock_memory_manager):
    """
    Test that RetrievalResult has all required fields.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4**
    """
    mock_entries = [
        create_mock_entry("1", "Memory 1", "work", ["tag1"]),
        create_mock_entry("2", "Memory 2", "work", ["tag2"])
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    result = adapter.retrieve(params={"query": "test", "category": "work", "limit": 10})
    
    # Verify top-level structure
    assert "memories" in result
    assert "total_count" in result
    assert "query_metadata" in result
    
    # Verify query_metadata structure
    metadata = result["query_metadata"]
    assert "execution_time_ms" in metadata, "Must include execution time"
    assert "filters_applied" in metadata, "Must include filters applied"
    assert "limit" in metadata, "Must include limit"
    assert "has_more" in metadata, "Must include has_more flag"
    
    # Verify metadata values
    assert isinstance(metadata["execution_time_ms"], (int, float))
    assert metadata["execution_time_ms"] >= 0
    assert isinstance(metadata["filters_applied"], dict)
    assert metadata["limit"] == 10
    assert isinstance(metadata["has_more"], bool)


def test_retrieve_metadata_includes_execution_time(adapter, mock_memory_manager):
    """
    Test that query_metadata includes execution_time_ms.
    
    **Validates: Requirement 10.2 (execution time tracking)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query="test")
    
    # Verify execution time is present and reasonable
    exec_time = result["query_metadata"]["execution_time_ms"]
    assert isinstance(exec_time, (int, float))
    assert exec_time >= 0
    assert exec_time < 10000  # Should be less than 10 seconds


def test_retrieve_metadata_includes_filters_applied(adapter, mock_memory_manager):
    """
    Test that query_metadata includes filters_applied.
    
    **Validates: Requirement 10.3 (filters tracking)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    params = {
        "query": "test",
        "category": "work",
        "tags": ["important"],
        "limit": 5
    }
    result = adapter.retrieve(params=params)
    
    # Verify filters are recorded (excluding limit)
    filters = result["query_metadata"]["filters_applied"]
    assert "query" in filters
    assert filters["query"] == "test"
    assert "category" in filters
    assert filters["category"] == "work"
    assert "tags" in filters
    assert filters["tags"] == ["important"]
    # limit should not be in filters_applied
    assert "limit" not in filters


def test_retrieve_metadata_total_count_matches_results(adapter, mock_memory_manager):
    """
    Test that total_count matches the number of memories returned.
    
    **Validates: Requirement 10.1 (total_count accuracy)**
    """
    mock_entries = [
        create_mock_entry("1", "Memory 1", "general", []),
        create_mock_entry("2", "Memory 2", "general", []),
        create_mock_entry("3", "Memory 3", "general", [])
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    result = adapter.retrieve(query="test")
    
    # Verify total_count matches actual count
    assert result["total_count"] == len(result["memories"])
    assert result["total_count"] == 3


def test_retrieve_metadata_has_more_flag(adapter, mock_memory_manager):
    """
    Test that has_more flag is present in metadata.
    
    **Validates: Requirement 10.4 (pagination support)**
    """
    mock_memory_manager.query_memories.return_value = []
    
    result = adapter.retrieve(query="test")
    
    # Verify has_more flag exists
    assert "has_more" in result["query_metadata"]
    assert isinstance(result["query_metadata"]["has_more"], bool)
    # Currently always False (pagination not implemented)
    assert result["query_metadata"]["has_more"] is False


# ============================================================================
# Memory Entry Structure Tests
# ============================================================================

def test_retrieve_memory_entry_has_all_required_fields(adapter, mock_memory_manager):
    """
    Test that each MemoryEntry has all required fields.
    
    **Validates: Requirement 10.1 (result completeness)**
    """
    mock_entry = create_mock_entry("123", "Test content", "education", ["python", "tutorial"])
    mock_memory_manager.query_memories.return_value = [mock_entry]
    
    result = adapter.retrieve(query="test")
    
    # Verify memory entry structure
    memory = result["memories"][0]
    assert "id" in memory
    assert "content" in memory
    assert "metadata" in memory
    assert "timestamp" in memory
    assert "category" in memory
    assert "tags" in memory
    
    # Verify field values
    assert memory["id"] == "123"
    assert memory["content"] == "Test content"
    assert memory["category"] == "education"
    assert memory["tags"] == ["python", "tutorial"]
    assert isinstance(memory["timestamp"], str)
    assert isinstance(memory["metadata"], dict)


def test_retrieve_memory_entry_timestamp_is_iso_format(adapter, mock_memory_manager):
    """
    Test that timestamp is in ISO 8601 format.
    
    **Validates: Requirement 10.1 (result completeness)**
    """
    mock_entry = create_mock_entry("1", "Test", "general", [])
    mock_memory_manager.query_memories.return_value = [mock_entry]
    
    result = adapter.retrieve(query="test")
    
    # Verify timestamp format
    timestamp = result["memories"][0]["timestamp"]
    assert isinstance(timestamp, str)
    # Should be parseable as ISO format
    datetime.fromisoformat(timestamp)  # Will raise if invalid


def test_retrieve_applies_default_category_when_missing(adapter, mock_memory_manager):
    """
    Test that default_category is applied when entry has no category.
    
    **Validates: Requirement 10.1 (result completeness)**
    """
    # Create adapter with custom default category
    adapter_custom = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category="custom_default"
    )
    
    # Create mock entry without category in context
    mock_entry = Mock()
    mock_entry.id = "1"
    mock_entry.action = "Test"
    mock_entry.tags = []
    mock_entry.context = {}  # No category
    mock_entry.created_at = datetime.now()
    mock_entry.timestamp = datetime.now()
    
    mock_memory_manager.query_memories.return_value = [mock_entry]
    
    result = adapter_custom.retrieve(query="test")
    
    # Verify default category was applied
    assert result["memories"][0]["category"] == "custom_default"


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_retrieve_raises_memory_retrieval_error_on_failure(adapter, mock_memory_manager):
    """
    Test that retrieve wraps exceptions in MemoryRetrievalError.
    
    **Validates: Requirement 1.5 (error handling)**
    """
    # Setup mock to raise exception
    mock_memory_manager.query_memories.side_effect = Exception("Database connection failed")
    
    # Verify MemoryRetrievalError is raised
    with pytest.raises(MemoryRetrievalError) as exc_info:
        adapter.retrieve(query="test")
    
    # Verify error message
    assert "Retrieval failed" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_retrieve_raises_value_error_on_invalid_params(adapter, mock_memory_manager):
    """
    Test that retrieve raises ValueError for invalid parameters.
    
    **Validates: Requirement 1.5 (parameter validation)**
    """
    # Test invalid limit
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        adapter.retrieve(params={"query": "test", "limit": -1})
    
    # Test invalid timestamp range
    with pytest.raises(ValueError, match="start_time.*must be <= end_time"):
        adapter.retrieve(params={
            "query": "test",
            "start_time": datetime(2024, 1, 31),
            "end_time": datetime(2024, 1, 1)
        })


# ============================================================================
# Multiple Results Tests
# ============================================================================

def test_retrieve_multiple_memories_preserves_order(adapter, mock_memory_manager):
    """
    Test that multiple memories are returned in the order from MemoryManager.
    
    **Validates: Requirement 10.1 (result completeness)**
    """
    mock_entries = [
        create_mock_entry("1", "First", "general", []),
        create_mock_entry("2", "Second", "general", []),
        create_mock_entry("3", "Third", "general", [])
    ]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    result = adapter.retrieve(query="test")
    
    # Verify order is preserved
    assert len(result["memories"]) == 3
    assert result["memories"][0]["id"] == "1"
    assert result["memories"][1]["id"] == "2"
    assert result["memories"][2]["id"] == "3"


def test_retrieve_respects_limit_parameter(adapter, mock_memory_manager):
    """
    Test that limit parameter is passed correctly to MemoryManager.
    
    **Validates: Requirement 10.4 (limit handling)**
    """
    mock_entries = [create_mock_entry(str(i), f"Memory {i}", "general", []) for i in range(5)]
    mock_memory_manager.query_memories.return_value = mock_entries
    
    result = adapter.retrieve(query="test", limit=5)
    
    # Verify limit was passed
    call_kwargs = mock_memory_manager.query_memories.call_args.kwargs
    assert call_kwargs["limit"] == 5
    assert result["query_metadata"]["limit"] == 5
