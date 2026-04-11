"""
Unit Tests for SQLiteMemoryAdapter Parameter Validation

This module tests the _validate_and_normalize_params method of SQLiteMemoryAdapter
to ensure proper validation of query parameters including type checking, range
validation, and edge case handling.

Feature: intent-based-memory-retrieval-enhancements
Task: 3.6 Write unit tests for parameter validation
Requirements: 2.6, 8.4, 8.5, 8.6, 9.3
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import QueryParameters


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing."""
    return Mock()


@pytest.fixture
def adapter(mock_memory_manager):
    """Create a SQLiteMemoryAdapter instance for testing."""
    return SQLiteMemoryAdapter(mock_memory_manager)


# ============================================================================
# Valid Parameter Tests
# ============================================================================

def test_validate_with_all_valid_parameters(adapter):
    """
    Test validation with all valid parameters.
    
    Validates: Requirement 2.6
    
    When all parameters are valid, validation should succeed and return
    a normalized dictionary with all parameters preserved.
    """
    # Create valid parameters
    params: QueryParameters = {
        "query": "test query",
        "category": "work",
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2024, 12, 31),
        "tags": ["important", "work"],
        "limit": 20
    }
    
    # Validate parameters
    result = adapter._validate_and_normalize_params(None, params)
    
    # Verify all parameters are preserved
    assert result["query"] == "test query"
    assert result["category"] == "work"
    assert result["start_time"] == datetime(2024, 1, 1)
    assert result["end_time"] == datetime(2024, 12, 31)
    assert result["tags"] == ["important", "work"]
    assert result["limit"] == 20


def test_validate_with_minimal_valid_parameters(adapter):
    """
    Test validation with minimal valid parameters (only query).
    
    Validates: Requirement 2.6
    
    When only a query string is provided, validation should succeed
    and apply default limit.
    """
    params: QueryParameters = {
        "query": "simple query"
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["query"] == "simple query"
    assert result["limit"] == 10  # Default limit


def test_validate_with_legacy_query_string(adapter):
    """
    Test validation with legacy query string parameter.
    
    Validates: Requirement 2.6
    
    When using the legacy API (query string only), validation should
    succeed and create a normalized dictionary.
    """
    result = adapter._validate_and_normalize_params("legacy query", None)
    
    assert result["query"] == "legacy query"
    assert result["limit"] == 10  # Default limit


def test_validate_with_no_parameters(adapter):
    """
    Test validation with no parameters.
    
    Validates: Requirement 2.6
    
    When no parameters are provided, validation should succeed
    and return a dictionary with default limit.
    """
    result = adapter._validate_and_normalize_params(None, None)
    
    assert result.get("query") is None
    assert result["limit"] == 10  # Default limit


def test_validate_with_custom_limit(adapter):
    """
    Test validation with custom limit value.
    
    Validates: Requirement 2.6
    
    When a custom limit is provided, it should be preserved in the result.
    """
    params: QueryParameters = {
        "query": "test",
        "limit": 50
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["limit"] == 50


# ============================================================================
# Invalid Type Tests
# ============================================================================

def test_validate_rejects_non_string_query(adapter):
    """
    Test that validation rejects non-string query parameter.
    
    Validates: Requirement 8.4
    
    When query is not a string, validation should raise ValueError
    with a clear error message.
    """
    params: QueryParameters = {
        "query": 123  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "query" in error_msg.lower()
    assert "string" in error_msg.lower()


def test_validate_rejects_non_integer_limit(adapter):
    """
    Test that validation rejects non-integer limit parameter.
    
    Validates: Requirement 8.4
    
    When limit is not an integer, validation should raise ValueError.
    """
    params: QueryParameters = {
        "limit": "10"  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "limit" in error_msg.lower()
    assert "integer" in error_msg.lower()


def test_validate_rejects_float_limit(adapter):
    """
    Test that validation rejects float limit parameter.
    
    Validates: Requirement 8.4
    
    When limit is a float, validation should raise ValueError.
    """
    params: QueryParameters = {
        "limit": 10.5  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "limit" in error_msg.lower()


def test_validate_rejects_non_list_tags(adapter):
    """
    Test that validation rejects non-list tags parameter.
    
    Validates: Requirement 8.4
    
    When tags is not a list, validation should raise ValueError.
    """
    params: QueryParameters = {
        "tags": "tag1,tag2"  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "tags" in error_msg.lower()
    assert "list" in error_msg.lower()


def test_validate_rejects_tags_with_non_string_elements(adapter):
    """
    Test that validation rejects tags list with non-string elements.
    
    Validates: Requirement 8.4
    
    When tags list contains non-string elements, validation should raise ValueError.
    """
    params: QueryParameters = {
        "tags": ["valid", 123, "another"]  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "tags" in error_msg.lower()
    assert "string" in error_msg.lower()


def test_validate_rejects_non_string_category(adapter):
    """
    Test that validation rejects non-string category parameter.
    
    Validates: Requirement 8.4
    
    When category is not a string, validation should raise ValueError.
    """
    params: QueryParameters = {
        "category": 123  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "category" in error_msg.lower()
    assert "string" in error_msg.lower()


def test_validate_rejects_non_datetime_start_time(adapter):
    """
    Test that validation rejects non-datetime start_time parameter.
    
    Validates: Requirement 8.4
    
    When start_time is not a datetime object, validation should raise ValueError.
    """
    params: QueryParameters = {
        "start_time": "2024-01-01",  # type: ignore
        "end_time": datetime(2024, 12, 31)
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "datetime" in error_msg.lower()


def test_validate_rejects_non_datetime_end_time(adapter):
    """
    Test that validation rejects non-datetime end_time parameter.
    
    Validates: Requirement 8.4
    
    When end_time is not a datetime object, validation should raise ValueError.
    """
    params: QueryParameters = {
        "start_time": datetime(2024, 1, 1),
        "end_time": "2024-12-31"  # type: ignore
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "datetime" in error_msg.lower()


# ============================================================================
# Invalid Range Tests
# ============================================================================

def test_validate_rejects_negative_limit(adapter):
    """
    Test that validation rejects negative limit values.
    
    Validates: Requirement 8.6
    
    When limit is negative, validation should raise ValueError.
    """
    params: QueryParameters = {
        "limit": -5
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "limit" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_validate_rejects_zero_limit(adapter):
    """
    Test that validation rejects zero limit value.
    
    Validates: Requirement 8.6
    
    When limit is zero, validation should raise ValueError.
    """
    params: QueryParameters = {
        "limit": 0
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "limit" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_validate_rejects_invalid_timestamp_range(adapter):
    """
    Test that validation rejects timestamp ranges where start > end.
    
    Validates: Requirement 8.5
    
    When start_time is after end_time, validation should raise ValueError.
    """
    params: QueryParameters = {
        "start_time": datetime(2024, 12, 31),
        "end_time": datetime(2024, 1, 1)  # Earlier than start_time
    }
    
    with pytest.raises(ValueError) as exc_info:
        adapter._validate_and_normalize_params(None, params)
    
    error_msg = str(exc_info.value)
    assert "start_time" in error_msg.lower()
    assert "end_time" in error_msg.lower()


def test_validate_accepts_equal_start_and_end_time(adapter):
    """
    Test that validation accepts equal start_time and end_time.
    
    Validates: Requirement 8.5
    
    When start_time equals end_time, validation should succeed (edge case).
    """
    same_time = datetime(2024, 6, 15, 12, 0, 0)
    params: QueryParameters = {
        "start_time": same_time,
        "end_time": same_time
    }
    
    # Should not raise exception
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["start_time"] == same_time
    assert result["end_time"] == same_time


# ============================================================================
# Edge Case Tests - Empty and Whitespace
# ============================================================================

def test_validate_normalizes_empty_query_to_none(adapter):
    """
    Test that validation normalizes empty query string to None.
    
    Validates: Requirement 8.1
    
    When query is an empty string, it should be normalized to None.
    """
    params: QueryParameters = {
        "query": ""
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["query"] is None


def test_validate_normalizes_whitespace_query_to_none(adapter):
    """
    Test that validation normalizes whitespace-only query to None.
    
    Validates: Requirement 8.3
    
    When query contains only whitespace, it should be normalized to None.
    """
    params: QueryParameters = {
        "query": "   \t\n  "
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["query"] is None


def test_validate_preserves_query_with_leading_trailing_whitespace(adapter):
    """
    Test that validation preserves query with actual content and whitespace.
    
    Validates: Requirement 2.6
    
    When query has content with leading/trailing whitespace, the original
    string should be preserved (not stripped).
    """
    params: QueryParameters = {
        "query": "  actual content  "
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    # Original string is preserved (strip() is only used for checking, not modifying)
    assert result["query"] == "  actual content  "


def test_validate_accepts_none_query(adapter):
    """
    Test that validation accepts None as query value.
    
    Validates: Requirement 8.2
    
    When query is explicitly None, validation should succeed.
    """
    params: QueryParameters = {
        "query": None
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result.get("query") is None


def test_validate_accepts_empty_tags_list(adapter):
    """
    Test that validation accepts empty tags list.
    
    Validates: Requirement 2.6
    
    When tags is an empty list, validation should succeed.
    """
    params: QueryParameters = {
        "tags": []
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["tags"] == []


# ============================================================================
# Embedding Parameter Handling Tests
# ============================================================================

def test_validate_ignores_embedding_parameter(adapter):
    """
    Test that validation accepts and ignores embedding parameter.
    
    Validates: Requirement 9.3
    
    When embedding parameter is provided, validation should accept it
    but remove it from the result (not yet implemented).
    """
    params: QueryParameters = {
        "query": "test",
        "embedding": [0.1, 0.2, 0.3, 0.4]  # type: ignore
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    # Embedding should be removed from result
    assert "embedding" not in result
    # Other parameters should be preserved
    assert result["query"] == "test"


def test_validate_logs_embedding_parameter_warning(adapter, caplog):
    """
    Test that validation logs a debug message when embedding is provided.
    
    Validates: Requirement 9.3
    
    When embedding parameter is provided, a debug log should indicate
    it's not yet implemented.
    """
    import logging
    
    params: QueryParameters = {
        "embedding": [0.1, 0.2, 0.3]  # type: ignore
    }
    
    with caplog.at_level(logging.DEBUG):
        result = adapter._validate_and_normalize_params(None, params)
    
    # Check that debug log was created
    assert any("embedding" in record.message.lower() for record in caplog.records)
    assert any("not yet implemented" in record.message.lower() or "ignoring" in record.message.lower() 
               for record in caplog.records)


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

def test_validate_params_takes_precedence_over_query(adapter):
    """
    Test that params parameter takes precedence over query parameter.
    
    Validates: Requirement 2.6
    
    When both query and params are provided, params should be used.
    """
    params: QueryParameters = {
        "query": "params query"
    }
    
    result = adapter._validate_and_normalize_params("legacy query", params)
    
    # params should take precedence
    assert result["query"] == "params query"


def test_validate_uses_query_when_params_is_none(adapter):
    """
    Test that query parameter is used when params is None.
    
    Validates: Requirement 2.6
    
    When params is None, the legacy query parameter should be used.
    """
    result = adapter._validate_and_normalize_params("legacy query", None)
    
    assert result["query"] == "legacy query"


# ============================================================================
# Optional Parameter Tests
# ============================================================================

def test_validate_accepts_only_start_time(adapter):
    """
    Test that validation accepts start_time without end_time.
    
    Validates: Requirement 2.6
    
    When only start_time is provided (no end_time), validation should succeed.
    """
    params: QueryParameters = {
        "start_time": datetime(2024, 1, 1)
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["start_time"] == datetime(2024, 1, 1)
    assert "end_time" not in result or result.get("end_time") is None


def test_validate_accepts_only_end_time(adapter):
    """
    Test that validation accepts end_time without start_time.
    
    Validates: Requirement 2.6
    
    When only end_time is provided (no start_time), validation should succeed.
    """
    params: QueryParameters = {
        "end_time": datetime(2024, 12, 31)
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result["end_time"] == datetime(2024, 12, 31)
    assert "start_time" not in result or result.get("start_time") is None


def test_validate_accepts_none_category(adapter):
    """
    Test that validation accepts None as category value.
    
    Validates: Requirement 2.6
    
    When category is explicitly None, validation should succeed.
    """
    params: QueryParameters = {
        "category": None
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result.get("category") is None


def test_validate_accepts_none_tags(adapter):
    """
    Test that validation accepts None as tags value.
    
    Validates: Requirement 2.6
    
    When tags is explicitly None, validation should succeed.
    """
    params: QueryParameters = {
        "tags": None
    }
    
    result = adapter._validate_and_normalize_params(None, params)
    
    assert result.get("tags") is None
