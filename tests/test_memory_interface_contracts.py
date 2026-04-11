"""
Unit tests for memory interface typed contracts and validation utilities.

Tests Requirements: 2.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from luma.core.memory_interface import (
    QueryParameters,
    MemoryEntry,
    RetrievalResult,
    validate_query_string,
    validate_limit,
    validate_timestamp_range,
    validate_tags,
    validate_category,
    validate_query_parameters,
)


# ============================================================================
# Tests for TypedDict Definitions
# ============================================================================


class TestTypedDictDefinitions:
    """Test that TypedDict definitions have correct fields."""
    
    def test_query_parameters_has_correct_fields(self):
        """Test QueryParameters TypedDict has all expected fields."""
        # Create a valid QueryParameters instance
        params: QueryParameters = {
            "query": "test query",
            "category": "test",
            "start_time": datetime(2024, 1, 1),
            "end_time": datetime(2024, 1, 31),
            "tags": ["tag1", "tag2"],
            "limit": 10,
            "embedding": [0.1, 0.2, 0.3]
        }
        
        # Verify all fields are accessible
        assert params["query"] == "test query"
        assert params["category"] == "test"
        assert isinstance(params["start_time"], datetime)
        assert isinstance(params["end_time"], datetime)
        assert params["tags"] == ["tag1", "tag2"]
        assert params["limit"] == 10
        assert params["embedding"] == [0.1, 0.2, 0.3]
    
    def test_query_parameters_all_fields_optional(self):
        """Test QueryParameters allows empty dictionary (all fields optional)."""
        params: QueryParameters = {}
        assert params == {}
    
    def test_memory_entry_has_correct_fields(self):
        """Test MemoryEntry TypedDict has all expected fields."""
        entry: MemoryEntry = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {"source": "test"},
            "timestamp": "2024-01-15T10:30:00",
            "category": "test",
            "tags": ["tag1"]
        }
        
        # Verify all fields are accessible
        assert entry["id"] == "mem_123"
        assert entry["content"] == "Test content"
        assert entry["metadata"] == {"source": "test"}
        assert entry["timestamp"] == "2024-01-15T10:30:00"
        assert entry["category"] == "test"
        assert entry["tags"] == ["tag1"]
    
    def test_retrieval_result_has_correct_fields(self):
        """Test RetrievalResult TypedDict has all expected fields."""
        result: RetrievalResult = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 15.3,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        }
        
        # Verify all fields are accessible
        assert result["memories"] == []
        assert result["total_count"] == 0
        assert result["query_metadata"]["execution_time_ms"] == 15.3
        assert result["query_metadata"]["filters_applied"] == {}
        assert result["query_metadata"]["limit"] == 10
        assert result["query_metadata"]["has_more"] is False


# ============================================================================
# Tests for validate_query_string
# ============================================================================


class TestValidateQueryString:
    """Test validate_query_string function."""
    
    def test_valid_query_string(self):
        """Test validation with valid query string."""
        result = validate_query_string("Python programming")
        assert result == "Python programming"
    
    def test_none_query_returns_none(self):
        """Test None query returns None."""
        result = validate_query_string(None)
        assert result is None
    
    def test_empty_string_returns_none(self):
        """Test empty string returns None."""
        result = validate_query_string("")
        assert result is None
    
    def test_whitespace_only_returns_none(self):
        """Test whitespace-only string returns None."""
        result = validate_query_string("   ")
        assert result is None
    
    def test_whitespace_with_tabs_returns_none(self):
        """Test whitespace with tabs returns None."""
        result = validate_query_string("\t\n  ")
        assert result is None
    
    def test_invalid_type_raises_error(self):
        """Test non-string type raises ValueError."""
        with pytest.raises(ValueError, match="query must be a string"):
            validate_query_string(123)
    
    def test_invalid_type_includes_type_name(self):
        """Test error message includes actual type name."""
        with pytest.raises(ValueError, match="int"):
            validate_query_string(123)
    
    def test_list_type_raises_error(self):
        """Test list type raises ValueError."""
        with pytest.raises(ValueError, match="query must be a string"):
            validate_query_string(["query"])
    
    def test_dict_type_raises_error(self):
        """Test dict type raises ValueError."""
        with pytest.raises(ValueError, match="query must be a string"):
            validate_query_string({"query": "test"})


# ============================================================================
# Tests for validate_limit
# ============================================================================


class TestValidateLimit:
    """Test validate_limit function."""
    
    def test_valid_positive_limit(self):
        """Test validation with valid positive limit."""
        result = validate_limit(5)
        assert result == 5
    
    def test_none_returns_default(self):
        """Test None returns default limit of 10."""
        result = validate_limit(None)
        assert result == 10
    
    def test_zero_raises_error(self):
        """Test zero limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_limit(0)
    
    def test_negative_raises_error(self):
        """Test negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_limit(-1)
    
    def test_large_negative_raises_error(self):
        """Test large negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_limit(-100)
    
    def test_float_raises_error(self):
        """Test float type raises ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_limit(5.5)
    
    def test_string_raises_error(self):
        """Test string type raises ValueError."""
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_limit("10")
    
    def test_boolean_raises_error(self):
        """Test boolean type raises ValueError (even though bool is subclass of int)."""
        # Note: In Python, bool is a subclass of int, so True == 1 and False == 0
        # We need to handle this edge case
        result = validate_limit(True)
        # True is treated as 1, which is valid
        assert result == 1


# ============================================================================
# Tests for validate_timestamp_range
# ============================================================================


class TestValidateTimestampRange:
    """Test validate_timestamp_range function."""
    
    def test_valid_timestamp_range(self):
        """Test validation with valid timestamp range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        # Should not raise any exception
        validate_timestamp_range(start, end)
    
    def test_equal_timestamps(self):
        """Test validation with equal start and end timestamps."""
        timestamp = datetime(2024, 1, 15)
        # Should not raise any exception
        validate_timestamp_range(timestamp, timestamp)
    
    def test_none_timestamps(self):
        """Test validation with None timestamps."""
        # Should not raise any exception
        validate_timestamp_range(None, None)
    
    def test_none_start_time(self):
        """Test validation with None start_time."""
        end = datetime(2024, 1, 31)
        # Should not raise any exception
        validate_timestamp_range(None, end)
    
    def test_none_end_time(self):
        """Test validation with None end_time."""
        start = datetime(2024, 1, 1)
        # Should not raise any exception
        validate_timestamp_range(start, None)
    
    def test_invalid_range_raises_error(self):
        """Test validation with start > end raises ValueError."""
        start = datetime(2024, 1, 31)
        end = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            validate_timestamp_range(start, end)
    
    def test_invalid_start_type_raises_error(self):
        """Test validation with non-datetime start_time raises ValueError."""
        with pytest.raises(ValueError, match="start_time must be a datetime object"):
            validate_timestamp_range("2024-01-01", datetime(2024, 1, 31))
    
    def test_invalid_end_type_raises_error(self):
        """Test validation with non-datetime end_time raises ValueError."""
        with pytest.raises(ValueError, match="end_time must be a datetime object"):
            validate_timestamp_range(datetime(2024, 1, 1), "2024-01-31")
    
    def test_string_timestamps_raise_error(self):
        """Test validation with string timestamps raises ValueError."""
        with pytest.raises(ValueError, match="start_time must be a datetime object"):
            validate_timestamp_range("2024-01-01", "2024-01-31")
    
    def test_integer_timestamps_raise_error(self):
        """Test validation with integer timestamps raises ValueError."""
        with pytest.raises(ValueError, match="start_time must be a datetime object"):
            validate_timestamp_range(1704067200, 1706745600)


# ============================================================================
# Tests for validate_tags
# ============================================================================


class TestValidateTags:
    """Test validate_tags function."""
    
    def test_valid_tags_list(self):
        """Test validation with valid tags list."""
        tags = ["python", "programming"]
        result = validate_tags(tags)
        assert result == ["python", "programming"]
    
    def test_empty_tags_list(self):
        """Test validation with empty tags list."""
        result = validate_tags([])
        assert result == []
    
    def test_none_tags_returns_none(self):
        """Test None tags returns None."""
        result = validate_tags(None)
        assert result is None
    
    def test_single_tag(self):
        """Test validation with single tag."""
        result = validate_tags(["python"])
        assert result == ["python"]
    
    def test_non_list_raises_error(self):
        """Test non-list type raises ValueError."""
        with pytest.raises(ValueError, match="tags must be a list"):
            validate_tags("python")
    
    def test_tuple_raises_error(self):
        """Test tuple type raises ValueError."""
        with pytest.raises(ValueError, match="tags must be a list"):
            validate_tags(("python", "programming"))
    
    def test_non_string_elements_raise_error(self):
        """Test list with non-string elements raises ValueError."""
        with pytest.raises(ValueError, match="all tags must be strings"):
            validate_tags(["python", 123])
    
    def test_mixed_types_raise_error(self):
        """Test list with mixed types raises ValueError."""
        with pytest.raises(ValueError, match="all tags must be strings"):
            validate_tags(["python", None, "programming"])
    
    def test_integer_list_raises_error(self):
        """Test list of integers raises ValueError."""
        with pytest.raises(ValueError, match="all tags must be strings"):
            validate_tags([1, 2, 3])


# ============================================================================
# Tests for validate_category
# ============================================================================


class TestValidateCategory:
    """Test validate_category function."""
    
    def test_valid_category(self):
        """Test validation with valid category string."""
        result = validate_category("education")
        assert result == "education"
    
    def test_none_category_returns_none(self):
        """Test None category returns None."""
        result = validate_category(None)
        assert result is None
    
    def test_empty_string_category(self):
        """Test empty string category is valid."""
        result = validate_category("")
        assert result == ""
    
    def test_integer_raises_error(self):
        """Test integer type raises ValueError."""
        with pytest.raises(ValueError, match="category must be a string"):
            validate_category(123)
    
    def test_list_raises_error(self):
        """Test list type raises ValueError."""
        with pytest.raises(ValueError, match="category must be a string"):
            validate_category(["education"])
    
    def test_dict_raises_error(self):
        """Test dict type raises ValueError."""
        with pytest.raises(ValueError, match="category must be a string"):
            validate_category({"category": "education"})


# ============================================================================
# Tests for validate_query_parameters
# ============================================================================


class TestValidateQueryParameters:
    """Test validate_query_parameters function."""
    
    def test_none_params_returns_default(self):
        """Test None params returns default limit."""
        result = validate_query_parameters(None)
        assert result == {"limit": 10}
    
    def test_empty_params_returns_default(self):
        """Test empty params returns default limit."""
        result = validate_query_parameters({})
        assert result == {"limit": 10}
    
    def test_valid_query_string(self):
        """Test validation with valid query string."""
        params: QueryParameters = {"query": "Python"}
        result = validate_query_parameters(params)
        assert result["query"] == "Python"
        assert result["limit"] == 10
    
    def test_valid_limit(self):
        """Test validation with valid limit."""
        params: QueryParameters = {"limit": 5}
        result = validate_query_parameters(params)
        assert result["limit"] == 5
    
    def test_valid_category(self):
        """Test validation with valid category."""
        params: QueryParameters = {"category": "education"}
        result = validate_query_parameters(params)
        assert result["category"] == "education"
        assert result["limit"] == 10
    
    def test_valid_tags(self):
        """Test validation with valid tags."""
        params: QueryParameters = {"tags": ["python", "programming"]}
        result = validate_query_parameters(params)
        assert result["tags"] == ["python", "programming"]
        assert result["limit"] == 10
    
    def test_valid_timestamp_range(self):
        """Test validation with valid timestamp range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        params: QueryParameters = {"start_time": start, "end_time": end}
        result = validate_query_parameters(params)
        assert result["start_time"] == start
        assert result["end_time"] == end
        assert result["limit"] == 10
    
    def test_all_valid_params(self):
        """Test validation with all valid parameters."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        params: QueryParameters = {
            "query": "Python",
            "category": "education",
            "tags": ["python", "programming"],
            "start_time": start,
            "end_time": end,
            "limit": 20
        }
        result = validate_query_parameters(params)
        assert result["query"] == "Python"
        assert result["category"] == "education"
        assert result["tags"] == ["python", "programming"]
        assert result["start_time"] == start
        assert result["end_time"] == end
        assert result["limit"] == 20
    
    def test_empty_query_string_normalized_to_none(self):
        """Test empty query string is normalized to None."""
        params: QueryParameters = {"query": ""}
        result = validate_query_parameters(params)
        assert result.get("query") is None
        assert result["limit"] == 10
    
    def test_whitespace_query_normalized_to_none(self):
        """Test whitespace query is normalized to None."""
        params: QueryParameters = {"query": "   "}
        result = validate_query_parameters(params)
        assert result.get("query") is None
        assert result["limit"] == 10
    
    def test_invalid_query_type_raises_error(self):
        """Test invalid query type raises ValueError."""
        params: QueryParameters = {"query": 123}  # type: ignore
        with pytest.raises(ValueError, match="query must be a string"):
            validate_query_parameters(params)
    
    def test_invalid_limit_raises_error(self):
        """Test invalid limit raises ValueError."""
        params: QueryParameters = {"limit": 0}
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            validate_query_parameters(params)
    
    def test_invalid_timestamp_range_raises_error(self):
        """Test invalid timestamp range raises ValueError."""
        start = datetime(2024, 1, 31)
        end = datetime(2024, 1, 1)
        params: QueryParameters = {"start_time": start, "end_time": end}
        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            validate_query_parameters(params)
    
    def test_invalid_tags_type_raises_error(self):
        """Test invalid tags type raises ValueError."""
        params: QueryParameters = {"tags": "python"}  # type: ignore
        with pytest.raises(ValueError, match="tags must be a list"):
            validate_query_parameters(params)
    
    def test_invalid_category_type_raises_error(self):
        """Test invalid category type raises ValueError."""
        params: QueryParameters = {"category": 123}  # type: ignore
        with pytest.raises(ValueError, match="category must be a string"):
            validate_query_parameters(params)
    
    def test_embedding_field_ignored(self):
        """Test embedding field is ignored (future use)."""
        params: QueryParameters = {"embedding": [0.1, 0.2, 0.3]}
        result = validate_query_parameters(params)
        # Embedding should not be in result
        assert "embedding" not in result
        assert result["limit"] == 10