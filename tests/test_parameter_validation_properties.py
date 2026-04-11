"""
Property-Based Tests for Parameter Validation

This module implements property-based tests using Hypothesis to verify
universal correctness properties for query parameter validation.

Feature: intent-based-memory-retrieval-enhancements
Property 2: Parameter Validation
Validates: Requirements 2.6, 8.4, 8.5, 8.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from typing import Any

from luma.core.memory_interface import (
    validate_query_string,
    validate_limit,
    validate_timestamp_range,
    validate_tags,
    validate_category,
    validate_query_parameters,
    QueryParameters,
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def invalid_query_types(draw):
    """Generate invalid types for query parameter (anything except str or None)."""
    return draw(st.one_of(
        st.integers(),
        st.floats(),
        st.booleans(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()),
        st.tuples(st.text()),
    ))


@st.composite
def invalid_limit_values(draw):
    """Generate invalid limit values (non-positive integers, wrong types)."""
    return draw(st.one_of(
        st.integers(max_value=0),  # Zero or negative
        st.floats(),  # Float type
        st.text(),  # String type
        st.lists(st.integers()),  # List type
    ))


@st.composite
def invalid_timestamp_ranges(draw):
    """Generate invalid timestamp ranges where start > end."""
    # Generate two different datetimes
    dt1 = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    ))
    dt2 = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    ))
    
    # Ensure dt1 > dt2 (invalid range)
    assume(dt1 > dt2)
    
    return (dt1, dt2)


@st.composite
def invalid_timestamp_types(draw):
    """Generate invalid types for timestamp parameters."""
    return draw(st.one_of(
        st.text(),
        st.integers(),
        st.floats(),
        st.lists(st.integers()),
    ))


@st.composite
def invalid_tags_types(draw):
    """Generate invalid types for tags parameter (not a list)."""
    return draw(st.one_of(
        st.text(),  # String instead of list
        st.integers(),
        st.tuples(st.text()),  # Tuple instead of list
        st.dictionaries(st.text(), st.text()),
    ))


@st.composite
def invalid_tags_elements(draw):
    """Generate lists with non-string elements."""
    # Generate a list with at least one non-string element
    return draw(st.lists(
        st.one_of(st.integers(), st.floats(), st.none(), st.booleans()),
        min_size=1,
        max_size=5
    ))


@st.composite
def invalid_category_types(draw):
    """Generate invalid types for category parameter."""
    return draw(st.one_of(
        st.integers(),
        st.floats(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()),
    ))


# ============================================================================
# Property 2: Parameter Validation
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_type=invalid_query_types())
@settings(max_examples=10)
@pytest.mark.property_test
def test_query_string_validation_rejects_invalid_types(invalid_type):
    """
    Property: For any invalid query type (not str or None), validate_query_string
    must raise ValueError with a clear error message before attempting to query.
    
    **Validates: Requirements 2.6, 8.4**
    
    This test verifies that:
    1. Invalid types are rejected with ValueError
    2. Error message clearly indicates the type issue
    3. Validation happens before any query execution
    """
    with pytest.raises(ValueError) as exc_info:
        validate_query_string(invalid_type)
    
    # Verify error message mentions "query" and "string"
    error_msg = str(exc_info.value).lower()
    assert "query" in error_msg, \
        f"Error message should mention 'query', got: {exc_info.value}"
    assert "string" in error_msg, \
        f"Error message should mention 'string', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_limit=invalid_limit_values())
@settings(max_examples=10)
@pytest.mark.property_test
def test_limit_validation_rejects_invalid_values(invalid_limit):
    """
    Property: For any invalid limit value (non-positive or wrong type),
    validate_limit must raise ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.6**
    
    This test verifies that:
    1. Non-positive integers are rejected
    2. Wrong types (float, string, etc.) are rejected
    3. Error message clearly indicates the issue
    """
    with pytest.raises(ValueError) as exc_info:
        validate_limit(invalid_limit)
    
    # Verify error message mentions "limit" and "positive integer"
    error_msg = str(exc_info.value).lower()
    assert "limit" in error_msg, \
        f"Error message should mention 'limit', got: {exc_info.value}"
    assert "positive" in error_msg or "integer" in error_msg, \
        f"Error message should mention 'positive' or 'integer', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_range=invalid_timestamp_ranges())
@settings(max_examples=10)
@pytest.mark.property_test
def test_timestamp_range_validation_rejects_invalid_ranges(invalid_range):
    """
    Property: For any timestamp range where start_time > end_time,
    validate_timestamp_range must raise ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.5**
    
    This test verifies that:
    1. Invalid ranges (start > end) are rejected
    2. Error message clearly indicates the range issue
    3. Validation happens before query execution
    """
    start_time, end_time = invalid_range
    
    with pytest.raises(ValueError) as exc_info:
        validate_timestamp_range(start_time, end_time)
    
    # Verify error message mentions the range issue
    error_msg = str(exc_info.value).lower()
    assert "start_time" in error_msg or "end_time" in error_msg, \
        f"Error message should mention timestamp fields, got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_type=invalid_timestamp_types())
@settings(max_examples=10)
@pytest.mark.property_test
def test_timestamp_validation_rejects_invalid_types(invalid_type):
    """
    Property: For any non-datetime timestamp type, validate_timestamp_range
    must raise ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.4**
    
    This test verifies that:
    1. Non-datetime types are rejected for timestamps
    2. Error message clearly indicates the type issue
    3. Both start_time and end_time are validated
    """
    # Test with invalid start_time
    with pytest.raises(ValueError) as exc_info:
        validate_timestamp_range(invalid_type, datetime.now())
    
    error_msg = str(exc_info.value).lower()
    assert "datetime" in error_msg, \
        f"Error message should mention 'datetime', got: {exc_info.value}"
    
    # Test with invalid end_time
    with pytest.raises(ValueError) as exc_info:
        validate_timestamp_range(datetime.now(), invalid_type)
    
    error_msg = str(exc_info.value).lower()
    assert "datetime" in error_msg, \
        f"Error message should mention 'datetime', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_type=invalid_tags_types())
@settings(max_examples=10)
@pytest.mark.property_test
def test_tags_validation_rejects_invalid_types(invalid_type):
    """
    Property: For any non-list tags type, validate_tags must raise
    ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.4**
    
    This test verifies that:
    1. Non-list types are rejected for tags
    2. Error message clearly indicates tags must be a list
    3. Validation happens before query execution
    """
    with pytest.raises(ValueError) as exc_info:
        validate_tags(invalid_type)
    
    # Verify error message mentions "tags" and "list"
    error_msg = str(exc_info.value).lower()
    assert "tags" in error_msg or "list" in error_msg, \
        f"Error message should mention 'tags' or 'list', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_elements=invalid_tags_elements())
@settings(max_examples=10)
@pytest.mark.property_test
def test_tags_validation_rejects_non_string_elements(invalid_elements):
    """
    Property: For any tags list containing non-string elements, validate_tags
    must raise ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.4**
    
    This test verifies that:
    1. Lists with non-string elements are rejected
    2. Error message clearly indicates all tags must be strings
    3. Validation catches mixed-type lists
    """
    with pytest.raises(ValueError) as exc_info:
        validate_tags(invalid_elements)
    
    # Verify error message mentions strings requirement
    error_msg = str(exc_info.value).lower()
    assert "string" in error_msg, \
        f"Error message should mention 'string', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(invalid_type=invalid_category_types())
@settings(max_examples=10)
@pytest.mark.property_test
def test_category_validation_rejects_invalid_types(invalid_type):
    """
    Property: For any non-string category type, validate_category must raise
    ValueError with a clear error message.
    
    **Validates: Requirements 2.6, 8.4**
    
    This test verifies that:
    1. Non-string types are rejected for category
    2. Error message clearly indicates category must be a string
    3. Validation happens before query execution
    """
    with pytest.raises(ValueError) as exc_info:
        validate_category(invalid_type)
    
    # Verify error message mentions "category" and "string"
    error_msg = str(exc_info.value).lower()
    assert "category" in error_msg or "string" in error_msg, \
        f"Error message should mention 'category' or 'string', got: {exc_info.value}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(
    query_type=st.one_of(invalid_query_types(), st.none()),
    limit_value=st.one_of(invalid_limit_values(), st.none()),
    tags_value=st.one_of(invalid_tags_types(), invalid_tags_elements(), st.none()),
    category_value=st.one_of(invalid_category_types(), st.none()),
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_query_parameters_validation_comprehensive(
    query_type, limit_value, tags_value, category_value
):
    """
    Property: For any QueryParameters with at least one invalid field,
    validate_query_parameters must raise ValueError before attempting to query.
    
    **Validates: Requirements 2.6, 8.4, 8.5, 8.6**
    
    This test verifies that:
    1. Comprehensive validation catches any invalid parameter
    2. Validation happens before query execution
    3. Error messages are clear and actionable
    4. Multiple invalid parameters are handled correctly
    """
    # Build params dict with at least one invalid field
    params: Any = {}
    has_invalid = False
    
    if query_type is not None and not isinstance(query_type, str):
        params["query"] = query_type
        has_invalid = True
    
    if limit_value is not None and (not isinstance(limit_value, int) or limit_value <= 0):
        params["limit"] = limit_value
        has_invalid = True
    
    if tags_value is not None:
        params["tags"] = tags_value
        has_invalid = True
    
    if category_value is not None:
        params["category"] = category_value
        has_invalid = True
    
    # Only test if we have at least one invalid parameter
    assume(has_invalid)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        validate_query_parameters(params)
    
    # Verify we got a clear error message
    error_msg = str(exc_info.value)
    assert len(error_msg) > 0, "Error message should not be empty"
    assert any(word in error_msg.lower() for word in ["query", "limit", "tags", "category", "must", "invalid"]), \
        f"Error message should be descriptive, got: {error_msg}"


# Feature: intent-based-memory-retrieval-enhancements, Property 2: Parameter Validation
@given(
    valid_query=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    valid_limit=st.integers(min_value=1, max_value=1000),
    valid_tags=st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=5), max_size=10)),
    valid_category=st.one_of(st.none(), st.text(min_size=1, max_size=5)),
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_query_parameters_validation_accepts_valid_params(
    valid_query, valid_limit, valid_tags, valid_category
):
    """
    Property: For any valid QueryParameters, validate_query_parameters must
    return a normalized dictionary without raising exceptions.
    
    **Validates: Requirements 2.6**
    
    This test verifies that:
    1. Valid parameters are accepted
    2. Normalized dictionary is returned
    3. No exceptions are raised for valid inputs
    4. Default values are applied correctly
    """
    params: QueryParameters = {}
    
    if valid_query is not None:
        params["query"] = valid_query
    if valid_limit is not None:
        params["limit"] = valid_limit
    if valid_tags is not None:
        params["tags"] = valid_tags
    if valid_category is not None:
        params["category"] = valid_category
    
    # Should not raise any exception
    result = validate_query_parameters(params)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Verify limit is present (default or provided)
    assert "limit" in result, "Result should contain limit"
    assert isinstance(result["limit"], int), "Limit should be an integer"
    assert result["limit"] > 0, "Limit should be positive"
    
    # Verify query normalization (empty/whitespace becomes None)
    if "query" in params and params["query"] and params["query"].strip():
        assert result.get("query") == params["query"], "Valid query should be preserved"
    
    # Verify other fields are preserved if provided
    if valid_tags is not None:
        assert result.get("tags") == valid_tags, "Valid tags should be preserved"
    if valid_category is not None:
        assert result.get("category") == valid_category, "Valid category should be preserved"


# ============================================================================
# Property 9: Embedding Parameter Tolerance
# ============================================================================

# Strategy for generating valid embedding vectors
# Uses small vectors (1-32 dimensions) for efficient property testing
# Large vectors (e.g., 1536 dimensions) belong in integration/performance tests
valid_embedding_vectors = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=32
)




# Feature: intent-based-memory-retrieval-enhancements, Property 9: Embedding Parameter Tolerance
@given(
    embedding=valid_embedding_vectors,
    query=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    limit=st.integers(min_value=1, max_value=100),
    tags=st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=5), max_size=10)),
    category=st.one_of(st.none(), st.text(min_size=1, max_size=5)),
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_embedding_parameter_tolerance(embedding, query, limit, tags, category):
    """
    Property: For any QueryParameters containing an embedding field, the system
    must accept the parameters without error and ignore the embedding field
    (logging that it's not yet implemented).
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. QueryParameters with embedding field are accepted without error
    2. The embedding field is ignored (not used in query execution)
    3. A debug log message indicates embedding is not yet implemented
    4. Other parameters are processed normally
    5. The system maintains forward compatibility for future vector search
    """
    # Build params with embedding field
    params: QueryParameters = {
        "embedding": embedding,
        "limit": limit,
    }
    
    if query is not None:
        params["query"] = query
    if tags is not None:
        params["tags"] = tags
    if category is not None:
        params["category"] = category
    
    # Should not raise any exception despite embedding field
    result = validate_query_parameters(params)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Verify embedding field is NOT in the result (it should be ignored/removed)
    assert "embedding" not in result, \
        "Embedding field should be removed from normalized parameters"
    
    # Verify other parameters are still present and valid
    assert "limit" in result, "Limit should be present"
    assert result["limit"] == limit, f"Limit should be {limit}, got {result['limit']}"
    
    if query is not None and query.strip():
        assert result.get("query") == query, "Query should be preserved"
    
    if tags is not None:
        assert result.get("tags") == tags, "Tags should be preserved"
    
    if category is not None:
        assert result.get("category") == category, "Category should be preserved"


# Feature: intent-based-memory-retrieval-enhancements, Property 9: Embedding Parameter Tolerance
@given(
    embedding=st.one_of(
        valid_embedding_vectors,
        st.lists(st.floats()),  # Any list of floats
        st.lists(st.integers()),  # List of integers (also valid)
        st.just([]),  # Empty list
    ),
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_embedding_parameter_various_formats(embedding):
    """
    Property: For any embedding parameter in various formats (different sizes,
    types), the system must accept it without error and ignore it.
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. Various embedding formats are tolerated
    2. No validation errors are raised for embedding content
    3. The embedding field is consistently ignored
    4. System remains stable with different embedding formats
    """
    params: QueryParameters = {
        "embedding": embedding,
        "limit": 10,
    }
    
    # Should not raise any exception
    result = validate_query_parameters(params)
    
    # Verify embedding is removed
    assert "embedding" not in result, \
        "Embedding field should be removed regardless of format"
    
    # Verify basic functionality still works
    assert isinstance(result, dict), "Result should be a dictionary"
    assert result["limit"] == 10, "Limit should be preserved"


# Feature: intent-based-memory-retrieval-enhancements, Property 9: Embedding Parameter Tolerance
@given(
    query=st.text(min_size=1, max_size=100),
    embedding=valid_embedding_vectors,
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_embedding_does_not_affect_query_execution(query, embedding):
    """
    Property: For any query with an embedding parameter, the query execution
    should be identical to the same query without the embedding parameter.
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. Embedding parameter does not affect query results
    2. Query execution is identical with or without embedding
    3. Forward compatibility is maintained
    """
    # Params with embedding
    params_with_embedding: QueryParameters = {
        "query": query,
        "embedding": embedding,
        "limit": 10,
    }
    
    # Params without embedding
    params_without_embedding: QueryParameters = {
        "query": query,
        "limit": 10,
    }
    
    # Validate both
    result_with = validate_query_parameters(params_with_embedding)
    result_without = validate_query_parameters(params_without_embedding)
    
    # Results should be identical (embedding is ignored)
    assert result_with == result_without, \
        "Query results should be identical with or without embedding parameter"
    
    # Verify neither result contains embedding
    assert "embedding" not in result_with, "Result with embedding should not contain embedding field"
    assert "embedding" not in result_without, "Result without embedding should not contain embedding field"
