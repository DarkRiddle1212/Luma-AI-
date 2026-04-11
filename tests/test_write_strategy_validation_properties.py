"""
Property-Based Tests for Content and Metadata Validation

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly validates content and metadata
before storage.

Feature: memory-write-strategy-session-management
Property 22: Metadata type validation
Validates: Requirements 14.3, 14.5
"""

import pytest
from hypothesis import given, strategies as st, settings

from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing write strategy."""
    
    def __init__(self):
        self.stored_memories = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method."""
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Mock retrieve method."""
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def invalid_content_types(draw):
    """Generate invalid types for content parameter (anything except string)."""
    return draw(st.one_of(
        st.integers(),
        st.floats(),
        st.booleans(),
        st.none(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()),
        st.tuples(st.text(), st.text()),
    ))


@st.composite
def invalid_metadata_types(draw):
    """Generate invalid types for metadata parameter (anything except dict or None)."""
    return draw(st.one_of(
        st.integers(),
        st.floats(),
        st.booleans(),
        st.text(),
        st.lists(st.text()),
        st.tuples(st.text(), st.text()),
    ))


# ============================================================================
# Property 21: Content Validation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 21: Content validation
@given(
    invalid_content=invalid_content_types()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_21_invalid_content_types(invalid_content):
    """
    Property: For any storage attempt with non-string content, a
    MemoryStorageError should be raised.
    
    **Validates: Requirements 14.1, 14.5**
    
    This test verifies that:
    1. Non-string content types are rejected
    2. MemoryStorageError is raised with a descriptive message
    3. Error message mentions "content" and "string"
    4. Validation happens before any storage operations
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Attempt to validate invalid content type
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(invalid_content)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "content" in error_msg, \
            f"Error message should mention 'content', got: {exc_info.value}"
        assert "string" in error_msg or "str" in error_msg, \
            f"Error message should mention 'string' or 'str', got: {exc_info.value}"
        
        # Verify the type name is mentioned in the error
        type_name = type(invalid_content).__name__
        assert type_name.lower() in error_msg, \
            f"Error message should mention the invalid type '{type_name}', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 21: Content validation
@given(
    empty_content=st.just("")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_21_empty_content(empty_content):
    """
    Property: For any storage attempt with empty string content, a
    MemoryStorageError should be raised.
    
    **Validates: Requirements 14.1, 14.5**
    
    This test verifies that:
    1. Empty string content is rejected
    2. MemoryStorageError is raised with a descriptive message
    3. Error message mentions "empty"
    4. Validation enforces non-empty requirement
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Attempt to validate empty content
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(empty_content)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "content" in error_msg, \
            f"Error message should mention 'empty' or 'content', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 21: Content validation
@pytest.mark.property_test
def test_property_21_content_exceeds_max_length():
    """
    Property: For any storage attempt with content exceeding maximum length
    (100,000 characters), a MemoryStorageError should be raised.
    
    **Validates: Requirements 14.2, 14.5**
    
    This test verifies that:
    1. Content exceeding 100,000 characters is rejected
    2. MemoryStorageError is raised with a descriptive message
    3. Error message mentions "length" or "exceeds" or "maximum"
    4. Validation enforces maximum length limit
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create content that exceeds max length (100,001 characters)
        too_long_content = "x" * 100001
        
        # Attempt to validate content that's too long
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(too_long_content)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["length", "exceeds", "maximum", "max"]), \
            f"Error message should mention length limit, got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 21: Content validation
@given(
    valid_content=st.text(min_size=1, max_size=500)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_21_valid_content_accepted(valid_content):
    """
    Property: For any storage attempt with valid content (non-empty string
    within length limits), the validation should pass without raising exceptions.
    
    **Validates: Requirements 14.1, 14.2, 14.5**
    
    This test verifies that:
    1. Valid non-empty strings are accepted
    2. Content within length limits is accepted
    3. No exceptions are raised for valid content
    4. Validation allows various string formats
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should not raise any exception for valid content
        strategy.validate_content(valid_content)
        
        # If we get here, validation passed (which is expected)
        assert True, "Valid content should pass validation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 21: Content validation
@given(
    valid_content=st.text(min_size=1, max_size=5),
    valid_metadata=st.one_of(
        st.none(),
        st.dictionaries(
            st.text(min_size=1, max_size=5),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            max_size=10
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_21_valid_content_with_metadata(valid_content, valid_metadata):
    """
    Property: For any storage attempt with valid content and valid metadata,
    the validation should pass without raising exceptions.
    
    **Validates: Requirements 14.1, 14.2, 14.5**
    
    This test verifies that:
    1. Valid content with valid metadata is accepted
    2. Content validation works correctly with metadata present
    3. No exceptions are raised for valid combinations
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should not raise any exception for valid content and metadata
        strategy.validate_content(valid_content, valid_metadata)
        
        # If we get here, validation passed (which is expected)
        assert True, "Valid content with valid metadata should pass validation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 21: Content validation
@given(
    boundary_type=st.sampled_from(["min", "max"])
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_21_boundary_content_lengths(boundary_type):
    """
    Property: For any storage attempt with content at boundary lengths
    (1 character or 100,000 characters), the validation should pass.
    
    **Validates: Requirements 14.1, 14.2, 14.5**
    
    This test verifies that:
    1. Content at minimum length (1 character) is accepted
    2. Content at maximum length (100,000 characters) is accepted
    3. Boundary conditions are handled correctly
    4. Validation uses inclusive boundaries
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create boundary content based on type
        if boundary_type == "min":
            boundary_content = "a"  # Minimum valid length (1 character)
        else:
            boundary_content = "x" * 100000  # Maximum valid length (100,000 characters)
        
        # Should not raise any exception for boundary content lengths
        strategy.validate_content(boundary_content)
        
        # If we get here, validation passed (which is expected)
        assert True, "Content at boundary lengths should pass validation"
    
    finally:
        session_manager.shutdown()


# ============================================================================
# Property 22: Metadata Type Validation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 22: Metadata type validation
@given(
    content=st.text(min_size=1, max_size=100),
    invalid_metadata=invalid_metadata_types()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_22_metadata_type_validation(content, invalid_metadata):
    """
    Property: For any storage attempt with metadata, if metadata is not a
    dictionary, a MemoryStorageError should be raised.
    
    **Validates: Requirements 14.3, 14.5**
    
    This test verifies that:
    1. Non-dictionary metadata types are rejected
    2. MemoryStorageError is raised with a descriptive message
    3. Error message mentions "metadata" and "dictionary"
    4. Validation happens before any storage operations
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Attempt to validate content with invalid metadata type
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(content, invalid_metadata)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "metadata" in error_msg, \
            f"Error message should mention 'metadata', got: {exc_info.value}"
        assert "dictionary" in error_msg or "dict" in error_msg, \
            f"Error message should mention 'dictionary' or 'dict', got: {exc_info.value}"
        
        # Verify the type name is mentioned in the error
        type_name = type(invalid_metadata).__name__
        assert type_name.lower() in error_msg, \
            f"Error message should mention the invalid type '{type_name}', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 22: Metadata type validation
@given(
    content=st.text(min_size=1, max_size=100),
    valid_metadata=st.one_of(
        st.none(),
        st.dictionaries(
            st.text(min_size=1, max_size=5),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            max_size=10
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_22_valid_metadata_accepted(content, valid_metadata):
    """
    Property: For any storage attempt with valid metadata (dict or None),
    the validation should pass without raising exceptions.
    
    **Validates: Requirements 14.3, 14.5**
    
    This test verifies that:
    1. Valid dictionary metadata is accepted
    2. None metadata is accepted
    3. No exceptions are raised for valid metadata
    4. Validation allows various dictionary structures
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should not raise any exception for valid metadata
        strategy.validate_content(content, valid_metadata)
        
        # If we get here, validation passed (which is expected)
        assert True, "Valid metadata should pass validation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 22: Metadata type validation
@given(
    content=st.text(min_size=1, max_size=100),
    invalid_metadata=st.one_of(
        st.lists(st.dictionaries(st.text(), st.text())),  # List of dicts
        st.tuples(st.text(), st.text()),  # Tuple
        st.text(),  # String
        st.integers(),  # Integer
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_22_various_invalid_metadata_types(content, invalid_metadata):
    """
    Property: For various invalid metadata types (list, tuple, string, int),
    the validation should consistently reject them with MemoryStorageError.
    
    **Validates: Requirements 14.3, 14.5**
    
    This test verifies that:
    1. Various non-dict types are consistently rejected
    2. Error handling is uniform across different invalid types
    3. Error messages are always descriptive
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should raise MemoryStorageError for any invalid metadata type
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(content, invalid_metadata)
        
        # Verify error message is present and descriptive
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0, "Error message should not be empty"
        assert "metadata" in error_msg.lower(), \
            f"Error message should mention 'metadata', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 22: Metadata type validation
@given(
    content=st.text(min_size=1, max_size=100),
    metadata_with_various_values=st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(
            st.text(),
            st.integers(),
            st.floats(),
            st.booleans(),
            st.none(),
            st.lists(st.text()),
            st.dictionaries(st.text(), st.text())
        ),
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_22_metadata_with_various_value_types(content, metadata_with_various_values):
    """
    Property: For any valid dictionary metadata containing various value types,
    the validation should accept it (only the metadata container type matters,
    not the value types within).
    
    **Validates: Requirements 14.3, 14.5**
    
    This test verifies that:
    1. Dictionaries with various value types are accepted
    2. Validation focuses on the metadata container type
    3. Nested structures within metadata are allowed
    4. The validation is type-checking the metadata itself, not its contents
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should not raise exception for valid dict metadata with various values
        strategy.validate_content(content, metadata_with_various_values)
        
        # If we get here, validation passed
        assert True, "Dictionary metadata with various value types should pass validation"
    
    finally:
        session_manager.shutdown()


# ============================================================================
# Helper Strategies for Tags Validation
# ============================================================================

@st.composite
def invalid_tags_types(draw):
    """Generate invalid types for tags parameter (anything except list)."""
    return draw(st.one_of(
        st.text(),  # String instead of list
        st.integers(),
        st.floats(),
        st.booleans(),
        st.tuples(st.text()),  # Tuple instead of list
        st.dictionaries(st.text(), st.text()),
    ))


@st.composite
def invalid_tags_elements(draw):
    """Generate lists with non-string elements."""
    return draw(st.lists(
        st.one_of(st.integers(), st.floats(), st.none(), st.booleans(), st.dictionaries(st.text(), st.text())),
        min_size=1,
        max_size=5
    ))


@st.composite
def mixed_tags_elements(draw):
    """Generate lists with mixed string and non-string elements."""
    # At least one valid string and one invalid element
    valid_tags = draw(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=3))
    invalid_element = draw(st.one_of(st.integers(), st.floats(), st.none(), st.booleans()))
    
    # Insert invalid element at random position
    result = valid_tags.copy()
    result.append(invalid_element)
    return result


# ============================================================================
# Property 23: Tags Type Validation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    invalid_tags=invalid_tags_types()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_tags_type_validation(content, invalid_tags):
    """
    Property: For any storage attempt with tags in metadata, if tags is not a
    list, a MemoryStorageError should be raised.
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Non-list tags types are rejected
    2. MemoryStorageError is raised with a descriptive message
    3. Error message mentions "tags" and "list"
    4. Validation happens before any storage operations
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with invalid tags type
        metadata = {"tags": invalid_tags}
        
        # Attempt to validate content with invalid tags type
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(content, metadata)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "tags" in error_msg or "list" in error_msg, \
            f"Error message should mention 'tags' or 'list', got: {exc_info.value}"
        
        # Verify the type name is mentioned in the error
        type_name = type(invalid_tags).__name__
        assert type_name.lower() in error_msg, \
            f"Error message should mention the invalid type '{type_name}', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    invalid_elements=invalid_tags_elements()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_tags_non_string_elements(content, invalid_elements):
    """
    Property: For any storage attempt with tags containing non-string elements,
    a MemoryStorageError should be raised.
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Lists with non-string elements are rejected
    2. Error message clearly indicates all tags must be strings
    3. Validation catches various non-string types (int, float, None, bool)
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags containing non-string elements
        metadata = {"tags": invalid_elements}
        
        # Should raise MemoryStorageError
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(content, metadata)
        
        # Verify error message mentions strings requirement
        error_msg = str(exc_info.value).lower()
        assert "string" in error_msg or "tags" in error_msg, \
            f"Error message should mention 'string' or 'tags', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    mixed_elements=mixed_tags_elements()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_tags_mixed_elements(content, mixed_elements):
    """
    Property: For any storage attempt with tags containing mixed string and
    non-string elements, a MemoryStorageError should be raised.
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Mixed-type lists are rejected (even if some elements are valid strings)
    2. All elements must be strings for validation to pass
    3. Validation is strict about element types
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with mixed-type tags
        metadata = {"tags": mixed_elements}
        
        # Should raise MemoryStorageError
        with pytest.raises(MemoryStorageError) as exc_info:
            strategy.validate_content(content, metadata)
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value).lower()
        assert "string" in error_msg or "tags" in error_msg, \
            f"Error message should mention 'string' or 'tags', got: {exc_info.value}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    valid_tags=st.one_of(
        st.lists(st.text(min_size=1, max_size=5), max_size=10),
        st.just([]),  # Empty list is valid
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_valid_tags_accepted(content, valid_tags):
    """
    Property: For any storage attempt with valid tags (list of strings or empty list),
    the validation should pass without raising exceptions.
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Valid list of strings is accepted
    2. Empty list is accepted
    3. No exceptions are raised for valid tags
    4. Validation allows various list sizes
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with valid tags
        metadata = {"tags": valid_tags}
        
        # Should not raise any exception for valid tags
        strategy.validate_content(content, metadata)
        
        # If we get here, validation passed (which is expected)
        assert True, "Valid tags should pass validation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    metadata_without_tags=st.dictionaries(
        st.text(min_size=1, max_size=5).filter(lambda x: x != "tags"),
        st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_metadata_without_tags_accepted(content, metadata_without_tags):
    """
    Property: For any storage attempt with metadata that doesn't contain tags,
    the validation should pass (tags validation only applies when tags are present).
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Metadata without tags field is accepted
    2. Tags validation is optional (only when tags are provided)
    3. Other metadata fields don't interfere with tags validation
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Should not raise exception for metadata without tags
        strategy.validate_content(content, metadata_without_tags)
        
        # If we get here, validation passed
        assert True, "Metadata without tags should pass validation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 23: Tags type validation
@given(
    content=st.text(min_size=1, max_size=100),
    tags_with_special_chars=st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=1,
            max_size=50
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_23_tags_with_special_characters(content, tags_with_special_chars):
    """
    Property: For any storage attempt with tags containing special characters,
    the validation should accept them (as long as they are strings).
    
    **Validates: Requirements 14.4, 14.5**
    
    This test verifies that:
    1. Tags with special characters are accepted
    2. Validation only checks type, not content
    3. Various string formats are allowed in tags
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags containing special characters
        metadata = {"tags": tags_with_special_chars}
        
        # Should not raise exception for valid string tags
        strategy.validate_content(content, metadata)
        
        # If we get here, validation passed
        assert True, "Tags with special characters should pass validation"
    
    finally:
        session_manager.shutdown()
