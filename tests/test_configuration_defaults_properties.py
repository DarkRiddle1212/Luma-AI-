"""
Property-Based Tests for Configuration Default Application

This module implements property-based tests using Hypothesis to verify
universal correctness properties for SQLiteMemoryAdapter configuration defaults.

Feature: intent-based-memory-retrieval-enhancements
Property 6: Configuration Default Application
Validates: Requirements 6.4, 6.5, 6.6
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock
from typing import Dict, List, Optional, Any

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def device_id_strategy(draw):
    """Generate valid device IDs."""
    return draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        ))
    ))


@st.composite
def category_strategy(draw):
    """Generate valid category strings."""
    return draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        ))
    ))


@st.composite
def tags_strategy(draw):
    """Generate valid tag lists."""
    return draw(st.one_of(
        st.none(),
        st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='-_'
            )),
            min_size=0,
            max_size=5
        )
    ))


@st.composite
def metadata_strategy(draw):
    """Generate metadata dictionaries with optional category and tags."""
    has_category = draw(st.booleans())
    has_tags = draw(st.booleans())
    
    metadata = {}
    
    if has_category:
        metadata["category"] = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        )))
    
    if has_tags:
        metadata["tags"] = draw(st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='-_'
            )),
            min_size=0,
            max_size=5
        ))
    
    # Add some other metadata fields
    num_extra_fields = draw(st.integers(min_value=0, max_value=3))
    for i in range(num_extra_fields):
        key = f"field_{i}"
        value = draw(st.one_of(st.text(max_size=5), st.integers(), st.booleans()))
        metadata[key] = value
    
    return metadata


@st.composite
def content_strategy(draw):
    """Generate content strings."""
    return draw(st.text(min_size=1, max_size=200, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')
    )))


# ============================================================================
# Property 6: Configuration Default Application
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 6: Configuration Default Application
@given(
    device_id=device_id_strategy(),
    default_category=category_strategy(),
    default_tags=tags_strategy(),
    content=content_strategy(),
    metadata=st.one_of(st.none(), metadata_strategy())
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_configuration_defaults_applied_correctly(
    device_id,
    default_category,
    default_tags,
    content,
    metadata
):
    """
    Property: For any memory stored through SQLiteMemoryAdapter, when default_category
    is configured and no category is provided in metadata, the default category must be
    applied; and when default_tags are configured, they must be merged with any provided tags.
    
    **Validates: Requirements 6.4, 6.5, 6.6**
    
    This test verifies that:
    1. device_id is passed to MemoryManager.create_memory() (Requirement 6.4)
    2. default_category is applied when no category in metadata (Requirement 6.5)
    3. default_tags are merged with metadata tags (Requirement 6.6)
    4. Explicit category in metadata overrides default_category
    5. Tags are deduplicated when merged
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_memory_id"
    
    # Create adapter with configuration
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=device_id,
        default_category=default_category,
        default_tags=default_tags
    )
    
    # Store memory
    adapter.store(content, metadata)
    
    # Verify create_memory was called
    assert mock_memory_manager.create_memory.called, \
        "create_memory should be called"
    
    call_args = mock_memory_manager.create_memory.call_args
    
    # Requirement 6.4: Verify device_id was passed
    expected_device_id = device_id if device_id else "reasoning-engine"
    actual_device_id = call_args.kwargs.get("device_id")
    assert actual_device_id == expected_device_id, \
        f"device_id should be '{expected_device_id}', got '{actual_device_id}'"
    
    # Requirement 6.5: Verify category handling
    context = call_args.kwargs.get("context", {})
    expected_category = default_category if default_category else "general"
    
    if metadata and "category" in metadata:
        # Explicit category should override default
        expected_category = metadata["category"]
    
    actual_category = context.get("category")
    assert actual_category == expected_category, \
        f"category should be '{expected_category}', got '{actual_category}'"
    
    # Requirement 6.6: Verify tag merging
    actual_tags = call_args.kwargs.get("tags", [])
    expected_default_tags = default_tags if default_tags else []
    expected_metadata_tags = metadata.get("tags", []) if metadata else []
    
    # Expected tags should be the union (deduplicated)
    expected_tags_set = set(expected_default_tags + expected_metadata_tags)
    actual_tags_set = set(actual_tags)
    
    assert actual_tags_set == expected_tags_set, \
        f"tags should be merged and deduplicated: expected {expected_tags_set}, got {actual_tags_set}"
    
    # Verify no duplicate tags
    assert len(actual_tags) == len(actual_tags_set), \
        f"tags should not contain duplicates: {actual_tags}"


# Feature: intent-based-memory-retrieval-enhancements, Property 6: Configuration Default Application
@given(
    default_category=category_strategy(),
    default_tags=tags_strategy(),
    content=content_strategy()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_configuration_defaults_applied_when_metadata_is_none(
    default_category,
    default_tags,
    content
):
    """
    Property: For any memory stored with metadata=None, the adapter must apply
    default_category and default_tags from configuration.
    
    **Validates: Requirements 6.5, 6.6**
    
    This test verifies that:
    1. default_category is applied when metadata is None
    2. default_tags are applied when metadata is None
    3. No errors occur when metadata is None
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_memory_id"
    
    # Create adapter with configuration
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category=default_category,
        default_tags=default_tags
    )
    
    # Store memory with None metadata
    adapter.store(content, metadata=None)
    
    # Verify create_memory was called
    assert mock_memory_manager.create_memory.called
    
    call_args = mock_memory_manager.create_memory.call_args
    
    # Verify category
    context = call_args.kwargs.get("context", {})
    expected_category = default_category if default_category else "general"
    actual_category = context.get("category")
    assert actual_category == expected_category, \
        f"category should be '{expected_category}' when metadata is None"
    
    # Verify tags
    actual_tags = call_args.kwargs.get("tags", [])
    expected_tags = default_tags if default_tags else []
    assert set(actual_tags) == set(expected_tags), \
        f"tags should be {expected_tags} when metadata is None"


# Feature: intent-based-memory-retrieval-enhancements, Property 6: Configuration Default Application
@given(
    default_tags=tags_strategy(),
    metadata_tags=tags_strategy(),
    content=content_strategy()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_tag_merging_deduplication(
    default_tags,
    metadata_tags,
    content
):
    """
    Property: For any combination of default_tags and metadata tags, the adapter
    must merge them and remove duplicates.
    
    **Validates: Requirement 6.6**
    
    This test verifies that:
    1. Tags from both sources are included
    2. Duplicate tags are removed
    3. Order doesn't matter (set equality)
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_memory_id"
    
    # Create adapter with default_tags
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=default_tags
    )
    
    # Create metadata with tags
    metadata = {"tags": metadata_tags} if metadata_tags is not None else None
    
    # Store memory
    adapter.store(content, metadata)
    
    # Verify tags were merged and deduplicated
    call_args = mock_memory_manager.create_memory.call_args
    actual_tags = call_args.kwargs.get("tags", [])
    
    expected_default = default_tags if default_tags else []
    expected_metadata = metadata_tags if metadata_tags else []
    expected_tags_set = set(expected_default + expected_metadata)
    
    assert set(actual_tags) == expected_tags_set, \
        f"tags should be merged: expected {expected_tags_set}, got {set(actual_tags)}"
    
    # Verify no duplicates
    assert len(actual_tags) == len(set(actual_tags)), \
        f"tags should not contain duplicates: {actual_tags}"


# Feature: intent-based-memory-retrieval-enhancements, Property 6: Configuration Default Application
@given(
    default_category=category_strategy(),
    explicit_category=category_strategy(),
    content=content_strategy()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_explicit_category_overrides_default(
    default_category,
    explicit_category,
    content
):
    """
    Property: For any memory stored with an explicit category in metadata,
    the explicit category must override the default_category.
    
    **Validates: Requirement 6.5**
    
    This test verifies that:
    1. Explicit category in metadata takes precedence
    2. default_category is ignored when explicit category is provided
    """
    # Skip if explicit_category is None (we're testing override behavior)
    if explicit_category is None:
        return
    
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_memory_id"
    
    # Create adapter with default_category
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category=default_category
    )
    
    # Store memory with explicit category
    metadata = {"category": explicit_category}
    adapter.store(content, metadata)
    
    # Verify explicit category was used
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context", {})
    actual_category = context.get("category")
    
    assert actual_category == explicit_category, \
        f"explicit category '{explicit_category}' should override default '{default_category}'"


# Feature: intent-based-memory-retrieval-enhancements, Property 6: Configuration Default Application
@given(
    device_id=device_id_strategy(),
    default_category=category_strategy(),
    default_tags=tags_strategy(),
    content=content_strategy(),
    metadata=metadata_strategy()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_other_metadata_fields_preserved(
    device_id,
    default_category,
    default_tags,
    content,
    metadata
):
    """
    Property: For any memory stored with additional metadata fields beyond
    category and tags, those fields must be preserved in the context.
    
    **Validates: Requirement 6.6 (implicit - preserve other fields)**
    
    This test verifies that:
    1. Configuration defaults don't interfere with other metadata fields
    2. All non-tag, non-category fields are preserved in context
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "test_memory_id"
    
    # Create adapter with configuration
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=device_id,
        default_category=default_category,
        default_tags=default_tags
    )
    
    # Store memory
    adapter.store(content, metadata)
    
    # Verify other metadata fields are preserved
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context", {})
    
    if metadata:
        for key, value in metadata.items():
            if key not in ["tags", "category"]:
                assert key in context, \
                    f"metadata field '{key}' should be preserved in context"
                assert context[key] == value, \
                    f"metadata field '{key}' value should be preserved: expected {value}, got {context[key]}"
