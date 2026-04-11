"""
Unit Tests for SQLiteMemoryAdapter Configuration

This module tests the configuration features of SQLiteMemoryAdapter including
device_id, default_category, and default_tags parameters.

Feature: intent-based-memory-retrieval-enhancements
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import MemoryStorageError


# ============================================================================
# Constructor Configuration Tests
# ============================================================================

def test_adapter_constructor_with_all_defaults():
    """
    Test adapter constructor with no configuration parameters.
    
    Validates: Requirements 6.1, 6.2, 6.3
    
    When no configuration parameters are provided, the adapter should use
    default values: device_id="reasoning-engine", default_category="general",
    default_tags=[].
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create adapter with no configuration
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Verify default values are set
    assert adapter.device_id == "reasoning-engine", \
        "Default device_id should be 'reasoning-engine'"
    assert adapter.default_category == "general", \
        "Default default_category should be 'general'"
    assert adapter.default_tags == [], \
        "Default default_tags should be empty list"
    assert adapter.memory_manager is mock_memory_manager, \
        "memory_manager should be stored"


def test_adapter_constructor_with_custom_device_id():
    """
    Test adapter constructor with custom device_id.
    
    Validates: Requirement 6.1
    
    When device_id is provided, the adapter should use it instead of the default.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create adapter with custom device_id
    custom_device_id = "mobile-device-123"
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=custom_device_id
    )
    
    # Verify custom device_id is set
    assert adapter.device_id == custom_device_id, \
        f"device_id should be '{custom_device_id}'"
    # Verify other defaults remain
    assert adapter.default_category == "general"
    assert adapter.default_tags == []


def test_adapter_constructor_with_custom_default_category():
    """
    Test adapter constructor with custom default_category.
    
    Validates: Requirement 6.2
    
    When default_category is provided, the adapter should use it instead of the default.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create adapter with custom default_category
    custom_category = "education"
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category=custom_category
    )
    
    # Verify custom default_category is set
    assert adapter.default_category == custom_category, \
        f"default_category should be '{custom_category}'"
    # Verify other defaults remain
    assert adapter.device_id == "reasoning-engine"
    assert adapter.default_tags == []


def test_adapter_constructor_with_custom_default_tags():
    """
    Test adapter constructor with custom default_tags.
    
    Validates: Requirement 6.3
    
    When default_tags is provided, the adapter should use it instead of the default.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create adapter with custom default_tags
    custom_tags = ["system", "automated", "v2"]
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=custom_tags
    )
    
    # Verify custom default_tags is set
    assert adapter.default_tags == custom_tags, \
        f"default_tags should be {custom_tags}"
    # Verify other defaults remain
    assert adapter.device_id == "reasoning-engine"
    assert adapter.default_category == "general"


def test_adapter_constructor_with_all_custom_config():
    """
    Test adapter constructor with all configuration parameters.
    
    Validates: Requirements 6.1, 6.2, 6.3
    
    When all configuration parameters are provided, the adapter should use
    all custom values.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    
    # Create adapter with all custom configuration
    custom_device_id = "server-node-42"
    custom_category = "analytics"
    custom_tags = ["production", "monitored"]
    
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=custom_device_id,
        default_category=custom_category,
        default_tags=custom_tags
    )
    
    # Verify all custom values are set
    assert adapter.device_id == custom_device_id
    assert adapter.default_category == custom_category
    assert adapter.default_tags == custom_tags


def test_adapter_constructor_rejects_none_memory_manager():
    """
    Test that adapter constructor raises ValueError for None memory_manager.
    
    Validates: Requirements 6.1, 6.2, 6.3
    
    The adapter requires a valid MemoryManager instance and should reject None.
    """
    with pytest.raises(ValueError) as exc_info:
        SQLiteMemoryAdapter(None)
    
    assert "memory_manager cannot be None" in str(exc_info.value)


# ============================================================================
# Default Application in store() Method Tests
# ============================================================================

def test_store_applies_device_id():
    """
    Test that store() passes device_id to MemoryManager.create_memory().
    
    Validates: Requirement 6.4
    
    When device_id is configured, store() should pass it to create_memory().
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with custom device_id
    custom_device_id = "test-device-456"
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=custom_device_id
    )
    
    # Store memory
    content = "Test content"
    metadata = {"tags": ["test"], "category": "test"}
    adapter.store(content, metadata)
    
    # Verify device_id was passed to create_memory
    call_args = mock_memory_manager.create_memory.call_args
    assert call_args.kwargs.get("device_id") == custom_device_id, \
        f"device_id should be '{custom_device_id}'"


def test_store_applies_default_category_when_missing():
    """
    Test that store() applies default_category when no category in metadata.
    
    Validates: Requirement 6.5
    
    When metadata doesn't include a category, store() should use default_category.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with custom default_category
    custom_category = "work"
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category=custom_category
    )
    
    # Store memory without category in metadata
    content = "Test content"
    metadata = {"tags": ["test"]}  # No category
    adapter.store(content, metadata)
    
    # Verify default_category was applied in context
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context")
    assert context.get("category") == custom_category, \
        f"category should be '{custom_category}' when not in metadata"


def test_store_preserves_explicit_category():
    """
    Test that store() preserves explicit category from metadata.
    
    Validates: Requirement 6.5
    
    When metadata includes a category, store() should use it instead of default.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with default_category
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category="general"
    )
    
    # Store memory with explicit category
    content = "Test content"
    explicit_category = "personal"
    metadata = {"tags": ["test"], "category": explicit_category}
    adapter.store(content, metadata)
    
    # Verify explicit category was used
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context")
    assert context.get("category") == explicit_category, \
        f"category should be '{explicit_category}' from metadata"


def test_store_applies_default_category_when_metadata_is_none():
    """
    Test that store() applies default_category when metadata is None.
    
    Validates: Requirement 6.5
    
    When no metadata is provided, store() should use default_category.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with custom default_category
    custom_category = "system"
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category=custom_category
    )
    
    # Store memory with no metadata
    content = "Test content"
    adapter.store(content, metadata=None)
    
    # Verify default_category was applied
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context")
    assert context.get("category") == custom_category, \
        f"category should be '{custom_category}' when metadata is None"


# ============================================================================
# Tag Merging Behavior Tests
# ============================================================================

def test_store_merges_default_tags_with_metadata_tags():
    """
    Test that store() merges default_tags with metadata tags.
    
    Validates: Requirement 6.6
    
    When both default_tags and metadata tags are provided, store() should
    merge them (union) and pass the combined list to create_memory().
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with default_tags
    default_tags = ["system", "automated"]
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=default_tags
    )
    
    # Store memory with additional tags
    content = "Test content"
    metadata_tags = ["user-generated", "important"]
    metadata = {"tags": metadata_tags, "category": "test"}
    adapter.store(content, metadata)
    
    # Verify tags were merged
    call_args = mock_memory_manager.create_memory.call_args
    merged_tags = call_args.kwargs.get("tags")
    
    # Check that all tags are present (order doesn't matter, duplicates removed)
    assert set(merged_tags) == set(default_tags + metadata_tags), \
        f"tags should be merged: {default_tags + metadata_tags}"


def test_store_uses_only_default_tags_when_metadata_has_no_tags():
    """
    Test that store() uses only default_tags when metadata has no tags.
    
    Validates: Requirement 6.6
    
    When metadata doesn't include tags, store() should use default_tags.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with default_tags
    default_tags = ["system", "automated"]
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=default_tags
    )
    
    # Store memory without tags in metadata
    content = "Test content"
    metadata = {"category": "test"}  # No tags
    adapter.store(content, metadata)
    
    # Verify only default_tags were used
    call_args = mock_memory_manager.create_memory.call_args
    tags = call_args.kwargs.get("tags")
    assert set(tags) == set(default_tags), \
        f"tags should be {default_tags} when not in metadata"


def test_store_uses_only_default_tags_when_metadata_is_none():
    """
    Test that store() uses only default_tags when metadata is None.
    
    Validates: Requirement 6.6
    
    When no metadata is provided, store() should use default_tags.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with default_tags
    default_tags = ["system", "automated"]
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=default_tags
    )
    
    # Store memory with no metadata
    content = "Test content"
    adapter.store(content, metadata=None)
    
    # Verify only default_tags were used
    call_args = mock_memory_manager.create_memory.call_args
    tags = call_args.kwargs.get("tags")
    assert set(tags) == set(default_tags), \
        f"tags should be {default_tags} when metadata is None"


def test_store_removes_duplicate_tags():
    """
    Test that store() removes duplicate tags when merging.
    
    Validates: Requirement 6.6
    
    When default_tags and metadata tags have duplicates, store() should
    deduplicate them.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with default_tags
    default_tags = ["system", "important"]
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_tags=default_tags
    )
    
    # Store memory with overlapping tags
    content = "Test content"
    metadata_tags = ["important", "user-generated"]  # "important" is duplicate
    metadata = {"tags": metadata_tags, "category": "test"}
    adapter.store(content, metadata)
    
    # Verify tags were deduplicated
    call_args = mock_memory_manager.create_memory.call_args
    merged_tags = call_args.kwargs.get("tags")
    
    # Check that duplicates were removed
    expected_unique_tags = {"system", "important", "user-generated"}
    assert set(merged_tags) == expected_unique_tags, \
        f"tags should be deduplicated: {expected_unique_tags}"
    assert len(merged_tags) == len(expected_unique_tags), \
        "tags list should not contain duplicates"


def test_store_uses_empty_tags_when_no_defaults_and_no_metadata_tags():
    """
    Test that store() uses empty tags when neither default nor metadata tags exist.
    
    Validates: Requirement 6.7
    
    When no default_tags are configured and metadata has no tags, store()
    should pass an empty list to create_memory().
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with no default_tags (uses default empty list)
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Store memory without tags
    content = "Test content"
    metadata = {"category": "test"}  # No tags
    adapter.store(content, metadata)
    
    # Verify empty tags were used
    call_args = mock_memory_manager.create_memory.call_args
    tags = call_args.kwargs.get("tags")
    assert tags == [], \
        "tags should be empty list when no defaults and no metadata tags"


# ============================================================================
# Integration Tests - Multiple Configuration Options
# ============================================================================

def test_store_applies_all_configuration_defaults():
    """
    Test that store() applies all configuration defaults together.
    
    Validates: Requirements 6.4, 6.5, 6.6, 6.7
    
    When all configuration options are set, store() should apply them all
    correctly in a single operation.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with all configuration options
    custom_device_id = "integration-test-device"
    custom_category = "integration"
    custom_tags = ["config-test", "integration"]
    
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id=custom_device_id,
        default_category=custom_category,
        default_tags=custom_tags
    )
    
    # Store memory with minimal metadata
    content = "Integration test content"
    metadata = {"tags": ["user-tag"]}  # No category, has one tag
    adapter.store(content, metadata)
    
    # Verify all configuration was applied
    call_args = mock_memory_manager.create_memory.call_args
    
    # Check device_id
    assert call_args.kwargs.get("device_id") == custom_device_id
    
    # Check category (should use default since not in metadata)
    context = call_args.kwargs.get("context")
    assert context.get("category") == custom_category
    
    # Check tags (should be merged)
    tags = call_args.kwargs.get("tags")
    expected_tags = set(custom_tags + ["user-tag"])
    assert set(tags) == expected_tags


def test_store_preserves_other_metadata_fields():
    """
    Test that store() preserves other metadata fields beyond tags and category.
    
    Validates: Requirement 6.7
    
    When metadata contains additional fields, store() should preserve them
    in the context while applying configuration defaults.
    """
    # Create mock MemoryManager
    mock_memory_manager = Mock()
    mock_memory_manager.create_memory.return_value = "memory_123"
    
    # Create adapter with configuration
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        default_category="general",
        default_tags=["system"]
    )
    
    # Store memory with extra metadata fields
    content = "Test content"
    metadata = {
        "tags": ["user"],
        "category": "custom",
        "source": "api",
        "priority": "high",
        "user_id": "user_123"
    }
    adapter.store(content, metadata)
    
    # Verify extra fields were preserved in context
    call_args = mock_memory_manager.create_memory.call_args
    context = call_args.kwargs.get("context")
    
    assert context.get("source") == "api"
    assert context.get("priority") == "high"
    assert context.get("user_id") == "user_123"
    assert context.get("category") == "custom"  # Explicit category preserved
