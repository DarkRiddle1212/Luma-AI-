"""
Property-Based Tests for Default Category Application

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly applies default_category when
no explicit category is provided in metadata.

Feature: memory-write-strategy-session-management
Property 19: Default category application
Validates: Requirements 6.6, 8.5
"""

import pytest
from hypothesis import given, strategies as st, settings

from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing write strategy."""
    
    def __init__(self, default_category=None, default_tags=None):
        self.stored_memories = []
        self.default_category = default_category or "general"
        self.default_tags = default_tags or []
    
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
# Property 19: Default Category Application
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 19: Default category application
@given(
    default_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_19_default_category_applied_when_missing(default_category):
    """
    Property: For any memory stored without an explicit category through
    an adapter with default_category configured, the memory should have
    the default_category applied.
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. When metadata has no category field, default_category is applied
    2. The default_category is normalized (trimmed and lowercased)
    3. The result always has a category field
    4. The category matches the configured default
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_category=default_category)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata WITHOUT category field
        metadata = {"tags": ["test"], "source": "user"}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify default_category is applied and normalized
        expected_category = default_category.strip().casefold()
        assert "category" in normalized, \
            "Normalized metadata should have a category field"
        assert normalized["category"] == expected_category, \
            f"Category should be default_category (normalized). Expected: '{expected_category}', Got: '{normalized['category']}'"
        
        # Verify category is not empty
        assert len(normalized["category"]) > 0, \
            "Category should not be empty after applying default"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 19: Default category application
@given(
    default_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_19_default_category_applied_when_empty(default_category):
    """
    Property: For any memory with an empty or whitespace-only category,
    the default_category should be applied.
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. Empty string category is replaced with default_category
    2. Whitespace-only category is replaced with default_category
    3. The default_category is normalized
    4. No empty categories remain after normalization
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_category=default_category)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Test with empty string
        metadata_empty = {"category": "", "tags": ["test"]}
        normalized_empty = strategy.normalize_metadata(metadata_empty)
        
        expected_category = default_category.strip().casefold()
        assert normalized_empty["category"] == expected_category, \
            f"Empty category should be replaced with default_category. Expected: '{expected_category}', Got: '{normalized_empty['category']}'"
        
        # Test with whitespace-only string
        metadata_whitespace = {"category": "   \t\n  ", "tags": ["test"]}
        normalized_whitespace = strategy.normalize_metadata(metadata_whitespace)
        
        assert normalized_whitespace["category"] == expected_category, \
            f"Whitespace-only category should be replaced with default_category. Expected: '{expected_category}', Got: '{normalized_whitespace['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 19: Default category application
@given(
    default_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    ),
    explicit_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_19_explicit_category_not_overridden(default_category, explicit_category):
    """
    Property: For any memory with an explicit category, the default_category
    should NOT override it.
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. Explicit categories are preserved
    2. Default category is only used when no category provided
    3. Explicit category is still normalized
    4. Default doesn't interfere with explicit values
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_category=default_category)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata WITH explicit category
        metadata = {"category": explicit_category, "tags": ["test"]}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify explicit category is preserved (but normalized)
        expected_category = explicit_category.strip().casefold()
        assert normalized["category"] == expected_category, \
            f"Explicit category should be preserved (normalized). Expected: '{expected_category}', Got: '{normalized['category']}'"
        
        # Verify it's NOT the default category (unless they happen to be the same after normalization)
        default_normalized = default_category.strip().casefold()
        if expected_category != default_normalized:
            assert normalized["category"] != default_normalized, \
                f"Explicit category should not be overridden by default. Got: '{normalized['category']}', Default: '{default_normalized}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 19: Default category application
@given(
    default_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    ),
    metadata_dict=st.dictionaries(
        st.text(min_size=1, max_size=5).filter(lambda x: x != "category"),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_19_default_category_preserves_other_metadata(default_category, metadata_dict):
    """
    Property: For any metadata without a category, applying default_category
    should not affect other metadata fields.
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. Other metadata fields remain unchanged
    2. Only the category field is added/modified
    3. Metadata structure is preserved
    4. No side effects on other fields
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_category=default_category)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata without category
        metadata = metadata_dict.copy()
        
        # Store original values
        original_fields = metadata.copy()
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify default_category is applied
        expected_category = default_category.strip().casefold()
        assert normalized["category"] == expected_category, \
            f"Default category should be applied. Expected: '{expected_category}', Got: '{normalized['category']}'"
        
        # Verify other fields are preserved (excluding timestamp and session_id which are added)
        for key, value in original_fields.items():
            if key not in ["timestamp", "session_id", "tags"]:  # tags might be normalized
                assert key in normalized, f"Field '{key}' should be preserved"
                assert normalized[key] == value, \
                    f"Field '{key}' should not be modified. Expected: {value}, Got: {normalized[key]}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 19: Default category application
@given(
    default_category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_19_default_category_idempotent(default_category):
    """
    Property: For any metadata, applying default_category normalization
    multiple times should produce the same result (idempotent).
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. Normalization is idempotent
    2. Applying default_category twice gives same result
    3. No accumulation of defaults
    4. Stable normalization behavior
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_category=default_category)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata without category
        metadata = {"tags": ["test"], "source": "user"}
        
        # Normalize once
        normalized_once = strategy.normalize_metadata(metadata)
        
        # Normalize again
        normalized_twice = strategy.normalize_metadata(normalized_once)
        
        # Verify both normalizations produce the same category
        assert normalized_once["category"] == normalized_twice["category"], \
            f"Normalization should be idempotent. First: '{normalized_once['category']}', Second: '{normalized_twice['category']}'"
        
        # Verify the category is the expected default
        expected_category = default_category.strip().casefold()
        assert normalized_once["category"] == expected_category, \
            f"Category should be default_category. Expected: '{expected_category}', Got: '{normalized_once['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 19: Default category application
@pytest.mark.property_test
def test_property_19_fallback_when_no_default_configured():
    """
    Property: When no default_category is configured on the adapter,
    a sensible fallback should be used (e.g., "general").
    
    **Validates: Requirements 6.6, 8.5**
    
    This test verifies that:
    1. System works even without configured default_category
    2. A sensible fallback is used
    3. No empty categories result
    4. Graceful handling of missing configuration
    """
    # Create write strategy with mock memory that has NO default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    # Don't set default_category, or set it to None
    memory.default_category = None
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata without category
        metadata = {"tags": ["test"]}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify a fallback category is applied
        assert "category" in normalized, \
            "Normalized metadata should have a category field"
        assert normalized["category"], \
            "Category should not be empty"
        assert len(normalized["category"]) > 0, \
            "Category should have content"
        
        # The fallback should be "general" based on the implementation
        assert normalized["category"] == "general", \
            f"Fallback category should be 'general'. Got: '{normalized['category']}'"
    
    finally:
        session_manager.shutdown()
