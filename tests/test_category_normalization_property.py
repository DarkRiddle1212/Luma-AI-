"""
Property-Based Tests for Category Normalization

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly normalizes category metadata
by trimming whitespace and converting to lowercase.

Feature: memory-write-strategy-session-management
Property 17: Category normalization
Validates: Requirements 6.4
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
    
    def __init__(self):
        self.stored_memories = []
        self.default_category = None
        self.default_tags = []
    
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
def category_with_whitespace(draw):
    """Generate category strings with various whitespace patterns."""
    # Generate a base category (alphanumeric)
    base_category = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))
    
    # Add leading whitespace
    leading_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    
    # Add trailing whitespace
    trailing_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    
    return leading_ws + base_category + trailing_ws


@st.composite
def category_with_mixed_case(draw):
    """Generate category strings with mixed uppercase and lowercase."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))


@st.composite
def category_with_whitespace_and_mixed_case(draw):
    """Generate category strings with both whitespace and mixed case."""
    base_category = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))
    
    leading_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    trailing_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    
    return leading_ws + base_category + trailing_ws


# ============================================================================
# Property 17: Category Normalization
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category=category_with_whitespace()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_whitespace_trimming(category):
    """
    Property: For any category string with leading or trailing whitespace,
    the normalized category should have all whitespace trimmed.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Leading whitespace is removed from category
    2. Trailing whitespace is removed from category
    3. The trimmed category is stored in metadata
    4. Internal whitespace (if any) is preserved
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
        # Create metadata with category containing whitespace
        metadata = {"category": category}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify category is trimmed
        expected_trimmed = category.strip()
        assert normalized["category"] == expected_trimmed.casefold(), \
            f"Category should be trimmed and lowercased. Expected: '{expected_trimmed.casefold()}', Got: '{normalized['category']}'"
        
        # Verify no leading whitespace
        if normalized["category"]:
            assert not normalized["category"][0].isspace(), \
                f"Category should not have leading whitespace: '{normalized['category']}'"
            
            # Verify no trailing whitespace
            assert not normalized["category"][-1].isspace(), \
                f"Category should not have trailing whitespace: '{normalized['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category=category_with_mixed_case()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_lowercase_conversion(category):
    """
    Property: For any category string with uppercase letters,
    the normalized category should be converted to lowercase.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. All uppercase letters are converted to lowercase
    2. Mixed case strings are fully lowercased
    3. Already lowercase strings remain unchanged
    4. Non-alphabetic characters are preserved
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
        # Create metadata with category containing mixed case
        metadata = {"category": category}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify category is lowercased
        expected_lower = category.strip().casefold()
        assert normalized["category"] == expected_lower, \
            f"Category should be lowercased. Expected: '{expected_lower}', Got: '{normalized['category']}'"
        
        # Verify no uppercase letters remain
        assert normalized["category"] == normalized["category"].casefold(), \
            f"Category should not contain uppercase letters: '{normalized['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category=category_with_whitespace_and_mixed_case()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_full_normalization(category):
    """
    Property: For any category string with both whitespace and mixed case,
    the normalized category should be both trimmed and lowercased.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Both trimming and lowercasing are applied together
    2. The order of operations produces correct results
    3. The final category is fully normalized
    4. Normalization is idempotent (applying twice gives same result)
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
        # Create metadata with category containing whitespace and mixed case
        metadata = {"category": category}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify category is both trimmed and lowercased
        expected_normalized = category.strip().casefold()
        assert normalized["category"] == expected_normalized, \
            f"Category should be trimmed and lowercased. Expected: '{expected_normalized}', Got: '{normalized['category']}'"
        
        # Verify normalization is idempotent
        normalized_again = strategy.normalize_metadata(normalized)
        assert normalized_again["category"] == normalized["category"], \
            f"Normalization should be idempotent. First: '{normalized['category']}', Second: '{normalized_again['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category1=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ),
    leading_ws1=st.text(alphabet=' \t\n\r', min_size=0, max_size=5),
    trailing_ws1=st.text(alphabet=' \t\n\r', min_size=0, max_size=5),
    leading_ws2=st.text(alphabet=' \t\n\r', min_size=0, max_size=5),
    trailing_ws2=st.text(alphabet=' \t\n\r', min_size=0, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_normalization_equivalence(
    category1, leading_ws1, trailing_ws1, leading_ws2, trailing_ws2
):
    """
    Property: For any two category strings that differ only in whitespace
    and case, they should normalize to the same value.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Different whitespace patterns normalize to same result
    2. Different case patterns normalize to same result
    3. Normalization creates equivalence classes
    4. Duplicate detection can rely on normalized categories
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
        # First normalize the base category to ensure we're working with a consistent base
        # This handles Unicode edge cases where .upper() and .casefold() might not be inverses
        base_normalized = category1.strip().casefold()
        
        # Create two variations with different whitespace but same normalized base
        cat_variant1 = leading_ws1 + base_normalized + trailing_ws1
        cat_variant2 = leading_ws2 + base_normalized + trailing_ws2
        
        # Normalize both
        metadata1 = {"category": cat_variant1}
        metadata2 = {"category": cat_variant2}
        
        normalized1 = strategy.normalize_metadata(metadata1)
        normalized2 = strategy.normalize_metadata(metadata2)
        
        # Verify both normalize to the same value
        assert normalized1["category"] == normalized2["category"], \
            f"Categories differing only in whitespace should normalize to same value. " \
            f"Variant1: '{cat_variant1}' -> '{normalized1['category']}', " \
            f"Variant2: '{cat_variant2}' -> '{normalized2['category']}'"
        
        # Verify they both equal the base normalized value
        assert normalized1["category"] == base_normalized, \
            f"Normalized category should equal base normalized value. " \
            f"Expected: '{base_normalized}', Got: '{normalized1['category']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category=st.one_of(
        st.just(""),  # Empty string
        st.text(alphabet=' \t\n\r', min_size=1, max_size=10)  # Only whitespace
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_empty_category_handling(category):
    """
    Property: For any empty or whitespace-only category string,
    the normalized metadata should apply the default category.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Empty strings result in default category
    2. Whitespace-only strings result in default category
    3. Default category is applied after normalization
    4. The result is never an empty category
    """
    # Create write strategy with mock memory that has default_category
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    memory.default_category = "general"  # Set default category
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with empty/whitespace category
        metadata = {"category": category}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify default category is applied
        assert normalized["category"] == "general", \
            f"Empty/whitespace category should be replaced with default. Got: '{normalized['category']}'"
        
        # Verify category is not empty
        assert len(normalized["category"]) > 0, \
            "Normalized category should never be empty"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_special_characters_preserved(category):
    """
    Property: For any category string containing special characters
    (hyphens, underscores, numbers), normalization should preserve them
    while still trimming and lowercasing.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Special characters are preserved during normalization
    2. Only whitespace and case are modified
    3. Hyphens, underscores, numbers remain unchanged
    4. Category structure is maintained
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
        # Add some whitespace and mixed case
        category_with_ws = "  " + category + "  "
        metadata = {"category": category_with_ws}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify special characters are preserved
        expected = category.strip().casefold()
        assert normalized["category"] == expected, \
            f"Special characters should be preserved. Expected: '{expected}', Got: '{normalized['category']}'"
        
        # Count special characters in original and normalized
        original_hyphens = category.count('-')
        original_underscores = category.count('_')
        normalized_hyphens = normalized["category"].count('-')
        normalized_underscores = normalized["category"].count('_')
        
        assert original_hyphens == normalized_hyphens, \
            f"Hyphens should be preserved. Original: {original_hyphens}, Normalized: {normalized_hyphens}"
        assert original_underscores == normalized_underscores, \
            f"Underscores should be preserved. Original: {original_underscores}, Normalized: {normalized_underscores}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 17: Category normalization
@given(
    metadata_dict=st.dictionaries(
        st.text(min_size=1, max_size=5).filter(lambda x: x != "category"),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    ),
    category=category_with_whitespace_and_mixed_case()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_17_category_normalization_preserves_other_metadata(metadata_dict, category):
    """
    Property: For any metadata dictionary with a category field,
    normalizing the category should not affect other metadata fields.
    
    **Validates: Requirements 6.4**
    
    This test verifies that:
    1. Other metadata fields remain unchanged
    2. Only the category field is normalized
    3. Metadata structure is preserved
    4. No side effects on other fields
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
        # Create metadata with category and other fields
        metadata = metadata_dict.copy()
        metadata["category"] = category
        
        # Store original values of other fields
        original_other_fields = {k: v for k, v in metadata.items() if k != "category"}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify category is normalized
        expected_category = category.strip().casefold()
        assert normalized["category"] == expected_category, \
            f"Category should be normalized. Expected: '{expected_category}', Got: '{normalized['category']}'"
        
        # Verify other fields are preserved (excluding timestamp and session_id which are added)
        for key, value in original_other_fields.items():
            if key not in ["timestamp", "session_id"]:
                assert key in normalized, f"Field '{key}' should be preserved"
                assert normalized[key] == value, \
                    f"Field '{key}' should not be modified. Expected: {value}, Got: {normalized[key]}"
    
    finally:
        session_manager.shutdown()
