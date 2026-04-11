"""
Property-Based Tests for Tag Normalization

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly normalizes tag metadata
by trimming whitespace, converting to lowercase, and removing duplicates.

Feature: memory-write-strategy-session-management
Property 18: Tag normalization and deduplication
Validates: Requirements 6.5
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
def tag_with_whitespace(draw):
    """Generate tag strings with various whitespace patterns."""
    # Generate a base tag (alphanumeric)
    base_tag = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))
    
    # Add leading whitespace
    leading_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    
    # Add trailing whitespace
    trailing_ws = draw(st.text(alphabet=' \t\n\r', min_size=0, max_size=5))
    
    return leading_ws + base_tag + trailing_ws


@st.composite
def tag_with_mixed_case(draw):
    """Generate tag strings with mixed uppercase and lowercase."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))


@st.composite
def tags_list_with_duplicates(draw):
    """Generate a list of tags that contains duplicates (case-insensitive or whitespace variations)."""
    # Generate a base tag
    base_tag = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=15
    ))
    
    # Create variations of the same tag
    num_variations = draw(st.integers(min_value=2, max_value=5))
    tags = []
    
    for _ in range(num_variations):
        # Randomly apply transformations
        variant = base_tag
        
        # Random case transformation
        case_choice = draw(st.integers(min_value=0, max_value=2))
        if case_choice == 0:
            variant = variant.upper()
        elif case_choice == 1:
            variant = variant.lower()
        # else: keep original case
        
        # Random whitespace
        if draw(st.booleans()):
            leading_ws = draw(st.text(alphabet=' \t', min_size=0, max_size=3))
            trailing_ws = draw(st.text(alphabet=' \t', min_size=0, max_size=3))
            variant = leading_ws + variant + trailing_ws
        
        tags.append(variant)
    
    return tags


# ============================================================================
# Property 18: Tag Normalization and Deduplication
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=st.lists(
        tag_with_whitespace(),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_whitespace_trimming(tags):
    """
    Property: For any list of tag strings with leading or trailing whitespace,
    the normalized tags should have all whitespace trimmed.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Leading whitespace is removed from all tags
    2. Trailing whitespace is removed from all tags
    3. The trimmed tags are stored in metadata
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
        # Create metadata with tags containing whitespace
        metadata = {"tags": tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify all tags are trimmed
        for tag in normalized["tags"]:
            # Verify no leading whitespace
            if tag:
                assert not tag[0].isspace(), \
                    f"Tag should not have leading whitespace: '{tag}'"
                
                # Verify no trailing whitespace
                assert not tag[-1].isspace(), \
                    f"Tag should not have trailing whitespace: '{tag}'"
        
        # Verify each normalized tag corresponds to a trimmed original
        expected_trimmed = [t.strip().casefold() for t in tags if t.strip()]
        # Remove duplicates while preserving order
        expected_unique = []
        seen = set()
        for t in expected_trimmed:
            if t not in seen:
                expected_unique.append(t)
                seen.add(t)
        
        assert normalized["tags"] == expected_unique, \
            f"Tags should be trimmed and deduplicated. Expected: {expected_unique}, Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=st.lists(
        tag_with_mixed_case(),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_lowercase_conversion(tags):
    """
    Property: For any list of tag strings with uppercase letters,
    the normalized tags should be converted to lowercase.
    
    **Validates: Requirements 6.5**
    
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
        # Create metadata with tags containing mixed case
        metadata = {"tags": tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify all tags are lowercased
        for tag in normalized["tags"]:
            assert tag == tag.casefold(), \
                f"Tag should be lowercased. Expected: '{tag.casefold()}', Got: '{tag}'"
        
        # Verify each normalized tag is lowercase version of original (after trim and dedup)
        expected_lower = []
        seen = set()
        for t in tags:
            normalized_t = t.strip().casefold()
            if normalized_t and normalized_t not in seen:
                expected_lower.append(normalized_t)
                seen.add(normalized_t)
        
        assert normalized["tags"] == expected_lower, \
            f"Tags should be lowercased and deduplicated. Expected: {expected_lower}, Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=tags_list_with_duplicates()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_deduplication(tags):
    """
    Property: For any list of tags containing duplicates (after normalization),
    the normalized tags should contain only unique values.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Exact duplicates are removed
    2. Case-insensitive duplicates are removed
    3. Whitespace-variation duplicates are removed
    4. Order of first occurrence is preserved
    5. Only one instance of each unique tag remains
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
        # Create metadata with duplicate tags
        metadata = {"tags": tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify no duplicates in normalized tags
        assert len(normalized["tags"]) == len(set(normalized["tags"])), \
            f"Normalized tags should not contain duplicates. Got: {normalized['tags']}"
        
        # Verify all tags are unique
        seen = set()
        for tag in normalized["tags"]:
            assert tag not in seen, \
                f"Tag '{tag}' appears multiple times in normalized tags"
            seen.add(tag)
        
        # Verify the normalized list contains exactly one instance of the base tag
        # All variations should normalize to the same value
        base_normalized = tags[0].strip().casefold()
        count = normalized["tags"].count(base_normalized)
        assert count == 1, \
            f"Should have exactly one instance of normalized tag '{base_normalized}', found {count}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ),
        min_size=1,
        max_size=10
    ),
    leading_ws=st.lists(
        st.text(alphabet=' \t\n\r', min_size=0, max_size=5),
        min_size=1,
        max_size=10
    ),
    trailing_ws=st.lists(
        st.text(alphabet=' \t\n\r', min_size=0, max_size=5),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_full_normalization(tags, leading_ws, trailing_ws):
    """
    Property: For any list of tags with whitespace and mixed case,
    the normalized tags should be trimmed, lowercased, and deduplicated.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. All three normalizations are applied together
    2. The order of operations produces correct results
    3. The final tags are fully normalized
    4. Normalization is idempotent
    """
    # Ensure lists are same length
    min_len = min(len(tags), len(leading_ws), len(trailing_ws))
    tags = tags[:min_len]
    leading_ws = leading_ws[:min_len]
    trailing_ws = trailing_ws[:min_len]
    
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Add whitespace to tags
        tags_with_ws = [
            leading_ws[i] + tags[i] + trailing_ws[i]
            for i in range(len(tags))
        ]
        
        # Create metadata with tags
        metadata = {"tags": tags_with_ws}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify tags are trimmed, lowercased, and deduplicated
        expected = []
        seen = set()
        for tag in tags_with_ws:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected.append(normalized_tag)
                seen.add(normalized_tag)
        
        assert normalized["tags"] == expected, \
            f"Tags should be fully normalized. Expected: {expected}, Got: {normalized['tags']}"
        
        # Verify normalization is idempotent
        normalized_again = strategy.normalize_metadata(normalized)
        assert normalized_again["tags"] == normalized["tags"], \
            f"Normalization should be idempotent. First: {normalized['tags']}, Second: {normalized_again['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=st.lists(
        st.one_of(
            st.just(""),  # Empty string
            st.text(alphabet=' \t\n\r', min_size=1, max_size=10)  # Only whitespace
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_empty_tag_filtering(tags):
    """
    Property: For any list containing empty or whitespace-only tags,
    the normalized tags should exclude these empty values.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Empty strings are filtered out
    2. Whitespace-only strings are filtered out
    3. The result contains no empty tags
    4. Valid tags are preserved
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
        # Create metadata with empty/whitespace tags
        metadata = {"tags": tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify no empty tags in result
        for tag in normalized["tags"]:
            assert len(tag) > 0, \
                "Normalized tags should not contain empty strings"
            assert not tag.isspace(), \
                "Normalized tags should not contain whitespace-only strings"
        
        # Since all input tags are empty/whitespace, result should be empty list
        assert normalized["tags"] == [], \
            f"All empty/whitespace tags should be filtered out. Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
            min_size=1,
            max_size=30
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_special_characters_preserved(tags):
    """
    Property: For any list of tags containing special characters
    (hyphens, underscores, numbers), normalization should preserve them
    while still trimming, lowercasing, and deduplicating.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Special characters are preserved during normalization
    2. Only whitespace and case are modified
    3. Hyphens, underscores, numbers remain unchanged
    4. Tag structure is maintained
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
        tags_with_ws = ["  " + tag + "  " for tag in tags]
        metadata = {"tags": tags_with_ws}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify special characters are preserved
        expected = []
        seen = set()
        for tag in tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected.append(normalized_tag)
                seen.add(normalized_tag)
        
        assert normalized["tags"] == expected, \
            f"Special characters should be preserved. Expected: {expected}, Got: {normalized['tags']}"
        
        # Count special characters in original and normalized
        for i, original_tag in enumerate(tags):
            if i < len(normalized["tags"]):
                original_hyphens = original_tag.count('-')
                original_underscores = original_tag.count('_')
                normalized_hyphens = normalized["tags"][i].count('-')
                normalized_underscores = normalized["tags"][i].count('_')
                
                # Note: Due to deduplication, we can only check the first occurrence
                if original_tag.strip().casefold() == normalized["tags"][i]:
                    assert original_hyphens == normalized_hyphens, \
                        f"Hyphens should be preserved in tag '{original_tag}'"
                    assert original_underscores == normalized_underscores, \
                        f"Underscores should be preserved in tag '{original_tag}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    metadata_dict=st.dictionaries(
        st.text(min_size=1, max_size=5).filter(lambda x: x != "tags"),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    ),
    tags=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_normalization_preserves_other_metadata(metadata_dict, tags):
    """
    Property: For any metadata dictionary with a tags field,
    normalizing the tags should not affect other metadata fields.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Other metadata fields remain unchanged
    2. Only the tags field is normalized
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
        # Create metadata with tags and other fields
        metadata = metadata_dict.copy()
        metadata["tags"] = tags
        
        # Store original values of other fields
        original_other_fields = {k: v for k, v in metadata.items() if k != "tags"}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify tags are normalized
        expected_tags = []
        seen = set()
        for tag in tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        assert normalized["tags"] == expected_tags, \
            f"Tags should be normalized. Expected: {expected_tags}, Got: {normalized['tags']}"
        
        # Verify other fields are preserved (excluding timestamp, session_id, category which are added/modified)
        for key, value in original_other_fields.items():
            if key not in ["timestamp", "session_id", "category"]:
                assert key in normalized, f"Field '{key}' should be preserved"
                assert normalized[key] == value, \
                    f"Field '{key}' should not be modified. Expected: {value}, Got: {normalized[key]}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 18: Tag normalization and deduplication
@given(
    tags1=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=15
        ),
        min_size=1,
        max_size=5
    ),
    tags2=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=15
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_18_tag_order_preservation(tags1, tags2):
    """
    Property: For any list of tags, normalization should preserve
    the order of first occurrence of each unique tag.
    
    **Validates: Requirements 6.5**
    
    This test verifies that:
    1. Order of first occurrence is maintained
    2. Duplicates are removed but order is preserved
    3. The first instance of each unique tag determines position
    4. Subsequent duplicates don't affect order
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
        # Combine tags with some duplicates
        all_tags = tags1 + tags2 + tags1  # tags1 appears twice
        
        # Create metadata
        metadata = {"tags": all_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected order: first occurrence of each unique normalized tag
        expected_order = []
        seen = set()
        for tag in all_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_order.append(normalized_tag)
                seen.add(normalized_tag)
        
        assert normalized["tags"] == expected_order, \
            f"Tag order should be preserved. Expected: {expected_order}, Got: {normalized['tags']}"
        
        # Verify each tag appears exactly once
        assert len(normalized["tags"]) == len(set(normalized["tags"])), \
            "Each tag should appear exactly once"
    
    finally:
        session_manager.shutdown()
