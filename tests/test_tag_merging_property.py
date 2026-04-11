"""
Property-Based Tests for Tag Merging with Defaults

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly merges provided tags with
default_tags from adapter configuration, applying normalization and deduplication.

Feature: memory-write-strategy-session-management
Property 20: Tag merging with defaults
Validates: Requirements 6.7, 8.6
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
    """Mock memory interface for testing write strategy with default_tags."""
    
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
# Helper Strategies
# ============================================================================

@st.composite
def tag_list_strategy(draw, min_size=0, max_size=10):
    """Generate a list of valid tag strings."""
    return draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd')),
            min_size=1,
            max_size=20
        ),
        min_size=min_size,
        max_size=max_size
    ))


# ============================================================================
# Property 20: Tag Merging with Defaults
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    default_tags=tag_list_strategy(min_size=1, max_size=5),
    metadata_tags=tag_list_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_tag_merging_with_defaults(default_tags, metadata_tags):
    """
    Property: For any memory stored through an adapter with default_tags configured,
    the final tags should be the union of provided tags and default_tags (after normalization).
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. default_tags from adapter configuration are included
    2. Provided tags from metadata are included
    3. The union of both sets is computed
    4. Duplicates are removed after normalization
    5. All tags are normalized (trimmed, lowercased)
    """
    # Create write strategy with mock memory interface that has default_tags
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected tags: union of default_tags and metadata_tags (normalized and deduplicated)
        all_tags = default_tags + metadata_tags
        expected_tags = []
        seen = set()
        for tag in all_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify the normalized tags match expected
        assert normalized["tags"] == expected_tags, \
            f"Tags should be merged from default_tags and metadata. Expected: {expected_tags}, Got: {normalized['tags']}"
        
        # Verify all default_tags are present (after normalization)
        for default_tag in default_tags:
            normalized_default = default_tag.strip().casefold()
            if normalized_default:
                assert normalized_default in normalized["tags"], \
                    f"Default tag '{normalized_default}' should be present in final tags"
        
        # Verify all metadata_tags are present (after normalization)
        for metadata_tag in metadata_tags:
            normalized_metadata = metadata_tag.strip().casefold()
            if normalized_metadata:
                assert normalized_metadata in normalized["tags"], \
                    f"Metadata tag '{normalized_metadata}' should be present in final tags"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    default_tags=tag_list_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_default_tags_applied_when_no_metadata_tags(default_tags):
    """
    Property: For any memory stored with no tags in metadata,
    the default_tags from adapter configuration should be applied.
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. default_tags are applied when metadata has no tags field
    2. default_tags are normalized (trimmed, lowercased, deduplicated)
    3. Empty metadata still gets default_tags
    """
    # Create write strategy with mock memory interface that has default_tags
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata without tags field
        metadata = {"category": "test"}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected tags: normalized default_tags
        expected_tags = []
        seen = set()
        for tag in default_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify the normalized tags match expected
        assert normalized["tags"] == expected_tags, \
            f"Default tags should be applied when no metadata tags. Expected: {expected_tags}, Got: {normalized['tags']}"
        
        # Verify all default_tags are present
        for default_tag in default_tags:
            normalized_default = default_tag.strip().casefold()
            if normalized_default:
                assert normalized_default in normalized["tags"], \
                    f"Default tag '{normalized_default}' should be present"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    overlapping_tag=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=15
    ),
    additional_default_tags=tag_list_strategy(min_size=0, max_size=3),
    additional_metadata_tags=tag_list_strategy(min_size=0, max_size=3)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_duplicate_tags_deduplicated_after_merge(
    overlapping_tag,
    additional_default_tags,
    additional_metadata_tags
):
    """
    Property: For any memory where default_tags and metadata tags contain
    the same tag (possibly with different case/whitespace), the final tags
    should contain only one normalized instance.
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. Overlapping tags between default_tags and metadata are deduplicated
    2. Case-insensitive deduplication works correctly
    3. Whitespace variations are normalized and deduplicated
    4. Only one instance of each unique tag remains
    """
    # Create variations of the overlapping tag
    default_variation = "  " + overlapping_tag.upper() + "  "
    metadata_variation = overlapping_tag.lower() + " "
    
    # Build tag lists with overlapping tag
    default_tags = [default_variation] + additional_default_tags
    metadata_tags = [metadata_variation] + additional_metadata_tags
    
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # The overlapping tag should appear only once (normalized)
        normalized_overlapping = overlapping_tag.strip().casefold()
        count = normalized["tags"].count(normalized_overlapping)
        
        assert count == 1, \
            f"Overlapping tag '{normalized_overlapping}' should appear exactly once, found {count} times"
        
        # Verify no duplicates in final tags
        assert len(normalized["tags"]) == len(set(normalized["tags"])), \
            f"Final tags should not contain duplicates. Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    metadata_tags=tag_list_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_metadata_tags_preserved_when_no_default_tags(metadata_tags):
    """
    Property: For any memory stored through an adapter with no default_tags configured,
    only the metadata tags should be present (after normalization).
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. When default_tags is empty, only metadata tags are used
    2. Metadata tags are still normalized
    3. No unexpected tags are added
    """
    # Create write strategy with mock memory interface with empty default_tags
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=[])
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected tags: normalized metadata_tags only
        expected_tags = []
        seen = set()
        for tag in metadata_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify the normalized tags match expected
        assert normalized["tags"] == expected_tags, \
            f"Only metadata tags should be present when no default_tags. Expected: {expected_tags}, Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    default_tags=st.lists(
        st.text(alphabet=' \t\n\r', min_size=1, max_size=5),
        min_size=1,
        max_size=3
    ),
    metadata_tags=tag_list_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_empty_default_tags_filtered_out(default_tags, metadata_tags):
    """
    Property: For any adapter with default_tags containing empty or whitespace-only strings,
    those empty tags should be filtered out during merging.
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. Empty default_tags are filtered out
    2. Whitespace-only default_tags are filtered out
    3. Valid metadata tags are preserved
    4. The merge process handles empty tags gracefully
    """
    # Create write strategy with whitespace-only default_tags
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with valid tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify no empty tags in result
        for tag in normalized["tags"]:
            assert len(tag) > 0, "Tags should not be empty"
            assert not tag.isspace(), "Tags should not be whitespace-only"
        
        # Build expected tags: only normalized metadata_tags (default_tags are all empty)
        expected_tags = []
        seen = set()
        for tag in metadata_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify the normalized tags match expected (no empty default_tags included)
        assert normalized["tags"] == expected_tags, \
            f"Empty default_tags should be filtered out. Expected: {expected_tags}, Got: {normalized['tags']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    default_tags=tag_list_strategy(min_size=1, max_size=5),
    metadata_tags=tag_list_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_tag_order_preserves_defaults_first(default_tags, metadata_tags):
    """
    Property: For any memory with both default_tags and metadata tags,
    the final tag order should preserve default_tags first, followed by metadata tags
    (with duplicates removed).
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. default_tags appear before metadata tags in the final list
    2. Order within each group is preserved
    3. Duplicates are removed while maintaining order
    4. First occurrence determines position
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected order: default_tags first, then metadata_tags (deduplicated)
        all_tags = default_tags + metadata_tags
        expected_order = []
        seen = set()
        for tag in all_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_order.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify order matches expected
        assert normalized["tags"] == expected_order, \
            f"Tag order should preserve defaults first. Expected: {expected_order}, Got: {normalized['tags']}"
        
        # Verify default_tags appear before any metadata-only tags
        normalized_defaults = [t.strip().casefold() for t in default_tags if t.strip()]
        normalized_metadata = [t.strip().casefold() for t in metadata_tags if t.strip()]
        
        # Find metadata-only tags (not in defaults)
        metadata_only = [t for t in normalized_metadata if t not in normalized_defaults]
        
        if metadata_only:
            # Find position of first metadata-only tag
            first_metadata_only_pos = None
            for tag in metadata_only:
                if tag in normalized["tags"]:
                    first_metadata_only_pos = normalized["tags"].index(tag)
                    break
            
            # Verify all default tags appear before first metadata-only tag
            if first_metadata_only_pos is not None:
                for default_tag in normalized_defaults:
                    if default_tag in normalized["tags"]:
                        default_pos = normalized["tags"].index(default_tag)
                        assert default_pos < first_metadata_only_pos, \
                            f"Default tag '{default_tag}' at position {default_pos} should appear before metadata-only tag at position {first_metadata_only_pos}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 20: Tag merging with defaults
@given(
    default_tags=tag_list_strategy(min_size=1, max_size=5),
    metadata_tags=st.lists(st.just(""), min_size=1, max_size=3)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_20_default_tags_applied_when_metadata_tags_empty(default_tags, metadata_tags):
    """
    Property: For any memory with empty strings in metadata tags,
    the default_tags should still be applied and empty metadata tags filtered out.
    
    **Validates: Requirements 6.7, 8.6**
    
    This test verifies that:
    1. Empty metadata tags are filtered out
    2. default_tags are still applied
    3. The merge handles empty metadata tags gracefully
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface(default_tags=default_tags)
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with empty tags
        metadata = {"tags": metadata_tags}
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Build expected tags: only normalized default_tags (metadata tags are all empty)
        expected_tags = []
        seen = set()
        for tag in default_tags:
            normalized_tag = tag.strip().casefold()
            if normalized_tag and normalized_tag not in seen:
                expected_tags.append(normalized_tag)
                seen.add(normalized_tag)
        
        # Verify the normalized tags match expected
        assert normalized["tags"] == expected_tags, \
            f"Default tags should be applied when metadata tags are empty. Expected: {expected_tags}, Got: {normalized['tags']}"
        
        # Verify no empty tags in result
        for tag in normalized["tags"]:
            assert len(tag) > 0, "Result should not contain empty tags"
    
    finally:
        session_manager.shutdown()
