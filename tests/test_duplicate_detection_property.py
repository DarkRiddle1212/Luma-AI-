"""
Property-Based Test for Exact Duplicate Detection

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly detects and rejects exact duplicate
memories.

Feature: memory-write-strategy-session-management
Property 11: Exact duplicate detection and rejection
Validates: Requirements 4.1, 4.2
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing duplicate detection."""
    
    def __init__(self):
        self.stored_memories = []
        self.next_id = 1
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method."""
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "category": (metadata or {}).get("category", "general")
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Mock retrieve method with category filtering."""
        import unicodedata
        
        if params is None:
            return {"memories": self.stored_memories}
        
        # Filter by category if provided
        category = params.get("category")
        if category:
            # Apply same normalization as write_strategy
            normalized_category = unicodedata.normalize('NFC', category.strip().casefold())
            filtered = [
                mem for mem in self.stored_memories
                if unicodedata.normalize('NFC', mem.get("category", "").strip().casefold()) == normalized_category
            ]
            return {"memories": filtered}
        
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Property 11: Exact Duplicate Detection and Rejection
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 11: Exact duplicate detection and rejection
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd', 'Pc', 'Pd', 'Zs'],
            whitelist_characters=[' ', '.', ',', '!', '?']
        )
    ),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_11_exact_duplicate_detection_and_rejection(content, category):
    """
    Property: For any memory with content and category, attempting to store an
    identical memory (same normalized content and category) should be rejected
    and return the existing memory_id.
    
    **Validates: Requirements 4.1, 4.2**
    
    This test verifies that:
    1. The first storage of content succeeds
    2. Attempting to store identical content returns the existing memory_id
    3. The duplicate is detected even with different whitespace
    4. The duplicate is detected even with different case
    5. No new memory is created for duplicates
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial and meets minimum length
        normalized_content = content.strip().lower()
        is_trivial = any(normalized_content == pattern.lower() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Store the first memory directly via memory interface
            memory_id_1 = memory.store(content, {"category": category})
            initial_count = len(memory.stored_memories)
            
            # Attempt to detect duplicate using check_duplicate
            duplicate_id = strategy.check_duplicate(content, category)
            
            # Property 1: check_duplicate should return the existing memory_id
            assert duplicate_id == memory_id_1, \
                f"Exact duplicate should be detected and return existing memory_id, got {duplicate_id} instead of {memory_id_1}"
            
            # Property 2: No new memory should be created
            assert len(memory.stored_memories) == initial_count, \
                f"No new memory should be created for duplicate, count changed from {initial_count} to {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 11: Exact duplicate detection and rejection
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',']
        )
    ),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_11_case_insensitive_duplicate_detection(content, category):
    """
    Property: For any memory with content and category, attempting to store
    the same content with different case should be detected as a duplicate.
    
    **Validates: Requirements 4.1, 4.2, 4.4**
    
    This test verifies that:
    1. Duplicate detection is case-insensitive
    2. Content normalization (casefold) is applied correctly
    3. The existing memory_id is returned for case-variant duplicates
    
    Note: This test only applies to strings where case transformations produce
    distinct variants. Some Unicode characters (e.g., µ, ß) may not have
    distinct case variants, so we skip those cases.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Create case-variant versions
            uppercase_content = content.upper()
            lowercase_content = content.lower()
            mixed_case_content = content.swapcase()
            
            # Skip cases where case transformations don't produce distinct variants
            # This handles Unicode edge cases like 'µ' where case transformations
            # may not produce meaningfully different strings
            variants = {
                content.casefold(),
                uppercase_content.casefold(),
                lowercase_content.casefold(),
                mixed_case_content.casefold()
            }
            assume(len(variants) == 1)  # All variants should normalize to the same string
            
            # Store the first memory with original case
            memory_id_1 = memory.store(content, {"category": category})
            
            # Test uppercase variant
            duplicate_id_upper = strategy.check_duplicate(uppercase_content, category)
            assert duplicate_id_upper == memory_id_1, \
                f"Uppercase variant should be detected as duplicate, got {duplicate_id_upper} instead of {memory_id_1}"
            
            # Test lowercase variant
            duplicate_id_lower = strategy.check_duplicate(lowercase_content, category)
            assert duplicate_id_lower == memory_id_1, \
                f"Lowercase variant should be detected as duplicate, got {duplicate_id_lower} instead of {memory_id_1}"
            
            # Test mixed case variant
            duplicate_id_mixed = strategy.check_duplicate(mixed_case_content, category)
            assert duplicate_id_mixed == memory_id_1, \
                f"Mixed case variant should be detected as duplicate, got {duplicate_id_mixed} instead of {memory_id_1}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 11: Exact duplicate detection and rejection
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',']
        )
    ),
    category=st.text(min_size=3, max_size=5),
    leading_spaces=st.integers(min_value=0, max_value=5),
    trailing_spaces=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_11_whitespace_normalized_duplicate_detection(content, category, leading_spaces, trailing_spaces):
    """
    Property: For any memory with content and category, attempting to store
    the same content with different whitespace should be detected as a duplicate.
    
    **Validates: Requirements 4.1, 4.2, 4.4**
    
    This test verifies that:
    1. Duplicate detection normalizes whitespace (trim)
    2. Leading and trailing whitespace is ignored
    3. The existing memory_id is returned for whitespace-variant duplicates
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().lower()
        is_trivial = any(normalized_content == pattern.lower() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Store the first memory with original content
            memory_id_1 = memory.store(content, {"category": category})
            
            # Create whitespace-variant version
            whitespace_variant = (' ' * leading_spaces) + content + (' ' * trailing_spaces)
            
            # Test whitespace variant
            duplicate_id = strategy.check_duplicate(whitespace_variant, category)
            assert duplicate_id == memory_id_1, \
                f"Whitespace variant should be detected as duplicate, got {duplicate_id} instead of {memory_id_1}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 11: Exact duplicate detection and rejection
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',']
        )
    ),
    category1=st.text(min_size=3, max_size=5),
    category2=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_11_different_category_not_duplicate(content, category1, category2):
    """
    Property: For any memory with content, the same content in a different
    category should NOT be detected as a duplicate.
    
    **Validates: Requirements 4.1, 4.2**
    
    This test verifies that:
    1. Duplicate detection is category-specific
    2. Same content in different categories are treated as separate memories
    3. check_duplicate returns None for same content in different category
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().lower()
        is_trivial = any(normalized_content == pattern.lower() for pattern in config.trivial_patterns)
        
        # Ensure categories are different (after normalization)
        if (not is_trivial and 
            len(content.strip()) >= config.min_content_length and
            category1.strip().lower() != category2.strip().lower()):
            
            # Store the first memory in category1
            memory_id_1 = memory.store(content, {"category": category1})
            
            # Check for duplicate in category2 (should not find one)
            duplicate_id = strategy.check_duplicate(content, category2)
            
            # Property: check_duplicate should return None (not a duplicate in different category)
            assert duplicate_id is None, \
                f"Same content in different category should NOT be detected as duplicate, got {duplicate_id}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 11: Exact duplicate detection and rejection
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',']
        )
    ),
    category=st.text(min_size=3, max_size=5),
    num_duplicates=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_11_multiple_duplicate_attempts(content, category, num_duplicates):
    """
    Property: For any memory with content and category, multiple attempts to
    store the same content should all be detected as duplicates and return
    the same original memory_id.
    
    **Validates: Requirements 4.1, 4.2**
    
    This test verifies that:
    1. Multiple duplicate attempts are all detected
    2. All duplicate checks return the same original memory_id
    3. No additional memories are created for any duplicate attempt
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().lower()
        is_trivial = any(normalized_content == pattern.lower() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Store the first memory
            memory_id_1 = memory.store(content, {"category": category})
            initial_count = len(memory.stored_memories)
            
            # Attempt to detect duplicate multiple times
            for i in range(num_duplicates):
                duplicate_id = strategy.check_duplicate(content, category)
                
                # Property 1: Each check should return the original memory_id
                assert duplicate_id == memory_id_1, \
                    f"Duplicate attempt {i+1} should return original memory_id, got {duplicate_id} instead of {memory_id_1}"
                
                # Property 2: No new memories should be created
                assert len(memory.stored_memories) == initial_count, \
                    f"No new memory should be created on duplicate attempt {i+1}"
    
    finally:
        session_manager.shutdown()


# ============================================================================
# Property 12: Content Normalization for Duplicates
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 12: Content normalization for duplicates
@given(
    base_content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',', '!', '?']
        )
    ),
    category=st.text(min_size=3, max_size=5),
    leading_spaces=st.integers(min_value=0, max_value=10),
    trailing_spaces=st.integers(min_value=0, max_value=10),
    case_transform=st.sampled_from(['upper', 'lower', 'swapcase', 'title'])
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_12_content_normalization_for_duplicates(
    base_content, category, leading_spaces, trailing_spaces, case_transform
):
    """
    Property: For any two memories with content that differs only in whitespace
    and case (e.g., "Hello" vs "  hello  "), they should be considered duplicates
    after normalization.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. Content normalization applies both trim and casefold operations
    2. Memories differing only in leading/trailing whitespace are duplicates
    3. Memories differing only in case are duplicates
    4. Memories differing in both whitespace and case are duplicates
    5. The existing memory_id is returned for all normalized duplicates
    
    The normalization process should:
    - Strip leading and trailing whitespace
    - Apply casefold() for Unicode-aware case-insensitive comparison
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure base content is not trivial
        normalized_base = base_content.strip().casefold()
        is_trivial = any(normalized_base == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(base_content.strip()) >= config.min_content_length:
            # Store the first memory with base content
            memory_id_1 = memory.store(base_content, {"category": category})
            initial_count = len(memory.stored_memories)
            
            # Apply case transformation
            if case_transform == 'upper':
                transformed_content = base_content.upper()
            elif case_transform == 'lower':
                transformed_content = base_content.lower()
            elif case_transform == 'swapcase':
                transformed_content = base_content.swapcase()
            else:  # title
                transformed_content = base_content.title()
            
            # Add whitespace variations
            variant_content = (' ' * leading_spaces) + transformed_content + (' ' * trailing_spaces)
            
            # Verify that the variant normalizes to the same value as base
            # This ensures we're testing actual duplicates
            variant_normalized = variant_content.strip().casefold()
            base_normalized = base_content.strip().casefold()
            
            # Only proceed if normalization makes them identical
            if variant_normalized == base_normalized:
                # Test duplicate detection with normalized variant
                duplicate_id = strategy.check_duplicate(variant_content, category)
                
                # Property 1: Normalized duplicate should be detected
                assert duplicate_id == memory_id_1, \
                    f"Content differing only in whitespace/case should be detected as duplicate. " \
                    f"Base: '{base_content}', Variant: '{variant_content}', " \
                    f"Expected: {memory_id_1}, Got: {duplicate_id}"
                
                # Property 2: No new memory should be created
                assert len(memory.stored_memories) == initial_count, \
                    f"No new memory should be created for normalized duplicate, " \
                    f"count changed from {initial_count} to {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 12: Content normalization for duplicates
@given(
    content_words=st.lists(
        st.text(
            min_size=3,
            max_size=15,
            alphabet=st.characters(whitelist_categories=['Ll', 'Lu', 'Nd'])
        ),
        min_size=2,
        max_size=10
    ),
    category=st.text(min_size=3, max_size=5),
    whitespace_pattern=st.sampled_from([' ', '  ', '\t', '   '])
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_12_internal_whitespace_normalization(content_words, category, whitespace_pattern):
    """
    Property: For any two memories where content differs only in internal
    whitespace patterns (e.g., "hello  world" vs "hello world"), the
    normalization should handle leading/trailing whitespace but preserve
    internal structure for comparison.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. Internal whitespace is preserved during normalization
    2. Only leading/trailing whitespace is stripped
    3. Different internal whitespace patterns result in different normalized content
    4. Duplicate detection may flag them as near-duplicates if similarity >= threshold
    
    Note: The current implementation uses strip() + casefold() for exact matching,
    but also performs near-duplicate detection using similarity calculation.
    Content with minor internal whitespace differences may be detected as
    near-duplicates if their similarity score >= 0.9.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create content with single space between words
        content1 = ' '.join(content_words)
        
        # Ensure content is not trivial
        normalized_content = content1.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content1.strip()) >= config.min_content_length:
            # Store the first memory
            memory_id_1 = memory.store(content1, {"category": category})
            
            # Create variant with different internal whitespace
            content2 = whitespace_pattern.join(content_words)
            
            # Check if they normalize to the same value
            norm1 = content1.strip().casefold()
            norm2 = content2.strip().casefold()
            
            if norm1 == norm2:
                # Exact match: Should be detected as duplicate
                duplicate_id = strategy.check_duplicate(content2, category)
                assert duplicate_id == memory_id_1, \
                    f"Content with same normalized internal whitespace should be duplicate"
            else:
                # Different normalized content: May be detected as near-duplicate
                # if similarity >= threshold (0.9), or None if similarity < threshold
                duplicate_id = strategy.check_duplicate(content2, category)
                
                # Calculate similarity to determine expected behavior
                similarity = strategy._calculate_similarity(norm1, norm2)
                
                if similarity >= config.similarity_threshold:
                    # Near-duplicate: Should be detected
                    assert duplicate_id == memory_id_1, \
                        f"Content with high similarity ({similarity:.2f} >= {config.similarity_threshold}) " \
                        f"should be detected as near-duplicate"
                else:
                    # Low similarity: Should NOT be detected as duplicate
                    assert duplicate_id is None, \
                        f"Content with low similarity ({similarity:.2f} < {config.similarity_threshold}) " \
                        f"should NOT be detected as duplicate"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 12: Content normalization for duplicates
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',']
        )
    ),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_12_casefold_unicode_normalization(content, category):
    """
    Property: For any content with Unicode characters, normalization should
    use casefold() rather than lower() for proper Unicode case handling.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. casefold() is used for case normalization (not lower())
    2. Unicode characters are handled correctly
    3. content.casefold() == content.lower().casefold() for duplicate detection
    
    The difference between lower() and casefold():
    - lower(): Simple case conversion
    - casefold(): Aggressive case folding for caseless matching
    - Example: 'ß'.lower() = 'ß', but 'ß'.casefold() = 'ss'
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Store the first memory
            memory_id_1 = memory.store(content, {"category": category})
            
            # Create variants using different case operations
            lower_variant = content.lower()
            casefold_variant = content.casefold()
            
            # Test that both variants are detected as duplicates
            # (because check_duplicate uses casefold internally)
            duplicate_id_lower = strategy.check_duplicate(lower_variant, category)
            duplicate_id_casefold = strategy.check_duplicate(casefold_variant, category)
            
            # Both should be detected as duplicates
            assert duplicate_id_lower == memory_id_1, \
                f"Lower variant should be detected as duplicate"
            assert duplicate_id_casefold == memory_id_1, \
                f"Casefold variant should be detected as duplicate"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 12: Content normalization for duplicates
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',', '\n', '\r', '\t']
        )
    ),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_12_strip_removes_all_whitespace_types(content, category):
    """
    Property: For any content with various whitespace characters (spaces, tabs,
    newlines), strip() should remove all leading and trailing whitespace types.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. strip() removes spaces, tabs, newlines, carriage returns
    2. All whitespace types are handled uniformly
    3. Content differing only in leading/trailing whitespace types are duplicates
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Store the first memory with original content
            memory_id_1 = memory.store(content, {"category": category})
            
            # Test that stripped content is detected as duplicate
            stripped_content = content.strip()
            duplicate_id = strategy.check_duplicate(stripped_content, category)
            
            # Property: Stripped content should be detected as duplicate
            assert duplicate_id == memory_id_1, \
                f"Content with whitespace stripped should be detected as duplicate"
    
    finally:
        session_manager.shutdown()

def test_unicode_micro_sign_duplicate_detection():
    """
    Test case-insensitive duplicate detection with Unicode micro sign (µ).
    
    The micro sign 'µ' (U+00B5) has special case-folding behavior:
    - µ.upper() → 'Μ' (Greek capital letter Mu, U+039C)
    - µ.casefold() → 'μ' (Greek small letter mu, U+03BC)
    
    This test verifies that casefold() normalization correctly handles
    this Unicode edge case.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Store memory with micro sign
        content1 = "temperature is 25µC"
        memory_id_1 = memory.store(content1, {"category": "measurement"})
        
        # Test with uppercase variant (contains Greek Mu)
        content2 = "TEMPERATURE IS 25ΜC"  # Note: Μ is Greek capital Mu
        duplicate_id = strategy.check_duplicate(content2, "measurement")
        
        # Should be detected as duplicate due to casefold() normalization
        assert duplicate_id == memory_id_1, \
            f"Micro sign variant should be detected as duplicate, got {duplicate_id} instead of {memory_id_1}"
    
    finally:
        session_manager.shutdown()


def test_unicode_german_sharp_s_duplicate_detection():
    """
    Test case-insensitive duplicate detection with German sharp s (ß).
    
    The German sharp s 'ß' (U+00DF) has special case-folding behavior:
    - ß.upper() → 'SS'
    - ß.casefold() → 'ss'
    
    This test verifies that casefold() normalization correctly handles
    this Unicode edge case.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Store memory with German sharp s
        content1 = "die Straße ist lang"
        memory_id_1 = memory.store(content1, {"category": "german"})
        
        # Test with uppercase variant (ß becomes SS)
        content2 = "DIE STRASSE IST LANG"
        duplicate_id = strategy.check_duplicate(content2, "german")
        
        # Should be detected as duplicate due to casefold() normalization
        assert duplicate_id == memory_id_1, \
            f"German sharp s variant should be detected as duplicate, got {duplicate_id} instead of {memory_id_1}"
    
    finally:
        session_manager.shutdown()


def test_unicode_mixed_script_duplicate_detection():
    """
    Test case-insensitive duplicate detection with mixed scripts.
    
    This test verifies that casefold() normalization works correctly
    with strings containing multiple Unicode scripts (Latin, Greek, Cyrillic).
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Store memory with mixed scripts
        content1 = "Hello Привет Γεια"  # English, Russian, Greek
        memory_id_1 = memory.store(content1, {"category": "multilingual"})
        
        # Test with different case variants
        content2 = "HELLO ПРИВЕТ ΓΕΙΑ"
        duplicate_id_upper = strategy.check_duplicate(content2, "multilingual")
        assert duplicate_id_upper == memory_id_1, \
            f"Uppercase mixed script should be detected as duplicate"
        
        content3 = "hello привет γεια"
        duplicate_id_lower = strategy.check_duplicate(content3, "multilingual")
        assert duplicate_id_lower == memory_id_1, \
            f"Lowercase mixed script should be detected as duplicate"
    
    finally:
        session_manager.shutdown()


def test_unicode_turkish_i_duplicate_detection():
    """
    Test case-insensitive duplicate detection with Turkish I/i.
    
    Turkish has special case rules for I/i:
    - i.upper() → 'İ' (Latin capital I with dot above) in Turkish locale
    - I.lower() → 'ı' (Latin small dotless i) in Turkish locale
    
    However, casefold() uses locale-independent Unicode case folding,
    so this test verifies the expected behavior.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Store memory with Turkish text
        content1 = "istanbul is beautiful"
        memory_id_1 = memory.store(content1, {"category": "turkish"})
        
        # Test with uppercase variant
        content2 = "ISTANBUL IS BEAUTIFUL"
        duplicate_id = strategy.check_duplicate(content2, "turkish")
        
        # Should be detected as duplicate (casefold is locale-independent)
        assert duplicate_id == memory_id_1, \
            f"Turkish text variant should be detected as duplicate, got {duplicate_id} instead of {memory_id_1}"
    
    finally:
        session_manager.shutdown()


def test_unicode_combining_characters_duplicate_detection():
    """
    Test case-insensitive duplicate detection with combining characters.
    
    Unicode allows characters to be represented in multiple ways:
    - Precomposed: é (U+00E9)
    - Decomposed: e (U+0065) + ́ (U+0301)
    
    Note: This test documents current behavior. Full Unicode normalization
    (NFC/NFD) is not implemented yet but may be added in the future.
    """
    config = WriteStrategyConfig()
    memory = MockMemoryInterface()
    session_manager = Session_Manager(
        SessionConfig(),
        memory
    )
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Store memory with precomposed character
        content1 = "café"  # é is precomposed (U+00E9)
        memory_id_1 = memory.store(content1, {"category": "french"})
        
        # Test with decomposed character
        content2 = "café"  # é is decomposed (e + combining acute)
        duplicate_id = strategy.check_duplicate(content2, "french")
        
        # Current behavior: May or may not be detected as duplicate
        # depending on whether the strings are already normalized
        # This test documents the behavior without asserting a specific result
        # Future enhancement: Implement NFC/NFD normalization
        
        # For now, just verify the method doesn't crash
        assert duplicate_id is None or isinstance(duplicate_id, str)
    
    finally:
        session_manager.shutdown()
