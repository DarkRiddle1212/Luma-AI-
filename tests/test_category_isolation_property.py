"""
Property-Based Tests for Category Isolation Invariant

This module implements property-based tests using Hypothesis to verify
that CategoryFilter correctly enforces category isolation - when enabled,
all output memories have categories in the allowed_categories list.

Feature: context-injection-engine
Property 4: Category Isolation Invariant
Validates: Requirements 4.1
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Optional

from luma.core.injection_engine import CategoryFilter, InjectionConfig


# ============================================================================
# Helper Classes and Strategies
# ============================================================================

class MockMemory:
    """Mock memory object for testing CategoryFilter.
    
    This simple mock provides the minimal interface needed by CategoryFilter:
    a category attribute that can be None or a string.
    """
    
    def __init__(self, memory_id: str, category: Optional[str] = None):
        self.memory_id = memory_id
        self.category = category
    
    def __repr__(self):
        return f"MockMemory(id={self.memory_id}, category={self.category})"


@st.composite
def category_strategy(draw):
    """Generate random category strings or None.
    
    Categories are alphanumeric strings with underscores and hyphens,
    representing realistic category names like "programming", "work_notes",
    "personal-diary", etc.
    """
    return draw(st.one_of(
        st.none(),
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_-'
            )
        )
    ))


@st.composite
def memory_list_strategy(draw, min_size=0, max_size=5):
    """Generate random lists of MockMemory objects with various categories.
    
    Creates realistic memory lists with diverse categories for testing
    category isolation filtering.
    
    Args:
        min_size: Minimum number of memories to generate
        max_size: Maximum number of memories to generate
    
    Returns:
        List of MockMemory objects with random categories
    """
    num_memories = draw(st.integers(min_value=min_size, max_value=max_size))
    memories = []
    
    for i in range(num_memories):
        memory_id = f"mem_{i}"
        category = draw(category_strategy())
        memories.append(MockMemory(memory_id, category))
    
    return memories


@st.composite
def allowed_categories_strategy(draw, min_size=1, max_size=10):
    """Generate random lists of allowed category names.
    
    Creates non-empty lists of category names for testing category isolation.
    
    Args:
        min_size: Minimum number of categories
        max_size: Maximum number of categories
    
    Returns:
        List of category name strings
    """
    return draw(st.lists(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='_-'
            )
        ),
        min_size=min_size,
        max_size=max_size,
        unique=True
    ))


@st.composite
def injection_config_with_isolation_strategy(draw):
    """Generate InjectionConfig with category isolation enabled.
    
    Creates valid InjectionConfig objects with enable_category_isolation=True
    and a non-empty allowed_categories list for testing the isolation invariant.
    
    Returns:
        InjectionConfig with category isolation enabled
    """
    allowed_categories = draw(allowed_categories_strategy())
    
    return InjectionConfig(
        max_token_budget=draw(st.integers(min_value=100, max_value=10000)),
        max_memory_count=draw(st.integers(min_value=1, max_value=100)),
        redundancy_similarity_threshold=draw(st.floats(min_value=0.0, max_value=1.0)),
        enable_category_isolation=True,
        allowed_categories=allowed_categories
    )


# ============================================================================
# Property 4: Category Isolation Invariant
# ============================================================================

# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    config=injection_config_with_isolation_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_invariant(memories, config):
    """
    Property: For any InjectionResult where enable_category_isolation is True,
    all memories in the output should have categories that are in the
    allowed_categories list.
    
    **Validates: Requirements 4.1**
    
    This test verifies that:
    1. CategoryFilter enforces category isolation when enabled
    2. All output memories have categories in allowed_categories
    3. No memories with disallowed categories pass through
    4. Memories with None category are filtered out (not in allowed list)
    5. The invariant holds for all possible input combinations
    """
    # Create CategoryFilter with isolation enabled
    filter = CategoryFilter(config)
    
    # Apply category filtering
    filtered = filter.filter(memories)
    
    # INVARIANT: All filtered memories must have categories in allowed_categories
    for memory in filtered:
        assert memory.category in config.allowed_categories, (
            f"Category isolation invariant violated: "
            f"Memory {memory.memory_id} has category '{memory.category}' "
            f"which is not in allowed_categories {config.allowed_categories}. "
            f"Input had {len(memories)} memories, output has {len(filtered)} memories."
        )
    
    # Additional verification: No disallowed categories should pass through
    disallowed_categories = set()
    for memory in memories:
        if memory.category not in config.allowed_categories:
            disallowed_categories.add(memory.category)
    
    for memory in filtered:
        assert memory.category not in disallowed_categories or memory.category in config.allowed_categories, (
            f"Disallowed category passed through filter: {memory.category}"
        )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=30),
    allowed_categories=allowed_categories_strategy(min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_with_explicit_allowed_list(memories, allowed_categories):
    """
    Property: When category isolation is enabled with a specific allowed_categories
    list, only memories with categories in that list should pass through.
    
    **Validates: Requirements 4.1**
    
    This test verifies the invariant with explicit control over the allowed
    categories list, ensuring the filter correctly matches categories.
    """
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=allowed_categories
    )
    
    filter = CategoryFilter(config)
    filtered = filter.filter(memories)
    
    # INVARIANT: All filtered memories must have categories in allowed_categories
    for memory in filtered:
        assert memory.category in allowed_categories, (
            f"Category isolation invariant violated: "
            f"Memory {memory.memory_id} has category '{memory.category}' "
            f"which is not in allowed_categories {allowed_categories}"
        )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    num_memories=st.integers(min_value=5, max_value=30),
    allowed_category=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_with_single_allowed_category(num_memories, allowed_category):
    """
    Property: When category isolation is enabled with a single allowed category,
    only memories with that exact category should pass through.
    
    **Validates: Requirements 4.1**
    
    This test verifies the invariant with a single allowed category,
    which is a common use case for category isolation.
    """
    # Create memories with various categories, some matching the allowed category
    memories = []
    for i in range(num_memories):
        # 30% chance of matching the allowed category
        if i % 3 == 0:
            category = allowed_category
        else:
            # Generate a different category
            category = f"other_category_{i}"
        memories.append(MockMemory(f"mem_{i}", category))
    
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=[allowed_category]
    )
    
    filter = CategoryFilter(config)
    filtered = filter.filter(memories)
    
    # INVARIANT: All filtered memories must have the allowed category
    for memory in filtered:
        assert memory.category == allowed_category, (
            f"Category isolation invariant violated: "
            f"Memory {memory.memory_id} has category '{memory.category}' "
            f"but only '{allowed_category}' is allowed"
        )
    
    # Verify that all memories with the allowed category are included
    expected_count = sum(1 for m in memories if m.category == allowed_category)
    assert len(filtered) == expected_count, (
        f"Expected {expected_count} memories with category '{allowed_category}', "
        f"but got {len(filtered)}"
    )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    config=injection_config_with_isolation_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_filters_none_categories(memories, config):
    """
    Property: When category isolation is enabled, memories with None category
    should be filtered out (unless None is explicitly in allowed_categories,
    which is not a valid configuration).
    
    **Validates: Requirements 4.1**
    
    This test verifies that memories with None category are correctly
    filtered out when category isolation is enabled.
    """
    filter = CategoryFilter(config)
    filtered = filter.filter(memories)
    
    # INVARIANT: No memory with None category should pass through
    # (unless None is in allowed_categories, which shouldn't happen)
    for memory in filtered:
        if memory.category is None:
            assert None in config.allowed_categories, (
                f"Memory {memory.memory_id} with None category passed through filter, "
                f"but None is not in allowed_categories {config.allowed_categories}"
            )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    memories=memory_list_strategy(min_size=0, max_size=5),
    config=injection_config_with_isolation_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_output_is_subset_of_input(memories, config):
    """
    Property: The filtered output should be a subset of the input memories,
    preserving order and only removing memories with disallowed categories.
    
    **Validates: Requirements 4.1**
    
    This test verifies that:
    1. All output memories exist in the input
    2. The order is preserved
    3. Only category filtering is applied (no other modifications)
    """
    filter = CategoryFilter(config)
    filtered = filter.filter(memories)
    
    # INVARIANT: All filtered memories must be in the input list
    input_ids = {m.memory_id for m in memories}
    for memory in filtered:
        assert memory.memory_id in input_ids, (
            f"Filtered memory {memory.memory_id} not found in input memories"
        )
    
    # INVARIANT: Order should be preserved
    # Build a mapping of memory_id to index in input
    input_indices = {m.memory_id: i for i, m in enumerate(memories)}
    
    # Check that filtered memories appear in the same relative order
    filtered_indices = [input_indices[m.memory_id] for m in filtered]
    assert filtered_indices == sorted(filtered_indices), (
        f"Order not preserved: input indices {filtered_indices} are not sorted"
    )
    
    # INVARIANT: All filtered memories have allowed categories
    for memory in filtered:
        assert memory.category in config.allowed_categories, (
            f"Memory {memory.memory_id} with category '{memory.category}' "
            f"should not be in filtered output"
        )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    allowed_categories=allowed_categories_strategy(min_size=1, max_size=5),
    num_matching=st.integers(min_value=0, max_value=20),
    num_non_matching=st.integers(min_value=0, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_with_controlled_distribution(
    allowed_categories, num_matching, num_non_matching
):
    """
    Property: When we control the distribution of matching vs non-matching
    categories, the filter should return exactly the memories with matching
    categories.
    
    **Validates: Requirements 4.1**
    
    This test creates a controlled scenario where we know exactly which
    memories should pass through the filter.
    """
    # Create memories with controlled categories
    memories = []
    
    # Add memories with matching categories
    for i in range(num_matching):
        category = allowed_categories[i % len(allowed_categories)]
        memories.append(MockMemory(f"match_{i}", category))
    
    # Add memories with non-matching categories
    for i in range(num_non_matching):
        category = f"non_matching_{i}"
        # Ensure it's not accidentally in allowed_categories
        while category in allowed_categories:
            category = f"non_matching_{i}_alt"
        memories.append(MockMemory(f"non_match_{i}", category))
    
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=100,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=allowed_categories
    )
    
    filter = CategoryFilter(config)
    filtered = filter.filter(memories)
    
    # INVARIANT: Should have exactly num_matching memories
    assert len(filtered) == num_matching, (
        f"Expected {num_matching} filtered memories, got {len(filtered)}"
    )
    
    # INVARIANT: All filtered memories should have matching categories
    for memory in filtered:
        assert memory.category in allowed_categories, (
            f"Memory {memory.memory_id} with category '{memory.category}' "
            f"should not be in filtered output"
        )
    
    # INVARIANT: All matching memories should be in the output
    matching_ids = {f"match_{i}" for i in range(num_matching)}
    filtered_ids = {m.memory_id for m in filtered}
    assert matching_ids == filtered_ids, (
        f"Filtered IDs {filtered_ids} don't match expected {matching_ids}"
    )


# Feature: context-injection-engine, Property 4: Category Isolation Invariant
@given(
    memories=memory_list_strategy(min_size=1, max_size=5),
    config=injection_config_with_isolation_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_category_isolation_idempotence(memories, config):
    """
    Property: Applying the category filter multiple times should produce
    the same result (idempotence).
    
    **Validates: Requirements 4.1**
    
    This test verifies that the filter is idempotent - applying it multiple
    times doesn't change the result after the first application.
    """
    filter = CategoryFilter(config)
    
    # Apply filter multiple times
    filtered1 = filter.filter(memories)
    filtered2 = filter.filter(filtered1)
    filtered3 = filter.filter(filtered2)
    
    # INVARIANT: All applications should produce the same result
    assert len(filtered1) == len(filtered2) == len(filtered3), (
        f"Filter is not idempotent: got {len(filtered1)}, {len(filtered2)}, {len(filtered3)} memories"
    )
    
    # Verify memory IDs are the same
    ids1 = [m.memory_id for m in filtered1]
    ids2 = [m.memory_id for m in filtered2]
    ids3 = [m.memory_id for m in filtered3]
    
    assert ids1 == ids2 == ids3, (
        f"Filter is not idempotent: memory IDs differ across applications"
    )
    
    # INVARIANT: All memories in all filtered results have allowed categories
    for filtered in [filtered1, filtered2, filtered3]:
        for memory in filtered:
            assert memory.category in config.allowed_categories, (
                f"Category isolation invariant violated in idempotence test"
            )
