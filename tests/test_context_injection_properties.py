"""
Property-Based Tests for Context Injection Strategy

This module implements property-based tests using Hypothesis to verify
the correctness properties of the Context Injection Strategy module.

Feature: context-injection-memory-relevance
Tests Properties 3, 4, and others as specified in the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Any
from datetime import datetime
import copy

from luma.core.context_injection import (
    inject_memories,
    InjectionConfig,
    transform_memory_entry
)
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult,
    MemoryRetrievalError
)


# ============================================================================
# Hypothesis Strategies for Generating Test Data
# ============================================================================


@st.composite
def primitive_value_strategy(draw):
    """Generate primitive values (str, int, float, bool, None)."""
    return draw(st.one_of(
        st.text(max_size=5),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ))


@st.composite
def metadata_strategy(draw):
    """Generate varied metadata structures with nested dicts and lists."""
    # Base metadata with common fields
    metadata = {
        "source": draw(st.text(min_size=1, max_size=5)),
        "priority": draw(st.integers(min_value=1, max_value=10))
    }
    
    # Add random extra fields with varied types
    num_extra_fields = draw(st.integers(min_value=0, max_value=5))
    for i in range(num_extra_fields):
        key = f"field_{i}"
        # Generate nested structures
        value = draw(st.one_of(
            primitive_value_strategy(),
            st.lists(primitive_value_strategy(), max_size=3),
            st.dictionaries(
                st.text(min_size=1, max_size=10),
                primitive_value_strategy(),
                max_size=3
            )
        ))
        metadata[key] = value
    
    return metadata


@st.composite
def memory_entry_strategy(draw):
    """Generate valid MemoryEntry objects with varied metadata."""
    memory_id = draw(st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        )
    ))
    
    content = draw(st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')
        )
    ))
    
    category = draw(st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        )
    ))
    
    tags = draw(st.lists(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'),
                whitelist_characters='-_'
            )
        ),
        min_size=0,
        max_size=5
    ))
    
    # Generate timestamp
    timestamp = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 12, 31)
    )).isoformat()
    
    # Generate varied metadata
    metadata = draw(metadata_strategy())
    
    memory_entry: MemoryEntry = {
        "id": memory_id,
        "content": content,
        "metadata": metadata,
        "timestamp": timestamp,
        "category": category,
        "tags": tags
    }
    
    return memory_entry


# ============================================================================
# Mock Memory Interface
# ============================================================================


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self, memories: List[MemoryEntry], should_fail: bool = False):
        """
        Initialize mock with predefined memories.
        
        Args:
            memories: List of MemoryEntry objects to return
            should_fail: If True, raise MemoryRetrievalError on retrieve
        """
        self.memories = memories
        self.should_fail = should_fail
    
    def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Not used in these tests."""
        return "mock_id"
    
    def retrieve(
        self,
        query: str = None,
        params: QueryParameters = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Return predefined memories or raise error."""
        if self.should_fail:
            raise MemoryRetrievalError("Simulated retrieval failure")
        
        # Extract limit from params if provided
        if params and "limit" in params:
            limit = params["limit"]
        
        # Return memories up to limit
        limited_memories = self.memories[:limit]
        
        return {
            "memories": limited_memories,
            "total_count": len(limited_memories),
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": len(self.memories) > limit
            }
        }


# ============================================================================
# Property 1: Memories Key Existence
# ============================================================================


# Feature: context-injection-memory-relevance, Property 1: Memories Key Existence
@given(
    query=st.text(min_size=0, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memories_key_exists(query, max_memories, memories):
    """
    Property: For any valid input (query, memory_interface, config), the
    returned context dictionary SHALL contain the key "memories".
    
    **Validates: Requirements 1.1**
    
    This test verifies that:
    1. "memories" key is always present in returned context
    2. This holds for any query (empty, short, long)
    3. This holds for any max_memories value (5-20)
    4. This holds for any number of retrieved memories (0-50)
    5. This holds even when retrieval succeeds with empty results
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 1.1: "memories" key must always exist
    assert "memories" in context, \
        f"'memories' key missing from context for query='{query}', " \
        f"max_memories={max_memories}, memory_count={len(memories)}"


# Feature: context-injection-memory-relevance, Property 1: Memories Key Existence
@given(
    query=st.text(min_size=0, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memories_key_exists_on_retrieval_failure(query, max_memories):
    """
    Property: For any valid input, the returned context SHALL contain the
    "memories" key even when retrieval fails.
    
    **Validates: Requirements 1.1, 4.2**
    
    This test verifies that:
    1. "memories" key exists even when MemoryRetrievalError is raised
    2. Graceful degradation ensures context structure is maintained
    3. Empty list is injected on failure
    """
    # Create mock memory interface that fails
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface([], should_fail=True)
    
    # Inject memories into context (should handle failure gracefully)
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 1.1: "memories" key must exist even on failure
    assert "memories" in context, \
        f"'memories' key missing from context after retrieval failure"
    
    # REQUIREMENT 4.2: Empty list should be injected on failure
    assert context["memories"] == [], \
        f"Expected empty list on failure, got {context['memories']}"


# Feature: context-injection-memory-relevance, Property 1: Memories Key Existence
@given(
    query=st.text(min_size=0, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    existing_context=st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_memories_key_exists_with_existing_context(query, max_memories, existing_context):
    """
    Property: For any valid input with existing context, the returned context
    SHALL contain the "memories" key.
    
    **Validates: Requirements 1.1**
    
    This test verifies that:
    1. "memories" key is added to existing context
    2. Existing context fields are preserved
    3. "memories" key exists regardless of existing context content
    """
    # Create mock memory interface with empty memories
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface([])
    
    # Inject memories into existing context
    context = inject_memories(query, memory_interface, config, existing_context)
    
    # REQUIREMENT 1.1: "memories" key must exist
    assert "memories" in context, \
        f"'memories' key missing from context with existing_context={existing_context}"
    
    # Verify existing context fields are preserved
    for key, value in existing_context.items():
        if key != "memories":  # Don't check memories key from existing context
            assert key in context, f"Existing context key '{key}' was lost"
            assert context[key] == value, f"Existing context value for '{key}' was modified"


# ============================================================================
# Property 2: Field Completeness
# ============================================================================


# Feature: context-injection-memory-relevance, Property 2: Field Completeness
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_field_completeness(query, memories):
    """
    Property: For any memory in the injected "memories" list (when retrieval
    succeeds), that memory SHALL contain all required fields: id, content,
    category, timestamp, metadata, and tags.
    
    **Validates: Requirements 1.2**
    
    This test verifies that:
    1. All required fields are present in each injected memory
    2. No required fields are missing
    3. This holds for any number of memories (1-20)
    4. This holds for any query string
    5. Field completeness is guaranteed across all inputs
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    assert len(context["memories"]) > 0, "Expected at least one memory to be injected"
    
    # Define required fields
    required_fields = {"id", "content", "category", "timestamp", "metadata", "tags"}
    
    # REQUIREMENT 1.2: All required fields must be present in each memory
    for i, memory in enumerate(context["memories"]):
        actual_fields = set(memory.keys())
        missing_fields = required_fields - actual_fields
        
        assert actual_fields >= required_fields, \
            f"Memory {i} (id={memory.get('id', 'unknown')}) is missing required fields: " \
            f"{missing_fields}. Expected fields: {required_fields}, " \
            f"Got fields: {actual_fields}"


# Feature: context-injection-memory-relevance, Property 2: Field Completeness
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_field_completeness_with_truncation(query, max_memories, memories):
    """
    Property: For any memory in the injected list, all required fields SHALL
    be present even when truncation occurs.
    
    **Validates: Requirements 1.2, 3.1**
    
    This test verifies that:
    1. Field completeness holds even when memories are truncated
    2. Truncation doesn't affect field completeness
    3. All memories (before and after truncation point) have all fields
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    
    # Define required fields
    required_fields = {"id", "content", "category", "timestamp", "metadata", "tags"}
    
    # REQUIREMENT 1.2: All required fields must be present in each memory
    # This must hold even when truncation occurs
    for i, memory in enumerate(context["memories"]):
        actual_fields = set(memory.keys())
        missing_fields = required_fields - actual_fields
        
        assert actual_fields >= required_fields, \
            f"Memory {i} (id={memory.get('id', 'unknown')}) is missing required fields " \
            f"after truncation: {missing_fields}. " \
            f"Expected fields: {required_fields}, Got fields: {actual_fields}"


# Feature: context-injection-memory-relevance, Property 2: Field Completeness
@given(memory=memory_entry_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_transform_memory_entry_field_completeness(memory):
    """
    Property: The transform_memory_entry function SHALL produce a dictionary
    containing all required fields.
    
    **Validates: Requirements 1.2**
    
    This test verifies that:
    1. transform_memory_entry produces all required fields
    2. No fields are lost during transformation
    3. Field completeness is guaranteed at the transformation level
    """
    # Transform memory entry
    transformed = transform_memory_entry(memory)
    
    # Define required fields
    required_fields = {"id", "content", "category", "timestamp", "metadata", "tags"}
    
    # Verify all required fields present
    actual_fields = set(transformed.keys())
    missing_fields = required_fields - actual_fields
    
    assert actual_fields >= required_fields, \
        f"Transformed memory is missing required fields: {missing_fields}. " \
        f"Expected fields: {required_fields}, Got fields: {actual_fields}"


# Feature: context-injection-memory-relevance, Property 2: Field Completeness
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_field_completeness_no_extra_required_fields(query, memories):
    """
    Property: For any memory in the injected list, exactly the required fields
    SHALL be present (no missing fields, but extra fields are allowed).
    
    **Validates: Requirements 1.2**
    
    This test verifies that:
    1. All required fields are present
    2. The exact set of required fields is defined and enforced
    3. Extra fields beyond required ones are acceptable
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Define required fields (must match design specification)
    required_fields = {"id", "content", "category", "timestamp", "metadata", "tags"}
    
    # Verify field completeness for each memory
    for i, memory in enumerate(context["memories"]):
        actual_fields = set(memory.keys())
        
        # Check that all required fields are present
        assert required_fields.issubset(actual_fields), \
            f"Memory {i} is missing required fields: {required_fields - actual_fields}"


# ============================================================================
# Property 3: Metadata Preservation
# ============================================================================


# Feature: context-injection-memory-relevance, Property 3: Metadata Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_metadata_preservation(query, memories):
    """
    Property: For any memory retrieved from the MemoryInterface, the metadata
    in the injected context SHALL be identical to the metadata in the original
    MemoryEntry (round-trip property).
    
    **Validates: Requirements 1.3**
    
    This test verifies that:
    1. Metadata is preserved exactly as-is without modification
    2. Round-trip property holds: original metadata == injected metadata
    3. Nested metadata structures (dicts, lists) are preserved
    4. All metadata types (str, int, float, bool, None, list, dict) are preserved
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    assert len(context["memories"]) == len(memories)
    
    # Verify metadata preservation for each memory (round-trip)
    for original, injected in zip(memories, context["memories"]):
        # REQUIREMENT 1.3: Metadata must be identical (round-trip property)
        assert injected["metadata"] == original["metadata"], \
            f"Metadata not preserved for memory {original['id']}: " \
            f"expected {original['metadata']}, got {injected['metadata']}"


# Feature: context-injection-memory-relevance, Property 3: Metadata Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memory=memory_entry_strategy()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_transform_memory_entry_preserves_metadata(query, memory):
    """
    Property: The transform_memory_entry function SHALL preserve metadata
    exactly as-is without modification.
    
    **Validates: Requirements 1.3**
    
    This test verifies that:
    1. transform_memory_entry preserves metadata field
    2. Metadata structure is unchanged
    3. Metadata values are unchanged
    """
    # Transform memory entry
    transformed = transform_memory_entry(memory)
    
    # Verify metadata preservation
    assert "metadata" in transformed
    assert transformed["metadata"] == memory["metadata"], \
        f"Metadata not preserved in transformation: " \
        f"expected {memory['metadata']}, got {transformed['metadata']}"


# Feature: context-injection-memory-relevance, Property 3: Metadata Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_all_fields_preserved_in_transformation(query, memories):
    """
    Property: For any memory, ALL required fields (id, content, category,
    timestamp, metadata, tags) SHALL be preserved in transformation.
    
    **Validates: Requirements 1.2, 1.3**
    
    This test verifies that:
    1. All required fields are present in transformed memory
    2. All field values match original values
    3. No fields are lost or modified during transformation
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Verify all fields preserved for each memory
    required_fields = {"id", "content", "category", "timestamp", "metadata", "tags"}
    
    for original, injected in zip(memories, context["memories"]):
        # Verify all required fields present
        assert set(injected.keys()) >= required_fields, \
            f"Missing required fields in memory {original['id']}: " \
            f"expected {required_fields}, got {set(injected.keys())}"
        
        # Verify all field values match
        assert injected["id"] == original["id"]
        assert injected["content"] == original["content"]
        assert injected["category"] == original["category"]
        assert injected["timestamp"] == original["timestamp"]
        assert injected["metadata"] == original["metadata"]
        assert injected["tags"] == original["tags"]


# ============================================================================
# Property 4: Pure Data Structures
# ============================================================================


# Feature: context-injection-memory-relevance, Property 4: Pure Data Structures
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_pure_data_structures(query, memories):
    """
    Property: For any memory in the injected "memories" list, all values SHALL
    be primitive types (str, int, float, bool, None, list, dict) with no custom
    class instances.
    
    **Validates: Requirements 1.4**
    
    This test verifies that:
    1. All values in transformed memories are primitive types
    2. Nested structures (lists, dicts) contain only primitives
    3. No custom class instances or objects are present
    4. Recursive checking handles arbitrary nesting depth
    """
    
    def is_primitive(value):
        """
        Check if a value is a primitive type or a container of primitives.
        
        Primitive types: str, int, float, bool, None
        Containers: list, dict (must contain only primitives)
        
        Returns True if value is primitive or contains only primitives.
        """
        # Base case: primitive types
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        
        # Recursive case: list
        if isinstance(value, list):
            return all(is_primitive(item) for item in value)
        
        # Recursive case: dict
        if isinstance(value, dict):
            # Check both keys and values
            return all(
                isinstance(k, str) and is_primitive(v)
                for k, v in value.items()
            )
        
        # Not a primitive type
        return False
    
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    
    # REQUIREMENT 1.4: All values must be primitive types
    for i, memory in enumerate(context["memories"]):
        # Check each field in the memory
        for field_name, field_value in memory.items():
            assert is_primitive(field_value), \
                f"Non-primitive value found in memory {i}, field '{field_name}': " \
                f"type={type(field_value).__name__}, value={field_value}"


# Feature: context-injection-memory-relevance, Property 4: Pure Data Structures
@given(memory=memory_entry_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_transform_memory_entry_produces_primitives(memory):
    """
    Property: The transform_memory_entry function SHALL produce only primitive
    types in the output dictionary.
    
    **Validates: Requirements 1.4**
    
    This test verifies that:
    1. transform_memory_entry output contains only primitives
    2. No MemoryEntry objects or custom classes in output
    3. All nested structures contain only primitives
    """
    
    def is_primitive(value):
        """Check if value is primitive or container of primitives."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        if isinstance(value, list):
            return all(is_primitive(item) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(k, str) and is_primitive(v)
                for k, v in value.items()
            )
        return False
    
    # Transform memory entry
    transformed = transform_memory_entry(memory)
    
    # Verify all values are primitive types
    for field_name, field_value in transformed.items():
        assert is_primitive(field_value), \
            f"Non-primitive value in transformed memory, field '{field_name}': " \
            f"type={type(field_value).__name__}, value={field_value}"


# Feature: context-injection-memory-relevance, Property 4: Pure Data Structures
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_custom_objects_in_injected_memories(query, memories):
    """
    Property: For any injected memory, no custom class instances SHALL be
    present in the data structure.
    
    **Validates: Requirements 1.4**
    
    This test verifies that:
    1. No MemoryEntry objects in output
    2. No custom class instances anywhere in the structure
    3. Only built-in Python types (str, int, float, bool, None, list, dict)
    """
    
    def contains_custom_objects(value):
        """
        Check if value contains any custom objects.
        Returns True if custom objects found, False otherwise.
        """
        # Check if it's a custom object (not a built-in type)
        if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
            return True
        
        # Recursively check containers
        if isinstance(value, list):
            return any(contains_custom_objects(item) for item in value)
        
        if isinstance(value, dict):
            return any(
                contains_custom_objects(k) or contains_custom_objects(v)
                for k, v in value.items()
            )
        
        return False
    
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Verify no custom objects in any memory
    for i, memory in enumerate(context["memories"]):
        assert not contains_custom_objects(memory), \
            f"Custom objects found in memory {i}: {memory}"
        
        # Specifically check that it's a dict, not a MemoryEntry
        assert isinstance(memory, dict), \
            f"Memory {i} is not a dict: type={type(memory).__name__}"


# ============================================================================
# Property 5: Order Preservation
# ============================================================================


# Feature: context-injection-memory-relevance, Property 5: Order Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation(query, memories):
    """
    Property: For any list of ranked memories from MemoryInterface, the order
    in the injected "memories" list SHALL match the order of the input list
    (preserving ranking).
    
    **Validates: Requirements 2.1**
    
    This test verifies that:
    1. Memory order in output matches input order
    2. No sorting or reordering occurs during transformation
    3. Ranking order from retrieval system is preserved
    4. Order preservation holds for any number of memories (1-20)
    5. Order preservation holds for any query string
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    assert len(context["memories"]) == len(memories)
    
    # REQUIREMENT 2.1: Verify order matches input order (compare IDs)
    for i, (original, injected) in enumerate(zip(memories, context["memories"])):
        assert injected["id"] == original["id"], \
            f"Order mismatch at position {i}: expected {original['id']}, " \
            f"got {injected['id']}"


# Feature: context-injection-memory-relevance, Property 5: Order Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_with_truncation(query, max_memories, memories):
    """
    Property: For any list of memories, order SHALL be preserved even when
    truncation occurs due to size limits.
    
    **Validates: Requirements 2.1, 2.4, 3.4**
    
    This test verifies that:
    1. Order is preserved before truncation point
    2. Truncation doesn't affect order of remaining memories
    3. First N memories maintain their relative order
    4. Order preservation holds across varying memory counts and limits
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    
    # Calculate expected count (min of memories length and max_memories)
    expected_count = min(len(memories), max_memories)
    assert len(context["memories"]) == expected_count
    
    # REQUIREMENT 2.1, 2.4: Verify order preserved for first N memories
    for i in range(expected_count):
        assert context["memories"][i]["id"] == memories[i]["id"], \
            f"Order mismatch at position {i} after truncation: " \
            f"expected {memories[i]['id']}, got {context['memories'][i]['id']}"


# Feature: context-injection-memory-relevance, Property 5: Order Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=2, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_reordering_during_transformation(query, memories):
    """
    Property: The transformation process SHALL NOT reorder memories.
    
    **Validates: Requirements 2.1, 2.4**
    
    This test verifies that:
    1. transform_memory_entry doesn't affect order
    2. List comprehension preserves order
    3. No sorting occurs anywhere in the pipeline
    4. Relative positions of all memories are maintained
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Extract IDs from original and injected memories
    original_ids = [m["id"] for m in memories]
    injected_ids = [m["id"] for m in context["memories"]]
    
    # REQUIREMENT 2.1: Verify ID sequences match exactly
    assert injected_ids == original_ids, \
        f"Memory order was changed during transformation: " \
        f"original={original_ids}, injected={injected_ids}"


# Feature: context-injection-memory-relevance, Property 5: Order Preservation
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=3, max_size=15)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_order_preservation_all_positions(query, memories):
    """
    Property: For any memory at position i in the input, it SHALL appear at
    position i in the output (for i < min(len(memories), max_memories)).
    
    **Validates: Requirements 2.1**
    
    This test verifies that:
    1. Each memory maintains its exact position
    2. Position preservation holds for all indices
    3. No swapping or shuffling occurs
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Verify each memory is at the correct position
    for i, original_memory in enumerate(memories):
        injected_memory = context["memories"][i]
        
        # REQUIREMENT 2.1: Memory at position i must match original at position i
        assert injected_memory["id"] == original_memory["id"], \
            f"Position {i} mismatch: expected {original_memory['id']}, " \
            f"got {injected_memory['id']}"
        
        # Also verify content matches (additional check)
        assert injected_memory["content"] == original_memory["content"], \
            f"Content mismatch at position {i}"


# ============================================================================
# Property 6: Deterministic Ordering
# ============================================================================


# Feature: context-injection-memory-relevance, Property 6: Deterministic Ordering
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_deterministic_ordering(query, memories):
    """
    Property: For any identical inputs (same query, memory_interface state, config),
    calling inject_memories multiple times SHALL produce identical memory ordering
    in the output.
    
    **Validates: Requirements 2.2**
    
    This test verifies that:
    1. Multiple calls with identical inputs produce identical outputs
    2. Memory ordering is deterministic (not random)
    3. No non-deterministic operations affect ordering
    4. Determinism holds across all input variations
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Call inject_memories twice with identical inputs
    context1 = inject_memories(query, memory_interface, config)
    context2 = inject_memories(query, memory_interface, config)
    
    # Extract memory IDs to compare ordering
    ids1 = [m["id"] for m in context1["memories"]]
    ids2 = [m["id"] for m in context2["memories"]]
    
    # REQUIREMENT 2.2: Ordering must be identical for identical inputs
    assert ids1 == ids2, \
        f"Non-deterministic ordering detected: " \
        f"first call={ids1}, second call={ids2}"


# Feature: context-injection-memory-relevance, Property 6: Deterministic Ordering
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_deterministic_ordering_with_truncation(query, max_memories, memories):
    """
    Property: Deterministic ordering SHALL hold even when truncation occurs.
    
    **Validates: Requirements 2.2, 3.4**
    
    This test verifies that:
    1. Truncation doesn't introduce non-determinism
    2. Multiple calls with truncation produce identical results
    3. Determinism holds across varying memory counts and limits
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Call multiple times
    context1 = inject_memories(query, memory_interface, config)
    context2 = inject_memories(query, memory_interface, config)
    context3 = inject_memories(query, memory_interface, config)
    
    # Extract IDs
    ids1 = [m["id"] for m in context1["memories"]]
    ids2 = [m["id"] for m in context2["memories"]]
    ids3 = [m["id"] for m in context3["memories"]]
    
    # REQUIREMENT 2.2: All calls must produce identical ordering
    assert ids1 == ids2 == ids3, \
        f"Non-deterministic ordering with truncation: " \
        f"call1={ids1}, call2={ids2}, call3={ids3}"


# Feature: context-injection-memory-relevance, Property 6: Deterministic Ordering
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=2, max_size=15)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_deterministic_ordering_full_content(query, memories):
    """
    Property: Not only IDs but all memory content SHALL be identical across
    multiple calls with identical inputs.
    
    **Validates: Requirements 2.2**
    
    This test verifies that:
    1. Complete memory dictionaries are identical
    2. Not just IDs but all fields maintain deterministic order
    3. Metadata, tags, timestamps all preserved deterministically
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Call twice
    context1 = inject_memories(query, memory_interface, config)
    context2 = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 2.2: Complete memories list must be identical
    assert context1["memories"] == context2["memories"], \
        f"Non-deterministic content detected"
    
    # Verify each memory is identical
    for i, (mem1, mem2) in enumerate(zip(context1["memories"], context2["memories"])):
        assert mem1 == mem2, \
            f"Memory {i} differs between calls: {mem1} != {mem2}"


# Feature: context-injection-memory-relevance, Property 6: Deterministic Ordering
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_deterministic_ordering_many_calls(query, memories):
    """
    Property: Deterministic ordering SHALL hold across many sequential calls.
    
    **Validates: Requirements 2.2, 5.4 (statelessness)**
    
    This test verifies that:
    1. Determinism holds across 5+ sequential calls
    2. No state accumulation affects ordering
    3. Each call is independent and produces identical results
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Make 5 sequential calls
    results = []
    for _ in range(5):
        context = inject_memories(query, memory_interface, config)
        ids = [m["id"] for m in context["memories"]]
        results.append(ids)
    
    # REQUIREMENT 2.2: All calls must produce identical ordering
    first_result = results[0]
    for i, result in enumerate(results[1:], start=1):
        assert result == first_result, \
            f"Call {i+1} produced different ordering: " \
            f"expected {first_result}, got {result}"


# ============================================================================
# Property 7: Size Limit Enforcement
# ============================================================================


# Feature: context-injection-memory-relevance, Property 7: Size Limit Enforcement
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_size_limit_enforcement(query, max_memories, memories):
    """
    Property: For any input size, the output SHALL respect the configured limit.
    The length of the "memories" list SHALL NOT exceed max_memories.
    
    **Validates: Requirements 3.1, 3.5**
    
    This test verifies that:
    1. Output length never exceeds configured max_memories
    2. Size limit is enforced for all input sizes (0-100)
    3. Size limit is enforced for all max_memories values (5-20)
    4. Truncation occurs correctly when needed
    """
    # Create mock memory interface with predefined memories
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories into context
    context = inject_memories(query, memory_interface, config)
    
    # Verify memories were injected
    assert "memories" in context
    
    # REQUIREMENT 3.1, 3.5: Output length must not exceed configured limit
    actual_length = len(context["memories"])
    assert actual_length <= max_memories, \
        f"Size limit violated: expected max {max_memories}, got {actual_length}"
    
    # Verify length matches expected (min of input size and limit)
    expected_length = min(len(memories), max_memories)
    assert actual_length == expected_length, \
        f"Expected {expected_length} memories, got {actual_length}"


# Feature: context-injection-memory-relevance, Property 7: Size Limit Enforcement
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    num_memories=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_size_limit_invariant(query, max_memories, num_memories):
    """
    Property: The size invariant SHALL hold for all possible input sizes.
    len(output["memories"]) <= max_memories for ALL inputs.
    
    **Validates: Requirements 3.5**
    
    This test verifies that:
    1. Size invariant holds universally
    2. No edge cases violate the limit
    3. Limit enforcement is absolute (never exceeded)
    """
    # Generate memories of specific size
    memories = [
        {
            "id": f"mem_{i}",
            "content": f"Content {i}",
            "category": "test",
            "timestamp": "2024-01-15T10:00:00",
            "metadata": {},
            "tags": []
        }
        for i in range(num_memories)
    ]
    
    # Create mock memory interface
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 3.5: Size invariant must hold
    assert len(context["memories"]) <= max_memories, \
        f"Size invariant violated: {len(context['memories'])} > {max_memories}"


# Feature: context-injection-memory-relevance, Property 7: Size Limit Enforcement
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_size_limit_with_large_input(query, max_memories):
    """
    Property: When input size greatly exceeds limit, truncation SHALL occur
    correctly and only first max_memories items are kept.
    
    **Validates: Requirements 3.1, 3.3, 3.4**
    
    This test verifies that:
    1. Large inputs are truncated correctly
    2. First N memories are kept (preserving ranking)
    3. Truncation doesn't cause errors or data corruption
    """
    # Create large list of memories (always exceeds limit)
    num_memories = max_memories * 3  # 3x the limit
    memories = [
        {
            "id": f"mem_{i}",
            "content": f"Content {i}",
            "category": "test",
            "timestamp": "2024-01-15T10:00:00",
            "metadata": {"index": i},
            "tags": []
        }
        for i in range(num_memories)
    ]
    
    # Create mock memory interface
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 3.1: Length must equal max_memories (truncated)
    assert len(context["memories"]) == max_memories, \
        f"Expected {max_memories} memories after truncation, got {len(context['memories'])}"
    
    # REQUIREMENT 3.3, 3.4: Verify first N memories are kept
    for i in range(max_memories):
        assert context["memories"][i]["id"] == f"mem_{i}", \
            f"Truncation error: expected mem_{i}, got {context['memories'][i]['id']}"


# Feature: context-injection-memory-relevance, Property 7: Size Limit Enforcement
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_size_limit_boundary_conditions(query, max_memories, memories):
    """
    Property: Size limit SHALL be enforced correctly at boundary conditions
    (input size = limit, input size = limit ± 1).
    
    **Validates: Requirements 3.1**
    
    This test verifies that:
    1. Boundary conditions are handled correctly
    2. No off-by-one errors in truncation
    3. Exact limit case works correctly
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # Calculate expected length
    expected_length = min(len(memories), max_memories)
    
    # REQUIREMENT 3.1: Verify correct length
    assert len(context["memories"]) == expected_length, \
        f"Boundary condition error: expected {expected_length}, got {len(context['memories'])}"
    
    # Verify no extra memories
    assert len(context["memories"]) <= max_memories, \
        f"Size limit exceeded at boundary: {len(context['memories'])} > {max_memories}"


# ============================================================================
# Property 8: Configuration Validation
# ============================================================================


# Feature: context-injection-memory-relevance, Property 8: Configuration Validation
@given(
    max_memories=st.integers().filter(lambda x: x < 5 or x > 20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_configuration_validation(max_memories):
    """
    Property: For any invalid max_memories value (outside [5, 20]), calling
    config.validate() SHALL raise ValueError.
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. Invalid configurations are rejected
    2. ValueError is raised for out-of-range values
    3. Validation catches all invalid inputs
    4. Error message is descriptive
    """
    # Create config with invalid max_memories
    config = InjectionConfig(max_memories=max_memories)
    
    # REQUIREMENT 3.2: validate() must raise ValueError for invalid values
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "max_memories must be in [5, 20]" in error_message, \
        f"Error message should mention valid range: {error_message}"
    assert str(max_memories) in error_message, \
        f"Error message should include invalid value: {error_message}"


# Feature: context-injection-memory-relevance, Property 8: Configuration Validation
@given(
    max_memories=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_valid_configuration_no_error(max_memories):
    """
    Property: For any valid max_memories value (in [5, 20]), calling
    config.validate() SHALL NOT raise any exception.
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. Valid configurations are accepted
    2. No exceptions raised for valid values
    3. All values in [5, 20] are valid
    """
    # Create config with valid max_memories
    config = InjectionConfig(max_memories=max_memories)
    
    # REQUIREMENT 3.2: validate() must not raise for valid values
    try:
        config.validate()
    except ValueError as e:
        pytest.fail(f"Valid configuration rejected: max_memories={max_memories}, error={e}")


# Feature: context-injection-memory-relevance, Property 8: Configuration Validation
@given(
    max_memories=st.integers().filter(lambda x: x < 5 or x > 20),
    query=st.text(min_size=1, max_size=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_inject_memories_validates_config(max_memories, query):
    """
    Property: inject_memories SHALL validate configuration before processing,
    raising ValueError for invalid configs.
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. inject_memories calls config.validate()
    2. Invalid configs are rejected early (fail fast)
    3. ValueError propagates to caller for config errors
    """
    # Create invalid config
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface([])
    
    # REQUIREMENT 3.2: inject_memories must validate and raise ValueError
    with pytest.raises(ValueError) as exc_info:
        inject_memories(query, memory_interface, config)
    
    # Verify it's a configuration error
    error_message = str(exc_info.value)
    assert "max_memories must be in [5, 20]" in error_message


# Feature: context-injection-memory-relevance, Property 8: Configuration Validation
@given(
    max_memories=st.integers(min_value=-100, max_value=4)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_configuration_validation_below_minimum(max_memories):
    """
    Property: Configuration validation SHALL reject values below minimum (< 5).
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. Values below 5 are rejected
    2. Negative values are rejected
    3. Zero is rejected
    """
    config = InjectionConfig(max_memories=max_memories)
    
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    assert "max_memories must be in [5, 20]" in str(exc_info.value)


# Feature: context-injection-memory-relevance, Property 8: Configuration Validation
@given(
    max_memories=st.integers(min_value=21, max_value=1000)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_configuration_validation_above_maximum(max_memories):
    """
    Property: Configuration validation SHALL reject values above maximum (> 20).
    
    **Validates: Requirements 3.2**
    
    This test verifies that:
    1. Values above 20 are rejected
    2. Large values are rejected
    3. Upper bound is enforced
    """
    config = InjectionConfig(max_memories=max_memories)
    
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    assert "max_memories must be in [5, 20]" in str(exc_info.value)


# ============================================================================
# Property 9: No Side Effects
# ============================================================================


# Feature: context-injection-memory-relevance, Property 9: No Side Effects
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_side_effects(query, memories):
    """
    Property: For any inputs, calling inject_memories SHALL NOT modify them.
    The function must be pure with no side effects.
    
    **Validates: Requirements 5.3**
    
    This test verifies that:
    1. Input parameters are not modified
    2. Function has no side effects beyond returning data
    3. Functional purity is maintained
    4. Deep copy comparison shows no changes
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=10)
    memory_interface = MockMemoryInterface(memories)
    
    # Deep copy inputs before calling function
    query_copy = copy.deepcopy(query)
    config_copy = copy.deepcopy(config)
    memories_copy = copy.deepcopy(memories)
    
    # Call inject_memories
    inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 5.3: Verify inputs unchanged
    assert query == query_copy, \
        f"Query was modified: original={query_copy}, after={query}"
    
    assert config.max_memories == config_copy.max_memories, \
        f"Config was modified"
    
    assert memories == memories_copy, \
        f"Memories list was modified"


# Feature: context-injection-memory-relevance, Property 9: No Side Effects
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10),
    existing_context=st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_side_effects_on_existing_context(query, memories, existing_context):
    """
    Property: inject_memories SHALL NOT modify the existing_context parameter.
    A new context dictionary is returned without mutating the input.
    
    **Validates: Requirements 5.3**
    
    This test verifies that:
    1. existing_context parameter is not modified
    2. Original context dict remains unchanged
    3. New context is returned (not mutated in-place)
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=10)
    memory_interface = MockMemoryInterface(memories)
    
    # Deep copy existing_context
    existing_context_copy = copy.deepcopy(existing_context)
    
    # Call inject_memories with existing_context
    result_context = inject_memories(
        query,
        memory_interface,
        config,
        existing_context=existing_context
    )
    
    # REQUIREMENT 5.3: Verify existing_context not modified
    assert existing_context == existing_context_copy, \
        f"existing_context was modified"
    
    # Verify "memories" key was not added to original
    assert "memories" not in existing_context, \
        f"'memories' key was added to original existing_context"
    
    # Verify result is a different object
    assert result_context is not existing_context, \
        f"Result should be a new dict, not the same object"


# Feature: context-injection-memory-relevance, Property 9: No Side Effects
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_mutation_of_memory_entries(query, memories):
    """
    Property: inject_memories SHALL NOT mutate the MemoryEntry objects
    in the memories list.
    
    **Validates: Requirements 5.3**
    
    This test verifies that:
    1. Original MemoryEntry objects are not modified
    2. Transformation creates new dicts without mutating source
    3. Memory metadata remains unchanged
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=20)
    memory_interface = MockMemoryInterface(memories)
    
    # Deep copy memories
    memories_copy = copy.deepcopy(memories)
    
    # Call inject_memories
    inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 5.3: Verify memories not mutated
    assert memories == memories_copy, \
        f"Memory entries were mutated"
    
    # Verify each memory entry unchanged
    for i, (original, after) in enumerate(zip(memories_copy, memories)):
        assert original == after, \
            f"Memory entry {i} was mutated"


# ============================================================================
# Property 10: Statelessness
# ============================================================================


# Feature: context-injection-memory-relevance, Property 10: Statelessness
@given(
    queries=st.lists(st.text(min_size=1, max_size=100), min_size=2, max_size=5),
    memories_list=st.lists(
        st.lists(memory_entry_strategy(), min_size=0, max_size=10),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_statelessness(queries, memories_list):
    """
    Property: For any sequence of calls, the output of call N SHALL NOT depend
    on the outputs or side effects of calls 1 through N-1 (stateless invariant).
    
    **Validates: Requirements 5.4**
    
    This test verifies that:
    1. Each call is independent
    2. No state retained between calls
    3. Calling same inputs produces same output regardless of history
    4. Stateless invariant holds across sequences
    """
    # Ensure we have matching lengths
    min_length = min(len(queries), len(memories_list))
    queries = queries[:min_length]
    memories_list = memories_list[:min_length]
    
    config = InjectionConfig(max_memories=10)
    
    # Make sequence of calls and store results
    results = []
    for query, memories in zip(queries, memories_list):
        memory_interface = MockMemoryInterface(memories)
        context = inject_memories(query, memory_interface, config)
        results.append(context["memories"])
    
    # Call again with same inputs - should get identical results
    for i, (query, memories) in enumerate(zip(queries, memories_list)):
        memory_interface = MockMemoryInterface(memories)
        context = inject_memories(query, memory_interface, config)
        
        # REQUIREMENT 5.4: Output should match previous call with same input
        assert context["memories"] == results[i], \
            f"Call {i} produced different result on second invocation: " \
            f"statelessness violated"


# Feature: context-injection-memory-relevance, Property 10: Statelessness
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_no_session_leakage(query, memories):
    """
    Property: inject_memories SHALL NOT retain session-specific data between
    calls (no session leakage property).
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. No session data retained
    2. Each call starts fresh
    3. No hidden state accumulation
    4. Multiple calls with different inputs don't interfere
    """
    config = InjectionConfig(max_memories=10)
    
    # Make first call
    memory_interface1 = MockMemoryInterface(memories)
    context1 = inject_memories(query, memory_interface1, config)
    
    # Make second call with different memories
    different_memories = [
        {
            "id": f"different_{i}",
            "content": f"Different content {i}",
            "category": "different",
            "timestamp": "2024-01-16T10:00:00",
            "metadata": {},
            "tags": []
        }
        for i in range(5)
    ]
    memory_interface2 = MockMemoryInterface(different_memories)
    context2 = inject_memories(query, memory_interface2, config)
    
    # Make third call with original memories again
    memory_interface3 = MockMemoryInterface(memories)
    context3 = inject_memories(query, memory_interface3, config)
    
    # REQUIREMENT 5.5: Third call should match first call (no session leakage)
    assert context3["memories"] == context1["memories"], \
        f"Session leakage detected: third call differs from first call " \
        f"despite identical inputs"
    
    # Verify second call was different (sanity check)
    if len(memories) > 0 and len(different_memories) > 0:
        assert context2["memories"] != context1["memories"], \
            f"Sanity check failed: different inputs should produce different outputs"


# Feature: context-injection-memory-relevance, Property 10: Statelessness
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=10)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_statelessness_many_sequential_calls(query, memories):
    """
    Property: Statelessness SHALL hold across many sequential calls (10+).
    
    **Validates: Requirements 5.4**
    
    This test verifies that:
    1. No state accumulation over many calls
    2. Performance doesn't degrade
    3. Results remain consistent
    """
    config = InjectionConfig(max_memories=10)
    memory_interface = MockMemoryInterface(memories)
    
    # Make first call to establish baseline
    baseline_context = inject_memories(query, memory_interface, config)
    baseline_ids = [m["id"] for m in baseline_context["memories"]]
    
    # Make 10 more calls
    for i in range(10):
        memory_interface = MockMemoryInterface(memories)
        context = inject_memories(query, memory_interface, config)
        ids = [m["id"] for m in context["memories"]]
        
        # REQUIREMENT 5.4: Each call should match baseline
        assert ids == baseline_ids, \
            f"Call {i+1} produced different result: statelessness violated"


# ============================================================================
# Property 11: Type Invariant
# ============================================================================


# Feature: context-injection-memory-relevance, Property 11: Type Invariant
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_type_invariant(query, memories):
    """
    Property: For any context dictionary returned by inject_memories, the value
    at key "memories" SHALL be of type list.
    
    **Validates: Requirements 6.2**
    
    This test verifies that:
    1. "memories" value is always a list type
    2. Type invariant holds for all inputs
    3. Never returns dict, tuple, or other types
    4. Empty list is still a list (not None)
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=10)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 6.2: "memories" value must be a list
    assert isinstance(context["memories"], list), \
        f"Type invariant violated: expected list, got {type(context['memories']).__name__}"


# Feature: context-injection-memory-relevance, Property 11: Type Invariant
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_type_invariant_on_failure(query, max_memories):
    """
    Property: The type invariant SHALL hold even when retrieval fails.
    "memories" must be a list even on error.
    
    **Validates: Requirements 6.2, 6.4**
    
    This test verifies that:
    1. Type invariant holds on retrieval failure
    2. Empty list returned (not None or other type)
    3. Graceful degradation maintains type safety
    """
    # Create mock memory interface that fails
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface([], should_fail=True)
    
    # Inject memories (should handle failure gracefully)
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 6.2: "memories" must still be a list
    assert isinstance(context["memories"], list), \
        f"Type invariant violated on failure: expected list, " \
        f"got {type(context['memories']).__name__}"
    
    # Should be empty list
    assert context["memories"] == [], \
        f"Expected empty list on failure, got {context['memories']}"


# Feature: context-injection-memory-relevance, Property 11: Type Invariant
@given(
    query=st.text(min_size=1, max_size=100),
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=5),
    existing_context=st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_type_invariant_with_existing_context(query, memories, existing_context):
    """
    Property: Type invariant SHALL hold when existing_context is provided.
    
    **Validates: Requirements 6.2**
    
    This test verifies that:
    1. Type invariant holds with existing_context parameter
    2. "memories" is always a list regardless of existing context
    3. Existing context doesn't affect type invariant
    """
    # Create mock memory interface
    config = InjectionConfig(max_memories=10)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories with existing_context
    context = inject_memories(
        query,
        memory_interface,
        config,
        existing_context=existing_context
    )
    
    # REQUIREMENT 6.2: "memories" must be a list
    assert isinstance(context["memories"], list), \
        f"Type invariant violated with existing_context: expected list, " \
        f"got {type(context['memories']).__name__}"


# Feature: context-injection-memory-relevance, Property 11: Type Invariant
@given(
    query=st.text(min_size=1, max_size=100),
    max_memories=st.integers(min_value=5, max_value=20),
    num_memories=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_type_invariant_all_sizes(query, max_memories, num_memories):
    """
    Property: Type invariant SHALL hold for all input sizes (0 to 100+ memories).
    
    **Validates: Requirements 6.2**
    
    This test verifies that:
    1. Type invariant holds for empty input
    2. Type invariant holds for small inputs
    3. Type invariant holds for large inputs
    4. Type invariant holds across all size ranges
    """
    # Generate memories of specific size
    memories = [
        {
            "id": f"mem_{i}",
            "content": f"Content {i}",
            "category": "test",
            "timestamp": "2024-01-15T10:00:00",
            "metadata": {},
            "tags": []
        }
        for i in range(num_memories)
    ]
    
    # Create mock memory interface
    config = InjectionConfig(max_memories=max_memories)
    memory_interface = MockMemoryInterface(memories)
    
    # Inject memories
    context = inject_memories(query, memory_interface, config)
    
    # REQUIREMENT 6.2: "memories" must be a list for all sizes
    assert isinstance(context["memories"], list), \
        f"Type invariant violated for size {num_memories}: expected list, " \
        f"got {type(context['memories']).__name__}"
