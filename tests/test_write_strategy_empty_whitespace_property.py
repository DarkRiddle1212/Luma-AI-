"""
Property-Based Test for Empty and Whitespace Rejection

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly rejects empty and whitespace-only
messages without storage.

Feature: memory-write-strategy-session-management
Property 2: Empty and whitespace rejection
Validates: Requirements 1.2
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
# Property 2: Empty and Whitespace Rejection
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 2: Empty and whitespace rejection
@given(
    whitespace_content=st.one_of(
        st.just(""),  # Empty string
        st.text(alphabet=st.sampled_from([' ', '\t', '\n', '\r', '\f', '\v']), min_size=1, max_size=5)  # Whitespace only
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_2_empty_and_whitespace_rejection(whitespace_content):
    """
    Property: For any string composed entirely of whitespace characters
    (including empty string), the Memory_Write_Strategy should reject it
    without storage.

    **Validates: Requirements 1.2**

    This test verifies that:
    1. Empty strings are rejected
    2. Whitespace-only strings are rejected
    3. The decision includes should_write=False
    4. The reason is "empty_or_whitespace"
    5. The message is NOT added to recent_messages
    """
    # Create write strategy with default config
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Track initial recent_messages count
        initial_recent_count = len(strategy.recent_messages)

        # Evaluate the empty/whitespace message
        decision = strategy.evaluate_write_trigger(whitespace_content)

        # Property 1: should_write should be False for empty/whitespace content
        assert decision.should_write is False, \
            f"Empty/whitespace message should be rejected, got should_write={decision.should_write}, content='{whitespace_content}'"

        # Property 2: reason should be "empty_or_whitespace"
        assert decision.reason == "empty_or_whitespace", \
            f"Expected reason 'empty_or_whitespace', got '{decision.reason}'"

        # Property 3: metadata should contain content_length
        assert "content_length" in decision.metadata, \
            f"Decision metadata should contain 'content_length', got {decision.metadata}"

        # Property 4: content_length should match actual length
        expected_length = len(whitespace_content) if whitespace_content else 0
        assert decision.metadata["content_length"] == expected_length, \
            f"Expected content_length={expected_length}, got {decision.metadata['content_length']}"

        # Property 5: message should NOT be added to recent_messages
        assert len(strategy.recent_messages) == initial_recent_count, \
            f"Empty/whitespace message should not be added to recent_messages"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 2: Empty and whitespace rejection
@given(
    whitespace_type=st.sampled_from([
        ' ',      # Space
        '\t',     # Tab
        '\n',     # Newline
        '\r',     # Carriage return
        '\f',     # Form feed
        '\v',     # Vertical tab
        '  ',     # Multiple spaces
        '\t\t',   # Multiple tabs
        '\n\n',   # Multiple newlines
        ' \t\n',  # Mixed whitespace
    ]),
    count=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_2_various_whitespace_types_rejection(whitespace_type, count):
    """
    Property: For any combination of whitespace characters (spaces, tabs,
    newlines, etc.), the Memory_Write_Strategy should reject them without
    storage.

    **Validates: Requirements 1.2**

    This test verifies that:
    1. All types of whitespace characters are rejected
    2. Multiple whitespace characters are rejected
    3. Mixed whitespace characters are rejected
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Create whitespace content by repeating the whitespace type
        whitespace_content = whitespace_type * count

        # Evaluate the whitespace message
        decision = strategy.evaluate_write_trigger(whitespace_content)

        # Property 1: should_write should be False
        assert decision.should_write is False, \
            f"Whitespace message should be rejected, got should_write={decision.should_write}"

        # Property 2: reason should be "empty_or_whitespace"
        assert decision.reason == "empty_or_whitespace", \
            f"Expected reason 'empty_or_whitespace', got '{decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 2: Empty and whitespace rejection
@given(
    leading_whitespace=st.text(alphabet=st.sampled_from([' ', '\t', '\n']), min_size=0, max_size=10),
    trailing_whitespace=st.text(alphabet=st.sampled_from([' ', '\t', '\n']), min_size=0, max_size=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_2_whitespace_with_leading_trailing(leading_whitespace, trailing_whitespace):
    """
    Property: For any string that contains only whitespace (with leading
    and trailing whitespace), the Memory_Write_Strategy should reject it.

    **Validates: Requirements 1.2**

    This test verifies that:
    1. Strings with only leading whitespace are rejected
    2. Strings with only trailing whitespace are rejected
    3. Strings with both leading and trailing whitespace are rejected
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Create content with leading and trailing whitespace only
        whitespace_content = leading_whitespace + trailing_whitespace

        # Evaluate the whitespace message
        decision = strategy.evaluate_write_trigger(whitespace_content)

        # Property 1: should_write should be False
        assert decision.should_write is False, \
            f"Whitespace-only message should be rejected, got should_write={decision.should_write}"

        # Property 2: reason should be "empty_or_whitespace"
        assert decision.reason == "empty_or_whitespace", \
            f"Expected reason 'empty_or_whitespace', got '{decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 2: Empty and whitespace rejection
@given(
    metadata=st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(st.text(), st.integers(), st.booleans())
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_2_empty_string_with_metadata_rejection(metadata):
    """
    Property: For an empty string with any metadata, the Memory_Write_Strategy
    should still reject it without storage.

    **Validates: Requirements 1.2**

    This test verifies that:
    1. Empty strings are rejected regardless of metadata
    2. Metadata does not affect empty string rejection
    3. The evaluation is based solely on content, not metadata
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Evaluate empty string with metadata
        decision = strategy.evaluate_write_trigger("", metadata)

        # Property 1: should_write should be False
        assert decision.should_write is False, \
            f"Empty string should be rejected regardless of metadata, got should_write={decision.should_write}"

        # Property 2: reason should be "empty_or_whitespace"
        assert decision.reason == "empty_or_whitespace", \
            f"Expected reason 'empty_or_whitespace', got '{decision.reason}'"

        # Property 3: content_length should be 0
        assert decision.metadata["content_length"] == 0, \
            f"Expected content_length=0, got {decision.metadata['content_length']}"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 2: Empty and whitespace rejection
@pytest.mark.property_test
def test_property_2_empty_string_not_in_recent_messages():
    """
    Property: When empty or whitespace-only messages are rejected, they
    should never be added to the recent_messages list for repetition
    detection.

    **Validates: Requirements 1.2**

    This test verifies that:
    1. Rejected messages don't pollute recent_messages
    2. The recent_messages list remains clean for valid repetition detection
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Test various empty/whitespace strings
        test_cases = [
            "",
            " ",
            "  ",
            "\t",
            "\n",
            "\r\n",
            "   \t\n   ",
            "\t\t\t",
            "\n\n\n"
        ]

        for content in test_cases:
            initial_count = len(strategy.recent_messages)
            decision = strategy.evaluate_write_trigger(content)

            # Property 1: should_write should be False
            assert decision.should_write is False, \
                f"Content '{repr(content)}' should be rejected"

            # Property 2: recent_messages should not grow
            assert len(strategy.recent_messages) == initial_count, \
                f"recent_messages should not grow for rejected content '{repr(content)}'"

    finally:
        session_manager.shutdown()
