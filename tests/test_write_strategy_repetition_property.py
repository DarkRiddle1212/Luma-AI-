"""
Property-Based Test for Repetition Detection

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly detects and rejects repetitive
messages (identical to recent messages within the repetition window).

Feature: memory-write-strategy-session-management
Property 3: Repetition detection
Validates: Requirements 1.4
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
# Property 3: Repetition Detection
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    content=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=3,
        max_size=100
    ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ]),
    repetition_window=st.integers(min_value=1, max_value=10),
    repeat_position=st.integers(min_value=0, max_value=9)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_repetition_detection_within_window(content, repetition_window, repeat_position):
    """
    Property: For any sequence of messages where a message is identical to one
    of the last N messages (within repetition window), the Memory_Write_Strategy
    should reject the duplicate message.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. Messages identical to recent messages are detected
    2. The detection works within the configured repetition window
    3. The decision includes should_write=False
    4. The reason is "repetitive"
    5. The repetitive message is NOT added to recent_messages
    """
    # Ensure repeat_position is within the window
    assume(repeat_position < repetition_window)
    
    # Create write strategy with custom repetition window
    config = WriteStrategyConfig(repetition_window=repetition_window)
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Fill the recent messages with unique content up to repeat_position
        for i in range(repeat_position + 1):
            unique_content = f"{content}_unique_{i}"
            decision = strategy.evaluate_write_trigger(unique_content)
            assert decision.should_write is True, \
                f"Unique message {i} should be approved"

        # Now try to repeat the message at repeat_position
        repeated_content = f"{content}_unique_{repeat_position}"
        
        # Track recent_messages count before repetition
        recent_count_before = len(strategy.recent_messages)
        
        # Evaluate the repeated message
        decision = strategy.evaluate_write_trigger(repeated_content)

        # Property 1: should_write should be False for repetitive content
        assert decision.should_write is False, \
            f"Repetitive message should be rejected, got should_write={decision.should_write}, reason={decision.reason}"

        # Property 2: reason should be "repetitive"
        assert decision.reason == "repetitive", \
            f"Expected reason 'repetitive', got '{decision.reason}'"

        # Property 3: metadata should contain repetition_window
        assert "repetition_window" in decision.metadata, \
            f"Decision metadata should contain 'repetition_window', got {decision.metadata}"

        # Property 4: repetitive message should NOT be added to recent_messages
        assert len(strategy.recent_messages) == recent_count_before, \
            f"Repetitive message should not be added to recent_messages"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    content=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=3,
        max_size=100
    ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ]),
    num_messages=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_immediate_repetition_detection(content, num_messages):
    """
    Property: When the same message is sent multiple times in a row,
    only the first occurrence should be approved, and all subsequent
    identical messages should be rejected as repetitive.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. The first message is approved
    2. All subsequent identical messages are rejected
    3. Each rejection has reason "repetitive"
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # First message should be approved
        first_decision = strategy.evaluate_write_trigger(content)
        assert first_decision.should_write is True, \
            f"First message should be approved, got should_write={first_decision.should_write}"
        assert first_decision.reason == "approved", \
            f"First message should have reason 'approved', got '{first_decision.reason}'"

        # All subsequent identical messages should be rejected
        for i in range(1, num_messages):
            decision = strategy.evaluate_write_trigger(content)
            
            # Property 1: should_write should be False
            assert decision.should_write is False, \
                f"Repetitive message {i} should be rejected, got should_write={decision.should_write}"
            
            # Property 2: reason should be "repetitive"
            assert decision.reason == "repetitive", \
                f"Expected reason 'repetitive' for message {i}, got '{decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    base_content=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=3,
        max_size=50
    ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ]),
    leading_whitespace=st.text(alphabet=st.sampled_from([' ', '\t']), min_size=0, max_size=5),
    trailing_whitespace=st.text(alphabet=st.sampled_from([' ', '\t']), min_size=0, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_repetition_detection_ignores_whitespace(base_content, leading_whitespace, trailing_whitespace):
    """
    Property: Repetition detection should normalize content by trimming
    whitespace and converting to lowercase, so messages that differ only
    in whitespace or case should be detected as repetitive.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. Messages with different whitespace are detected as repetitive
    2. Content normalization (trim + lowercase) is applied
    3. The second message with different whitespace is rejected
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # First message without extra whitespace
        first_decision = strategy.evaluate_write_trigger(base_content)
        assert first_decision.should_write is True, \
            f"First message should be approved"

        # Second message with leading/trailing whitespace
        content_with_whitespace = leading_whitespace + base_content + trailing_whitespace
        
        # Only test if the whitespace version is different from the original
        if content_with_whitespace != base_content:
            decision = strategy.evaluate_write_trigger(content_with_whitespace)
            
            # Property 1: should_write should be False (detected as repetitive)
            assert decision.should_write is False, \
                f"Message with different whitespace should be detected as repetitive, got should_write={decision.should_write}"
            
            # Property 2: reason should be "repetitive"
            assert decision.reason == "repetitive", \
                f"Expected reason 'repetitive', got '{decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    content=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=3,
        max_size=100
    ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ]),
    repetition_window=st.integers(min_value=2, max_value=10),
    num_unique_messages=st.integers(min_value=1, max_value=15)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_repetition_detection_outside_window(content, repetition_window, num_unique_messages):
    """
    Property: Messages that are identical to messages outside the repetition
    window should NOT be detected as repetitive and should be approved.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. Messages outside the window are not detected as repetitive
    2. The repetition window is properly enforced
    3. Old messages can be repeated after the window expires
    """
    # Ensure we have enough messages to push the original outside the window
    assume(num_unique_messages > repetition_window)
    
    config = WriteStrategyConfig(repetition_window=repetition_window)
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # Store the first message
        first_decision = strategy.evaluate_write_trigger(content)
        assert first_decision.should_write is True, \
            f"First message should be approved"

        # Fill the window with unique messages to push the first message out
        for i in range(num_unique_messages):
            unique_content = f"{content}_filler_{i}"
            decision = strategy.evaluate_write_trigger(unique_content)
            assert decision.should_write is True, \
                f"Filler message {i} should be approved"

        # Now repeat the original message - it should be approved since it's outside the window
        repeat_decision = strategy.evaluate_write_trigger(content)
        
        # Property 1: should_write should be True (not detected as repetitive)
        assert repeat_decision.should_write is True, \
            f"Message outside repetition window should be approved, got should_write={repeat_decision.should_write}, reason={repeat_decision.reason}"
        
        # Property 2: reason should be "approved"
        assert repeat_decision.reason == "approved", \
            f"Expected reason 'approved', got '{repeat_decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    base_content=st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=3,
        max_size=50
    ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ])
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_repetition_detection_case_insensitive(base_content):
    """
    Property: Repetition detection should be case-insensitive, so messages
    that differ only in case should be detected as repetitive.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. Messages with different case are detected as repetitive
    2. Case normalization (lowercase) is applied
    3. The second message with different case is rejected
    """
    # Skip if content is already all lowercase or has no letters
    assume(base_content.casefold() != base_content)
    assume(any(c.isalpha() for c in base_content))
    
    # Skip if swapcase doesn't produce a different normalized result
    # (handles Unicode edge cases like German ß)
    different_case_content = base_content.swapcase()
    assume(base_content.strip().casefold() == different_case_content.strip().casefold())
    
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        # First message in original case
        first_decision = strategy.evaluate_write_trigger(base_content)
        assert first_decision.should_write is True, \
            f"First message should be approved"

        # Second message in different case
        decision = strategy.evaluate_write_trigger(different_case_content)
        
        # Property 1: should_write should be False (detected as repetitive)
        assert decision.should_write is False, \
            f"Message with different case should be detected as repetitive, got should_write={decision.should_write}"
        
        # Property 2: reason should be "repetitive"
        assert decision.reason == "repetitive", \
            f"Expected reason 'repetitive', got '{decision.reason}'"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 3: Repetition detection
@given(
    messages=st.lists(
        st.text(
            alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
            min_size=3,
            max_size=50
        ).filter(lambda x: x.strip() and len(x.strip()) >= 3 and x.strip().casefold() not in [
            "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
        ]),
        min_size=2,
        max_size=10,
        unique=True
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_3_no_false_positives_for_unique_messages(messages):
    """
    Property: Unique messages (not identical to any recent message) should
    never be rejected as repetitive.

    **Validates: Requirements 1.4**

    This test verifies that:
    1. Unique messages are always approved
    2. No false positives in repetition detection
    3. Each unique message is added to recent_messages
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)

    try:
        for i, message in enumerate(messages):
            decision = strategy.evaluate_write_trigger(message)
            
            # Property 1: should_write should be True for unique messages
            assert decision.should_write is True, \
                f"Unique message {i} should be approved, got should_write={decision.should_write}, reason={decision.reason}"
            
            # Property 2: reason should be "approved"
            assert decision.reason == "approved", \
                f"Expected reason 'approved' for message {i}, got '{decision.reason}'"
            
            # Property 3: message should be in recent_messages
            normalized = message.strip().casefold()
            assert normalized in strategy.recent_messages, \
                f"Approved message {i} should be in recent_messages"

    finally:
        session_manager.shutdown()
