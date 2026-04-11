"""
Property-Based Test for Non-Trivial Message Evaluation

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly evaluates non-trivial messages
for storage.

Feature: memory-write-strategy-session-management
Property 1: Non-trivial message evaluation
Validates: Requirements 1.1, 1.5
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
# Property 1: Non-Trivial Message Evaluation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 1: Non-trivial message evaluation
@given(
    content=st.text(
        min_size=10,
        max_size=200,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd', 'Pc', 'Pd', 'Zs'],
            whitelist_characters=[' ', '.', ',', '!', '?']
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_1_non_trivial_message_evaluation(content):
    """
    Property: For any non-empty, non-whitespace message that is not in the
    trivial patterns list and is not repetitive, the Memory_Write_Strategy
    should evaluate it and approve it for storage.
    
    **Validates: Requirements 1.1, 1.5**
    
    This test verifies that:
    1. Non-trivial messages are approved for storage
    2. The decision includes should_write=True
    3. The reason is "approved"
    4. The message is added to recent_messages for repetition detection
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
        # Ensure content is not a trivial pattern
        normalized_content = content.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Evaluate the message
            decision = strategy.evaluate_write_trigger(content)
            
            # Property 1: should_write should be True for non-trivial content
            assert decision.should_write is True, \
                f"Non-trivial message should be approved for storage, got should_write={decision.should_write}, reason={decision.reason}"
            
            # Property 2: reason should be "approved"
            assert decision.reason == "approved", \
                f"Expected reason 'approved', got '{decision.reason}'"
            
            # Property 3: metadata should be empty for approved messages
            assert decision.metadata == {}, \
                f"Approved message should have empty metadata, got {decision.metadata}"
            
            # Property 4: message should be added to recent_messages
            normalized_for_comparison = content.strip().casefold()
            assert normalized_for_comparison in strategy.recent_messages, \
                f"Approved message should be added to recent_messages"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 1: Non-trivial message evaluation
@given(
    content=st.text(
        min_size=5,
        max_size=100,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '_']
        )
    ),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_1_non_trivial_messages_with_categories(content, category):
    """
    Property: For any non-trivial message with various content types and
    categories, the Memory_Write_Strategy should correctly evaluate and
    approve it for storage.
    
    **Validates: Requirements 1.1, 1.5**
    
    This test verifies that:
    1. Messages with different content types are approved
    2. Category information doesn't affect the write trigger evaluation
    3. The evaluation is based solely on content characteristics
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().lower()
        is_trivial = any(normalized_content == pattern.lower() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Evaluate with metadata containing category
            metadata = {"category": category}
            decision = strategy.evaluate_write_trigger(content, metadata)
            
            # Property 1: should_write should be True
            assert decision.should_write is True, \
                f"Non-trivial message with category should be approved, got should_write={decision.should_write}"
            
            # Property 2: reason should be "approved"
            assert decision.reason == "approved", \
                f"Expected reason 'approved', got '{decision.reason}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 1: Non-trivial message evaluation
@given(
    content=st.text(
        min_size=5,
        max_size=50,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd', 'Pc', 'Po', 'Zs'],
            whitelist_characters=[' ', '.', ',', '!', '?', ':', ';', '-', '(', ')']
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_1_complex_non_trivial_messages(content):
    """
    Property: For complex non-trivial messages with various punctuation,
    sentence structures, and content types, the Memory_Write_Strategy
    should correctly evaluate and approve them for storage.
    
    **Validates: Requirements 1.1, 1.5**
    
    This test verifies that:
    1. Complex messages with punctuation are approved
    2. Messages with multiple sentences are approved
    3. Messages with special characters are approved
    4. The evaluation handles various text formats correctly
    """
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure content is not trivial
        normalized_content = content.strip().casefold()
        is_trivial = any(normalized_content == pattern.casefold() for pattern in config.trivial_patterns)
        
        if not is_trivial and len(content.strip()) >= config.min_content_length:
            # Evaluate the complex message
            decision = strategy.evaluate_write_trigger(content)
            
            # Property 1: should_write should be True for complex non-trivial content
            assert decision.should_write is True, \
                f"Complex non-trivial message should be approved, got should_write={decision.should_write}, reason={decision.reason}"
            
            # Property 2: reason should be "approved"
            assert decision.reason == "approved", \
                f"Expected reason 'approved', got '{decision.reason}'"
            
            # Property 3: message should be added to recent_messages
            normalized_for_comparison = content.strip().casefold()
            assert normalized_for_comparison in strategy.recent_messages, \
                f"Approved complex message should be added to recent_messages"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 1: Non-trivial message evaluation
@given(
    content=st.text(
        min_size=5,
        max_size=100,
        alphabet=st.characters(
            whitelist_categories=['Ll', 'Lu', 'Nd'],
            whitelist_characters=[' ', '.', ',', '!', '?']
        )
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_1_multiple_non_trivial_messages_in_sequence(content):
    """
    Property: For a sequence of non-trivial messages, each should be
    evaluated independently and approved if it meets the criteria.
    
    **Validates: Requirements 1.1, 1.5**
    
    This test verifies that:
    1. Multiple non-trivial messages are all approved
    2. Each message is evaluated independently
    3. The repetition detection works correctly across multiple messages
    """
    config = WriteStrategyConfig(repetition_window=5)
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Generate multiple variations of the content
        messages = [f"{content} variation {i}" for i in range(5)]
        
        for i, message in enumerate(messages):
            # Ensure message is not trivial
            normalized_message = message.strip().casefold()
            is_trivial = any(normalized_message == pattern.casefold() for pattern in config.trivial_patterns)
            
            if not is_trivial and len(message.strip()) >= config.min_content_length:
                decision = strategy.evaluate_write_trigger(message)
                
                # Property 1: should_write should be True
                assert decision.should_write is True, \
                    f"Message {i+1} should be approved, got should_write={decision.should_write}, reason={decision.reason}"
                
                # Property 2: reason should be "approved"
                assert decision.reason == "approved", \
                    f"Expected reason 'approved' for message {i+1}, got '{decision.reason}'"
    
    finally:
        session_manager.shutdown()
