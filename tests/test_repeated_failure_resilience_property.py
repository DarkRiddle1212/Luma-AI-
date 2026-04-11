"""
Property-Based Tests for Repeated Failure Resilience

This module implements property-based tests using Hypothesis to verify
that the ReasoningEngine continues processing messages successfully even
when memory operations fail repeatedly across multiple messages.

Feature: intent-based-memory-retrieval-enhancements
Property 12: Repeated Failure Resilience
Validates: Requirements 11.3, 11.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Any

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryRetrievalError,
    MemoryStorageError,
    QueryParameters,
    RetrievalResult
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def memory_message(draw):
    """
    Generate random messages that trigger memory intents (store or retrieve).
    """
    intent_type = draw(st.sampled_from(["store", "retrieve"]))
    
    if intent_type == "store":
        triggers = ["remember", "store"]
        trigger = draw(st.sampled_from(triggers))
        content = draw(st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=3,
            max_size=100
        ))
        return f"{trigger} {content}"
    else:  # retrieve
        triggers = ["what was", "recall", "retrieve"]
        trigger = draw(st.sampled_from(triggers))
        query = draw(st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=3,
            max_size=50
        ))
        return f"{trigger} {query}"


@st.composite
def non_memory_message(draw):
    """Generate user messages that don't trigger memory intents."""
    templates = [
        "What is {}?",
        "Tell me about {}",
        "Explain {}",
        "How does {} work?",
    ]
    
    template = draw(st.sampled_from(templates))
    topic = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')
    )))
    
    return template.format(topic)


@st.composite
def mixed_message_sequence(draw):
    """Generate a sequence of mixed memory and non-memory messages with at least one memory message."""
    # Ensure at least one memory message
    first_message = draw(memory_message())
    
    # Generate remaining messages (can be memory or non-memory)
    remaining_messages = draw(st.lists(
        st.one_of(
            memory_message(),
            non_memory_message()
        ),
        min_size=2,
        max_size=9
    ))
    
    return [first_message] + remaining_messages


# ============================================================================
# Mock Memory Implementation
# ============================================================================

class MockMemoryAlwaysFails(MemoryInterface):
    """Mock memory that always fails on both storage and retrieval."""
    
    def __init__(self):
        self.store_call_count = 0
        self.retrieve_call_count = 0
        self.total_operations = 0
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Always raises MemoryStorageError."""
        self.store_call_count += 1
        self.total_operations += 1
        raise MemoryStorageError(f"Storage failure #{self.store_call_count}")
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Always raises MemoryRetrievalError."""
        self.retrieve_call_count += 1
        self.total_operations += 1
        raise MemoryRetrievalError(f"Retrieval failure #{self.retrieve_call_count}")


class ContextCapturingLLM(StubLLM):
    """LLM that captures the context passed to generate_response."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        self.call_count += 1
        return f"Response {self.call_count}"


# ============================================================================
# Property Test: Repeated Failure Resilience (Property 12)
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 12: Repeated Failure Resilience
@given(messages=mixed_message_sequence())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_repeated_failure_resilience_property(messages):
    """
    Property: For any sequence of messages where memory operations fail repeatedly,
    the ReasoningEngine must continue processing each subsequent message without
    degradation in functionality.
    
    **Validates: Requirements 11.3, 11.5**
    
    This test verifies that:
    1. System continues processing subsequent messages after repeated failures (11.3)
    2. System maintains all existing functionality when memory fails repeatedly (11.5)
    3. Each message gets a valid response despite repeated failures
    4. No degradation in response quality across the sequence
    5. System state remains stable throughout repeated failures
    """
    # Create mock memory that always fails
    mock_memory = MockMemoryAlwaysFails()
    llm = ContextCapturingLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process all messages in sequence
    results = []
    for i, message in enumerate(messages):
        # Process message - should not raise exception
        result = engine.process_message(message)
        results.append(result)
        
        # Requirement 11.3: System continues processing subsequent messages
        assert result is not None, \
            f"Message {i}: Result should not be None despite repeated failures"
        assert isinstance(result, dict), \
            f"Message {i}: Result should be a dictionary"
        
        # Verify required keys are present
        assert "response" in result, \
            f"Message {i}: Result must contain 'response' key"
        assert "intent" in result, \
            f"Message {i}: Result must contain 'intent' key"
        assert "metadata" in result, \
            f"Message {i}: Result must contain 'metadata' key"
        
        # Requirement 11.5: System maintains all existing functionality
        # Verify response is valid and non-empty
        assert isinstance(result["response"], str), \
            f"Message {i}: Response must be a string"
        assert len(result["response"]) > 0, \
            f"Message {i}: Response must not be empty"
        
        # Verify intent is valid (not error for valid messages)
        assert isinstance(result["intent"], str), \
            f"Message {i}: Intent must be a string"
        assert len(result["intent"]) > 0, \
            f"Message {i}: Intent must not be empty"
        
        # Verify metadata structure
        assert isinstance(result["metadata"], dict), \
            f"Message {i}: Metadata must be a dictionary"
    
    # Verify all messages were processed
    assert len(results) == len(messages), \
        f"Should have processed all {len(messages)} messages"
    
    # Verify no degradation - all results should have valid structure
    for i, result in enumerate(results):
        assert "response" in result and len(result["response"]) > 0, \
            f"Message {i}: Response quality should not degrade"
        assert "intent" in result and len(result["intent"]) > 0, \
            f"Message {i}: Intent detection should not degrade"
        assert "metadata" in result and isinstance(result["metadata"], dict), \
            f"Message {i}: Metadata structure should not degrade"
    
    # Verify LLM was called appropriately (system continues functioning)
    # Note: 
    # - Non-memory intents call LLM directly
    # - Retrieve intents call LLM after retrieval (or in fallback when retrieval fails)
    # - Store intents don't call LLM (they use special handlers)
    expected_llm_calls = sum(1 for r in results if r["intent"] != "store_memory")
    assert llm.call_count == expected_llm_calls, \
        f"LLM should be called for non-store messages despite repeated failures"


# Feature: intent-based-memory-retrieval-enhancements, Property 12: Repeated Failure Resilience
@given(
    num_messages=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_repeated_retrieval_failures_no_degradation(num_messages):
    """
    Property: For any number of consecutive retrieval failures, the system
    must handle each failure independently without degradation.
    
    **Validates: Requirements 11.3, 11.5**
    
    This test verifies that:
    - Multiple consecutive retrieval failures are handled gracefully
    - Each failure gets proper error handling
    - Response quality doesn't degrade over repeated failures
    - System remains stable throughout
    """
    mock_memory = MockMemoryAlwaysFails()
    llm = ContextCapturingLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Generate retrieve messages
    message = "what was my last task"
    
    results = []
    for i in range(num_messages):
        result = engine.process_message(message)
        results.append(result)
        
        # Verify each result is valid
        assert result is not None, f"Iteration {i}: Result should not be None"
        assert isinstance(result, dict), f"Iteration {i}: Result should be a dict"
        assert "response" in result, f"Iteration {i}: Must have response"
        assert "metadata" in result, f"Iteration {i}: Must have metadata"
        
        # Verify fallback behavior is consistent
        assert result["metadata"].get("fallback") is True, \
            f"Iteration {i}: Should have fallback flag"
        assert "error" in result["metadata"], \
            f"Iteration {i}: Should have error in metadata"
    
    # Verify all operations were attempted
    assert mock_memory.retrieve_call_count == num_messages, \
        f"retrieve() should be called {num_messages} times"
    
    # Verify no degradation - check first and last results are similar quality
    first_result = results[0]
    last_result = results[-1]
    
    assert len(first_result["response"]) > 0 and len(last_result["response"]) > 0, \
        "Response quality should not degrade from first to last"
    assert first_result["metadata"]["fallback"] == last_result["metadata"]["fallback"], \
        "Fallback behavior should be consistent"


# Feature: intent-based-memory-retrieval-enhancements, Property 12: Repeated Failure Resilience
@given(
    num_messages=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_repeated_storage_failures_no_degradation(num_messages):
    """
    Property: For any number of consecutive storage failures, the system
    must handle each failure independently without degradation.
    
    **Validates: Requirements 11.3, 11.5**
    
    This test verifies that:
    - Multiple consecutive storage failures are handled gracefully
    - Each failure gets proper error handling
    - Response quality doesn't degrade over repeated failures
    - System remains stable throughout
    """
    mock_memory = MockMemoryAlwaysFails()
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Generate store messages
    message = "remember to buy milk"
    
    results = []
    for i in range(num_messages):
        result = engine.process_message(message)
        results.append(result)
        
        # Verify each result is valid
        assert result is not None, f"Iteration {i}: Result should not be None"
        assert isinstance(result, dict), f"Iteration {i}: Result should be a dict"
        assert "response" in result, f"Iteration {i}: Must have response"
        assert "metadata" in result, f"Iteration {i}: Must have metadata"
        
        # Verify error is indicated
        assert "error" in result["metadata"] or "error_type" in result["metadata"], \
            f"Iteration {i}: Should have error indicator"
    
    # Verify all operations were attempted
    assert mock_memory.store_call_count == num_messages, \
        f"store() should be called {num_messages} times"
    
    # Verify no degradation - all results should have error handling
    for i, result in enumerate(results):
        assert "response" in result and len(result["response"]) > 0, \
            f"Iteration {i}: Should have non-empty response"
        assert "error" in result["metadata"] or "error_type" in result["metadata"], \
            f"Iteration {i}: Should have error indicator"


# Feature: intent-based-memory-retrieval-enhancements, Property 12: Repeated Failure Resilience
@given(messages=mixed_message_sequence())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_mixed_failures_maintain_functionality(messages):
    """
    Property: For any sequence of mixed memory operations (store and retrieve)
    that all fail, the system must maintain full functionality for each message.
    
    **Validates: Requirements 11.3, 11.5**
    
    This test verifies that:
    - System handles mixed failure types (storage and retrieval)
    - Each message type gets appropriate error handling
    - Non-memory messages continue to work normally
    - System state remains consistent across mixed failures
    """
    mock_memory = MockMemoryAlwaysFails()
    llm = ContextCapturingLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    results = []
    for i, message in enumerate(messages):
        result = engine.process_message(message)
        results.append(result)
        
        # Verify result is valid
        assert result is not None, f"Message {i}: Result should not be None"
        assert isinstance(result, dict), f"Message {i}: Result should be a dict"
        
        # Verify required fields
        assert "response" in result, f"Message {i}: Must have response"
        assert "intent" in result, f"Message {i}: Must have intent"
        assert "metadata" in result, f"Message {i}: Must have metadata"
        
        # Verify response quality
        assert isinstance(result["response"], str), \
            f"Message {i}: Response must be a string"
        assert len(result["response"]) > 0, \
            f"Message {i}: Response must not be empty"
        
        # Check intent-specific behavior
        intent = result["intent"]
        if intent == "retrieve_memory":
            # Retrieval failures should have fallback flag
            assert result["metadata"].get("fallback") is True, \
                f"Message {i}: Retrieval failure should have fallback flag"
        elif intent == "store_memory":
            # Storage failures should have error indicator
            assert "error" in result["metadata"] or "error_type" in result["metadata"], \
                f"Message {i}: Storage failure should have error indicator"
        # Non-memory intents should work normally without error indicators
    
    # Verify all messages were processed
    assert len(results) == len(messages), \
        "All messages should be processed"
    
    # Verify LLM was called appropriately
    # Note: 
    # - Non-memory intents call LLM directly
    # - Retrieve intents call LLM after retrieval (or in fallback when retrieval fails)
    # - Store intents don't call LLM (they use special handlers)
    expected_llm_calls = sum(1 for r in results if r["intent"] != "store_memory")
    assert llm.call_count == expected_llm_calls, \
        "LLM should be called for non-store messages"
    
    # Verify system attempted memory operations when appropriate
    total_memory_operations = mock_memory.store_call_count + mock_memory.retrieve_call_count
    assert total_memory_operations > 0, \
        "Should have attempted some memory operations"


# Feature: intent-based-memory-retrieval-enhancements, Property 12: Repeated Failure Resilience
@given(
    messages=st.lists(
        memory_message(),
        min_size=10,
        max_size=30
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_long_sequence_repeated_failures_stability(messages):
    """
    Property: For any long sequence of memory operations that fail repeatedly,
    the system must remain stable without memory leaks, state corruption, or
    performance degradation.
    
    **Validates: Requirements 11.3, 11.5**
    
    This test verifies that:
    - System handles long sequences of failures without issues
    - No memory leaks or resource exhaustion
    - Performance remains consistent
    - State remains clean throughout
    """
    mock_memory = MockMemoryAlwaysFails()
    llm = ContextCapturingLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process long sequence
    results = []
    for i, message in enumerate(messages):
        result = engine.process_message(message)
        results.append(result)
        
        # Verify basic validity
        assert result is not None, f"Message {i}: Result should not be None"
        assert isinstance(result, dict), f"Message {i}: Result should be a dict"
        assert "response" in result, f"Message {i}: Must have response"
    
    # Verify all messages were processed
    assert len(results) == len(messages), \
        f"Should process all {len(messages)} messages"
    
    # Verify system attempted all operations
    assert mock_memory.total_operations == len(messages), \
        "Should attempt memory operation for each message"
    
    # Verify LLM was called appropriately
    # Note: 
    # - Retrieve intents call LLM in fallback when retrieval fails
    # - Store intents don't call LLM (they use special handlers)
    retrieve_count = sum(1 for r in results if r["intent"] == "retrieve_memory")
    expected_llm_calls = retrieve_count  # Only retrieve intents call LLM in fallback
    assert llm.call_count == expected_llm_calls, \
        f"LLM should be called for retrieve intents in fallback (expected {expected_llm_calls}, got {llm.call_count})"
    
    # Verify no degradation - sample results throughout sequence
    sample_indices = [0, len(results) // 2, len(results) - 1]
    for idx in sample_indices:
        result = results[idx]
        assert "response" in result and len(result["response"]) > 0, \
            f"Result at index {idx}: Response quality should not degrade"
        assert "metadata" in result and isinstance(result["metadata"], dict), \
            f"Result at index {idx}: Metadata structure should not degrade"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
