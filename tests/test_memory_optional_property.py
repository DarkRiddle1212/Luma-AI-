"""
Property-Based Tests for Memory-Optional Operation

This module implements property-based tests using Hypothesis to verify
that the ReasoningEngine operates correctly when memory is not configured,
processing messages successfully and returning informative messages for
memory-related intents.

Feature: intent-based-memory-retrieval-enhancements
Property 11: Memory-Optional Operation
Validates: Requirements 11.1, 11.2, 11.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict
from datetime import datetime

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def non_memory_message_strategy(draw):
    """Generate user messages that don't trigger memory intents."""
    templates = [
        "What is {}?",
        "Tell me about {}",
        "Explain {}",
        "How does {} work?",
        "Can you help me with {}?",
        "I need information about {}",
        "Teach me {}",
        "Show me how to {}",
        "What are the benefits of {}?",
        "Why is {} important?"
    ]
    
    template = draw(st.sampled_from(templates))
    topic = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')
    )))
    
    return template.format(topic)


@st.composite
def memory_store_message_strategy(draw):
    """Generate user messages that trigger store_memory intent."""
    # Use keywords that match detect_intent: "remember" or "store"
    # Avoid "remember" alone as it can be ambiguous with retrieve
    triggers = ["remember that", "store this information", "please store"]
    trigger = draw(st.sampled_from(triggers))
    
    content = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')
    )))
    
    return f"{trigger} {content}"


@st.composite
def memory_retrieve_message_strategy(draw):
    """Generate user messages that trigger retrieve_memory intent."""
    # Use only retrieve-specific triggers to avoid ambiguity with store
    triggers = ["what was", "recall", "retrieve"]
    trigger = draw(st.sampled_from(triggers))
    
    query = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')
    )))
    
    return f"{trigger} {query}"


# ============================================================================
# Property 11: Memory-Optional Operation
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 11: Memory-Optional Operation
@given(message=non_memory_message_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_non_memory_messages_processed_without_memory(message):
    """
    Property: For any non-memory message processed by ReasoningEngine when memory
    is None, the system must process the message successfully using LLM-only mode.
    
    **Validates: Requirements 11.1, 11.4**
    
    This test verifies that:
    1. ReasoningEngine processes all messages using LLM-only mode when memory=None (Requirement 11.1)
    2. System maintains all existing functionality when memory is disabled (Requirement 11.4)
    3. No errors occur when memory is not configured
    4. Response structure is valid and complete
    """
    # Create ReasoningEngine without memory (memory=None)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process non-memory message
    result = engine.process_message(message)
    
    # Verify response is valid and complete
    assert result is not None, "Result must not be None"
    assert isinstance(result, dict), "Result must be a dictionary"
    
    # Verify required keys are present
    assert "response" in result, "Result must contain 'response' key"
    assert "intent" in result, "Result must contain 'intent' key"
    assert "metadata" in result, "Result must contain 'metadata' key"
    
    # Verify response is a non-empty string
    assert isinstance(result["response"], str), "Response must be a string"
    assert len(result["response"]) > 0, "Response must not be empty"
    
    # Verify intent is valid (not error or invalid)
    assert result["intent"] not in ["error", "invalid"], \
        f"Intent should not be error or invalid for valid message, got: {result['intent']}"
    
    # Verify metadata structure
    assert isinstance(result["metadata"], dict), "Metadata must be a dictionary"
    assert "timestamp" in result["metadata"], "Metadata must contain timestamp"
    
    # Verify no error in metadata (successful processing)
    assert "error" not in result["metadata"] or result["metadata"].get("error") is None, \
        f"Should not have error in metadata for successful processing, got: {result['metadata'].get('error')}"
    
    # Verify system processed message successfully without memory
    # The response should be from LLM, not an error message about missing memory
    assert "not available" not in result["response"].lower(), \
        "Response should not indicate unavailability for non-memory intents"


# Feature: intent-based-memory-retrieval-enhancements, Property 11: Memory-Optional Operation
@given(message=memory_store_message_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_store_memory_intent_returns_informative_message_without_memory(message):
    """
    Property: For any store_memory message processed by ReasoningEngine when memory
    is None, the system must return an informative message indicating memory is not
    available.
    
    **Validates: Requirement 11.2**
    
    This test verifies that:
    1. ReasoningEngine returns informative messages for memory-related intents when memory=None (Requirement 11.2)
    2. System doesn't crash or throw errors
    3. Response clearly indicates memory is not configured
    4. Metadata includes appropriate error information
    """
    # Create ReasoningEngine without memory (memory=None)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process store_memory message
    result = engine.process_message(message)
    
    # Verify response is valid
    assert result is not None, "Result must not be None"
    assert isinstance(result, dict), "Result must be a dictionary"
    
    # Verify required keys are present
    assert "response" in result, "Result must contain 'response' key"
    assert "intent" in result, "Result must contain 'intent' key"
    assert "metadata" in result, "Result must contain 'metadata' key"
    
    # Verify intent is store_memory
    assert result["intent"] == "store_memory", \
        f"Intent should be 'store_memory', got: {result['intent']}"
    
    # Verify response is informative about memory not being available
    response_lower = result["response"].lower()
    assert "not available" in response_lower or "not configured" in response_lower, \
        f"Response should indicate memory is not available, got: {result['response']}"
    
    # Verify metadata contains error information
    assert isinstance(result["metadata"], dict), "Metadata must be a dictionary"
    assert "error" in result["metadata"], \
        "Metadata should contain 'error' key indicating no memory configured"
    assert result["metadata"]["error"] == "no_memory_configured", \
        f"Error should be 'no_memory_configured', got: {result['metadata']['error']}"
    
    # Verify system didn't crash (we got a valid response)
    assert isinstance(result["response"], str), "Response must be a string"
    assert len(result["response"]) > 0, "Response must not be empty"


# Feature: intent-based-memory-retrieval-enhancements, Property 11: Memory-Optional Operation
@given(message=memory_retrieve_message_strategy())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_retrieve_memory_intent_returns_informative_message_without_memory(message):
    """
    Property: For any retrieve_memory message processed by ReasoningEngine when memory
    is None, the system must return an informative message indicating memory is not
    available.
    
    **Validates: Requirement 11.2**
    
    This test verifies that:
    1. ReasoningEngine returns informative messages for memory-related intents when memory=None (Requirement 11.2)
    2. System doesn't crash or throw errors
    3. Response clearly indicates memory is not configured
    4. Metadata includes appropriate error information
    """
    # Create ReasoningEngine without memory (memory=None)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process retrieve_memory message
    result = engine.process_message(message)
    
    # Verify response is valid
    assert result is not None, "Result must not be None"
    assert isinstance(result, dict), "Result must be a dictionary"
    
    # Verify required keys are present
    assert "response" in result, "Result must contain 'response' key"
    assert "intent" in result, "Result must contain 'intent' key"
    assert "metadata" in result, "Result must contain 'metadata' key"
    
    # Verify intent is retrieve_memory
    assert result["intent"] == "retrieve_memory", \
        f"Intent should be 'retrieve_memory', got: {result['intent']}"
    
    # Verify response is informative about memory not being available
    response_lower = result["response"].lower()
    assert "not available" in response_lower or "not configured" in response_lower, \
        f"Response should indicate memory is not available, got: {result['response']}"
    
    # Verify metadata contains error information
    assert isinstance(result["metadata"], dict), "Metadata must be a dictionary"
    assert "error" in result["metadata"], \
        "Metadata should contain 'error' key indicating no memory configured"
    assert result["metadata"]["error"] == "no_memory_configured", \
        f"Error should be 'no_memory_configured', got: {result['metadata']['error']}"
    
    # Verify system didn't crash (we got a valid response)
    assert isinstance(result["response"], str), "Response must be a string"
    assert len(result["response"]) > 0, "Response must not be empty"


# Feature: intent-based-memory-retrieval-enhancements, Property 11: Memory-Optional Operation
@given(
    messages=st.lists(
        st.one_of(
            non_memory_message_strategy(),
            memory_store_message_strategy(),
            memory_retrieve_message_strategy()
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_multiple_messages_processed_successfully_without_memory(messages):
    """
    Property: For any sequence of messages processed by ReasoningEngine when memory
    is None, all messages must be processed successfully without degradation.
    
    **Validates: Requirements 11.1, 11.4**
    
    This test verifies that:
    1. System can process multiple messages in sequence without memory
    2. No degradation in functionality across multiple calls
    3. Each message gets a valid response
    4. System remains stable throughout the sequence
    """
    # Create ReasoningEngine without memory (memory=None)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process all messages in sequence
    results = []
    for message in messages:
        result = engine.process_message(message)
        results.append(result)
    
    # Verify all messages were processed successfully
    assert len(results) == len(messages), \
        f"Should have {len(messages)} results, got {len(results)}"
    
    # Verify each result is valid
    for i, result in enumerate(results):
        assert result is not None, f"Result {i} must not be None"
        assert isinstance(result, dict), f"Result {i} must be a dictionary"
        
        # Verify required keys
        assert "response" in result, f"Result {i} must contain 'response' key"
        assert "intent" in result, f"Result {i} must contain 'intent' key"
        assert "metadata" in result, f"Result {i} must contain 'metadata' key"
        
        # Verify response is valid
        assert isinstance(result["response"], str), f"Result {i} response must be a string"
        assert len(result["response"]) > 0, f"Result {i} response must not be empty"
        
        # Verify no system errors (intent should not be 'error' for valid messages)
        # Note: Memory intents will have error metadata, but that's expected behavior
        if result["intent"] not in ["store_memory", "retrieve_memory"]:
            assert result["intent"] != "error", \
                f"Result {i} should not have error intent for non-memory message"


# Feature: intent-based-memory-retrieval-enhancements, Property 11: Memory-Optional Operation
@given(message=st.text(min_size=1, max_size=200))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_any_message_processed_without_crash_when_memory_none(message):
    """
    Property: For any message (including edge cases) processed by ReasoningEngine
    when memory is None, the system must not crash and must return a valid response.
    
    **Validates: Requirements 11.1, 11.4**
    
    This test verifies that:
    1. System handles all message types without crashing when memory=None
    2. Edge cases (special characters, long messages, etc.) are handled
    3. Always returns a valid response structure
    4. No unhandled exceptions occur
    """
    # Create ReasoningEngine without memory (memory=None)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process message - should not raise any exceptions
    try:
        result = engine.process_message(message)
    except Exception as e:
        pytest.fail(f"Processing message should not raise exception when memory=None, got: {e}")
    
    # Verify result is valid
    assert result is not None, "Result must not be None"
    assert isinstance(result, dict), "Result must be a dictionary"
    
    # Verify required keys are present
    assert "response" in result, "Result must contain 'response' key"
    assert "intent" in result, "Result must contain 'intent' key"
    assert "metadata" in result, "Result must contain 'metadata' key"
    
    # Verify response is a string (even if empty for invalid input)
    assert isinstance(result["response"], str), "Response must be a string"
    
    # Verify intent is a string
    assert isinstance(result["intent"], str), "Intent must be a string"
    
    # Verify metadata is a dictionary
    assert isinstance(result["metadata"], dict), "Metadata must be a dictionary"
