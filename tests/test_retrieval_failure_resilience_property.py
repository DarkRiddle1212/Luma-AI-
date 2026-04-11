"""
Property-Based Tests for Retrieval Failure Resilience

This module implements property-based tests using Hypothesis to verify
that the ReasoningEngine handles MemoryRetrievalError gracefully with
proper logging, fallback behavior, and metadata reporting.

Feature: intent-based-memory-retrieval-enhancements
Property 4: Retrieval Failure Resilience
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
    QueryParameters,
    RetrievalResult
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def retrieve_memory_message(draw):
    """
    Generate random messages with retrieve_memory intent.
    
    These messages should trigger the retrieve_memory intent in ReasoningEngine.
    Uses triggers that are recognized by detect_intent.
    """
    # Retrieve memory trigger words
    triggers = ["what was", "recall", "retrieve"]
    trigger = draw(st.sampled_from(triggers))
    
    # Query content
    query = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=3,
        max_size=100
    ))
    
    # Construct message with trigger + query
    message = f"{trigger} {query}"
    
    return message


@st.composite
def error_message(draw):
    """Generate random error messages for MemoryRetrievalError."""
    error_types = [
        "Database connection failed",
        "Timeout while querying memories",
        "SQLite error: disk I/O error",
        "Memory index corrupted",
        "Network error during retrieval"
    ]
    
    return draw(st.sampled_from(error_types))


# ============================================================================
# Mock Memory Implementation
# ============================================================================

class MockMemoryRetrievalFailure(MemoryInterface):
    """Mock memory that always fails on retrieval with MemoryRetrievalError."""
    
    def __init__(self, error_msg: str = "Database connection failed"):
        self.error_msg = error_msg
        self.retrieve_called = False
        self.retrieve_call_count = 0
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store succeeds (not tested here)."""
        return "mem_1"
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Always raises MemoryRetrievalError."""
        self.retrieve_called = True
        self.retrieve_call_count += 1
        raise MemoryRetrievalError(self.error_msg)


class ContextCapturingLLM(StubLLM):
    """LLM that captures the context passed to generate_response."""
    
    def __init__(self):
        super().__init__()
        self.captured_contexts = []
        self.call_count = 0
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        self.captured_contexts.append(context)
        self.call_count += 1
        return f"Fallback response {self.call_count}"


# ============================================================================
# Property Test: Retrieval Failure Resilience (Property 4)
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 4: Retrieval Failure Resilience
@given(
    message=retrieve_memory_message(),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_failure_resilience_property(message, error_msg):
    """
    Property: For any MemoryRetrievalError during retrieval, the ReasoningEngine
    must catch the exception, log the error details, fall back to LLM-only processing,
    and return a valid response with metadata containing fallback=true and the error message.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
    
    This test verifies that:
    1. MemoryRetrievalError is caught and doesn't crash the system (4.1, 4.6)
    2. Error is logged with full details (4.2)
    3. System falls back to LLM-only processing without memories (4.3)
    4. Response metadata includes fallback=true flag (4.4)
    5. Response metadata includes the error message (4.5)
    6. A valid response is returned to the user (4.6)
    """
    # Create mock memory that fails with the given error message
    mock_memory = MockMemoryRetrievalFailure(error_msg)
    llm = ContextCapturingLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Patch logger to capture log calls
    with patch('luma.core.reasoning.logger') as mock_logger:
        # Process the message - should not raise exception
        result = engine.process_message(message)
        
        # Requirement 4.1: MemoryRetrievalError is caught and handled
        assert mock_memory.retrieve_called, \
            "memory.retrieve() should be called"
        
        # Requirement 4.6: System doesn't crash - valid response returned
        assert isinstance(result, dict), \
            f"Result must be a dict, got {type(result)}"
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "intent" in result, \
            "Result must contain 'intent' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # Verify intent is correct
        assert result["intent"] == "retrieve_memory", \
            f"Intent should be 'retrieve_memory', got '{result['intent']}'"
        
        # Verify response is a valid string
        assert isinstance(result["response"], str), \
            f"Response must be a string, got {type(result['response'])}"
        assert len(result["response"]) > 0, \
            "Response should not be empty"
        
        # Requirement 4.2: Error is logged with full details
        # Check that logger.error was called
        assert mock_logger.error.called, \
            "logger.error should be called when retrieval fails"
        
        # Verify error logging includes the error message
        error_call_args = mock_logger.error.call_args
        assert error_call_args is not None, \
            "logger.error should have been called with arguments"
        
        # Check that error message is in the log
        log_message = str(error_call_args[0][0])
        assert "retrieval failed" in log_message.lower() or "error" in log_message.lower(), \
            f"Error log should mention retrieval failure, got: {log_message}"
        
        # Verify exc_info=True was passed for stack trace
        if 'exc_info' in error_call_args[1]:
            assert error_call_args[1]['exc_info'] is True, \
                "logger.error should be called with exc_info=True for stack trace"
        
        # Requirement 4.3: System falls back to LLM-only processing
        assert llm.call_count > 0, \
            "LLM should be called for fallback processing"
        
        # Verify LLM was called with context
        assert len(llm.captured_contexts) > 0, \
            "LLM should receive context during fallback"
        
        # Verify context has empty memories (fallback behavior)
        context = llm.captured_contexts[0]
        assert "memories" in context, \
            "Context should contain 'memories' key even in fallback"
        assert context["memories"] == [], \
            "Memories should be empty list in fallback mode"
        
        # Requirement 4.4: Response metadata includes fallback=true flag
        assert "fallback" in result["metadata"], \
            "Metadata must contain 'fallback' key"
        assert result["metadata"]["fallback"] is True, \
            f"Fallback flag should be True, got {result['metadata']['fallback']}"
        
        # Requirement 4.5: Response metadata includes error message
        assert "error" in result["metadata"], \
            "Metadata must contain 'error' key"
        assert isinstance(result["metadata"]["error"], str), \
            f"Error must be a string, got {type(result['metadata']['error'])}"
        assert len(result["metadata"]["error"]) > 0, \
            "Error message should not be empty"
        
        # Verify the error message matches what was raised
        assert error_msg in result["metadata"]["error"], \
            f"Error message should contain '{error_msg}', got '{result['metadata']['error']}'"


# Feature: intent-based-memory-retrieval-enhancements, Property 4: Retrieval Failure Resilience
@given(message=retrieve_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_failure_does_not_crash_system(message):
    """
    Property: For any retrieve_memory message, when retrieval fails,
    the system must not crash and must continue processing subsequent messages.
    
    **Validates: Requirements 4.6**
    
    This test verifies that:
    1. System doesn't crash on retrieval error
    2. System can process multiple messages even when retrieval fails repeatedly
    3. Each message gets a valid response despite failures
    """
    # Create mock memory that always fails
    mock_memory = MockMemoryRetrievalFailure("Persistent database error")
    llm = StubLLM()
    
    # Create ReasoningEngine
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the same message multiple times - should all succeed
    for i in range(3):
        result = engine.process_message(message)
        
        # Verify valid response each time
        assert isinstance(result, dict), \
            f"Iteration {i}: Result must be a dict"
        assert "response" in result, \
            f"Iteration {i}: Result must contain 'response'"
        assert "metadata" in result, \
            f"Iteration {i}: Result must contain 'metadata'"
        assert result["metadata"]["fallback"] is True, \
            f"Iteration {i}: Fallback flag should be True"
        
        # Verify retrieve was called each time
        assert mock_memory.retrieve_call_count == i + 1, \
            f"Iteration {i}: retrieve should be called {i + 1} times"


# Feature: intent-based-memory-retrieval-enhancements, Property 4: Retrieval Failure Resilience
@given(
    message=retrieve_memory_message(),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_failure_fallback_continues_with_llm(message, error_msg):
    """
    Property: For any MemoryRetrievalError, the system must continue
    processing with LLM and inject empty memories into context.
    
    **Validates: Requirements 4.3**
    
    This test verifies that:
    1. LLM is called even when retrieval fails
    2. Context is built without memories (empty list)
    3. LLM receives the user message for processing
    """
    # Create mock memory that fails
    mock_memory = MockMemoryRetrievalFailure(error_msg)
    llm = ContextCapturingLLM()
    
    # Create ReasoningEngine
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process message
    result = engine.process_message(message)
    
    # Verify LLM was called
    assert llm.call_count == 1, \
        "LLM should be called exactly once for fallback"
    
    # Verify context was passed to LLM
    assert len(llm.captured_contexts) == 1, \
        "LLM should receive context"
    
    context = llm.captured_contexts[0]
    
    # Verify context structure
    assert isinstance(context, dict), \
        f"Context must be a dict, got {type(context)}"
    
    # Verify memories key exists and is empty
    assert "memories" in context, \
        "Context must contain 'memories' key"
    assert context["memories"] == [], \
        "Memories should be empty list in fallback mode"
    
    # Verify response contains LLM output
    assert "Fallback response" in result["response"], \
        "Response should contain LLM-generated fallback response"


# Feature: intent-based-memory-retrieval-enhancements, Property 4: Retrieval Failure Resilience
@given(
    messages=st.lists(
        retrieve_memory_message(),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_failure_resilience_across_multiple_messages(messages):
    """
    Property: For any sequence of retrieve_memory messages, when retrieval
    fails for all of them, the system must handle each failure independently
    and return valid responses for all messages.
    
    **Validates: Requirements 4.1, 4.6**
    
    This test verifies that:
    1. System handles multiple consecutive failures
    2. Each message gets independent error handling
    3. System remains stable across repeated failures
    """
    # Create mock memory that always fails
    mock_memory = MockMemoryRetrievalFailure("Database unavailable")
    llm = StubLLM()
    
    # Create ReasoningEngine
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process all messages
    results = []
    for i, message in enumerate(messages):
        result = engine.process_message(message)
        results.append(result)
        
        # Verify each result is valid
        assert isinstance(result, dict), \
            f"Message {i}: Result must be a dict"
        assert "response" in result, \
            f"Message {i}: Result must contain 'response'"
        assert "metadata" in result, \
            f"Message {i}: Result must contain 'metadata'"
        assert result["metadata"]["fallback"] is True, \
            f"Message {i}: Fallback flag should be True"
        assert "error" in result["metadata"], \
            f"Message {i}: Error should be in metadata"
    
    # Verify all messages were processed
    assert len(results) == len(messages), \
        f"Should process all {len(messages)} messages"
    
    # Verify retrieve was called for each message
    assert mock_memory.retrieve_call_count == len(messages), \
        f"retrieve should be called {len(messages)} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
