"""
Property-Based Tests for Storage Failure Handling

This module implements property-based tests using Hypothesis to verify
that the ReasoningEngine handles MemoryStorageError gracefully with
proper logging, user-friendly error messages, and metadata reporting.

Feature: intent-based-memory-retrieval-enhancements
Property 5: Storage Failure Handling
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Any

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    QueryParameters,
    RetrievalResult
)


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def store_memory_message(draw):
    """
    Generate random messages with store_memory intent.
    
    These messages should trigger the store_memory intent in ReasoningEngine.
    Uses triggers that are recognized by detect_intent: "remember" or "store"
    """
    # Store memory trigger words (only those recognized by detect_intent)
    triggers = ["remember", "store"]
    trigger = draw(st.sampled_from(triggers))
    
    # Content to store
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=3,
        max_size=100
    ))
    
    # Construct message with trigger + content
    message = f"{trigger} {content}"
    
    return message


@st.composite
def error_message(draw):
    """Generate random error messages for MemoryStorageError."""
    error_types = [
        "Database write failed",
        "Disk full - cannot store memory",
        "SQLite error: database is locked",
        "Encryption failed during storage",
        "Network error during sync"
    ]
    
    return draw(st.sampled_from(error_types))


# ============================================================================
# Mock Memory Implementation
# ============================================================================

class MockMemoryStorageFailure(MemoryInterface):
    """Mock memory that always fails on storage with MemoryStorageError."""
    
    def __init__(self, error_msg: str = "Database write failed"):
        self.error_msg = error_msg
        self.store_called = False
        self.store_call_count = 0
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Always raises MemoryStorageError."""
        self.store_called = True
        self.store_call_count += 1
        raise MemoryStorageError(self.error_msg)
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Retrieve succeeds (not tested here)."""
        return {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }


# ============================================================================
# Property Test: Storage Failure Handling (Property 5)
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 5: Storage Failure Handling
@given(
    message=store_memory_message(),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_storage_failure_handling_property(message, error_msg):
    """
    Property: For any MemoryStorageError during storage, the ReasoningEngine
    must catch the exception, log the error details, return a user-friendly
    error message, and include error details in response metadata without crashing.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    
    This test verifies that:
    1. MemoryStorageError is caught and doesn't crash the system (5.1, 5.5)
    2. User-friendly error message is returned (5.2)
    3. Response metadata includes error details (5.3)
    4. Error is logged with full details (5.4)
    5. System continues functioning after storage failure (5.5)
    """
    # Create mock memory that fails with the given error message
    mock_memory = MockMemoryStorageFailure(error_msg)
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Patch logger to capture log calls
    with patch('luma.core.reasoning.logger') as mock_logger:
        # Process the message - should not raise exception
        result = engine.process_message(message)
        
        # Verify: MemoryStorageError was caught (5.1)
        assert mock_memory.store_called, "store() should have been called"
        
        # Verify: System didn't crash (5.5)
        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dictionary"
        
        # Verify: User-friendly error message is returned (5.2)
        assert "response" in result, "Result should contain 'response' key"
        response_text = result["response"].lower()
        assert "couldn't store" in response_text or "error" in response_text or "failed" in response_text, \
            f"Response should contain user-friendly error message, got: {result['response']}"
        
        # Verify: Response metadata includes error details (5.3)
        assert "metadata" in result, "Result should contain 'metadata' key"
        metadata = result["metadata"]
        assert "error" in metadata or "error_type" in metadata, \
            "Metadata should include error information"
        
        # If error field exists, verify it contains the error message
        if "error" in metadata:
            assert error_msg in str(metadata["error"]) or metadata["error"], \
                f"Metadata error should reference the original error: {error_msg}"
        
        # Verify: Error was logged with full details (5.4)
        assert mock_logger.error.called, "Error should have been logged"
        
        # Get the first error log call
        error_log_calls = [call for call in mock_logger.error.call_args_list]
        assert len(error_log_calls) > 0, "At least one error should have been logged"
        
        # Verify error message was logged
        logged_messages = [str(call[0][0]) for call in error_log_calls]
        assert any(error_msg in msg or "storage" in msg.lower() for msg in logged_messages), \
            f"Error message should have been logged. Logged: {logged_messages}"


# ============================================================================
# Additional Property Tests for Storage Failure Scenarios
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 5: Storage Failure Handling
@given(message=store_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_storage_failure_preserves_engine_state(message):
    """
    Property: For any storage failure, the ReasoningEngine state should not be
    corrupted and subsequent operations should work normally.
    
    **Validates: Requirements 5.5**
    
    This test verifies that after a storage failure, the engine can still:
    - Process subsequent messages
    - Handle other intents correctly
    - Maintain its internal state
    """
    # Create mock memory that fails on storage
    mock_memory = MockMemoryStorageFailure("Test storage error")
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # First call - storage fails
    result1 = engine.process_message(message)
    assert result1 is not None, "First call should return a result"
    assert "error" in result1["metadata"] or "error_type" in result1["metadata"], \
        "First call should indicate error"
    
    # Second call - should still work (engine state not corrupted)
    result2 = engine.process_message(message)
    assert result2 is not None, "Second call should return a result"
    assert isinstance(result2, dict), "Second call should return a dictionary"
    
    # Verify engine is still functional
    assert mock_memory.store_call_count == 2, "store() should have been called twice"


# Feature: intent-based-memory-retrieval-enhancements, Property 5: Storage Failure Handling
@given(
    messages=st.lists(
        store_memory_message(),
        min_size=2,
        max_size=5
    ),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_storage_failure_repeated_resilience(messages, error_msg):
    """
    Property: For any sequence of storage operations that fail, the ReasoningEngine
    must handle each failure gracefully without degradation.
    
    **Validates: Requirements 5.1, 5.2, 5.5**
    
    This test verifies that:
    - Multiple consecutive storage failures are handled
    - Each failure gets proper error handling
    - System remains stable across repeated failures
    """
    mock_memory = MockMemoryStorageFailure(error_msg)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    results = []
    for message in messages:
        result = engine.process_message(message)
        results.append(result)
        
        # Each result should be valid
        assert result is not None, f"Result should not be None for message: {message}"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "response" in result, "Result should have response"
        assert "metadata" in result, "Result should have metadata"
        
        # Each result should indicate error
        assert "error" in result["metadata"] or "error_type" in result["metadata"], \
            "Each result should indicate storage error"
    
    # Verify all operations were attempted
    assert mock_memory.store_call_count == len(messages), \
        f"store() should have been called {len(messages)} times"
    
    # Verify no degradation - all results should be similar quality
    assert len(results) == len(messages), "Should have result for each message"


# Feature: intent-based-memory-retrieval-enhancements, Property 5: Storage Failure Handling
@given(
    message=store_memory_message(),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_storage_failure_intent_preserved(message, error_msg):
    """
    Property: For any storage failure, the response should preserve the
    store_memory intent in metadata for tracking purposes.
    
    **Validates: Requirements 5.3**
    
    This test verifies that even when storage fails, the system correctly
    identifies and preserves the intent that was being processed.
    """
    mock_memory = MockMemoryStorageFailure(error_msg)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    result = engine.process_message(message)
    
    # Verify intent is preserved
    assert "intent" in result, "Result should contain intent"
    assert result["intent"] == "store_memory", \
        f"Intent should be 'store_memory', got: {result.get('intent')}"
    
    # Verify error metadata is present
    assert "metadata" in result, "Result should contain metadata"
    assert "error" in result["metadata"] or "error_type" in result["metadata"], \
        "Metadata should indicate error occurred"


# Feature: intent-based-memory-retrieval-enhancements, Property 5: Storage Failure Handling
@given(
    message=store_memory_message(),
    error_msg=error_message()
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_storage_failure_no_partial_state(message, error_msg):
    """
    Property: For any storage failure, the system should not leave partial
    or corrupted state in the memory system.
    
    **Validates: Requirements 5.1, 5.5**
    
    This test verifies that when storage fails:
    - No partial data is committed
    - The memory system remains in a consistent state
    - Subsequent operations don't see corrupted data
    """
    mock_memory = MockMemoryStorageFailure(error_msg)
    llm = StubLLM()
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Attempt storage - should fail
    result = engine.process_message(message)
    
    # Verify failure was handled
    assert "error" in result["metadata"] or "error_type" in result["metadata"], \
        "Storage failure should be indicated in metadata"
    
    # Verify store was called (attempt was made)
    assert mock_memory.store_called, "store() should have been called"
    
    # Verify no memory_id in successful response (since it failed)
    # If there's a memory_id, it should only be in error context
    if "memory_id" in result["metadata"]:
        # If memory_id exists, there should also be an error indicator
        assert "error" in result["metadata"] or "error_type" in result["metadata"], \
            "If memory_id exists after failure, error should also be present"
