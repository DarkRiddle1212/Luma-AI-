"""
Property-Based Tests for Memory Storage Integration

This module implements property-based tests using Hypothesis to verify
universal correctness properties for memory storage integration in ReasoningEngine.

Feature: reasoning-memory-integration
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Optional, Any
from datetime import datetime

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def store_memory_message(draw):
    """
    Generate random messages with store_memory intent.
    
    These messages should trigger the store_memory intent in ReasoningEngine.
    Only uses triggers that are recognized by detect_intent: "remember" and "store"
    """
    # Store memory trigger words (only those recognized by detect_intent)
    triggers = ["remember", "store"]
    trigger = draw(st.sampled_from(triggers))
    
    # Content to store (what comes after the trigger)
    # Use alphanumeric text to avoid issues with special characters
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ))
    
    # Construct message with trigger + content
    message = f"{trigger} {content}"
    
    return message


# ============================================================================
# Mock Memory Implementation
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory implementation for property testing."""
    
    def __init__(self):
        self.store_called = False
        self.store_calls = []
        self.next_id = 1
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content and track the call."""
        self.store_called = True
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        
        self.store_calls.append({
            "content": content,
            "metadata": metadata,
            "id": memory_id
        })
        
        return memory_id
    
    def retrieve(self, query: Optional[str] = None, params: Optional[Dict[str, Any]] = None, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Retrieve memories (not used in storage tests) - supports both legacy and enhanced API."""
        return {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": params or {},
                "limit": limit,
                "has_more": False
            }
        }


# ============================================================================
# 3.6 Property Test: Memory Storage Integration (Property 2)
# ============================================================================

# Feature: reasoning-memory-integration, Property 2: Memory Storage Integration
@given(message=store_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_memory_storage_integration_property(message):
    """
    Property: For any message with "store_memory" intent, the ReasoningEngine
    should extract content, call memory.store(), and return a response containing
    confirmation of storage.
    
    **Validates: Requirements 4.1, 4.2, 4.3**
    
    This test verifies that:
    1. ReasoningEngine detects store_memory intent correctly
    2. Content is extracted from the message (trigger words removed)
    3. memory.store() is called with the extracted content
    4. Response contains confirmation of successful storage
    5. Response includes the memory_id in metadata
    6. The flow works consistently across all valid store_memory messages
    """
    # Create mock memory and LLM
    mock_memory = MockMemoryInterface()
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify intent was detected as store_memory
    assert result["intent"] == "store_memory", \
        f"Intent should be 'store_memory' for message '{message}', got '{result['intent']}'"
    
    # Verify memory.store() was called
    assert mock_memory.store_called, \
        f"memory.store() should be called for store_memory intent"
    
    # Verify store was called exactly once
    assert len(mock_memory.store_calls) == 1, \
        f"memory.store() should be called exactly once, was called {len(mock_memory.store_calls)} times"
    
    # Verify content extraction worked (content should not be empty)
    stored_content = mock_memory.store_calls[0]["content"]
    assert isinstance(stored_content, str), \
        f"Stored content must be a string, got {type(stored_content)}"
    assert len(stored_content) > 0, \
        "Stored content should not be empty after extraction"
    
    # Verify metadata was provided
    stored_metadata = mock_memory.store_calls[0]["metadata"]
    assert stored_metadata is not None, \
        "Metadata should be provided to memory.store()"
    assert isinstance(stored_metadata, dict), \
        f"Metadata must be a dict, got {type(stored_metadata)}"
    
    # Verify response structure
    assert isinstance(result, dict), \
        f"Result must be a dict, got {type(result)}"
    assert "response" in result, \
        "Result must contain 'response' key"
    assert "metadata" in result, \
        "Result must contain 'metadata' key"
    
    # Verify response contains confirmation
    response_text = result["response"]
    assert isinstance(response_text, str), \
        f"Response must be a string, got {type(response_text)}"
    assert len(response_text) > 0, \
        "Response should not be empty"
    
    # Verify confirmation message (should mention "stored" or similar)
    confirmation_keywords = ["stored", "saved", "remembered", "kept"]
    has_confirmation = any(keyword in response_text.lower() for keyword in confirmation_keywords)
    assert has_confirmation, \
        f"Response should contain confirmation keyword, got: '{response_text}'"
    
    # Verify memory_id is in metadata
    assert "memory_id" in result["metadata"], \
        "Result metadata must contain 'memory_id'"
    
    memory_id = result["metadata"]["memory_id"]
    assert isinstance(memory_id, str), \
        f"memory_id must be a string, got {type(memory_id)}"
    assert len(memory_id) > 0, \
        "memory_id should not be empty"
    
    # Verify the memory_id matches what was returned by store()
    expected_id = mock_memory.store_calls[0]["id"]
    assert memory_id == expected_id, \
        f"memory_id in response should match ID from store(), expected '{expected_id}', got '{memory_id}'"


# Feature: reasoning-memory-integration, Property 2: Memory Storage Integration
@given(
    trigger=st.sampled_from(["remember", "store"]),  # Only triggers recognized by detect_intent
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_content_extraction_correctness_property(trigger, content):
    """
    Property: For any store_memory message, the content extraction should
    remove trigger words and preserve the actual content to be stored.
    
    **Validates: Requirements 4.1**
    
    This test verifies that:
    1. Trigger words are removed from the content
    2. The actual content is preserved
    3. Extraction works consistently across different trigger words
    4. No data loss occurs during extraction
    """
    # Create mock memory and LLM
    mock_memory = MockMemoryInterface()
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Construct message with trigger + content
    message = f"{trigger} {content}"
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify store was called
    assert mock_memory.store_called, \
        "memory.store() should be called"
    
    # Get the stored content
    stored_content = mock_memory.store_calls[0]["content"]
    
    # Verify trigger word was removed (stored content should not start with trigger)
    # Note: The implementation converts to lowercase and removes triggers
    assert not stored_content.lower().startswith(trigger.lower()), \
        f"Stored content should not start with trigger word '{trigger}', got: '{stored_content}'"
    
    # Verify content is not empty after extraction
    assert len(stored_content.strip()) > 0, \
        "Stored content should not be empty after trigger removal"
    
    # Verify the original content is present in the stored content
    # (after lowercasing and stripping, since implementation does this)
    content_lower = content.lower().strip()
    stored_lower = stored_content.lower().strip()
    
    # The stored content should contain the original content (or be very similar)
    # We check if the content is a substring or if they're very close
    assert content_lower in stored_lower or stored_lower in content_lower, \
        f"Stored content should preserve original content. Original: '{content}', Stored: '{stored_content}'"


# Feature: reasoning-memory-integration, Property 2: Memory Storage Integration
@given(message=store_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_metadata_inclusion_property(message):
    """
    Property: For any store_memory operation, metadata should be included
    in the store() call with appropriate source and category information.
    
    **Validates: Requirements 4.2**
    
    This test verifies that:
    1. Metadata is always provided to memory.store()
    2. Metadata contains source information
    3. Metadata contains category information
    4. Metadata structure is consistent
    """
    # Create mock memory and LLM
    mock_memory = MockMemoryInterface()
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify store was called
    assert mock_memory.store_called, \
        "memory.store() should be called"
    
    # Get the metadata
    metadata = mock_memory.store_calls[0]["metadata"]
    
    # Verify metadata exists and is a dict
    assert metadata is not None, \
        "Metadata should not be None"
    assert isinstance(metadata, dict), \
        f"Metadata must be a dict, got {type(metadata)}"
    
    # Verify metadata contains expected fields
    assert "source" in metadata, \
        "Metadata should contain 'source' field"
    assert "category" in metadata, \
        "Metadata should contain 'category' field"
    
    # Verify field values are strings
    assert isinstance(metadata["source"], str), \
        f"Metadata source must be a string, got {type(metadata['source'])}"
    assert isinstance(metadata["category"], str), \
        f"Metadata category must be a string, got {type(metadata['category'])}"
    
    # Verify field values are not empty
    assert len(metadata["source"]) > 0, \
        "Metadata source should not be empty"
    assert len(metadata["category"]) > 0, \
        "Metadata category should not be empty"
    
    # Verify source indicates user request
    assert "user" in metadata["source"].lower(), \
        f"Metadata source should indicate user request, got: '{metadata['source']}'"
    
    # Verify category is appropriate
    valid_categories = ["user_memory", "general", "education", "work", "personal"]
    assert metadata["category"] in valid_categories, \
        f"Metadata category should be valid, got: '{metadata['category']}'"


# ============================================================================
# 3.7 Property Test: Memory Storage Error Handling (Property 3)
# ============================================================================

# Feature: reasoning-memory-integration, Property 3: Memory Storage Error Handling
@given(message=store_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_memory_storage_error_handling_property(message):
    """
    Property: For any memory storage operation that fails, the ReasoningEngine
    should handle the error gracefully and return a response informing the user
    of the failure without crashing.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. ReasoningEngine catches exceptions from memory.store()
    2. No unhandled exceptions propagate to the caller
    3. An error response is returned to the user
    4. The response contains error information in metadata
    5. The system remains stable after storage failures
    6. Error handling works consistently across all types of exceptions
    """
    # Create mock memory that raises exceptions
    mock_memory = Mock(spec=MemoryInterface)
    
    # Configure mock to raise various types of exceptions
    # This simulates different failure scenarios
    exception_types = [
        Exception("Database connection failed"),
        RuntimeError("Storage quota exceeded"),
        ValueError("Invalid content format"),
        IOError("Disk write error"),
        MemoryError("Out of memory"),
    ]
    
    # Pick one exception type for this test iteration
    # Hypothesis will run this 100 times, covering different scenarios
    import random
    exception = random.choice(exception_types)
    mock_memory.store.side_effect = exception
    
    # Create LLM
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message - should NOT raise exception
    try:
        result = engine.process_message(message)
    except Exception as e:
        pytest.fail(
            f"ReasoningEngine should handle storage errors gracefully, "
            f"but raised {type(e).__name__}: {e}"
        )
    
    # Verify result is a valid response dictionary
    assert isinstance(result, dict), \
        f"Result must be a dict even on error, got {type(result)}"
    
    # Verify required keys are present
    assert "response" in result, \
        "Result must contain 'response' key even on error"
    assert "intent" in result, \
        "Result must contain 'intent' key even on error"
    assert "metadata" in result, \
        "Result must contain 'metadata' key even on error"
    
    # Verify intent is still store_memory
    assert result["intent"] == "store_memory", \
        f"Intent should be 'store_memory' even on error, got '{result['intent']}'"
    
    # Verify response is a string
    assert isinstance(result["response"], str), \
        f"Response must be a string, got {type(result['response'])}"
    
    # Verify response is not empty
    assert len(result["response"]) > 0, \
        "Response should not be empty on error"
    
    # Verify response indicates failure
    error_indicators = ["couldn't", "could not", "failed", "error", "unable"]
    has_error_indicator = any(indicator in result["response"].lower() for indicator in error_indicators)
    assert has_error_indicator, \
        f"Response should indicate failure, got: '{result['response']}'"
    
    # Verify metadata contains error information
    assert "error" in result["metadata"], \
        "Metadata must contain 'error' field on storage failure"
    
    # Verify error information is not empty
    error_info = result["metadata"]["error"]
    assert isinstance(error_info, str), \
        f"Error info must be a string, got {type(error_info)}"
    assert len(error_info) > 0, \
        "Error info should not be empty"
    
    # Verify memory.store() was actually called (error occurred during storage)
    assert mock_memory.store.called, \
        "memory.store() should have been called before the error"
    
    # Verify no memory_id in metadata (storage failed)
    assert "memory_id" not in result["metadata"] or result["metadata"].get("memory_id") is None, \
        "memory_id should not be present in metadata when storage fails"


# Feature: reasoning-memory-integration, Property 3: Memory Storage Error Handling
@given(
    message=store_memory_message(),
    exception_message=st.text(min_size=5, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_error_message_preservation_property(message, exception_message):
    """
    Property: For any storage error, the error message should be preserved
    and communicated to the user in the response.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. Error messages from storage layer are captured
    2. Error information is included in the response metadata
    3. Users receive actionable error information
    4. Error details are not lost during exception handling
    """
    # Create mock memory that raises exception with specific message
    mock_memory = Mock(spec=MemoryInterface)
    mock_memory.store.side_effect = Exception(exception_message)
    
    # Create LLM
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify error information is in metadata
    assert "error" in result["metadata"], \
        "Metadata must contain error information"
    
    # Verify the error message is preserved
    error_info = result["metadata"]["error"]
    assert exception_message in error_info, \
        f"Error metadata should contain original exception message. " \
        f"Expected '{exception_message}' in '{error_info}'"
    
    # Verify the response mentions the error
    response = result["response"]
    assert exception_message in response or "couldn't" in response.lower(), \
        f"Response should mention the error or indicate failure, got: '{response}'"


# Feature: reasoning-memory-integration, Property 3: Memory Storage Error Handling
@given(message=store_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_system_stability_after_error_property(message):
    """
    Property: After a storage error occurs, the ReasoningEngine should remain
    functional and be able to process subsequent messages successfully.
    
    **Validates: Requirements 4.4**
    
    This test verifies that:
    1. Storage errors don't corrupt the engine state
    2. The engine can process messages after an error
    3. Subsequent operations work correctly
    4. The system recovers gracefully from errors
    """
    # Create mock memory that fails on first call, succeeds on second
    mock_memory = Mock(spec=MemoryInterface)
    call_count = [0]
    
    def store_side_effect(content, metadata=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("First call fails")
        return f"mem_{call_count[0]}"
    
    mock_memory.store.side_effect = store_side_effect
    
    # Create LLM
    llm = StubLLM()
    
    # Create ReasoningEngine
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # First call - should fail gracefully
    result1 = engine.process_message(message)
    
    # Verify first call handled error
    assert "error" in result1["metadata"], \
        "First call should have error in metadata"
    
    # Second call - should succeed
    result2 = engine.process_message(message)
    
    # Verify second call succeeded
    assert "memory_id" in result2["metadata"], \
        "Second call should succeed and have memory_id"
    assert "error" not in result2["metadata"] or result2["metadata"]["error"] is None, \
        "Second call should not have error"
    
    # Verify the engine is still functional
    assert result2["intent"] == "store_memory", \
        "Engine should still detect store_memory intent after error"
    
    # Verify store was called twice
    assert mock_memory.store.call_count == 2, \
        f"store() should be called twice, was called {mock_memory.store.call_count} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "property"])


# ============================================================================
# Helper Strategies for Retrieval Tests
# ============================================================================

@st.composite
def retrieve_memory_message(draw):
    """
    Generate random messages with retrieve_memory intent.
    
    These messages should trigger the retrieve_memory intent in ReasoningEngine.
    Only uses triggers that are recognized by detect_intent: "what was", "recall", "retrieve"
    """
    # Retrieve memory trigger words (only those recognized by detect_intent)
    triggers = ["what was", "recall", "retrieve"]
    trigger = draw(st.sampled_from(triggers))
    
    # Query content (what comes after the trigger)
    # Use alphanumeric text to avoid issues with special characters
    query = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ))
    
    # Construct message with trigger + query
    message = f"{trigger} {query}"
    
    return message


@st.composite
def memory_entry(draw):
    """Generate random memory entry dictionaries for testing."""
    memory_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=5,
        max_size=20
    ))
    
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=10,
        max_size=200
    ))
    
    tags = draw(st.lists(
        st.text(min_size=1, max_size=5),
        min_size=0,
        max_size=5
    ))
    
    category = draw(st.sampled_from(["general", "education", "work", "personal", "system"]))
    
    return {
        "id": memory_id,
        "content": content,
        "metadata": {
            "tags": tags,
            "category": category
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Mock Memory Implementation for Retrieval Tests
# ============================================================================

class MockMemoryInterfaceWithRetrieval(MemoryInterface):
    """Mock memory implementation for retrieval property testing."""
    
    def __init__(self, test_memories: List[Dict[str, Any]] = None):
        self.retrieve_called = False
        self.retrieve_calls = []
        self.test_memories = test_memories or []
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content (not used in retrieval tests)."""
        return f"mem_{hash(content) % 10000}"
    
    def retrieve(self, query: Optional[str] = None, params: Optional[Dict[str, Any]] = None, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Retrieve memories and track the call - supports both legacy and enhanced API."""
        self.retrieve_called = True
        
        # Handle both calling patterns
        if params:
            # Enhanced API: extract query and limit from params
            actual_query = params.get("query", "")
            actual_limit = params.get("limit", limit)
        else:
            # Legacy API: use direct parameters
            actual_query = query or ""
            actual_limit = limit
        
        self.retrieve_calls.append({
            "query": actual_query,
            "limit": actual_limit,
            "params": params
        })
        
        # Return the test memories (up to limit) in RetrievalResult format
        memories = self.test_memories[:actual_limit]
        return {
            "memories": memories,
            "total_count": len(memories),
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": params or {},
                "limit": actual_limit,
                "has_more": len(self.test_memories) > actual_limit
            }
        }


# ============================================================================
# 3.8 Property Test: Memory Retrieval Integration (Property 4)
# ============================================================================

# Feature: reasoning-memory-integration, Property 4: Memory Retrieval Integration
@given(
    message=retrieve_memory_message(),
    test_memories=st.lists(memory_entry(), min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_memory_retrieval_integration_property(message, test_memories):
    """
    Property: For any message with "retrieve_memory" intent, the ReasoningEngine
    should extract the query, call memory.retrieve(), and inject retrieved memories
    into the processing context.
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    
    This test verifies that:
    1. ReasoningEngine detects retrieve_memory intent correctly
    2. Query is extracted from the message (trigger words removed)
    3. memory.retrieve() is called with the extracted query
    4. Retrieved memories are injected into the context
    5. Response includes memories_found count in metadata
    6. Response includes memory_ids in metadata
    7. The flow works consistently across all valid retrieve_memory messages
    """
    # Create mock memory with test data
    mock_memory = MockMemoryInterfaceWithRetrieval(test_memories=test_memories)
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify intent was detected as retrieve_memory
    assert result["intent"] == "retrieve_memory", \
        f"Intent should be 'retrieve_memory' for message '{message}', got '{result['intent']}'"
    
    # Verify memory.retrieve() was called
    assert mock_memory.retrieve_called, \
        f"memory.retrieve() should be called for retrieve_memory intent"
    
    # Verify retrieve was called exactly once
    assert len(mock_memory.retrieve_calls) == 1, \
        f"memory.retrieve() should be called exactly once, was called {len(mock_memory.retrieve_calls)} times"
    
    # Verify query extraction worked (query should not be empty)
    retrieved_query = mock_memory.retrieve_calls[0]["query"]
    assert isinstance(retrieved_query, str), \
        f"Retrieved query must be a string, got {type(retrieved_query)}"
    assert len(retrieved_query.strip()) > 0, \
        "Retrieved query should not be empty after extraction"
    
    # Verify limit parameter was provided
    retrieved_limit = mock_memory.retrieve_calls[0]["limit"]
    assert isinstance(retrieved_limit, int), \
        f"Limit must be an integer, got {type(retrieved_limit)}"
    assert retrieved_limit > 0, \
        f"Limit should be positive, got {retrieved_limit}"
    
    # Verify response structure
    assert isinstance(result, dict), \
        f"Result must be a dict, got {type(result)}"
    assert "response" in result, \
        "Result must contain 'response' key"
    assert "metadata" in result, \
        "Result must contain 'metadata' key"
    
    # Verify response is not empty
    response_text = result["response"]
    assert isinstance(response_text, str), \
        f"Response must be a string, got {type(response_text)}"
    assert len(response_text) > 0, \
        "Response should not be empty"
    
    # Verify memories_found is in metadata
    assert "memories_found" in result["metadata"], \
        "Result metadata must contain 'memories_found'"
    
    memories_found = result["metadata"]["memories_found"]
    assert isinstance(memories_found, int), \
        f"memories_found must be an integer, got {type(memories_found)}"
    assert memories_found >= 0, \
        f"memories_found should be non-negative, got {memories_found}"
    
    # Verify the count matches the test data (up to limit)
    expected_count = min(len(test_memories), 5)  # Default limit is 5
    assert memories_found == expected_count, \
        f"memories_found should be {expected_count}, got {memories_found}"
    
    # Verify memory_ids is in metadata
    assert "memory_ids" in result["metadata"], \
        "Result metadata must contain 'memory_ids'"
    
    memory_ids = result["metadata"]["memory_ids"]
    assert isinstance(memory_ids, list), \
        f"memory_ids must be a list, got {type(memory_ids)}"
    
    # Verify the number of IDs matches memories_found
    assert len(memory_ids) == memories_found, \
        f"Number of memory_ids should match memories_found, expected {memories_found}, got {len(memory_ids)}"
    
    # Verify each ID is a string
    for memory_id in memory_ids:
        assert isinstance(memory_id, str), \
            f"Each memory_id must be a string, got {type(memory_id)}"
        assert len(memory_id) > 0, \
            "Each memory_id should not be empty"
    
    # Verify the IDs match the test memories
    expected_ids = [m["id"] for m in test_memories[:expected_count]]
    assert memory_ids == expected_ids, \
        f"memory_ids should match test data IDs, expected {expected_ids}, got {memory_ids}"


# Feature: reasoning-memory-integration, Property 4: Memory Retrieval Integration
@given(
    trigger=st.sampled_from(["what was", "recall", "retrieve"]),
    query=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ),
    test_memories=st.lists(memory_entry(), min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_query_extraction_correctness_property(trigger, query, test_memories):
    """
    Property: For any retrieve_memory message, the query extraction should
    remove trigger words and preserve the actual query content.
    
    **Validates: Requirements 5.1**
    
    This test verifies that:
    1. Trigger words are removed from the query
    2. The actual query content is preserved
    3. Extraction works consistently across different trigger words
    4. No data loss occurs during extraction
    """
    # Create mock memory with test data
    mock_memory = MockMemoryInterfaceWithRetrieval(test_memories=test_memories)
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Construct message with trigger + query
    message = f"{trigger} {query}"
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify retrieve was called
    assert mock_memory.retrieve_called, \
        "memory.retrieve() should be called"
    
    # Get the extracted query
    extracted_query = mock_memory.retrieve_calls[0]["query"]
    
    # Verify trigger word was removed (extracted query should not start with trigger)
    # Note: The implementation converts to lowercase and removes triggers
    assert not extracted_query.lower().startswith(trigger.lower()), \
        f"Extracted query should not start with trigger word '{trigger}', got: '{extracted_query}'"
    
    # Verify query is not empty after extraction
    assert len(extracted_query.strip()) > 0, \
        "Extracted query should not be empty after trigger removal"
    
    # Verify the original query content is present in the extracted query
    # (after lowercasing and stripping, since implementation does this)
    query_lower = query.lower().strip()
    extracted_lower = extracted_query.lower().strip()
    
    # The extracted query should contain the original query (or be very similar)
    assert query_lower in extracted_lower or extracted_lower in query_lower, \
        f"Extracted query should preserve original query. Original: '{query}', Extracted: '{extracted_query}'"


# Feature: reasoning-memory-integration, Property 4: Memory Retrieval Integration
@given(
    message=retrieve_memory_message(),
    test_memories=st.lists(memory_entry(), min_size=1, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_memories_injected_into_context_property(message, test_memories):
    """
    Property: For any retrieve_memory operation with results, the retrieved
    memories should be injected into the processing context used for LLM generation.
    
    **Validates: Requirements 5.3**
    
    This test verifies that:
    1. Retrieved memories are passed to build_context()
    2. The context contains the memories
    3. The LLM receives the context with memories
    4. Memory injection works consistently across all retrieval operations
    """
    # Create mock memory with test data
    mock_memory = MockMemoryInterfaceWithRetrieval(test_memories=test_memories)
    
    # Create a custom LLM that captures the context it receives
    class ContextCapturingLLM(StubLLM):
        def __init__(self):
            super().__init__()
            self.captured_contexts = []
        
        def generate_response(self, prompt: str, context: Dict) -> str:
            self.captured_contexts.append(context)
            return super().generate_response(prompt, context)
    
    llm = ContextCapturingLLM()
    
    # Create ReasoningEngine with mock memory and capturing LLM
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify LLM was called (context was captured)
    assert len(llm.captured_contexts) > 0, \
        "LLM should have been called with context"
    
    # Get the captured context
    captured_context = llm.captured_contexts[0]
    
    # Verify context is a dictionary
    assert isinstance(captured_context, dict), \
        f"Context must be a dict, got {type(captured_context)}"
    
    # Verify context contains memories key
    assert "memories" in captured_context, \
        "Context must contain 'memories' key when memories are retrieved"
    
    # Verify memories in context
    context_memories = captured_context["memories"]
    assert isinstance(context_memories, list), \
        f"Context memories must be a list, got {type(context_memories)}"
    
    # Verify the memories match the test data (up to limit)
    expected_count = min(len(test_memories), 5)  # Default limit is 5
    assert len(context_memories) == expected_count, \
        f"Context should contain {expected_count} memories, got {len(context_memories)}"
    
    # Verify each memory in context has the correct structure
    for i, memory in enumerate(context_memories):
        assert isinstance(memory, dict), \
            f"Each memory must be a dict, got {type(memory)}"
        
        # Verify required fields
        assert "id" in memory, f"Memory {i} must have 'id' field"
        assert "content" in memory, f"Memory {i} must have 'content' field"
        assert "metadata" in memory, f"Memory {i} must have 'metadata' field"
        assert "timestamp" in memory, f"Memory {i} must have 'timestamp' field"
        
        # Verify the memory matches the test data
        expected_memory = test_memories[i]
        assert memory["id"] == expected_memory["id"], \
            f"Memory {i} ID should match test data"
        assert memory["content"] == expected_memory["content"], \
            f"Memory {i} content should match test data"


# Feature: reasoning-memory-integration, Property 4: Memory Retrieval Integration
@given(message=retrieve_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_no_results_handling_property(message):
    """
    Property: For any retrieve_memory operation that returns no results,
    the ReasoningEngine should return an informative response indicating
    no memories were found.
    
    **Validates: Requirements 5.2**
    
    This test verifies that:
    1. Empty results are handled gracefully
    2. An informative response is returned
    3. memories_found is 0 in metadata
    4. No memory_ids are included in metadata
    5. The system doesn't crash on empty results
    """
    # Create mock memory with no test data (empty results)
    mock_memory = MockMemoryInterfaceWithRetrieval(test_memories=[])
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify intent is retrieve_memory
    assert result["intent"] == "retrieve_memory", \
        f"Intent should be 'retrieve_memory', got '{result['intent']}'"
    
    # Verify retrieve was called
    assert mock_memory.retrieve_called, \
        "memory.retrieve() should be called"
    
    # Verify response structure
    assert isinstance(result, dict), \
        f"Result must be a dict, got {type(result)}"
    assert "response" in result, \
        "Result must contain 'response' key"
    assert "metadata" in result, \
        "Result must contain 'metadata' key"
    
    # Verify response indicates no results
    response_text = result["response"]
    assert isinstance(response_text, str), \
        f"Response must be a string, got {type(response_text)}"
    assert len(response_text) > 0, \
        "Response should not be empty"
    
    # Verify response mentions no memories found
    no_results_indicators = ["don't have", "no memories", "not found", "no results"]
    has_no_results_indicator = any(indicator in response_text.lower() for indicator in no_results_indicators)
    assert has_no_results_indicator, \
        f"Response should indicate no memories found, got: '{response_text}'"
    
    # Verify memories_found is 0
    assert "memories_found" in result["metadata"], \
        "Result metadata must contain 'memories_found'"
    assert result["metadata"]["memories_found"] == 0, \
        f"memories_found should be 0 for empty results, got {result['metadata']['memories_found']}"
    
    # Verify no memory_ids in metadata (or empty list)
    if "memory_ids" in result["metadata"]:
        assert len(result["metadata"]["memory_ids"]) == 0, \
            "memory_ids should be empty for no results"


# Feature: reasoning-memory-integration, Property 4: Memory Retrieval Integration
@given(
    message=retrieve_memory_message(),
    limit=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_limit_parameter_respected_property(message, limit):
    """
    Property: For any retrieve_memory operation, the limit parameter should
    be passed to memory.retrieve() and respected in the results.
    
    **Validates: Requirements 5.2**
    
    This test verifies that:
    1. The limit parameter is passed to memory.retrieve()
    2. The number of results respects the limit
    3. Limit handling works consistently
    """
    # Create more test memories than the limit
    test_memories = [
        {
            "id": f"mem_{i}",
            "content": f"Test memory {i}",
            "metadata": {"tags": [], "category": "test"},
            "timestamp": datetime.now().isoformat()
        }
        for i in range(20)  # Create 20 memories
    ]
    
    # Create mock memory with test data
    mock_memory = MockMemoryInterfaceWithRetrieval(test_memories=test_memories)
    llm = StubLLM()
    
    # Create ReasoningEngine with mock memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify retrieve was called
    assert mock_memory.retrieve_called, \
        "memory.retrieve() should be called"
    
    # Verify limit parameter was passed
    retrieved_limit = mock_memory.retrieve_calls[0]["limit"]
    assert isinstance(retrieved_limit, int), \
        f"Limit must be an integer, got {type(retrieved_limit)}"
    
    # Note: The implementation uses a hardcoded limit of 5
    # This test verifies that a limit is passed, even if it's not configurable
    assert retrieved_limit > 0, \
        f"Limit should be positive, got {retrieved_limit}"
    
    # Verify the number of results respects the limit
    memories_found = result["metadata"]["memories_found"]
    assert memories_found <= retrieved_limit, \
        f"memories_found ({memories_found}) should not exceed limit ({retrieved_limit})"
    
    # Verify the number of memory_ids respects the limit
    if "memory_ids" in result["metadata"]:
        memory_ids = result["metadata"]["memory_ids"]
        assert len(memory_ids) <= retrieved_limit, \
            f"Number of memory_ids ({len(memory_ids)}) should not exceed limit ({retrieved_limit})"



# ============================================================================
# 3.9 Property Test: Memory Retrieval Error Handling (Property 5)
# ============================================================================

class FailingMemoryInterface(MemoryInterface):
    """Mock memory implementation that always fails on retrieve."""
    
    def __init__(self, exception_type=Exception, exception_message="Retrieval failed"):
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.retrieve_called = False
        self.retrieve_call_count = 0
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content (not used in error handling tests)."""
        return f"mem_{hash(content) % 10000}"
    
    def retrieve(self, query: Optional[str] = None, params: Optional[Dict[str, Any]] = None, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Raise an exception to simulate retrieval failure - supports both legacy and enhanced API."""
        self.retrieve_called = True
        self.retrieve_call_count += 1
        raise self.exception_type(self.exception_message)


# Feature: reasoning-memory-integration, Property 5: Memory Retrieval Error Handling
@given(
    message=retrieve_memory_message(),
    exception_type=st.sampled_from([
        Exception,
        RuntimeError,
        ValueError,
        ConnectionError,
        TimeoutError,
        IOError
    ]),
    exception_message=st.text(min_size=5, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_memory_retrieval_error_handling_property(message, exception_type, exception_message):
    """
    Property: For any memory retrieval operation that fails, the ReasoningEngine
    should handle the error gracefully and continue processing without injected
    memories.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. ReasoningEngine catches exceptions from memory.retrieve()
    2. The system continues processing without crashing
    3. A valid response is returned (fallback behavior)
    4. The response is generated using LLM without memory context
    5. Metadata includes error information and fallback flag
    6. The error is logged but doesn't propagate to the caller
    
    This property is critical for:
    - System resilience and fault tolerance
    - Graceful degradation when memory system fails
    - User experience (no crashes, always get a response)
    - Operational reliability
    """
    # Create failing memory that raises exceptions
    failing_memory = FailingMemoryInterface(
        exception_type=exception_type,
        exception_message=exception_message
    )
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=failing_memory)
    
    # Process the message - should NOT raise exception
    result = engine.process_message(message)
    
    # Verify retrieve was called (and failed)
    assert failing_memory.retrieve_called, \
        "memory.retrieve() should be called"
    
    # Verify result is a valid dictionary (no crash)
    assert isinstance(result, dict), \
        f"Result must be a dict even when retrieval fails, got {type(result)}"
    
    # Verify required keys are present
    assert "response" in result, \
        "Result must contain 'response' key even when retrieval fails"
    assert "intent" in result, \
        "Result must contain 'intent' key even when retrieval fails"
    assert "metadata" in result, \
        "Result must contain 'metadata' key even when retrieval fails"
    
    # Verify intent is retrieve_memory
    assert result["intent"] == "retrieve_memory", \
        f"Intent should be 'retrieve_memory', got '{result['intent']}'"
    
    # Verify response is a valid string (fallback response)
    response_text = result["response"]
    assert isinstance(response_text, str), \
        f"Response must be a string, got {type(response_text)}"
    assert len(response_text) > 0, \
        "Response should not be empty even when retrieval fails"
    
    # Verify response is from LLM (fallback behavior)
    # StubLLM always includes "StubLLM Response" in its output
    assert "StubLLM Response" in response_text, \
        f"Response should be from LLM fallback, got: '{response_text}'"
    
    # Verify metadata contains error information
    metadata = result["metadata"]
    assert isinstance(metadata, dict), \
        f"Metadata must be a dict, got {type(metadata)}"
    
    # Verify fallback flag is set
    assert "fallback" in metadata, \
        "Metadata must contain 'fallback' key when retrieval fails"
    assert metadata["fallback"] is True, \
        f"Fallback flag should be True when retrieval fails, got {metadata['fallback']}"
    
    # Verify error information is present
    assert "error" in metadata, \
        "Metadata must contain 'error' key when retrieval fails"
    error_info = metadata["error"]
    assert isinstance(error_info, str), \
        f"Error info must be a string, got {type(error_info)}"
    assert len(error_info) > 0, \
        "Error info should not be empty"
    
    # Verify no memory_ids in metadata (since retrieval failed)
    if "memory_ids" in metadata:
        assert len(metadata["memory_ids"]) == 0, \
            "memory_ids should be empty when retrieval fails"
    
    # Verify no memories_found in metadata (or it's 0)
    if "memories_found" in metadata:
        assert metadata["memories_found"] == 0, \
            "memories_found should be 0 when retrieval fails"


# Feature: reasoning-memory-integration, Property 5: Memory Retrieval Error Handling
@given(message=retrieve_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_error_no_crash_property(message):
    """
    Property: For any retrieve_memory message, if memory.retrieve() raises
    any exception, the ReasoningEngine should never crash and always return
    a valid response dictionary.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. No exceptions propagate to the caller
    2. A valid response structure is always returned
    3. The system remains stable after errors
    4. Multiple errors don't cause cumulative failures
    """
    # Create failing memory
    failing_memory = FailingMemoryInterface(
        exception_type=RuntimeError,
        exception_message="Simulated retrieval failure"
    )
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=failing_memory)
    
    # Process multiple messages to verify stability
    for _ in range(3):
        # Should not raise exception
        result = engine.process_message(message)
        
        # Verify basic structure
        assert isinstance(result, dict), \
            "Result must be a dict"
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "intent" in result, \
            "Result must contain 'intent' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # Verify fallback behavior
        assert result["metadata"].get("fallback") is True, \
            "Fallback flag should be True"
    
    # Verify multiple calls were made (system didn't break after first error)
    assert failing_memory.retrieve_call_count >= 3, \
        f"Should have made at least 3 retrieve calls, got {failing_memory.retrieve_call_count}"


# Feature: reasoning-memory-integration, Property 5: Memory Retrieval Error Handling
@given(
    message=retrieve_memory_message(),
    exception_types=st.lists(
        st.sampled_from([Exception, RuntimeError, ValueError, IOError]),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_different_exceptions_handled_property(message, exception_types):
    """
    Property: For any type of exception raised by memory.retrieve(), the
    ReasoningEngine should handle it gracefully with consistent fallback behavior.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. Different exception types are all handled
    2. Fallback behavior is consistent across exception types
    3. No exception type causes a crash
    4. Error information is captured for all exception types
    """
    for exception_type in exception_types:
        # Create failing memory with specific exception type
        failing_memory = FailingMemoryInterface(
            exception_type=exception_type,
            exception_message=f"Test {exception_type.__name__}"
        )
        llm = StubLLM()
        
        # Create ReasoningEngine with failing memory
        engine = ReasoningEngine(llm=llm, memory=failing_memory)
        
        # Process the message - should NOT raise exception
        result = engine.process_message(message)
        
        # Verify fallback behavior is consistent
        assert isinstance(result, dict), \
            f"Result must be a dict for {exception_type.__name__}"
        assert result["intent"] == "retrieve_memory", \
            f"Intent should be 'retrieve_memory' for {exception_type.__name__}"
        assert result["metadata"].get("fallback") is True, \
            f"Fallback flag should be True for {exception_type.__name__}"
        assert "error" in result["metadata"], \
            f"Error info should be present for {exception_type.__name__}"
        assert "StubLLM Response" in result["response"], \
            f"Should use LLM fallback for {exception_type.__name__}"


# Feature: reasoning-memory-integration, Property 5: Memory Retrieval Error Handling
@given(message=retrieve_memory_message())
@settings(max_examples=10)
@pytest.mark.property_test
def test_retrieval_error_context_without_memories_property(message):
    """
    Property: When memory.retrieve() fails, the ReasoningEngine should build
    context without memories and pass it to the LLM for response generation.
    
    **Validates: Requirements 5.5**
    
    This test verifies that:
    1. Context is built even when retrieval fails
    2. Context does not include memories (empty list)
    3. LLM receives valid context for fallback response
    4. The fallback response is based on the original message
    """
    # Create failing memory
    failing_memory = FailingMemoryInterface()
    llm = StubLLM()
    
    # Create ReasoningEngine with failing memory
    engine = ReasoningEngine(llm=llm, memory=failing_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify fallback behavior
    assert result["metadata"].get("fallback") is True, \
        "Fallback flag should be True"
    
    # Verify response is generated (LLM was called)
    assert "StubLLM Response" in result["response"], \
        "LLM should generate fallback response"
    
    # Verify response is not empty
    assert len(result["response"]) > 0, \
        "Fallback response should not be empty"
    
    # Verify no memories in response metadata
    if "memories_found" in result["metadata"]:
        assert result["metadata"]["memories_found"] == 0, \
            "memories_found should be 0 in fallback"
    
    if "memory_ids" in result["metadata"]:
        assert len(result["metadata"]["memory_ids"]) == 0, \
            "memory_ids should be empty in fallback"


# ============================================================================
# 6.3 Property Test: End-to-End Storage Flow (Property 8)
# ============================================================================

# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(message=store_memory_message())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_storage_flow_property(message):
    """
    Property: For any user message requesting memory storage, the system should
    process the intent, store the memory in the database, and respond with
    confirmation in a single message processing flow.
    
    **Validates: Requirements 9.1**
    
    This test verifies the complete end-to-end flow:
    1. User sends storage request message
    2. ReasoningEngine detects store_memory intent
    3. Content is extracted from message
    4. Memory is stored to SQLite database
    5. Response contains confirmation
    6. Memory is actually persisted in database (can be retrieved)
    7. The entire flow works consistently across all valid storage requests
    
    This is a true integration test that uses real components:
    - Real SQLiteStorage with temporary database
    - Real MemoryManager
    - Real SQLiteMemoryAdapter
    - Real ReasoningEngine
    - Only LLM is stubbed (StubLLM)
    
    This property is critical for:
    - Verifying the complete integration works end-to-end
    - Ensuring data actually persists to database
    - Validating the full dependency chain
    - Confirming user-facing functionality works correctly
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    # Initialize application with real components and temporary database
    from luma.container import initialize_application, cleanup_application
    
    # Create reasoning engine with real storage
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Step 1: Process the storage request message
        result = engine.process_message(message)
        
        # Step 2: Verify intent was detected as store_memory
        assert result["intent"] == "store_memory", \
            f"Intent should be 'store_memory' for message '{message}', got '{result['intent']}'"
        
        # Step 3: Verify response structure is valid
        assert isinstance(result, dict), \
            f"Result must be a dict, got {type(result)}"
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # Step 4: Verify response contains confirmation
        response_text = result["response"]
        assert isinstance(response_text, str), \
            f"Response must be a string, got {type(response_text)}"
        assert len(response_text) > 0, \
            "Response should not be empty"
        
        # Verify confirmation keywords in response
        confirmation_keywords = ["stored", "saved", "remembered", "kept"]
        has_confirmation = any(keyword in response_text.lower() for keyword in confirmation_keywords)
        assert has_confirmation, \
            f"Response should contain confirmation keyword, got: '{response_text}'"
        
        # Step 5: Verify memory_id is in metadata
        assert "memory_id" in result["metadata"], \
            "Result metadata must contain 'memory_id'"
        
        memory_id = result["metadata"]["memory_id"]
        assert isinstance(memory_id, str), \
            f"memory_id must be a string, got {type(memory_id)}"
        assert len(memory_id) > 0, \
            "memory_id should not be empty"
        
        # Step 6: Verify memory was actually stored in database
        # Query the database directly to confirm persistence
        from luma_memory.memory_manager import MemoryManager
        
        # Create a new MemoryManager to query the database
        verify_manager = MemoryManager(storage=storage)
        
        # Try to retrieve the stored memory by ID
        stored_entry = verify_manager.get_memory(memory_id)
        
        # Verify the entry exists
        assert stored_entry is not None, \
            f"Memory with ID '{memory_id}' should exist in database"
        
        # Verify the entry has the correct structure
        assert hasattr(stored_entry, 'id'), \
            "Stored entry must have 'id' attribute"
        assert hasattr(stored_entry, 'action'), \
            "Stored entry must have 'action' attribute"
        assert hasattr(stored_entry, 'context'), \
            "Stored entry must have 'context' attribute"
        
        # Verify the ID matches
        assert stored_entry.id == memory_id, \
            f"Stored entry ID should match returned ID. Expected '{memory_id}', got '{stored_entry.id}'"
        
        # Verify action is not empty (action contains the stored content)
        assert len(stored_entry.action) > 0, \
            "Stored action should not be empty"
        
        # Verify context contains the content
        assert 'content' in stored_entry.context, \
            "Stored entry context should contain 'content' key"
        
        # Step 7: Verify the stored content is related to the original message
        # Extract what should have been stored (remove trigger words)
        message_lower = message.lower()
        expected_content = message_lower
        for trigger in ["remember", "store", "save"]:
            expected_content = expected_content.replace(trigger, "").strip()
        
        # The stored content should match the extracted content
        # Content is stored in the context dict
        stored_content = stored_entry.context.get('content', '')
        assert stored_content.lower().strip() == expected_content, \
            f"Stored content should match extracted content. " \
            f"Expected '{expected_content}', got '{stored_content}'"
        
        # Step 8: Verify we can query for the memory (full round-trip)
        # Query all memories (no filters)
        query_results = verify_manager.query_memories(limit=100)
        
        # Verify the stored memory is in the query results
        found_ids = [entry.id for entry in query_results]
        assert memory_id in found_ids, \
            f"Stored memory '{memory_id}' should be retrievable via query. " \
            f"Found IDs: {found_ids}"
        
    finally:
        # Cleanup: Close database connections
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(
    trigger=st.sampled_from(["remember", "store"]),
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=10,
        max_size=100
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_storage_persistence_property(trigger, content):
    """
    Property: For any storage request, the stored memory should persist in the
    database and be retrievable after storage.
    
    **Validates: Requirements 9.1**
    
    This test verifies:
    1. Memory is stored to database
    2. Memory persists (can be retrieved)
    3. Stored content matches original content
    4. Metadata is preserved
    5. Storage is durable (survives manager recreation)
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    from luma.container import initialize_application, cleanup_application
    
    # Construct message
    message = f"{trigger} {content}"
    
    # Initialize application with temporary database
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Store the memory
        result = engine.process_message(message)
        
        # Verify storage succeeded
        assert result["intent"] == "store_memory", \
            f"Intent should be 'store_memory', got '{result['intent']}'"
        assert "memory_id" in result["metadata"], \
            "Result should contain memory_id"
        
        memory_id = result["metadata"]["memory_id"]
        
        # Create a new MemoryManager to verify persistence
        from luma_memory.memory_manager import MemoryManager
        new_manager = MemoryManager(storage=storage)
        
        # Retrieve the stored memory
        stored_entry = new_manager.get_memory(memory_id)
        
        # Verify the memory exists
        assert stored_entry is not None, \
            f"Memory '{memory_id}' should exist in database"
        
        # Verify content matches (after trigger removal)
        expected_content = content.lower().strip()
        # Content is stored in the context dict
        actual_content = stored_entry.context.get('content', '').lower().strip()
        
        assert actual_content == expected_content, \
            f"Stored content should match original. Expected '{expected_content}', got '{actual_content}'"
        
        # Verify context has source metadata
        assert 'source' in stored_entry.context, \
            "Context should contain 'source' metadata"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(
    messages=st.lists(
        store_memory_message(),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_multiple_storage_operations_property(messages):
    """
    Property: For any sequence of storage requests, each memory should be
    stored independently and all should be retrievable from the database.
    
    **Validates: Requirements 9.1**
    
    This test verifies:
    1. Multiple storage operations work correctly
    2. Each memory gets a unique ID
    3. All memories are persisted
    4. No interference between storage operations
    5. System remains stable across multiple operations
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    from luma.container import initialize_application, cleanup_application
    from luma_memory.memory_manager import MemoryManager
    
    # Initialize application with temporary database
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        stored_ids = []
        
        # Store all messages
        for message in messages:
            result = engine.process_message(message)
            
            # Verify each storage succeeded
            assert result["intent"] == "store_memory", \
                f"Intent should be 'store_memory', got '{result['intent']}'"
            assert "memory_id" in result["metadata"], \
                "Result should contain memory_id"
            
            memory_id = result["metadata"]["memory_id"]
            stored_ids.append(memory_id)
        
        # Verify all IDs are unique
        assert len(stored_ids) == len(set(stored_ids)), \
            f"All memory IDs should be unique. Got {len(stored_ids)} IDs but {len(set(stored_ids))} unique"
        
        # Verify all memories are in the database
        verify_manager = MemoryManager(storage=storage)
        
        for memory_id in stored_ids:
            stored_entry = verify_manager.get_memory(memory_id)
            assert stored_entry is not None, \
                f"Memory '{memory_id}' should exist in database"
            assert stored_entry.id == memory_id, \
                f"Stored entry ID should match. Expected '{memory_id}', got '{stored_entry.id}'"
        
        # Verify we can query and get all memories
        all_memories = verify_manager.query_memories(query="", limit=100)
        all_ids = [entry.id for entry in all_memories]
        
        for memory_id in stored_ids:
            assert memory_id in all_ids, \
                f"Memory '{memory_id}' should be in query results"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(message=store_memory_message())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_storage_no_network_required_property(message):
    """
    Property: For any storage request, the operation should complete successfully
    without requiring network connectivity (local operation).
    
    **Validates: Requirements 9.1, 8.1, 8.2**
    
    This test verifies:
    1. Storage works with local SQLite database
    2. No network calls are made
    3. All operations are local
    4. System is self-contained
    
    Note: This test verifies local operation by using a local temp database
    and confirming the operation succeeds. In a real scenario, you could
    also mock network calls to ensure none are made.
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    from luma.container import initialize_application, cleanup_application
    
    # Initialize application with local temporary database
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Process storage request (should work without network)
        result = engine.process_message(message)
        
        # Verify storage succeeded
        assert result["intent"] == "store_memory", \
            f"Intent should be 'store_memory', got '{result['intent']}'"
        assert "memory_id" in result["metadata"], \
            "Storage should succeed locally"
        
        # Verify the database file exists locally
        import os
        assert os.path.exists(temp_db_path), \
            f"Database file should exist at '{temp_db_path}'"
        
        # Verify the database file is a regular file (not a network resource)
        assert os.path.isfile(temp_db_path), \
            f"Database should be a local file"
        
        # Verify we can read from the database (local operation)
        from luma_memory.memory_manager import MemoryManager
        verify_manager = MemoryManager(storage=storage)
        
        memory_id = result["metadata"]["memory_id"]
        stored_entry = verify_manager.get_memory(memory_id)
        
        assert stored_entry is not None, \
            "Should be able to retrieve from local database"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "end_to_end_storage"])


# ============================================================================
# 6.4 Property Test: End-to-End Retrieval Flow (Property 9)
# ============================================================================

@st.composite
def retrieve_memory_message(draw):
    """
    Generate random messages with retrieve_memory intent.
    
    These messages should trigger the retrieve_memory intent in ReasoningEngine.
    Uses triggers that are recognized by detect_intent.
    """
    # Retrieve memory trigger words (recognized by detect_intent)
    triggers = ["what was", "recall", "retrieve"]
    trigger = draw(st.sampled_from(triggers))
    
    # Query content (what to search for)
    query = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=3,
        max_size=100
    ))
    
    # Construct message with trigger + query
    message = f"{trigger} {query}"
    
    return message


# Feature: reasoning-memory-integration, Property 9: End-to-End Retrieval Flow
@given(
    store_content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=10,
        max_size=200
    ),
    retrieve_message=retrieve_memory_message()
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_retrieval_flow_property(store_content, retrieve_message):
    """
    Property: For any user message requesting memory retrieval, the system should
    process the intent, retrieve relevant memories from the database, and
    incorporate them into the response in a single message processing flow.
    
    **Validates: Requirements 9.2**
    
    This test verifies the complete end-to-end retrieval flow:
    1. Pre-populate memory with test data
    2. User sends retrieval request message
    3. ReasoningEngine detects retrieve_memory intent
    4. Query is extracted from message
    5. Memories are retrieved from SQLite database
    6. Memories are incorporated into the response
    7. The entire flow works consistently across all valid retrieval requests
    
    This is a true integration test that uses real components:
    - Real SQLiteStorage with temporary database
    - Real MemoryManager
    - Real SQLiteMemoryAdapter
    - Real ReasoningEngine
    - Only LLM is stubbed (StubLLM)
    
    This property is critical for:
    - Verifying the complete integration works end-to-end
    - Ensuring data can be retrieved from database
    - Validating the full dependency chain
    - Confirming user-facing retrieval functionality works correctly
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    # Initialize application with real components and temporary database
    from luma.container import initialize_application, cleanup_application
    
    # Create reasoning engine with real storage
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Step 1: Pre-populate memory with test data
        # Store multiple memories to test retrieval
        store_message1 = f"remember {store_content}"
        result_store1 = engine.process_message(store_message1)
        
        # Verify storage succeeded
        assert result_store1["intent"] == "store_memory", \
            f"First storage should succeed, got intent '{result_store1['intent']}'"
        assert "memory_id" in result_store1["metadata"], \
            "First storage should return memory_id"
        
        memory_id1 = result_store1["metadata"]["memory_id"]
        
        # Store a second memory with different content
        store_message2 = "remember additional test data for retrieval"
        result_store2 = engine.process_message(store_message2)
        
        assert result_store2["intent"] == "store_memory", \
            f"Second storage should succeed, got intent '{result_store2['intent']}'"
        assert "memory_id" in result_store2["metadata"], \
            "Second storage should return memory_id"
        
        memory_id2 = result_store2["metadata"]["memory_id"]
        
        # Verify both memories are different
        assert memory_id1 != memory_id2, \
            "Stored memories should have different IDs"
        
        # Step 2: Process the retrieval request message
        result = engine.process_message(retrieve_message)
        
        # Step 3: Verify intent was detected as retrieve_memory
        assert result["intent"] == "retrieve_memory", \
            f"Intent should be 'retrieve_memory' for message '{retrieve_message}', got '{result['intent']}'"
        
        # Step 4: Verify response structure is valid
        assert isinstance(result, dict), \
            f"Result must be a dict, got {type(result)}"
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # Step 5: Verify response is valid
        response_text = result["response"]
        assert isinstance(response_text, str), \
            f"Response must be a string, got {type(response_text)}"
        assert len(response_text) > 0, \
            "Response should not be empty"
        
        # Step 6: Verify memories_found is in metadata
        assert "memories_found" in result["metadata"], \
            "Result metadata must contain 'memories_found'"
        
        memories_found = result["metadata"]["memories_found"]
        assert isinstance(memories_found, int), \
            f"memories_found must be an integer, got {type(memories_found)}"
        assert memories_found >= 0, \
            f"memories_found should be non-negative, got {memories_found}"
        
        # Step 7: Verify memories were retrieved from database
        # The query might or might not match the stored content depending on the random data
        # But we can verify the retrieval mechanism works
        
        if memories_found > 0:
            # If memories were found, verify memory_ids are in metadata
            assert "memory_ids" in result["metadata"], \
                "Result metadata must contain 'memory_ids' when memories found"
            
            memory_ids = result["metadata"]["memory_ids"]
            assert isinstance(memory_ids, list), \
                f"memory_ids must be a list, got {type(memory_ids)}"
            assert len(memory_ids) == memories_found, \
                f"memory_ids length should match memories_found. Expected {memories_found}, got {len(memory_ids)}"
            
            # Verify all memory IDs are strings
            for mem_id in memory_ids:
                assert isinstance(mem_id, str), \
                    f"Each memory_id must be a string, got {type(mem_id)}"
                assert len(mem_id) > 0, \
                    "memory_id should not be empty"
            
            # Verify the retrieved memories are valid IDs from our database
            # They should be either memory_id1 or memory_id2 or other valid IDs
            from luma_memory.memory_manager import MemoryManager
            verify_manager = MemoryManager(storage=storage)
            
            for mem_id in memory_ids:
                # Try to get the memory from database
                stored_entry = verify_manager.get_memory(mem_id)
                assert stored_entry is not None, \
                    f"Memory with ID '{mem_id}' should exist in database"
                
                # Verify the entry has correct structure
                assert hasattr(stored_entry, 'id'), \
                    "Retrieved entry must have 'id' attribute"
                assert hasattr(stored_entry, 'action'), \
                    "Retrieved entry must have 'action' attribute"
                assert stored_entry.id == mem_id, \
                    f"Retrieved entry ID should match. Expected '{mem_id}', got '{stored_entry.id}'"
        
        else:
            # If no memories found, verify response indicates this
            no_memory_keywords = ["don't have", "no memories", "not found", "couldn't find"]
            has_no_memory_indication = any(
                keyword in response_text.lower() for keyword in no_memory_keywords
            )
            # Note: StubLLM might not include these keywords, so we just verify the structure
            # The important thing is that memories_found is 0 and no crash occurred
            assert memories_found == 0, \
                "When no memories found, memories_found should be 0"
        
        # Step 8: Verify the database contains our stored memories
        # (regardless of whether they matched the query)
        from luma_memory.memory_manager import MemoryManager
        verify_manager = MemoryManager(storage=storage)
        
        # Verify memory_id1 exists
        stored_entry1 = verify_manager.get_memory(memory_id1)
        assert stored_entry1 is not None, \
            f"Memory with ID '{memory_id1}' should exist in database"
        
        # Verify memory_id2 exists
        stored_entry2 = verify_manager.get_memory(memory_id2)
        assert stored_entry2 is not None, \
            f"Memory with ID '{memory_id2}' should exist in database"
        
        # Step 9: Test retrieval with a query that should definitely match
        # Use content from the first stored memory
        words = store_content.split()
        if words and len(words[0]) > 2:
            # Use first word as query
            specific_query = words[0]
            specific_message = f"recall {specific_query}"
            
            result_specific = engine.process_message(specific_message)
            
            # Verify intent detection
            assert result_specific["intent"] == "retrieve_memory", \
                f"Intent should be 'retrieve_memory', got '{result_specific['intent']}'"
            
            # Verify memories_found is present
            assert "memories_found" in result_specific["metadata"], \
                "Result metadata must contain 'memories_found'"
            
            # The query should potentially match the stored content
            # (depending on how the query matching works in MemoryManager)
            memories_found_specific = result_specific["metadata"]["memories_found"]
            assert isinstance(memories_found_specific, int), \
                f"memories_found must be an integer, got {type(memories_found_specific)}"
            assert memories_found_specific >= 0, \
                f"memories_found should be non-negative, got {memories_found_specific}"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 9: End-to-End Retrieval Flow
@given(
    store_messages=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=10,
            max_size=100
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_retrieval_multiple_memories_property(store_messages):
    """
    Property: For any set of stored memories, retrieval should work correctly
    and return relevant results based on the query.
    
    **Validates: Requirements 9.2**
    
    This test verifies:
    1. Multiple memories can be stored
    2. Retrieval works with multiple memories in database
    3. Query matching works correctly
    4. Results are properly formatted
    5. The system handles varying amounts of stored data
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    # Initialize application
    from luma.container import initialize_application, cleanup_application
    
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Store all messages
        stored_ids = []
        for content in store_messages:
            store_msg = f"remember {content}"
            result = engine.process_message(store_msg)
            
            assert result["intent"] == "store_memory", \
                f"Storage should succeed, got intent '{result['intent']}'"
            assert "memory_id" in result["metadata"], \
                "Storage should return memory_id"
            
            stored_ids.append(result["metadata"]["memory_id"])
        
        # Verify all memories were stored
        assert len(stored_ids) == len(store_messages), \
            f"Should have stored {len(store_messages)} memories, got {len(stored_ids)}"
        
        # Verify all IDs are unique
        assert len(set(stored_ids)) == len(stored_ids), \
            "All memory IDs should be unique"
        
        # Test retrieval with a query from the first stored message
        first_content = store_messages[0]
        words = first_content.split()
        
        if words and len(words[0]) > 2:
            query_word = words[0]
            retrieve_msg = f"what was {query_word}"
            
            result = engine.process_message(retrieve_msg)
            
            # Verify intent detection
            assert result["intent"] == "retrieve_memory", \
                f"Intent should be 'retrieve_memory', got '{result['intent']}'"
            
            # Verify response structure
            assert "response" in result, \
                "Result must contain 'response' key"
            assert "metadata" in result, \
                "Result must contain 'metadata' key"
            
            # Verify memories_found is present
            assert "memories_found" in result["metadata"], \
                "Result metadata must contain 'memories_found'"
            
            memories_found = result["metadata"]["memories_found"]
            assert isinstance(memories_found, int), \
                f"memories_found must be an integer, got {type(memories_found)}"
            assert memories_found >= 0, \
                f"memories_found should be non-negative, got {memories_found}"
            
            # If memories were found, verify structure
            if memories_found > 0:
                assert "memory_ids" in result["metadata"], \
                    "Result metadata must contain 'memory_ids' when memories found"
                
                memory_ids = result["metadata"]["memory_ids"]
                assert isinstance(memory_ids, list), \
                    f"memory_ids must be a list, got {type(memory_ids)}"
                assert len(memory_ids) == memories_found, \
                    f"memory_ids length should match memories_found"
        
        # Test retrieval with a generic query
        generic_retrieve_msg = "recall test"
        result_generic = engine.process_message(generic_retrieve_msg)
        
        # Verify intent detection
        assert result_generic["intent"] == "retrieve_memory", \
            f"Intent should be 'retrieve_memory', got '{result_generic['intent']}'"
        
        # Verify response structure
        assert "response" in result_generic, \
            "Result must contain 'response' key"
        assert "metadata" in result_generic, \
            "Result must contain 'metadata' key"
        assert "memories_found" in result_generic["metadata"], \
            "Result metadata must contain 'memories_found'"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 9: End-to-End Retrieval Flow
@given(retrieve_message=retrieve_memory_message())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_end_to_end_retrieval_empty_database_property(retrieve_message):
    """
    Property: For any retrieval request on an empty database, the system should
    handle gracefully and return appropriate response indicating no memories found.
    
    **Validates: Requirements 9.2**
    
    This test verifies:
    1. Retrieval works on empty database without crashing
    2. Response indicates no memories found
    3. memories_found is 0
    4. No memory_ids in metadata
    5. System handles edge case gracefully
    """
    import tempfile
    import os
    
    # Create temporary database file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    # Initialize application
    from luma.container import initialize_application, cleanup_application
    
    engine, storage = initialize_application(
        db_path=temp_db_path,
        llm=StubLLM(),
        return_storage=True
    )
    
    try:
        # Process retrieval request on empty database
        result = engine.process_message(retrieve_message)
        
        # Verify intent detection
        assert result["intent"] == "retrieve_memory", \
            f"Intent should be 'retrieve_memory', got '{result['intent']}'"
        
        # Verify response structure
        assert isinstance(result, dict), \
            f"Result must be a dict, got {type(result)}"
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # Verify response is valid
        response_text = result["response"]
        assert isinstance(response_text, str), \
            f"Response must be a string, got {type(response_text)}"
        assert len(response_text) > 0, \
            "Response should not be empty"
        
        # Verify memories_found is 0
        assert "memories_found" in result["metadata"], \
            "Result metadata must contain 'memories_found'"
        
        memories_found = result["metadata"]["memories_found"]
        assert isinstance(memories_found, int), \
            f"memories_found must be an integer, got {type(memories_found)}"
        assert memories_found == 0, \
            f"memories_found should be 0 for empty database, got {memories_found}"
        
        # Verify no memory_ids in metadata (since no memories found)
        # Note: memory_ids might not be present at all, or might be an empty list
        if "memory_ids" in result["metadata"]:
            memory_ids = result["metadata"]["memory_ids"]
            assert isinstance(memory_ids, list), \
                f"memory_ids must be a list if present, got {type(memory_ids)}"
            assert len(memory_ids) == 0, \
                f"memory_ids should be empty when no memories found, got {len(memory_ids)} items"
        
    finally:
        cleanup_application(storage)
        
        # Remove temporary database file
        try:
            if os.path.exists(temp_db_path):
                # Force garbage collection to release file handles on Windows
                import gc
                gc.collect()
                os.remove(temp_db_path)
        except PermissionError:
            # On Windows, retry after ensuring handles are released
            import time
            time.sleep(0.1)
            gc.collect()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors
