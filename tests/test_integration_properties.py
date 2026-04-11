"""
Property-Based Tests for Integration Testing and Validation

This module implements property-based tests using Hypothesis to verify
universal correctness properties for the complete reasoning-memory integration.

Feature: reasoning-memory-integration
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import os

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import MemoryInterface
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def user_message_with_intent(draw):
    """Generate random user messages with various intents."""
    intent_type = draw(st.sampled_from(["store_memory", "retrieve_memory", "general"]))
    
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ))
    
    if intent_type == "store_memory":
        trigger = draw(st.sampled_from(["remember", "store"]))
        message = f"{trigger} {content}"
    elif intent_type == "retrieve_memory":
        trigger = draw(st.sampled_from(["what was", "recall", "retrieve"]))
        message = f"{trigger} {content}"
    else:
        message = content
    
    return message, intent_type


# ============================================================================
# Mock Memory Implementation for Swappability Tests
# ============================================================================

class AlternativeMemoryImplementation(MemoryInterface):
    """Alternative memory implementation for testing swappability."""
    
    def __init__(self):
        self.memories = {}
        self.next_id = 1
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content in memory."""
        memory_id = f"alt_mem_{self.next_id}"
        self.next_id += 1
        
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return memory_id
    
    def retrieve(self, query: Optional[str] = None, params: Optional[Dict[str, Any]] = None, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Retrieve memories matching the query - supports both legacy and enhanced API."""
        # Handle both calling patterns
        if params:
            actual_query = params.get("query", "")
            actual_limit = params.get("limit", limit)
        else:
            actual_query = query or ""
            actual_limit = limit
        
        # Simple substring matching
        results = []
        for memory in self.memories.values():
            if actual_query.lower() in memory["content"].lower():
                results.append(memory)
                if len(results) >= actual_limit:
                    break
        
        return {
            "memories": results,
            "total_count": len(results),
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": params or {},
                "limit": actual_limit,
                "has_more": len(results) >= actual_limit
            }
        }


# ============================================================================
# 6.1 Property Test: Implementation Swappability (Property 6)
# ============================================================================

# Feature: reasoning-memory-integration, Property 6: Implementation Swappability
@given(message_data=user_message_with_intent())
@settings(max_examples=10)
@pytest.mark.property_test
def test_implementation_swappability_property(message_data):
    """
    Property: For any implementation of MemoryInterface, the ReasoningEngine
    should function correctly when provided that implementation, without
    requiring code changes to the ReasoningEngine.
    
    **Validates: Requirements 7.5**
    
    This test verifies that:
    1. ReasoningEngine works with alternative MemoryInterface implementations
    2. All operations work correctly with the alternative implementation
    3. Swapping implementations doesn't break functionality
    4. The abstraction layer successfully decouples implementation details
    5. Clean architecture principles are upheld
    """
    message, expected_intent = message_data
    
    # Create LLM
    llm = StubLLM()
    
    # Test with alternative memory implementation
    alt_memory = AlternativeMemoryImplementation()
    engine_with_alt = ReasoningEngine(llm=llm, memory=alt_memory)
    
    # Process message with alternative implementation
    result_alt = engine_with_alt.process_message(message)
    
    # Verify result is valid
    assert isinstance(result_alt, dict), \
        f"Result must be a dict with alternative implementation, got {type(result_alt)}"
    
    assert "response" in result_alt, \
        "Result must contain 'response' key with alternative implementation"
    assert "intent" in result_alt, \
        "Result must contain 'intent' key with alternative implementation"
    assert "metadata" in result_alt, \
        "Result must contain 'metadata' key with alternative implementation"
    
    # Verify response is valid
    assert isinstance(result_alt["response"], str), \
        f"Response must be a string, got {type(result_alt['response'])}"
    assert len(result_alt["response"]) > 0, \
        "Response should not be empty"
    
    # Verify intent detection works
    assert isinstance(result_alt["intent"], str), \
        f"Intent must be a string, got {type(result_alt['intent'])}"
    
    # For store_memory intent, verify storage worked
    if result_alt["intent"] == "store_memory":
        assert "memory_id" in result_alt["metadata"], \
            "store_memory should include memory_id in metadata"
        
        memory_id = result_alt["metadata"]["memory_id"]
        assert memory_id in alt_memory.memories, \
            f"Memory should be stored in alternative implementation"
        
        # Verify the stored memory has correct structure
        stored_memory = alt_memory.memories[memory_id]
        assert "content" in stored_memory, \
            "Stored memory should have content"
        assert "metadata" in stored_memory, \
            "Stored memory should have metadata"
        assert "timestamp" in stored_memory, \
            "Stored memory should have timestamp"
    
    # For retrieve_memory intent, verify retrieval worked
    elif result_alt["intent"] == "retrieve_memory":
        assert "memories_found" in result_alt["metadata"], \
            "retrieve_memory should include memories_found in metadata"
        
        memories_found = result_alt["metadata"]["memories_found"]
        assert isinstance(memories_found, int), \
            f"memories_found must be an integer, got {type(memories_found)}"
        assert memories_found >= 0, \
            f"memories_found should be non-negative, got {memories_found}"
    
    # Now test with SQLite implementation (if we have a temp database)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_swap.db")
        
        # Create SQLite-based memory
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        sqlite_memory = SQLiteMemoryAdapter(memory_manager=memory_manager)
        
        # Create engine with SQLite implementation
        engine_with_sqlite = ReasoningEngine(llm=StubLLM(), memory=sqlite_memory)
        
        # Process the same message with SQLite implementation
        result_sqlite = engine_with_sqlite.process_message(message)
        
        # Verify result is valid
        assert isinstance(result_sqlite, dict), \
            f"Result must be a dict with SQLite implementation, got {type(result_sqlite)}"
        
        assert "response" in result_sqlite, \
            "Result must contain 'response' key with SQLite implementation"
        assert "intent" in result_sqlite, \
            "Result must contain 'intent' key with SQLite implementation"
        assert "metadata" in result_sqlite, \
            "Result must contain 'metadata' key with SQLite implementation"
        
        # Verify both implementations detected the same intent
        assert result_alt["intent"] == result_sqlite["intent"], \
            f"Both implementations should detect same intent. " \
            f"Alternative: '{result_alt['intent']}', SQLite: '{result_sqlite['intent']}'"
        
        # Verify both implementations produced valid responses
        assert isinstance(result_sqlite["response"], str), \
            f"Response must be a string, got {type(result_sqlite['response'])}"
        assert len(result_sqlite["response"]) > 0, \
            "Response should not be empty"
        
        # Verify metadata structure is consistent
        if result_sqlite["intent"] == "store_memory":
            assert "memory_id" in result_sqlite["metadata"], \
                "store_memory should include memory_id in metadata with SQLite"
        elif result_sqlite["intent"] == "retrieve_memory":
            assert "memories_found" in result_sqlite["metadata"], \
                "retrieve_memory should include memories_found in metadata with SQLite"
        
        # Clean up
        storage.close()


# Feature: reasoning-memory-integration, Property 6: Implementation Swappability
@given(
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_swappability_store_operations_property(content):
    """
    Property: For any content to store, both alternative and SQLite implementations
    should successfully store the content and return valid IDs.
    
    **Validates: Requirements 7.5**
    
    This test verifies that:
    1. Different implementations can store content
    2. Both return valid memory IDs
    3. The ReasoningEngine works identically with both
    4. Implementation details are properly abstracted
    """
    message = f"remember {content}"
    
    # Test with alternative implementation
    alt_memory = AlternativeMemoryImplementation()
    engine_alt = ReasoningEngine(llm=StubLLM(), memory=alt_memory)
    result_alt = engine_alt.process_message(message)
    
    # Verify alternative implementation worked
    assert result_alt["intent"] == "store_memory", \
        f"Intent should be store_memory, got '{result_alt['intent']}'"
    assert "memory_id" in result_alt["metadata"], \
        "Alternative implementation should return memory_id"
    
    alt_memory_id = result_alt["metadata"]["memory_id"]
    assert isinstance(alt_memory_id, str), \
        f"memory_id must be a string, got {type(alt_memory_id)}"
    assert len(alt_memory_id) > 0, \
        "memory_id should not be empty"
    
    # Test with SQLite implementation
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        sqlite_memory = SQLiteMemoryAdapter(memory_manager=memory_manager)
        
        engine_sqlite = ReasoningEngine(llm=StubLLM(), memory=sqlite_memory)
        result_sqlite = engine_sqlite.process_message(message)
        
        # Verify SQLite implementation worked
        assert result_sqlite["intent"] == "store_memory", \
            f"Intent should be store_memory, got '{result_sqlite['intent']}'"
        assert "memory_id" in result_sqlite["metadata"], \
            "SQLite implementation should return memory_id"
        
        sqlite_memory_id = result_sqlite["metadata"]["memory_id"]
        assert isinstance(sqlite_memory_id, str), \
            f"memory_id must be a string, got {type(sqlite_memory_id)}"
        assert len(sqlite_memory_id) > 0, \
            "memory_id should not be empty"
        
        # Verify both implementations produced valid responses
        assert "stored" in result_alt["response"].lower() or "saved" in result_alt["response"].lower(), \
            f"Alternative implementation should confirm storage: '{result_alt['response']}'"
        assert "stored" in result_sqlite["response"].lower() or "saved" in result_sqlite["response"].lower(), \
            f"SQLite implementation should confirm storage: '{result_sqlite['response']}'"
        
        # Clean up
        storage.close()


# Feature: reasoning-memory-integration, Property 6: Implementation Swappability
@given(
    store_content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ),
    retrieve_query=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=3,
        max_size=50
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_swappability_retrieve_operations_property(store_content, retrieve_query):
    """
    Property: For any stored content and retrieval query, both alternative and
    SQLite implementations should handle retrieval operations correctly.
    
    **Validates: Requirements 7.5**
    
    This test verifies that:
    1. Different implementations can retrieve content
    2. Both handle empty results gracefully
    3. Both return results in the correct format
    4. The ReasoningEngine works identically with both
    """
    # Test with alternative implementation
    alt_memory = AlternativeMemoryImplementation()
    engine_alt = ReasoningEngine(llm=StubLLM(), memory=alt_memory)
    
    # Store content first
    store_message = f"remember {store_content}"
    engine_alt.process_message(store_message)
    
    # Retrieve content
    retrieve_message = f"recall {retrieve_query}"
    result_alt = engine_alt.process_message(retrieve_message)
    
    # Verify alternative implementation worked
    assert result_alt["intent"] == "retrieve_memory", \
        f"Intent should be retrieve_memory, got '{result_alt['intent']}'"
    assert "memories_found" in result_alt["metadata"], \
        "Alternative implementation should return memories_found"
    
    memories_found_alt = result_alt["metadata"]["memories_found"]
    assert isinstance(memories_found_alt, int), \
        f"memories_found must be an integer, got {type(memories_found_alt)}"
    assert memories_found_alt >= 0, \
        f"memories_found should be non-negative, got {memories_found_alt}"
    
    # Test with SQLite implementation
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_retrieve.db")
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        sqlite_memory = SQLiteMemoryAdapter(memory_manager=memory_manager)
        
        engine_sqlite = ReasoningEngine(llm=StubLLM(), memory=sqlite_memory)
        
        # Store content first
        engine_sqlite.process_message(store_message)
        
        # Retrieve content
        result_sqlite = engine_sqlite.process_message(retrieve_message)
        
        # Verify SQLite implementation worked
        assert result_sqlite["intent"] == "retrieve_memory", \
            f"Intent should be retrieve_memory, got '{result_sqlite['intent']}'"
        assert "memories_found" in result_sqlite["metadata"], \
            "SQLite implementation should return memories_found"
        
        memories_found_sqlite = result_sqlite["metadata"]["memories_found"]
        assert isinstance(memories_found_sqlite, int), \
            f"memories_found must be an integer, got {type(memories_found_sqlite)}"
        assert memories_found_sqlite >= 0, \
            f"memories_found should be non-negative, got {memories_found_sqlite}"
        
        # Verify both implementations produced valid responses
        assert isinstance(result_alt["response"], str), \
            f"Alternative response must be a string, got {type(result_alt['response'])}"
        assert isinstance(result_sqlite["response"], str), \
            f"SQLite response must be a string, got {type(result_sqlite['response'])}"
        
        assert len(result_alt["response"]) > 0, \
            "Alternative response should not be empty"
        assert len(result_sqlite["response"]) > 0, \
            "SQLite response should not be empty"
        
        # Clean up
        storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "property"])


# ============================================================================
# 6.2 Property Test: Memory Persistence Round-Trip (Property 7)
# ============================================================================

# Feature: reasoning-memory-integration, Property 7: Memory Persistence Round-Trip
@given(
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_memory_persistence_round_trip_property(content):
    """
    Property: For any valid memory content, storing it to memory and then
    restarting the application should result in the memory being retrievable
    from local storage.
    
    **Validates: Requirements 8.3, 8.4**
    
    This test verifies that:
    1. Memories are persisted to local storage
    2. Application restart doesn't lose data
    3. Stored memories can be retrieved after restart
    4. Content matches original after round-trip
    5. Persistence works consistently across all valid inputs
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_persistence.db")
        
        # Phase 1: Store memory
        storage1 = SQLiteStorage(db_path=db_path)
        memory_manager1 = MemoryManager(storage=storage1)
        memory1 = SQLiteMemoryAdapter(memory_manager=memory_manager1)
        engine1 = ReasoningEngine(llm=StubLLM(), memory=memory1)
        
        # Store the content
        store_message = f"remember {content}"
        result_store = engine1.process_message(store_message)
        
        # Verify storage succeeded
        assert result_store["intent"] == "store_memory", \
            f"Intent should be store_memory, got '{result_store['intent']}'"
        assert "memory_id" in result_store["metadata"], \
            "Storage should return memory_id"
        
        memory_id = result_store["metadata"]["memory_id"]
        
        # Close the first application instance
        storage1.close()
        del engine1
        del memory1
        del memory_manager1
        del storage1
        
        # Phase 2: Simulate application restart - reinitialize components
        storage2 = SQLiteStorage(db_path=db_path)
        memory_manager2 = MemoryManager(storage=storage2)
        memory2 = SQLiteMemoryAdapter(memory_manager=memory_manager2)
        engine2 = ReasoningEngine(llm=StubLLM(), memory=memory2)
        
        # Retrieve the content using a query that should match
        # Extract a keyword from the content for the query
        words = content.split()
        if words:
            query_word = words[0] if len(words[0]) > 2 else content[:5]
        else:
            query_word = content[:5]
        
        retrieve_message = f"recall {query_word}"
        result_retrieve = engine2.process_message(retrieve_message)
        
        # Verify retrieval succeeded
        assert result_retrieve["intent"] == "retrieve_memory", \
            f"Intent should be retrieve_memory, got '{result_retrieve['intent']}'"
        assert "memories_found" in result_retrieve["metadata"], \
            "Retrieval should return memories_found"
        
        memories_found = result_retrieve["metadata"]["memories_found"]
        
        # Verify at least one memory was found (the one we stored)
        # Note: The query might not match if the content doesn't contain the query word
        # But the memory should still be in the database
        assert isinstance(memories_found, int), \
            f"memories_found must be an integer, got {type(memories_found)}"
        assert memories_found >= 0, \
            f"memories_found should be non-negative, got {memories_found}"
        
        # If memories were found, verify the content is present
        if memories_found > 0:
            assert "memory_ids" in result_retrieve["metadata"], \
                "Retrieval should return memory_ids when memories found"
            
            memory_ids = result_retrieve["metadata"]["memory_ids"]
            assert isinstance(memory_ids, list), \
                f"memory_ids must be a list, got {type(memory_ids)}"
            assert len(memory_ids) > 0, \
                "memory_ids should not be empty when memories found"
        
        # Verify the database file exists and has data
        assert os.path.exists(db_path), \
            "Database file should exist after restart"
        
        # Verify we can query the database directly
        all_memories = memory_manager2.query_memories(action_type="", limit=100)
        assert len(all_memories) > 0, \
            "Database should contain at least one memory after restart"
        
        # Verify the stored memory is in the database
        found_memory = False
        for memory in all_memories:
            if memory.id == memory_id:
                found_memory = True
                # Verify content is preserved
                assert content.lower() in memory.action.lower(), \
                    f"Stored content should be preserved. Original: '{content}', Retrieved: '{memory.action}'"
                break
        
        assert found_memory, \
            f"Memory with ID '{memory_id}' should be found in database after restart"
        
        # Clean up
        storage2.close()