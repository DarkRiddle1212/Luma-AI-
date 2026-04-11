"""
Unit tests for inject_memories context dictionary handling.

Tests that inject_memories correctly:
1. Adds "memories" key to context with transformed list
2. Handles existing_context parameter (merge if provided)
3. Ensures "memories" key always exists (even if empty)

Requirements tested:
- 1.1: Inject memories under "memories" key
- 1.5: Inject empty list when no results exist
- Task 4.3: Inject memories into context dictionary

Feature: context-injection-memory-relevance
"""

import pytest
from typing import Dict, Any, List

from luma.core.context_injection import (
    inject_memories,
    InjectionConfig
)
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult,
    MemoryRetrievalError
)


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
# Helper Functions
# ============================================================================


def create_test_memory(
    memory_id: str,
    content: str,
    category: str = "test",
    tags: List[str] = None
) -> MemoryEntry:
    """Create a test MemoryEntry."""
    return {
        "id": memory_id,
        "content": content,
        "category": category,
        "timestamp": "2024-01-15T10:30:00",
        "metadata": {"source": "test"},
        "tags": tags or []
    }


# ============================================================================
# Test Suite: Context Dictionary Handling
# ============================================================================


class TestInjectMemoriesContextHandling:
    """Test suite for inject_memories context dictionary handling."""
    
    def test_memories_key_added_to_new_context(self):
        """
        Test that "memories" key is added to context when no existing_context provided.
        
        Validates: Requirement 1.1
        """
        # Create test data
        memories = [
            create_test_memory("mem_1", "Test memory 1"),
            create_test_memory("mem_2", "Test memory 2")
        ]
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories (no existing_context)
        context = inject_memories("test query", memory_interface, config)
        
        # Verify "memories" key exists
        assert "memories" in context, "Context must contain 'memories' key"
        
        # Verify memories list is correct
        assert isinstance(context["memories"], list)
        assert len(context["memories"]) == 2
        assert context["memories"][0]["id"] == "mem_1"
        assert context["memories"][1]["id"] == "mem_2"
    
    def test_memories_key_added_to_existing_context(self):
        """
        Test that "memories" key is added when existing_context is provided.
        
        Validates: Requirement 1.1, Task 4.3
        """
        # Create test data
        memories = [create_test_memory("mem_1", "Test memory")]
        
        # Create existing context with other keys
        existing_context = {
            "message": "user message",
            "timestamp": "2024-01-15T10:00:00",
            "session_id": "session_123"
        }
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories with existing_context
        context = inject_memories(
            "test query",
            memory_interface,
            config,
            existing_context=existing_context
        )
        
        # Verify "memories" key was added
        assert "memories" in context, "Context must contain 'memories' key"
        assert len(context["memories"]) == 1
        assert context["memories"][0]["id"] == "mem_1"
        
        # Verify existing keys are preserved
        assert context["message"] == "user message"
        assert context["timestamp"] == "2024-01-15T10:00:00"
        assert context["session_id"] == "session_123"
    
    def test_existing_context_merged_correctly(self):
        """
        Test that existing_context is merged with new memories.
        
        Validates: Task 4.3 (merge if provided)
        """
        # Create test data
        memories = [
            create_test_memory("mem_1", "Memory 1"),
            create_test_memory("mem_2", "Memory 2")
        ]
        
        # Create existing context with multiple keys
        existing_context = {
            "message": "test message",
            "message_length": 12,
            "user_context": {"user_id": "user_123"},
            "timestamp": "2024-01-15T10:00:00",
            "session_id": "session_456",
            "system_state": {"cpu": 50}
        }
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories
        context = inject_memories(
            "test query",
            memory_interface,
            config,
            existing_context=existing_context
        )
        
        # Verify all existing keys are preserved
        assert context["message"] == "test message"
        assert context["message_length"] == 12
        assert context["user_context"] == {"user_id": "user_123"}
        assert context["timestamp"] == "2024-01-15T10:00:00"
        assert context["session_id"] == "session_456"
        assert context["system_state"] == {"cpu": 50}
        
        # Verify memories were added
        assert "memories" in context
        assert len(context["memories"]) == 2
        assert context["memories"][0]["id"] == "mem_1"
        assert context["memories"][1]["id"] == "mem_2"
    
    def test_memories_key_always_exists_with_empty_list(self):
        """
        Test that "memories" key exists even when no memories retrieved.
        
        Validates: Requirement 1.5, Task 4.3
        """
        # Create mock interface with no memories
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface([])  # Empty list
        
        # Inject memories
        context = inject_memories("test query", memory_interface, config)
        
        # Verify "memories" key exists with empty list
        assert "memories" in context, "Context must contain 'memories' key even when empty"
        assert isinstance(context["memories"], list)
        assert len(context["memories"]) == 0
    
    def test_memories_key_exists_on_retrieval_failure(self):
        """
        Test that "memories" key exists even when retrieval fails.
        
        Validates: Requirement 1.5, Task 4.3
        """
        # Create mock interface that fails
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface([], should_fail=True)
        
        # Inject memories (should handle failure gracefully)
        context = inject_memories("test query", memory_interface, config)
        
        # Verify "memories" key exists with empty list
        assert "memories" in context, "Context must contain 'memories' key even on failure"
        assert isinstance(context["memories"], list)
        assert len(context["memories"]) == 0
    
    def test_memories_key_exists_with_existing_context_on_failure(self):
        """
        Test that "memories" key is added to existing_context even on failure.
        
        Validates: Requirement 1.5, Task 4.3
        """
        # Create existing context
        existing_context = {
            "message": "test message",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        # Create mock interface that fails
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface([], should_fail=True)
        
        # Inject memories
        context = inject_memories(
            "test query",
            memory_interface,
            config,
            existing_context=existing_context
        )
        
        # Verify "memories" key exists with empty list
        assert "memories" in context, "Context must contain 'memories' key even on failure"
        assert isinstance(context["memories"], list)
        assert len(context["memories"]) == 0
        
        # Verify existing keys are preserved
        assert context["message"] == "test message"
        assert context["timestamp"] == "2024-01-15T10:00:00"
    
    def test_existing_context_not_modified(self):
        """
        Test that the original existing_context dict is not modified.
        
        Validates: Requirement 5.3 (no side effects)
        """
        # Create test data
        memories = [create_test_memory("mem_1", "Test memory")]
        
        # Create existing context
        existing_context = {
            "message": "test message",
            "timestamp": "2024-01-15T10:00:00"
        }
        
        # Store original keys
        original_keys = set(existing_context.keys())
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories
        context = inject_memories(
            "test query",
            memory_interface,
            config,
            existing_context=existing_context
        )
        
        # Verify original existing_context was not modified
        assert set(existing_context.keys()) == original_keys, \
            "Original existing_context should not be modified"
        assert "memories" not in existing_context, \
            "Original existing_context should not have 'memories' key added"
        
        # Verify returned context has memories
        assert "memories" in context
        assert len(context["memories"]) == 1
    
    def test_empty_existing_context_dict(self):
        """
        Test that empty existing_context dict is handled correctly.
        
        Validates: Task 4.3
        """
        # Create test data
        memories = [create_test_memory("mem_1", "Test memory")]
        
        # Create empty existing context
        existing_context = {}
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories
        context = inject_memories(
            "test query",
            memory_interface,
            config,
            existing_context=existing_context
        )
        
        # Verify "memories" key was added
        assert "memories" in context
        assert len(context["memories"]) == 1
        assert context["memories"][0]["id"] == "mem_1"
    
    def test_context_is_dict_type(self):
        """
        Test that returned context is always a dict.
        
        Validates: Requirement 6.1, 6.2
        """
        # Create test data
        memories = [create_test_memory("mem_1", "Test memory")]
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories
        context = inject_memories("test query", memory_interface, config)
        
        # Verify context is a dict
        assert isinstance(context, dict), "Context must be a dictionary"
        
        # Verify "memories" value is a list
        assert isinstance(context["memories"], list), \
            "Context['memories'] must be a list"
    
    def test_multiple_memories_transformed_correctly(self):
        """
        Test that multiple memories are all transformed and added to context.
        
        Validates: Requirement 1.1, 1.2
        """
        # Create test data with varied content
        memories = [
            create_test_memory("mem_1", "First memory", "category1", ["tag1"]),
            create_test_memory("mem_2", "Second memory", "category2", ["tag2", "tag3"]),
            create_test_memory("mem_3", "Third memory", "category1", [])
        ]
        
        # Create mock interface
        config = InjectionConfig(max_memories=10)
        memory_interface = MockMemoryInterface(memories)
        
        # Inject memories
        context = inject_memories("test query", memory_interface, config)
        
        # Verify all memories are in context
        assert "memories" in context
        assert len(context["memories"]) == 3
        
        # Verify each memory has correct structure
        for i, memory in enumerate(context["memories"]):
            assert "id" in memory
            assert "content" in memory
            assert "category" in memory
            assert "timestamp" in memory
            assert "metadata" in memory
            assert "tags" in memory
            
            # Verify values match original
            assert memory["id"] == memories[i]["id"]
            assert memory["content"] == memories[i]["content"]
            assert memory["category"] == memories[i]["category"]
            assert memory["tags"] == memories[i]["tags"]
