"""
Unit tests for ReasoningEngine enhanced memory integration.

Tests the enhanced memory integration functionality including:
- Retrieval with successful memory operations
- Retrieval with MemoryRetrievalError
- Storage with MemoryStorageError
- Operation without memory configured
- Context injection with memories
- Metadata in responses

Requirements: 3.1, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 11.1, 11.2, 11.4
"""

import pytest
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError,
    QueryParameters,
    RetrievalResult,
    MemoryEntry
)
from typing import Dict, List, Optional, Any


class MockMemorySuccess(MemoryInterface):
    """Mock memory implementation that succeeds."""
    
    def __init__(self):
        self.stored_memories = []
        self.next_id = 1
        self.retrieve_calls = []
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content and return ID."""
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        
        memory_entry = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": "2024-01-15T10:00:00",
            "category": metadata.get("category", "general") if metadata else "general",
            "tags": metadata.get("tags", []) if metadata else []
        }
        self.stored_memories.append(memory_entry)
        return memory_id
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Retrieve memories matching query - enhanced API."""
        import time
        start_time = time.time()
        
        # Track retrieve calls
        self.retrieve_calls.append({"query": query, "params": params, "limit": limit})
        
        # Handle both legacy and enhanced API
        if params:
            query = params.get("query")
            limit = params.get("limit", limit)
            category = params.get("category")
            tags = params.get("tags")
        else:
            category = None
            tags = None
        
        # Simple substring matching for testing
        results = []
        for mem in self.stored_memories:
            match = True
            
            # Query matching
            if query and query.lower() not in mem["content"].lower():
                match = False
            
            # Category matching
            if category and mem["category"] != category:
                match = False
            
            # Tags matching (must contain all specified tags)
            if tags:
                mem_tags = set(mem["tags"])
                required_tags = set(tags)
                if not required_tags.issubset(mem_tags):
                    match = False
            
            if match:
                results.append(mem)
                if len(results) >= limit:
                    break
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Build filters_applied
        filters_applied = {}
        if query:
            filters_applied["query"] = query
        if category:
            filters_applied["category"] = category
        if tags:
            filters_applied["tags"] = tags
        
        # Return RetrievalResult
        return {
            "memories": results,
            "total_count": len(results),
            "query_metadata": {
                "execution_time_ms": execution_time_ms,
                "filters_applied": filters_applied,
                "limit": limit,
                "has_more": False
            }
        }


class MockMemoryRetrievalFailure(MemoryInterface):
    """Mock memory that fails on retrieval."""
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return "mem_1"
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """Always raises MemoryRetrievalError."""
        raise MemoryRetrievalError("Database connection failed")


class MockMemoryStorageFailure(MemoryInterface):
    """Mock memory that fails on storage."""
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Always raises MemoryStorageError."""
        raise MemoryStorageError("Disk full")
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10
    ) -> RetrievalResult:
        return {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }


class ContextCapturingLLM(StubLLM):
    """LLM that captures the context passed to generate_response."""
    
    def __init__(self):
        super().__init__()
        self.captured_contexts = []
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        self.captured_contexts.append(context)
        return f"Response based on context with {len(context.get('memories', []))} memories"


class TestRetrievalWithSuccessfulMemoryOperations:
    """Test retrieval with successful memory operations (Requirement 3.1, 3.3, 3.4, 3.5)."""
    
    def test_retrieve_with_matching_memories(self):
        """Test successful retrieval with matching memories."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store test memories
        memory.store("Python is a programming language", metadata={"category": "tech"})
        memory.store("JavaScript is also a language", metadata={"category": "tech"})
        memory.store("Buy groceries tomorrow", metadata={"category": "personal"})
        
        # Retrieve memories - query will be "python" after trigger word removal
        result = engine.process_message("Recall Python")
        
        # Verify intent
        assert result["intent"] == "retrieve_memory"
        
        # Verify memories were found
        assert "memories_found" in result["metadata"]
        assert result["metadata"]["memories_found"] == 1
        
        # Verify memory IDs are included
        assert "memory_ids" in result["metadata"]
        assert len(result["metadata"]["memory_ids"]) == 1
        assert result["metadata"]["memory_ids"][0] == "mem_1"
    
    def test_retrieve_uses_enhanced_api_with_params(self):
        """Test that retrieve uses enhanced API with QueryParameters."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store test memory
        memory.store("Python tutorial", metadata={})
        
        # Retrieve
        result = engine.process_message("Recall Python")
        
        # Verify retrieve was called with params
        assert len(memory.retrieve_calls) == 1
        call = memory.retrieve_calls[0]
        assert call["params"] is not None
        assert "query" in call["params"]
        assert "limit" in call["params"]
    
    def test_retrieve_extracts_query_from_message(self):
        """Test that query is correctly extracted from message."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store test memory
        memory.store("Python programming", metadata={})
        
        # Test various trigger words
        test_cases = [
            ("Recall Python", "python"),
            ("Retrieve Python notes", "python notes"),
        ]
        
        for message, expected_query_part in test_cases:
            memory.retrieve_calls.clear()
            result = engine.process_message(message)
            
            # Verify query was extracted (trigger words removed)
            call = memory.retrieve_calls[0]
            actual_query = call["params"]["query"]
            assert expected_query_part in actual_query.lower()


class TestRetrievalMetadata:
    """Test retrieval result metadata (Requirement 10.5)."""
    
    def test_metadata_includes_execution_time(self):
        """Test that metadata includes execution time."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        memory.store("Test memory", metadata={})
        
        result = engine.process_message("Recall test")
        
        # Verify execution time is included
        assert "execution_time_ms" in result["metadata"]
        assert isinstance(result["metadata"]["execution_time_ms"], (int, float))
        assert result["metadata"]["execution_time_ms"] >= 0
    
    def test_metadata_includes_filters_applied(self):
        """Test that metadata includes filters applied."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        memory.store("Python tutorial", metadata={})
        
        result = engine.process_message("Recall Python")
        
        # Verify filters are included
        assert "filters_applied" in result["metadata"]
        assert isinstance(result["metadata"]["filters_applied"], dict)
    
    def test_metadata_includes_memory_count(self):
        """Test that metadata includes memory count."""
        llm = StubLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        memory.store("Memory 1", metadata={})
        memory.store("Memory 2", metadata={})
        
        result = engine.process_message("Recall Memory")
        
        # Verify count is included
        assert "memories_found" in result["metadata"]
        assert result["metadata"]["memories_found"] == 2


class TestContextInjectionWithMemories:
    """Test context injection with memories (Requirement 3.1, 3.5)."""
    
    def test_memories_injected_into_context(self):
        """Test that retrieved memories are injected into LLM context."""
        llm = ContextCapturingLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store test memories
        memory.store("Python is great", metadata={"category": "tech", "tags": ["python"]})
        memory.store("JavaScript is fun", metadata={"category": "tech", "tags": ["js"]})
        
        # Retrieve memories
        result = engine.process_message("Recall Python")
        
        # Verify context was captured
        assert len(llm.captured_contexts) == 1
        context = llm.captured_contexts[0]
        
        # Verify memories are in context
        assert "memories" in context
        assert len(context["memories"]) == 1
        
        # Verify memory structure
        memory_entry = context["memories"][0]
        assert "id" in memory_entry
        assert "content" in memory_entry
        assert "metadata" in memory_entry
        assert "timestamp" in memory_entry
        assert "category" in memory_entry
        assert "tags" in memory_entry
    
    def test_all_memory_metadata_preserved(self):
        """Test that all memory metadata is preserved in context."""
        llm = ContextCapturingLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store memory with rich metadata
        memory.store(
            "Python tutorial",
            metadata={
                "category": "education",
                "tags": ["python", "tutorial", "programming"],
                "custom_field": "custom_value"
            }
        )
        
        # Retrieve
        result = engine.process_message("Recall Python")
        
        # Verify metadata is preserved
        context = llm.captured_contexts[0]
        memory_entry = context["memories"][0]
        
        assert memory_entry["category"] == "education"
        assert "python" in memory_entry["tags"]
        assert "tutorial" in memory_entry["tags"]
        assert "programming" in memory_entry["tags"]
        assert memory_entry["metadata"]["custom_field"] == "custom_value"
    
    def test_empty_memories_list_when_no_results(self):
        """Test that empty list is injected when no memories match."""
        llm = ContextCapturingLLM()
        memory = MockMemorySuccess()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Don't store any memories
        
        # Retrieve
        result = engine.process_message("Recall Python")
        
        # Verify response indicates no memories
        assert "don't have any memories" in result["response"].lower()
        assert result["metadata"]["memories_found"] == 0


class TestRetrievalWithMemoryRetrievalError:
    """Test retrieval with MemoryRetrievalError (Requirement 4.1, 4.2, 4.3, 4.4, 4.5, 4.6)."""
    
    def test_retrieval_error_caught_and_handled(self):
        """Test that MemoryRetrievalError is caught and handled gracefully."""
        llm = StubLLM()
        memory = MockMemoryRetrievalFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Should not raise exception
        result = engine.process_message("What was my task?")
        
        # Verify intent is correct
        assert result["intent"] == "retrieve_memory"
        
        # Verify response is valid (not an error response)
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_retrieval_error_includes_fallback_flag(self):
        """Test that fallback flag is set when retrieval fails."""
        llm = StubLLM()
        memory = MockMemoryRetrievalFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Recall something")
        
        # Verify fallback flag
        assert "fallback" in result["metadata"]
        assert result["metadata"]["fallback"] is True
    
    def test_retrieval_error_includes_error_message(self):
        """Test that error message is included in metadata."""
        llm = StubLLM()
        memory = MockMemoryRetrievalFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("What was test?")
        
        # Verify error message
        assert "error" in result["metadata"]
        assert "Database connection failed" in result["metadata"]["error"]
    
    def test_retrieval_error_continues_with_llm_processing(self):
        """Test that processing continues with LLM when retrieval fails."""
        llm = ContextCapturingLLM()
        memory = MockMemoryRetrievalFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Recall Python")
        
        # Verify LLM was called
        assert len(llm.captured_contexts) == 1
        
        # Verify context has empty memories (fallback)
        context = llm.captured_contexts[0]
        assert "memories" in context
        assert context["memories"] == []
    
    def test_retrieval_error_does_not_crash_system(self):
        """Test that system doesn't crash on retrieval error."""
        llm = StubLLM()
        memory = MockMemoryRetrievalFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Process multiple messages - should all succeed
        for i in range(3):
            result = engine.process_message(f"What was task {i}?")
            assert result["intent"] == "retrieve_memory"
            assert "fallback" in result["metadata"]


class TestStorageWithMemoryStorageError:
    """Test storage with MemoryStorageError (Requirement 5.1, 5.2, 5.3, 5.4, 5.5)."""
    
    def test_storage_error_caught_and_handled(self):
        """Test that MemoryStorageError is caught and handled gracefully."""
        llm = StubLLM()
        memory = MockMemoryStorageFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Should not raise exception
        result = engine.process_message("Remember to buy milk")
        
        # Verify intent is correct
        assert result["intent"] == "store_memory"
        
        # Verify response is valid
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_storage_error_returns_user_friendly_message(self):
        """Test that user-friendly error message is returned."""
        llm = StubLLM()
        memory = MockMemoryStorageFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Store this information")
        
        # Verify user-friendly message
        assert "couldn't store" in result["response"].lower()
    
    def test_storage_error_includes_error_details_in_metadata(self):
        """Test that error details are included in metadata."""
        llm = StubLLM()
        memory = MockMemoryStorageFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Remember this")
        
        # Verify error details
        assert "error" in result["metadata"]
        assert "Disk full" in result["metadata"]["error"]
        assert "error_type" in result["metadata"]
    
    def test_storage_error_does_not_crash_system(self):
        """Test that system doesn't crash on storage error."""
        llm = StubLLM()
        memory = MockMemoryStorageFailure()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Process multiple messages - should all succeed
        for i in range(3):
            result = engine.process_message(f"Remember task {i}")
            assert result["intent"] == "store_memory"
            assert "error" in result["metadata"]


class TestOperationWithoutMemoryConfigured:
    """Test operation without memory configured (Requirement 11.1, 11.2, 11.4)."""
    
    def test_store_memory_intent_without_memory(self):
        """Test store_memory intent when memory is None."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        result = engine.process_message("Remember to buy milk")
        
        # Verify intent is detected
        assert result["intent"] == "store_memory"
        
        # Verify informative message
        assert "not available" in result["response"].lower()
        
        # Verify error metadata
        assert "error" in result["metadata"]
        assert result["metadata"]["error"] == "no_memory_configured"
    
    def test_retrieve_memory_intent_without_memory(self):
        """Test retrieve_memory intent when memory is None."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        result = engine.process_message("What was my last task?")
        
        # Verify intent is detected
        assert result["intent"] == "retrieve_memory"
        
        # Verify informative message
        assert "not available" in result["response"].lower()
        
        # Verify error metadata
        assert "error" in result["metadata"]
        assert result["metadata"]["error"] == "no_memory_configured"
    
    def test_non_memory_intents_work_without_memory(self):
        """Test that non-memory intents work normally without memory."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        # Education intent
        result = engine.process_message("Teach me Python")
        assert result["intent"] == "education"
        assert "StubLLM Response" in result["response"]
        
        # Scheduling intent
        result = engine.process_message("Schedule a meeting")
        assert result["intent"] == "scheduling"
        assert "StubLLM Response" in result["response"]
        
        # General intent
        result = engine.process_message("Hello")
        assert result["intent"] == "general"
        assert "StubLLM Response" in result["response"]
    
    def test_system_processes_all_messages_without_memory(self):
        """Test that system processes all message types without memory."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        messages = [
            "Teach me Python",
            "Remember this",
            "What was that?",
            "Schedule a meeting",
            "Hello there"
        ]
        
        for message in messages:
            result = engine.process_message(message)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
