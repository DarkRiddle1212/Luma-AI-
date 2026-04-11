"""
Test ReasoningEngine memory integration.

Tests the memory integration functionality added to ReasoningEngine,
including constructor injection, memory storage, and memory retrieval.
"""

import pytest
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult
)
from typing import Dict, List, Optional, Any


class MockMemory(MemoryInterface):
    """Mock memory implementation for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.next_id = 1
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content and return ID."""
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": "2024-01-15T10:00:00",
            "category": metadata.get("category", "general") if metadata else "general",
            "tags": metadata.get("tags", []) if metadata else []
        })
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
        
        # Handle both legacy and enhanced API
        if params:
            query = params.get("query")
            limit = params.get("limit", limit)
        
        # Simple substring matching for testing
        results = []
        for mem in self.stored_memories:
            if query and query.lower() in mem["content"].lower():
                results.append(mem)
                if len(results) >= limit:
                    break
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Return RetrievalResult
        return {
            "memories": results,
            "total_count": len(results),
            "query_metadata": {
                "execution_time_ms": execution_time_ms,
                "filters_applied": {"query": query} if query else {},
                "limit": limit,
                "has_more": False
            }
        }


class TestReasoningEngineMemoryIntegration:
    """Test memory integration in ReasoningEngine."""
    
    def test_constructor_accepts_memory_parameter(self):
        """Test that constructor accepts optional memory parameter."""
        llm = StubLLM()
        memory = MockMemory()
        
        # Should work with memory
        engine = ReasoningEngine(llm=llm, memory=memory)
        assert engine.llm is llm
        assert engine.memory is memory
        
        # Should work without memory
        engine2 = ReasoningEngine(llm=llm)
        assert engine2.llm is llm
        assert engine2.memory is None
    
    def test_build_context_accepts_retrieved_memories(self):
        """Test that build_context accepts retrieved_memories parameter."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm)
        
        # Without memories
        context = engine.build_context("Test message")
        assert "memories" in context
        assert context["memories"] == []
        
        # With memories
        memories = [
            {"id": "1", "content": "Test memory", "metadata": {}, "timestamp": "2024-01-15T10:00:00"}
        ]
        context = engine.build_context("Test message", retrieved_memories=memories)
        assert context["memories"] == memories
        assert len(context["memories"]) == 1
    
    def test_store_memory_intent_with_memory_configured(self):
        """Test store_memory intent when memory is configured."""
        llm = StubLLM()
        memory = MockMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Remember to buy milk")
        
        assert result["intent"] == "store_memory"
        assert "stored that information" in result["response"].lower()
        assert "memory_id" in result["metadata"]
        assert len(memory.stored_memories) == 1
        assert "to buy milk" in memory.stored_memories[0]["content"]
    
    def test_store_memory_intent_without_memory_configured(self):
        """Test store_memory intent when memory is not configured."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm)  # No memory
        
        result = engine.process_message("Remember to buy milk")
        
        assert result["intent"] == "store_memory"
        assert "not available" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_retrieve_memory_intent_with_memory_configured(self):
        """Test retrieve_memory intent when memory is configured."""
        llm = StubLLM()
        memory = MockMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Store some memories first
        memory.store("python is a programming language", metadata={})
        memory.store("javascript is also a language", metadata={})
        
        result = engine.process_message("Recall python")
        
        assert result["intent"] == "retrieve_memory"
        assert "memories_found" in result["metadata"]
        assert result["metadata"]["memories_found"] == 1
        assert "memory_ids" in result["metadata"]
    
    def test_retrieve_memory_intent_without_memory_configured(self):
        """Test retrieve_memory intent when memory is not configured."""
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm)  # No memory
        
        result = engine.process_message("What was my last task?")
        
        assert result["intent"] == "retrieve_memory"
        assert "not available" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_retrieve_memory_with_no_results(self):
        """Test retrieve_memory when no memories match."""
        llm = StubLLM()
        memory = MockMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("What was about Python?")
        
        assert result["intent"] == "retrieve_memory"
        assert "don't have any memories" in result["response"].lower()
        assert result["metadata"]["memories_found"] == 0
    
    def test_non_memory_intents_unchanged(self):
        """Test that non-memory intents work as before."""
        llm = StubLLM()
        memory = MockMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        # Education intent
        result = engine.process_message("Teach me Python")
        assert result["intent"] == "education"
        assert "StubLLM Response" in result["response"]
        
        # Scheduling intent
        result = engine.process_message("Schedule a meeting")
        assert result["intent"] == "scheduling"
        
        # General intent
        result = engine.process_message("Hello")
        assert result["intent"] == "general"
    
    def test_memory_storage_error_handling(self):
        """Test error handling when memory storage fails."""
        
        class FailingMemory(MemoryInterface):
            def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
                raise Exception("Storage failed")
            
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
        
        llm = StubLLM()
        memory = FailingMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("Remember this")
        
        assert result["intent"] == "store_memory"
        assert "couldn't store" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_memory_retrieval_error_handling(self):
        """Test error handling when memory retrieval fails."""
        
        class FailingMemory(MemoryInterface):
            def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
                return "mem_1"
            
            def retrieve(
                self,
                query: Optional[str] = None,
                params: Optional[QueryParameters] = None,
                limit: int = 10
            ) -> RetrievalResult:
                raise Exception("Retrieval failed")
        
        llm = StubLLM()
        memory = FailingMemory()
        engine = ReasoningEngine(llm=llm, memory=memory)
        
        result = engine.process_message("What was my task?")
        
        assert result["intent"] == "retrieve_memory"
        # Should fallback to processing without memories
        assert "fallback" in result["metadata"]
        assert result["metadata"]["fallback"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
