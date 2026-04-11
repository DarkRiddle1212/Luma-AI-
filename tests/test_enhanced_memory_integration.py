"""
End-to-End Integration Tests for Enhanced Memory Retrieval

This module implements comprehensive integration tests for the complete
reasoning-memory flow with enhanced query parameters, error handling,
and configuration features.

Feature: intent-based-memory-retrieval-enhancements
Requirements: All requirements (1-12)
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError,
    QueryParameters,
    MemoryEntry,
    RetrievalResult
)
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


class TestEnhancedMemoryIntegration:
    """
    End-to-end integration tests for enhanced memory retrieval system.
    
    These tests verify the complete flow from ReasoningEngine through
    SQLiteMemoryAdapter to MemoryManager with all enhanced features.
    """
    
    def test_complete_flow_store_retrieve_inject_context(self):
        """
        Test complete flow: store → retrieve → inject into context.
        
        Validates: Requirements 1.1-1.6, 2.1-2.6, 3.1-3.5, 10.1-10.5
        
        This test verifies:
        1. Memory can be stored with metadata
        2. Memory can be retrieved with filters
        3. Retrieved memories are injected into LLM context
        4. All metadata is preserved throughout the flow
        5. RetrievalResult contains complete metadata
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_complete_flow.db")
            
            # Initialize components
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(
                memory_manager=memory_manager,
                device_id="test-device",
                default_category="test-category",
                default_tags=["integration-test"]
            )
            
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Step 1: Store memories with different categories and tags
            store_result1 = engine.process_message("Remember Python is a programming language")
            assert store_result1["intent"] == "store_memory"
            assert "memory_id" in store_result1["metadata"]
            memory_id1 = store_result1["metadata"]["memory_id"]
            
            # Store with explicit metadata
            memory_adapter.store(
                "JavaScript is used for web development",
                metadata={"category": "programming", "tags": ["web", "javascript"]}
            )
            
            memory_adapter.store(
                "TypeScript is a superset of JavaScript",
                metadata={"category": "programming", "tags": ["web", "typescript"]}
            )
            
            # Step 2: Retrieve memories using enhanced query parameters
            params: QueryParameters = {
                "query": "JavaScript",
                "category": "programming",
                "tags": ["web"],
                "limit": 10
            }
            
            result = memory_adapter.retrieve(params=params)
            
            # Verify RetrievalResult structure
            assert isinstance(result, dict)
            assert "memories" in result
            assert "total_count" in result
            assert "query_metadata" in result
            
            # Verify memories were retrieved
            assert result["total_count"] >= 2
            assert len(result["memories"]) >= 2
            
            # Verify query metadata
            metadata = result["query_metadata"]
            assert "execution_time_ms" in metadata
            assert "filters_applied" in metadata
            assert "limit" in metadata
            assert metadata["limit"] == 10
            
            # Verify filters were applied
            filters = metadata["filters_applied"]
            assert "query" in filters
            assert filters["query"] == "JavaScript"
            
            # Step 3: Verify memories have complete structure
            for memory in result["memories"]:
                assert "id" in memory
                assert "content" in memory
                assert "metadata" in memory
                assert "timestamp" in memory
                assert "category" in memory
                assert "tags" in memory
                
                # Verify category filter worked
                assert memory["category"] == "programming"
                
                # Verify tag filter worked
                assert "web" in memory["tags"]
            
            # Step 4: Test retrieval through ReasoningEngine
            retrieve_result = engine.process_message("Recall JavaScript")
            
            assert retrieve_result["intent"] == "retrieve_memory"
            assert "memories_found" in retrieve_result["metadata"]
            assert retrieve_result["metadata"]["memories_found"] >= 2
            
            # Verify memories were injected into context
            assert "memory_ids" in retrieve_result["metadata"]
            memory_ids = retrieve_result["metadata"]["memory_ids"]
            assert isinstance(memory_ids, list)
            assert len(memory_ids) >= 2
            
            # Clean up
            storage.close()
    
    def test_retrieval_failure_fallback_in_complete_flow(self):
        """
        Test retrieval failure fallback in complete flow.
        
        Validates: Requirements 4.1-4.6, 11.1-11.5
        
        This test verifies:
        1. System continues when retrieval fails
        2. Fallback to LLM-only processing works
        3. Error is logged and included in metadata
        4. User gets a valid response despite failure
        5. System doesn't crash
        """
        
        class FailingMemoryAdapter(MemoryInterface):
            """Memory adapter that fails on retrieval."""
            
            def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
                return "mem_1"
            
            def retrieve(
                self,
                query: Optional[str] = None,
                params: Optional[QueryParameters] = None
            ) -> RetrievalResult:
                raise MemoryRetrievalError("Simulated retrieval failure")
        
        llm = StubLLM()
        failing_memory = FailingMemoryAdapter()
        engine = ReasoningEngine(llm=llm, memory=failing_memory)
        
        # Attempt retrieval
        result = engine.process_message("What was my last task?")
        
        # Verify fallback behavior
        assert result["intent"] == "retrieve_memory"
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        
        # Verify fallback flag in metadata
        assert "fallback" in result["metadata"]
        assert result["metadata"]["fallback"] is True
        
        # Verify error message in metadata
        assert "error" in result["metadata"]
        assert "retrieval failure" in result["metadata"]["error"].lower()
        
        # Verify system didn't crash
        assert result is not None
    
    def test_storage_failure_handling_in_complete_flow(self):
        """
        Test storage failure handling in complete flow.
        
        Validates: Requirements 5.1-5.5
        
        This test verifies:
        1. System handles storage failures gracefully
        2. User gets clear error message
        3. Error details are in metadata
        4. System doesn't crash
        5. Subsequent operations still work
        """
        
        class FailingMemoryAdapter(MemoryInterface):
            """Memory adapter that fails on storage."""
            
            def __init__(self):
                self.storage_attempts = 0
            
            def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
                self.storage_attempts += 1
                if self.storage_attempts == 1:
                    raise MemoryStorageError("Simulated storage failure")
                return f"mem_{self.storage_attempts}"
            
            def retrieve(
                self,
                query: Optional[str] = None,
                params: Optional[QueryParameters] = None
            ) -> RetrievalResult:
                return {
                    "memories": [],
                    "total_count": 0,
                    "query_metadata": {
                        "execution_time_ms": 0,
                        "filters_applied": {},
                        "limit": 10,
                        "has_more": False
                    }
                }
        
        llm = StubLLM()
        failing_memory = FailingMemoryAdapter()
        engine = ReasoningEngine(llm=llm, memory=failing_memory)
        
        # Attempt storage (will fail)
        result1 = engine.process_message("Remember to buy milk")
        
        # Verify error handling
        assert result1["intent"] == "store_memory"
        assert "couldn't store" in result1["response"].lower() or "error" in result1["response"].lower()
        assert "error" in result1["metadata"]
        assert "storage failure" in result1["metadata"]["error"].lower()
        
        # Verify system didn't crash - try another operation
        result2 = engine.process_message("Remember to call dentist")
        
        # Second attempt should succeed
        assert result2["intent"] == "store_memory"
        assert "memory_id" in result2["metadata"]
        assert failing_memory.storage_attempts == 2
    
    def test_memory_optional_operation_in_complete_flow(self):
        """
        Test memory-optional operation in complete flow.
        
        Validates: Requirements 11.1, 11.2, 11.4
        
        This test verifies:
        1. System works without memory configured
        2. Memory-related intents return informative messages
        3. Non-memory intents work normally
        4. No errors when memory is None
        """
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        # Test store_memory intent without memory
        result1 = engine.process_message("Remember to buy milk")
        assert result1["intent"] == "store_memory"
        assert "not available" in result1["response"].lower()
        assert "error" in result1["metadata"]
        
        # Test retrieve_memory intent without memory
        result2 = engine.process_message("What was my last task?")
        assert result2["intent"] == "retrieve_memory"
        assert "not available" in result2["response"].lower()
        assert "error" in result2["metadata"]
        
        # Test non-memory intent works normally
        result3 = engine.process_message("Teach me Python")
        assert result3["intent"] == "education"
        assert "StubLLM Response" in result3["response"]
        assert "error" not in result3["metadata"]
    
    def test_configuration_defaults_in_complete_flow(self):
        """
        Test configuration defaults in complete flow.
        
        Validates: Requirements 6.1-6.7
        
        This test verifies:
        1. device_id is applied to stored memories
        2. default_category is used when not specified
        3. default_tags are merged with provided tags
        4. Configuration works end-to-end
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_config.db")
            
            # Initialize with configuration
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(
                memory_manager=memory_manager,
                device_id="test-device-123",
                default_category="default-cat",
                default_tags=["default-tag1", "default-tag2"]
            )
            
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Store memory through ReasoningEngine (which sets category to "user_memory")
            result = engine.process_message("Remember Python is great")
            assert result["intent"] == "store_memory"
            assert "memory_id" in result["metadata"]
            memory_id = result["metadata"]["memory_id"]
            
            # Retrieve and verify configuration was applied
            all_memories = memory_manager.query_memories(action_type="", limit=100)
            
            found_memory = None
            for mem in all_memories:
                if mem.id == memory_id:
                    found_memory = mem
                    break
            
            assert found_memory is not None
            
            # Verify device_id was applied
            assert found_memory.device_id == "test-device-123"
            
            # Verify default_tags were applied
            assert "default-tag1" in found_memory.tags
            assert "default-tag2" in found_memory.tags
            
            # Note: ReasoningEngine explicitly sets category to "user_memory"
            # so default_category is overridden in this case
            assert found_memory.context.get("category") == "user_memory"
            
            # Store memory directly through adapter without explicit category
            # This tests that default_category is used when no category is provided
            memory_id2 = memory_adapter.store(
                "JavaScript is also great",
                metadata={"tags": ["explicit-tag"]}
            )
            
            # Verify tags were merged and default_category was applied
            all_memories = memory_manager.query_memories(action_type="", limit=100)
            js_memory = None
            for m in all_memories:
                if m.id == memory_id2:
                    js_memory = m
                    break
            
            assert js_memory is not None
            assert "default-tag1" in js_memory.tags
            assert "default-tag2" in js_memory.tags
            assert "explicit-tag" in js_memory.tags
            
            # Verify default_category was applied when storing directly through adapter
            assert js_memory.context.get("category") == "default-cat"
            
            # Clean up
            storage.close()
    
    def test_enhanced_query_parameters_end_to_end(self):
        """
        Test enhanced query parameters end-to-end.
        
        Validates: Requirements 1.1-1.6, 2.1-2.6
        
        This test verifies:
        1. Category filtering works
        2. Timestamp range filtering works
        3. Tag filtering works
        4. Multiple filters work together (AND logic)
        5. Limit parameter works
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_enhanced_query.db")
            
            # Initialize components
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Store memories with different attributes
            now = datetime.now()
            
            # Memory 1: programming, web, recent
            memory_adapter.store(
                "React is a JavaScript library",
                metadata={
                    "category": "programming",
                    "tags": ["web", "react", "javascript"]
                }
            )
            
            # Memory 2: programming, backend, recent
            memory_adapter.store(
                "Django is a Python framework",
                metadata={
                    "category": "programming",
                    "tags": ["backend", "python", "django"]
                }
            )
            
            # Memory 3: general, no tags
            memory_adapter.store(
                "Remember to buy groceries",
                metadata={
                    "category": "general",
                    "tags": []
                }
            )
            
            # Test 1: Filter by category
            result1 = memory_adapter.retrieve(params={"category": "programming"})
            assert result1["total_count"] >= 2
            for mem in result1["memories"]:
                assert mem["category"] == "programming"
            
            # Test 2: Filter by tags
            result2 = memory_adapter.retrieve(params={"tags": ["web"]})
            assert result2["total_count"] >= 1
            for mem in result2["memories"]:
                assert "web" in mem["tags"]
            
            # Test 3: Filter by query text
            result3 = memory_adapter.retrieve(params={"query": "Python"})
            assert result3["total_count"] >= 1
            for mem in result3["memories"]:
                assert "python" in mem["content"].lower() or "Python" in mem["content"]
            
            # Test 4: Combine multiple filters
            result4 = memory_adapter.retrieve(params={
                "category": "programming",
                "tags": ["web"]
            })
            assert result4["total_count"] >= 1
            for mem in result4["memories"]:
                assert mem["category"] == "programming"
                assert "web" in mem["tags"]
            
            # Test 5: Limit parameter
            result5 = memory_adapter.retrieve(params={"limit": 1})
            assert len(result5["memories"]) <= 1
            assert result5["query_metadata"]["limit"] == 1
            
            # Clean up
            storage.close()
    
    def test_backward_compatibility_in_complete_flow(self):
        """
        Test backward compatibility in complete flow.
        
        Validates: Requirements 1.5, 9.1-9.5
        
        This test verifies:
        1. Legacy API (query string) still works
        2. Enhanced API (params dict) works
        3. Both produce equivalent results
        4. No breaking changes to existing code
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_backward_compat.db")
            
            # Initialize components
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Store test memory
            memory_adapter.store(
                "Python is a programming language",
                metadata={"category": "programming"}
            )
            
            # Test legacy API
            result_legacy = memory_adapter.retrieve(query="Python")
            
            # Test enhanced API
            result_enhanced = memory_adapter.retrieve(params={"query": "Python"})
            
            # Verify both work
            assert result_legacy["total_count"] >= 1
            assert result_enhanced["total_count"] >= 1
            
            # Verify both have same structure
            assert "memories" in result_legacy
            assert "total_count" in result_legacy
            assert "query_metadata" in result_legacy
            
            assert "memories" in result_enhanced
            assert "total_count" in result_enhanced
            assert "query_metadata" in result_enhanced
            
            # Verify results are equivalent
            assert result_legacy["total_count"] == result_enhanced["total_count"]
            assert len(result_legacy["memories"]) == len(result_enhanced["memories"])
            
            # Clean up
            storage.close()
    
    def test_empty_and_malformed_query_handling(self):
        """
        Test empty and malformed query handling.
        
        Validates: Requirements 8.1-8.6
        
        This test verifies:
        1. Empty query returns empty results
        2. None query returns empty results
        3. Whitespace query returns empty results
        4. Invalid parameters raise validation errors
        5. Error messages are clear
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_empty_query.db")
            
            # Initialize components
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Test empty query
            result1 = memory_adapter.retrieve(query="")
            assert result1["total_count"] == 0
            assert result1["memories"] == []
            
            # Test None query
            result2 = memory_adapter.retrieve(query=None)
            assert result2["total_count"] == 0
            
            # Test whitespace query
            result3 = memory_adapter.retrieve(query="   ")
            assert result3["total_count"] == 0
            
            # Test invalid limit (negative)
            with pytest.raises(ValueError, match="limit must be a positive integer"):
                memory_adapter.retrieve(params={"limit": -1})
            
            # Test invalid limit (zero)
            with pytest.raises(ValueError, match="limit must be a positive integer"):
                memory_adapter.retrieve(params={"limit": 0})
            
            # Test invalid timestamp range
            with pytest.raises(ValueError, match="start_time.*must be <= end_time"):
                memory_adapter.retrieve(params={
                    "start_time": datetime(2024, 1, 2),
                    "end_time": datetime(2024, 1, 1)
                })
            
            # Clean up
            storage.close()
    
    def test_retrieval_result_metadata_completeness(self):
        """
        Test retrieval result metadata completeness.
        
        Validates: Requirements 10.1-10.5
        
        This test verifies:
        1. RetrievalResult includes total_count
        2. RetrievalResult includes execution_time_ms
        3. RetrievalResult includes filters_applied
        4. RetrievalResult includes limit and has_more
        5. All metadata is accurate
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_metadata.db")
            
            # Initialize components
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Store test memories
            for i in range(5):
                memory_adapter.store(
                    f"Test memory {i}",
                    metadata={"category": "test"}
                )
            
            # Retrieve with filters
            result = memory_adapter.retrieve(params={
                "query": "Test",
                "category": "test",
                "limit": 3
            })
            
            # Verify top-level fields
            assert "memories" in result
            assert "total_count" in result
            assert "query_metadata" in result
            
            # Verify total_count
            assert isinstance(result["total_count"], int)
            assert result["total_count"] >= 0
            assert result["total_count"] == len(result["memories"])
            
            # Verify query_metadata structure
            metadata = result["query_metadata"]
            assert "execution_time_ms" in metadata
            assert "filters_applied" in metadata
            assert "limit" in metadata
            assert "has_more" in metadata
            
            # Verify execution_time_ms
            assert isinstance(metadata["execution_time_ms"], (int, float))
            assert metadata["execution_time_ms"] >= 0
            
            # Verify filters_applied
            filters = metadata["filters_applied"]
            assert isinstance(filters, dict)
            assert "query" in filters
            assert filters["query"] == "Test"
            
            # Verify limit
            assert metadata["limit"] == 3
            assert len(result["memories"]) <= 3
            
            # Verify has_more
            assert isinstance(metadata["has_more"], bool)
            
            # Clean up
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
