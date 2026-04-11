"""
Test memory failure paths and error isolation.

These tests validate that the ReasoningEngine handles memory operation failures
gracefully without crashing, and that errors are properly isolated and reported
in response metadata.
"""

import pytest
from unittest.mock import Mock, MagicMock
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError
)
from typing import Dict, List, Optional, Any


class TestMemoryStorageFailurePaths:
    """Test graceful handling of memory storage failures."""
    
    def test_storage_failure_returns_graceful_response(self):
        """
        Test that when store() fails, ReasoningEngine returns a graceful
        response with error metadata instead of crashing.
        
        Validates: Memory errors don't crash reasoning engine (Requirement 4.4)
        """
        # Create a mock memory that raises MemoryStorageError
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.store.side_effect = MemoryStorageError("Database connection failed")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # Process a store_memory intent message
        result = engine.process_message("Remember to buy groceries")
        
        # Verify graceful response
        assert result is not None, "Engine should return a response, not crash"
        assert result["intent"] == "store_memory"
        assert "response" in result
        assert "couldn't store" in result["response"].lower() or "error" in result["response"].lower()
        
        # Verify error metadata is present
        assert "metadata" in result
        assert "error" in result["metadata"]
        assert "Database connection failed" in str(result["metadata"]["error"])
        
        # Verify store was attempted
        mock_memory.store.assert_called_once()
    
    def test_storage_generic_exception_handled(self):
        """
        Test that generic exceptions during storage are caught and handled.
        
        Validates: All exception types are handled, not just MemoryStorageError
        """
        # Create a mock memory that raises a generic exception
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.store.side_effect = RuntimeError("Unexpected error")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        result = engine.process_message("Store this information")
        
        # Should still return graceful response
        assert result is not None
        assert result["intent"] == "store_memory"
        assert "error" in result["metadata"]
        assert "Unexpected error" in str(result["metadata"]["error"])
    
    def test_storage_failure_with_empty_content(self):
        """
        Test storage failure when content extraction results in empty string.
        
        Validates: Edge case handling in error paths
        """
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.store.side_effect = MemoryStorageError("Empty content not allowed")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # Message with only trigger words
        result = engine.process_message("Remember")
        
        assert result is not None
        assert result["intent"] == "store_memory"
        assert "error" in result["metadata"]
    
    def test_storage_failure_preserves_engine_state(self):
        """
        Test that storage failure doesn't corrupt engine state for subsequent calls.
        
        Validates: Engine remains functional after errors
        """
        mock_memory = Mock(spec=MemoryInterface)
        # First call fails, second succeeds
        mock_memory.store.side_effect = [
            MemoryStorageError("Temporary failure"),
            "mem_123"
        ]
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # First call should fail gracefully
        result1 = engine.process_message("Remember task 1")
        assert "error" in result1["metadata"]
        
        # Second call should succeed
        result2 = engine.process_message("Remember task 2")
        assert result2["intent"] == "store_memory"
        # If it succeeded, it should have memory_id
        if "error" not in result2["metadata"]:
            assert "memory_id" in result2["metadata"]


class TestMemoryRetrievalFailurePaths:
    """Test graceful handling of memory retrieval failures."""
    
    def test_retrieval_failure_falls_back_to_llm(self):
        """
        Test that when retrieve() fails, ReasoningEngine falls back to
        normal LLM response without crashing.
        
        Validates: Retrieval errors trigger fallback behavior (Requirement 5.5)
        """
        # Create a mock memory that raises MemoryRetrievalError
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.retrieve.side_effect = MemoryRetrievalError("Query timeout")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # Process a retrieve_memory intent message
        result = engine.process_message("What was my last task?")
        
        # Verify graceful fallback
        assert result is not None, "Engine should return a response, not crash"
        assert result["intent"] == "retrieve_memory"
        assert "response" in result
        
        # Should contain LLM-generated response (fallback behavior)
        assert "StubLLM Response" in result["response"]
        
        # Verify error metadata and fallback flag
        assert "metadata" in result
        assert "error" in result["metadata"]
        assert "fallback" in result["metadata"]
        assert result["metadata"]["fallback"] is True
        
        # Verify retrieve was attempted
        mock_memory.retrieve.assert_called_once()
    
    def test_retrieval_generic_exception_handled(self):
        """
        Test that generic exceptions during retrieval are caught and handled.
        
        Validates: All exception types trigger fallback, not just MemoryRetrievalError
        """
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.retrieve.side_effect = ConnectionError("Network unavailable")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        result = engine.process_message("Recall my notes")
        
        # Should fallback to LLM
        assert result is not None
        assert result["intent"] == "retrieve_memory"
        assert "StubLLM Response" in result["response"]
        assert result["metadata"]["fallback"] is True
        assert "Network unavailable" in str(result["metadata"]["error"])
    
    def test_retrieval_failure_with_complex_query(self):
        """
        Test retrieval failure with complex query strings.
        
        Validates: Error handling works with various query formats
        """
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.retrieve.side_effect = MemoryRetrievalError("Invalid query syntax")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # Complex query with special characters
        result = engine.process_message("What was about Python & JavaScript?")
        
        assert result is not None
        assert result["intent"] == "retrieve_memory"
        assert result["metadata"]["fallback"] is True
    
    def test_retrieval_failure_preserves_engine_state(self):
        """
        Test that retrieval failure doesn't corrupt engine state for subsequent calls.
        
        Validates: Engine remains functional after retrieval errors
        """
        mock_memory = Mock(spec=MemoryInterface)
        # First call fails, second succeeds
        mock_memory.retrieve.side_effect = [
            MemoryRetrievalError("Temporary failure"),
            [{"id": "mem_1", "content": "Test", "metadata": {}, "timestamp": "2024-01-15T10:00:00"}]
        ]
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # First call should fail and fallback
        result1 = engine.process_message("What was my task?")
        assert result1["metadata"]["fallback"] is True
        
        # Second call should succeed
        result2 = engine.process_message("Recall my notes")
        assert result2["intent"] == "retrieve_memory"
        # If it succeeded, it should have memories_found
        if "fallback" not in result2["metadata"]:
            assert "memories_found" in result2["metadata"]


class TestMemoryAdapterLifecycle:
    """Test adapter lifecycle management."""
    
    def test_adapter_close_method_exists(self):
        """Test that SQLiteMemoryAdapter has a close() method."""
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        from luma_memory.memory_manager import MemoryManager
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        import tempfile
        import os
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name
        
        try:
            storage = SQLiteStorage(db_path)
            manager = MemoryManager(storage=storage)
            adapter = SQLiteMemoryAdapter(manager)
            
            # Verify close method exists
            assert hasattr(adapter, 'close')
            assert callable(adapter.close)
            
            # Call close
            adapter.close()
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except:
                    pass
    
    def test_adapter_close_is_idempotent(self):
        """Test that calling close() multiple times is safe."""
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        from luma_memory.memory_manager import MemoryManager
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name
        
        try:
            storage = SQLiteStorage(db_path)
            manager = MemoryManager(storage=storage)
            adapter = SQLiteMemoryAdapter(manager)
            
            # Call close multiple times - should not raise errors
            adapter.close()
            adapter.close()
            adapter.close()
            
        finally:
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except:
                    pass
    
    def test_adapter_close_with_no_storage(self):
        """Test that close() handles cases where storage doesn't have close method."""
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        
        # Create a mock manager without storage or with storage without close
        mock_manager = Mock()
        mock_manager.storage = None
        
        adapter = SQLiteMemoryAdapter(mock_manager)
        
        # Should not raise error
        adapter.close()


class TestCombinedFailureScenarios:
    """Test combined failure scenarios and edge cases."""
    
    def test_both_store_and_retrieve_failures(self):
        """
        Test that engine handles both storage and retrieval failures gracefully.
        
        Validates: Multiple failure types don't compound into crashes
        """
        mock_memory = Mock(spec=MemoryInterface)
        mock_memory.store.side_effect = MemoryStorageError("Storage failed")
        mock_memory.retrieve.side_effect = MemoryRetrievalError("Retrieval failed")
        
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=mock_memory)
        
        # Test storage failure
        result1 = engine.process_message("Remember this")
        assert result1 is not None
        assert "error" in result1["metadata"]
        
        # Test retrieval failure
        result2 = engine.process_message("What was that?")
        assert result2 is not None
        assert result2["metadata"]["fallback"] is True
        
        # Test non-memory intent still works
        result3 = engine.process_message("Teach me Python")
        assert result3 is not None
        assert result3["intent"] == "education"
    
    def test_memory_none_with_memory_intents(self):
        """
        Test that memory intents work gracefully when memory is None.
        
        Validates: Missing memory dependency is handled gracefully
        """
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=None)
        
        # Store intent
        result1 = engine.process_message("Remember to call John")
        assert result1 is not None
        assert result1["intent"] == "store_memory"
        assert "not available" in result1["response"].lower()
        
        # Retrieve intent
        result2 = engine.process_message("What was my task?")
        assert result2 is not None
        assert result2["intent"] == "retrieve_memory"
        assert "not available" in result2["response"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
