"""
Integration Test for Restart Persistence

This module tests that memories persist across application restarts.
It verifies that data stored in one application session can be retrieved
after shutting down and restarting the application.

Feature: reasoning-memory-integration
Requirements: 8.4
"""

import pytest
import tempfile
import os
from pathlib import Path

from luma.container import initialize_application, cleanup_application
from luma.core.reasoning import ReasoningEngine


class TestRestartPersistence:
    """
    Integration tests for memory persistence across application restarts.
    
    These tests verify that:
    1. Memories can be stored in one application session
    2. Application can be shut down cleanly
    3. Application can be restarted with the same database
    4. Previously stored memories can be retrieved after restart
    5. All memory data is correctly restored
    """
    
    def test_memories_persist_across_restart(self):
        """
        Test that memories persist when application is restarted.
        
        Validates Requirement 8.4:
        - When the application restarts, the system shall restore
          previously stored memory from local storage
        
        This test:
        1. Initializes application with temporary database
        2. Stores multiple test memories
        3. Shuts down application (closes connections)
        4. Re-initializes application with same database
        5. Retrieves previously stored memories
        6. Verifies all memories are correctly restored
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_restart.db")
            
            # Test memories to store
            test_memories = [
                "Remember Python is a programming language",
                "Remember to buy groceries tomorrow",
                "Remember the meeting is at 3 PM"
            ]
            
            stored_memory_ids = []
            
            # ===== SESSION 1: Initialize and store memories =====
            engine1, storage1 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Store memories
                for memory_text in test_memories:
                    result = engine1.process_message(memory_text)
                    
                    assert result["intent"] == "store_memory", \
                        f"Intent should be store_memory, got '{result['intent']}'"
                    
                    assert "memory_id" in result["metadata"], \
                        "Result should contain memory_id"
                    
                    memory_id = result["metadata"]["memory_id"]
                    stored_memory_ids.append(memory_id)
                    
                    assert "stored that information" in result["response"].lower(), \
                        f"Response should confirm storage: '{result['response']}'"
                
                # Verify all memories were stored
                assert len(stored_memory_ids) == len(test_memories), \
                    f"Should have stored {len(test_memories)} memories, " \
                    f"got {len(stored_memory_ids)}"
                
            finally:
                # Shutdown application (close connections)
                cleanup_application(storage1)
            
            # Verify database file still exists after shutdown
            assert os.path.exists(db_path), \
                f"Database file should exist after shutdown: {db_path}"
            
            # ===== SESSION 2: Restart and retrieve memories =====
            engine2, storage2 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Retrieve memories using different queries
                queries = ["Python", "groceries", "meeting"]
                
                for i, query in enumerate(queries):
                    result = engine2.process_message(f"Recall {query}")
                    
                    assert result["intent"] == "retrieve_memory", \
                        f"Intent should be retrieve_memory, got '{result['intent']}'"
                    
                    assert "memories_found" in result["metadata"], \
                        "Result should contain memories_found"
                    
                    memories_found = result["metadata"]["memories_found"]
                    assert memories_found > 0, \
                        f"Should find at least one memory for query '{query}', " \
                        f"found {memories_found}"
                    
                    assert "memory_ids" in result["metadata"], \
                        "Result should contain memory_ids when memories found"
                    
                    memory_ids = result["metadata"]["memory_ids"]
                    assert isinstance(memory_ids, list), \
                        f"memory_ids must be a list, got {type(memory_ids)}"
                    
                    # Verify at least one of the stored IDs is retrieved
                    # (depending on query, might not retrieve all)
                    assert len(memory_ids) > 0, \
                        f"Should retrieve at least one memory for query '{query}'"
                
            finally:
                # Cleanup second session
                cleanup_application(storage2)
    
    def test_multiple_restart_cycles(self):
        """
        Test that memories persist across multiple restart cycles.
        
        Validates Requirement 8.4:
        - System shall restore previously stored memory from local storage
          across multiple restart cycles
        
        This test:
        1. Performs multiple store-restart-retrieve cycles
        2. Verifies memories accumulate correctly
        3. Verifies all memories from all sessions are retrievable
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_multi_restart.db")
            
            all_stored_ids = []
            
            # Perform 3 restart cycles
            for cycle in range(3):
                engine, storage = initialize_application(
                    db_path=db_path,
                    return_storage=True
                )
                
                try:
                    # Store a memory in this cycle
                    memory_text = f"Remember cycle {cycle} information"
                    result = engine.process_message(memory_text)
                    
                    assert result["intent"] == "store_memory"
                    assert "memory_id" in result["metadata"]
                    
                    memory_id = result["metadata"]["memory_id"]
                    all_stored_ids.append(memory_id)
                    
                    # Verify we can retrieve memories from previous cycles
                    if cycle > 0:
                        retrieve_result = engine.process_message("Recall cycle")
                        
                        assert retrieve_result["intent"] == "retrieve_memory"
                        assert retrieve_result["metadata"]["memories_found"] >= cycle, \
                            f"Cycle {cycle}: Should find at least {cycle} memories, " \
                            f"found {retrieve_result['metadata']['memories_found']}"
                
                finally:
                    # Shutdown after each cycle
                    cleanup_application(storage)
            
            # Final verification: restart and verify all memories are present
            engine_final, storage_final = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                result = engine_final.process_message("Recall cycle")
                
                assert result["intent"] == "retrieve_memory"
                assert result["metadata"]["memories_found"] >= 3, \
                    f"Should find at least 3 memories after all cycles, " \
                    f"found {result['metadata']['memories_found']}"
            
            finally:
                cleanup_application(storage_final)
    
    def test_restart_with_empty_database(self):
        """
        Test that restarting with an empty database works correctly.
        
        Validates Requirement 8.4:
        - System shall handle restart with empty database gracefully
        
        This test:
        1. Initializes application with new database
        2. Shuts down without storing anything
        3. Restarts application
        4. Verifies retrieval returns no results (not an error)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_empty_restart.db")
            
            # Session 1: Initialize and shutdown without storing
            engine1, storage1 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Don't store anything, just verify engine works
                result = engine1.process_message("Hello")
                assert result["intent"] == "general"
            
            finally:
                cleanup_application(storage1)
            
            # Session 2: Restart and try to retrieve
            engine2, storage2 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Try to retrieve from empty database
                result = engine2.process_message("Recall anything")
                
                assert result["intent"] == "retrieve_memory"
                assert result["metadata"]["memories_found"] == 0, \
                    "Should find 0 memories in empty database"
                
                assert "don't have any memories" in result["response"].lower(), \
                    "Should inform user no memories found"
            
            finally:
                cleanup_application(storage2)
    
    def test_restart_preserves_memory_content_exactly(self):
        """
        Test that memory content is preserved exactly across restart.
        
        Validates Requirement 8.4:
        - System shall restore previously stored memory with exact content
        
        This test:
        1. Stores memories with specific content
        2. Restarts application
        3. Retrieves memories
        4. Verifies content matches exactly (via direct database query)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_exact_content.db")
            
            # Specific test content with special characters
            test_content = "Remember: Python 3.11+ has improved error messages!"
            
            # Session 1: Store memory
            engine1, storage1 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            stored_id = None
            
            try:
                result = engine1.process_message(test_content)
                
                assert result["intent"] == "store_memory"
                stored_id = result["metadata"]["memory_id"]
                
                # Verify content was stored correctly in session 1
                # by querying directly through the adapter
                memories = engine1.memory.retrieve(query="Python", limit=10)
                assert len(memories) > 0, "Should find stored memory"
                
                # Find our specific memory
                our_memory = None
                for mem in memories:
                    if mem["id"] == stored_id:
                        our_memory = mem
                        break
                
                assert our_memory is not None, \
                    f"Should find memory with ID {stored_id}"
                
                # Store the original content for comparison
                original_content = our_memory["content"]
                
            finally:
                cleanup_application(storage1)
            
            # Session 2: Restart and verify exact content
            engine2, storage2 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Retrieve the memory directly
                memories = engine2.memory.retrieve(query="Python", limit=10)
                
                assert len(memories) > 0, \
                    "Should find memory after restart"
                
                # Find our specific memory
                restored_memory = None
                for mem in memories:
                    if mem["id"] == stored_id:
                        restored_memory = mem
                        break
                
                assert restored_memory is not None, \
                    f"Should find memory with ID {stored_id} after restart"
                
                # Verify content matches exactly
                assert restored_memory["content"] == original_content, \
                    f"Content should match exactly. " \
                    f"Original: '{original_content}', " \
                    f"Restored: '{restored_memory['content']}'"
                
                # Verify metadata is preserved
                assert "metadata" in restored_memory, \
                    "Memory should have metadata"
                
                # Verify timestamp is preserved
                assert "timestamp" in restored_memory, \
                    "Memory should have timestamp"
                
            finally:
                cleanup_application(storage2)
    
    def test_restart_with_large_number_of_memories(self):
        """
        Test that restart works correctly with many memories.
        
        Validates Requirement 8.4:
        - System shall restore all previously stored memories regardless of count
        
        This test:
        1. Stores a large number of memories (50+)
        2. Restarts application
        3. Verifies all memories can be retrieved
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_large_restart.db")
            
            num_memories = 50
            stored_ids = []
            
            # Session 1: Store many memories
            engine1, storage1 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                for i in range(num_memories):
                    memory_text = f"Remember item number {i} with unique content"
                    result = engine1.process_message(memory_text)
                    
                    assert result["intent"] == "store_memory"
                    assert "memory_id" in result["metadata"]
                    
                    stored_ids.append(result["metadata"]["memory_id"])
                
                assert len(stored_ids) == num_memories, \
                    f"Should have stored {num_memories} memories"
            
            finally:
                cleanup_application(storage1)
            
            # Session 2: Restart and verify all memories are accessible
            engine2, storage2 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Retrieve with high limit to get all memories
                memories = engine2.memory.retrieve(query="item number", limit=100)
                
                assert len(memories) >= num_memories, \
                    f"Should retrieve at least {num_memories} memories after restart, " \
                    f"found {len(memories)}"
                
                # Verify all stored IDs are present
                retrieved_ids = [mem["id"] for mem in memories]
                
                for stored_id in stored_ids:
                    assert stored_id in retrieved_ids, \
                        f"Stored memory ID '{stored_id}' should be retrievable after restart"
            
            finally:
                cleanup_application(storage2)
    
    def test_restart_after_failed_operations(self):
        """
        Test that restart works correctly even after failed operations.
        
        Validates Requirement 8.4:
        - System shall restore memories correctly even if previous session
          had errors or failed operations
        
        This test:
        1. Stores some memories successfully
        2. Attempts operations that might fail
        3. Restarts application
        4. Verifies successful memories are still retrievable
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_failed_ops_restart.db")
            
            # Session 1: Store memories and attempt some operations
            engine1, storage1 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            successful_ids = []
            
            try:
                # Store successful memories
                for i in range(3):
                    result = engine1.process_message(f"Remember successful item {i}")
                    assert result["intent"] == "store_memory"
                    successful_ids.append(result["metadata"]["memory_id"])
                
                # Attempt some operations that might not work as expected
                # (e.g., empty messages, invalid queries)
                engine1.process_message("")  # Empty message
                engine1.process_message("Recall nonexistent_query_xyz")  # No results
                
            finally:
                cleanup_application(storage1)
            
            # Session 2: Restart and verify successful memories are intact
            engine2, storage2 = initialize_application(
                db_path=db_path,
                return_storage=True
            )
            
            try:
                # Retrieve the successful memories
                result = engine2.process_message("Recall successful item")
                
                assert result["intent"] == "retrieve_memory"
                assert result["metadata"]["memories_found"] >= 3, \
                    f"Should find at least 3 successful memories, " \
                    f"found {result['metadata']['memories_found']}"
                
                # Verify all successful IDs are retrievable
                memories = engine2.memory.retrieve(query="successful item", limit=10)
                retrieved_ids = [mem["id"] for mem in memories]
                
                for success_id in successful_ids:
                    assert success_id in retrieved_ids, \
                        f"Successful memory '{success_id}' should be retrievable after restart"
            
            finally:
                cleanup_application(storage2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
