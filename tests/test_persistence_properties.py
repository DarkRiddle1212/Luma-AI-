"""
Property-Based Tests for Memory Persistence

This module implements property-based tests using Hypothesis to verify
memory persistence across application restarts.

Feature: reasoning-memory-integration
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import os
import time
import gc

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


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
@settings(max_examples=10, deadline=None)
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
        storage1 = None
        storage2 = None
        
        try:
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
            storage1 = None
            
            # Force garbage collection to release file handles
            gc.collect()
            # Small delay to allow Windows to release file handles
            time.sleep(0.1)
            
            # Phase 2: Simulate application restart - reinitialize components
            storage2 = SQLiteStorage(db_path=db_path)
            memory_manager2 = MemoryManager(storage=storage2)
            memory2 = SQLiteMemoryAdapter(memory_manager=memory_manager2)
            engine2 = ReasoningEngine(llm=StubLLM(), memory=memory2)
            
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
                    # Verify content is preserved (strip whitespace for comparison)
                    assert content.lower().strip() in memory.action.lower().strip(), \
                        f"Stored content should be preserved. Original: '{content}', Retrieved: '{memory.action}'"
                    break
            
            assert found_memory, \
                f"Memory with ID '{memory_id}' should be found in database after restart"
            
            # Clean up
            storage2.close()
            del engine2
            del memory2
            del memory_manager2
            del storage2
            storage2 = None
            
            # Force garbage collection to release file handles
            gc.collect()
            # Small delay to allow Windows to release file handles
            time.sleep(0.1)
            
        finally:
            # Ensure storages are closed before cleanup
            if storage1 is not None:
                try:
                    storage1.close()
                except Exception:
                    pass
            if storage2 is not None:
                try:
                    storage2.close()
                except Exception:
                    pass
            
            # Final cleanup - force garbage collection
            gc.collect()
            # Give Windows time to release file handles before TemporaryDirectory cleanup
            time.sleep(0.1)


# Feature: reasoning-memory-integration, Property 7: Memory Persistence Round-Trip
@given(
    contents=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=5,
            max_size=100
        ),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_multiple_memories_persistence_property(contents):
    """
    Property: For any list of memory contents, storing multiple memories and
    then restarting the application should result in all memories being
    retrievable from local storage.
    
    **Validates: Requirements 8.3, 8.4**
    
    This test verifies that:
    1. Multiple memories can be stored
    2. All memories persist across restart
    3. No data loss occurs during restart
    4. All memories are retrievable after restart
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_multi_persistence.db")
        storage1 = None
        storage2 = None
        
        try:
            # Phase 1: Store multiple memories
            storage1 = SQLiteStorage(db_path=db_path)
            memory_manager1 = MemoryManager(storage=storage1)
            memory1 = SQLiteMemoryAdapter(memory_manager=memory_manager1)
            engine1 = ReasoningEngine(llm=StubLLM(), memory=memory1)
            
            stored_ids = []
            for content in contents:
                store_message = f"remember {content}"
                result = engine1.process_message(store_message)
                
                assert result["intent"] == "store_memory", \
                    f"Intent should be store_memory, got '{result['intent']}'"
                assert "memory_id" in result["metadata"], \
                    "Storage should return memory_id"
                
                stored_ids.append(result["metadata"]["memory_id"])
            
            # Verify all memories were stored
            assert len(stored_ids) == len(contents), \
                f"Should have stored {len(contents)} memories, got {len(stored_ids)}"
            
            # Close the first application instance
            storage1.close()
            del engine1
            del memory1
            del memory_manager1
            del storage1
            storage1 = None
            
            # Force garbage collection to release file handles
            gc.collect()
            # Small delay to allow Windows to release file handles
            time.sleep(0.1)
            
            # Phase 2: Simulate application restart
            storage2 = SQLiteStorage(db_path=db_path)
            memory_manager2 = MemoryManager(storage=storage2)
            memory2 = SQLiteMemoryAdapter(memory_manager=memory_manager2)
            engine2 = ReasoningEngine(llm=StubLLM(), memory=memory2)
            
            # Query all memories from the database
            all_memories = memory_manager2.query_memories(action_type="", limit=100)
            
            # Verify all memories are present
            assert len(all_memories) >= len(contents), \
                f"Should have at least {len(contents)} memories after restart, got {len(all_memories)}"
            
            # Verify each stored ID is found
            found_ids = [m.id for m in all_memories]
            for stored_id in stored_ids:
                assert stored_id in found_ids, \
                    f"Memory with ID '{stored_id}' should be found after restart"
            
            # Verify content is preserved for each memory
            for i, content in enumerate(contents):
                memory_id = stored_ids[i]
                memory = next((m for m in all_memories if m.id == memory_id), None)
                
                assert memory is not None, \
                    f"Memory with ID '{memory_id}' should exist"
                
                # Strip whitespace for comparison since memory system may normalize whitespace
                assert content.lower().strip() in memory.action.lower().strip(), \
                    f"Content should be preserved. Original: '{content}', Retrieved: '{memory.action}'"
            
            # Clean up
            storage2.close()
            del engine2
            del memory2
            del memory_manager2
            del storage2
            storage2 = None
            
            # Force garbage collection to release file handles
            gc.collect()
            # Small delay to allow Windows to release file handles
            time.sleep(0.1)
            
        finally:
            # Ensure storages are closed before cleanup
            if storage1 is not None:
                try:
                    storage1.close()
                except Exception:
                    pass
            if storage2 is not None:
                try:
                    storage2.close()
                except Exception:
                    pass
            
            # Final cleanup - force garbage collection
            gc.collect()
            # Give Windows time to release file handles before TemporaryDirectory cleanup
            time.sleep(0.1)


# Feature: reasoning-memory-integration, Property 7: Memory Persistence Round-Trip
@given(
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_persistence_across_multiple_restarts_property(content):
    """
    Property: For any memory content, the data should persist across multiple
    application restarts without data loss or corruption.
    
    **Validates: Requirements 8.3, 8.4**
    
    This test verifies that:
    1. Data persists across multiple restart cycles
    2. No data corruption occurs
    3. Content remains intact after multiple restarts
    4. The persistence mechanism is robust
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_multi_restart.db")
        storage = None
        
        try:
            # Initial store
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory = SQLiteMemoryAdapter(memory_manager=memory_manager)
            engine = ReasoningEngine(llm=StubLLM(), memory=memory)
            
            store_message = f"remember {content}"
            result = engine.process_message(store_message)
            memory_id = result["metadata"]["memory_id"]
            
            storage.close()
            del engine
            del memory
            del memory_manager
            del storage
            storage = None
            
            # Force garbage collection to release file handles
            gc.collect()
            # Small delay to allow Windows to release file handles
            time.sleep(0.1)
            
            # Simulate 3 restart cycles
            for restart_num in range(3):
                # Restart application
                storage = SQLiteStorage(db_path=db_path)
                memory_manager = MemoryManager(storage=storage)
                memory = SQLiteMemoryAdapter(memory_manager=memory_manager)
                engine = ReasoningEngine(llm=StubLLM(), memory=memory)
                
                # Verify memory still exists
                all_memories = memory_manager.query_memories(action_type="", limit=100)
                
                found = False
                for mem in all_memories:
                    if mem.id == memory_id:
                        found = True
                        # Strip whitespace for comparison since memory system may normalize whitespace
                        assert content.lower().strip() in mem.action.lower().strip(), \
                            f"Content should be preserved after restart {restart_num + 1}. " \
                            f"Original: '{content}', Retrieved: '{mem.action}'"
                        break
                
                assert found, \
                    f"Memory should exist after restart {restart_num + 1}"
                
                # Close for next restart
                storage.close()
                del engine
                del memory
                del memory_manager
                del storage
                storage = None
                
                # Force garbage collection to release file handles
                gc.collect()
                # Small delay to allow Windows to release file handles
                time.sleep(0.1)
                
        finally:
            # Ensure storage is closed before cleanup
            if storage is not None:
                try:
                    storage.close()
                except Exception:
                    pass
            
            # Final cleanup - force garbage collection
            gc.collect()
            # Give Windows time to release file handles before TemporaryDirectory cleanup
            time.sleep(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "property"])
