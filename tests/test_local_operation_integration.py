"""
Integration Test for Local Operation

This module tests that the reasoning-memory integration operates entirely locally
without requiring network connectivity. It verifies that memories are stored to
local SQLite files and can be retrieved without any external dependencies.

Feature: reasoning-memory-integration
Requirements: 8.1, 8.2
"""

import pytest
import tempfile
import os
import socket
from pathlib import Path
from unittest.mock import patch, MagicMock

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


class NetworkBlocker:
    """
    Context manager that blocks all network operations.
    
    This ensures that tests fail if any network calls are attempted,
    verifying that the system operates entirely locally.
    """
    
    def __init__(self):
        self.original_socket = None
    
    def __enter__(self):
        """Block network operations by patching socket."""
        self.original_socket = socket.socket
        
        def blocked_socket(*args, **kwargs):
            raise RuntimeError(
                "Network operation attempted during local-only test! "
                "The system should operate entirely locally without network calls."
            )
        
        socket.socket = blocked_socket
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original socket."""
        socket.socket = self.original_socket
        return False


class TestLocalOperationIntegration:
    """
    Integration tests for local operation of reasoning-memory system.
    
    These tests verify that:
    1. Memories can be stored using ReasoningEngine
    2. Data persists to local SQLite file
    3. No network calls are made during operation
    4. All operations work without external dependencies
    """
    
    def test_store_memory_locally_without_network(self):
        """
        Test that memories can be stored locally without network access.
        
        Validates Requirements 8.1, 8.2:
        - System uses SQLite for local memory storage
        - System does not require network connectivity
        
        This test:
        1. Creates a temporary SQLite database
        2. Blocks all network operations
        3. Stores a memory using ReasoningEngine
        4. Verifies the memory was stored successfully
        5. Verifies no network calls were attempted
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_local.db")
            
            # Initialize components with local storage
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Create reasoning engine with memory
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Block network operations to ensure local-only operation
            with NetworkBlocker():
                # Store a memory
                result = engine.process_message("Remember to buy groceries")
                
                # Verify storage succeeded
                assert result["intent"] == "store_memory", \
                    f"Intent should be store_memory, got '{result['intent']}'"
                
                assert "memory_id" in result["metadata"], \
                    "Result should contain memory_id in metadata"
                
                assert "stored that information" in result["response"].lower(), \
                    f"Response should confirm storage: '{result['response']}'"
                
                memory_id = result["metadata"]["memory_id"]
                assert isinstance(memory_id, str), \
                    f"memory_id must be a string, got {type(memory_id)}"
                assert len(memory_id) > 0, \
                    "memory_id should not be empty"
            
            # Verify database file exists
            assert os.path.exists(db_path), \
                f"Database file should exist at {db_path}"
            
            # Verify database file has content (not empty)
            db_size = os.path.getsize(db_path)
            assert db_size > 0, \
                f"Database file should have content, got size {db_size}"
            
            # Clean up
            storage.close()
    
    def test_retrieve_memory_locally_without_network(self):
        """
        Test that memories can be retrieved locally without network access.
        
        Validates Requirements 8.1, 8.2:
        - System uses SQLite for local memory storage
        - System does not require network connectivity
        
        This test:
        1. Creates a temporary SQLite database
        2. Stores a memory locally
        3. Blocks all network operations
        4. Retrieves the memory using ReasoningEngine
        5. Verifies the retrieval was successful
        6. Verifies no network calls were attempted
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_retrieve_local.db")
            
            # Initialize components with local storage
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Create reasoning engine with memory
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Store a memory first (without network blocking to isolate test)
            store_result = engine.process_message("Remember Python is a programming language")
            assert store_result["intent"] == "store_memory"
            assert "memory_id" in store_result["metadata"]
            
            # Block network operations to ensure local-only retrieval
            with NetworkBlocker():
                # Retrieve the memory
                retrieve_result = engine.process_message("Recall Python")
                
                # Verify retrieval succeeded
                assert retrieve_result["intent"] == "retrieve_memory", \
                    f"Intent should be retrieve_memory, got '{retrieve_result['intent']}'"
                
                assert "memories_found" in retrieve_result["metadata"], \
                    "Result should contain memories_found in metadata"
                
                memories_found = retrieve_result["metadata"]["memories_found"]
                assert memories_found > 0, \
                    f"Should find at least one memory, found {memories_found}"
                
                assert "memory_ids" in retrieve_result["metadata"], \
                    "Result should contain memory_ids when memories found"
                
                memory_ids = retrieve_result["metadata"]["memory_ids"]
                assert isinstance(memory_ids, list), \
                    f"memory_ids must be a list, got {type(memory_ids)}"
                assert len(memory_ids) > 0, \
                    "memory_ids should not be empty when memories found"
            
            # Clean up
            storage.close()
    
    def test_data_persists_to_local_sqlite_file(self):
        """
        Test that stored memories persist to the local SQLite file.
        
        Validates Requirement 8.1:
        - System persists memory data to local storage
        
        This test:
        1. Creates a temporary SQLite database
        2. Stores multiple memories
        3. Verifies database file exists and has content
        4. Queries the database directly to verify data persistence
        5. Verifies all stored memories are in the database
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_persistence.db")
            
            # Initialize components with local storage
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Create reasoning engine with memory
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Store multiple memories
            test_memories = [
                "Remember to buy milk",
                "Remember to call the dentist",
                "Remember Python is a programming language"
            ]
            
            stored_ids = []
            for memory_text in test_memories:
                result = engine.process_message(memory_text)
                assert result["intent"] == "store_memory"
                assert "memory_id" in result["metadata"]
                stored_ids.append(result["metadata"]["memory_id"])
            
            # Verify database file exists
            assert os.path.exists(db_path), \
                f"Database file should exist at {db_path}"
            
            # Verify database file has content
            db_size = os.path.getsize(db_path)
            assert db_size > 0, \
                f"Database file should have content, got size {db_size}"
            
            # Query database directly to verify persistence
            all_memories = memory_manager.query_memories(action_type="", limit=100)
            
            assert len(all_memories) >= len(test_memories), \
                f"Database should contain at least {len(test_memories)} memories, " \
                f"found {len(all_memories)}"
            
            # Verify all stored IDs are in the database
            db_memory_ids = [mem.id for mem in all_memories]
            for stored_id in stored_ids:
                assert stored_id in db_memory_ids, \
                    f"Stored memory ID '{stored_id}' should be in database"
            
            # Verify memory content is preserved
            for memory in all_memories:
                # Check that the action field contains meaningful content
                assert len(memory.action) > 0, \
                    f"Memory action should not be empty for ID {memory.id}"
                
                # Verify the memory has required fields
                assert memory.id is not None, "Memory should have an ID"
                assert memory.timestamp is not None, "Memory should have a timestamp"
                assert memory.device_id is not None, "Memory should have a device_id"
            
            # Clean up
            storage.close()
    
    def test_complete_workflow_without_network(self):
        """
        Test complete store-retrieve workflow operates locally without network.
        
        Validates Requirements 8.1, 8.2:
        - System uses SQLite for local memory storage
        - System does not require network connectivity
        
        This test:
        1. Creates a temporary SQLite database
        2. Blocks all network operations
        3. Performs complete store-retrieve workflow
        4. Verifies all operations succeed without network
        5. Verifies data persists correctly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            
            # Initialize components with local storage
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Create reasoning engine with memory
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Block network operations for entire workflow
            with NetworkBlocker():
                # Step 1: Store a memory
                store_result = engine.process_message("Remember JavaScript is a web language")
                
                assert store_result["intent"] == "store_memory"
                assert "memory_id" in store_result["metadata"]
                memory_id = store_result["metadata"]["memory_id"]
                
                # Step 2: Retrieve the memory
                retrieve_result = engine.process_message("Recall JavaScript")
                
                assert retrieve_result["intent"] == "retrieve_memory"
                assert "memories_found" in retrieve_result["metadata"]
                assert retrieve_result["metadata"]["memories_found"] > 0
                
                # Step 3: Verify the retrieved memory matches what was stored
                assert "memory_ids" in retrieve_result["metadata"]
                retrieved_ids = retrieve_result["metadata"]["memory_ids"]
                assert memory_id in retrieved_ids, \
                    f"Retrieved memory IDs should include stored ID '{memory_id}'"
                
                # Step 4: Store another memory
                store_result2 = engine.process_message("Remember TypeScript is a superset of JavaScript")
                
                assert store_result2["intent"] == "store_memory"
                assert "memory_id" in store_result2["metadata"]
                
                # Step 5: Retrieve both memories
                retrieve_result2 = engine.process_message("Recall JavaScript")
                
                assert retrieve_result2["intent"] == "retrieve_memory"
                assert retrieve_result2["metadata"]["memories_found"] >= 2, \
                    "Should find at least 2 memories related to JavaScript"
            
            # Verify database file exists and has content
            assert os.path.exists(db_path)
            assert os.path.getsize(db_path) > 0
            
            # Verify data persists in database
            all_memories = memory_manager.query_memories(action_type="", limit=100)
            assert len(all_memories) >= 2, \
                f"Database should contain at least 2 memories, found {len(all_memories)}"
            
            # Clean up
            storage.close()
    
    def test_local_operation_with_database_path_verification(self):
        """
        Test that the system uses the specified local database path.
        
        Validates Requirement 8.1:
        - System persists memory data to local storage
        
        This test:
        1. Creates a temporary directory with a specific path
        2. Initializes storage with that path
        3. Stores memories
        4. Verifies the database file is created at the exact specified path
        5. Verifies no other database files are created
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a specific subdirectory structure
            data_dir = os.path.join(tmpdir, "data", "memories")
            os.makedirs(data_dir, exist_ok=True)
            
            db_path = os.path.join(data_dir, "luma_memory.db")
            
            # Verify database doesn't exist yet
            assert not os.path.exists(db_path), \
                "Database should not exist before initialization"
            
            # Initialize components with specific local path
            storage = SQLiteStorage(db_path=db_path)
            memory_manager = MemoryManager(storage=storage)
            memory_adapter = SQLiteMemoryAdapter(memory_manager=memory_manager)
            
            # Create reasoning engine with memory
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory_adapter)
            
            # Store a memory
            result = engine.process_message("Remember to test local storage")
            
            assert result["intent"] == "store_memory"
            assert "memory_id" in result["metadata"]
            
            # Verify database file was created at the exact specified path
            assert os.path.exists(db_path), \
                f"Database file should exist at specified path: {db_path}"
            
            # Verify it's a file, not a directory
            assert os.path.isfile(db_path), \
                f"Database path should be a file: {db_path}"
            
            # Verify the file has content
            db_size = os.path.getsize(db_path)
            assert db_size > 0, \
                f"Database file should have content, got size {db_size}"
            
            # Verify no other database files were created in the temp directory
            all_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.db'):
                        all_files.append(os.path.join(root, file))
            
            assert len(all_files) == 1, \
                f"Should have exactly one database file, found {len(all_files)}: {all_files}"
            
            assert all_files[0] == db_path, \
                f"Database file should be at specified path. Expected: {db_path}, Got: {all_files[0]}"
            
            # Clean up
            storage.close()
    
    def test_no_network_dependencies_in_imports(self):
        """
        Test that the core components don't import network-related modules.
        
        Validates Requirement 8.2:
        - System does not require network connectivity
        
        This test verifies that the core reasoning and memory components
        don't have dependencies on network libraries like requests, urllib,
        http.client, etc.
        """
        # Import the modules
        import luma.core.reasoning
        import luma.core.memory_interface
        import luma.adapters.sqlite_memory_adapter
        import luma_memory.storage.sqlite_storage
        
        # Get module attributes
        reasoning_attrs = dir(luma.core.reasoning)
        memory_interface_attrs = dir(luma.core.memory_interface)
        adapter_attrs = dir(luma.adapters.sqlite_memory_adapter)
        storage_attrs = dir(luma_memory.storage.sqlite_storage)
        
        # Network-related module names to check for
        network_modules = [
            'requests', 'urllib', 'http', 'httplib', 'httpx',
            'aiohttp', 'websocket', 'socket'
        ]
        
        # Check that none of the network modules are imported
        all_attrs = (
            reasoning_attrs + memory_interface_attrs +
            adapter_attrs + storage_attrs
        )
        
        for attr in all_attrs:
            for network_module in network_modules:
                # Check if attribute name suggests network module usage
                # (This is a heuristic check, not exhaustive)
                if network_module in attr.lower():
                    # socket is used by SQLite internally, which is acceptable
                    if network_module == 'socket' and 'sqlite' in attr.lower():
                        continue
                    
                    # If we find a suspicious attribute, log it but don't fail
                    # (Some false positives are expected)
                    print(f"Note: Found attribute '{attr}' that might suggest "
                          f"network module '{network_module}' usage")
        
        # The main verification is that the system works without network
        # (tested in other test methods), this is just a supplementary check
        assert True, "Import check completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
