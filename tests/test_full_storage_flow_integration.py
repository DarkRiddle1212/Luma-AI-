"""
Integration Test for Full Storage Flow

This module implements an integration test for the complete memory write strategy
storage flow, verifying end-to-end functionality from message input through all
processing stages to final storage.

Feature: memory-write-strategy-session-management, Integration Test: Full Storage Flow
"""

import pytest
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

from luma.core.write_strategy import (
    Memory_Write_Strategy,
    WriteStrategyConfig,
    SessionConfig
)
from luma.core.session_manager import Session_Manager
from luma.core.memory_interface import MemoryInterface, MemoryStorageError
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


# ============================================================================
# Integration Test: Full Storage Flow
# ============================================================================


def test_full_storage_flow_integration():
    """
    Integration test for complete storage flow through Memory_Write_Strategy.
    
    This test verifies the end-to-end flow:
    1. Message input
    2. Write trigger evaluation (non-trivial, non-repetitive)
    3. Content validation
    4. Duplicate detection
    5. Metadata normalization
    6. Storage (either buffered or immediate)
    
    The test covers the happy path where a valid message flows through all
    stages and gets stored successfully, verifying that all transformations
    (normalization, metadata enrichment) happen correctly.
    
    **Validates: All write strategy requirements**
    """
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_integration.db")
        
        # Initialize storage components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        memory_adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager,
            device_id="test-device-001",
            default_category="test_category",
            default_tags=["integration_test"]
        )
        
        # Initialize session manager
        session_config = SessionConfig(
            timeout_seconds=1800,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        session_manager = Session_Manager(
            config=session_config,
            memory_interface=memory_adapter
        )
        
        # Initialize write strategy
        write_config = WriteStrategyConfig(
            trivial_patterns=["hello", "hi", "hey", "thanks", "ok"],
            min_content_length=3,
            repetition_window=5,
            immediate_persist_patterns=[],
            similarity_threshold=0.9,
            enable_conflict_detection=True
        )
        write_strategy = Memory_Write_Strategy(
            config=write_config,
            session_manager=session_manager,
            memory_interface=memory_adapter
        )
        
        try:
            # ================================================================
            # Stage 1: Message Input
            # ================================================================
            message_content = "This is an important piece of information about Python programming"
            message_metadata = {
                "category": "Programming",
                "tags": ["Python", "Learning"]
            }
            
            # ================================================================
            # Stage 2-6: Full Storage Flow
            # ================================================================
            # The store_memory method will internally:
            # - Evaluate write trigger (Stage 2)
            # - Validate content (Stage 3)
            # - Check for duplicates (Stage 4)
            # - Normalize metadata (Stage 5)
            # - Store the memory (Stage 6)
            #
            # We'll verify each stage by examining the results and logs
            # Since there's no active session, storage should be immediate
            memory_id = write_strategy.store_memory(
                content=message_content,
                metadata=message_metadata,
                immediate=False  # Let strategy decide
            )
            
            assert memory_id is not None, \
                "Memory ID should be returned from storage"
            assert isinstance(memory_id, str), \
                f"Memory ID should be a string, got: {type(memory_id)}"
            assert len(memory_id) > 0, \
                "Memory ID should not be empty"
            
            # ================================================================
            # Verification: Retrieve and Verify Stored Memory
            # ================================================================
            # Retrieve the stored memory to verify all transformations
            result = memory_adapter.retrieve(
                params={
                    "query": "Python programming",
                    "limit": 10
                }
            )
            
            assert "memories" in result, \
                "Result should contain 'memories' key"
            assert len(result["memories"]) > 0, \
                "At least one memory should be retrieved"
            
            stored_memory = result["memories"][0]
            
            # Verify content is preserved
            assert "content" in stored_memory, \
                "Stored memory should have 'content' field"
            # Content might be transformed by adapter, but should contain original text
            assert "python programming" in stored_memory["content"].lower(), \
                f"Stored content should contain original text, got: {stored_memory['content']}"
            
            # Verify metadata is present
            assert "metadata" in stored_memory, \
                "Stored memory should have 'metadata' field"
            
            # Verify category was normalized (Stage 5 verification)
            assert "category" in stored_memory, \
                "Stored memory should have 'category' field"
            assert stored_memory["category"] == "programming", \
                f"Category should be normalized to lowercase, got: {stored_memory['category']}"
            
            # Verify tags were normalized and merged (Stage 5 verification)
            assert "tags" in stored_memory, \
                "Stored memory should have 'tags' field"
            stored_tags = set(stored_memory["tags"])
            assert "python" in stored_tags, \
                f"Tags should include 'python', got: {stored_tags}"
            assert "learning" in stored_tags, \
                f"Tags should include 'learning', got: {stored_tags}"
            assert "integration_test" in stored_tags, \
                f"Tags should include default tag 'integration_test', got: {stored_tags}"
            
            # Verify timestamp is present (Stage 5 verification)
            assert "timestamp" in stored_memory, \
                "Stored memory should have 'timestamp' field"
            
            # ================================================================
            # Stage 2 Verification: Write Trigger Evaluation
            # ================================================================
            # The message passed trigger evaluation (otherwise store_memory would have raised)
            # Verify by checking that the memory was stored successfully
            assert memory_id is not None and len(memory_id) > 0, \
                "Write trigger evaluation passed (memory was stored)"
            
            # ================================================================
            # Stage 3 Verification: Content Validation
            # ================================================================
            # Content validation passed (otherwise store_memory would have raised)
            # Verify by checking that content is valid in stored memory
            assert len(stored_memory["content"]) > 0, \
                "Content validation passed (content is non-empty)"
            
            # ================================================================
            # Test Duplicate Detection on Second Storage Attempt
            # ================================================================
            # Attempt to store the same content again (with slight variation to avoid repetition check)
            # The duplicate detection should catch this based on content similarity
            duplicate_content = message_content  # Same content
            duplicate_metadata = {
                "category": "Programming",  # Same category
                "tags": ["Python"]
            }
            
            # To avoid the repetition check, we need to clear the recent messages
            # or use the check_duplicate method directly
            duplicate_id_check = write_strategy.check_duplicate(
                duplicate_content,
                "programming"  # normalized category
            )
            
            # Should find the existing memory as a duplicate
            assert duplicate_id_check == memory_id, \
                f"Duplicate detection should find existing memory. Expected: {memory_id}, Got: {duplicate_id_check}"
            
            # ================================================================
            # Test Session Buffering Flow
            # ================================================================
            # Create a session and test buffered storage
            session_id = session_manager.create_session(
                metadata={"user_id": "test_user"}
            )
            
            # Set the current session ID on the session manager
            # (In real usage, this would be tracked by the application)
            session_manager.current_session_id = session_id
            
            # Store a different message (should be buffered)
            buffered_content = "Another important message about data structures"
            buffered_metadata = {
                "category": "Computer Science",
                "tags": ["Data Structures"]
            }
            
            buffered_id = write_strategy.store_memory(
                content=buffered_content,
                metadata=buffered_metadata,
                immediate=False
            )
            
            # Should return a buffer ID (not a database ID)
            assert buffered_id is not None, \
                "Buffered memory should return an ID"
            assert "buffered:" in buffered_id, \
                f"Buffered memory ID should contain 'buffered:', got: {buffered_id}"
            
            # Verify memory is in session buffer
            session_memories = session_manager.get_session_memories(session_id)
            assert len(session_memories) == 1, \
                f"Session should have 1 buffered memory, got: {len(session_memories)}"
            
            buffered_entry = session_memories[0]
            assert buffered_entry["content"] == buffered_content, \
                "Buffered content should match original"
            
            # ================================================================
            # Test Session End Persistence
            # ================================================================
            # End the session and verify buffered memories are persisted
            persisted_count = session_manager.end_session(session_id, persist=True)
            
            assert persisted_count == 1, \
                f"Should persist 1 memory on session end, got: {persisted_count}"
            
            # Verify the buffered memory is now in the database
            result = memory_adapter.retrieve(
                params={
                    "query": "data structures",
                    "limit": 10
                }
            )
            
            assert len(result["memories"]) > 0, \
                "Buffered memory should be persisted to database"
            
            # Find the persisted memory
            found_buffered = False
            for memory in result["memories"]:
                if "data structures" in memory["content"].lower():
                    found_buffered = True
                    # Verify category was normalized
                    assert memory["category"] == "computer science", \
                        f"Category should be normalized, got: {memory['category']}"
                    # Verify tags were normalized
                    assert "data structures" in [tag.lower() for tag in memory["tags"]], \
                        f"Tags should include 'data structures', got: {memory['tags']}"
                    break
            
            assert found_buffered, \
                "Buffered memory should be found in database after session end"
            
            # ================================================================
            # Test Immediate Persistence with Pattern
            # ================================================================
            # Configure immediate persist pattern
            write_strategy.config.immediate_persist_patterns = ["urgent"]
            
            # Create a new session
            session_id2 = session_manager.create_session()
            session_manager.current_session_id = session_id2
            
            # Store a message with immediate persist pattern
            urgent_content = "This is an urgent message that needs immediate storage"
            urgent_metadata = {"category": "alerts"}
            
            urgent_id = write_strategy.store_memory(
                content=urgent_content,
                metadata=urgent_metadata,
                immediate=False  # Pattern should trigger immediate persistence
            )
            
            # Should return a database ID (not buffered)
            assert urgent_id is not None, \
                "Urgent message should return an ID"
            assert "buffered:" not in urgent_id, \
                f"Urgent message should be persisted immediately, got: {urgent_id}"
            
            # Verify it's in the database immediately
            result = memory_adapter.retrieve(
                params={
                    "query": "urgent message",
                    "limit": 10
                }
            )
            
            assert len(result["memories"]) > 0, \
                "Urgent message should be in database immediately"
            
            # Clean up session
            session_manager.end_session(session_id2, persist=True)
            
            # ================================================================
            # Success: All stages verified
            # ================================================================
            print("✓ Full storage flow integration test passed")
            print(f"  - Write trigger evaluation: PASS")
            print(f"  - Content validation: PASS")
            print(f"  - Duplicate detection: PASS")
            print(f"  - Metadata normalization: PASS")
            print(f"  - Immediate storage: PASS")
            print(f"  - Session buffering: PASS")
            print(f"  - Session end persistence: PASS")
            print(f"  - Immediate persist pattern: PASS")
            
        finally:
            # Cleanup
            session_manager.shutdown()
            storage.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
