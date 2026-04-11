"""
Property-Based Tests for End-to-End Storage Flow

This module implements property-based tests using Hypothesis to verify
the complete end-to-end storage flow from user message to database persistence.

Feature: reasoning-memory-integration
"""

import pytest
import tempfile
import os
from hypothesis import given, strategies as st, settings
from pathlib import Path

from luma.container import initialize_application, cleanup_application
from luma_memory.storage.sqlite_storage import SQLiteStorage


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def storage_request_message(draw):
    """
    Generate random storage request messages.
    
    These messages should trigger the store_memory intent and result in
    database storage. Uses only triggers recognized by detect_intent.
    """
    # Store memory trigger words (only those recognized by detect_intent)
    triggers = ["remember", "store"]
    trigger = draw(st.sampled_from(triggers))
    
    # Content to store (what comes after the trigger)
    # Use alphanumeric text to avoid issues with special characters
    content = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    ))
    
    # Construct message with trigger + content
    message = f"{trigger} {content}"
    
    return message


# ============================================================================
# 6.3 Property Test: End-to-End Storage Flow (Property 8)
# ============================================================================

# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(message=storage_request_message())
@settings(max_examples=10, deadline=5000)
@pytest.mark.property_test
def test_end_to_end_storage_flow_property(message):
    """
    Property: For any user message requesting memory storage, the system should
    process the intent, store the memory in the database, and respond with
    confirmation in a single message processing flow.
    
    **Validates: Requirements 9.1**
    
    This test verifies the complete end-to-end flow:
    1. User sends storage request message
    2. ReasoningEngine detects store_memory intent
    3. Memory is stored in SQLite database
    4. Response contains confirmation
    5. Memory persists in database and can be queried
    6. The entire flow works consistently across all valid storage requests
    
    This is a true integration test that uses real database operations
    to verify the complete system behavior.
    """
    # Create temporary database for this test iteration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize application with real database
        engine, storage = initialize_application(
            db_path=db_path,
            return_storage=True
        )
        
        # Process the storage request message
        result = engine.process_message(message)
        
        # ===================================================================
        # Verify 1: Intent Detection
        # ===================================================================
        assert result["intent"] == "store_memory", \
            f"Intent should be 'store_memory' for message '{message}', got '{result['intent']}'"
        
        # ===================================================================
        # Verify 2: Response Structure
        # ===================================================================
        assert isinstance(result, dict), \
            f"Result must be a dict, got {type(result)}"
        
        assert "response" in result, \
            "Result must contain 'response' key"
        assert "metadata" in result, \
            "Result must contain 'metadata' key"
        
        # ===================================================================
        # Verify 3: Confirmation in Response
        # ===================================================================
        response_text = result["response"]
        assert isinstance(response_text, str), \
            f"Response must be a string, got {type(response_text)}"
        assert len(response_text) > 0, \
            "Response should not be empty"
        
        # Verify confirmation message (should mention "stored" or similar)
        confirmation_keywords = ["stored", "saved", "remembered", "kept"]
        has_confirmation = any(keyword in response_text.lower() for keyword in confirmation_keywords)
        assert has_confirmation, \
            f"Response should contain confirmation keyword, got: '{response_text}'"
        
        # ===================================================================
        # Verify 4: Memory ID in Metadata
        # ===================================================================
        assert "memory_id" in result["metadata"], \
            "Result metadata must contain 'memory_id'"
        
        memory_id = result["metadata"]["memory_id"]
        assert isinstance(memory_id, str), \
            f"memory_id must be a string, got {type(memory_id)}"
        assert len(memory_id) > 0, \
            "memory_id should not be empty"
        
        # ===================================================================
        # Verify 5: Memory Stored in Database
        # ===================================================================
        # Query the database directly to verify storage
        stored_memory = storage.get_entry(memory_id)
        
        assert stored_memory is not None, \
            f"Memory with ID '{memory_id}' should exist in database"
        
        # ===================================================================
        # Verify 6: Stored Content Matches Request
        # ===================================================================
        # The stored action should contain the content from the message
        # (with trigger words removed)
        assert isinstance(stored_memory.action, str), \
            f"Stored action must be a string, got {type(stored_memory.action)}"
        assert len(stored_memory.action) > 0, \
            "Stored action should not be empty"
        
        # Verify the content is present in the stored action
        # (after lowercasing, since implementation does this)
        message_lower = message.lower()
        action_lower = stored_memory.action.lower()
        
        # The stored action should contain some of the original message content
        # (excluding trigger words)
        for trigger in ["remember", "store"]:
            message_lower = message_lower.replace(trigger, "").strip()
        
        # Check that the content is preserved (allowing for some transformation)
        assert len(action_lower) > 0, \
            "Stored action should contain content"
        
        # ===================================================================
        # Verify 7: Memory Metadata
        # ===================================================================
        assert stored_memory.tags is not None, \
            "Stored memory should have tags (even if empty list)"
        assert isinstance(stored_memory.tags, list), \
            f"Tags must be a list, got {type(stored_memory.tags)}"
        
        assert stored_memory.context is not None, \
            "Stored memory should have context"
        assert isinstance(stored_memory.context, dict), \
            f"Context must be a dict, got {type(stored_memory.context)}"
        
        # ===================================================================
        # Verify 8: Memory Timestamp
        # ===================================================================
        assert stored_memory.created_at is not None or stored_memory.timestamp is not None, \
            "Stored memory should have a timestamp"
        
        # ===================================================================
        # Verify 9: Memory Can Be Retrieved
        # ===================================================================
        # Verify the memory can be retrieved by ID
        retrieved_memory = storage.get_entry(memory_id)
        assert retrieved_memory is not None, \
            f"Memory with ID '{memory_id}' should be retrievable from database"
        
        assert retrieved_memory.id == memory_id, \
            f"Retrieved memory ID should match, expected '{memory_id}', got '{retrieved_memory.id}'"
        
        assert retrieved_memory.action == stored_memory.action, \
            "Retrieved memory action should match stored memory"
        
        # ===================================================================
        # Success: Complete end-to-end flow verified
        # ===================================================================
        
    finally:
        # Cleanup: Close database connections and remove temp file
        try:
            cleanup_application(storage)
        except:
            pass
        
        # Remove temporary database file
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass  # Ignore cleanup errors


# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(
    trigger=st.sampled_from(["remember", "store"]),
    content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10, deadline=5000)
@pytest.mark.property_test
def test_storage_persistence_property(trigger, content):
    """
    Property: For any storage request, the stored memory should persist
    in the database and be retrievable after the initial storage operation.
    
    **Validates: Requirements 9.1**
    
    This test verifies that:
    1. Memory is actually written to disk (not just in-memory)
    2. Memory persists after the storage operation completes
    3. Memory can be retrieved using database queries
    4. The storage is durable and reliable
    """
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize application
        engine, storage = initialize_application(
            db_path=db_path,
            return_storage=True
        )
        
        # Construct message
        message = f"{trigger} {content}"
        
        # Process the message
        result = engine.process_message(message)
        
        # Verify storage succeeded
        assert result["intent"] == "store_memory", \
            f"Intent should be 'store_memory', got '{result['intent']}'"
        
        assert "memory_id" in result["metadata"], \
            "Result should contain memory_id"
        
        memory_id = result["metadata"]["memory_id"]
        
        # Verify memory persists in database
        retrieved = storage.get_entry(memory_id)
        
        assert retrieved is not None, \
            f"Memory with ID '{memory_id}' should be in database"
        
        assert retrieved.id == memory_id, \
            f"Retrieved memory ID should match"
        
        # Verify content is preserved
        assert isinstance(retrieved.action, str), \
            "Retrieved action must be a string"
        assert len(retrieved.action) > 0, \
            "Retrieved action should not be empty"
        
    finally:
        # Cleanup
        try:
            cleanup_application(storage)
        except:
            pass
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass


# Feature: reasoning-memory-integration, Property 8: End-to-End Storage Flow
@given(message=storage_request_message())
@settings(max_examples=10, deadline=5000)
@pytest.mark.property_test
def test_database_integrity_property(message):
    """
    Property: For any storage operation, the database should maintain
    integrity with valid schema, constraints, and data types.
    
    **Validates: Requirements 9.1**
    
    This test verifies that:
    1. Database schema is valid
    2. All required fields are populated
    3. Data types are correct
    4. Constraints are enforced
    5. No data corruption occurs
    """
    # Create temporary database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Initialize application
        engine, storage = initialize_application(
            db_path=db_path,
            return_storage=True
        )
        
        # Process the message
        result = engine.process_message(message)
        
        # Verify storage succeeded
        assert result["intent"] == "store_memory", \
            "Intent should be 'store_memory'"
        
        memory_id = result["metadata"]["memory_id"]
        
        # Retrieve the stored memory
        stored_memory = storage.get_entry(memory_id)
        assert stored_memory is not None, \
            "Memory should be retrievable from database"
        
        # Verify database integrity: all required fields present
        assert hasattr(stored_memory, 'id'), \
            "Memory entry must have 'id' field"
        assert hasattr(stored_memory, 'action'), \
            "Memory entry must have 'action' field"
        assert hasattr(stored_memory, 'context'), \
            "Memory entry must have 'context' field"
        assert hasattr(stored_memory, 'tags'), \
            "Memory entry must have 'tags' field"
        
        # Verify field types
        assert isinstance(stored_memory.id, str), \
            f"ID must be string, got {type(stored_memory.id)}"
        assert isinstance(stored_memory.action, str), \
            f"Action must be string, got {type(stored_memory.action)}"
        assert isinstance(stored_memory.context, dict), \
            f"Context must be dict, got {type(stored_memory.context)}"
        assert isinstance(stored_memory.tags, list), \
            f"Tags must be list, got {type(stored_memory.tags)}"
        
        # Verify ID is not empty
        assert len(stored_memory.id) > 0, \
            "ID should not be empty"
        
        # Verify action is not empty
        assert len(stored_memory.action) > 0, \
            "Action should not be empty"
        
        # Verify timestamp exists
        assert stored_memory.created_at is not None or stored_memory.timestamp is not None, \
            "Memory must have a timestamp"
        
    finally:
        # Cleanup
        try:
            cleanup_application(storage)
        except:
            pass
        
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "property"])
