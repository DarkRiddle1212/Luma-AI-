"""
Property-Based Test for Unique Session ID Generation

This module implements property-based tests using Hypothesis to verify
that the Session_Manager generates unique session IDs with no collisions.

Feature: memory-write-strategy-session-management
Property 4: Unique session creation
Validates: Requirements 2.1, 13.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Set

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing session manager."""
    
    def __init__(self):
        self.stored_memories = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method."""
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Mock retrieve method."""
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Property 4: Unique Session Creation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 4: Unique session creation
@given(num_sessions=st.integers(min_value=1, max_value=100))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_4_unique_session_creation(num_sessions):
    """
    Property: For any number of session creation requests, the Session_Manager
    should generate unique session_ids with no collisions.
    
    **Validates: Requirements 2.1, 13.5**
    
    This test verifies that:
    1. Each session creation returns a unique session_id
    2. No two sessions have the same session_id
    3. Session IDs are properly formatted (UUID strings)
    4. All created sessions can be retrieved by their IDs
    """
    # Create session manager with test configuration
    config = SessionConfig(
        timeout_seconds=3600,  # 1 hour (long enough for test)
        cleanup_interval_seconds=3600,  # Don't cleanup during test
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create multiple sessions
        session_ids: Set[str] = set()
        created_sessions = []
        
        for _ in range(num_sessions):
            session_id = session_manager.create_session()
            created_sessions.append(session_id)
            session_ids.add(session_id)
        
        # Property 1: All session IDs should be unique (set size == list size)
        assert len(session_ids) == num_sessions, \
            f"Expected {num_sessions} unique session IDs, but got {len(session_ids)} unique IDs"
        
        # Property 2: Each session ID should be a non-empty string
        for session_id in created_sessions:
            assert isinstance(session_id, str), \
                f"Session ID should be a string, got {type(session_id)}"
            assert len(session_id) > 0, \
                "Session ID should not be empty"
        
        # Property 3: Each session ID should be retrievable
        for session_id in created_sessions:
            session = session_manager.get_session(session_id)
            assert session is not None, \
                f"Session {session_id} should be retrievable after creation"
            assert session.session_id == session_id, \
                f"Retrieved session ID {session.session_id} should match requested ID {session_id}"
        
        # Property 4: Session IDs should follow UUID format (36 characters with hyphens)
        for session_id in created_sessions:
            assert len(session_id) == 36, \
                f"Session ID should be 36 characters (UUID format), got {len(session_id)}"
            assert session_id.count('-') == 4, \
                f"Session ID should have 4 hyphens (UUID format), got {session_id.count('-')}"
    
    finally:
        # Cleanup: shutdown session manager to stop background threads
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 4: Unique session creation
@given(
    num_sessions=st.integers(min_value=2, max_value=50),
    metadata_list=st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=5),
            values=st.one_of(st.text(max_size=5), st.integers(), st.booleans())
        ),
        min_size=2,
        max_size=50
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_4_unique_sessions_with_metadata(num_sessions, metadata_list):
    """
    Property: For any number of session creation requests with various metadata,
    the Session_Manager should generate unique session_ids regardless of metadata.
    
    **Validates: Requirements 2.1, 13.5**
    
    This test verifies that:
    1. Session IDs are unique even when metadata is identical
    2. Metadata does not affect session ID generation
    3. Each session preserves its own metadata correctly
    """
    # Ensure we have enough metadata entries
    while len(metadata_list) < num_sessions:
        metadata_list.append({})
    
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create sessions with metadata
        session_ids: Set[str] = set()
        session_metadata_map = {}
        
        for i in range(num_sessions):
            metadata = metadata_list[i]
            session_id = session_manager.create_session(metadata=metadata)
            session_ids.add(session_id)
            session_metadata_map[session_id] = metadata
        
        # Property 1: All session IDs should be unique
        assert len(session_ids) == num_sessions, \
            f"Expected {num_sessions} unique session IDs, got {len(session_ids)}"
        
        # Property 2: Each session should preserve its metadata
        for session_id, expected_metadata in session_metadata_map.items():
            session = session_manager.get_session(session_id)
            assert session is not None, \
                f"Session {session_id} should be retrievable"
            assert session.metadata == expected_metadata, \
                f"Session metadata mismatch: expected {expected_metadata}, got {session.metadata}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 4: Unique session creation
@given(num_sessions=st.integers(min_value=10, max_value=100))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_4_concurrent_session_creation_uniqueness(num_sessions):
    """
    Property: For any number of concurrent session creation requests,
    the Session_Manager should generate unique session_ids with thread safety.
    
    **Validates: Requirements 2.1, 13.5**
    
    This test verifies that:
    1. Concurrent session creation maintains uniqueness
    2. Thread-safe session ID generation
    3. No race conditions in session creation
    """
    import threading
    
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Shared list to collect session IDs from threads
        session_ids = []
        lock = threading.Lock()
        
        def create_session_thread():
            """Thread function to create a session."""
            session_id = session_manager.create_session()
            with lock:
                session_ids.append(session_id)
        
        # Create threads
        threads = []
        for _ in range(num_sessions):
            thread = threading.Thread(target=create_session_thread)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Property 1: All session IDs should be unique
        unique_ids = set(session_ids)
        assert len(unique_ids) == num_sessions, \
            f"Expected {num_sessions} unique session IDs from concurrent creation, " \
            f"got {len(unique_ids)} unique IDs (duplicates detected)"
        
        # Property 2: All sessions should be retrievable
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            assert session is not None, \
                f"Session {session_id} should be retrievable after concurrent creation"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 4: Unique session creation
@given(num_sessions=st.integers(min_value=1, max_value=50))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_4_session_id_format_consistency(num_sessions):
    """
    Property: For any number of session creation requests, all generated
    session IDs should follow a consistent format (UUID v4).
    
    **Validates: Requirements 2.1, 13.5**
    
    This test verifies that:
    1. Session IDs follow UUID format consistently
    2. Session IDs are valid UUIDs
    3. Session IDs can be parsed as UUIDs
    """
    import uuid
    
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create sessions
        session_ids = []
        for _ in range(num_sessions):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        # Property 1: All session IDs should be valid UUIDs
        for session_id in session_ids:
            try:
                parsed_uuid = uuid.UUID(session_id)
                # Verify it's a valid UUID by converting back to string
                assert str(parsed_uuid) == session_id, \
                    f"UUID round-trip failed: {session_id} != {str(parsed_uuid)}"
            except ValueError as e:
                pytest.fail(f"Session ID {session_id} is not a valid UUID: {e}")
        
        # Property 2: All session IDs should be lowercase (UUID standard)
        for session_id in session_ids:
            assert session_id == session_id.lower(), \
                f"Session ID should be lowercase: {session_id}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 4: Unique session creation
@given(
    num_iterations=st.integers(min_value=2, max_value=10),
    sessions_per_iteration=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_4_session_uniqueness_across_multiple_managers(num_iterations, sessions_per_iteration):
    """
    Property: For any number of Session_Manager instances created sequentially,
    each should generate unique session IDs with no collisions across managers.
    
    **Validates: Requirements 2.1, 13.5**
    
    This test verifies that:
    1. Session IDs are globally unique (not just per manager instance)
    2. UUID generation is properly random
    3. No predictable patterns in session ID generation
    """
    all_session_ids: Set[str] = set()
    total_sessions = 0
    
    for _ in range(num_iterations):
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create sessions with this manager
            for _ in range(sessions_per_iteration):
                session_id = session_manager.create_session()
                all_session_ids.add(session_id)
                total_sessions += 1
        finally:
            session_manager.shutdown()
    
    # Property: All session IDs across all managers should be unique
    assert len(all_session_ids) == total_sessions, \
        f"Expected {total_sessions} unique session IDs across all managers, " \
        f"got {len(all_session_ids)} unique IDs (collisions detected)"
