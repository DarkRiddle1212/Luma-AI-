"""
Property-Based Test for Session Memory Buffering

This module implements property-based tests using Hypothesis to verify
that the Session_Manager correctly buffers memories during active sessions
and maintains buffer integrity.

Feature: memory-write-strategy-session-management
Property 7: Session memory buffering
Validates: Requirements 3.1, 11.1, 11.2
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any

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
# Property 7: Session Memory Buffering
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    num_memories=st.integers(min_value=1, max_value=50),
    content_prefix=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_buffer_memory_adds_to_session(num_memories, content_prefix):
    """
    Property: For any message approved for storage during an active session,
    it should be added to the session buffer and not immediately persisted.
    
    **Validates: Requirements 3.1, 11.1, 11.2**
    
    This test verifies that:
    1. Buffered memories are added to the session buffer
    2. Buffered memories are NOT immediately persisted to long-term storage
    3. Buffer size matches the number of buffered memories
    4. Each buffered memory has the correct content and metadata
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer multiple memories
        for i in range(num_memories):
            content = f"{content_prefix}_memory_{i}"
            metadata = {"category": "test", "index": i}
            session_manager.buffer_memory(
                session_id=session_id,
                content=content,
                metadata=metadata
            )
        
        # Property 1: All memories should be in the session buffer
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_memories, \
            f"Expected {num_memories} buffered memories, got {len(buffered)}"
        
        # Property 2: Buffered memories should have correct content
        for i in range(num_memories):
            expected_content = f"{content_prefix}_memory_{i}"
            assert buffered[i]["content"] == expected_content, \
                f"Memory {i} content mismatch: expected '{expected_content}', got '{buffered[i]['content']}'"
        
        # Property 3: Buffered memories should have correct metadata
        for i in range(num_memories):
            assert buffered[i]["metadata"]["category"] == "test", \
                f"Memory {i} category mismatch"
            assert buffered[i]["metadata"]["index"] == i, \
                f"Memory {i} index mismatch"
        
        # Property 4: Buffered memories should have buffered_at timestamp
        for memory_entry in buffered:
            assert "buffered_at" in memory_entry, \
                "Buffered memory should have buffered_at timestamp"
        
        # Property 5: Memories should NOT be persisted yet (buffering behavior)
        assert len(memory.stored_memories) == 0, \
            f"Expected no persisted memories during buffering, got {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    num_sessions=st.integers(min_value=2, max_value=10),
    memories_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_session_isolation(num_sessions, memories_per_session):
    """
    Property: For any set of concurrent sessions, each session should maintain
    separate buffer state with no interference between sessions.
    
    **Validates: Requirements 3.1, 11.1, 11.2**
    
    This test verifies that:
    1. Each session maintains its own independent buffer
    2. Buffering in one session doesn't affect other sessions
    3. Session isolation is maintained even with concurrent operations
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create multiple sessions
        session_ids: List[str] = []
        for _ in range(num_sessions):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        # Buffer different numbers of memories in each session
        for i, session_id in enumerate(session_ids):
            memories_in_session = memories_per_session + i  # Vary by session
            for j in range(memories_in_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )
        
        # Property 1: Each session should have its own memories
        for i, session_id in enumerate(session_ids):
            buffered = session_manager.get_session_memories(session_id)
            expected_count = memories_per_session + i
            assert len(buffered) == expected_count, \
                f"Session {i} should have {expected_count} memories, got {len(buffered)}"
            
            # Property 2: Each session's memories should be distinct
            for j in range(expected_count):
                assert buffered[j]["content"] == f"Session_{i}_Memory_{j}", \
                    f"Session {i} memory {j} content mismatch"
        
        # Property 3: No memories should be persisted yet
        assert len(memory.stored_memories) == 0, \
            f"Expected no persisted memories during buffering, got {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    num_operations=st.integers(min_value=1, max_value=100),
    content=st.text(min_size=5, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_buffer_metadata_integrity(num_operations, content):
    """
    Property: For any memory buffered during an active session, the metadata
    should be preserved exactly as provided without modification.
    
    **Validates: Requirements 3.1, 11.1**
    
    This test verifies that:
    1. Metadata is stored exactly as provided
    2. No fields are added, removed, or modified during buffering
    3. Complex metadata structures are preserved
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer memories with various metadata
        for i in range(num_operations):
            metadata = {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 5)],
                "custom_field": f"value_{i}",
                "number_field": i * 10,
                "bool_field": i % 2 == 0,
                "nested": {"key": f"nested_value_{i}"}
            }
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"{content}_{i}",
                metadata=metadata
            )
        
        # Retrieve buffered memories
        buffered = session_manager.get_session_memories(session_id)
        
        # Property 1: All memories should be buffered
        assert len(buffered) == num_operations, \
            f"Expected {num_operations} buffered memories, got {len(buffered)}"
        
        # Property 2: Metadata should be preserved exactly
        for i in range(num_operations):
            expected_metadata = {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 5)],
                "custom_field": f"value_{i}",
                "number_field": i * 10,
                "bool_field": i % 2 == 0,
                "nested": {"key": f"nested_value_{i}"}
            }
            assert buffered[i]["metadata"] == expected_metadata, \
                f"Metadata mismatch for memory {i}: expected {expected_metadata}, got {buffered[i]['metadata']}"
        
        # Property 3: Content should be preserved
        for i in range(num_operations):
            assert buffered[i]["content"] == f"{content}_{i}", \
                f"Content mismatch for memory {i}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    session_id=st.text(min_size=1, max_size=5),
    num_memories=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_get_session_memories_nonexistent(session_id, num_memories):
    """
    Property: For any session that doesn't exist, get_session_memories
    should return an empty list without raising an error.
    
    **Validates: Requirements 11.5**
    
    This test verifies that:
    1. Nonexistent sessions return empty list
    2. No exception is raised for nonexistent sessions
    3. The method is safe to call on any session_id
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Try to get memories for a session (may or may not exist)
        # Note: We don't create the session, so it will be nonexistent
        buffered = session_manager.get_session_memories(session_id)
        
        # Property 1: Should return empty list for nonexistent session
        assert isinstance(buffered, list), \
            f"Expected list, got {type(buffered)}"
        assert len(buffered) == 0, \
            f"Expected empty list for nonexistent session, got {len(buffered)} memories"
        
        # Property 2: No exception should be raised
        # (if we got here, no exception was raised)
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    num_memories=st.integers(min_value=1, max_value=20),
    content=st.text(min_size=10, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_buffer_copy_isolation(num_memories, content):
    """
    Property: For any call to get_session_memories, the returned list
    should be a copy, not the internal buffer, preventing external modification.
    
    **Validates: Requirements 11.5**
    
    This test verifies that:
    1. get_session_memories returns a copy of the buffer
    2. Modifying the returned list doesn't affect the internal buffer
    3. Buffer integrity is maintained
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer some memories
        for i in range(num_memories):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"{content}_{i}",
                metadata={"index": i}
            )
        
        # Get buffered memories
        buffered1 = session_manager.get_session_memories(session_id)
        original_count = len(buffered1)
        
        # Modify the returned list
        buffered1.append({"content": "modified", "metadata": {}})
        buffered1.clear()
        
        # Property 1: Original buffer should be unaffected
        buffered2 = session_manager.get_session_memories(session_id)
        assert len(buffered2) == original_count, \
            f"Internal buffer should be unaffected by external modification, " \
            f"expected {original_count}, got {len(buffered2)}"
        
        # Property 2: All original memories should still be accessible
        for i in range(original_count):
            assert buffered2[i]["content"] == f"{content}_{i}", \
                f"Memory {i} should still be accessible"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 7: Session memory buffering
@given(
    num_sessions=st.integers(min_value=1, max_value=5),
    operations_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_7_concurrent_buffer_operations(num_sessions, operations_per_session):
    """
    Property: For any number of concurrent buffer operations across sessions,
    each session's buffer should remain consistent and isolated.
    
    **Validates: Requirements 3.1, 11.1, 11.2**
    
    This test verifies that:
    1. Concurrent buffer operations don't corrupt session state
    2. Each session maintains correct buffer size
    3. Thread safety is maintained
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
        # Create sessions
        session_ids: List[str] = []
        for _ in range(num_sessions):
            session_id = session_manager.create_session()
            session_ids.append(session_id)
        
        # Track expected buffer sizes
        expected_sizes: Dict[str, int] = {sid: 0 for sid in session_ids}
        lock = threading.Lock()
        
        def buffer_memories(session_id: str, count: int):
            """Thread function to buffer memories in a session."""
            for i in range(count):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{session_id[:8]}_Memory_{i}",
                    metadata={"thread": threading.current_thread().name, "index": i}
                )
            with lock:
                expected_sizes[session_id] += count
        
        # Create threads for each session
        threads: List[threading.Thread] = []
        for session_id in session_ids:
            thread = threading.Thread(
                target=buffer_memories,
                args=(session_id, operations_per_session),
                name=f"Thread-{session_id[:8]}"
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Property 1: Each session should have correct buffer size
        for session_id in session_ids:
            buffered = session_manager.get_session_memories(session_id)
            expected = expected_sizes[session_id]
            assert len(buffered) == expected, \
                f"Session {session_id} should have {expected} buffered memories, got {len(buffered)}"
        
        # Property 2: No memories should be persisted yet
        assert len(memory.stored_memories) == 0, \
            f"Expected no persisted memories during buffering, got {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()
