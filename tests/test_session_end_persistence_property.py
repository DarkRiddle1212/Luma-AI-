"""
Property-Based Test for Session End Persistence

This module implements property-based tests using Hypothesis to verify
that the Session_Manager correctly persists buffered memories when a session
ends normally.

Feature: memory-write-strategy-session-management
Property 5: Session end persistence
Validates: Requirements 2.4, 3.2, 11.3
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing session manager."""
    
    def __init__(self):
        self.stored_memories = []
        self.store_calls = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method."""
        memory_id = f"mem_{len(self.stored_memories)}"
        memory_entry = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        }
        self.stored_memories.append(memory_entry)
        self.store_calls.append(memory_entry)
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
# Property 5: Session End Persistence
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_memories=st.integers(min_value=1, max_value=50),
    content_prefix=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_session_end_persists_all_buffered_memories(num_memories, content_prefix):
    """
    Property: For any session with buffered memories, when the session ends
    normally, all buffered memories should be persisted to long-term storage.
    
    **Validates: Requirements 2.4, 3.2, 11.3**
    
    This test verifies that:
    1. All buffered memories are persisted when session ends with persist=True
    2. Persisted memories have correct content
    3. Persisted memories have correct metadata
    4. Buffer is cleared after persistence
    5. Session is removed from active tracking
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
        
        # Verify memories are buffered, not persisted
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_memories
        assert len(memory.stored_memories) == 0
        
        # End session with persist=True
        persisted_count = session_manager.end_session(session_id, persist=True)
        
        # Property 1: All memories should be persisted
        assert persisted_count == num_memories, \
            f"Expected {num_memories} persisted memories, got {persisted_count}"
        
        # Property 2: All memories should be in long-term storage
        assert len(memory.stored_memories) == num_memories, \
            f"Expected {num_memories} stored memories, got {len(memory.stored_memories)}"
        
        # Property 3: Persisted memories should have correct content
        for i in range(num_memories):
            expected_content = f"{content_prefix}_memory_{i}"
            assert memory.stored_memories[i]["content"] == expected_content, \
                f"Memory {i} content mismatch: expected '{expected_content}', got '{memory.stored_memories[i]['content']}'"
        
        # Property 4: Persisted memories should have correct metadata
        for i in range(num_memories):
            assert memory.stored_memories[i]["metadata"]["category"] == "test", \
                f"Memory {i} category mismatch"
            assert memory.stored_memories[i]["metadata"]["index"] == i, \
                f"Memory {i} index mismatch"
        
        # Property 5: Session should be removed from active tracking
        session = session_manager.get_session(session_id)
        assert session is None, \
            f"Session {session_id} should be removed after ending"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_sessions=st.integers(min_value=2, max_value=10),
    memories_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_multiple_sessions_persist_independently(num_sessions, memories_per_session):
    """
    Property: For any set of multiple sessions, when each session ends,
    only that session's buffered memories should be persisted.
    
    **Validates: Requirements 2.4, 3.2, 11.3**
    
    This test verifies that:
    1. Each session's memories are persisted independently
    2. No cross-contamination between sessions
    3. Each session's memory count is correct
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
        expected_counts: Dict[str, int] = {}
        for i, session_id in enumerate(session_ids):
            memories_in_session = memories_per_session + i  # Vary by session
            expected_counts[session_id] = memories_in_session
            for j in range(memories_in_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )
        
        # End each session and verify independent persistence
        for i, session_id in enumerate(session_ids):
            expected_count = expected_counts[session_id]
            persisted_count = session_manager.end_session(session_id, persist=True)
            
            # Property 1: Correct number of memories persisted for this session
            assert persisted_count == expected_count, \
                f"Session {i} should persist {expected_count} memories, got {persisted_count}"
        
        # Property 2: Total stored memories should match sum of all sessions
        total_expected = sum(expected_counts.values())
        assert len(memory.stored_memories) == total_expected, \
            f"Expected {total_expected} total stored memories, got {len(memory.stored_memories)}"
        
        # Property 3: All sessions should be removed
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            assert session is None, \
                f"Session {session_id} should be removed after ending"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_memories=st.integers(min_value=1, max_value=30),
    content=st.text(min_size=10, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_buffer_cleared_after_persistence(num_memories, content):
    """
    Property: After a session ends and memories are persisted, the session's
    buffer should be cleared to prevent duplicate persistence.
    
    **Validates: Requirements 3.2, 11.3**
    
    This test verifies that:
    1. Buffer is cleared after successful persistence
    2. Calling end_session again doesn't persist duplicate memories
    3. Buffer is empty after session ends
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
        
        # Buffer memories
        for i in range(num_memories):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"{content}_{i}",
                metadata={"index": i}
            )
        
        # Get initial buffer size
        initial_buffer = session_manager.get_session_memories(session_id)
        assert len(initial_buffer) == num_memories
        
        # End session
        persisted_count = session_manager.end_session(session_id, persist=True)
        assert persisted_count == num_memories
        
        # Property 1: Buffer should be cleared (session no longer exists)
        session = session_manager.get_session(session_id)
        assert session is None, \
            "Session should be removed after ending"
        
        # Property 2: No duplicate persistence on second call
        # (Calling end_session on non-existent session should return 0)
        second_persisted = session_manager.end_session(session_id, persist=True)
        assert second_persisted == 0, \
            f"Second end_session call should persist 0 memories, got {second_persisted}"
        
        # Property 3: Total stored should equal initial buffer size
        assert len(memory.stored_memories) == num_memories, \
            f"Expected {num_memories} stored memories (no duplicates), got {len(memory.stored_memories)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_memories=st.integers(min_value=1, max_value=20),
    content=st.text(min_size=5, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_persist_false_discards_buffer(num_memories, content):
    """
    Property: When a session ends with persist=False, buffered memories
    should be discarded without persistence.
    
    **Validates: Requirements 11.4**
    
    This test verifies that:
    1. No memories are persisted when persist=False
    2. Buffer is cleared even when persist=False
    3. Session is removed from active tracking
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
        
        # Buffer memories
        for i in range(num_memories):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"{content}_{i}",
                metadata={"index": i}
            )
        
        # Verify memories are buffered
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_memories
        assert len(memory.stored_memories) == 0
        
        # End session with persist=False
        persisted_count = session_manager.end_session(session_id, persist=False)
        
        # Property 1: No memories should be persisted
        assert persisted_count == 0, \
            f"Expected 0 persisted memories with persist=False, got {persisted_count}"
        
        # Property 2: No memories should be in long-term storage
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 stored memories with persist=False, got {len(memory.stored_memories)}"
        
        # Property 3: Session should be removed
        session = session_manager.get_session(session_id)
        assert session is None, \
            "Session should be removed even with persist=False"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_failures=st.integers(min_value=1, max_value=5),
    num_memories=st.integers(min_value=5, max_value=30)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_persistence_continues_on_errors(num_failures, num_memories):
    """
    Property: When persisting buffered memories, if some store operations fail,
    the system should continue persisting remaining memories and return the
    count of successfully persisted memories.
    
    **Validates: Requirements 2.4, 11.3**
    
    This test verifies that:
    1. Persistence continues after individual failures
    2. Successfully persisted memories are counted correctly
    3. Failed memories are logged but don't stop the process
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    
    # Create mock that will fail on specific indices
    # Use a fresh class instance for each test run to avoid state pollution
    class FailingMockMemoryInterface(MockMemoryInterface):
        def __init__(self, fail_count):
            super().__init__()
            self.fail_count = fail_count
            self.call_count = 0
        
        def store(self, content: str, metadata: dict = None) -> str:
            self.call_count += 1
            if self.call_count <= self.fail_count:
                raise MemoryStorageError(f"Simulated failure at call {self.call_count}")
            return super().store(content, metadata)
    
    memory = FailingMockMemoryInterface(fail_count=num_failures)
    
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer memories
        for i in range(num_memories):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"Memory_{i}",
                metadata={"index": i}
            )
        
        # End session with persist=True
        persisted_count = session_manager.end_session(session_id, persist=True)
        
        # Property 1: Some memories should be persisted (successful ones)
        successful_count = num_memories - num_failures
        assert persisted_count == successful_count, \
            f"Expected {successful_count} persisted memories, got {persisted_count}"
        
        # Property 2: Total stored should equal successful count
        assert len(memory.stored_memories) == successful_count, \
            f"Expected {successful_count} stored memories, got {len(memory.stored_memories)}"
        
        # Property 3: Session should still be removed despite errors
        session = session_manager.get_session(session_id)
        assert session is None, \
            "Session should be removed even with some persistence failures"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_memories=st.integers(min_value=1, max_value=20),
    content=st.text(min_size=10, max_size=100),
    metadata_keys=st.lists(
        st.text(min_size=3, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=1,
        max_size=5,
        unique=True
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_metadata_integrity_on_persistence(num_memories, content, metadata_keys):
    """
    Property: When buffered memories are persisted on session end, all
    metadata should be preserved exactly as it was when buffered.
    
    **Validates: Requirements 2.4, 3.2**
    
    This test verifies that:
    1. All metadata fields are preserved
    2. Complex metadata structures are maintained
    3. No fields are added, removed, or modified during persistence
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
        
        # Buffer memories with complex metadata
        for i in range(num_memories):
            metadata = {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 5)],
                "custom_field": f"value_{i}",
                "number_field": i * 10,
                "bool_field": i % 2 == 0,
                "nested": {"key": f"nested_value_{i}", "level2": {"deep": i}},
            }
            # Add dynamic metadata keys
            for key in metadata_keys:
                metadata[f"dynamic_{key}"] = f"dynamic_value_{i}_{key}"
            
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"{content}_{i}",
                metadata=metadata
            )
        
        # End session
        persisted_count = session_manager.end_session(session_id, persist=True)
        assert persisted_count == num_memories
        
        # Property 1: All memories should be persisted
        assert len(memory.stored_memories) == num_memories
        
        # Property 2: Metadata should be preserved exactly
        for i in range(num_memories):
            expected_metadata = {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 5)],
                "custom_field": f"value_{i}",
                "number_field": i * 10,
                "bool_field": i % 2 == 0,
                "nested": {"key": f"nested_value_{i}", "level2": {"deep": i}},
            }
            for key in metadata_keys:
                expected_metadata[f"dynamic_{key}"] = f"dynamic_value_{i}_{key}"
            
            actual_metadata = memory.stored_memories[i]["metadata"]
            assert actual_metadata == expected_metadata, \
                f"Metadata mismatch for memory {i}:\nExpected: {expected_metadata}\nActual: {actual_metadata}"
        
        # Property 3: Content should be preserved
        for i in range(num_memories):
            assert memory.stored_memories[i]["content"] == f"{content}_{i}", \
                f"Content mismatch for memory {i}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 5: Session end persistence
@given(
    num_sessions=st.integers(min_value=1, max_value=5),
    operations_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_5_concurrent_session_end_operations(num_sessions, operations_per_session):
    """
    Property: For any number of concurrent session end operations,
    each session's buffered memories should be persisted correctly
    without race conditions or data corruption.
    
    **Validates: Requirements 2.4, 3.2, 11.3**
    
    This test verifies that:
    1. Concurrent session endings don't corrupt persistence
    2. Each session's memories are persisted independently
    3. Thread safety is maintained during persistence
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
        
        # Buffer memories in each session
        expected_counts: Dict[str, int] = {}
        for i, session_id in enumerate(session_ids):
            memories_in_session = operations_per_session + i
            expected_counts[session_id] = memories_in_session
            for j in range(memories_in_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )
        
        # End all sessions concurrently
        results: Dict[str, int] = {}
        lock = threading.Lock()
        
        def end_session_thread(session_id: str):
            """Thread function to end a session."""
            count = session_manager.end_session(session_id, persist=True)
            with lock:
                results[session_id] = count
        
        threads: List[threading.Thread] = []
        for session_id in session_ids:
            thread = threading.Thread(target=end_session_thread, args=(session_id,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Property 1: Each session should have correct persisted count
        for session_id in session_ids:
            expected = expected_counts[session_id]
            actual = results[session_id]
            assert actual == expected, \
                f"Session {session_id} should persist {expected} memories, got {actual}"
        
        # Property 2: Total stored should match sum of all sessions
        total_expected = sum(expected_counts.values())
        assert len(memory.stored_memories) == total_expected, \
            f"Expected {total_expected} total stored memories, got {len(memory.stored_memories)}"
        
        # Property 3: All sessions should be removed
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            assert session is None, \
                f"Session {session_id} should be removed after ending"
    
    finally:
        session_manager.shutdown()
