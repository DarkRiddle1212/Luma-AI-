"""
Property-Based Test for Session Cancellation

This module implements property-based tests using Hypothesis to verify
that the Session_Manager correctly discards buffered memories when a session
is cancelled without persisting them to long-term storage.

Feature: memory-write-strategy-session-management
Property 10: Session cancellation discards buffer
Validates: Requirements 11.4
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
# Property 10: Session Cancellation Discards Buffer
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
@given(
    num_memories=st.integers(min_value=1, max_value=50),
    content_prefix=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_10_cancel_session_discards_buffer(num_memories, content_prefix):
    """
    Property: For any session that is cancelled or aborted, all buffered
    memories should be discarded without persistence.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. Buffered memories are NOT persisted when session is cancelled
    2. Buffer is cleared after cancellation
    3. Session is removed from active tracking
    4. No memories reach long-term storage
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

        # Verify memories are buffered
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_memories, \
            f"Expected {num_memories} buffered memories, got {len(buffered)}"

        # Verify no memories persisted yet
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories before cancellation, got {len(memory.stored_memories)}"

        # Cancel the session
        session_manager.cancel_session(session_id)

        # Property 1: No memories should be persisted after cancellation
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories after cancellation, got {len(memory.stored_memories)}"

        # Property 2: Session should be removed from active tracking
        session = session_manager.get_session(session_id)
        assert session is None, \
            f"Session {session_id} should be removed after cancellation"

        # Property 3: Attempting to get memories from cancelled session returns empty list
        buffered_after = session_manager.get_session_memories(session_id)
        assert len(buffered_after) == 0, \
            f"Expected 0 memories after cancellation, got {len(buffered_after)}"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
@given(
    num_sessions=st.integers(min_value=2, max_value=10),
    memories_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_10_cancel_one_session_preserves_others(num_sessions, memories_per_session):
    """
    Property: For any set of multiple sessions, cancelling one session
    should only discard that session's buffer without affecting other sessions.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. Cancelling one session doesn't affect other sessions
    2. Other sessions can still be ended normally with persistence
    3. Session isolation is maintained during cancellation
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

        # Buffer memories in each session
        for i, session_id in enumerate(session_ids):
            for j in range(memories_per_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )

        # Cancel the first session
        cancelled_session_id = session_ids[0]
        session_manager.cancel_session(cancelled_session_id)

        # Property 1: Cancelled session's memories should NOT be persisted
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories after cancelling first session, got {len(memory.stored_memories)}"

        # Property 2: Other sessions should still have their buffered memories
        for i in range(1, num_sessions):
            session_id = session_ids[i]
            buffered = session_manager.get_session_memories(session_id)
            assert len(buffered) == memories_per_session, \
                f"Session {i} should still have {memories_per_session} buffered memories, got {len(buffered)}"

        # Property 3: End remaining sessions normally (with persistence)
        for i in range(1, num_sessions):
            session_id = session_ids[i]
            persisted_count = session_manager.end_session(session_id, persist=True)
            assert persisted_count == memories_per_session, \
                f"Session {i} should persist {memories_per_session} memories, got {persisted_count}"

        # Property 4: Only non-cancelled sessions' memories should be persisted
        expected_total = (num_sessions - 1) * memories_per_session
        assert len(memory.stored_memories) == expected_total, \
            f"Expected {expected_total} total persisted memories, got {len(memory.stored_memories)}"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
@given(
    num_memories=st.integers(min_value=1, max_value=30),
    content=st.text(min_size=10, max_size=100)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_10_cancel_vs_end_with_persist_false(num_memories, content):
    """
    Property: Cancelling a session should have the same effect as ending
    a session with persist=False - both discard the buffer without persistence.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. cancel_session() behaves identically to end_session(persist=False)
    2. Both methods discard buffered memories
    3. Both methods remove the session from tracking
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory1 = MockMemoryInterface()
    memory2 = MockMemoryInterface()
    session_manager1 = Session_Manager(config, memory1)
    session_manager2 = Session_Manager(config, memory2)

    try:
        # Create two identical sessions
        session_id1 = session_manager1.create_session()
        session_id2 = session_manager2.create_session()

        # Buffer identical memories in both sessions
        for i in range(num_memories):
            session_manager1.buffer_memory(
                session_id=session_id1,
                content=f"{content}_{i}",
                metadata={"index": i}
            )
            session_manager2.buffer_memory(
                session_id=session_id2,
                content=f"{content}_{i}",
                metadata={"index": i}
            )

        # Cancel first session, end second with persist=False
        session_manager1.cancel_session(session_id1)
        session_manager2.end_session(session_id2, persist=False)

        # Property 1: Both should have 0 persisted memories
        assert len(memory1.stored_memories) == 0, \
            f"cancel_session should persist 0 memories, got {len(memory1.stored_memories)}"
        assert len(memory2.stored_memories) == 0, \
            f"end_session(persist=False) should persist 0 memories, got {len(memory2.stored_memories)}"

        # Property 2: Both sessions should be removed
        session1 = session_manager1.get_session(session_id1)
        session2 = session_manager2.get_session(session_id2)
        assert session1 is None, "Session 1 should be removed after cancel_session"
        assert session2 is None, "Session 2 should be removed after end_session(persist=False)"

    finally:
        session_manager1.shutdown()
        session_manager2.shutdown()


# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
@given(
    num_memories=st.integers(min_value=1, max_value=20),
    content=st.text(min_size=5, max_size=5)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_10_cancel_nonexistent_session_safe(num_memories, content):
    """
    Property: Cancelling a non-existent session should be safe and not
    raise an error.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. Cancelling a non-existent session doesn't raise an exception
    2. No side effects occur from cancelling non-existent sessions
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
        # Try to cancel a session that doesn't exist
        fake_session_id = "nonexistent-session-id-12345"
        
        # Property 1: Should not raise an exception
        session_manager.cancel_session(fake_session_id)

        # Property 2: No memories should be persisted
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories, got {len(memory.stored_memories)}"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
@given(
    num_sessions=st.integers(min_value=1, max_value=5),
    operations_per_session=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_10_concurrent_cancellations(num_sessions, operations_per_session):
    """
    Property: For any number of concurrent session cancellations,
    each session's buffer should be discarded correctly without race
    conditions or data corruption.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. Concurrent cancellations don't corrupt session state
    2. No memories are persisted from any cancelled session
    3. Thread safety is maintained during cancellation
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
        for i, session_id in enumerate(session_ids):
            for j in range(operations_per_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )

        # Cancel all sessions concurrently
        def cancel_session_thread(session_id: str):
            """Thread function to cancel a session."""
            session_manager.cancel_session(session_id)

        threads: List[threading.Thread] = []
        for session_id in session_ids:
            thread = threading.Thread(target=cancel_session_thread, args=(session_id,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Property 1: No memories should be persisted
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories after concurrent cancellations, got {len(memory.stored_memories)}"

        # Property 2: All sessions should be removed
        for session_id in session_ids:
            session = session_manager.get_session(session_id)
            assert session is None, \
                f"Session {session_id} should be removed after cancellation"

    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 10: Session cancellation discards buffer
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
def test_property_10_complex_metadata_discarded_on_cancel(num_memories, content, metadata_keys):
    """
    Property: When a session is cancelled, all buffered memories with
    complex metadata should be discarded without any persistence attempts.

    **Validates: Requirements 11.4**

    This test verifies that:
    1. Complex metadata doesn't affect cancellation behavior
    2. No partial persistence occurs
    3. All buffered data is completely discarded
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

        # Verify memories are buffered
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_memories

        # Cancel the session
        session_manager.cancel_session(session_id)

        # Property 1: No memories should be persisted
        assert len(memory.stored_memories) == 0, \
            f"Expected 0 persisted memories after cancellation, got {len(memory.stored_memories)}"

        # Property 2: Session should be removed
        session = session_manager.get_session(session_id)
        assert session is None, \
            "Session should be removed after cancellation"

    finally:
        session_manager.shutdown()
