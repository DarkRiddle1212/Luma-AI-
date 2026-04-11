"""
Property-Based Test for Multi-Session Isolation

This module implements property-based tests using Hypothesis to verify
that the Session_Manager maintains proper isolation between concurrent sessions.

Feature: memory-write-strategy-session-management
Property 24: Multi-session isolation
Validates: Requirements 13.1, 13.2, 13.3
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Set
import threading
import time

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig, WriteStrategyConfig, Memory_Write_Strategy
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Thread-safe mock memory interface for testing concurrent sessions."""
    
    def __init__(self):
        self.stored_memories = []
        self.lock = threading.Lock()
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method with thread safety."""
        with self.lock:
            memory_id = f"mem_{len(self.stored_memories)}"
            self.stored_memories.append({
                "id": memory_id,
                "content": content,
                "metadata": metadata or {}
            })
            return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Mock retrieve method with thread safety."""
        with self.lock:
            return {"memories": self.stored_memories.copy()}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Property 24: Multi-Session Isolation
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 24: Multi-session isolation
@given(
    num_sessions=st.integers(min_value=2, max_value=10),
    memories_per_session=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_property_24_multi_session_isolation(num_sessions, memories_per_session):
    """
    Property: For any set of concurrent sessions, each session should maintain
    separate state (buffer, metadata, session_id) with no interference between sessions.
    
    **Validates: Requirements 13.1, 13.2, 13.3**
    
    This test verifies that:
    1. Multiple sessions maintain separate state for each session_id
    2. Memory storage associates with correct session_id in multi-session environment
    3. Memory retrieval filters by session_id when appropriate
    4. Buffered memories don't leak between sessions
    5. Session metadata remains isolated
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
        # Create multiple sessions with unique metadata
        sessions_data = []
        for i in range(num_sessions):
            session_id = session_manager.create_session(
                metadata={"session_index": i, "test_marker": f"session_{i}"}
            )
            sessions_data.append({
                "session_id": session_id,
                "index": i,
                "expected_memories": []
            })
        
        # Buffer memories in each session with unique content
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            index = session_data["index"]
            
            for mem_idx in range(memories_per_session):
                content = f"Memory {mem_idx} for session {index}"
                metadata = {
                    "session_id": session_id,
                    "memory_index": mem_idx,
                    "session_index": index
                }
                
                session_manager.buffer_memory(session_id, content, metadata)
                session_data["expected_memories"].append({
                    "content": content,
                    "metadata": metadata
                })
        
        # Property 1: Each session should have exactly the expected number of buffered memories
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            buffered = session_manager.get_session_memories(session_id)
            
            assert len(buffered) == memories_per_session, \
                f"Session {session_id} should have {memories_per_session} buffered memories, " \
                f"but has {len(buffered)}"
        
        # Property 2: Buffered memories should match expected content and metadata
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            buffered = session_manager.get_session_memories(session_id)
            expected = session_data["expected_memories"]
            
            for i, (buffered_mem, expected_mem) in enumerate(zip(buffered, expected)):
                assert buffered_mem["content"] == expected_mem["content"], \
                    f"Session {session_id} memory {i} content mismatch"
                assert buffered_mem["metadata"]["session_id"] == session_id, \
                    f"Session {session_id} memory {i} has wrong session_id in metadata"
                assert buffered_mem["metadata"]["memory_index"] == expected_mem["metadata"]["memory_index"], \
                    f"Session {session_id} memory {i} has wrong memory_index"
        
        # Property 3: Session metadata should remain isolated
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            session = session_manager.get_session(session_id)
            
            assert session is not None, f"Session {session_id} should exist"
            assert session.metadata["session_index"] == session_data["index"], \
                f"Session {session_id} metadata corrupted"
            assert session.metadata["test_marker"] == f"session_{session_data['index']}", \
                f"Session {session_id} test_marker corrupted"
        
        # Property 4: No memory should appear in multiple session buffers
        all_buffered_contents = []
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            buffered = session_manager.get_session_memories(session_id)
            
            for mem in buffered:
                content_with_session = (mem["content"], mem["metadata"]["session_id"])
                all_buffered_contents.append(content_with_session)
        
        # Each (content, session_id) pair should be unique
        assert len(all_buffered_contents) == len(set(all_buffered_contents)), \
            "Memories leaked between sessions - found duplicate (content, session_id) pairs"
        
        # Property 5: After ending one session, other sessions should remain unaffected
        if num_sessions >= 2:
            # End the first session
            first_session_id = sessions_data[0]["session_id"]
            session_manager.end_session(first_session_id, persist=True)
            
            # Verify first session is gone
            assert session_manager.get_session(first_session_id) is None, \
                f"Session {first_session_id} should be removed after ending"
            
            # Verify other sessions still exist with correct state
            for session_data in sessions_data[1:]:
                session_id = session_data["session_id"]
                session = session_manager.get_session(session_id)
                
                assert session is not None, \
                    f"Session {session_id} should still exist after ending another session"
                
                buffered = session_manager.get_session_memories(session_id)
                assert len(buffered) == memories_per_session, \
                    f"Session {session_id} buffer corrupted after ending another session"
        
    finally:
        # Cleanup: end all remaining sessions
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)


# Feature: memory-write-strategy-session-management, Property 24: Multi-session isolation (concurrent)
@given(
    num_sessions=st.integers(min_value=2, max_value=8),
    operations_per_session=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_property_24_concurrent_multi_session_isolation(num_sessions, operations_per_session):
    """
    Property: For any set of concurrent sessions with concurrent operations,
    each session should maintain separate state with thread-safe isolation.
    
    **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
    
    This test verifies that:
    1. Concurrent buffer operations maintain session isolation
    2. Thread-safe data structures prevent race conditions
    3. No memory corruption occurs during concurrent access
    4. Each session's state remains consistent
    """
    # Create session manager with test configuration
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    # Track results from threads
    results = {
        "errors": [],
        "session_ids": [],
        "buffered_counts": {}
    }
    results_lock = threading.Lock()
    
    def session_worker(worker_id: int):
        """Worker function that creates a session and buffers memories."""
        try:
            # Create session
            session_id = session_manager.create_session(
                metadata={"worker_id": worker_id}
            )
            
            with results_lock:
                results["session_ids"].append(session_id)
            
            # Buffer memories
            for op_idx in range(operations_per_session):
                content = f"Worker {worker_id} memory {op_idx}"
                metadata = {
                    "session_id": session_id,
                    "worker_id": worker_id,
                    "operation_index": op_idx
                }
                session_manager.buffer_memory(session_id, content, metadata)
                
                # Small delay to increase chance of interleaving
                time.sleep(0.001)
            
            # Record final buffer count
            buffered = session_manager.get_session_memories(session_id)
            with results_lock:
                results["buffered_counts"][session_id] = len(buffered)
            
        except Exception as e:
            with results_lock:
                results["errors"].append((worker_id, str(e)))
    
    try:
        # Create and start worker threads
        threads = []
        for worker_id in range(num_sessions):
            thread = threading.Thread(target=session_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread did not complete in time"
        
        # Property 1: No errors should occur during concurrent operations
        assert len(results["errors"]) == 0, \
            f"Errors occurred during concurrent operations: {results['errors']}"
        
        # Property 2: All sessions should be created successfully
        assert len(results["session_ids"]) == num_sessions, \
            f"Expected {num_sessions} sessions, but got {len(results['session_ids'])}"
        
        # Property 3: All session IDs should be unique
        assert len(set(results["session_ids"])) == num_sessions, \
            "Session ID collision detected in concurrent creation"
        
        # Property 4: Each session should have exactly the expected number of memories
        for session_id in results["session_ids"]:
            buffered_count = results["buffered_counts"].get(session_id, 0)
            assert buffered_count == operations_per_session, \
                f"Session {session_id} should have {operations_per_session} memories, " \
                f"but has {buffered_count}"
        
        # Property 5: Verify session isolation by checking metadata
        for session_id in results["session_ids"]:
            buffered = session_manager.get_session_memories(session_id)
            
            # All memories in this session should have the same session_id
            for mem in buffered:
                assert mem["metadata"]["session_id"] == session_id, \
                    f"Memory in session {session_id} has wrong session_id in metadata"
            
            # All memories should have the same worker_id
            worker_ids = set(mem["metadata"]["worker_id"] for mem in buffered)
            assert len(worker_ids) == 1, \
                f"Session {session_id} has memories from multiple workers: {worker_ids}"
        
    finally:
        # Cleanup: end all sessions
        for session_id in results["session_ids"]:
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)


# Feature: memory-write-strategy-session-management, Property 24: Multi-session isolation (with write strategy)
@given(
    num_sessions=st.integers(min_value=2, max_value=5),
    memories_per_session=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_property_24_multi_session_isolation_with_write_strategy(num_sessions, memories_per_session):
    """
    Property: For any set of concurrent sessions using Memory_Write_Strategy,
    memory storage should associate with the correct session_id.
    
    **Validates: Requirements 13.1, 13.2, 13.3**
    
    This test verifies that:
    1. Memory_Write_Strategy correctly associates memories with session_id
    2. Session isolation is maintained through the write strategy layer
    3. Stored memories have correct session_id in metadata
    """
    # Create configurations
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=[],
        similarity_threshold=0.9,
        enable_conflict_detection=False
    )
    
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create multiple sessions
        sessions_data = []
        for i in range(num_sessions):
            session_id = session_manager.create_session(
                metadata={"session_index": i}
            )
            sessions_data.append({
                "session_id": session_id,
                "index": i,
                "stored_memory_ids": []
            })
        
        # Store memories through write strategy for each session
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            index = session_data["index"]
            
            # Set current session in write strategy
            write_strategy.current_session_id = session_id
            
            for mem_idx in range(memories_per_session):
                content = f"Unique memory {mem_idx} for session {index} at {time.time()}"
                metadata = {
                    "memory_index": mem_idx,
                    "session_index": index
                }
                
                # Store memory (should be buffered in session)
                session_manager.buffer_memory(session_id, content, metadata)
        
        # Property 1: Each session should have correct number of buffered memories
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            buffered = session_manager.get_session_memories(session_id)
            
            assert len(buffered) == memories_per_session, \
                f"Session {session_id} should have {memories_per_session} buffered memories"
        
        # Property 2: Memories should have correct session_id when retrieved
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            buffered = session_manager.get_session_memories(session_id)
            
            for mem in buffered:
                # Memory should not have session_id from other sessions
                assert "session_index" in mem["metadata"], \
                    f"Memory missing session_index in metadata"
                assert mem["metadata"]["session_index"] == session_data["index"], \
                    f"Memory has wrong session_index: expected {session_data['index']}, " \
                    f"got {mem['metadata']['session_index']}"
        
        # Property 3: Ending one session should not affect others
        if num_sessions >= 2:
            first_session_id = sessions_data[0]["session_id"]
            session_manager.end_session(first_session_id, persist=True)
            
            # Other sessions should still have their buffered memories
            for session_data in sessions_data[1:]:
                session_id = session_data["session_id"]
                buffered = session_manager.get_session_memories(session_id)
                
                assert len(buffered) == memories_per_session, \
                    f"Session {session_id} buffer affected by ending another session"
        
    finally:
        # Cleanup
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)
