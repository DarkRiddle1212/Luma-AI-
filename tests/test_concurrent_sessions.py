"""
Concurrent unit tests for Session_Manager.

Tests thread-safety and concurrent session support including:
- Multiple threads creating sessions simultaneously
- Concurrent memory storage across sessions
- Thread-safe buffer operations
- Session state isolation between concurrent sessions

Feature: memory-write-strategy-session-management
Requirements: 13.1, 13.2, 13.3, 13.4
"""

import threading
import time
from typing import List, Dict, Any
from datetime import datetime

import pytest

from luma.core.session_manager import Session_Manager, Session
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


class ThreadSafeMemoryInterface(MemoryInterface):
    """Thread-safe mock memory interface for concurrent testing."""
    
    def __init__(self):
        self.stored_memories: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.store_count = 0
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Thread-safe store operation."""
        with self.lock:
            memory_id = f"mem_{self.store_count}"
            self.store_count += 1
            self.stored_memories.append({
                "id": memory_id,
                "content": content,
                "metadata": metadata or {},
                "thread_id": threading.current_thread().ident
            })
            # Simulate some processing time
            time.sleep(0.001)
            return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Thread-safe retrieve operation."""
        with self.lock:
            return {"memories": self.stored_memories.copy()}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        return True
    
    def delete(self, memory_id: str) -> bool:
        return True
    
    def get_store_count(self) -> int:
        """Get the total number of store operations."""
        with self.lock:
            return self.store_count


class TestConcurrentSessionCreation:
    """Test multiple threads creating sessions simultaneously."""
    
    def test_concurrent_session_creation_no_collisions(self):
        """
        Test that multiple threads can create sessions simultaneously without ID collisions.
        
        Requirements: 13.1, 13.5
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            num_threads = 10
            sessions_per_thread = 5
            created_session_ids: List[str] = []
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def create_sessions(thread_id: int):
                """Create multiple sessions in a thread."""
                try:
                    for i in range(sessions_per_thread):
                        session_id = session_manager.create_session(
                            metadata={"thread_id": thread_id, "index": i}
                        )
                        with lock:
                            created_session_ids.append(session_id)
                        # Small delay to increase chance of race conditions
                        time.sleep(0.001)
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=create_sessions, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent session creation: {errors}"
            
            # Verify all sessions were created
            expected_count = num_threads * sessions_per_thread
            assert len(created_session_ids) == expected_count, \
                f"Expected {expected_count} sessions, got {len(created_session_ids)}"
            
            # Verify all session IDs are unique (no collisions)
            unique_ids = set(created_session_ids)
            assert len(unique_ids) == expected_count, \
                f"Session ID collisions detected: {expected_count} sessions but only {len(unique_ids)} unique IDs"
            
            # Verify all sessions can be retrieved
            for session_id in created_session_ids:
                session = session_manager.get_session(session_id)
                assert session is not None, f"Session {session_id} not found"
                assert session.session_id == session_id
        
        finally:
            session_manager.shutdown()
    
    def test_concurrent_session_creation_with_activity(self):
        """
        Test concurrent session creation with immediate activity updates.
        
        Requirements: 13.1, 13.4
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            num_threads = 8
            created_sessions: List[Dict[str, Any]] = []
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def create_and_use_session(thread_id: int):
                """Create a session and immediately update activity."""
                try:
                    session_id = session_manager.create_session(
                        metadata={"thread_id": thread_id}
                    )
                    
                    # Immediately update activity multiple times
                    for i in range(5):
                        session_manager.update_activity(session_id)
                        time.sleep(0.001)
                    
                    # Get final session state
                    session = session_manager.get_session(session_id)
                    with lock:
                        created_sessions.append({
                            "session_id": session_id,
                            "message_count": session.message_count,
                            "thread_id": thread_id
                        })
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=create_and_use_session, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
            
            # Verify all sessions were created and updated
            assert len(created_sessions) == num_threads
            
            # Verify each session has correct message count
            for session_info in created_sessions:
                assert session_info["message_count"] == 5, \
                    f"Session {session_info['session_id']} has incorrect message count: {session_info['message_count']}"
        
        finally:
            session_manager.shutdown()


class TestConcurrentMemoryStorage:
    """Test concurrent memory storage across sessions."""
    
    def test_concurrent_memory_buffering_across_sessions(self):
        """
        Test that multiple threads can buffer memories in different sessions concurrently.
        
        Requirements: 13.1, 13.2, 13.3
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            num_threads = 10
            memories_per_thread = 10
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def buffer_memories(thread_id: int):
                """Create a session and buffer memories."""
                try:
                    # Each thread creates its own session
                    session_id = session_manager.create_session(
                        metadata={"thread_id": thread_id}
                    )
                    
                    # Buffer multiple memories
                    for i in range(memories_per_thread):
                        session_manager.buffer_memory(
                            session_id=session_id,
                            content=f"Memory from thread {thread_id}, index {i}",
                            metadata={
                                "thread_id": thread_id,
                                "index": i,
                                "category": "test"
                            }
                        )
                        # Small delay to increase concurrency
                        time.sleep(0.001)
                    
                    # Verify buffer size
                    buffered = session_manager.get_session_memories(session_id)
                    assert len(buffered) == memories_per_thread, \
                        f"Thread {thread_id}: Expected {memories_per_thread} buffered memories, got {len(buffered)}"
                
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=buffer_memories, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent buffering: {errors}"
            
            # Verify no memories were persisted yet (all buffered)
            assert memory.get_store_count() == 0, \
                "Memories should be buffered, not persisted"
        
        finally:
            session_manager.shutdown()
    
    def test_concurrent_session_end_with_persistence(self):
        """
        Test that multiple threads can end sessions and persist memories concurrently.
        
        Requirements: 13.1, 13.2, 13.3
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            num_threads = 8
            memories_per_session = 5
            persisted_counts: List[int] = []
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def create_buffer_and_end_session(thread_id: int):
                """Create session, buffer memories, and end session."""
                try:
                    # Create session
                    session_id = session_manager.create_session(
                        metadata={"thread_id": thread_id}
                    )
                    
                    # Buffer memories
                    for i in range(memories_per_session):
                        session_manager.buffer_memory(
                            session_id=session_id,
                            content=f"Memory from thread {thread_id}, index {i}",
                            metadata={"thread_id": thread_id, "index": i}
                        )
                    
                    # Small delay before ending
                    time.sleep(0.01)
                    
                    # End session and persist
                    count = session_manager.end_session(session_id, persist=True)
                    with lock:
                        persisted_counts.append(count)
                
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=create_buffer_and_end_session, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent session ending: {errors}"
            
            # Verify all memories were persisted
            assert len(persisted_counts) == num_threads
            assert all(count == memories_per_session for count in persisted_counts), \
                f"Not all memories were persisted correctly: {persisted_counts}"
            
            # Verify total persisted count
            total_expected = num_threads * memories_per_session
            assert memory.get_store_count() == total_expected, \
                f"Expected {total_expected} persisted memories, got {memory.get_store_count()}"
        
        finally:
            session_manager.shutdown()


class TestThreadSafeBufferOperations:
    """Test thread-safe buffer operations."""
    
    def test_concurrent_buffer_operations_same_session(self):
        """
        Test that multiple threads can safely buffer memories to the same session.
        
        This tests the RLock protection for buffer operations.
        
        Requirements: 13.4
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=1000,  # Large buffer to avoid overflow
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a single session
            session_id = session_manager.create_session(metadata={"test": "concurrent_buffer"})
            
            num_threads = 10
            memories_per_thread = 10
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def buffer_to_shared_session(thread_id: int):
                """Buffer memories to the shared session."""
                try:
                    for i in range(memories_per_thread):
                        session_manager.buffer_memory(
                            session_id=session_id,
                            content=f"Memory from thread {thread_id}, index {i}",
                            metadata={"thread_id": thread_id, "index": i}
                        )
                        # Very small delay to maximize concurrency
                        time.sleep(0.0001)
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=buffer_to_shared_session, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent buffer operations: {errors}"
            
            # Verify all memories were buffered
            buffered = session_manager.get_session_memories(session_id)
            expected_count = num_threads * memories_per_thread
            assert len(buffered) == expected_count, \
                f"Expected {expected_count} buffered memories, got {len(buffered)}"
            
            # Verify each thread's memories are present
            thread_counts = {}
            for mem in buffered:
                thread_id = mem["metadata"]["thread_id"]
                thread_counts[thread_id] = thread_counts.get(thread_id, 0) + 1
            
            assert len(thread_counts) == num_threads, \
                f"Not all threads contributed memories: {thread_counts}"
            
            for thread_id, count in thread_counts.items():
                assert count == memories_per_thread, \
                    f"Thread {thread_id} has {count} memories, expected {memories_per_thread}"
        
        finally:
            session_manager.shutdown()
    
    def test_concurrent_buffer_overflow_handling(self):
        """
        Test that buffer overflow is handled correctly under concurrent load.
        
        Requirements: 13.4
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=20,  # Small buffer to trigger overflow
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a single session
            session_id = session_manager.create_session(metadata={"test": "overflow"})
            
            num_threads = 5
            memories_per_thread = 10
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def buffer_with_overflow(thread_id: int):
                """Buffer memories that will trigger overflow."""
                try:
                    for i in range(memories_per_thread):
                        session_manager.buffer_memory(
                            session_id=session_id,
                            content=f"Memory from thread {thread_id}, index {i}",
                            metadata={"thread_id": thread_id, "index": i}
                        )
                        time.sleep(0.001)
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=buffer_with_overflow, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent overflow handling: {errors}"
            
            # Verify buffer size is within limits
            buffered = session_manager.get_session_memories(session_id)
            assert len(buffered) <= config.max_buffer_size, \
                f"Buffer size {len(buffered)} exceeds max {config.max_buffer_size}"
            
            # Verify some memories were flushed to storage
            assert memory.get_store_count() > 0, \
                "Expected some memories to be flushed due to overflow"
            
            # Verify total memories (buffered + persisted) equals what was added
            total_added = num_threads * memories_per_thread
            total_stored = len(buffered) + memory.get_store_count()
            assert total_stored == total_added, \
                f"Memory loss detected: added {total_added}, stored {total_stored}"
        
        finally:
            session_manager.shutdown()


class TestSessionIsolation:
    """Test session state isolation between concurrent sessions."""
    
    def test_concurrent_sessions_maintain_isolation(self):
        """
        Test that concurrent sessions maintain separate state with no interference.
        
        Requirements: 13.1, 13.2, 13.3
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            num_threads = 8
            session_data: List[Dict[str, Any]] = []
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def create_isolated_session(thread_id: int):
                """Create a session with unique data and verify isolation."""
                try:
                    # Create session with unique metadata
                    session_id = session_manager.create_session(
                        metadata={"thread_id": thread_id, "unique_value": thread_id * 100}
                    )
                    
                    # Buffer unique memories
                    for i in range(5):
                        session_manager.buffer_memory(
                            session_id=session_id,
                            content=f"Unique content for thread {thread_id}, message {i}",
                            metadata={
                                "thread_id": thread_id,
                                "message_index": i,
                                "unique_marker": f"thread_{thread_id}_msg_{i}"
                            }
                        )
                        session_manager.update_activity(session_id)
                        time.sleep(0.001)
                    
                    # Retrieve session state
                    session = session_manager.get_session(session_id)
                    buffered = session_manager.get_session_memories(session_id)
                    
                    with lock:
                        session_data.append({
                            "thread_id": thread_id,
                            "session_id": session_id,
                            "metadata": session.metadata,
                            "message_count": session.message_count,
                            "buffered_count": len(buffered),
                            "buffered_memories": buffered
                        })
                
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=create_isolated_session, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent session operations: {errors}"
            
            # Verify all sessions were created
            assert len(session_data) == num_threads
            
            # Verify each session has unique data
            session_ids = set()
            for data in session_data:
                # Check session ID is unique
                assert data["session_id"] not in session_ids, \
                    f"Duplicate session ID: {data['session_id']}"
                session_ids.add(data["session_id"])
                
                # Check metadata matches thread
                assert data["metadata"]["thread_id"] == data["thread_id"], \
                    f"Metadata mismatch for thread {data['thread_id']}"
                assert data["metadata"]["unique_value"] == data["thread_id"] * 100, \
                    f"Unique value mismatch for thread {data['thread_id']}"
                
                # Check message count
                assert data["message_count"] == 5, \
                    f"Thread {data['thread_id']} has incorrect message count: {data['message_count']}"
                
                # Check buffered memories count
                assert data["buffered_count"] == 5, \
                    f"Thread {data['thread_id']} has incorrect buffered count: {data['buffered_count']}"
                
                # Verify all buffered memories belong to this thread
                for mem in data["buffered_memories"]:
                    assert mem["metadata"]["thread_id"] == data["thread_id"], \
                        f"Memory from wrong thread found in session {data['session_id']}"
                    assert f"thread_{data['thread_id']}_" in mem["metadata"]["unique_marker"], \
                        f"Memory marker mismatch in session {data['session_id']}"
        
        finally:
            session_manager.shutdown()
    
    def test_concurrent_get_session_operations(self):
        """
        Test that concurrent get_session operations are thread-safe.
        
        Requirements: 13.4
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = ThreadSafeMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create multiple sessions
            num_sessions = 10
            session_ids = []
            for i in range(num_sessions):
                session_id = session_manager.create_session(
                    metadata={"session_index": i}
                )
                session_ids.append(session_id)
            
            num_threads = 20
            read_results: List[bool] = []
            errors: List[Exception] = []
            lock = threading.Lock()
            
            def read_sessions(thread_id: int):
                """Read all sessions multiple times."""
                try:
                    all_found = True
                    for _ in range(10):
                        for session_id in session_ids:
                            session = session_manager.get_session(session_id)
                            if session is None:
                                all_found = False
                        time.sleep(0.001)
                    
                    with lock:
                        read_results.append(all_found)
                
                except Exception as e:
                    with lock:
                        errors.append(e)
            
            # Create and start threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=read_sessions, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors occurred
            assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
            
            # Verify all reads were successful
            assert len(read_results) == num_threads
            assert all(read_results), "Some sessions were not found during concurrent reads"
        
        finally:
            session_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
