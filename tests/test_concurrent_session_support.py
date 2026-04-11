"""
Concurrent Unit Tests for Session Support

This module implements unit tests to verify thread-safe concurrent session
operations including session creation, memory storage, and buffer operations.

Feature: memory-write-strategy-session-management
Validates: Requirements 13.1, 13.2, 13.3, 13.4
"""

import pytest
import threading
import time
from typing import List, Dict, Set

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
        self.store_count = 0
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method with thread safety."""
        with self.lock:
            self.store_count += 1
            memory_id = f"mem_{self.store_count}"
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
# Concurrent Unit Tests
# ============================================================================

class TestConcurrentSessionCreation:
    """Tests for concurrent session creation."""
    
    def test_multiple_threads_creating_sessions_simultaneously(self):
        """
        Test that multiple threads can create sessions simultaneously without
        race conditions or ID collisions.
        
        **Validates: Requirements 13.1, 13.4, 13.5**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        num_threads = 10
        sessions_per_thread = 5
        results = {"session_ids": [], "errors": []}
        results_lock = threading.Lock()
        
        def create_sessions_worker(worker_id: int):
            """Worker that creates multiple sessions."""
            try:
                for i in range(sessions_per_thread):
                    session_id = session_manager.create_session(
                        metadata={"worker_id": worker_id, "session_index": i}
                    )
                    with results_lock:
                        results["session_ids"].append(session_id)
                    # Small delay to increase interleaving
                    time.sleep(0.001)
            except Exception as e:
                with results_lock:
                    results["errors"].append((worker_id, str(e)))
        
        try:
            # Create and start threads
            threads = []
            for worker_id in range(num_threads):
                thread = threading.Thread(target=create_sessions_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
                assert not thread.is_alive(), "Thread did not complete in time"
            
            # Verify results
            assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
            assert len(results["session_ids"]) == num_threads * sessions_per_thread
            assert len(set(results["session_ids"])) == num_threads * sessions_per_thread, \
                "Session ID collision detected"
            
            # Verify all sessions exist and have correct metadata
            for session_id in results["session_ids"]:
                session = session_manager.get_session(session_id)
                assert session is not None, f"Session {session_id} not found"
                assert "worker_id" in session.metadata
                assert "session_index" in session.metadata
        
        finally:
            # Cleanup
            for session_id in results["session_ids"]:
                if session_manager.get_session(session_id) is not None:
                    session_manager.end_session(session_id, persist=False)
    
    def test_concurrent_session_creation_with_same_metadata(self):
        """
        Test that concurrent sessions with identical metadata remain isolated.
        
        **Validates: Requirements 13.1, 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        num_threads = 5
        shared_metadata = {"user_id": "test_user", "device": "test_device"}
        results = {"session_ids": []}
        results_lock = threading.Lock()
        
        def create_session_worker():
            """Worker that creates a session with shared metadata."""
            session_id = session_manager.create_session(metadata=shared_metadata.copy())
            with results_lock:
                results["session_ids"].append(session_id)
        
        try:
            # Create and start threads
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=create_session_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=5.0)
            
            # Verify all sessions are unique despite same metadata
            assert len(results["session_ids"]) == num_threads
            assert len(set(results["session_ids"])) == num_threads, \
                "Sessions with same metadata should still have unique IDs"
        
        finally:
            # Cleanup
            for session_id in results["session_ids"]:
                if session_manager.get_session(session_id) is not None:
                    session_manager.end_session(session_id, persist=False)


class TestConcurrentMemoryStorage:
    """Tests for concurrent memory storage across sessions."""
    
    def test_concurrent_memory_storage_across_sessions(self):
        """
        Test that multiple threads can store memories in different sessions
        concurrently without interference.
        
        **Validates: Requirements 13.1, 13.2, 13.3, 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        num_sessions = 5
        memories_per_session = 10
        results = {"session_data": {}, "errors": []}
        results_lock = threading.Lock()
        
        def storage_worker(worker_id: int):
            """Worker that creates a session and stores memories."""
            try:
                # Create session
                session_id = session_manager.create_session(
                    metadata={"worker_id": worker_id}
                )
                
                with results_lock:
                    results["session_data"][session_id] = {
                        "worker_id": worker_id,
                        "expected_count": memories_per_session
                    }
                
                # Store memories
                for i in range(memories_per_session):
                    content = f"Worker {worker_id} memory {i}"
                    metadata = {
                        "session_id": session_id,
                        "worker_id": worker_id,
                        "memory_index": i
                    }
                    session_manager.buffer_memory(session_id, content, metadata)
                    time.sleep(0.001)  # Small delay
            
            except Exception as e:
                with results_lock:
                    results["errors"].append((worker_id, str(e)))
        
        try:
            # Create and start threads
            threads = []
            for worker_id in range(num_sessions):
                thread = threading.Thread(target=storage_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
                assert not thread.is_alive(), "Thread did not complete in time"
            
            # Verify results
            assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
            assert len(results["session_data"]) == num_sessions
            
            # Verify each session has correct number of memories
            for session_id, data in results["session_data"].items():
                buffered = session_manager.get_session_memories(session_id)
                assert len(buffered) == memories_per_session, \
                    f"Session {session_id} should have {memories_per_session} memories"
                
                # Verify all memories belong to correct worker
                for mem in buffered:
                    assert mem["metadata"]["worker_id"] == data["worker_id"], \
                        f"Memory in session {session_id} has wrong worker_id"
                    assert mem["metadata"]["session_id"] == session_id, \
                        f"Memory has wrong session_id"
        
        finally:
            # Cleanup
            for session_id in results["session_data"].keys():
                if session_manager.get_session(session_id) is not None:
                    session_manager.end_session(session_id, persist=False)
    
    def test_concurrent_storage_in_same_session(self):
        """
        Test that multiple threads can store memories in the same session
        concurrently with proper synchronization.
        
        **Validates: Requirements 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        # Create a single session
        session_id = session_manager.create_session()
        
        num_threads = 5
        memories_per_thread = 10
        results = {"errors": [], "stored_count": 0}
        results_lock = threading.Lock()
        
        def storage_worker(worker_id: int):
            """Worker that stores memories in the shared session."""
            try:
                for i in range(memories_per_thread):
                    content = f"Worker {worker_id} memory {i}"
                    metadata = {
                        "worker_id": worker_id,
                        "memory_index": i
                    }
                    session_manager.buffer_memory(session_id, content, metadata)
                    
                    with results_lock:
                        results["stored_count"] += 1
                    
                    time.sleep(0.001)
            
            except Exception as e:
                with results_lock:
                    results["errors"].append((worker_id, str(e)))
        
        try:
            # Create and start threads
            threads = []
            for worker_id in range(num_threads):
                thread = threading.Thread(target=storage_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
                assert not thread.is_alive(), "Thread did not complete in time"
            
            # Verify results
            assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
            
            # Verify all memories were stored
            buffered = session_manager.get_session_memories(session_id)
            expected_total = num_threads * memories_per_thread
            assert len(buffered) == expected_total, \
                f"Expected {expected_total} memories, got {len(buffered)}"
            
            # Verify all worker IDs are present
            worker_ids = set(mem["metadata"]["worker_id"] for mem in buffered)
            assert len(worker_ids) == num_threads, \
                f"Expected memories from {num_threads} workers"
        
        finally:
            # Cleanup
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)


class TestThreadSafeBufferOperations:
    """Tests for thread-safe buffer operations."""
    
    def test_concurrent_buffer_and_retrieval(self):
        """
        Test that concurrent buffer writes and reads maintain consistency.
        
        **Validates: Requirements 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        session_id = session_manager.create_session()
        
        num_writers = 3
        num_readers = 2
        operations_per_thread = 20
        results = {"write_errors": [], "read_errors": [], "read_counts": []}
        results_lock = threading.Lock()
        
        def writer_worker(worker_id: int):
            """Worker that writes to buffer."""
            try:
                for i in range(operations_per_thread):
                    content = f"Writer {worker_id} memory {i}"
                    metadata = {"writer_id": worker_id, "index": i}
                    session_manager.buffer_memory(session_id, content, metadata)
                    time.sleep(0.001)
            except Exception as e:
                with results_lock:
                    results["write_errors"].append((worker_id, str(e)))
        
        def reader_worker(worker_id: int):
            """Worker that reads from buffer."""
            try:
                for _ in range(operations_per_thread):
                    buffered = session_manager.get_session_memories(session_id)
                    with results_lock:
                        results["read_counts"].append(len(buffered))
                    time.sleep(0.001)
            except Exception as e:
                with results_lock:
                    results["read_errors"].append((worker_id, str(e)))
        
        try:
            # Create and start threads
            threads = []
            
            # Start writers
            for worker_id in range(num_writers):
                thread = threading.Thread(target=writer_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Start readers
            for worker_id in range(num_readers):
                thread = threading.Thread(target=reader_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=15.0)
                assert not thread.is_alive(), "Thread did not complete in time"
            
            # Verify no errors
            assert len(results["write_errors"]) == 0, \
                f"Write errors occurred: {results['write_errors']}"
            assert len(results["read_errors"]) == 0, \
                f"Read errors occurred: {results['read_errors']}"
            
            # Verify final buffer state
            final_buffered = session_manager.get_session_memories(session_id)
            expected_total = num_writers * operations_per_thread
            assert len(final_buffered) == expected_total, \
                f"Expected {expected_total} memories in final buffer"
            
            # Verify read counts are monotonically increasing or stable
            # (reads should never see fewer memories than before)
            for i in range(1, len(results["read_counts"])):
                assert results["read_counts"][i] >= results["read_counts"][i-1] or \
                       results["read_counts"][i] == 0, \
                    "Buffer count decreased unexpectedly"
        
        finally:
            # Cleanup
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)
    
    def test_concurrent_buffer_overflow_handling(self):
        """
        Test that buffer overflow is handled correctly under concurrent load.
        
        **Validates: Requirements 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=20,  # Small buffer to trigger overflow
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        session_id = session_manager.create_session()
        
        num_threads = 5
        memories_per_thread = 10
        results = {"errors": []}
        results_lock = threading.Lock()
        
        def storage_worker(worker_id: int):
            """Worker that stores memories to trigger overflow."""
            try:
                for i in range(memories_per_thread):
                    content = f"Worker {worker_id} memory {i}"
                    metadata = {"worker_id": worker_id, "index": i}
                    session_manager.buffer_memory(session_id, content, metadata)
                    time.sleep(0.001)
            except Exception as e:
                with results_lock:
                    results["errors"].append((worker_id, str(e)))
        
        try:
            # Create and start threads
            threads = []
            for worker_id in range(num_threads):
                thread = threading.Thread(target=storage_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)
                assert not thread.is_alive(), "Thread did not complete in time"
            
            # Verify no errors occurred
            assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
            
            # Verify buffer size is within limits
            buffered = session_manager.get_session_memories(session_id)
            assert len(buffered) <= config.max_buffer_size, \
                f"Buffer size {len(buffered)} exceeds max {config.max_buffer_size}"
            
            # Verify some memories were flushed to storage
            stored = memory.retrieve()["memories"]
            total_memories = len(buffered) + len(stored)
            expected_total = num_threads * memories_per_thread
            assert total_memories == expected_total, \
                f"Expected {expected_total} total memories, got {total_memories}"
        
        finally:
            # Cleanup
            if session_manager.get_session(session_id) is not None:
                session_manager.end_session(session_id, persist=False)
    
    def test_concurrent_session_end_operations(self):
        """
        Test that concurrent session end operations are handled safely.
        
        **Validates: Requirements 13.4**
        """
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        num_sessions = 10
        memories_per_session = 5
        results = {"session_ids": [], "errors": [], "persisted_counts": {}}
        results_lock = threading.Lock()
        
        def session_lifecycle_worker(worker_id: int):
            """Worker that creates session, stores memories, and ends session."""
            try:
                # Create session
                session_id = session_manager.create_session(
                    metadata={"worker_id": worker_id}
                )
                
                with results_lock:
                    results["session_ids"].append(session_id)
                
                # Store memories
                for i in range(memories_per_session):
                    content = f"Worker {worker_id} memory {i}"
                    metadata = {"worker_id": worker_id, "index": i}
                    session_manager.buffer_memory(session_id, content, metadata)
                
                # End session
                persisted_count = session_manager.end_session(session_id, persist=True)
                
                with results_lock:
                    results["persisted_counts"][session_id] = persisted_count
            
            except Exception as e:
                with results_lock:
                    results["errors"].append((worker_id, str(e)))
        
        # Create and start threads
        threads = []
        for worker_id in range(num_sessions):
            thread = threading.Thread(target=session_lifecycle_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
            assert not thread.is_alive(), "Thread did not complete in time"
        
        # Verify results
        assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
        assert len(results["session_ids"]) == num_sessions
        
        # Verify all sessions were ended
        for session_id in results["session_ids"]:
            assert session_manager.get_session(session_id) is None, \
                f"Session {session_id} should be ended"
        
        # Verify all memories were persisted
        for session_id, persisted_count in results["persisted_counts"].items():
            assert persisted_count == memories_per_session, \
                f"Session {session_id} should have persisted {memories_per_session} memories"
        
        # Verify total stored memories
        stored = memory.retrieve()["memories"]
        expected_total = num_sessions * memories_per_session
        assert len(stored) == expected_total, \
            f"Expected {expected_total} stored memories, got {len(stored)}"
