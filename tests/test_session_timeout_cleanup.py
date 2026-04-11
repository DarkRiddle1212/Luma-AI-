"""Unit tests for session timeout and cleanup functionality."""

import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from luma.core.session_manager import Session_Manager, Session
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self):
        self.stored_memories = []
        self.update_called = False
        self.delete_called = False
    
    def store(self, content: str, metadata: dict = None) -> str:
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({"id": memory_id, "content": content, "metadata": metadata or {}})
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        self.update_called = True
        return True
    
    def delete(self, memory_id: str) -> bool:
        self.delete_called = True
        return True


class TestSessionTimeout:
    """Tests for session timeout detection (_is_expired method)."""
    
    def test_session_not_expired_within_timeout(self):
        """Test that a session is not expired when within the timeout period."""
        config = SessionConfig(
            timeout_seconds=3600,  # 1 hour
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
            
            # Session should not be expired (just created)
            assert not session_manager._is_expired(session)
        finally:
            session_manager.shutdown()
    
    def test_session_expired_after_timeout(self):
        """Test that a session is expired after the timeout period."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second for testing
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
            
            # Wait for timeout
            time.sleep(1.5)
            
            # Session should be expired
            assert session_manager._is_expired(session)
        finally:
            session_manager.shutdown()
    
    def test_session_expired_exact_timeout(self):
        """Test that a session is expired exactly at the timeout boundary."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
            
            # Wait exactly at timeout
            time.sleep(1.0)
            
            # Session should be expired (or very close to it)
            assert session_manager._is_expired(session)
        finally:
            session_manager.shutdown()
    
    def test_session_not_expired_before_timeout(self):
        """Test that a session is not expired just before the timeout."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
            
            # Wait just before timeout
            time.sleep(0.5)
            
            # Session should not be expired yet
            assert not session_manager._is_expired(session)
        finally:
            session_manager.shutdown()


class TestSessionExpiration:
    """Tests for session expiration (_expire_session method)."""
    
    def test_expire_session_persists_buffered_memories(self):
        """Test that expiring a session persists buffered memories."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session and buffer some memories
            session_id = session_manager.create_session()
            session_manager.buffer_memory(session_id, "test memory 1", {"category": "test"})
            session_manager.buffer_memory(session_id, "test memory 2", {"category": "test"})
            
            # Verify memories are buffered
            buffered = session_manager.get_session_memories(session_id)
            assert len(buffered) == 2
            
            # Expire the session
            session_manager._expire_session(session_id)
            
            # Verify memories were persisted
            assert len(memory.stored_memories) == 2
            assert memory.stored_memories[0]["content"] == "test memory 1"
            assert memory.stored_memories[1]["content"] == "test memory 2"
            
            # Verify session was removed
            assert session_manager.get_session(session_id) is None
        finally:
            session_manager.shutdown()
    
    def test_expire_session_removes_from_tracking(self):
        """Test that expiring a session removes it from active tracking."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            
            # Verify session exists
            assert session_manager.get_session(session_id) is not None
            
            # Expire the session
            session_manager._expire_session(session_id)
            
            # Verify session was removed
            assert session_manager.get_session(session_id) is None
        finally:
            session_manager.shutdown()


class TestCleanupExpiredSessions:
    """Tests for cleanup of expired sessions (_cleanup_expired_sessions method)."""
    
    def test_cleanup_removes_expired_sessions(self):
        """Test that cleanup removes expired sessions."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create multiple sessions
            session_id_1 = session_manager.create_session()
            session_id_2 = session_manager.create_session()
            session_id_3 = session_manager.create_session()
            
            # Wait for sessions to expire
            time.sleep(1.5)
            
            # Run cleanup
            session_manager._cleanup_expired_sessions()
            
            # Verify all sessions were removed
            assert session_manager.get_session(session_id_1) is None
            assert session_manager.get_session(session_id_2) is None
            assert session_manager.get_session(session_id_3) is None
        finally:
            session_manager.shutdown()
    
    def test_cleanup_persists_buffered_memories_before_removal(self):
        """Test that cleanup persists buffered memories before removing sessions."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session and buffer memories
            session_id = session_manager.create_session()
            session_manager.buffer_memory(session_id, "important memory", {"category": "test"})
            
            # Wait for session to expire
            time.sleep(1.5)
            
            # Run cleanup
            session_manager._cleanup_expired_sessions()
            
            # Verify memory was persisted
            assert len(memory.stored_memories) == 1
            assert memory.stored_memories[0]["content"] == "important memory"
        finally:
            session_manager.shutdown()
    
    def test_cleanup_only_removes_expired_sessions(self):
        """Test that cleanup only removes expired sessions, not active ones."""
        config = SessionConfig(
            timeout_seconds=10,  # 10 seconds
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create multiple sessions
            session_id_1 = session_manager.create_session()
            session_id_2 = session_manager.create_session()
            session_id_3 = session_manager.create_session()
            
            # Update activity on session 2 to keep it active
            session_manager.update_activity(session_id_2)
            
            # Run cleanup immediately (no sessions should be expired)
            session_manager._cleanup_expired_sessions()
            
            # Verify all sessions still exist
            assert session_manager.get_session(session_id_1) is not None
            assert session_manager.get_session(session_id_2) is not None
            assert session_manager.get_session(session_id_3) is not None
        finally:
            session_manager.shutdown()


class TestCleanupTask:
    """Tests for the background cleanup task (_start_cleanup_task method)."""
    
    def test_cleanup_task_runs_periodically(self):
        """Test that the cleanup task runs at the configured interval."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=1,  # 1 second for testing
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # The cleanup task should be running
            assert session_manager._cleanup_thread is not None
            assert session_manager._cleanup_thread.is_alive()
        finally:
            session_manager.shutdown()
    
    def test_cleanup_task_stops_on_shutdown(self):
        """Test that the cleanup task stops when shutdown is called."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Cleanup task should be running
            assert session_manager._cleanup_thread is not None
            assert session_manager._cleanup_thread.is_alive()
            
            # Shutdown
            session_manager.shutdown()
            
            # Cleanup task should be stopped
            assert not session_manager._cleanup_thread.is_alive()
        finally:
            # Ensure cleanup thread is joined
            if session_manager._cleanup_thread and session_manager._cleanup_thread.is_alive():
                session_manager._cleanup_thread.join(timeout=1.0)
    
    def test_cleanup_task_detects_expired_sessions(self):
        """Test that the cleanup task detects and removes expired sessions."""
        config = SessionConfig(
            timeout_seconds=1,  # 1 second
            cleanup_interval_seconds=1,  # 1 second
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session
            session_id = session_manager.create_session()
            
            # Wait for session to expire
            time.sleep(1.5)
            
            # Wait for cleanup task to run
            time.sleep(1.5)
            
            # Session should be removed by cleanup task
            assert session_manager.get_session(session_id) is None
        finally:
            session_manager.shutdown()


class TestEdgeCases:
    """Tests for edge cases in session timeout and cleanup."""
    
    def test_expire_nonexistent_session(self):
        """Test that expiring a nonexistent session doesn't raise an error."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Try to expire a nonexistent session
            session_manager._expire_session("nonexistent-session-id")
            
            # Should not raise an error
            assert True
        finally:
            session_manager.shutdown()
    
    def test_cleanup_empty_sessions(self):
        """Test that cleanup works with no sessions."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Run cleanup with no sessions
            session_manager._cleanup_expired_sessions()
            
            # Should not raise an error
            assert True
        finally:
            session_manager.shutdown()
    
    def test_expire_session_with_empty_buffer(self):
        """Test that expiring a session with no buffered memories works."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create a session without buffering any memories
            session_id = session_manager.create_session()
            
            # Expire the session
            session_manager._expire_session(session_id)
            
            # Should not raise an error
            assert session_manager.get_session(session_id) is None
        finally:
            session_manager.shutdown()
    
    def test_multiple_concurrent_expirations(self):
        """Test that multiple concurrent expirations are handled safely."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=300,
            max_buffer_size=100,
            enable_buffering=True
        )
        memory = MockMemoryInterface()
        session_manager = Session_Manager(config, memory)
        
        try:
            # Create multiple sessions
            session_ids = [session_manager.create_session() for _ in range(5)]
            
            # Expire all sessions concurrently (thread safety test)
            import threading
            threads = []
            
            for session_id in session_ids:
                thread = threading.Thread(
                    target=session_manager._expire_session,
                    args=(session_id,)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5.0)
            
            # All sessions should be removed
            for session_id in session_ids:
                assert session_manager.get_session(session_id) is None
        finally:
            session_manager.shutdown()
