"""
Unit tests for logging in Memory Write Strategy and Session Management.

Tests that log messages are generated for key operations with appropriate
log levels and required context.

Feature: memory-write-strategy-session-management
Requirements: 12.1, 12.2, 12.3, 12.4, 12.5
"""

import pytest
import logging
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch
from luma.core.write_strategy import (
    WriteStrategyConfig,
    Memory_Write_Strategy
)
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


# ============================================================================
# Mock Memory Interface
# ============================================================================


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self, should_fail=False):
        self.stored_memories = []
        self.next_id = 0
        self.should_fail = should_fail
        self.default_category = None
        self.default_tags = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        if self.should_fail:
            raise MemoryStorageError("Simulated storage failure")
        
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        return True
    
    def delete(self, memory_id: str) -> bool:
        return True


# ============================================================================
# Memory_Write_Strategy Logging Tests
# ============================================================================


class TestMemoryWriteStrategyLogging:
    """Test logging in Memory_Write_Strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(
                timeout_seconds=3600,
                cleanup_interval_seconds=3600
            ),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_storage_success_logging(self, strategy, caplog):
        """
        Test that successful storage logs with INFO level and required context.
        
        Requirement 12.1: Log memory storage operations (memory_id, content length, category, tags)
        """
        caplog.set_level(logging.INFO)
        
        # Store a memory with immediate persistence
        memory_id = strategy.store_memory(
            "Test content for storage",
            metadata={"category": "test", "tags": ["tag1", "tag2"]},
            immediate=True
        )
        
        # Find the storage success log
        storage_logs = [r for r in caplog.records if "Memory stored immediately" in r.message]
        assert len(storage_logs) > 0, "Should have storage success log"
        
        log_record = storage_logs[0]
        assert log_record.levelname == "INFO"
        assert log_record.memory_id == memory_id
        assert log_record.content_length == len("Test content for storage")
        assert log_record.category == "test"
        assert log_record.tags == ["tag1", "tag2"]
    
    def test_rejection_trivial_logging(self, strategy, caplog):
        """
        Test that trivial pattern rejection logs with INFO level.
        
        Requirement 12.2: Log rejection reasons (trivial, duplicate, repetitive)
        """
        caplog.set_level(logging.INFO)
        
        # Try to store a trivial message
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("hello")
        
        # Check for rejection log
        rejection_logs = [r for r in caplog.records if "trivial pattern" in r.message.lower()]
        assert len(rejection_logs) > 0, "Should have trivial rejection log"
        
        log_record = rejection_logs[0]
        assert log_record.levelname == "INFO"
        assert log_record.reason == "trivial_pattern"
        assert hasattr(log_record, 'matched_pattern')
    
    def test_rejection_empty_logging(self, strategy, caplog):
        """
        Test that empty content rejection logs with INFO level.
        
        Requirement 12.2: Log rejection reasons (trivial, duplicate, repetitive)
        """
        caplog.set_level(logging.INFO)
        
        # Try to store empty content
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("")
        
        # Check for rejection log
        rejection_logs = [r for r in caplog.records if "empty or whitespace" in r.message.lower()]
        assert len(rejection_logs) > 0, "Should have empty rejection log"
        
        log_record = rejection_logs[0]
        assert log_record.levelname == "INFO"
        assert log_record.reason == "empty_or_whitespace"
    
    def test_rejection_repetitive_logging(self, strategy, caplog):
        """
        Test that repetitive content rejection logs with INFO level.
        
        Requirement 12.2: Log rejection reasons (trivial, duplicate, repetitive)
        """
        caplog.set_level(logging.INFO)
        
        # Store a message first
        strategy.store_memory("Unique message", immediate=True)
        
        caplog.clear()
        
        # Try to store the same message again
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("Unique message")
        
        # Check for repetitive rejection log
        rejection_logs = [r for r in caplog.records if "repetitive" in r.message.lower()]
        assert len(rejection_logs) > 0, "Should have repetitive rejection log"
        
        log_record = rejection_logs[0]
        assert log_record.levelname == "INFO"
        assert log_record.reason == "repetitive"
    
    def test_storage_failure_logging(self, strategy, caplog):
        """
        Test that storage failures log with ERROR level and exception details.
        
        Requirement 12.3: Log storage failures with full exception details
        """
        caplog.set_level(logging.ERROR)
        
        # Replace memory interface with one that fails
        strategy.memory = MockMemoryInterface(should_fail=True)
        
        # Try to store a memory
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("Test content", immediate=True)
        
        # Check for error log
        error_logs = [r for r in caplog.records if r.levelname == "ERROR" and "storage failed" in r.message.lower()]
        assert len(error_logs) > 0, "Should have storage failure log"
        
        log_record = error_logs[0]
        assert log_record.levelname == "ERROR"
        assert log_record.exc_info is not None, "Should include exception info"
        assert hasattr(log_record, 'content_length')
    
    def test_validation_failure_logging(self, strategy, caplog):
        """
        Test that validation failures log with WARNING level.
        
        Requirement 12.3: Log validation failures
        """
        caplog.set_level(logging.WARNING)
        
        # Try to store invalid content (non-string)
        with pytest.raises(MemoryStorageError):
            strategy.validate_content(123)  # type: ignore
        
        # Check for validation warning log
        warning_logs = [r for r in caplog.records if r.levelname == "WARNING" and "validation failed" in r.message.lower()]
        assert len(warning_logs) > 0, "Should have validation failure log"
        
        log_record = warning_logs[0]
        assert log_record.levelname == "WARNING"
        assert hasattr(log_record, 'validation_error')
    
    def test_duplicate_detection_logging(self, strategy, caplog):
        """
        Test that duplicate detection logs with INFO level.
        
        Requirement 12.2: Log rejection reasons (duplicate)
        """
        caplog.set_level(logging.INFO)
        
        # Store a memory first
        memory_id = strategy.store_memory(
            "Duplicate test content",
            metadata={"category": "test"},
            immediate=True
        )
        
        # Clear recent messages to avoid repetition detection
        strategy.recent_messages.clear()
        
        caplog.clear()
        
        # Try to store duplicate
        duplicate_id = strategy.store_memory(
            "Duplicate test content",
            metadata={"category": "test"},
            immediate=True
        )
        
        # Should return same ID
        assert duplicate_id == memory_id
        
        # Check for duplicate detection log
        duplicate_logs = [r for r in caplog.records if "duplicate" in r.message.lower()]
        assert len(duplicate_logs) > 0, "Should have duplicate detection log"
        
        log_record = [r for r in duplicate_logs if "Exact duplicate detected" in r.message][0]
        assert log_record.levelname == "INFO"
        assert log_record.memory_id == memory_id
    
    def test_conflict_detection_logging(self, strategy, caplog):
        """
        Test that conflict detection logs with INFO level.
        
        Requirement 12.1: Log memory operations including conflicts
        """
        caplog.set_level(logging.INFO)
        
        # Store a memory first
        strategy.store_memory(
            "User likes Python programming",
            metadata={"category": "preferences"},
            immediate=True
        )
        
        # Clear recent messages to avoid repetition detection
        strategy.recent_messages.clear()
        
        caplog.clear()
        
        # Store a conflicting memory
        strategy.store_memory(
            "User doesn't like Python programming",
            metadata={"category": "preferences"},
            immediate=True
        )
        
        # Check for conflict detection log
        conflict_logs = [r for r in caplog.records if "conflict detected" in r.message.lower()]
        assert len(conflict_logs) > 0, "Should have conflict detection log"
        
        # Find the "Potential conflict detected" log which has the extra fields
        potential_conflict_logs = [r for r in conflict_logs if "Potential conflict detected" in r.message]
        assert len(potential_conflict_logs) > 0, "Should have potential conflict log"
        
        log_record = potential_conflict_logs[0]
        assert log_record.levelname == "INFO"
        # Check that conflict_id is in the extra fields
        assert hasattr(log_record, 'memory_id') or "memory_id" in log_record.message


# ============================================================================
# Session_Manager Logging Tests
# ============================================================================


class TestSessionManagerLogging:
    """Test logging in Session_Manager."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a Session_Manager instance for testing."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=3600
        )
        memory = MockMemoryInterface()
        return Session_Manager(config, memory)
    
    def test_session_creation_logging(self, session_manager, caplog):
        """
        Test that session creation logs with INFO level.
        
        Requirement 12.4: Log session lifecycle events (creation)
        """
        caplog.set_level(logging.INFO)
        
        # Create a session
        session_id = session_manager.create_session(metadata={"user_id": "test_user"})
        
        # Check for creation log
        creation_logs = [r for r in caplog.records if "Session created" in r.message]
        assert len(creation_logs) > 0, "Should have session creation log"
        
        log_record = creation_logs[0]
        assert log_record.levelname == "INFO"
        assert session_id in log_record.message
    
    def test_session_end_logging(self, session_manager, caplog):
        """
        Test that session end logs with INFO level and session summary.
        
        Requirement 12.4: Log session lifecycle events (end) and session summaries
        """
        caplog.set_level(logging.INFO)
        
        # Create and use a session
        session_id = session_manager.create_session()
        session_manager.update_activity(session_id)
        session_manager.update_activity(session_id)
        
        # Buffer some memories
        session_manager.buffer_memory(
            session_id,
            "Test memory 1",
            {"category": "test"}
        )
        session_manager.buffer_memory(
            session_id,
            "Test memory 2",
            {"category": "test"}
        )
        
        caplog.clear()
        
        # End the session
        persisted_count = session_manager.end_session(session_id)
        
        # Check for end log
        end_logs = [r for r in caplog.records if "Session ended" in r.message]
        assert len(end_logs) > 0, "Should have session end log"
        
        log_record = end_logs[0]
        assert log_record.levelname == "INFO"
        assert session_id in log_record.message
        # Check that summary includes persisted_count and message_count
        assert "persisted_count" in log_record.message or hasattr(log_record, 'persisted_count')
        assert "message_count" in log_record.message or hasattr(log_record, 'message_count')
    
    def test_session_expiration_logging(self, session_manager, caplog):
        """
        Test that session expiration logs with INFO level.
        
        Requirement 12.4: Log session lifecycle events (expiration)
        """
        caplog.set_level(logging.INFO)
        
        # Create a session with short timeout
        short_timeout_manager = Session_Manager(
            SessionConfig(timeout_seconds=1, cleanup_interval_seconds=3600),
            MockMemoryInterface()
        )
        
        session_id = short_timeout_manager.create_session()
        
        # Buffer a memory
        short_timeout_manager.buffer_memory(
            session_id,
            "Test memory",
            {"category": "test"}
        )
        
        caplog.clear()
        
        # Mock time to make session expire
        with patch('luma.core.session_manager.datetime') as mock_datetime:
            # Set current time to 2 seconds after session creation
            future_time = datetime.now(UTC) + timedelta(seconds=2)
            mock_datetime.utcnow.return_value = future_time
            
            # Trigger cleanup
            short_timeout_manager._cleanup_expired_sessions()
        
        # Check for expiration log
        expiration_logs = [r for r in caplog.records if "Session expired" in r.message]
        assert len(expiration_logs) > 0, "Should have session expiration log"
        
        log_record = expiration_logs[0]
        assert log_record.levelname == "INFO"
        assert "reason=timeout" in log_record.message or hasattr(log_record, 'reason')
    
    def test_cleanup_task_logging(self, session_manager, caplog):
        """
        Test that cleanup task logs when expired sessions are found.
        
        Requirement 12.4: Log cleanup operations
        """
        caplog.set_level(logging.INFO)
        
        # Create a session with short timeout
        short_timeout_manager = Session_Manager(
            SessionConfig(timeout_seconds=1, cleanup_interval_seconds=3600),
            MockMemoryInterface()
        )
        
        session_id = short_timeout_manager.create_session()
        
        caplog.clear()
        
        # Mock time to make session expire
        with patch('luma.core.session_manager.datetime') as mock_datetime:
            future_time = datetime.now(UTC) + timedelta(seconds=2)
            mock_datetime.utcnow.return_value = future_time
            
            # Trigger cleanup
            short_timeout_manager._cleanup_expired_sessions()
        
        # Check for cleanup log
        cleanup_logs = [r for r in caplog.records if "Cleanup" in r.message and "expired sessions" in r.message]
        assert len(cleanup_logs) > 0, "Should have cleanup log"
        
        log_record = cleanup_logs[0]
        assert log_record.levelname == "INFO"
    
    def test_buffer_memory_logging(self, session_manager, caplog):
        """
        Test that buffer memory operations log with DEBUG level.
        
        Requirement 12.5: Support configurable log levels for different operation types
        """
        caplog.set_level(logging.DEBUG)
        
        # Create a session
        session_id = session_manager.create_session()
        
        caplog.clear()
        
        # Buffer a memory
        session_manager.buffer_memory(
            session_id,
            "Test memory content",
            {"category": "test"}
        )
        
        # Check for buffer log
        buffer_logs = [r for r in caplog.records if "Memory buffered" in r.message]
        assert len(buffer_logs) > 0, "Should have buffer memory log"
        
        log_record = buffer_logs[0]
        assert log_record.levelname == "DEBUG"
        # Check that session_id is in the message or as an attribute
        assert session_id in log_record.message or hasattr(log_record, 'session_id')
        # Check that buffer_size and content_length are in the message
        assert "buffer_size" in log_record.message
        assert "content_length" in log_record.message
    
    def test_buffer_overflow_logging(self, session_manager, caplog):
        """
        Test that buffer overflow/flush logs with INFO level.
        
        Requirement 12.4: Log session operations including buffer management
        """
        caplog.set_level(logging.INFO)
        
        # Create a session with small buffer
        small_buffer_manager = Session_Manager(
            SessionConfig(max_buffer_size=2, timeout_seconds=3600, cleanup_interval_seconds=3600),
            MockMemoryInterface()
        )
        
        session_id = small_buffer_manager.create_session()
        
        # Buffer memories to trigger overflow
        small_buffer_manager.buffer_memory(session_id, "Memory 1", {"category": "test"})
        small_buffer_manager.buffer_memory(session_id, "Memory 2", {"category": "test"})
        
        caplog.clear()
        
        # This should trigger overflow
        small_buffer_manager.buffer_memory(session_id, "Memory 3", {"category": "test"})
        
        # Check for overflow log
        overflow_logs = [r for r in caplog.records if "Buffer overflow" in r.message]
        assert len(overflow_logs) > 0, "Should have buffer overflow log"
        
        log_record = overflow_logs[0]
        assert log_record.levelname == "INFO"
        assert session_id in log_record.message or hasattr(log_record, 'session_id')


# ============================================================================
# Log Level Tests
# ============================================================================


class TestLogLevels:
    """Test that log levels are appropriate for different operations."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(timeout_seconds=3600, cleanup_interval_seconds=3600),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_info_level_for_normal_operations(self, strategy, caplog):
        """
        Test that normal operations use INFO level.
        
        Requirement 12.5: Support configurable log levels for different operation types
        """
        caplog.set_level(logging.INFO)
        
        # Perform normal storage operation
        strategy.store_memory("Normal content", immediate=True)
        
        # Check that INFO logs were generated
        info_logs = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_logs) > 0, "Should have INFO level logs for normal operations"
    
    def test_error_level_for_failures(self, strategy, caplog):
        """
        Test that failures use ERROR level.
        
        Requirement 12.5: Support configurable log levels for different operation types
        """
        caplog.set_level(logging.ERROR)
        
        # Replace memory interface with one that fails
        strategy.memory = MockMemoryInterface(should_fail=True)
        
        # Try to store a memory (should fail)
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("Test content", immediate=True)
        
        # Check that ERROR logs were generated
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0, "Should have ERROR level logs for failures"
    
    def test_debug_level_for_detailed_info(self, strategy, caplog):
        """
        Test that detailed information uses DEBUG level.
        
        Requirement 12.5: Support configurable log levels for different operation types
        """
        caplog.set_level(logging.DEBUG)
        
        # Perform operation that generates debug logs
        strategy.validate_content("Test content", {"tags": ["test"]})
        
        # Check that DEBUG logs were generated
        debug_logs = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(debug_logs) > 0, "Should have DEBUG level logs for detailed information"


# ============================================================================
# Log Content Validation Tests
# ============================================================================


class TestLogContentValidation:
    """Test that log messages contain required context fields."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance for testing."""
        config = WriteStrategyConfig()
        session_manager = Session_Manager(
            SessionConfig(timeout_seconds=3600, cleanup_interval_seconds=3600),
            MockMemoryInterface()
        )
        memory = MockMemoryInterface()
        return Memory_Write_Strategy(config, session_manager, memory)
    
    def test_storage_log_contains_required_fields(self, strategy, caplog):
        """
        Test that storage logs contain all required context fields.
        
        Requirement 12.1: Log memory storage operations (memory_id, content length, category, tags)
        """
        caplog.set_level(logging.INFO)
        
        # Store a memory
        memory_id = strategy.store_memory(
            "Test content",
            metadata={"category": "test", "tags": ["tag1"]},
            immediate=True
        )
        
        # Find storage log
        storage_logs = [r for r in caplog.records if "Memory stored" in r.message]
        assert len(storage_logs) > 0
        
        log_record = storage_logs[0]
        # Check required fields
        assert hasattr(log_record, 'memory_id')
        assert hasattr(log_record, 'content_length')
        assert hasattr(log_record, 'category')
        assert hasattr(log_record, 'tags')
    
    def test_rejection_log_contains_reason(self, strategy, caplog):
        """
        Test that rejection logs contain the rejection reason.
        
        Requirement 12.2: Log rejection reasons
        """
        caplog.set_level(logging.INFO)
        
        # Try to store trivial content
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("hello")
        
        # Find rejection log
        rejection_logs = [r for r in caplog.records if "rejected" in r.message.lower()]
        assert len(rejection_logs) > 0
        
        log_record = rejection_logs[0]
        assert hasattr(log_record, 'reason')
    
    def test_error_log_contains_exception_info(self, strategy, caplog):
        """
        Test that error logs include exception details.
        
        Requirement 12.3: Log storage failures with full exception details
        """
        caplog.set_level(logging.ERROR)
        
        # Replace memory interface with one that fails
        strategy.memory = MockMemoryInterface(should_fail=True)
        
        # Try to store a memory
        with pytest.raises(MemoryStorageError):
            strategy.store_memory("Test content", immediate=True)
        
        # Find error log
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0
        
        log_record = error_logs[0]
        # Check that exception info is included
        assert log_record.exc_info is not None
