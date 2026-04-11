"""Tests for error tracking functionality."""

import pytest
from datetime import datetime

from luma_memory.utils.error_tracker import (
    ErrorTracker,
    ErrorCategory,
    ErrorSeverity,
    ErrorRecord,
    get_error_tracker,
    reset_error_tracker
)
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.backend import StorageError
from luma_memory.processing.validation import ValidationError
from luma_memory.config import MemoryModuleConfig


class TestErrorTracker:
    """Test ErrorTracker class."""
    
    def test_error_tracker_initialization(self):
        """Test that error tracker initializes correctly."""
        tracker = ErrorTracker(max_history=100)
        
        assert tracker.max_history == 100
        assert tracker._total_errors == 0
        assert len(tracker._error_history) == 0
    
    def test_track_error_basic(self):
        """Test basic error tracking."""
        tracker = ErrorTracker()
        
        error = ValueError("Test error")
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            operation="test_operation"
        )
        
        assert tracker._total_errors == 1
        assert len(tracker._error_history) == 1
        assert tracker._error_counts[ErrorCategory.VALIDATION] == 1
        assert tracker._severity_counts[ErrorSeverity.MEDIUM] == 1
    
    def test_track_error_with_context(self):
        """Test error tracking with context."""
        tracker = ErrorTracker()
        
        error = ValueError("Test error")
        context = {"entry_id": "abc-123", "field": "action"}
        
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            operation="create_memory",
            context=context
        )
        
        recent = tracker.get_recent_errors(limit=1)
        assert len(recent) == 1
        assert recent[0]['context'] == context
        assert recent[0]['operation'] == "create_memory"
    
    def test_track_multiple_errors(self):
        """Test tracking multiple errors."""
        tracker = ErrorTracker()
        
        # Track different types of errors
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=ValueError("Validation error"),
            operation="create_memory"
        )
        
        tracker.track_error(
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            error=StorageError("Storage error"),
            operation="get_memory"
        )
        
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            error=ValueError("Another validation error"),
            operation="update_memory"
        )
        
        assert tracker._total_errors == 3
        assert tracker._error_counts[ErrorCategory.VALIDATION] == 2
        assert tracker._error_counts[ErrorCategory.STORAGE] == 1
        assert tracker._severity_counts[ErrorSeverity.MEDIUM] == 1
        assert tracker._severity_counts[ErrorSeverity.HIGH] == 1
        assert tracker._severity_counts[ErrorSeverity.LOW] == 1
    
    def test_get_error_stats(self):
        """Test getting error statistics."""
        tracker = ErrorTracker()
        
        # Track some errors
        for i in range(5):
            tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=StorageError(f"Error {i}"),
                operation="create_memory"
            )
        
        stats = tracker.get_error_stats()
        
        assert stats['total_errors'] == 5
        assert stats['errors_by_category']['storage'] == 5
        assert stats['errors_by_severity']['high'] == 5
        assert stats['errors_by_operation']['create_memory'] == 5
        assert stats['history_size'] == 5
        assert 'error_rate_per_minute' in stats
    
    def test_get_recent_errors(self):
        """Test getting recent errors."""
        tracker = ErrorTracker()
        
        # Track multiple errors
        for i in range(10):
            tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=ValueError(f"Error {i}"),
                operation="test_operation"
            )
        
        # Get recent 5 errors
        recent = tracker.get_recent_errors(limit=5)
        
        assert len(recent) == 5
        # Should be in reverse order (most recent first)
        assert "Error 9" in recent[0]['message']
        assert "Error 5" in recent[4]['message']
    
    def test_get_errors_by_category(self):
        """Test filtering errors by category."""
        tracker = ErrorTracker()
        
        # Track errors in different categories
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=ValueError("Validation error"),
            operation="create_memory"
        )
        
        tracker.track_error(
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            error=StorageError("Storage error"),
            operation="get_memory"
        )
        
        validation_errors = tracker.get_errors_by_category(ErrorCategory.VALIDATION)
        storage_errors = tracker.get_errors_by_category(ErrorCategory.STORAGE)
        
        assert len(validation_errors) == 1
        assert len(storage_errors) == 1
        assert validation_errors[0]['category'] == 'validation'
        assert storage_errors[0]['category'] == 'storage'
    
    def test_get_errors_by_operation(self):
        """Test filtering errors by operation."""
        tracker = ErrorTracker()
        
        # Track errors for different operations
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=ValueError("Error 1"),
            operation="create_memory"
        )
        
        tracker.track_error(
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            error=StorageError("Error 2"),
            operation="create_memory"
        )
        
        tracker.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=ValueError("Error 3"),
            operation="get_memory"
        )
        
        create_errors = tracker.get_errors_by_operation("create_memory")
        get_errors = tracker.get_errors_by_operation("get_memory")
        
        assert len(create_errors) == 2
        assert len(get_errors) == 1
    
    def test_max_history_limit(self):
        """Test that error history respects max_history limit."""
        tracker = ErrorTracker(max_history=5)
        
        # Track more errors than max_history
        for i in range(10):
            tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=ValueError(f"Error {i}"),
                operation="test_operation"
            )
        
        # History should be limited to max_history
        assert len(tracker._error_history) == 5
        # Total count should still be accurate
        assert tracker._total_errors == 10
        
        # Should keep most recent errors
        recent = tracker.get_recent_errors(limit=5)
        assert "Error 9" in recent[0]['message']
        assert "Error 5" in recent[4]['message']
    
    def test_clear_history(self):
        """Test clearing error history."""
        tracker = ErrorTracker()
        
        # Track some errors
        for i in range(5):
            tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=ValueError(f"Error {i}"),
                operation="test_operation"
            )
        
        tracker.clear_history()
        
        # History should be cleared
        assert len(tracker._error_history) == 0
        # But counters should remain
        assert tracker._total_errors == 5
        assert tracker._error_counts[ErrorCategory.VALIDATION] == 5
    
    def test_reset(self):
        """Test resetting error tracker."""
        tracker = ErrorTracker()
        
        # Track some errors
        for i in range(5):
            tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=ValueError(f"Error {i}"),
                operation="test_operation"
            )
        
        tracker.reset()
        
        # Everything should be reset
        assert len(tracker._error_history) == 0
        assert tracker._total_errors == 0
        assert len(tracker._error_counts) == 0
        assert tracker._first_error_time is None
        assert tracker._last_error_time is None
    
    def test_global_error_tracker(self):
        """Test global error tracker functions."""
        # Reset first to ensure clean state
        reset_error_tracker()
        
        tracker1 = get_error_tracker()
        tracker2 = get_error_tracker()
        
        # Should return the same instance
        assert tracker1 is tracker2
        
        # Track an error
        tracker1.track_error(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            error=ValueError("Test error"),
            operation="test_operation"
        )
        
        # Should be visible in tracker2
        assert tracker2._total_errors == 1


class TestMemoryManagerErrorTracking:
    """Test error tracking integration in MemoryManager."""
    
    def test_memory_manager_has_error_tracker(self):
        """Test that MemoryManager initializes with error tracker."""
        storage = MemoryStorage()
        config = MemoryModuleConfig()
        
        manager = MemoryManager(storage=storage, config=config)
        
        assert manager.error_tracker is not None
        assert isinstance(manager.error_tracker, ErrorTracker)
    
    def test_validation_error_tracked(self):
        """Test that validation errors are tracked."""
        storage = MemoryStorage()
        config = MemoryModuleConfig()
        manager = MemoryManager(storage=storage, config=config)
        
        # Try to create memory with invalid data (empty action)
        with pytest.raises(ValidationError):
            manager.create_memory(
                action="",  # Empty action should fail validation
                context={},
                device_id="test-device"
            )
        
        # Check that error was tracked
        stats = manager.error_tracker.get_error_stats()
        assert stats['total_errors'] > 0
        assert stats['errors_by_category'].get('validation', 0) > 0
    
    def test_error_stats_in_manager_stats(self):
        """Test that error tracking stats are included in manager stats."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create a memory successfully
        manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="test-device"
        )
        
        # Get stats
        stats = manager.get_stats()
        
        # Error tracking stats should be included
        assert 'error_tracking' in stats
        assert 'total_errors' in stats['error_tracking']
        assert 'errors_by_category' in stats['error_tracking']
    
    def test_multiple_operations_error_tracking(self):
        """Test error tracking across multiple operations."""
        storage = MemoryStorage()
        config = MemoryModuleConfig()
        manager = MemoryManager(storage=storage, config=config)
        
        # Create a valid memory
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="test-device"
        )
        
        # Try invalid operations
        with pytest.raises(ValidationError):
            manager.create_memory(
                action="",  # Invalid
                context={},
                device_id="test-device"
            )
        
        with pytest.raises(ValidationError):
            manager.update_memory(entry_id, {"action": ""})  # Invalid update
        
        # Check error tracking
        stats = manager.error_tracker.get_error_stats()
        assert stats['total_errors'] == 2
        assert stats['errors_by_operation']['create_memory'] == 1
        assert stats['errors_by_operation']['update_memory'] == 1
