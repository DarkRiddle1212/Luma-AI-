"""
Error tracking for Luma Memory Module.

This module provides error tracking capabilities to capture, categorize,
and report errors that occur during system operation.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur in the system."""
    VALIDATION = "validation"
    STORAGE = "storage"
    ENCRYPTION = "encryption"
    API = "api"
    SUMMARIZATION = "summarization"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """
    Record of a single error occurrence.
    
    Attributes:
        timestamp: When the error occurred
        category: Category of the error
        severity: Severity level
        error_type: Type/class of the exception
        message: Error message
        operation: Operation that was being performed
        context: Additional contextual information
        stack_trace: Optional stack trace
    """
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str
    message: str
    operation: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data


class ErrorTracker:
    """
    Tracks errors that occur during system operation.
    
    Provides:
    - Error recording with categorization and severity
    - Error frequency tracking
    - Error statistics and reporting
    - Recent error history
    - Error rate monitoring
    
    Example:
        >>> tracker = ErrorTracker(max_history=100)
        >>> tracker.track_error(
        ...     category=ErrorCategory.STORAGE,
        ...     severity=ErrorSeverity.HIGH,
        ...     error=StorageError("Database connection failed"),
        ...     operation="create_memory",
        ...     context={"entry_id": "abc-123"}
        ... )
        >>> stats = tracker.get_error_stats()
        >>> print(f"Total errors: {stats['total_errors']}")
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the error tracker.
        
        Args:
            max_history: Maximum number of error records to keep in history
        """
        self.max_history = max_history
        
        # Error history (most recent errors)
        self._error_history: List[ErrorRecord] = []
        
        # Error frequency by category
        self._error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        
        # Error frequency by type
        self._error_type_counts: Dict[str, int] = defaultdict(int)
        
        # Error frequency by operation
        self._operation_error_counts: Dict[str, int] = defaultdict(int)
        
        # Error frequency by severity
        self._severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        
        # Total error count
        self._total_errors = 0
        
        # First and last error timestamps
        self._first_error_time: Optional[datetime] = None
        self._last_error_time: Optional[datetime] = None
        
        logger.info(f"ErrorTracker initialized with max_history={max_history}")
    
    def track_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        include_stack_trace: bool = False
    ) -> None:
        """
        Track an error occurrence.
        
        Args:
            category: Category of the error
            severity: Severity level
            error: The exception that occurred
            operation: Operation that was being performed
            context: Additional contextual information
            include_stack_trace: Whether to include stack trace
        
        Example:
            >>> tracker.track_error(
            ...     category=ErrorCategory.VALIDATION,
            ...     severity=ErrorSeverity.MEDIUM,
            ...     error=ValidationError("Invalid field"),
            ...     operation="create_memory",
            ...     context={"field": "action"}
            ... )
        """
        import traceback
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            operation=operation,
            context=context or {},
            stack_trace=traceback.format_exc() if include_stack_trace else None
        )
        
        # Add to history (maintain max size)
        self._error_history.append(error_record)
        if len(self._error_history) > self.max_history:
            self._error_history.pop(0)
        
        # Update counters
        self._error_counts[category] += 1
        self._error_type_counts[error_record.error_type] += 1
        self._operation_error_counts[operation] += 1
        self._severity_counts[severity] += 1
        self._total_errors += 1
        
        # Update timestamps
        if self._first_error_time is None:
            self._first_error_time = error_record.timestamp
        self._last_error_time = error_record.timestamp
        
        # Log the error
        log_level = self._get_log_level(severity)
        logger.log(
            log_level,
            f"Error tracked: {category.value}/{severity.value} - {error_record.error_type}: {error_record.message}",
            extra={
                "error_category": category.value,
                "error_severity": severity.value,
                "error_type": error_record.error_type,
                "operation": operation,
                "context": context or {}
            }
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics.
        
        Returns:
            Dictionary containing:
                - total_errors: Total number of errors tracked
                - errors_by_category: Error counts by category
                - errors_by_type: Error counts by error type
                - errors_by_operation: Error counts by operation
                - errors_by_severity: Error counts by severity
                - first_error_time: Timestamp of first error
                - last_error_time: Timestamp of last error
                - error_rate: Errors per minute (if applicable)
                - recent_errors: List of recent error records
        
        Example:
            >>> stats = tracker.get_error_stats()
            >>> print(f"Total errors: {stats['total_errors']}")
            >>> print(f"Storage errors: {stats['errors_by_category']['storage']}")
        """
        stats = {
            'total_errors': self._total_errors,
            'errors_by_category': {
                cat.value: count for cat, count in self._error_counts.items()
            },
            'errors_by_type': dict(self._error_type_counts),
            'errors_by_operation': dict(self._operation_error_counts),
            'errors_by_severity': {
                sev.value: count for sev, count in self._severity_counts.items()
            },
            'first_error_time': self._first_error_time.isoformat() if self._first_error_time else None,
            'last_error_time': self._last_error_time.isoformat() if self._last_error_time else None,
            'history_size': len(self._error_history),
            'max_history': self.max_history
        }
        
        # Calculate error rate (errors per minute)
        if self._first_error_time and self._last_error_time:
            time_diff = (self._last_error_time - self._first_error_time).total_seconds()
            if time_diff > 0:
                stats['error_rate_per_minute'] = round((self._total_errors / time_diff) * 60, 2)
            else:
                stats['error_rate_per_minute'] = 0.0
        else:
            stats['error_rate_per_minute'] = 0.0
        
        return stats
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            limit: Maximum number of errors to return
        
        Returns:
            List of error records as dictionaries
        
        Example:
            >>> recent = tracker.get_recent_errors(limit=5)
            >>> for error in recent:
            ...     print(f"{error['timestamp']}: {error['message']}")
        """
        recent = self._error_history[-limit:] if self._error_history else []
        return [error.to_dict() for error in reversed(recent)]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[Dict[str, Any]]:
        """
        Get all errors for a specific category.
        
        Args:
            category: Error category to filter by
        
        Returns:
            List of error records matching the category
        
        Example:
            >>> storage_errors = tracker.get_errors_by_category(ErrorCategory.STORAGE)
        """
        filtered = [
            error for error in self._error_history
            if error.category == category
        ]
        return [error.to_dict() for error in filtered]
    
    def get_errors_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """
        Get all errors for a specific operation.
        
        Args:
            operation: Operation name to filter by
        
        Returns:
            List of error records for the operation
        
        Example:
            >>> create_errors = tracker.get_errors_by_operation("create_memory")
        """
        filtered = [
            error for error in self._error_history
            if error.operation == operation
        ]
        return [error.to_dict() for error in filtered]
    
    def clear_history(self) -> None:
        """
        Clear error history but keep counters.
        
        Useful for resetting history while maintaining aggregate statistics.
        """
        self._error_history.clear()
        logger.info("Error history cleared")
    
    def reset(self) -> None:
        """
        Reset all error tracking data.
        
        Clears history and resets all counters to zero.
        """
        self._error_history.clear()
        self._error_counts.clear()
        self._error_type_counts.clear()
        self._operation_error_counts.clear()
        self._severity_counts.clear()
        self._total_errors = 0
        self._first_error_time = None
        self._last_error_time = None
        logger.info("Error tracker reset")
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """
        Map error severity to logging level.
        
        Args:
            severity: Error severity
        
        Returns:
            Logging level constant
        """
        severity_to_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_to_level.get(severity, logging.ERROR)


# Global error tracker instance
_global_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """
    Get the global error tracker instance.
    
    Creates a new instance if one doesn't exist.
    
    Returns:
        Global ErrorTracker instance
    
    Example:
        >>> tracker = get_error_tracker()
        >>> tracker.track_error(...)
    """
    global _global_error_tracker
    if _global_error_tracker is None:
        _global_error_tracker = ErrorTracker()
    return _global_error_tracker


def reset_error_tracker() -> None:
    """
    Reset the global error tracker.
    
    Useful for testing or when starting a new monitoring period.
    """
    global _global_error_tracker
    if _global_error_tracker is not None:
        _global_error_tracker.reset()
