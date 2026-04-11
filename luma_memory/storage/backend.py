"""
Abstract storage backend interface for Luma Memory Module.

This module defines the abstract base class that all storage backends
must implement, providing a consistent interface for memory operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from luma_memory.models import MemoryEntry


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    All storage implementations (SQLite, JSON, cloud, etc.) must implement
    this interface to ensure consistent behavior across different storage
    mechanisms.
    
    The interface provides CRUD operations for memory entries along with
    query capabilities and storage statistics.
    """
    
    @abstractmethod
    def create_entry(self, entry: MemoryEntry) -> str:
        """
        Store a new memory entry and return its ID.
        
        Args:
            entry: The MemoryEntry instance to store.
        
        Returns:
            The ID of the stored entry.
        
        Raises:
            ValueError: If the entry is invalid or already exists.
            StorageError: If the storage operation fails.
        """
        pass
    
    @abstractmethod
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.
        
        Args:
            entry_id: The unique identifier of the entry to retrieve.
        
        Returns:
            The MemoryEntry if found, None otherwise.
        
        Raises:
            StorageError: If the retrieval operation fails.
        """
        pass
    
    @abstractmethod
    def query_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        action_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """
        Query memory entries with filters.
        
        Returns entries in reverse chronological order (newest first).
        Supports pagination through limit and offset parameters.
        
        Args:
            start_time: Optional start of time range filter (inclusive).
            end_time: Optional end of time range filter (inclusive).
            tags: Optional list of tags to filter by (entries must have at least one).
            action_type: Optional action type to filter by (partial match).
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).
        
        Returns:
            List of MemoryEntry instances matching the filters.
        
        Raises:
            ValueError: If filter parameters are invalid.
            StorageError: If the query operation fails.
        """
        pass
    
    @abstractmethod
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.
        
        Only the fields specified in the updates dictionary will be modified.
        The updated_at timestamp is automatically set to the current time.
        
        Args:
            entry_id: The unique identifier of the entry to update.
            updates: Dictionary of field names and new values.
        
        Returns:
            True if the entry was updated, False if not found.
        
        Raises:
            ValueError: If the updates are invalid.
            StorageError: If the update operation fails.
        """
        pass
    
    @abstractmethod
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The unique identifier of the entry to delete.
        
        Returns:
            True if the entry was deleted, False if not found.
        
        Raises:
            StorageError: If the delete operation fails.
        """
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Return storage statistics.
        
        Provides information about storage usage, entry counts, and
        performance metrics.
        
        Returns:
            Dictionary containing storage statistics with keys:
                - total_entries: Total number of entries stored
                - storage_size_bytes: Total storage size in bytes
                - oldest_entry: Timestamp of oldest entry (or None)
                - newest_entry: Timestamp of newest entry (or None)
                - entries_by_sensitivity: Count by sensitivity level
                - entries_by_sync_status: Count by sync status
        
        Raises:
            StorageError: If the stats operation fails.
        """
        pass


class StorageError(Exception):
    """
    Exception raised for storage operation failures.
    
    This exception is raised when a storage operation fails due to
    I/O errors, database errors, or other storage-related issues.
    """
    pass
