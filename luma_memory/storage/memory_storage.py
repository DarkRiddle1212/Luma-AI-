"""
In-memory storage backend for Luma Memory Module.

This module provides an in-memory implementation of the StorageBackend
interface, primarily for testing purposes. It stores all data in memory
using dictionaries and provides thread-safe operations.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from threading import Lock
from collections import defaultdict

from luma_memory.storage.backend import StorageBackend, StorageError
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend implementation.
    
    This backend stores all memory entries in memory using dictionaries.
    It's designed for testing purposes and provides fast access without
    any file I/O operations.
    
    Thread-safety is ensured through a lock that protects all operations.
    
    Attributes:
        _entries: Dictionary mapping entry IDs to MemoryEntry instances
        _lock: Threading lock for thread-safe operations
    """
    
    def __init__(self):
        """Initialize the in-memory storage backend."""
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = Lock()
    
    def create_entry(self, entry: MemoryEntry) -> str:
        """
        Store a new memory entry in memory.
        
        Args:
            entry: The MemoryEntry instance to store.
        
        Returns:
            The ID of the stored entry.
        
        Raises:
            ValueError: If the entry is invalid or already exists.
            StorageError: If the storage operation fails.
        """
        # Validate the entry
        is_valid, error_msg = entry.validate()
        if not is_valid:
            raise ValueError(f"Invalid memory entry: {error_msg}")
        
        with self._lock:
            # Check if entry already exists
            if entry.id in self._entries:
                raise ValueError(f"Entry with ID {entry.id} already exists")
            
            try:
                # Store the entry
                self._entries[entry.id] = entry
                return entry.id
            except Exception as e:
                raise StorageError(f"Failed to create entry: {str(e)}")
    
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
        with self._lock:
            try:
                return self._entries.get(entry_id)
            except Exception as e:
                raise StorageError(f"Failed to retrieve entry: {str(e)}")
    
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
        # Validate parameters
        if limit < 0:
            raise ValueError("Limit must be non-negative")
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        
        with self._lock:
            try:
                # Get all entries
                results = list(self._entries.values())
                
                # Apply filters
                if start_time:
                    results = [e for e in results if e.timestamp >= start_time]
                
                if end_time:
                    results = [e for e in results if e.timestamp <= end_time]
                
                if tags:
                    results = [e for e in results if any(tag in e.tags for tag in tags)]
                
                if action_type:
                    results = [e for e in results if action_type.lower() in e.action.lower()]
                
                # Sort by timestamp (newest first)
                results.sort(key=lambda e: e.timestamp, reverse=True)
                
                # Apply pagination
                start_idx = offset
                end_idx = offset + limit
                return results[start_idx:end_idx]
                
            except Exception as e:
                raise StorageError(f"Failed to query entries: {str(e)}")
    
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The unique identifier of the entry to update.
            updates: Dictionary of field names and new values.
        
        Returns:
            True if the entry was updated, False if not found.
        
        Raises:
            ValueError: If the updates are invalid.
            StorageError: If the update operation fails.
        """
        with self._lock:
            try:
                # Check if entry exists
                if entry_id not in self._entries:
                    return False
                
                entry = self._entries[entry_id]
                
                # Apply updates
                for field, value in updates.items():
                    if not hasattr(entry, field):
                        raise ValueError(f"Invalid field: {field}")
                    
                    # Handle enum conversions
                    if field == "sensitivity" and isinstance(value, str):
                        value = SensitivityLevel(value)
                    elif field == "sync_status" and isinstance(value, str):
                        value = SyncStatus(value)
                    elif field in ["timestamp", "created_at", "updated_at"] and isinstance(value, str):
                        value = datetime.fromisoformat(value)
                    
                    setattr(entry, field, value)
                
                # Update the updated_at timestamp
                entry.updated_at = datetime.now(UTC) if hasattr(datetime, 'UTC') else datetime.now(UTC)
                
                # Validate the updated entry
                is_valid, error_msg = entry.validate()
                if not is_valid:
                    raise ValueError(f"Invalid update: {error_msg}")
                
                return True
                
            except ValueError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to update entry: {str(e)}")
    
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
        with self._lock:
            try:
                if entry_id in self._entries:
                    del self._entries[entry_id]
                    return True
                return False
            except Exception as e:
                raise StorageError(f"Failed to delete entry: {str(e)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Return storage statistics.
        
        Returns:
            Dictionary containing storage statistics.
        
        Raises:
            StorageError: If the stats operation fails.
        """
        with self._lock:
            try:
                entries = list(self._entries.values())
                
                if not entries:
                    return {
                        "total_entries": 0,
                        "storage_size_bytes": 0,
                        "oldest_entry": None,
                        "newest_entry": None,
                        "entries_by_sensitivity": {},
                        "entries_by_sync_status": {}
                    }
                
                # Calculate storage size (approximate)
                storage_size = sum(
                    len(str(e.to_dict())) for e in entries
                )
                
                # Find oldest and newest entries
                sorted_entries = sorted(entries, key=lambda e: e.timestamp)
                oldest = sorted_entries[0].timestamp
                newest = sorted_entries[-1].timestamp
                
                # Count by sensitivity
                sensitivity_counts = defaultdict(int)
                for entry in entries:
                    sensitivity_counts[entry.sensitivity.value] += 1
                
                # Count by sync status
                sync_status_counts = defaultdict(int)
                for entry in entries:
                    sync_status_counts[entry.sync_status.value] += 1
                
                return {
                    "total_entries": len(entries),
                    "storage_size_bytes": storage_size,
                    "oldest_entry": oldest.isoformat(),
                    "newest_entry": newest.isoformat(),
                    "entries_by_sensitivity": dict(sensitivity_counts),
                    "entries_by_sync_status": dict(sync_status_counts)
                }
                
            except Exception as e:
                raise StorageError(f"Failed to get storage stats: {str(e)}")
    
    def clear(self) -> None:
        """
        Clear all entries from storage.
        
        This method is useful for testing purposes to reset the storage state.
        """
        with self._lock:
            self._entries.clear()
    
    def get_entry_count(self) -> int:
        """
        Get the total number of entries in storage.
        
        Returns:
            The number of entries stored.
        """
        with self._lock:
            return len(self._entries)

