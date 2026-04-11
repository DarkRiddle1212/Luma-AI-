"""
Sync Coordinator for Luma Memory Module.

This module provides the interface for future cross-device synchronization
of memory entries. It manages sync queues, conflict resolution, and
incremental synchronization.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC

from luma_memory.models import MemoryEntry


class ConflictResolution(Enum):
    """
    Strategies for resolving synchronization conflicts.
    
    When the same memory entry is modified on multiple devices,
    these strategies determine how to resolve the conflict.
    """
    LATEST_WINS = "latest_wins"  # Most recent update wins
    MERGE = "merge"              # Attempt to merge changes
    MANUAL = "manual"            # Require manual resolution


class SyncCoordinator:
    """
    Coordinates synchronization of memory entries across devices.
    
    This class provides the interface for future cloud synchronization
    functionality. It manages marking entries for sync, retrieving pending
    sync items, resolving conflicts, and performing incremental syncs.
    
    Attributes:
        resolution_strategy: The conflict resolution strategy to use
    """
    
    def __init__(self, resolution_strategy: ConflictResolution = ConflictResolution.LATEST_WINS, storage_backend=None):
        """
        Initialize the SyncCoordinator.

        Args:
            resolution_strategy: Strategy for resolving sync conflicts
            storage_backend: Optional storage backend for sync queue management
        """
        self.resolution_strategy = resolution_strategy
        self.storage_backend = storage_backend

    
    def mark_for_sync(self, entry_id: str, operation: str = "update") -> None:
        """
        Mark a memory entry for synchronization.
        
        Adds the entry to the sync queue so it will be synchronized
        with other devices on the next sync operation.
        
        Args:
            entry_id: The ID of the memory entry to mark for sync
            operation: The type of operation (create, update, delete)
        
        Raises:
            ValueError: If operation is not one of: create, update, delete
            RuntimeError: If storage backend is not configured
        
        Note:
            This implementation adds the entry to the sync_queue table
            in the database. The entry will remain in the queue until
            it is successfully synchronized.
        """
        # Validate operation type
        valid_operations = ["create", "update", "delete"]
        if operation not in valid_operations:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of: {', '.join(valid_operations)}"
            )
        
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot mark entry for sync."
            )
        
        # Add entry to sync queue
        import sqlite3
        from datetime import datetime, UTC
        
        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert into sync_queue table
                cursor.execute("""
                    INSERT INTO sync_queue (entry_id, operation, queued_at)
                    VALUES (?, ?, ?)
                """, (entry_id, operation, datetime.now(UTC)))
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark entry for sync: {e}")
    
    def get_pending_sync(self) -> List[MemoryEntry]:
        """
        Get all memory entries pending synchronization.

        Retrieves entries that have been marked for sync but haven't
        been successfully synchronized yet.

        Returns:
            List of MemoryEntry objects pending synchronization

        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database query fails

        Note:
            This method queries the sync_queue table for entries where
            synced_at IS NULL and joins with memory_entries to return
            the full MemoryEntry objects.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot retrieve pending sync entries."
            )

        import sqlite3

        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Query sync_queue for pending entries and join with memory_entries
                cursor.execute("""
                    SELECT me.*
                    FROM sync_queue sq
                    INNER JOIN memory_entries me ON sq.entry_id = me.id
                    WHERE sq.synced_at IS NULL
                    ORDER BY sq.queued_at ASC
                """)

                rows = cursor.fetchall()

                # Convert rows to MemoryEntry objects using storage backend's helper
                entries = [self.storage_backend._row_to_entry(row) for row in rows]

                return entries

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to retrieve pending sync entries: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error retrieving pending sync entries: {e}")

    def mark_as_synced(self, entry_id: str) -> None:
        """
        Mark a memory entry as successfully synchronized.

        Updates the sync_queue table to record when the entry was synced,
        indicating that synchronization completed successfully.

        Args:
            entry_id: The ID of the memory entry that was synced

        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database update fails
            ValueError: If entry is not in sync queue

        Note:
            This sets the synced_at timestamp for the entry in the sync_queue.
            Entries with synced_at set are considered completed and won't be
            returned by get_pending_sync().
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot mark entry as synced."
            )

        import sqlite3

        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Update sync_queue to mark as synced
                cursor.execute("""
                    UPDATE sync_queue
                    SET synced_at = ?
                    WHERE entry_id = ? AND synced_at IS NULL
                """, (datetime.now(UTC), entry_id))

                if cursor.rowcount == 0:
                    raise ValueError(
                        f"Entry {entry_id} not found in sync queue or already synced"
                    )

                conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark entry as synced: {e}")

    def remove_from_queue(self, entry_id: str) -> None:
        """
        Remove a memory entry from the sync queue.

        Completely removes the entry from the sync_queue table. This is useful
        for cleaning up entries that no longer need to be synced or for
        removing failed sync attempts.

        Args:
            entry_id: The ID of the memory entry to remove from queue

        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database deletion fails

        Note:
            This permanently removes the entry from the sync queue. Use
            mark_as_synced() if you want to keep a record of successful syncs.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot remove entry from queue."
            )

        import sqlite3

        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Delete from sync_queue
                cursor.execute("""
                    DELETE FROM sync_queue
                    WHERE entry_id = ?
                """, (entry_id,))

                conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove entry from queue: {e}")

    def clear_synced_entries(self, older_than: Optional[datetime] = None) -> int:
        """
        Clear successfully synced entries from the sync queue.

        Removes entries that have been successfully synchronized to reduce
        queue size and improve performance. Can optionally only clear entries
        synced before a certain date.

        Args:
            older_than: Optional datetime; only clear entries synced before this time.
                       If None, clears all synced entries.

        Returns:
            Number of entries removed from the queue

        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database deletion fails

        Note:
            This only removes entries where synced_at IS NOT NULL. Pending
            entries (synced_at IS NULL) are never removed by this method.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot clear synced entries."
            )

        import sqlite3

        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                if older_than is None:
                    # Clear all synced entries
                    cursor.execute("""
                        DELETE FROM sync_queue
                        WHERE synced_at IS NOT NULL
                    """)
                else:
                    # Clear only entries synced before the specified time
                    cursor.execute("""
                        DELETE FROM sync_queue
                        WHERE synced_at IS NOT NULL AND synced_at < ?
                    """, (older_than,))

                removed_count = cursor.rowcount
                conn.commit()

                return removed_count

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to clear synced entries: {e}")

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sync queue.

        Provides information about the current state of the sync queue,
        including counts of pending and completed syncs, oldest pending
        entry, and operation breakdown.

        Returns:
            Dictionary containing queue statistics with keys:
                - total_entries: Total number of entries in queue
                - pending_count: Number of entries waiting to be synced
                - synced_count: Number of successfully synced entries
                - oldest_pending: Timestamp of oldest pending entry (or None)
                - operations: Dict with counts by operation type (create/update/delete)

        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database query fails

        Note:
            This provides a snapshot of the queue state at query time.
            The statistics may change as sync operations progress.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot get queue stats."
            )

        import sqlite3

        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                # Get total count
                cursor.execute("SELECT COUNT(*) FROM sync_queue")
                total_entries = cursor.fetchone()[0]

                # Get pending count
                cursor.execute("""
                    SELECT COUNT(*) FROM sync_queue
                    WHERE synced_at IS NULL
                """)
                pending_count = cursor.fetchone()[0]

                # Get synced count
                cursor.execute("""
                    SELECT COUNT(*) FROM sync_queue
                    WHERE synced_at IS NOT NULL
                """)
                synced_count = cursor.fetchone()[0]

                # Get oldest pending entry timestamp
                cursor.execute("""
                    SELECT MIN(queued_at) FROM sync_queue
                    WHERE synced_at IS NULL
                """)
                oldest_pending_row = cursor.fetchone()
                oldest_pending = oldest_pending_row[0] if oldest_pending_row[0] else None

                # Get operation breakdown
                cursor.execute("""
                    SELECT operation, COUNT(*) as count
                    FROM sync_queue
                    WHERE synced_at IS NULL
                    GROUP BY operation
                """)
                operations = {row[0]: row[1] for row in cursor.fetchall()}

                return {
                    "total_entries": total_entries,
                    "pending_count": pending_count,
                    "synced_count": synced_count,
                    "oldest_pending": oldest_pending,
                    "operations": operations
                }

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get queue stats: {e}")


    
    def resolve_conflict(
        self,
        local_entry: MemoryEntry,
        remote_entry: MemoryEntry
    ) -> MemoryEntry:
        """
        Resolve a synchronization conflict between local and remote entries.

        When the same entry has been modified on multiple devices, this method
        determines which version should be kept based on the configured
        resolution strategy.

        Args:
            local_entry: The local version of the memory entry
            remote_entry: The remote version of the memory entry

        Returns:
            The resolved MemoryEntry that should be kept

        Raises:
            ValueError: If entries have different IDs or if manual resolution
                       is required but not implemented

        Strategies:
            LATEST_WINS: Keeps the entry with the most recent updated_at timestamp
            MERGE: Attempts to merge non-conflicting changes from both entries
            MANUAL: Raises an error requiring manual intervention
        """
        # Validate that we're comparing the same entry
        if local_entry.id != remote_entry.id:
            raise ValueError(
                f"Cannot resolve conflict between different entries: "
                f"{local_entry.id} vs {remote_entry.id}"
            )

        if self.resolution_strategy == ConflictResolution.LATEST_WINS:
            return self._resolve_latest_wins(local_entry, remote_entry)

        elif self.resolution_strategy == ConflictResolution.MERGE:
            return self._resolve_merge(local_entry, remote_entry)

        elif self.resolution_strategy == ConflictResolution.MANUAL:
            # Manual resolution requires user intervention
            raise ValueError(
                f"Manual conflict resolution required for entry {local_entry.id}. "
                "This feature is not yet implemented."
            )

        # Default fallback to latest wins
        return self._resolve_latest_wins(local_entry, remote_entry)

    def _resolve_latest_wins(
        self,
        local_entry: MemoryEntry,
        remote_entry: MemoryEntry
    ) -> MemoryEntry:
        """
        Resolve conflict by keeping the entry with the most recent timestamp.
        
        Compares updated_at timestamps and keeps the most recent version.
        Falls back to created_at if updated_at is not available.
        
        Args:
            local_entry: The local version of the memory entry
            remote_entry: The remote version of the memory entry
        
        Returns:
            The entry with the most recent timestamp
        """
        # Compare updated_at timestamps and keep the most recent
        if local_entry.updated_at is None and remote_entry.updated_at is None:
            # If neither has updated_at, compare created_at
            if local_entry.created_at >= remote_entry.created_at:
                return local_entry
            return remote_entry
        elif local_entry.updated_at is None:
            return remote_entry
        elif remote_entry.updated_at is None:
            return local_entry
        else:
            # Both have updated_at, compare them
            if local_entry.updated_at >= remote_entry.updated_at:
                return local_entry
            return remote_entry

    def _resolve_merge(
        self,
        local_entry: MemoryEntry,
        remote_entry: MemoryEntry
    ) -> MemoryEntry:
        """
        Resolve conflict by merging non-conflicting changes from both entries.
        
        This strategy attempts to intelligently merge changes:
        - Uses the most recent timestamp for updated_at
        - Merges tags from both entries (union)
        - Merges context dictionaries, preferring values from the newer entry
        - Uses the newer summary if different
        - Preserves other fields from the newer entry
        
        Args:
            local_entry: The local version of the memory entry
            remote_entry: The remote version of the memory entry
        
        Returns:
            A merged MemoryEntry combining non-conflicting changes
        """
        from copy import deepcopy
        
        # Determine which entry is newer
        local_time = local_entry.updated_at or local_entry.created_at
        remote_time = remote_entry.updated_at or remote_entry.created_at
        
        if local_time >= remote_time:
            newer_entry = local_entry
            older_entry = remote_entry
        else:
            newer_entry = remote_entry
            older_entry = local_entry
        
        # Start with a copy of the newer entry
        merged_entry = deepcopy(newer_entry)
        
        # Merge tags: union of both tag sets
        local_tags = set(local_entry.tags) if local_entry.tags else set()
        remote_tags = set(remote_entry.tags) if remote_entry.tags else set()
        merged_entry.tags = sorted(list(local_tags | remote_tags))
        
        # Merge context dictionaries
        # Start with older context, then overlay newer context
        merged_context = {}
        if older_entry.context:
            merged_context.update(older_entry.context)
        if newer_entry.context:
            merged_context.update(newer_entry.context)
        merged_entry.context = merged_context
        
        # Use newer summary if it exists and is different
        if newer_entry.summary:
            merged_entry.summary = newer_entry.summary
        elif older_entry.summary and not newer_entry.summary:
            # If newer doesn't have summary but older does, keep older summary
            merged_entry.summary = older_entry.summary
        
        # Update the updated_at timestamp to reflect the merge
        merged_entry.updated_at = datetime.now(UTC)
        
        return merged_entry

    
    def sync_incremental(self, since: datetime) -> Dict[str, Any]:
        """
        Perform incremental synchronization since a given timestamp.
        
        Synchronizes only entries that have been created or modified
        since the specified timestamp, reducing bandwidth and processing.
        
        Args:
            since: Timestamp to sync from (only entries modified after this)
        
        Returns:
            Dictionary containing sync results with keys:
                - synced_count: Number of entries synchronized
                - conflicts: Number of conflicts encountered
                - errors: List of error messages if any
        
        Note:
            This is a placeholder implementation for future functionality.
            In a full implementation, this would:
            1. Query local entries modified since timestamp
            2. Fetch remote entries modified since timestamp
            3. Resolve conflicts
            4. Update both local and remote storage
            5. Update sync_queue table
        """
        # TODO: Implement incremental sync
        # This would typically:
        # 1. Get local entries where updated_at > since
        # 2. Fetch remote entries where updated_at > since
        # 3. For each entry, check for conflicts
        # 4. Resolve conflicts using resolve_conflict()
        # 5. Push local changes to remote
        # 6. Pull remote changes to local
        # 7. Update sync_queue and mark entries as synced
        
        return {
            "synced_count": 0,
            "conflicts": 0,
            "errors": []
        }
    
    def mark_as_synced(self, entry_id: str) -> None:
        """
        Mark a memory entry as successfully synchronized.
        
        Updates the sync_queue table to record when the entry was synced,
        indicating that synchronization completed successfully.
        
        Args:
            entry_id: The ID of the memory entry that was synced
        
        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database update fails
            ValueError: If entry is not in sync queue
        
        Note:
            This sets the synced_at timestamp for the entry in the sync_queue.
            Entries with synced_at set are considered completed and won't be
            returned by get_pending_sync().
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot mark entry as synced."
            )
        
        import sqlite3
        
        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Update sync_queue to mark as synced
                cursor.execute("""
                    UPDATE sync_queue
                    SET synced_at = ?
                    WHERE entry_id = ? AND synced_at IS NULL
                """, (datetime.now(UTC), entry_id))
                
                if cursor.rowcount == 0:
                    raise ValueError(
                        f"Entry {entry_id} not found in sync queue or already synced"
                    )
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark entry as synced: {e}")
    
    def remove_from_queue(self, entry_id: str) -> None:
        """
        Remove a memory entry from the sync queue.
        
        Completely removes the entry from the sync_queue table. This is useful
        for cleaning up entries that no longer need to be synced or for
        removing failed sync attempts.
        
        Args:
            entry_id: The ID of the memory entry to remove from queue
        
        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database deletion fails
        
        Note:
            This permanently removes the entry from the sync queue. Use
            mark_as_synced() if you want to keep a record of successful syncs.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot remove entry from queue."
            )
        
        import sqlite3
        
        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete from sync_queue
                cursor.execute("""
                    DELETE FROM sync_queue
                    WHERE entry_id = ?
                """, (entry_id,))
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove entry from queue: {e}")
    
    def clear_synced_entries(self, older_than: Optional[datetime] = None) -> int:
        """
        Clear successfully synced entries from the sync queue.
        
        Removes entries that have been successfully synchronized to reduce
        queue size and improve performance. Can optionally only clear entries
        synced before a certain date.
        
        Args:
            older_than: Optional datetime; only clear entries synced before this time.
                       If None, clears all synced entries.
        
        Returns:
            Number of entries removed from the queue
        
        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database deletion fails
        
        Note:
            This only removes entries where synced_at IS NOT NULL. Pending
            entries (synced_at IS NULL) are never removed by this method.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot clear synced entries."
            )
        
        import sqlite3
        
        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                if older_than is None:
                    # Clear all synced entries
                    cursor.execute("""
                        DELETE FROM sync_queue
                        WHERE synced_at IS NOT NULL
                    """)
                else:
                    # Clear only entries synced before the specified time
                    cursor.execute("""
                        DELETE FROM sync_queue
                        WHERE synced_at IS NOT NULL AND synced_at < ?
                    """, (older_than,))
                
                removed_count = cursor.rowcount
                conn.commit()
                
                return removed_count
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to clear synced entries: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the sync queue.
        
        Provides information about the current state of the sync queue,
        including counts of pending and completed syncs, oldest pending
        entry, and operation breakdown.
        
        Returns:
            Dictionary containing queue statistics with keys:
                - total_entries: Total number of entries in queue
                - pending_count: Number of entries waiting to be synced
                - synced_count: Number of successfully synced entries
                - oldest_pending: Timestamp of oldest pending entry (or None)
                - operations: Dict with counts by operation type (create/update/delete)
        
        Raises:
            RuntimeError: If storage backend is not configured
            RuntimeError: If database query fails
        
        Note:
            This provides a snapshot of the queue state at query time.
            The statistics may change as sync operations progress.
        """
        # Check if storage backend is configured
        if self.storage_backend is None:
            raise RuntimeError(
                "Storage backend not configured. Cannot get queue stats."
            )
        
        import sqlite3
        
        try:
            # Get a connection from the storage backend's connection pool
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM sync_queue")
                total_entries = cursor.fetchone()[0]
                
                # Get pending count
                cursor.execute("""
                    SELECT COUNT(*) FROM sync_queue
                    WHERE synced_at IS NULL
                """)
                pending_count = cursor.fetchone()[0]
                
                # Get synced count
                cursor.execute("""
                    SELECT COUNT(*) FROM sync_queue
                    WHERE synced_at IS NOT NULL
                """)
                synced_count = cursor.fetchone()[0]
                
                # Get oldest pending entry timestamp
                cursor.execute("""
                    SELECT MIN(queued_at) FROM sync_queue
                    WHERE synced_at IS NULL
                """)
                oldest_pending_row = cursor.fetchone()
                oldest_pending = oldest_pending_row[0] if oldest_pending_row[0] else None
                
                # Get operation breakdown
                cursor.execute("""
                    SELECT operation, COUNT(*) as count
                    FROM sync_queue
                    WHERE synced_at IS NULL
                    GROUP BY operation
                """)
                operations = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_entries": total_entries,
                    "pending_count": pending_count,
                    "synced_count": synced_count,
                    "oldest_pending": oldest_pending,
                    "operations": operations
                }
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to get queue stats: {e}")
