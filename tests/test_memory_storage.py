"""Unit tests for in-memory storage backend."""

import pytest
from datetime import datetime, timedelta
from threading import Thread
import time

from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.backend import StorageError
from luma_memory.models import (
    create_memory_entry,
    MemoryEntry,
    SensitivityLevel,
    SyncStatus
)


class TestMemoryStorage:
    """Tests for MemoryStorage backend."""
    
    def test_initialization(self):
        """Test that MemoryStorage initializes correctly."""
        storage = MemoryStorage()
        
        # Verify storage is empty
        assert storage.get_entry_count() == 0
        
        # Verify storage stats
        stats = storage.get_storage_stats()
        assert stats["total_entries"] == 0
        assert stats["storage_size_bytes"] == 0
    
    def test_create_entry_success(self):
        """Test successful creation of a memory entry."""
        storage = MemoryStorage()
        
        # Create a test entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["test", "example"]
        )
        
        # Store the entry
        entry_id = storage.create_entry(entry)
        
        # Verify the entry was stored
        assert entry_id == entry.id
        assert storage.get_entry_count() == 1
        
        # Retrieve and verify
        retrieved = storage.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.action == "test_action"
        assert retrieved.context == {"key": "value"}
        assert retrieved.device_id == "test_device"
        assert retrieved.sensitivity == SensitivityLevel.PUBLIC
        assert retrieved.tags == ["test", "example"]
    
    def test_create_entry_with_invalid_entry(self):
        """Test that create_entry rejects invalid entries."""
        storage = MemoryStorage()
        
        # Create an invalid entry (empty action)
        entry = MemoryEntry(
            id="test_id",
            timestamp=datetime.now(),
            action="",  # Invalid: empty action
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="test_device",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        # Attempt to store should raise ValueError
        with pytest.raises(ValueError, match="Invalid memory entry"):
            storage.create_entry(entry)
        
        # Verify entry was not stored
        assert storage.get_entry_count() == 0
    
    def test_create_entry_duplicate_id(self):
        """Test that create_entry rejects duplicate entry IDs."""
        storage = MemoryStorage()
        
        # Create and store first entry
        entry1 = create_memory_entry(
            action="action1",
            context={"key": "value1"},
            device_id="device1",
            entry_id="duplicate_id"
        )
        storage.create_entry(entry1)
        
        # Create second entry with same ID
        entry2 = create_memory_entry(
            action="action2",
            context={"key": "value2"},
            device_id="device2",
            entry_id="duplicate_id"
        )
        
        # Attempt to store should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            storage.create_entry(entry2)
        
        # Verify only one entry exists
        assert storage.get_entry_count() == 1
    
    def test_get_entry_success(self):
        """Test successful retrieval of a memory entry."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        # Retrieve entry
        retrieved = storage.get_entry(entry_id)
        
        # Verify entry was retrieved correctly
        assert retrieved is not None
        assert retrieved.id == entry_id
        assert retrieved.action == "test_action"
        assert retrieved.context == {"key": "value"}
    
    def test_get_entry_not_found(self):
        """Test that get_entry returns None for non-existent entry."""
        storage = MemoryStorage()
        
        # Try to retrieve non-existent entry
        retrieved = storage.get_entry("non_existent_id")
        
        # Verify None is returned
        assert retrieved is None
    
    def test_query_entries_no_filters(self):
        """Test query_entries returns all entries when no filters are applied."""
        storage = MemoryStorage()
        
        # Create multiple entries
        entries = []
        for i in range(5):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            storage.create_entry(entry)
            entries.append(entry)
        
        # Query all entries
        results = storage.query_entries()
        
        # Verify all entries are returned
        assert len(results) == 5
        
        # Verify reverse chronological order (newest first)
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp
    
    def test_query_entries_time_range_filter(self):
        """Test query_entries filters by time range correctly."""
        storage = MemoryStorage()
        
        # Create entries with different timestamps
        now = datetime.now()
        timestamps = [
            now - timedelta(hours=3),
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now,
            now + timedelta(hours=1)
        ]
        
        for i, ts in enumerate(timestamps):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            entry.timestamp = ts
            storage.create_entry(entry)
        
        # Query with time range
        start_time = now - timedelta(hours=2, minutes=30)
        end_time = now + timedelta(minutes=30)
        
        results = storage.query_entries(start_time=start_time, end_time=end_time)
        
        # Should return entries at indices 1, 2, 3 (3 entries)
        assert len(results) == 3
        
        # Verify all results are within time range
        for result in results:
            assert result.timestamp >= start_time
            assert result.timestamp <= end_time
    
    def test_query_entries_tags_filter(self):
        """Test query_entries filters by tags correctly."""
        storage = MemoryStorage()
        
        # Create entries with different tags
        entry1 = create_memory_entry(
            action="action1",
            context={"key": "value1"},
            device_id="test_device",
            tags=["work", "important"]
        )
        storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="action2",
            context={"key": "value2"},
            device_id="test_device",
            tags=["personal", "important"]
        )
        storage.create_entry(entry2)
        
        entry3 = create_memory_entry(
            action="action3",
            context={"key": "value3"},
            device_id="test_device",
            tags=["work"]
        )
        storage.create_entry(entry3)
        
        # Query by single tag
        results = storage.query_entries(tags=["work"])
        assert len(results) == 2
        assert all("work" in r.tags for r in results)
        
        # Query by multiple tags (OR logic - at least one match)
        results = storage.query_entries(tags=["work", "personal"])
        assert len(results) == 3
    
    def test_query_entries_action_type_filter(self):
        """Test query_entries filters by action type with partial matching."""
        storage = MemoryStorage()
        
        # Create entries with different action types
        actions = ["user_login", "user_logout", "file_upload", "file_download", "system_update"]
        
        for action in actions:
            entry = create_memory_entry(
                action=action,
                context={"key": "value"},
                device_id="test_device"
            )
            storage.create_entry(entry)
        
        # Query with partial match
        results = storage.query_entries(action_type="user")
        assert len(results) == 2
        assert all("user" in r.action for r in results)
        
        # Query with different partial match
        results = storage.query_entries(action_type="file")
        assert len(results) == 2
        assert all("file" in r.action for r in results)
    
    def test_query_entries_pagination(self):
        """Test query_entries pagination with limit and offset."""
        storage = MemoryStorage()
        
        # Create 10 entries
        for i in range(10):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            storage.create_entry(entry)
        
        # Query first page
        page1 = storage.query_entries(limit=3, offset=0)
        assert len(page1) == 3
        
        # Query second page
        page2 = storage.query_entries(limit=3, offset=3)
        assert len(page2) == 3
        
        # Verify no overlap
        page1_ids = {e.id for e in page1}
        page2_ids = {e.id for e in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0
        
        # Query last page (partial)
        page4 = storage.query_entries(limit=3, offset=9)
        assert len(page4) == 1
    
    def test_query_entries_combined_filters(self):
        """Test query_entries with multiple filters combined."""
        storage = MemoryStorage()
        
        now = datetime.now()
        
        # Create entries with various attributes
        entry1 = create_memory_entry(
            action="user_login",
            context={"key": "value1"},
            device_id="test_device",
            tags=["work"]
        )
        entry1.timestamp = now - timedelta(hours=1)
        storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="user_logout",
            context={"key": "value2"},
            device_id="test_device",
            tags=["work"]
        )
        entry2.timestamp = now
        storage.create_entry(entry2)
        
        entry3 = create_memory_entry(
            action="file_upload",
            context={"key": "value3"},
            device_id="test_device",
            tags=["personal"]
        )
        entry3.timestamp = now - timedelta(hours=2)
        storage.create_entry(entry3)
        
        # Query with combined filters
        results = storage.query_entries(
            start_time=now - timedelta(hours=1, minutes=30),
            end_time=now + timedelta(minutes=30),
            tags=["work"],
            action_type="user"
        )
        
        # Should return only entry1 and entry2
        assert len(results) == 2
        assert all("user" in r.action for r in results)
        assert all("work" in r.tags for r in results)
    
    def test_query_entries_empty_result(self):
        """Test query_entries returns empty list when no matches found."""
        storage = MemoryStorage()
        
        # Create some entries
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device",
            tags=["test"]
        )
        storage.create_entry(entry)
        
        # Query with non-matching filter
        results = storage.query_entries(tags=["nonexistent"])
        
        # Should return empty list
        assert results == []
    
    def test_query_entries_invalid_limit(self):
        """Test query_entries raises ValueError for invalid limit."""
        storage = MemoryStorage()
        
        # Test with negative limit
        with pytest.raises(ValueError, match="Limit must be non-negative"):
            storage.query_entries(limit=-1)
    
    def test_query_entries_invalid_offset(self):
        """Test query_entries raises ValueError for invalid offset."""
        storage = MemoryStorage()
        
        # Test with negative offset
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            storage.query_entries(offset=-1)
    
    def test_query_entries_invalid_time_range(self):
        """Test query_entries raises ValueError when start_time > end_time."""
        storage = MemoryStorage()
        
        now = datetime.now()
        start_time = now
        end_time = now - timedelta(hours=1)
        
        # Test with invalid time range
        with pytest.raises(ValueError, match="Start time must be before or equal to end time"):
            storage.query_entries(start_time=start_time, end_time=end_time)
    
    def test_update_entry_success(self):
        """Test successful update of a memory entry."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="original_action",
            context={"key": "original_value"},
            device_id="test_device",
            tags=["original"]
        )
        entry_id = storage.create_entry(entry)
        
        # Update the entry
        updates = {
            "action": "updated_action",
            "context": {"key": "updated_value", "new_key": "new_value"},
            "tags": ["updated", "new_tag"],
            "summary": "This is a summary"
        }
        
        result = storage.update_entry(entry_id, updates)
        assert result is True
        
        # Retrieve and verify updates
        updated_entry = storage.get_entry(entry_id)
        assert updated_entry is not None
        assert updated_entry.action == "updated_action"
        assert updated_entry.context == {"key": "updated_value", "new_key": "new_value"}
        assert updated_entry.tags == ["updated", "new_tag"]
        assert updated_entry.summary == "This is a summary"
        assert updated_entry.updated_at is not None
    
    def test_update_entry_not_found(self):
        """Test that update_entry returns False for non-existent entry."""
        storage = MemoryStorage()
        
        # Try to update non-existent entry
        updates = {"action": "new_action"}
        result = storage.update_entry("non_existent_id", updates)
        
        assert result is False
    
    def test_update_entry_invalid_field(self):
        """Test that update_entry raises ValueError for invalid fields."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        # Try to update with invalid field
        updates = {"invalid_field": "value"}
        
        with pytest.raises(ValueError, match="Invalid field"):
            storage.update_entry(entry_id, updates)
    
    def test_update_entry_sensitivity_enum(self):
        """Test updating sensitivity with enum value."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device",
            sensitivity=SensitivityLevel.PUBLIC
        )
        entry_id = storage.create_entry(entry)
        
        # Update sensitivity
        updates = {"sensitivity": "sensitive"}
        result = storage.update_entry(entry_id, updates)
        assert result is True
        
        # Verify update
        updated_entry = storage.get_entry(entry_id)
        assert updated_entry.sensitivity == SensitivityLevel.SENSITIVE
    
    def test_update_entry_sync_status_enum(self):
        """Test updating sync_status with enum value."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        # Update sync status
        updates = {"sync_status": "synced"}
        result = storage.update_entry(entry_id, updates)
        assert result is True
        
        # Verify update
        updated_entry = storage.get_entry(entry_id)
        assert updated_entry.sync_status == SyncStatus.SYNCED
    
    def test_update_entry_invalid_update(self):
        """Test that update_entry validates the updated entry."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        # Try to update with invalid value (empty action)
        updates = {"action": ""}
        
        with pytest.raises(ValueError, match="Invalid update"):
            storage.update_entry(entry_id, updates)
    
    def test_delete_entry_success(self):
        """Test successful deletion of a memory entry."""
        storage = MemoryStorage()
        
        # Create and store an entry
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        # Verify entry exists
        assert storage.get_entry(entry_id) is not None
        assert storage.get_entry_count() == 1
        
        # Delete the entry
        result = storage.delete_entry(entry_id)
        assert result is True
        
        # Verify entry was deleted
        assert storage.get_entry(entry_id) is None
        assert storage.get_entry_count() == 0
    
    def test_delete_entry_not_found(self):
        """Test that delete_entry returns False for non-existent entry."""
        storage = MemoryStorage()
        
        # Try to delete non-existent entry
        result = storage.delete_entry("non_existent_id")
        
        assert result is False
    
    def test_get_storage_stats_empty(self):
        """Test get_storage_stats returns correct stats for empty storage."""
        storage = MemoryStorage()
        
        stats = storage.get_storage_stats()
        
        assert stats["total_entries"] == 0
        assert stats["storage_size_bytes"] == 0
        assert stats["oldest_entry"] is None
        assert stats["newest_entry"] is None
        assert stats["entries_by_sensitivity"] == {}
        assert stats["entries_by_sync_status"] == {}
    
    def test_get_storage_stats_with_entries(self):
        """Test get_storage_stats returns correct stats with entries."""
        storage = MemoryStorage()
        
        # Create entries with different attributes
        now = datetime.now()
        
        entry1 = create_memory_entry(
            action="action1",
            context={"key": "value1"},
            device_id="device1",
            sensitivity=SensitivityLevel.PUBLIC
        )
        entry1.timestamp = now - timedelta(hours=2)
        storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="action2",
            context={"key": "value2"},
            device_id="device2",
            sensitivity=SensitivityLevel.PRIVATE
        )
        entry2.timestamp = now - timedelta(hours=1)
        storage.create_entry(entry2)
        
        entry3 = create_memory_entry(
            action="action3",
            context={"key": "value3"},
            device_id="device3",
            sensitivity=SensitivityLevel.PUBLIC
        )
        entry3.timestamp = now
        storage.create_entry(entry3)
        
        # Get stats
        stats = storage.get_storage_stats()
        
        # Verify stats
        assert stats["total_entries"] == 3
        assert stats["storage_size_bytes"] > 0
        assert stats["oldest_entry"] == entry1.timestamp.isoformat()
        assert stats["newest_entry"] == entry3.timestamp.isoformat()
        assert stats["entries_by_sensitivity"]["public"] == 2
        assert stats["entries_by_sensitivity"]["private"] == 1
        assert stats["entries_by_sync_status"]["pending"] == 3
    
    def test_clear(self):
        """Test clear method removes all entries."""
        storage = MemoryStorage()
        
        # Create multiple entries
        for i in range(5):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            storage.create_entry(entry)
        
        # Verify entries exist
        assert storage.get_entry_count() == 5
        
        # Clear storage
        storage.clear()
        
        # Verify storage is empty
        assert storage.get_entry_count() == 0
        assert storage.query_entries() == []
    
    def test_get_entry_count(self):
        """Test get_entry_count returns correct count."""
        storage = MemoryStorage()
        
        # Initially empty
        assert storage.get_entry_count() == 0
        
        # Add entries
        for i in range(3):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            storage.create_entry(entry)
        
        assert storage.get_entry_count() == 3
        
        # Delete one entry
        entries = storage.query_entries()
        storage.delete_entry(entries[0].id)
        
        assert storage.get_entry_count() == 2
    
    def test_thread_safety_concurrent_creates(self):
        """Test thread-safety with concurrent create operations."""
        storage = MemoryStorage()
        num_threads = 10
        entries_per_thread = 10
        
        def create_entries(thread_id):
            for i in range(entries_per_thread):
                entry = create_memory_entry(
                    action=f"action_thread{thread_id}_entry{i}",
                    context={"thread": thread_id, "index": i},
                    device_id=f"device_{thread_id}"
                )
                storage.create_entry(entry)
        
        # Create threads
        threads = []
        for i in range(num_threads):
            thread = Thread(target=create_entries, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all entries were created
        assert storage.get_entry_count() == num_threads * entries_per_thread
    
    def test_thread_safety_concurrent_reads(self):
        """Test thread-safety with concurrent read operations."""
        storage = MemoryStorage()
        
        # Create some entries
        entry_ids = []
        for i in range(10):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            entry_id = storage.create_entry(entry)
            entry_ids.append(entry_id)
        
        results = []
        
        def read_entries():
            for entry_id in entry_ids:
                entry = storage.get_entry(entry_id)
                results.append(entry)
        
        # Create threads
        threads = []
        for i in range(5):
            thread = Thread(target=read_entries)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all reads succeeded
        assert len(results) == 5 * len(entry_ids)
        assert all(r is not None for r in results)
    
    def test_thread_safety_concurrent_updates(self):
        """Test thread-safety with concurrent update operations."""
        storage = MemoryStorage()
        
        # Create an entry
        entry = create_memory_entry(
            action="original_action",
            context={"counter": 0},
            device_id="test_device"
        )
        entry_id = storage.create_entry(entry)
        
        num_threads = 10
        
        def update_entry(thread_id):
            updates = {"summary": f"Updated by thread {thread_id}"}
            storage.update_entry(entry_id, updates)
        
        # Create threads
        threads = []
        for i in range(num_threads):
            thread = Thread(target=update_entry, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify entry still exists and has been updated
        updated_entry = storage.get_entry(entry_id)
        assert updated_entry is not None
        assert updated_entry.summary is not None
        assert "Updated by thread" in updated_entry.summary
    
    def test_thread_safety_mixed_operations(self):
        """Test thread-safety with mixed create, read, update, delete operations."""
        storage = MemoryStorage()
        
        # Pre-populate with some entries
        initial_entries = []
        for i in range(5):
            entry = create_memory_entry(
                action=f"initial_action_{i}",
                context={"index": i},
                device_id="test_device"
            )
            entry_id = storage.create_entry(entry)
            initial_entries.append(entry_id)
        
        def mixed_operations(thread_id):
            # Create
            entry = create_memory_entry(
                action=f"action_thread{thread_id}",
                context={"thread": thread_id},
                device_id=f"device_{thread_id}"
            )
            storage.create_entry(entry)
            
            # Read
            for entry_id in initial_entries[:2]:
                storage.get_entry(entry_id)
            
            # Update
            if initial_entries:
                storage.update_entry(initial_entries[0], {"summary": f"Thread {thread_id}"})
            
            # Query
            storage.query_entries(limit=5)
        
        # Create threads
        threads = []
        for i in range(5):
            thread = Thread(target=mixed_operations, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify storage is in a consistent state
        assert storage.get_entry_count() == 10  # 5 initial + 5 created by threads
        stats = storage.get_storage_stats()
        assert stats["total_entries"] == 10
