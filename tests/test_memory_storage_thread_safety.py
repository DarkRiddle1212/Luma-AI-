"""Thread-safety tests for in-memory storage backend."""

import pytest
import threading
import time
from datetime import datetime
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class TestMemoryStorageThreadSafety:
    """Tests for thread-safety of in-memory storage backend."""
    
    def test_concurrent_create_entries(self):
        """Test that concurrent create operations are thread-safe."""
        storage = MemoryStorage()
        errors = []
        created_ids = []
        
        def create_entry(index):
            try:
                entry = MemoryEntry(
                    id=f"test-{index}",
                    timestamp=datetime.now(),
                    action=f"test_action_{index}",
                    context={"data": f"value_{index}"},
                    sensitivity=SensitivityLevel.PUBLIC,
                    device_id="test-device",
                    sync_status=SyncStatus.PENDING,
                    tags=["test"]
                )
                entry_id = storage.create_entry(entry)
                created_ids.append(entry_id)
            except Exception as e:
                errors.append(e)
        
        # Create 50 entries concurrently
        threads = []
        for i in range(50):
            thread = threading.Thread(target=create_entry, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all entries were created
        assert len(created_ids) == 50
        assert storage.get_entry_count() == 50
    
    def test_concurrent_read_write(self):
        """Test that concurrent read and write operations are thread-safe."""
        storage = MemoryStorage()
        
        # Create initial entry
        entry = MemoryEntry(
            id="test-entry",
            timestamp=datetime.now(),
            action="test_action",
            context={"counter": 0},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="test-device",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        storage.create_entry(entry)
        
        errors = []
        read_results = []
        
        def read_entry():
            try:
                for _ in range(10):
                    result = storage.get_entry("test-entry")
                    if result:
                        read_results.append(result)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def update_entry(value):
            try:
                for _ in range(10):
                    storage.update_entry("test-entry", {"context": {"counter": value}})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start concurrent readers and writers
        threads = []
        for i in range(5):
            read_thread = threading.Thread(target=read_entry)
            write_thread = threading.Thread(target=update_entry, args=(i,))
            threads.extend([read_thread, write_thread])
            read_thread.start()
            write_thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify reads were successful
        assert len(read_results) > 0
    
    def test_concurrent_query_operations(self):
        """Test that concurrent query operations are thread-safe."""
        storage = MemoryStorage()
        
        # Create multiple entries
        for i in range(20):
            entry = MemoryEntry(
                id=f"test-{i}",
                timestamp=datetime.now(),
                action=f"test_action_{i}",
                context={"data": f"value_{i}"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="test-device",
                sync_status=SyncStatus.PENDING,
                tags=["test"]
            )
            storage.create_entry(entry)
        
        errors = []
        query_results = []
        
        def query_entries():
            try:
                for _ in range(10):
                    results = storage.query_entries(limit=10)
                    query_results.append(len(results))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start concurrent queries
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=query_entries)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify queries returned results
        assert len(query_results) == 100  # 10 threads * 10 queries each
        assert all(count == 10 for count in query_results)
    
    def test_concurrent_delete_operations(self):
        """Test that concurrent delete operations are thread-safe."""
        storage = MemoryStorage()
        
        # Create entries
        for i in range(20):
            entry = MemoryEntry(
                id=f"test-{i}",
                timestamp=datetime.now(),
                action=f"test_action_{i}",
                context={"data": f"value_{i}"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="test-device",
                sync_status=SyncStatus.PENDING,
                tags=["test"]
            )
            storage.create_entry(entry)
        
        errors = []
        delete_results = []
        
        def delete_entry(index):
            try:
                result = storage.delete_entry(f"test-{index}")
                delete_results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Delete entries concurrently
        threads = []
        for i in range(20):
            thread = threading.Thread(target=delete_entry, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all entries were deleted
        assert storage.get_entry_count() == 0
        assert all(result is True for result in delete_results)
