"""Tests for storage layer performance metrics collection."""

import pytest
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.models import create_memory_entry, SensitivityLevel
import tempfile
import os


class TestStorageMetrics:
    """Test storage layer metrics collection."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except:
            pass
    
    def test_cache_hit_metrics(self, temp_db):
        """Test that cache hits are tracked correctly."""
        storage = SQLiteStorage(temp_db, cache_size=10)
        
        # Create an entry
        entry = create_memory_entry(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        entry_id = storage.create_entry(entry)
        
        # First retrieval should be a cache miss (entry was just created and cached)
        # Actually, create_entry caches the entry, so this will be a hit
        initial_hits = storage._metrics['cache_hits']
        initial_misses = storage._metrics['cache_misses']
        
        # Retrieve the entry (should be a cache hit)
        retrieved = storage.get_entry(entry_id)
        assert retrieved is not None
        assert storage._metrics['cache_hits'] == initial_hits + 1
        
        # Retrieve again (should be another cache hit)
        retrieved = storage.get_entry(entry_id)
        assert storage._metrics['cache_hits'] == initial_hits + 2
    
    def test_cache_miss_metrics(self, temp_db):
        """Test that cache misses are tracked correctly."""
        storage = SQLiteStorage(temp_db, cache_size=10)
        
        # Create an entry
        entry = create_memory_entry(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        entry_id = storage.create_entry(entry)
        
        # Clear the cache to force a miss
        storage.cache.clear()
        
        initial_misses = storage._metrics['cache_misses']
        
        # Retrieve the entry (should be a cache miss)
        retrieved = storage.get_entry(entry_id)
        assert retrieved is not None
        assert storage._metrics['cache_misses'] == initial_misses + 1
    
    def test_insert_metrics(self, temp_db):
        """Test that insert operations are tracked."""
        storage = SQLiteStorage(temp_db)
        
        initial_inserts = storage._metrics['total_inserts']
        
        # Create multiple entries
        for i in range(5):
            entry = create_memory_entry(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
            storage.create_entry(entry)
        
        assert storage._metrics['total_inserts'] == initial_inserts + 5
    
    def test_query_metrics(self, temp_db):
        """Test that query operations are tracked."""
        storage = SQLiteStorage(temp_db)
        
        # Create some entries
        for i in range(3):
            entry = create_memory_entry(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
            storage.create_entry(entry)
        
        initial_queries = storage._metrics['total_queries']
        
        # Query entries
        storage.query_entries(limit=10)
        assert storage._metrics['total_queries'] == initial_queries + 1
        
        # Query again
        storage.query_entries(action_type="Test")
        assert storage._metrics['total_queries'] == initial_queries + 2
    
    def test_update_metrics(self, temp_db):
        """Test that update operations are tracked."""
        storage = SQLiteStorage(temp_db)
        
        # Create an entry
        entry = create_memory_entry(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        entry_id = storage.create_entry(entry)
        
        initial_updates = storage._metrics['total_updates']
        
        # Update the entry
        storage.update_entry(entry_id, {"tags": ["test"]})
        assert storage._metrics['total_updates'] == initial_updates + 1
        
        # Update again
        storage.update_entry(entry_id, {"summary": "Updated summary"})
        assert storage._metrics['total_updates'] == initial_updates + 2
    
    def test_delete_metrics(self, temp_db):
        """Test that delete operations are tracked."""
        storage = SQLiteStorage(temp_db)
        
        # Create an entry
        entry = create_memory_entry(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        entry_id = storage.create_entry(entry)
        
        initial_deletes = storage._metrics['total_deletes']
        
        # Delete the entry
        storage.delete_entry(entry_id)
        assert storage._metrics['total_deletes'] == initial_deletes + 1
    
    def test_storage_stats_includes_metrics(self, temp_db):
        """Test that get_storage_stats includes performance metrics."""
        storage = SQLiteStorage(temp_db)
        
        # Create some entries and perform operations
        for i in range(5):
            entry = create_memory_entry(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
            storage.create_entry(entry)
        
        # Query to generate cache hits/misses
        storage.query_entries()
        
        # Get stats
        stats = storage.get_storage_stats()
        
        # Check that storage metrics are included
        assert 'storage_metrics' in stats
        metrics = stats['storage_metrics']
        
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'total_queries' in metrics
        assert 'total_inserts' in metrics
        assert 'total_updates' in metrics
        assert 'total_deletes' in metrics
        
        # Verify values
        assert metrics['total_inserts'] == 5
        assert metrics['total_queries'] >= 1
    
    def test_cache_hit_rate_calculation(self, temp_db):
        """Test that cache hit rate is calculated correctly."""
        storage = SQLiteStorage(temp_db, cache_size=10)
        
        # Create entries
        entry_ids = []
        for i in range(5):
            entry = create_memory_entry(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
            entry_ids.append(storage.create_entry(entry))
        
        # Generate some cache hits (entries are already cached from create)
        for entry_id in entry_ids[:3]:
            storage.get_entry(entry_id)
        
        # Clear cache and generate misses
        storage.cache.clear()
        for entry_id in entry_ids[3:]:
            storage.get_entry(entry_id)
        
        # Get stats
        stats = storage.get_storage_stats()
        metrics = stats['storage_metrics']
        
        # Verify hit rate calculation
        total_requests = metrics['cache_hits'] + metrics['cache_misses']
        expected_rate = (metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        assert metrics['cache_hit_rate'] == round(expected_rate, 2)
    
    def test_metrics_initialization(self, temp_db):
        """Test that metrics are initialized to zero."""
        storage = SQLiteStorage(temp_db)
        
        assert storage._metrics['cache_hits'] == 0
        assert storage._metrics['cache_misses'] == 0
        assert storage._metrics['total_queries'] == 0
        assert storage._metrics['total_inserts'] == 0
        assert storage._metrics['total_updates'] == 0
        assert storage._metrics['total_deletes'] == 0
