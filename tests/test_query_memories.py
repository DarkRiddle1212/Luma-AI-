"""
Unit tests for MemoryManager.query_memories() method.
"""

import pytest
from datetime import datetime, timedelta
import tempfile
import os

from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.models import SensitivityLevel, create_memory_entry


class TestQueryMemories:
    """Test suite for MemoryManager.query_memories() method."""
    
    @pytest.fixture
    def memory_storage(self):
        """Create an in-memory storage for testing."""
        storage = MemoryStorage()
        yield storage
    
    @pytest.fixture
    def memory_manager(self, memory_storage):
        """Create a MemoryManager instance for testing."""
        manager = MemoryManager(storage=memory_storage)
        yield manager
    
    def test_query_memories_no_filters(self, memory_manager):
        """Test query_memories returns all entries when no filters are applied."""
        # Create test entries
        for i in range(5):
            entry = create_memory_entry(
                action=f"test_action_{i}",
                context={"index": i},
                device_id="device-1"
            )
            memory_manager.storage.create_entry(entry)
        
        # Query without filters
        results = memory_manager.query_memories()
        
        assert len(results) == 5
        # Results should be in reverse chronological order
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp
    
    def test_query_memories_time_range_filter(self, memory_manager):
        """Test query_memories filters by time range correctly."""
        now = datetime.now()
        
        # Create entries at different times
        entry1 = create_memory_entry(
            action="old_action",
            context={"time": "old"},
            device_id="device-1"
        )
        entry1.timestamp = now - timedelta(days=2)
        memory_manager.storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="recent_action",
            context={"time": "recent"},
            device_id="device-1"
        )
        entry2.timestamp = now - timedelta(hours=1)
        memory_manager.storage.create_entry(entry2)
        
        # Query with time range
        start_time = now - timedelta(days=1)
        results = memory_manager.query_memories(start_time=start_time)
        
        assert len(results) == 1
        assert results[0].action == "recent_action"
    
    def test_query_memories_tags_filter(self, memory_manager):
        """Test query_memories filters by tags correctly."""
        # Create entries with different tags
        entry1 = create_memory_entry(
            action="work_action",
            context={"type": "work"},
            device_id="device-1",
            tags=["work", "important"]
        )
        memory_manager.storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="personal_action",
            context={"type": "personal"},
            device_id="device-1",
            tags=["personal"]
        )
        memory_manager.storage.create_entry(entry2)
        
        # Query by tags
        results = memory_manager.query_memories(tags=["work"])
        
        assert len(results) == 1
        assert "work" in results[0].tags
    
    def test_query_memories_action_type_filter(self, memory_manager):
        """Test query_memories filters by action type."""
        # Create entries with different action types
        entry1 = create_memory_entry(
            action="file_open",
            context={"file": "test.txt"},
            device_id="device-1"
        )
        memory_manager.storage.create_entry(entry1)
        
        entry2 = create_memory_entry(
            action="browser_navigate",
            context={"url": "example.com"},
            device_id="device-1"
        )
        memory_manager.storage.create_entry(entry2)
        
        # Query by action type
        results = memory_manager.query_memories(action_type="file")
        
        assert len(results) == 1
        assert "file" in results[0].action
    
    def test_query_memories_pagination(self, memory_manager):
        """Test query_memories pagination with limit and offset."""
        # Create 10 entries
        for i in range(10):
            entry = create_memory_entry(
                action=f"action_{i}",
                context={"index": i},
                device_id="device-1"
            )
            memory_manager.storage.create_entry(entry)
        
        # Query first page
        page1 = memory_manager.query_memories(limit=3, offset=0)
        assert len(page1) == 3
        
        # Query second page
        page2 = memory_manager.query_memories(limit=3, offset=3)
        assert len(page2) == 3
        
        # Ensure no overlap
        page1_ids = {e.id for e in page1}
        page2_ids = {e.id for e in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0
    
    def test_query_memories_empty_result(self, memory_manager):
        """Test query_memories returns empty list when no matches found."""
        # Create an entry
        entry = create_memory_entry(
            action="test_action",
            context={"test": "data"},
            device_id="device-1",
            tags=["test"]
        )
        memory_manager.storage.create_entry(entry)
        
        # Query with non-matching filter
        results = memory_manager.query_memories(tags=["nonexistent"])
        
        assert results == []
