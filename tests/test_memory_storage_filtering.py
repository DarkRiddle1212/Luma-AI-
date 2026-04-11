"""Comprehensive tests for query filtering functionality."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.models import create_memory_entry, SensitivityLevel


class TestQueryFiltering:
    """Comprehensive tests for query filtering by time, tags, and action type."""
    
    def test_time_filter_with_only_start_time(self):
        """Test filtering with only start_time (no end_time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries at different times
            for i in range(5):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                entry.timestamp = now - timedelta(hours=i)
                storage.create_entry(entry)
            
            # Query with only start_time (should get entries after start_time)
            start_time = now - timedelta(hours=2, minutes=30)
            results = storage.query_entries(start_time=start_time)
            
            # Should return entries 0, 1, 2 (3 entries)
            assert len(results) == 3
            for result in results:
                assert result.timestamp >= start_time
            
            storage.close()
    
    def test_time_filter_with_only_end_time(self):
        """Test filtering with only end_time (no start_time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries at different times
            for i in range(5):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                entry.timestamp = now - timedelta(hours=i)
                storage.create_entry(entry)
            
            # Query with only end_time (should get entries before end_time)
            end_time = now - timedelta(hours=2, minutes=30)
            results = storage.query_entries(end_time=end_time)
            
            # Should return entries 3, 4 (2 entries)
            assert len(results) == 2
            for result in results:
                assert result.timestamp <= end_time
            
            storage.close()
    
    def test_time_filter_exact_boundary_matches(self):
        """Test that time filter includes entries exactly at boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            start_time = now - timedelta(hours=2)
            end_time = now
            
            # Create entry exactly at start_time
            entry1 = create_memory_entry(
                action="at_start",
                context={"position": "start"},
                device_id="test_device"
            )
            entry1.timestamp = start_time
            storage.create_entry(entry1)
            
            # Create entry exactly at end_time
            entry2 = create_memory_entry(
                action="at_end",
                context={"position": "end"},
                device_id="test_device"
            )
            entry2.timestamp = end_time
            storage.create_entry(entry2)
            
            # Create entry in the middle
            entry3 = create_memory_entry(
                action="in_middle",
                context={"position": "middle"},
                device_id="test_device"
            )
            entry3.timestamp = now - timedelta(hours=1)
            storage.create_entry(entry3)
            
            # Create entries outside range
            entry4 = create_memory_entry(
                action="before_start",
                context={"position": "before"},
                device_id="test_device"
            )
            entry4.timestamp = start_time - timedelta(seconds=1)
            storage.create_entry(entry4)
            
            entry5 = create_memory_entry(
                action="after_end",
                context={"position": "after"},
                device_id="test_device"
            )
            entry5.timestamp = end_time + timedelta(seconds=1)
            storage.create_entry(entry5)
            
            # Query with time range
            results = storage.query_entries(start_time=start_time, end_time=end_time)
            
            # Should include entries at boundaries and in middle (3 entries)
            assert len(results) == 3
            result_actions = {r.action for r in results}
            assert result_actions == {"at_start", "at_end", "in_middle"}
            
            storage.close()
    
    def test_time_filter_with_no_matching_entries(self):
        """Test time filter when no entries match the time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries in the past
            for i in range(3):
                entry = create_memory_entry(
                    action=f"old_action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                entry.timestamp = now - timedelta(days=10 + i)
                storage.create_entry(entry)
            
            # Query for recent entries (should find none)
            start_time = now - timedelta(hours=1)
            end_time = now
            results = storage.query_entries(start_time=start_time, end_time=end_time)
            
            assert len(results) == 0
            assert results == []
            
            storage.close()
    
    def test_tags_filter_with_empty_list(self):
        """Test filtering with empty tags list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            # Create entries with tags
            for i in range(3):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device",
                    tags=["tag1", "tag2"]
                )
                storage.create_entry(entry)
            
            # Query with empty tags list (should return all entries)
            results = storage.query_entries(tags=[])
            assert len(results) == 3
            
            storage.close()
    
    def test_tags_filter_with_no_matching_tags(self):
        """Test tags filter when no entries have the specified tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            # Create entries with specific tags
            for i in range(3):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device",
                    tags=["work", "important"]
                )
                storage.create_entry(entry)
            
            # Query with non-existent tags
            results = storage.query_entries(tags=["nonexistent", "missing"])
            
            assert len(results) == 0
            assert results == []
            
            storage.close()
    
    def test_tags_filter_with_entries_having_no_tags(self):
        """Test tags filter with entries that have empty tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            # Create entries without tags
            for i in range(2):
                entry = create_memory_entry(
                    action=f"no_tags_{i}",
                    context={"index": i},
                    device_id="test_device",
                    tags=[]
                )
                storage.create_entry(entry)
            
            # Create entries with tags
            for i in range(2):
                entry = create_memory_entry(
                    action=f"with_tags_{i}",
                    context={"index": i},
                    device_id="test_device",
                    tags=["work"]
                )
                storage.create_entry(entry)
            
            # Query for specific tag
            results = storage.query_entries(tags=["work"])
            
            # Should only return entries with the tag
            assert len(results) == 2
            assert all("work" in r.tags for r in results)
            
            storage.close()
    
    def test_tags_filter_case_insensitivity(self):
        """Test that tags filter is case-insensitive (SQLite LIKE behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            try:
                # Create entries with different case tags
                entry1 = create_memory_entry(
                    action="action1",
                    context={"key": "value1"},
                    device_id="test_device",
                    tags=["Work", "Important"]
                )
                storage.create_entry(entry1)
                
                entry2 = create_memory_entry(
                    action="action2",
                    context={"key": "value2"},
                    device_id="test_device",
                    tags=["work", "important"]
                )
                storage.create_entry(entry2)
                
                # Query with lowercase (should match both due to case-insensitive LIKE)
                results = storage.query_entries(tags=["work"])
                assert len(results) == 2
                
                # Query with uppercase (should also match both)
                results = storage.query_entries(tags=["Work"])
                assert len(results) == 2
            finally:
                storage.close()
    
    def test_action_type_filter_case_insensitivity(self):
        """Test that action_type filter is case-insensitive (SQLite LIKE behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            try:
                # Create entries with different case actions
                entry1 = create_memory_entry(
                    action="User_Login",
                    context={"key": "value1"},
                    device_id="test_device"
                )
                storage.create_entry(entry1)
                
                entry2 = create_memory_entry(
                    action="user_login",
                    context={"key": "value2"},
                    device_id="test_device"
                )
                storage.create_entry(entry2)
                
                # Query with lowercase (should match both due to case-insensitive LIKE)
                results = storage.query_entries(action_type="user")
                assert len(results) == 2
                
                # Query with uppercase (should also match both)
                results = storage.query_entries(action_type="USER")
                assert len(results) == 2
            finally:
                storage.close()
    
    def test_action_type_filter_with_special_characters(self):
        """Test action_type filter with special characters (partial matching with LIKE)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            try:
                # Create entries with special characters in action
                actions = [
                    "user-login",
                    "user_login",
                    "user.login",
                    "user@login",
                    "file/upload"
                ]
                
                for action in actions:
                    entry = create_memory_entry(
                        action=action,
                        context={"key": "value"},
                        device_id="test_device"
                    )
                    storage.create_entry(entry)
                
                # Query with "login" - should match all user actions (partial match)
                results = storage.query_entries(action_type="login")
                assert len(results) == 4
                assert all("login" in r.action for r in results)
                
                # Query with "user" - should match all user actions
                results = storage.query_entries(action_type="user")
                assert len(results) == 4
                assert all("user" in r.action for r in results)
                
                # Query with "file" - should match file/upload only
                results = storage.query_entries(action_type="file")
                assert len(results) == 1
                assert results[0].action == "file/upload"
                
                # Query with "/" - should match file/upload only
                results = storage.query_entries(action_type="/")
                assert len(results) == 1
                assert results[0].action == "file/upload"
            finally:
                storage.close()
    
    def test_action_type_filter_with_no_matches(self):
        """Test action_type filter when no entries match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            # Create entries
            for i in range(3):
                entry = create_memory_entry(
                    action=f"user_action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                storage.create_entry(entry)
            
            # Query with non-matching action type
            results = storage.query_entries(action_type="file")
            
            assert len(results) == 0
            assert results == []
            
            storage.close()
    
    def test_action_type_filter_exact_match(self):
        """Test action_type filter with exact action name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            # Create entries
            entry1 = create_memory_entry(
                action="login",
                context={"key": "value1"},
                device_id="test_device"
            )
            storage.create_entry(entry1)
            
            entry2 = create_memory_entry(
                action="user_login",
                context={"key": "value2"},
                device_id="test_device"
            )
            storage.create_entry(entry2)
            
            # Query with exact action name (should match both due to partial matching)
            results = storage.query_entries(action_type="login")
            assert len(results) == 2
            
            storage.close()
    
    def test_combined_filters_all_match(self):
        """Test combined filters where all filters match some entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create matching entry
            entry1 = create_memory_entry(
                action="user_login",
                context={"key": "value1"},
                device_id="test_device",
                tags=["work", "auth"]
            )
            entry1.timestamp = now - timedelta(hours=1)
            storage.create_entry(entry1)
            
            # Create entry that matches time and action but not tags
            entry2 = create_memory_entry(
                action="user_logout",
                context={"key": "value2"},
                device_id="test_device",
                tags=["personal"]
            )
            entry2.timestamp = now - timedelta(minutes=30)
            storage.create_entry(entry2)
            
            # Create entry that matches time and tags but not action
            entry3 = create_memory_entry(
                action="file_upload",
                context={"key": "value3"},
                device_id="test_device",
                tags=["work"]
            )
            entry3.timestamp = now - timedelta(minutes=45)
            storage.create_entry(entry3)
            
            # Query with all filters
            results = storage.query_entries(
                start_time=now - timedelta(hours=2),
                end_time=now,
                tags=["work"],
                action_type="user"
            )
            
            # Should only return entry1
            assert len(results) == 1
            assert results[0].id == entry1.id
            
            storage.close()
    
    def test_combined_filters_no_matches(self):
        """Test combined filters where no entries match all criteria."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries that partially match
            entry1 = create_memory_entry(
                action="user_login",
                context={"key": "value1"},
                device_id="test_device",
                tags=["work"]
            )
            entry1.timestamp = now - timedelta(hours=1)
            storage.create_entry(entry1)
            
            entry2 = create_memory_entry(
                action="file_upload",
                context={"key": "value2"},
                device_id="test_device",
                tags=["personal"]
            )
            entry2.timestamp = now - timedelta(minutes=30)
            storage.create_entry(entry2)
            
            # Query with filters that don't match any entry completely
            results = storage.query_entries(
                start_time=now - timedelta(hours=2),
                end_time=now,
                tags=["personal"],
                action_type="user"
            )
            
            # Should return no entries
            assert len(results) == 0
            assert results == []
            
            storage.close()
    
    def test_combined_filters_with_multiple_tags_or_logic(self):
        """Test combined filters with multiple tags using OR logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries with different tags
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
                tags=["personal"]
            )
            entry2.timestamp = now - timedelta(minutes=30)
            storage.create_entry(entry2)
            
            entry3 = create_memory_entry(
                action="user_update",
                context={"key": "value3"},
                device_id="test_device",
                tags=["urgent"]
            )
            entry3.timestamp = now - timedelta(minutes=45)
            storage.create_entry(entry3)
            
            # Query with multiple tags (OR logic) and action filter
            results = storage.query_entries(
                start_time=now - timedelta(hours=2),
                end_time=now,
                tags=["work", "personal"],
                action_type="user"
            )
            
            # Should return entry1 and entry2 (both match action and have one of the tags)
            assert len(results) == 2
            result_ids = {r.id for r in results}
            assert result_ids == {entry1.id, entry2.id}
            
            storage.close()
    
    def test_filters_preserve_reverse_chronological_order(self):
        """Test that filtered results maintain reverse chronological order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteStorage(db_path=db_path, cache_size=10)
            
            now = datetime.now()
            
            # Create entries with specific timestamps
            for i in range(10):
                entry = create_memory_entry(
                    action="user_action",
                    context={"index": i},
                    device_id="test_device",
                    tags=["work"]
                )
                entry.timestamp = now - timedelta(hours=i)
                storage.create_entry(entry)
            
            # Query with filters
            results = storage.query_entries(
                tags=["work"],
                action_type="user"
            )
            
            # Verify all entries are returned
            assert len(results) == 10
            
            # Verify reverse chronological order (newest first)
            for i in range(len(results) - 1):
                assert results[i].timestamp >= results[i + 1].timestamp
            
            storage.close()
