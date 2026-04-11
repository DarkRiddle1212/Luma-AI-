"""Tests for essential information preservation in summaries."""

import pytest
from datetime import datetime, timedelta
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class TestEssentialInformationPreservation:
    """Tests that verify essential information is preserved in summaries."""
    
    def test_summary_preserves_entry_ids(self):
        """Test that summary metadata includes all original entry IDs."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="entry-1",
                timestamp=now,
                action="action1",
                context={"key": "value1"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag1"]
            ),
            MemoryEntry(
                id="entry-2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={"key": "value2"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag2"]
            ),
            MemoryEntry(
                id="entry-3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={"key": "value3"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag3"]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify metadata is present
        assert "_summary_metadata" in summary.context, "Summary should contain metadata"
        metadata = summary.context["_summary_metadata"]
        
        # Verify all entry IDs are preserved
        assert "entry_ids" in metadata, "Metadata should contain entry IDs"
        assert set(metadata["entry_ids"]) == {"entry-1", "entry-2", "entry-3"}, \
            "All entry IDs should be preserved"
    
    def test_summary_preserves_time_range(self):
        """Test that summary metadata includes time range information."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(hours=2),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        metadata = summary.context["_summary_metadata"]
        
        # Verify time range is preserved
        assert "time_range" in metadata, "Metadata should contain time range"
        assert "start" in metadata["time_range"], "Time range should have start"
        assert "end" in metadata["time_range"], "Time range should have end"
        
        # Verify time range values
        start_time = datetime.fromisoformat(metadata["time_range"]["start"])
        end_time = datetime.fromisoformat(metadata["time_range"]["end"])
        
        assert abs((start_time - now).total_seconds()) < 1, "Start time should match earliest entry"
        assert abs((end_time - (now + timedelta(hours=2))).total_seconds()) < 1, \
            "End time should match latest entry"
    
    def test_summary_preserves_unique_actions(self):
        """Test that summary metadata includes all unique actions."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="open_file",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="edit_file",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="3",
                timestamp=now + timedelta(minutes=10),
                action="open_file",  # Duplicate action
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        metadata = summary.context["_summary_metadata"]
        
        # Verify unique actions are preserved
        assert "unique_actions" in metadata, "Metadata should contain unique actions"
        assert set(metadata["unique_actions"]) == {"open_file", "edit_file"}, \
            "All unique actions should be preserved"
    
    def test_summary_preserves_device_information(self):
        """Test that summary metadata includes all devices involved."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="phone",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",  # Duplicate device
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        metadata = summary.context["_summary_metadata"]
        
        # Verify devices are preserved
        assert "devices" in metadata, "Metadata should contain devices"
        assert set(metadata["devices"]) == {"laptop", "phone"}, \
            "All unique devices should be preserved"
    
    def test_summary_preserves_sensitivity_levels(self):
        """Test that summary metadata includes all sensitivity levels."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PRIVATE,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={},
                sensitivity=SensitivityLevel.SENSITIVE,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        metadata = summary.context["_summary_metadata"]
        
        # Verify sensitivity levels are preserved
        assert "sensitivity_levels" in metadata, "Metadata should contain sensitivity levels"
        assert set(metadata["sensitivity_levels"]) == {"public", "private", "sensitive"}, \
            "All unique sensitivity levels should be preserved"
    
    def test_summary_preserves_context_value_history(self):
        """Test that context merging preserves value history for changed fields."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={"status": "pending", "count": 1},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={"status": "processing", "count": 2},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={"status": "completed", "count": 3},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify value history is preserved
        assert "_value_history" in summary.context, \
            "Context should contain value history for changed fields"
        
        value_history = summary.context["_value_history"]
        
        # Verify status history
        assert "status" in value_history, "Value history should track status changes"
        assert value_history["status"] == ["pending", "processing", "completed"], \
            "Status history should preserve all values in order"
        
        # Verify count history
        assert "count" in value_history, "Value history should track count changes"
        assert value_history["count"] == [1, 2, 3], \
            "Count history should preserve all values in order"
    
    def test_summary_text_includes_device_info(self):
        """Test that summary text includes device information."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="phone",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify summary text includes device information
        assert "Devices:" in summary.summary or "Device:" in summary.summary, \
            "Summary text should include device information"
        assert "laptop" in summary.summary and "phone" in summary.summary, \
            "Summary text should list all devices"
    
    def test_summary_text_includes_common_context_keys(self):
        """Test that summary text includes common context keys."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={"file": "test.py", "line": 10},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={"file": "test.py", "line": 20},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={"file": "main.py", "line": 5},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify summary text includes common context keys
        assert "Common context:" in summary.summary, \
            "Summary text should include common context keys"
        assert "file" in summary.summary and "line" in summary.summary, \
            "Summary text should list common context keys"
    
    def test_context_merge_preserves_list_order(self):
        """Test that list merging preserves order while deduplicating."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="1",
                timestamp=now,
                action="action1",
                context={"items": ["a", "b", "c"]},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={"items": ["b", "d", "e"]},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify list merging preserves order and deduplicates
        assert "items" in summary.context, "Context should contain items list"
        items = summary.context["items"]
        
        # Should have all unique items
        assert set(items) == {"a", "b", "c", "d", "e"}, \
            "All unique items should be preserved"
        
        # Should preserve order (a, b, c from first, then d, e from second, no duplicate b)
        assert items == ["a", "b", "c", "d", "e"], \
            "List order should be preserved while deduplicating"
