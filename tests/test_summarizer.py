"""Unit tests for context summarizer."""

import pytest
from datetime import datetime, timedelta
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
from luma_memory.config import MemoryModuleConfig


class TestContextSummarizer:
    """Tests for ContextSummarizer."""
    
    def test_similarity_detection_identical_entries(self):
        """Test that identical entries have high similarity."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="open_file",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
        
        similarity = summarizer._calculate_similarity(entry1, entry2)
        assert similarity >= 0.8, f"Expected similarity >= 0.8, got {similarity}"
    
    def test_similarity_detection_different_entries(self):
        """Test that completely different entries have low similarity."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(days=2),
            action="send_email",
            context={"recipient": "user@example.com"},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="phone",
            sync_status=SyncStatus.PENDING,
            tags=["communication"]
        )
        
        similarity = summarizer._calculate_similarity(entry1, entry2)
        assert similarity < 0.5, f"Expected similarity < 0.5, got {similarity}"
    
    def test_similarity_detection_similar_actions(self):
        """Test that entries with similar actions have moderate similarity."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=10),
            action="open_file",
            context={"file": "main.py"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        similarity = summarizer._calculate_similarity(entry1, entry2)
        assert 0.5 <= similarity <= 1.0, f"Expected similarity between 0.5 and 1.0, got {similarity}"
    
    def test_similarity_detection_context_overlap(self):
        """Test that context key overlap affects similarity."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="edit_file",
            context={"file": "test.py", "line": 10, "column": 5},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=2),
            action="edit_file",
            context={"file": "test.py", "line": 15},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        similarity = summarizer._calculate_similarity(entry1, entry2)
        assert similarity > 0.6, f"Expected similarity > 0.6 due to context overlap, got {similarity}"
    
    def test_similarity_detection_temporal_proximity(self):
        """Test that temporal proximity affects similarity."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Recent entries
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        entry2_recent = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=30),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        entry3_old = MemoryEntry(
            id="3",
            timestamp=now + timedelta(days=2),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        similarity_recent = summarizer._calculate_similarity(entry1, entry2_recent)
        similarity_old = summarizer._calculate_similarity(entry1, entry3_old)
        
        assert similarity_recent > similarity_old, "Recent entries should have higher similarity"
    
    def test_identify_redundant_entries_groups_similar_entries(self):
        """Test that identify_redundant_entries correctly groups similar entries."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create similar entries that should be grouped
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="open_file",
            context={"file": "test.py", "line": 15},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
        
        # Create a different entry that should not be grouped
        entry3 = MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=10),
            action="send_email",
            context={"recipient": "user@example.com"},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="phone",
            sync_status=SyncStatus.PENDING,
            tags=["communication"]
        )
        
        groups = summarizer.identify_redundant_entries([entry1, entry2, entry3])
        
        # Should have one group with entry1 and entry2
        assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
        summary_id, entry_ids = groups[0]
        assert len(entry_ids) == 2, f"Expected 2 entries in group, got {len(entry_ids)}"
        assert "1" in entry_ids and "2" in entry_ids, "Expected entries 1 and 2 to be grouped"
    
    def test_identify_redundant_entries_empty_list(self):
        """Test that identify_redundant_entries handles empty list."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        groups = summarizer.identify_redundant_entries([])
        assert groups == [], "Expected empty list for empty input"
    
    def test_identify_redundant_entries_no_similar_entries(self):
        """Test that identify_redundant_entries returns empty when no entries are similar."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(days=1),
            action="send_email",
            context={"recipient": "user@example.com"},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="phone",
            sync_status=SyncStatus.PENDING,
            tags=["communication"]
        )
        
        groups = summarizer.identify_redundant_entries([entry1, entry2])
        assert groups == [], "Expected no groups when entries are not similar"
    
    def test_summarize_entries_basic(self):
        """Test basic summarization of multiple entries."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="edit_file",
            context={"file": "test.py", "line": 15},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify summary entry structure
        assert summary.id.startswith("summary-"), "Summary ID should start with 'summary-'"
        assert summary.timestamp == now, "Summary timestamp should be earliest timestamp"
        assert "Summary of 2 entries" in summary.action, "Action should indicate number of entries"
        assert summary.sensitivity == SensitivityLevel.PUBLIC, "Should use highest sensitivity"
        assert summary.device_id == "laptop", "Should use first device_id"
        assert summary.sync_status == SyncStatus.PENDING, "Should be marked as pending sync"
        assert "coding" in summary.tags and "python" in summary.tags, "Should merge all tags"
        assert summary.summary is not None, "Should have summary text"
        assert summary.parent_id is None, "Summary should not have parent_id"
        
        # Verify entry IDs to link
        assert len(entry_ids) == 2, "Should return 2 entry IDs to link"
        assert "1" in entry_ids and "2" in entry_ids, "Should include both entry IDs"
    
    def test_summarize_entries_empty_list(self):
        """Test that summarize_entries raises error for empty list."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        with pytest.raises(ValueError, match="Cannot summarize empty list"):
            summarizer.summarize_entries([])
    
    def test_summarize_entries_merges_contexts(self):
        """Test that contexts are properly merged."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="action1",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["tag1"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="action2",
            context={"file": "test.py", "column": 5},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["tag2"]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify context merging
        assert "file" in summary.context, "Should preserve file key"
        assert "line" in summary.context, "Should preserve line key"
        assert "column" in summary.context, "Should preserve column key"
        assert summary.context["file"] == "test.py", "Should preserve file value"
        
        # Verify entry IDs
        assert entry_ids == ["1", "2"], "Should return entry IDs in chronological order"
    
    def test_summarize_entries_uses_highest_sensitivity(self):
        """Test that summary uses the highest sensitivity level."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="action1",
            context={"key": "value1"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="action2",
            context={"key": "value2"},
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry3 = MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=10),
            action="action3",
            context={"key": "value3"},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
        
        assert summary.sensitivity == SensitivityLevel.SENSITIVE, "Should use highest sensitivity level"
        assert len(entry_ids) == 3, "Should return all 3 entry IDs"
    
    def test_summarize_entries_preserves_unique_tags(self):
        """Test that all unique tags are preserved."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="action1",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["tag1", "tag2", "common"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="action2",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["tag3", "common"]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Should have all unique tags
        assert set(summary.tags) == {"common", "tag1", "tag2", "tag3"}, "Should preserve all unique tags"
        assert len(entry_ids) == 2, "Should return 2 entry IDs"
    
    def test_summarize_entries_chronological_order(self):
        """Test that entries are processed in chronological order."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create entries out of order
        entry1 = MemoryEntry(
            id="1",
            timestamp=now + timedelta(minutes=10),
            action="action1",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now,
            action="action2",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry3 = MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=5),
            action="action3",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
        
        # Should use earliest timestamp
        assert summary.timestamp == now, "Should use earliest timestamp"
        # Should use latest timestamp for created_at
        assert summary.created_at == now + timedelta(minutes=10), "Should use latest timestamp for created_at"
        # Entry IDs should be in chronological order
        assert entry_ids == ["2", "3", "1"], "Should return entry IDs in chronological order"
    
    def test_summarize_entries_single_entry(self):
        """Test summarization of a single entry."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry = MemoryEntry(
            id="1",
            timestamp=now,
            action="single_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["tag1"]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry])
        
        assert summary.id.startswith("summary-"), "Should have summary ID"
        assert "Summary of 1 entries" in summary.action or "Summary of 1 entry" in summary.action, "Should indicate single entry"
        assert summary.sensitivity == SensitivityLevel.PRIVATE, "Should preserve sensitivity"
        assert summary.tags == ["tag1"], "Should preserve tags"
        assert entry_ids == ["1"], "Should return single entry ID"
    
    def test_summarize_entries_returns_entry_ids_for_linking(self):
        """Test that summarize_entries returns entry IDs that should be linked to the summary."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = []
        for i in range(5):
            entry = MemoryEntry(
                id=f"entry-{i}",
                timestamp=now + timedelta(minutes=i),
                action=f"action-{i}",
                context={"index": i},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[f"tag-{i}"]
            )
            entries.append(entry)
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify all entry IDs are returned
        assert len(entry_ids) == 5, "Should return all 5 entry IDs"
        assert entry_ids == ["entry-0", "entry-1", "entry-2", "entry-3", "entry-4"], \
            "Should return entry IDs in chronological order"
        
        # Verify summary has no parent_id (it's the parent)
        assert summary.parent_id is None, "Summary should not have a parent_id"
        
        # Verify summary metadata includes entry IDs
        assert "_summary_metadata" in summary.context, "Should have summary metadata"
        assert summary.context["_summary_metadata"]["entry_ids"] == entry_ids, \
            "Metadata should include entry IDs"
    
    def test_summarize_entries_preserves_chronological_order_in_entry_ids(self):
        """Test that entry IDs are returned in chronological order regardless of input order."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create entries with timestamps out of order
        entry1 = MemoryEntry(
            id="latest",
            timestamp=now + timedelta(minutes=20),
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="earliest",
            timestamp=now,
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry3 = MemoryEntry(
            id="middle",
            timestamp=now + timedelta(minutes=10),
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        # Pass entries in non-chronological order
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
        
        # Entry IDs should be in chronological order
        assert entry_ids == ["earliest", "middle", "latest"], \
            "Entry IDs should be sorted chronologically"

    
    def test_summarize_entries_preserves_essential_context_information(self):
        """Test that essential context information is preserved in summaries."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create entries with important context that should be preserved
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="edit_file",
            context={"file": "important.py", "line": 10, "user": "alice"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="edit_file",
            context={"file": "important.py", "line": 20, "user": "alice"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding"]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify essential information is preserved
        assert "file" in summary.context, "File information should be preserved"
        assert summary.context["file"] == "important.py", "File name should be preserved"
        assert "user" in summary.context, "User information should be preserved"
        
        # Verify metadata is present
        assert "_summary_metadata" in summary.context, "Summary metadata should be present"
        assert summary.context["_summary_metadata"]["entry_count"] == 2
        assert summary.context["_summary_metadata"]["entry_ids"] == ["1", "2"]
    
    def test_summarize_entries_tracks_value_changes(self):
        """Test that value changes are tracked in value history."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create entries where a value changes
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="status_update",
            context={"status": "pending", "priority": "low"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="status_update",
            context={"status": "in_progress", "priority": "low"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry3 = MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=10),
            action="status_update",
            context={"status": "completed", "priority": "low"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
        
        # Verify value history is tracked for changed fields
        assert "_value_history" in summary.context, "Value history should be tracked"
        assert "status" in summary.context["_value_history"], "Status changes should be tracked"
        assert summary.context["_value_history"]["status"] == ["pending", "in_progress", "completed"]
        
        # Verify unchanged field doesn't have history
        assert "priority" not in summary.context["_value_history"], "Unchanged fields shouldn't have history"
        
        # Verify most recent value is used
        assert summary.context["status"] == "completed", "Should use most recent status"
    
    def test_summarize_entries_merges_list_contexts(self):
        """Test that list values in context are properly merged."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="process_items",
            context={"items": ["item1", "item2"]},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="process_items",
            context={"items": ["item2", "item3"]},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify lists are merged with unique values preserved
        assert "items" in summary.context
        assert set(summary.context["items"]) == {"item1", "item2", "item3"}
        assert len(summary.context["items"]) == 3, "Should have 3 unique items"
    
    def test_summarize_entries_merges_nested_dict_contexts(self):
        """Test that nested dictionary contexts are properly merged."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="config_update",
            context={"settings": {"theme": "dark", "font_size": 12}},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="config_update",
            context={"settings": {"theme": "dark", "line_height": 1.5}},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify nested dicts are merged
        assert "settings" in summary.context
        assert isinstance(summary.context["settings"], dict)
        assert summary.context["settings"]["theme"] == "dark"
        assert "font_size" in summary.context["settings"]
        assert "line_height" in summary.context["settings"]
    
    def test_summarize_entries_summary_text_includes_key_information(self):
        """Test that summary text includes all key information."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = []
        for i in range(3):
            entry = MemoryEntry(
                id=f"entry-{i}",
                timestamp=now + timedelta(minutes=i * 5),
                action=f"action_{i}",
                context={"common_field": f"value_{i}"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[f"tag{i}"]
            )
            entries.append(entry)
        
        summary, entry_ids = summarizer.summarize_entries(entries)
        
        # Verify summary text contains key information
        assert summary.summary is not None
        assert "3 related entries" in summary.summary or "3 entries" in summary.summary
        assert "action_0" in summary.summary or "action_1" in summary.summary or "action_2" in summary.summary
        assert "laptop" in summary.summary
        assert "common_field" in summary.summary
    
    def test_summarize_entries_handles_multiple_devices(self):
        """Test that summaries properly handle entries from multiple devices."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=now,
            action="sync_action",
            context={"data": "value1"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="sync_action",
            context={"data": "value2"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="phone",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry3 = MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=10),
            action="sync_action",
            context={"data": "value3"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="tablet",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
        
        # Verify metadata includes all devices
        assert "_summary_metadata" in summary.context
        assert "devices" in summary.context["_summary_metadata"]
        assert set(summary.context["_summary_metadata"]["devices"]) == {"laptop", "phone", "tablet"}
        
        # Verify summary text mentions multiple devices
        assert "Devices:" in summary.summary or "laptop" in summary.summary
    
    def test_summarize_entries_preserves_time_range(self):
        """Test that time range is properly preserved in summary metadata."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        start_time = now
        end_time = now + timedelta(hours=2)
        
        entry1 = MemoryEntry(
            id="1",
            timestamp=start_time,
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        entry2 = MemoryEntry(
            id="2",
            timestamp=end_time,
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids = summarizer.summarize_entries([entry1, entry2])
        
        # Verify time range in metadata
        assert "_summary_metadata" in summary.context
        assert "time_range" in summary.context["_summary_metadata"]
        assert summary.context["_summary_metadata"]["time_range"]["start"] == start_time.isoformat()
        assert summary.context["_summary_metadata"]["time_range"]["end"] == end_time.isoformat()
        
        # Verify summary uses earliest timestamp
        assert summary.timestamp == start_time
        
        # Verify created_at uses latest timestamp
        assert summary.created_at == end_time
    
    def test_should_trigger_summarization_entry_count_threshold(self):
        """Test that summarization triggers when entry count exceeds threshold."""
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=1000,
            storage_size_threshold_mb=100
        )
        
        # Below threshold
        assert not summarizer.should_trigger_summarization(999, 50_000_000)
        
        # At threshold
        assert summarizer.should_trigger_summarization(1000, 50_000_000)
        
        # Above threshold
        assert summarizer.should_trigger_summarization(1500, 50_000_000)
    
    def test_should_trigger_summarization_storage_size_threshold(self):
        """Test that summarization triggers when storage size exceeds threshold."""
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=1000,
            storage_size_threshold_mb=100
        )
        
        # Below threshold (100 MB = 100_000_000 bytes)
        assert not summarizer.should_trigger_summarization(500, 99_999_999)
        
        # At threshold
        assert summarizer.should_trigger_summarization(500, 100_000_000)
        
        # Above threshold
        assert summarizer.should_trigger_summarization(500, 150_000_000)
    
    def test_should_trigger_summarization_either_threshold(self):
        """Test that summarization triggers when either threshold is exceeded."""
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=1000,
            storage_size_threshold_mb=100
        )
        
        # Neither threshold exceeded
        assert not summarizer.should_trigger_summarization(500, 50_000_000)
        
        # Only entry count threshold exceeded
        assert summarizer.should_trigger_summarization(1500, 50_000_000)
        
        # Only storage size threshold exceeded
        assert summarizer.should_trigger_summarization(500, 150_000_000)
        
        # Both thresholds exceeded
        assert summarizer.should_trigger_summarization(1500, 150_000_000)
    
    def test_should_trigger_summarization_custom_thresholds(self):
        """Test that custom thresholds work correctly."""
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=500,
            storage_size_threshold_mb=50
        )
        
        # Below custom thresholds
        assert not summarizer.should_trigger_summarization(499, 49_999_999)
        
        # At custom entry count threshold
        assert summarizer.should_trigger_summarization(500, 10_000_000)
        
        # At custom storage size threshold (50 MB = 50_000_000 bytes)
        assert summarizer.should_trigger_summarization(100, 50_000_000)
    
    def test_should_trigger_summarization_zero_values(self):
        """Test that zero values don't trigger summarization."""
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=1000,
            storage_size_threshold_mb=100
        )
        
        assert not summarizer.should_trigger_summarization(0, 0)
    
    def test_init_validates_thresholds(self):
        """Test that __init__ validates threshold parameters."""
        # Valid thresholds
        summarizer = ContextSummarizer(
            similarity_threshold=0.8,
            entry_count_threshold=1000,
            storage_size_threshold_mb=100
        )
        assert summarizer.entry_count_threshold == 1000
        assert summarizer.storage_size_threshold_bytes == 100_000_000
        
        # Invalid entry count threshold
        with pytest.raises(ValueError, match="entry_count_threshold must be a positive integer"):
            ContextSummarizer(entry_count_threshold=0)
        
        with pytest.raises(ValueError, match="entry_count_threshold must be a positive integer"):
            ContextSummarizer(entry_count_threshold=-1)
        
        # Invalid storage size threshold
        with pytest.raises(ValueError, match="storage_size_threshold_mb must be a positive integer"):
            ContextSummarizer(storage_size_threshold_mb=0)
        
        with pytest.raises(ValueError, match="storage_size_threshold_mb must be a positive integer"):
            ContextSummarizer(storage_size_threshold_mb=-1)
    
    def test_from_config_creates_summarizer_with_config_values(self):
        """Test that from_config creates a summarizer with values from config."""
        # Create a config with custom values
        config = MemoryModuleConfig(
            similarity_threshold=0.75,
            summarization_threshold=500,
            max_storage_size_mb=50
        )
        
        # Create summarizer from config
        summarizer = ContextSummarizer.from_config(config)
        
        # Verify that config values are used
        assert summarizer.similarity_threshold == 0.75
        assert summarizer.entry_count_threshold == 500
        assert summarizer.storage_size_threshold_bytes == 50_000_000
    
    def test_from_config_uses_default_config_values(self):
        """Test that from_config works with default config values."""
        # Create config with defaults
        config = MemoryModuleConfig()
        
        # Create summarizer from config
        summarizer = ContextSummarizer.from_config(config)
        
        # Verify default values
        assert summarizer.similarity_threshold == 0.8
        assert summarizer.entry_count_threshold == 1000
        assert summarizer.storage_size_threshold_bytes == 1000_000_000  # 1000 MB
    
    def test_from_config_respects_config_validation(self):
        """Test that from_config respects config validation rules."""
        # Config validation should prevent invalid similarity_threshold
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            MemoryModuleConfig(similarity_threshold=1.5)
        
        # Config validation should prevent invalid thresholds
        with pytest.raises(ValueError, match="must be a positive integer"):
            MemoryModuleConfig(summarization_threshold=-1)

