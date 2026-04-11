"""Tests for parent-child linking in summarization."""

import pytest
from datetime import datetime, timedelta
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class TestParentChildLinking:
    """Tests that verify parent-child linking functionality in summarization."""
    
    def test_summarize_entries_returns_entry_ids_for_parent_linking(self):
        """Test that summarize_entries returns entry IDs that should link to the summary as parent."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="child-1",
                timestamp=now,
                action="action1",
                context={"key": "value1"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag1"]
            ),
            MemoryEntry(
                id="child-2",
                timestamp=now + timedelta(minutes=5),
                action="action2",
                context={"key": "value2"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag2"]
            ),
            MemoryEntry(
                id="child-3",
                timestamp=now + timedelta(minutes=10),
                action="action3",
                context={"key": "value3"},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["tag3"]
            )
        ]
        
        summary, entry_ids_to_link = summarizer.summarize_entries(entries)
        
        # Verify the summary is created correctly
        assert summary.id.startswith("summary-"), "Summary should have summary- prefix"
        assert summary.parent_id is None, "Summary should not have a parent (it IS the parent)"
        
        # Verify all child entry IDs are returned for linking
        assert len(entry_ids_to_link) == 3, "Should return all 3 child entry IDs"
        assert set(entry_ids_to_link) == {"child-1", "child-2", "child-3"}, \
            "Should return all child entry IDs"
        
        # Verify entry IDs are in chronological order
        assert entry_ids_to_link == ["child-1", "child-2", "child-3"], \
            "Entry IDs should be in chronological order"
    
    def test_parent_child_relationship_workflow(self):
        """Test the complete workflow of creating parent-child relationships."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Step 1: Create original entries (children)
        child_entries = [
            MemoryEntry(
                id="original-1",
                timestamp=now,
                action="open_file",
                context={"file": "test.py", "line": 10},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["coding"],
                parent_id=None  # Initially no parent
            ),
            MemoryEntry(
                id="original-2",
                timestamp=now + timedelta(minutes=5),
                action="open_file",
                context={"file": "test.py", "line": 20},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=["coding"],
                parent_id=None  # Initially no parent
            )
        ]
        
        # Step 2: Create summary (parent)
        summary, entry_ids_to_link = summarizer.summarize_entries(child_entries)
        
        # Step 3: Verify the parent-child relationship structure
        # The summary is the parent
        assert summary.parent_id is None, "Parent summary should not have a parent_id"
        
        # The original entries should be updated to link to this summary
        assert entry_ids_to_link == ["original-1", "original-2"], \
            "Should return IDs of entries that need to be linked to this parent"
        
        # Step 4: Simulate updating the child entries with parent_id
        # (In real usage, the storage layer would update these entries)
        for entry in child_entries:
            if entry.id in entry_ids_to_link:
                entry.parent_id = summary.id
        
        # Step 5: Verify the relationship is established
        for entry in child_entries:
            assert entry.parent_id == summary.id, \
                f"Child entry {entry.id} should link to parent {summary.id}"
    
    def test_summary_metadata_includes_child_entry_ids(self):
        """Test that summary metadata includes child entry IDs for traceability."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entries = [
            MemoryEntry(
                id="entry-a",
                timestamp=now,
                action="action",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="entry-b",
                timestamp=now + timedelta(minutes=5),
                action="action",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids_to_link = summarizer.summarize_entries(entries)
        
        # Verify metadata includes entry IDs
        assert "_summary_metadata" in summary.context, "Summary should have metadata"
        metadata = summary.context["_summary_metadata"]
        assert "entry_ids" in metadata, "Metadata should include entry_ids"
        
        # The metadata entry_ids should match the returned entry_ids_to_link
        assert metadata["entry_ids"] == entry_ids_to_link, \
            "Metadata entry_ids should match returned entry_ids_to_link"
    
    def test_multiple_summaries_can_exist_independently(self):
        """Test that multiple summaries can be created without interfering with each other."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create first group of entries
        group1 = [
            MemoryEntry(
                id="g1-1",
                timestamp=now,
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="g1-2",
                timestamp=now + timedelta(minutes=5),
                action="action1",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        # Create second group of entries
        group2 = [
            MemoryEntry(
                id="g2-1",
                timestamp=now + timedelta(hours=1),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="g2-2",
                timestamp=now + timedelta(hours=1, minutes=5),
                action="action2",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        # Create summaries for both groups
        summary1, entry_ids1 = summarizer.summarize_entries(group1)
        summary2, entry_ids2 = summarizer.summarize_entries(group2)
        
        # Verify summaries are independent
        assert summary1.id != summary2.id, "Summaries should have different IDs"
        assert summary1.parent_id is None, "Summary 1 should not have parent"
        assert summary2.parent_id is None, "Summary 2 should not have parent"
        
        # Verify entry IDs are correctly separated
        assert set(entry_ids1) == {"g1-1", "g1-2"}, "Group 1 entry IDs should be correct"
        assert set(entry_ids2) == {"g2-1", "g2-2"}, "Group 2 entry IDs should be correct"
        assert set(entry_ids1).isdisjoint(set(entry_ids2)), \
            "Entry ID sets should not overlap"
    
    def test_entry_ids_maintain_chronological_order_for_linking(self):
        """Test that entry IDs are returned in chronological order for proper linking."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        # Create entries with non-sequential timestamps
        entries = [
            MemoryEntry(
                id="third",
                timestamp=now + timedelta(minutes=20),
                action="action",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="first",
                timestamp=now,
                action="action",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            ),
            MemoryEntry(
                id="second",
                timestamp=now + timedelta(minutes=10),
                action="action",
                context={},
                sensitivity=SensitivityLevel.PUBLIC,
                device_id="laptop",
                sync_status=SyncStatus.PENDING,
                tags=[]
            )
        ]
        
        summary, entry_ids_to_link = summarizer.summarize_entries(entries)
        
        # Entry IDs should be in chronological order
        assert entry_ids_to_link == ["first", "second", "third"], \
            "Entry IDs should be sorted chronologically for consistent linking"
    
    def test_single_entry_summarization_returns_single_id(self):
        """Test that summarizing a single entry returns a single ID for linking."""
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        now = datetime.now()
        
        entry = MemoryEntry(
            id="solo-entry",
            timestamp=now,
            action="action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        summary, entry_ids_to_link = summarizer.summarize_entries([entry])
        
        assert len(entry_ids_to_link) == 1, "Should return single entry ID"
        assert entry_ids_to_link[0] == "solo-entry", "Should return correct entry ID"
        assert summary.parent_id is None, "Summary should not have parent"
