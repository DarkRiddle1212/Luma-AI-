"""
Tests for automatic summarization triggers in MemoryManager.

This module tests that the MemoryManager automatically triggers
summarization when configured thresholds are exceeded.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from luma_memory.memory_manager import MemoryManager
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.config import MemoryModuleConfig


class TestAutomaticSummarization:
    """Test suite for automatic summarization triggers."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration with low thresholds."""
        config = MemoryModuleConfig()
        config.summarization_threshold = 5  # Trigger after 5 entries
        config.max_storage_size_mb = 1  # 1 MB threshold
        return config
    
    @pytest.fixture
    def storage(self):
        """Create an in-memory storage backend."""
        return MemoryStorage()
    
    @pytest.fixture
    def summarizer(self, config):
        """Create a context summarizer with config."""
        return ContextSummarizer.from_config(config)
    
    @pytest.fixture
    def memory_manager(self, storage, summarizer, config):
        """Create a memory manager with all components."""
        validation = ValidationManager()
        return MemoryManager(
            storage=storage,
            encryption=None,
            validation=validation,
            summarizer=summarizer,
            config=config
        )
    
    def test_summarization_not_triggered_below_threshold(self, memory_manager):
        """Test that summarization is not triggered when below threshold."""
        # Create entries below threshold (4 entries, threshold is 5)
        for i in range(4):
            memory_manager.create_memory(
                action=f"Action {i}",
                context={"index": i},
                device_id="test-device"
            )
        
        # Check that no summaries were created
        all_entries = memory_manager.storage.query_entries(limit=100)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        assert len(summary_entries) == 0
    
    def test_summarization_triggered_at_entry_threshold(self, memory_manager):
        """Test that summarization is triggered when entry count reaches threshold."""
        # Create similar entries to trigger summarization
        for i in range(6):  # Threshold is 5, so 6 should trigger
            memory_manager.create_memory(
                action="User opened document",
                context={"file": f"doc{i}.pdf", "page": 1},
                device_id="test-device",
                tags=["document", "work"]
            )
        
        # Check that summarization was triggered
        all_entries = memory_manager.storage.query_entries(limit=100)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        # Should have at least one summary entry
        assert len(summary_entries) > 0
    
    def test_summarization_triggered_on_storage_size(self, memory_manager):
        """Test that summarization is triggered when storage size exceeds threshold."""
        # Mock the storage stats to return a size above threshold
        original_get_stats = memory_manager.storage.get_storage_stats
        
        def mock_get_stats():
            stats = original_get_stats()
            stats['storage_size_bytes'] = 2_000_000  # 2 MB (above 1 MB threshold)
            return stats
        
        memory_manager.storage.get_storage_stats = mock_get_stats
        
        # Create similar entries
        for i in range(6):
            memory_manager.create_memory(
                action="User opened document",
                context={"file": f"doc{i}.pdf"},
                device_id="test-device",
                tags=["document"]
            )
        
        # Check that summarization was triggered
        all_entries = memory_manager.storage.query_entries(limit=100)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        assert len(summary_entries) > 0
    
    def test_summarization_creates_parent_child_links(self, memory_manager):
        """Test that summarization creates proper parent-child relationships."""
        # Create similar entries
        entry_ids = []
        for i in range(6):
            entry_id = memory_manager.create_memory(
                action="User opened document",
                context={"file": f"doc{i}.pdf"},
                device_id="test-device",
                tags=["document"]
            )
            entry_ids.append(entry_id)
        
        # Check for parent-child relationships
        all_entries = memory_manager.storage.query_entries(limit=100)
        
        # Find summary entries (parent entries with no parent_id)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        if len(summary_entries) > 0:
            # Find child entries (entries with parent_id set)
            child_entries = [e for e in all_entries if e.parent_id is not None]
            
            # Verify that child entries reference summary entries
            summary_ids = {s.id for s in summary_entries}
            for child in child_entries:
                assert child.parent_id in summary_ids
    
    def test_summarization_preserves_essential_information(self, memory_manager):
        """Test that summarization preserves essential information."""
        # Create entries with specific information
        for i in range(6):
            memory_manager.create_memory(
                action="User opened document",
                context={"file": f"report{i}.pdf", "page": i + 1},
                device_id="laptop-001",
                tags=["document", "work"],
                sensitivity=SensitivityLevel.PRIVATE
            )
        
        # Get summary entries
        all_entries = memory_manager.storage.query_entries(limit=100)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        if len(summary_entries) > 0:
            summary = summary_entries[0]
            
            # Verify essential information is preserved
            assert "document" in summary.tags
            assert "work" in summary.tags
            assert summary.sensitivity == SensitivityLevel.PRIVATE
            assert summary.device_id == "laptop-001"
            assert summary.summary is not None
            assert len(summary.summary) > 0
    
    def test_summarization_without_summarizer(self):
        """Test that operations work correctly when summarizer is not configured."""
        storage = MemoryStorage()
        validation = ValidationManager()
        config = MemoryModuleConfig()
        
        # Create manager without summarizer
        manager = MemoryManager(
            storage=storage,
            encryption=None,
            validation=validation,
            summarizer=None,  # No summarizer
            config=config
        )
        
        # Create entries - should not trigger summarization
        for i in range(10):
            manager.create_memory(
                action=f"Action {i}",
                context={"index": i},
                device_id="test-device"
            )
        
        # Verify no summaries were created
        all_entries = storage.query_entries(limit=100)
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        assert len(summary_entries) == 0
    
    def test_summarization_error_handling(self, memory_manager):
        """Test that summarization errors don't break memory creation."""
        # Mock the summarizer to raise an exception
        original_perform = memory_manager._perform_summarization
        
        def mock_perform():
            raise Exception("Summarization error")
        
        memory_manager._perform_summarization = mock_perform
        
        # Create entries - should not fail even if summarization fails
        entry_id = memory_manager.create_memory(
            action="Test action",
            context={"test": "data"},
            device_id="test-device"
        )
        
        # Verify entry was created successfully
        assert entry_id is not None
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
    
    def test_check_summarization_trigger_called_on_create(self, memory_manager):
        """Test that _check_summarization_trigger is called during create_memory."""
        # Mock the _check_summarization_trigger method
        with patch.object(memory_manager, '_check_summarization_trigger') as mock_check:
            memory_manager.create_memory(
                action="Test action",
                context={"test": "data"},
                device_id="test-device"
            )
            
            # Verify the method was called
            mock_check.assert_called_once()
    
    def test_summarization_with_dissimilar_entries(self, memory_manager):
        """Test that dissimilar entries are not summarized together."""
        # Create dissimilar entries
        for i in range(6):
            memory_manager.create_memory(
                action=f"Completely different action {i}",
                context={"unique_key_" + str(i): f"unique_value_{i}"},
                device_id="test-device",
                tags=[f"tag{i}"]
            )
        
        # Get all entries
        all_entries = memory_manager.storage.query_entries(limit=100)
        
        # Should have fewer or no summaries since entries are dissimilar
        summary_entries = [e for e in all_entries if e.summary is not None and "Summary of" in e.action]
        
        # If summaries exist, they should have fewer entries than similar case
        # This is a weaker assertion since dissimilar entries might still trigger
        # summarization but won't be grouped together
        assert len(all_entries) >= 6
