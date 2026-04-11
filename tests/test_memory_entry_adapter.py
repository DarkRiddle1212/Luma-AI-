"""
Unit tests for MemoryEntry to RankedMemory adapter function.

Tests the conversion logic that bridges the memory storage layer
and the ranking engine.
"""

import pytest
from datetime import datetime, timezone, UTC
from luma.core.ranking_engine import memory_entry_to_ranked_memory, RankedMemory
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


def test_basic_conversion():
    """Test basic conversion from MemoryEntry to RankedMemory."""
    entry = MemoryEntry(
        id="mem_123",
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
        action="User searched for Python tutorials",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=["search", "python"]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.85)
    
    assert ranked.memory_id == "mem_123"
    assert ranked.timestamp == datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    assert ranked.content == "User searched for Python tutorials"
    assert ranked.namespace is None
    assert ranked.similarity_score == 0.85
    assert ranked.importance_score == 0.0  # Default when not in context
    assert ranked.recency_score == 0.0  # Not yet computed
    assert ranked.final_score == 0.0  # Not yet computed
    assert ranked.memory_entry is entry


def test_conversion_with_namespace():
    """Test conversion with explicit namespace."""
    entry = MemoryEntry(
        id="mem_456",
        timestamp=datetime.now(UTC),
        action="User action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(
        entry,
        similarity_score=0.75,
        namespace="conversation"
    )
    
    assert ranked.namespace == "conversation"
    assert ranked.similarity_score == 0.75


def test_importance_extraction_from_context():
    """Test that importance score is extracted from context metadata."""
    entry = MemoryEntry(
        id="mem_789",
        timestamp=datetime.now(UTC),
        action="Important user action",
        context={"importance": 0.8},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.9)
    
    assert ranked.importance_score == 0.8


def test_importance_extraction_with_integer():
    """Test that integer importance values are converted to float."""
    entry = MemoryEntry(
        id="mem_int",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"importance": 1},  # Integer
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.5)
    
    assert ranked.importance_score == 1.0
    assert isinstance(ranked.importance_score, float)


def test_importance_clamped_to_valid_range():
    """Test that importance scores outside [0, 1] are clamped."""
    # Test upper bound
    entry_high = MemoryEntry(
        id="mem_high",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"importance": 1.5},  # Above 1.0
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked_high = memory_entry_to_ranked_memory(entry_high, similarity_score=0.5)
    assert ranked_high.importance_score == 1.0
    
    # Test lower bound
    entry_low = MemoryEntry(
        id="mem_low",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"importance": -0.5},  # Below 0.0
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked_low = memory_entry_to_ranked_memory(entry_low, similarity_score=0.5)
    assert ranked_low.importance_score == 0.0


def test_missing_importance_defaults_to_zero():
    """Test that missing importance in context defaults to 0.0."""
    entry = MemoryEntry(
        id="mem_no_importance",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"other_field": "value"},  # No importance field
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.5)
    
    assert ranked.importance_score == 0.0


def test_invalid_importance_type_defaults_to_zero():
    """Test that non-numeric importance values default to 0.0."""
    entry = MemoryEntry(
        id="mem_invalid",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"importance": "high"},  # String instead of number
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.5)
    
    assert ranked.importance_score == 0.0


def test_empty_context():
    """Test conversion with empty context dictionary."""
    entry = MemoryEntry(
        id="mem_empty",
        timestamp=datetime.now(UTC),
        action="Action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.5)
    
    assert ranked.importance_score == 0.0


def test_preserves_original_memory_entry():
    """Test that the original MemoryEntry is preserved in memory_entry field."""
    entry = MemoryEntry(
        id="mem_preserve",
        timestamp=datetime.now(UTC),
        action="Action",
        context={"importance": 0.7},
        sensitivity=SensitivityLevel.PRIVATE,
        device_id="device_1",
        sync_status=SyncStatus.PENDING,
        tags=["tag1", "tag2"]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.8)
    
    # Verify the original entry is preserved
    assert ranked.memory_entry is entry
    assert ranked.memory_entry.sensitivity == SensitivityLevel.PRIVATE
    assert ranked.memory_entry.sync_status == SyncStatus.PENDING
    assert ranked.memory_entry.tags == ["tag1", "tag2"]


def test_similarity_score_range():
    """Test that various similarity scores are handled correctly."""
    entry = MemoryEntry(
        id="mem_sim",
        timestamp=datetime.now(UTC),
        action="Action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    # Test minimum
    ranked_min = memory_entry_to_ranked_memory(entry, similarity_score=0.0)
    assert ranked_min.similarity_score == 0.0
    
    # Test maximum
    ranked_max = memory_entry_to_ranked_memory(entry, similarity_score=1.0)
    assert ranked_max.similarity_score == 1.0
    
    # Test middle
    ranked_mid = memory_entry_to_ranked_memory(entry, similarity_score=0.5)
    assert ranked_mid.similarity_score == 0.5


def test_content_from_action_field():
    """Test that content is extracted from the action field."""
    action_text = "User performed a complex action with details"
    entry = MemoryEntry(
        id="mem_action",
        timestamp=datetime.now(UTC),
        action=action_text,
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device_1",
        sync_status=SyncStatus.SYNCED,
        tags=[]
    )
    
    ranked = memory_entry_to_ranked_memory(entry, similarity_score=0.6)
    
    assert ranked.content == action_text
