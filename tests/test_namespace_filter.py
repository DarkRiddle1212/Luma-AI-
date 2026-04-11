"""Unit tests for NamespaceFilter component."""

import pytest
from datetime import datetime, timezone
from luma.core.ranking_engine import NamespaceFilter, RankedMemory


def create_test_memory(memory_id: str, namespace: str = None) -> RankedMemory:
    """Helper to create test memory."""
    return RankedMemory(
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc),
        content="test content",
        namespace=namespace,
        similarity_score=0.8,
        importance_score=0.5,
        recency_score=0.0,
        final_score=0.0,
        memory_entry=None
    )


def test_filter_with_specific_namespace():
    """Test filtering with specific namespace."""
    memories = [
        create_test_memory("1", "conversation"),
        create_test_memory("2", "system"),
        create_test_memory("3", "conversation"),
        create_test_memory("4", "user"),
    ]
    
    result = NamespaceFilter.filter(memories, "conversation")
    
    assert len(result) == 2
    assert all(m.namespace == "conversation" for m in result)
    assert result[0].memory_id == "1"
    assert result[1].memory_id == "3"


def test_filter_with_none_namespace():
    """Test no filtering when namespace is None."""
    memories = [
        create_test_memory("1", "conversation"),
        create_test_memory("2", "system"),
        create_test_memory("3", "user"),
    ]
    
    result = NamespaceFilter.filter(memories, None)
    
    assert len(result) == 3
    assert result == memories


def test_filter_empty_result():
    """Test empty result when no memories match namespace."""
    memories = [
        create_test_memory("1", "conversation"),
        create_test_memory("2", "system"),
    ]
    
    result = NamespaceFilter.filter(memories, "nonexistent")
    
    assert len(result) == 0


def test_filter_empty_input():
    """Test filtering with empty input list."""
    result = NamespaceFilter.filter([], "conversation")
    
    assert len(result) == 0


def test_filter_preserves_order():
    """Test that filtering preserves input order."""
    memories = [
        create_test_memory("1", "conversation"),
        create_test_memory("2", "conversation"),
        create_test_memory("3", "conversation"),
    ]
    
    result = NamespaceFilter.filter(memories, "conversation")
    
    assert len(result) == 3
    assert [m.memory_id for m in result] == ["1", "2", "3"]
