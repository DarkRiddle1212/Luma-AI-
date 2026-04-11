"""
Unit tests for InjectedMemory dataclass.

**Validates: Requirements 5.1, 5.2, 5.4**
"""

import pytest
from datetime import datetime, timezone
from luma.core.injection_engine import InjectedMemory


def test_injected_memory_creation():
    """Test that InjectedMemory can be created with all required fields."""
    memory = InjectedMemory(
        memory_id="mem_123",
        content="Test memory content",
        metadata={"source": "test", "token_count": 10},
        similarity_score=0.85,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        category="programming"
    )
    
    assert memory.memory_id == "mem_123"
    assert memory.content == "Test memory content"
    assert memory.metadata == {"source": "test", "token_count": 10}
    assert memory.similarity_score == 0.85
    assert memory.timestamp == datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    assert memory.category == "programming"


def test_injected_memory_optional_category():
    """Test that InjectedMemory can be created without category (optional field)."""
    memory = InjectedMemory(
        memory_id="mem_456",
        content="Test content",
        metadata={},
        similarity_score=0.75,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    )
    
    assert memory.category is None


def test_to_dict_serialization():
    """Test that to_dict() produces correct dictionary representation."""
    memory = InjectedMemory(
        memory_id="mem_789",
        content="Serialization test",
        metadata={"key": "value", "count": 42},
        similarity_score=0.92,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        category="education"
    )
    
    result = memory.to_dict()
    
    assert result["memory_id"] == "mem_789"
    assert result["content"] == "Serialization test"
    assert result["metadata"] == {"key": "value", "count": 42}
    assert result["similarity_score"] == 0.92
    assert result["timestamp"] == "2024-01-15T10:30:00+00:00"
    assert result["category"] == "education"


def test_to_dict_with_none_category():
    """Test that to_dict() handles None category correctly."""
    memory = InjectedMemory(
        memory_id="mem_999",
        content="Test",
        metadata={},
        similarity_score=0.5,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        category=None
    )
    
    result = memory.to_dict()
    
    assert result["category"] is None


def test_to_dict_preserves_metadata():
    """Test that to_dict() preserves metadata exactly (immutability requirement)."""
    original_metadata = {
        "source": "user_input",
        "token_count": 45,
        "embedding": [0.1, 0.2, 0.3],
        "custom_field": "value"
    }
    
    memory = InjectedMemory(
        memory_id="mem_meta",
        content="Metadata test",
        metadata=original_metadata,
        similarity_score=0.88,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        category="test"
    )
    
    result = memory.to_dict()
    
    # Metadata should be preserved exactly
    assert result["metadata"] == original_metadata
    assert result["metadata"]["embedding"] == [0.1, 0.2, 0.3]
    assert result["metadata"]["custom_field"] == "value"


def test_to_dict_datetime_iso_format():
    """Test that to_dict() converts datetime to ISO format string."""
    memory = InjectedMemory(
        memory_id="mem_time",
        content="Time test",
        metadata={},
        similarity_score=0.7,
        timestamp=datetime(2024, 3, 20, 15, 45, 30, tzinfo=timezone.utc),
        category="test"
    )
    
    result = memory.to_dict()
    
    # Should be ISO format string
    assert isinstance(result["timestamp"], str)
    assert result["timestamp"] == "2024-03-20T15:45:30+00:00"


def test_to_dict_all_fields_present():
    """Test that to_dict() includes all required fields (Requirement 5.2)."""
    memory = InjectedMemory(
        memory_id="mem_complete",
        content="Complete test",
        metadata={"test": "data"},
        similarity_score=0.95,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        category="complete"
    )
    
    result = memory.to_dict()
    
    # All required fields must be present
    required_fields = ["memory_id", "content", "metadata", "similarity_score", "timestamp", "category"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"


def test_empty_metadata():
    """Test that InjectedMemory handles empty metadata correctly."""
    memory = InjectedMemory(
        memory_id="mem_empty",
        content="Empty metadata test",
        metadata={},
        similarity_score=0.6,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    )
    
    result = memory.to_dict()
    assert result["metadata"] == {}


def test_complex_metadata():
    """Test that InjectedMemory handles complex nested metadata."""
    complex_metadata = {
        "nested": {
            "level1": {
                "level2": "value"
            }
        },
        "list": [1, 2, 3],
        "mixed": {"a": [1, 2], "b": {"c": 3}}
    }
    
    memory = InjectedMemory(
        memory_id="mem_complex",
        content="Complex metadata test",
        metadata=complex_metadata,
        similarity_score=0.8,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    )
    
    result = memory.to_dict()
    assert result["metadata"] == complex_metadata
