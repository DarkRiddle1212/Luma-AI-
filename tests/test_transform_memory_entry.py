"""
Unit tests for transform_memory_entry() function.

Tests the transformation of MemoryEntry objects to pure dictionaries,
verifying that all required fields are extracted and metadata is preserved.

Requirements tested:
- 1.2: Extract all required fields (id, content, category, timestamp, metadata, tags)
- 1.3: Preserve metadata exactly as-is (no modifications)
- 1.4: Return pure dictionary with only primitive types
"""

import pytest
from luma.core.context_injection import transform_memory_entry
from luma.core.memory_interface import MemoryEntry


class TestTransformMemoryEntry:
    """Test suite for transform_memory_entry() function."""
    
    def test_extracts_all_required_fields(self):
        """Test that all required fields are extracted from MemoryEntry."""
        # Requirement 1.2: Extract id, content, category, timestamp, metadata, tags
        entry: MemoryEntry = {
            "id": "mem_123",
            "content": "Python is a programming language",
            "category": "education",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {"source": "user_input", "confidence": 0.95},
            "tags": ["programming", "python"]
        }
        
        result = transform_memory_entry(entry)
        
        # Verify all required fields are present
        assert "id" in result
        assert "content" in result
        assert "category" in result
        assert "timestamp" in result
        assert "metadata" in result
        assert "tags" in result
        
        # Verify field values match
        assert result["id"] == "mem_123"
        assert result["content"] == "Python is a programming language"
        assert result["category"] == "education"
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"source": "user_input", "confidence": 0.95}
        assert result["tags"] == ["programming", "python"]
    
    def test_preserves_metadata_exactly(self):
        """Test that metadata is preserved exactly as-is without modifications."""
        # Requirement 1.3: Preserve metadata exactly as-is
        complex_metadata = {
            "source": "user_input",
            "confidence": 0.95,
            "nested": {
                "level1": {
                    "level2": "deep_value"
                }
            },
            "list_data": [1, 2, 3],
            "mixed": ["string", 42, True, None]
        }
        
        entry: MemoryEntry = {
            "id": "mem_456",
            "content": "Test content",
            "category": "test",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": complex_metadata,
            "tags": ["test"]
        }
        
        result = transform_memory_entry(entry)
        
        # Verify metadata is preserved exactly (round-trip property)
        assert result["metadata"] == complex_metadata
        # Note: The implementation preserves the reference, which is acceptable
        # as long as the metadata is not modified
    
    def test_returns_pure_dictionary(self):
        """Test that result contains only primitive types."""
        # Requirement 1.4: Return pure dictionary with only primitive types
        entry: MemoryEntry = {
            "id": "mem_789",
            "content": "Test content",
            "category": "test",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {"key": "value"},
            "tags": ["tag1", "tag2"]
        }
        
        result = transform_memory_entry(entry)
        
        # Verify result is a dictionary
        assert isinstance(result, dict)
        
        # Verify all values are primitive types
        assert isinstance(result["id"], str)
        assert isinstance(result["content"], str)
        assert isinstance(result["category"], str)
        assert isinstance(result["timestamp"], str)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["tags"], list)
        
        # Verify no custom objects
        for value in result.values():
            assert not hasattr(value, '__dict__') or isinstance(value, (dict, list))
    
    def test_handles_empty_metadata(self):
        """Test transformation with empty metadata."""
        entry: MemoryEntry = {
            "id": "mem_empty",
            "content": "Content with empty metadata",
            "category": "test",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {},
            "tags": []
        }
        
        result = transform_memory_entry(entry)
        
        assert result["metadata"] == {}
        assert result["tags"] == []
    
    def test_handles_various_primitive_types_in_metadata(self):
        """Test that various primitive types in metadata are preserved."""
        entry: MemoryEntry = {
            "id": "mem_types",
            "content": "Test content",
            "category": "test",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "none": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"}
            },
            "tags": ["tag1"]
        }
        
        result = transform_memory_entry(entry)
        
        # Verify all primitive types are preserved
        assert result["metadata"]["string"] == "text"
        assert result["metadata"]["integer"] == 42
        assert result["metadata"]["float"] == 3.14
        assert result["metadata"]["boolean"] is True
        assert result["metadata"]["none"] is None
        assert result["metadata"]["list"] == [1, 2, 3]
        assert result["metadata"]["dict"] == {"nested": "value"}
