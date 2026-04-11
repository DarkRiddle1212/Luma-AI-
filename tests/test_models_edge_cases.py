"""Edge case tests for data models."""

import pytest
from datetime import datetime, UTC
import sys
sys.path.insert(0, 'luma_memory')

from models import (
    MemoryEntry,
    SensitivityLevel,
    SyncStatus,
    create_memory_entry
)


class TestMemoryEntryEdgeCases:
    """Comprehensive tests for edge cases and invalid data."""
    
    def test_validation_with_whitespace_only_id(self):
        """Test validation fails for whitespace-only ID."""
        entry = MemoryEntry(
            id="   ",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "ID cannot be empty" in error
    
    def test_validation_with_whitespace_only_action(self):
        """Test validation fails for whitespace-only action."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="   ",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Action cannot be empty" in error
    
    def test_validation_with_whitespace_only_device_id(self):
        """Test validation fails for whitespace-only device ID."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="   ",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Device ID cannot be empty" in error
    
    def test_validation_with_whitespace_only_parent_id(self):
        """Test validation fails for whitespace-only parent ID."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            parent_id="   "
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Parent ID must be a non-empty string if provided" in error
    
    def test_validation_with_none_timestamp(self):
        """Test validation fails for None timestamp."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=None,
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Timestamp must be a datetime object" in error
    
    def test_validation_with_none_context(self):
        """Test validation fails for None context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context=None,
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Context must be a dictionary" in error
    
    def test_validation_with_none_tags(self):
        """Test validation fails for None tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=None
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Tags must be a list" in error
    
    def test_validation_with_empty_string_summary(self):
        """Test validation allows empty string summary."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=""
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_very_long_strings(self):
        """Test validation with extremely long string values."""
        long_string = "x" * 100000
        entry = MemoryEntry(
            id=long_string,
            timestamp=datetime.now(UTC),
            action=long_string,
            context={"key": long_string},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id=long_string,
            sync_status=SyncStatus.PENDING,
            tags=[long_string]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_special_characters_in_strings(self):
        """Test validation with special characters."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~\n\t\r"
        entry = MemoryEntry(
            id=special_chars,
            timestamp=datetime.now(UTC),
            action=special_chars,
            context={"key": special_chars},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id=special_chars,
            sync_status=SyncStatus.PENDING,
            tags=[special_chars]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_unicode_characters(self):
        """Test validation with unicode characters."""
        unicode_string = "Hello 世界 🌍 مرحبا Привет"
        entry = MemoryEntry(
            id=unicode_string,
            timestamp=datetime.now(UTC),
            action=unicode_string,
            context={"key": unicode_string},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id=unicode_string,
            sync_status=SyncStatus.PENDING,
            tags=[unicode_string]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_mixed_type_tags(self):
        """Test validation fails with mixed type tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["string", 123, None, True, {"dict": "value"}]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "All tags must be strings" in error
    
    def test_validation_with_empty_tags_list(self):
        """Test validation allows empty tags list."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_large_tags_list(self):
        """Test validation with very large tags list."""
        large_tags = [f"tag-{i}" for i in range(10000)]
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=large_tags
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_deeply_nested_context(self):
        """Test validation with deeply nested context dictionary."""
        nested_context = {"level1": {"level2": {"level3": {"level4": {"level5": "value"}}}}}
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context=nested_context,
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_context_containing_none_values(self):
        """Test validation allows None values in context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={"key1": None, "key2": "value", "key3": None},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_context_containing_mixed_types(self):
        """Test validation allows mixed types in context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={
                "string": "value",
                "int": 123,
                "float": 45.67,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "none": None
            },
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_serialization_with_empty_context(self):
        """Test serialization with empty context."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1"
        )
        
        result = entry.to_dict()
        assert result["context"] == {}
    
    def test_serialization_with_context_containing_special_types(self):
        """Test serialization preserves special types in context."""
        entry = create_memory_entry(
            action="test",
            context={
                "boolean": True,
                "none": None,
                "integer": 42,
                "float": 3.14,
                "negative": -100,
                "zero": 0
            },
            device_id="device-1"
        )
        
        result = entry.to_dict()
        assert result["context"]["boolean"] is True
        assert result["context"]["none"] is None
        assert result["context"]["integer"] == 42
        assert result["context"]["float"] == 3.14
        assert result["context"]["negative"] == -100
        assert result["context"]["zero"] == 0
    
    def test_deserialization_with_missing_optional_fields(self):
        """Test deserialization when optional fields are missing from dict."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": []
        }
        
        entry = MemoryEntry.from_dict(data)
        assert entry.summary is None
        assert entry.parent_id is None
    
    def test_deserialization_with_invalid_datetime_format(self):
        """Test deserialization fails with invalid datetime format."""
        data = {
            "id": "test-id",
            "timestamp": "not-a-valid-datetime",
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": []
        }
        
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)
    
    def test_create_memory_entry_with_none_tags(self):
        """Test factory function converts None tags to empty list."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=None
        )
        
        assert entry.tags == []
    
    def test_create_memory_entry_generates_different_ids(self):
        """Test factory function generates unique IDs for each call."""
        entries = [
            create_memory_entry(action="test", context={}, device_id="device-1")
            for _ in range(100)
        ]
        
        ids = [entry.id for entry in entries]
        assert len(ids) == len(set(ids))  # All IDs should be unique
    
    def test_create_memory_entry_with_empty_context(self):
        """Test factory function accepts empty context."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1"
        )
        
        assert entry.context == {}
        assert entry.is_valid()
    
    def test_timestamp_precision(self):
        """Test that timestamps preserve microsecond precision."""
        timestamp = datetime(2024, 1, 1, 12, 30, 45, 123456)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            timestamp=timestamp
        )
        
        assert entry.timestamp.microsecond == 123456
        
        # Test round-trip
        serialized = entry.to_dict()
        restored = MemoryEntry.from_dict(serialized)
        assert restored.timestamp.microsecond == 123456
    
    def test_validation_with_integer_summary(self):
        """Test validation fails for integer summary."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=12345
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Summary must be a string" in error
    
    def test_validation_with_list_summary(self):
        """Test validation fails for list summary."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=["not", "a", "string"]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Summary must be a string" in error
    
    def test_validation_with_dict_summary(self):
        """Test validation fails for dict summary."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary={"not": "a string"}
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Summary must be a string" in error
    
    def test_validation_with_integer_parent_id(self):
        """Test validation fails for integer parent ID."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            parent_id=12345
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Parent ID must be a non-empty string if provided" in error
    
    def test_context_with_empty_string_keys(self):
        """Test validation allows empty string keys in context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={"": "value", "key": ""},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_tags_with_empty_strings(self):
        """Test validation allows empty strings in tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["", "valid-tag", ""]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_tags_with_duplicate_values(self):
        """Test validation allows duplicate tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["tag1", "tag1", "tag2", "tag2"]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_serialization_preserves_context_order(self):
        """Test that serialization preserves context dictionary order."""
        context = {"z": 1, "a": 2, "m": 3}
        entry = create_memory_entry(
            action="test",
            context=context,
            device_id="device-1"
        )
        
        result = entry.to_dict()
        # In Python 3.7+, dict order is preserved
        assert list(result["context"].keys()) == ["z", "a", "m"]
    
    def test_validation_with_future_timestamp(self):
        """Test validation allows future timestamps."""
        future_time = datetime(2099, 12, 31, 23, 59, 59)
        entry = MemoryEntry(
            id="test-id",
            timestamp=future_time,
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_very_old_timestamp(self):
        """Test validation allows very old timestamps."""
        old_time = datetime(1970, 1, 1, 0, 0, 0)
        entry = MemoryEntry(
            id="test-id",
            timestamp=old_time,
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
