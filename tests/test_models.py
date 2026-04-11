"""Unit tests for data models."""

import pytest
from datetime import datetime, UTC
import sys
sys.path.insert(0, 'luma_memory')

from models import (
    MemoryEntry,
    SensitivityLevel,
    SyncStatus,
    MemoryType,
    create_memory_entry
)


class TestMemoryType:
    """Tests for MemoryType enum."""
    
    def test_memory_type_values(self):
        """Test that MemoryType has all expected values."""
        assert MemoryType.ACTION.value == "action"
        assert MemoryType.CONTEXT.value == "context"
        assert MemoryType.CONVERSATION.value == "conversation"
        assert MemoryType.TASK.value == "task"
        assert MemoryType.SYSTEM.value == "system"


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""
    
    def test_sensitivity_level_values(self):
        """Test that SensitivityLevel has all expected values."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.PRIVATE.value == "private"
        assert SensitivityLevel.SENSITIVE.value == "sensitive"


class TestSyncStatus:
    """Tests for SyncStatus enum."""
    
    def test_sync_status_values(self):
        """Test that SyncStatus has all expected values."""
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.SYNCED.value == "synced"
        assert SyncStatus.CONFLICT.value == "conflict"


class TestMemoryEntry:
    """Tests for MemoryEntry model."""
    
    def test_create_memory_entry(self):
        """Test creating a memory entry with factory function."""
        entry = create_memory_entry(
            action="user_search",
            context={"query": "test"},
            device_id="device-1",
            tags=["test"]
        )
        
        assert entry.action == "user_search"
        assert entry.device_id == "device-1"
        assert entry.tags == ["test"]
        assert entry.sync_status == SyncStatus.PENDING
        assert entry.sensitivity == SensitivityLevel.PUBLIC
    
    def test_memory_entry_validation_success(self):
        """Test validation of a valid memory entry."""
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1"
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_memory_entry_validation_empty_id(self):
        """Test validation fails for empty ID."""
        entry = MemoryEntry(
            id="",
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
    
    def test_memory_entry_validation_empty_action(self):
        """Test validation fails for empty action."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Action cannot be empty" in error
    
    def test_memory_entry_serialization(self):
        """Test serialization to dictionary."""
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1",
            sensitivity=SensitivityLevel.PRIVATE,
            tags=["tag1", "tag2"]
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["action"] == "test_action"
        assert entry_dict["device_id"] == "device-1"
        assert entry_dict["sensitivity"] == "private"
        assert entry_dict["sync_status"] == "pending"
        assert entry_dict["tags"] == ["tag1", "tag2"]
        assert isinstance(entry_dict["timestamp"], str)
    
    def test_memory_entry_deserialization(self):
        """Test deserialization from dictionary."""
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1"
        )
        
        entry_dict = entry.to_dict()
        restored = MemoryEntry.from_dict(entry_dict)
        
        assert restored.id == entry.id
        assert restored.action == entry.action
        assert restored.device_id == entry.device_id
        assert isinstance(restored.sensitivity, SensitivityLevel)
        assert isinstance(restored.sync_status, SyncStatus)
        assert isinstance(restored.timestamp, datetime)
    
    def test_memory_entry_timestamps(self):
        """Test that timestamps are automatically set."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1"
        )
        
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)
    
    def test_memory_entry_creation_with_all_fields(self):
        """Test creating a memory entry with all fields specified."""
        timestamp = datetime.now(UTC)
        created_at = datetime.now(UTC)
        updated_at = datetime.now(UTC)
        
        entry = MemoryEntry(
            id="test-id-123",
            timestamp=timestamp,
            action="user_action",
            context={"key": "value", "nested": {"data": 123}},
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="device-abc",
            sync_status=SyncStatus.SYNCED,
            tags=["tag1", "tag2", "tag3"],
            summary="Test summary",
            parent_id="parent-id-456",
            created_at=created_at,
            updated_at=updated_at
        )
        
        assert entry.id == "test-id-123"
        assert entry.timestamp == timestamp
        assert entry.action == "user_action"
        assert entry.context == {"key": "value", "nested": {"data": 123}}
        assert entry.sensitivity == SensitivityLevel.SENSITIVE
        assert entry.device_id == "device-abc"
        assert entry.sync_status == SyncStatus.SYNCED
        assert entry.tags == ["tag1", "tag2", "tag3"]
        assert entry.summary == "Test summary"
        assert entry.parent_id == "parent-id-456"
        assert entry.created_at == created_at
        assert entry.updated_at == updated_at
    
    def test_memory_entry_creation_with_minimal_fields(self):
        """Test creating a memory entry with only required fields."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        assert entry.id == "test-id"
        assert entry.action == "test_action"
        assert entry.context == {}
        assert entry.sensitivity == SensitivityLevel.PUBLIC
        assert entry.device_id == "device-1"
        assert entry.sync_status == SyncStatus.PENDING
        assert entry.tags == []
        assert entry.summary is None
        assert entry.parent_id is None
        assert entry.created_at is not None
        assert entry.updated_at is not None
    
    def test_memory_entry_validation_empty_device_id(self):
        """Test validation fails for empty device ID."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Device ID cannot be empty" in error
    
    def test_memory_entry_validation_invalid_timestamp(self):
        """Test validation fails for invalid timestamp."""
        entry = MemoryEntry(
            id="test-id",
            timestamp="not-a-datetime",
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
    
    def test_memory_entry_validation_invalid_context(self):
        """Test validation fails for non-dict context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context="not-a-dict",
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Context must be a dictionary" in error
    
    def test_memory_entry_validation_invalid_sensitivity(self):
        """Test validation fails for invalid sensitivity."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity="invalid",
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Sensitivity must be a SensitivityLevel enum" in error
    
    def test_memory_entry_validation_invalid_sync_status(self):
        """Test validation fails for invalid sync status."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status="invalid",
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Sync status must be a SyncStatus enum" in error
    
    def test_memory_entry_validation_invalid_tags_type(self):
        """Test validation fails for non-list tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags="not-a-list"
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Tags must be a list" in error
    
    def test_memory_entry_validation_invalid_tags_content(self):
        """Test validation fails for non-string tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["valid", 123, "another"]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "All tags must be strings" in error
    
    def test_memory_entry_validation_invalid_summary(self):
        """Test validation fails for non-string summary."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=123
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Summary must be a string" in error
    
    def test_memory_entry_validation_empty_parent_id(self):
        """Test validation fails for empty parent ID."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            parent_id=""
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Parent ID must be a non-empty string if provided" in error
    
    def test_memory_entry_is_valid_method(self):
        """Test is_valid convenience method."""
        valid_entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1"
        )
        assert valid_entry.is_valid() is True
        
        invalid_entry = MemoryEntry(
            id="",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        assert invalid_entry.is_valid() is False
    
    def test_create_memory_entry_with_custom_id(self):
        """Test factory function with custom ID."""
        custom_id = "custom-id-123"
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            entry_id=custom_id
        )
        
        assert entry.id == custom_id
    
    def test_create_memory_entry_with_custom_timestamp(self):
        """Test factory function with custom timestamp."""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            timestamp=custom_timestamp
        )
        
        assert entry.timestamp == custom_timestamp
    
    def test_create_memory_entry_with_sensitivity(self):
        """Test factory function with different sensitivity levels."""
        for sensitivity in [SensitivityLevel.PUBLIC, SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
            entry = create_memory_entry(
                action="test",
                context={},
                device_id="device-1",
                sensitivity=sensitivity
            )
            assert entry.sensitivity == sensitivity
    
    def test_create_memory_entry_generates_uuid(self):
        """Test that factory function generates unique IDs."""
        entry1 = create_memory_entry(action="test", context={}, device_id="device-1")
        entry2 = create_memory_entry(action="test", context={}, device_id="device-1")
        
        assert entry1.id != entry2.id
        assert len(entry1.id) > 0
        assert len(entry2.id) > 0
    
    def test_memory_entry_complex_context(self):
        """Test memory entry with complex nested context."""
        complex_context = {
            "user": {"name": "test", "id": 123},
            "action_details": {
                "type": "search",
                "query": "test query",
                "filters": ["filter1", "filter2"]
            },
            "metadata": {
                "source": "web",
                "timestamp": "2024-01-01T00:00:00"
            }
        }
        
        entry = create_memory_entry(
            action="complex_action",
            context=complex_context,
            device_id="device-1"
        )
        
        assert entry.context == complex_context
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None


class TestMemoryEntrySerialization:
    """Comprehensive tests for MemoryEntry serialization/deserialization."""
    
    def test_serialization_with_all_fields(self):
        """Test serialization of entry with all fields populated."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45)
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        updated_at = datetime(2024, 1, 15, 10, 31, 0)
        
        entry = MemoryEntry(
            id="test-id-123",
            timestamp=timestamp,
            action="user_search",
            context={"query": "test", "results": 5},
            sensitivity=SensitivityLevel.PRIVATE,
            device_id="device-abc",
            sync_status=SyncStatus.SYNCED,
            tags=["search", "user"],
            summary="User searched for test",
            parent_id="parent-123",
            created_at=created_at,
            updated_at=updated_at
        )
        
        result = entry.to_dict()
        
        assert result["id"] == "test-id-123"
        assert result["timestamp"] == "2024-01-15T10:30:45"
        assert result["action"] == "user_search"
        assert result["context"] == {"query": "test", "results": 5}
        assert result["sensitivity"] == "private"
        assert result["device_id"] == "device-abc"
        assert result["sync_status"] == "synced"
        assert result["tags"] == ["search", "user"]
        assert result["summary"] == "User searched for test"
        assert result["parent_id"] == "parent-123"
        assert result["created_at"] == "2024-01-15T10:30:00"
        assert result["updated_at"] == "2024-01-15T10:31:00"
    
    def test_serialization_with_minimal_fields(self):
        """Test serialization of entry with only required fields."""
        entry = create_memory_entry(
            action="test_action",
            context={},
            device_id="device-1"
        )
        
        result = entry.to_dict()
        
        assert result["id"] is not None
        assert result["timestamp"] is not None
        assert result["action"] == "test_action"
        assert result["context"] == {}
        assert result["sensitivity"] == "public"
        assert result["device_id"] == "device-1"
        assert result["sync_status"] == "pending"
        assert result["tags"] == []
        assert result["summary"] is None
        assert result["parent_id"] is None
        assert result["created_at"] is not None
        assert result["updated_at"] is not None
    
    def test_serialization_with_complex_context(self):
        """Test serialization preserves complex nested context."""
        complex_context = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                    "number": 42,
                    "boolean": True
                },
                "list": [1, 2, 3]
            },
            "string": "test",
            "null_value": None
        }
        
        entry = create_memory_entry(
            action="complex",
            context=complex_context,
            device_id="device-1"
        )
        
        result = entry.to_dict()
        assert result["context"] == complex_context
    
    def test_serialization_with_empty_tags(self):
        """Test serialization with empty tags list."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        
        result = entry.to_dict()
        assert result["tags"] == []
    
    def test_serialization_with_multiple_tags(self):
        """Test serialization with multiple tags."""
        tags = ["tag1", "tag2", "tag3", "tag4"]
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=tags
        )
        
        result = entry.to_dict()
        assert result["tags"] == tags
    
    def test_deserialization_with_all_fields(self):
        """Test deserialization of dictionary with all fields."""
        data = {
            "id": "test-id-456",
            "timestamp": "2024-02-20T15:45:30",
            "action": "user_action",
            "context": {"key": "value"},
            "sensitivity": "sensitive",
            "device_id": "device-xyz",
            "sync_status": "conflict",
            "tags": ["important", "urgent"],
            "summary": "Test summary",
            "parent_id": "parent-456",
            "created_at": "2024-02-20T15:45:00",
            "updated_at": "2024-02-20T15:46:00"
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert entry.id == "test-id-456"
        assert entry.timestamp == datetime(2024, 2, 20, 15, 45, 30)
        assert entry.action == "user_action"
        assert entry.context == {"key": "value"}
        assert entry.sensitivity == SensitivityLevel.SENSITIVE
        assert entry.device_id == "device-xyz"
        assert entry.sync_status == SyncStatus.CONFLICT
        assert entry.tags == ["important", "urgent"]
        assert entry.summary == "Test summary"
        assert entry.parent_id == "parent-456"
        assert entry.created_at == datetime(2024, 2, 20, 15, 45, 0)
        assert entry.updated_at == datetime(2024, 2, 20, 15, 46, 0)
    
    def test_deserialization_with_minimal_fields(self):
        """Test deserialization with only required fields."""
        data = {
            "id": "min-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "minimal",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert entry.id == "min-id"
        assert entry.action == "minimal"
        assert entry.summary is None
        assert entry.parent_id is None
    
    def test_deserialization_preserves_enum_types(self):
        """Test that deserialization converts strings to proper enum types."""
        data = {
            "id": "enum-test",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {},
            "sensitivity": "private",
            "device_id": "device-1",
            "sync_status": "synced",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert isinstance(entry.sensitivity, SensitivityLevel)
        assert entry.sensitivity == SensitivityLevel.PRIVATE
        assert isinstance(entry.sync_status, SyncStatus)
        assert entry.sync_status == SyncStatus.SYNCED
    
    def test_deserialization_preserves_datetime_types(self):
        """Test that deserialization converts ISO strings to datetime objects."""
        data = {
            "id": "datetime-test",
            "timestamp": "2024-03-15T12:30:45",
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": "2024-03-15T12:30:00",
            "updated_at": "2024-03-15T12:31:00"
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp == datetime(2024, 3, 15, 12, 30, 45)
        assert isinstance(entry.created_at, datetime)
        assert entry.created_at == datetime(2024, 3, 15, 12, 30, 0)
        assert isinstance(entry.updated_at, datetime)
        assert entry.updated_at == datetime(2024, 3, 15, 12, 31, 0)
    
    def test_round_trip_serialization(self):
        """Test that serialization and deserialization are inverse operations."""
        original = create_memory_entry(
            action="round_trip_test",
            context={"data": "test", "nested": {"value": 123}},
            device_id="device-round-trip",
            sensitivity=SensitivityLevel.SENSITIVE,
            tags=["test", "round-trip"]
        )
        
        # Serialize to dict
        serialized = original.to_dict()
        
        # Deserialize back to object
        restored = MemoryEntry.from_dict(serialized)
        
        # Verify all fields match
        assert restored.id == original.id
        assert restored.timestamp == original.timestamp
        assert restored.action == original.action
        assert restored.context == original.context
        assert restored.sensitivity == original.sensitivity
        assert restored.device_id == original.device_id
        assert restored.sync_status == original.sync_status
        assert restored.tags == original.tags
        assert restored.summary == original.summary
        assert restored.parent_id == original.parent_id
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at
    
    def test_round_trip_with_all_sensitivity_levels(self):
        """Test round-trip serialization for all sensitivity levels."""
        for sensitivity in [SensitivityLevel.PUBLIC, SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
            entry = create_memory_entry(
                action="test",
                context={},
                device_id="device-1",
                sensitivity=sensitivity
            )
            
            serialized = entry.to_dict()
            restored = MemoryEntry.from_dict(serialized)
            
            assert restored.sensitivity == sensitivity
    
    def test_round_trip_with_all_sync_statuses(self):
        """Test round-trip serialization for all sync statuses."""
        for sync_status in [SyncStatus.PENDING, SyncStatus.SYNCED, SyncStatus.CONFLICT]:
            entry = create_memory_entry(
                action="test",
                context={},
                device_id="device-1"
            )
            entry.sync_status = sync_status
            
            serialized = entry.to_dict()
            restored = MemoryEntry.from_dict(serialized)
            
            assert restored.sync_status == sync_status
    
    def test_serialization_does_not_modify_original(self):
        """Test that serialization doesn't modify the original entry."""
        entry = create_memory_entry(
            action="test",
            context={"key": "value"},
            device_id="device-1",
            tags=["tag1"]
        )
        
        original_id = entry.id
        original_action = entry.action
        original_context = entry.context.copy()
        original_tags = entry.tags.copy()
        
        # Serialize
        entry.to_dict()
        
        # Verify original is unchanged
        assert entry.id == original_id
        assert entry.action == original_action
        assert entry.context == original_context
        assert entry.tags == original_tags
    
    def test_deserialization_does_not_modify_input(self):
        """Test that deserialization doesn't modify the input dictionary."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {"key": "value"},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": ["tag1"],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        original_data = data.copy()
        
        # Deserialize
        MemoryEntry.from_dict(data)
        
        # Verify input is unchanged
        assert data == original_data
    
    def test_serialization_with_none_timestamps(self):
        """Test serialization handles None timestamps correctly."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            created_at=None,
            updated_at=None
        )
        
        # Override the __post_init__ values
        entry.created_at = None
        entry.updated_at = None
        
        result = entry.to_dict()
        
        assert result["created_at"] is None
        assert result["updated_at"] is None
    
    def test_deserialization_with_datetime_objects(self):
        """Test that deserialization handles datetime objects (not just strings)."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        created = datetime(2024, 1, 1, 11, 59, 0)
        updated = datetime(2024, 1, 1, 12, 1, 0)
        
        data = {
            "id": "test-id",
            "timestamp": timestamp,
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": created,
            "updated_at": updated
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert entry.timestamp == timestamp
        assert entry.created_at == created
        assert entry.updated_at == updated
    
    def test_deserialization_with_enum_objects(self):
        """Test that deserialization handles enum objects (not just strings)."""
        data = {
            "id": "test-id",
            "timestamp": datetime.now(UTC),
            "action": "test",
            "context": {},
            "sensitivity": SensitivityLevel.PRIVATE,
            "device_id": "device-1",
            "sync_status": SyncStatus.SYNCED,
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        entry = MemoryEntry.from_dict(data)
        
        assert entry.sensitivity == SensitivityLevel.PRIVATE
        assert entry.sync_status == SyncStatus.SYNCED


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
    
    def test_validation_with_special_characters(self):
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
    
    def test_validation_with_context_mixed_types(self):
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
    
    def test_create_memory_entry_generates_unique_ids(self):
        """Test factory function generates unique IDs."""
        entries = [
            create_memory_entry(action="test", context={}, device_id="device-1")
            for _ in range(100)
        ]
        
        ids = [entry.id for entry in entries]
        assert len(ids) == len(set(ids))
    
    def test_timestamp_precision(self):
        """Test timestamps preserve microsecond precision."""
        timestamp = datetime(2024, 1, 1, 12, 30, 45, 123456)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            timestamp=timestamp
        )
        
        assert entry.timestamp.microsecond == 123456
        
        serialized = entry.to_dict()
        restored = MemoryEntry.from_dict(serialized)
        assert restored.timestamp.microsecond == 123456
    
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


class TestEnumConversions:
    """Comprehensive tests for enum conversions in serialization/deserialization."""
    
    def test_sensitivity_level_to_string_conversion(self):
        """Test that SensitivityLevel enums convert to strings in serialization."""
        for sensitivity in [SensitivityLevel.PUBLIC, SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
            entry = create_memory_entry(
                action="test",
                context={},
                device_id="device-1",
                sensitivity=sensitivity
            )
            
            result = entry.to_dict()
            assert isinstance(result["sensitivity"], str)
            assert result["sensitivity"] == sensitivity.value
    
    def test_sync_status_to_string_conversion(self):
        """Test that SyncStatus enums convert to strings in serialization."""
        for sync_status in [SyncStatus.PENDING, SyncStatus.SYNCED, SyncStatus.CONFLICT]:
            entry = create_memory_entry(
                action="test",
                context={},
                device_id="device-1"
            )
            entry.sync_status = sync_status
            
            result = entry.to_dict()
            assert isinstance(result["sync_status"], str)
            assert result["sync_status"] == sync_status.value
    
    def test_string_to_sensitivity_level_conversion(self):
        """Test that sensitivity strings convert to SensitivityLevel enums in deserialization."""
        for sensitivity_str, expected_enum in [
            ("public", SensitivityLevel.PUBLIC),
            ("private", SensitivityLevel.PRIVATE),
            ("sensitive", SensitivityLevel.SENSITIVE)
        ]:
            data = {
                "id": "test-id",
                "timestamp": "2024-01-01T00:00:00",
                "action": "test",
                "context": {},
                "sensitivity": sensitivity_str,
                "device_id": "device-1",
                "sync_status": "pending",
                "tags": [],
                "summary": None,
                "parent_id": None,
                "created_at": None,
                "updated_at": None
            }
            
            entry = MemoryEntry.from_dict(data)
            assert isinstance(entry.sensitivity, SensitivityLevel)
            assert entry.sensitivity == expected_enum
    
    def test_string_to_sync_status_conversion(self):
        """Test that sync status strings convert to SyncStatus enums in deserialization."""
        for status_str, expected_enum in [
            ("pending", SyncStatus.PENDING),
            ("synced", SyncStatus.SYNCED),
            ("conflict", SyncStatus.CONFLICT)
        ]:
            data = {
                "id": "test-id",
                "timestamp": "2024-01-01T00:00:00",
                "action": "test",
                "context": {},
                "sensitivity": "public",
                "device_id": "device-1",
                "sync_status": status_str,
                "tags": [],
                "summary": None,
                "parent_id": None,
                "created_at": None,
                "updated_at": None
            }
            
            entry = MemoryEntry.from_dict(data)
            assert isinstance(entry.sync_status, SyncStatus)
            assert entry.sync_status == expected_enum
    
    def test_enum_to_enum_passthrough_sensitivity(self):
        """Test that SensitivityLevel enum objects pass through deserialization unchanged."""
        for sensitivity in [SensitivityLevel.PUBLIC, SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
            data = {
                "id": "test-id",
                "timestamp": datetime.now(UTC),
                "action": "test",
                "context": {},
                "sensitivity": sensitivity,  # Already an enum
                "device_id": "device-1",
                "sync_status": SyncStatus.PENDING,
                "tags": [],
                "summary": None,
                "parent_id": None,
                "created_at": None,
                "updated_at": None
            }
            
            entry = MemoryEntry.from_dict(data)
            assert isinstance(entry.sensitivity, SensitivityLevel)
            assert entry.sensitivity == sensitivity
    
    def test_enum_to_enum_passthrough_sync_status(self):
        """Test that SyncStatus enum objects pass through deserialization unchanged."""
        for sync_status in [SyncStatus.PENDING, SyncStatus.SYNCED, SyncStatus.CONFLICT]:
            data = {
                "id": "test-id",
                "timestamp": datetime.now(UTC),
                "action": "test",
                "context": {},
                "sensitivity": SensitivityLevel.PUBLIC,
                "device_id": "device-1",
                "sync_status": sync_status,  # Already an enum
                "tags": [],
                "summary": None,
                "parent_id": None,
                "created_at": None,
                "updated_at": None
            }
            
            entry = MemoryEntry.from_dict(data)
            assert isinstance(entry.sync_status, SyncStatus)
            assert entry.sync_status == sync_status
    
    def test_round_trip_enum_conversion_all_combinations(self):
        """Test round-trip conversion for all enum combinations."""
        for sensitivity in [SensitivityLevel.PUBLIC, SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
            for sync_status in [SyncStatus.PENDING, SyncStatus.SYNCED, SyncStatus.CONFLICT]:
                entry = create_memory_entry(
                    action="test",
                    context={},
                    device_id="device-1",
                    sensitivity=sensitivity
                )
                entry.sync_status = sync_status
                
                # Serialize to dict (enums -> strings)
                serialized = entry.to_dict()
                assert isinstance(serialized["sensitivity"], str)
                assert isinstance(serialized["sync_status"], str)
                
                # Deserialize back (strings -> enums)
                restored = MemoryEntry.from_dict(serialized)
                assert isinstance(restored.sensitivity, SensitivityLevel)
                assert isinstance(restored.sync_status, SyncStatus)
                assert restored.sensitivity == sensitivity
                assert restored.sync_status == sync_status
    
    def test_invalid_sensitivity_string_raises_error(self):
        """Test that invalid sensitivity strings raise ValueError."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {},
            "sensitivity": "invalid_sensitivity",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)
    
    def test_invalid_sync_status_string_raises_error(self):
        """Test that invalid sync status strings raise ValueError."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "invalid_status",
            "tags": [],
            "summary": None,
            "parent_id": None,
            "created_at": None,
            "updated_at": None
        }
        
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)
    
    def test_enum_value_attribute_access(self):
        """Test that enum .value attribute returns the correct string."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.PRIVATE.value == "private"
        assert SensitivityLevel.SENSITIVE.value == "sensitive"
        
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.SYNCED.value == "synced"
        assert SyncStatus.CONFLICT.value == "conflict"
    
    def test_enum_comparison_with_strings(self):
        """Test that enums can be compared by value."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            sensitivity=SensitivityLevel.PRIVATE
        )
        
        # Enum value should match string
        assert entry.sensitivity.value == "private"
        assert entry.sync_status.value == "pending"
    
    def test_serialization_preserves_enum_values(self):
        """Test that serialization correctly extracts enum values."""
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        entry.sync_status = SyncStatus.CONFLICT
        
        result = entry.to_dict()
        
        # Should be strings, not enum objects
        assert result["sensitivity"] == "sensitive"
        assert result["sync_status"] == "conflict"
        assert not isinstance(result["sensitivity"], SensitivityLevel)
        assert not isinstance(result["sync_status"], SyncStatus)



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
    
    def test_validation_allows_empty_strings_in_tags(self):
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
        """Test validation allows duplicate tag values."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["tag1", "tag1", "tag2"]  # Duplicate tags
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_tags_with_empty_strings(self):
        """Test validation allows empty string tags."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["", "tag1", ""]  # Empty string tags
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_context_with_empty_string_keys(self):
        """Test validation allows empty string keys in context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={"": "value", "key": ""},  # Empty string key
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
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
            parent_id=12345  # Integer instead of string
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "Parent ID must be a non-empty string" in error
    
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
    
    def test_unique_id_generation(self):
        """Test that generated IDs are unique."""
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
        assert entry.tags == []
    
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
            for _ in range(10)
        ]
        
        ids = [entry.id for entry in entries]
        assert len(ids) == len(set(ids))  # All IDs should be unique
    
    def test_deserialization_ignores_extra_fields(self):
        """Test deserialization ignores extra fields not in model."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "action": "test",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-1",
            "sync_status": "pending",
            "tags": [],
            "extra_field_1": "should be ignored",
            "extra_field_2": 123
        }
        
        entry = MemoryEntry.from_dict(data)
        assert entry.id == "test-id"
        assert not hasattr(entry, "extra_field_1")
        assert not hasattr(entry, "extra_field_2")
    
    def test_deserialization_with_missing_optional_fields(self):
        """Test deserialization works with missing optional fields."""
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
    
    def test_serialization_with_context_containing_special_values(self):
        """Test serialization preserves special values in context."""
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
        
        result = entry.to_dict()
        assert result["context"]["string"] == "value"
        assert result["context"]["int"] == 123
        assert result["context"]["float"] == 45.67
        assert result["context"]["bool"] is True
        assert result["context"]["list"] == [1, 2, 3]
        assert result["context"]["dict"] == {"nested": "value"}
        assert result["context"]["none"] is None
    
    def test_validation_with_none_values_in_context(self):
        """Test validation allows None values in context."""
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test",
            context={"key1": "value", "key2": None, "key3": None},
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
            context={"string": "value", "int": 42, "float": 3.14, "bool": True, "none": None},
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
    
    def test_validation_with_non_string_tags(self):
        """Test validation fails with non-string type tags."""
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
    
    def test_validation_with_unicode_in_all_fields(self):
        """Test validation with unicode characters in all string fields."""
        unicode_string = "Hello 世界 🌍"
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
            action="test action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["string", 123, True]  # Mixed types should fail
        )
        
        is_valid, error = entry.validate()
        assert is_valid is False
        assert "tags must be strings" in error.lower()
    
    def test_validation_with_unicode_characters(self):
        """Test validation with unicode characters."""
        unicode_string = "Hello 世界 🌍 مرحبا Привет"
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action=unicode_string,
            context={"key": unicode_string},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[unicode_string]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None
    
    def test_validation_with_special_characters_in_strings(self):
        """Test validation with special characters."""
        special_chars = r"!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action=special_chars,
            context={"key": special_chars},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[special_chars]
        )
        
        is_valid, error = entry.validate()
        assert is_valid is True
        assert error is None 