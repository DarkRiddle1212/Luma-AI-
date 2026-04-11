"""Unit tests for validation manager."""

import pytest
from datetime import datetime, timedelta, UTC

from luma_memory.processing.validation import ValidationManager, ValidationError
from luma_memory.models import (
    MemoryEntry,
    SensitivityLevel,
    SyncStatus,
    create_memory_entry
)


class TestValidationManager:
    """Tests for ValidationManager class."""
    
    def test_validate_valid_entry(self):
        """Test validation passes for a valid entry."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1",
            tags=["test"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_validate_entry_empty_id(self):
        """Test validation fails for empty ID."""
        validator = ValidationManager()
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
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "ID" in error and "empty" in error.lower()

    def test_validate_entry_empty_action(self):
        """Test validation fails for empty action."""
        validator = ValidationManager()
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
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "Action" in error and "empty" in error.lower()
    
    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        validator = ValidationManager()
        data = {"action": "  test  ", "device_id": "device-1"}
        sanitized = validator.sanitize_input(data)
        assert sanitized["action"] == "test"


class TestSanitization:
    """Tests for input sanitization."""
    
    def test_sanitize_strips_whitespace(self):
        """Test sanitization strips leading and trailing whitespace."""
        validator = ValidationManager()
        data = {"action": "  test action  ", "device_id": "  device-1  "}
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["action"] == "test action"
        assert sanitized["device_id"] == "device-1"
    
    def test_sanitize_removes_null_bytes(self):
        """Test sanitization removes null bytes."""
        validator = ValidationManager()
        data = {"action": "test\x00action", "device_id": "device\x00-1"}
        sanitized = validator.sanitize_input(data)
        
        assert "\x00" not in sanitized["action"]
        assert "\x00" not in sanitized["device_id"]
        assert sanitized["action"] == "testaction"
        assert sanitized["device_id"] == "device-1"
    
    def test_sanitize_removes_control_characters(self):
        """Test sanitization removes control characters except newlines and tabs."""
        validator = ValidationManager()
        # Include various control characters
        data = {"action": "test\x01\x02\x03action", "device_id": "device\x1f-1"}
        sanitized = validator.sanitize_input(data)
        
        assert "\x01" not in sanitized["action"]
        assert "\x02" not in sanitized["action"]
        assert "\x03" not in sanitized["action"]
        assert "\x1f" not in sanitized["device_id"]
    
    def test_sanitize_preserves_newlines_and_tabs(self):
        """Test sanitization preserves newlines and tabs."""
        validator = ValidationManager()
        data = {"action": "test\naction\twith\ttabs", "device_id": "device-1"}
        sanitized = validator.sanitize_input(data)
        
        assert "\n" in sanitized["action"]
        assert "\t" in sanitized["action"]
        assert sanitized["action"] == "test\naction\twith\ttabs"
    
    def test_sanitize_prevents_path_traversal(self):
        """Test sanitization removes path traversal patterns."""
        validator = ValidationManager()
        data = {"action": "../../../etc/passwd", "device_id": "..\\..\\windows\\system32"}
        sanitized = validator.sanitize_input(data)
        
        assert "../" not in sanitized["action"]
        assert "..\\" not in sanitized["device_id"]
    
    def test_sanitize_html_escaping(self):
        """Test sanitization performs HTML escaping to prevent XSS."""
        validator = ValidationManager()
        data = {"action": "<script>alert('xss')</script>", "device_id": "device<>1"}
        sanitized = validator.sanitize_input(data)
        
        # HTML entities should be escaped
        assert "&lt;" in sanitized["action"] or "<" not in sanitized["action"]
        assert "&gt;" in sanitized["action"] or ">" not in sanitized["action"]
        assert "&lt;" in sanitized["device_id"] or "<" not in sanitized["device_id"]
        assert "&gt;" in sanitized["device_id"] or ">" not in sanitized["device_id"]
    
    def test_sanitize_removes_script_tags(self):
        """Test sanitization removes script tags."""
        validator = ValidationManager()
        data = {"action": "test<script>alert('xss')</script>action"}
        sanitized = validator.sanitize_input(data)
        
        # Script tags should be removed or escaped
        assert "script" not in sanitized["action"].lower() or "&lt;script" in sanitized["action"]
    
    def test_sanitize_removes_javascript_protocol(self):
        """Test sanitization removes javascript: protocol."""
        validator = ValidationManager()
        data = {"action": "javascript:alert('xss')"}
        sanitized = validator.sanitize_input(data)
        
        # javascript: should be removed or escaped
        assert "javascript:" not in sanitized["action"].lower()
    
    def test_sanitize_removes_event_handlers(self):
        """Test sanitization removes event handler attributes."""
        validator = ValidationManager()
        data = {
            "action": "test onerror=alert('xss')",
            "device_id": "device onload=malicious()",
            "summary": "text onclick=bad()"
        }
        sanitized = validator.sanitize_input(data)
        
        # Event handlers should be removed
        assert "onerror=" not in sanitized["action"].lower()
        assert "onload=" not in sanitized["device_id"].lower()
        assert "onclick=" not in sanitized["summary"].lower()
    
    def test_sanitize_nested_dictionary(self):
        """Test sanitization works on nested dictionaries."""
        validator = ValidationManager()
        data = {
            "action": "test",
            "context": {
                "nested": "  value  ",
                "deep": {
                    "level": "<script>xss</script>"
                }
            }
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["context"]["nested"] == "value"
        assert "script" not in sanitized["context"]["deep"]["level"].lower() or "&lt;script" in sanitized["context"]["deep"]["level"]
    
    def test_sanitize_list_of_strings(self):
        """Test sanitization works on lists of strings."""
        validator = ValidationManager()
        data = {
            "tags": ["  tag1  ", "tag<script>2</script>", "tag\x003"]
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["tags"][0] == "tag1"
        assert "script" not in sanitized["tags"][1].lower() or "&lt;script" in sanitized["tags"][1]
        assert "\x00" not in sanitized["tags"][2]
    
    def test_sanitize_list_of_dicts(self):
        """Test sanitization works on lists of dictionaries."""
        validator = ValidationManager()
        data = {
            "items": [
                {"name": "  item1  "},
                {"name": "<script>xss</script>"}
            ]
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["items"][0]["name"] == "item1"
        assert "script" not in sanitized["items"][1]["name"].lower() or "&lt;script" in sanitized["items"][1]["name"]
    
    def test_sanitize_nested_lists(self):
        """Test sanitization works on nested lists."""
        validator = ValidationManager()
        data = {
            "nested": [["  item1  ", "item<>2"], ["item\x003"]]
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["nested"][0][0] == "item1"
        assert "&lt;" in sanitized["nested"][0][1] or "<" not in sanitized["nested"][0][1]
        assert "\x00" not in sanitized["nested"][1][0]
    
    def test_sanitize_preserves_non_string_types(self):
        """Test sanitization preserves non-string types."""
        validator = ValidationManager()
        data = {
            "count": 42,
            "active": True,
            "ratio": 3.14,
            "empty": None
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["count"] == 42
        assert sanitized["active"] is True
        assert sanitized["ratio"] == 3.14
        assert sanitized["empty"] is None
    
    def test_sanitize_mixed_types_in_dict(self):
        """Test sanitization handles mixed types in dictionary."""
        validator = ValidationManager()
        data = {
            "action": "  test  ",
            "count": 10,
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "  value  "},
            "active": True
        }
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["action"] == "test"
        assert sanitized["count"] == 10
        assert sanitized["tags"] == ["tag1", "tag2"]
        assert sanitized["metadata"]["key"] == "value"
        assert sanitized["active"] is True
    
    def test_sanitize_empty_string(self):
        """Test sanitization handles empty strings."""
        validator = ValidationManager()
        data = {"action": "", "device_id": ""}
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["action"] == ""
        assert sanitized["device_id"] == ""
    
    def test_sanitize_whitespace_only_string(self):
        """Test sanitization converts whitespace-only strings to empty."""
        validator = ValidationManager()
        data = {"action": "   ", "device_id": "\t\n  "}
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["action"] == ""
        assert sanitized["device_id"] == ""
    
    def test_sanitize_empty_dict(self):
        """Test sanitization handles empty dictionary."""
        validator = ValidationManager()
        data = {}
        sanitized = validator.sanitize_input(data)
        
        assert sanitized == {}
    
    def test_sanitize_empty_list(self):
        """Test sanitization handles empty list."""
        validator = ValidationManager()
        data = {"tags": []}
        sanitized = validator.sanitize_input(data)
        
        assert sanitized["tags"] == []
    
    def test_sanitize_complex_xss_attempt(self):
        """Test sanitization handles complex XSS attempts."""
        validator = ValidationManager()
        data = {
            "action": "<img src=x onerror=alert('xss')>",
            "device_id": "javascript:void(0)",
            "summary": "<iframe src='evil.com'></iframe>"
        }
        sanitized = validator.sanitize_input(data)
        
        # All dangerous patterns should be removed or escaped
        assert "onerror=" not in sanitized["action"].lower()
        assert "javascript:" not in sanitized["device_id"].lower()
        assert "iframe" not in sanitized["summary"].lower() or "&lt;iframe" in sanitized["summary"]
    
    def test_sanitize_sql_injection_patterns(self):
        """Test sanitization handles SQL injection patterns."""
        validator = ValidationManager()
        # Note: The sanitizer focuses on XSS, not SQL injection
        # SQL injection prevention should be handled by parameterized queries
        data = {
            "action": "test'; DROP TABLE users; --",
            "device_id": "1' OR '1'='1"
        }
        sanitized = validator.sanitize_input(data)
        
        # Sanitization should at least strip whitespace and escape HTML
        assert sanitized["action"].strip() != ""
        assert sanitized["device_id"].strip() != ""
    
    def test_sanitize_unicode_characters(self):
        """Test sanitization preserves valid unicode characters."""
        validator = ValidationManager()
        data = {
            "action": "测试动作",
            "device_id": "устройство-1",
            "summary": "résumé with émojis 🎉"
        }
        sanitized = validator.sanitize_input(data)
        
        # Unicode should be preserved
        assert "测试" in sanitized["action"]
        assert "устройство" in sanitized["device_id"]
        assert "résumé" in sanitized["summary"]
        assert "🎉" in sanitized["summary"]
    
    def test_sanitize_special_characters_in_context(self):
        """Test sanitization handles special characters in context."""
        validator = ValidationManager()
        data = {
            "context": {
                "url": "https://example.com?param=value&other=123",
                "email": "user@example.com",
                "path": "/home/user/file.txt"
            }
        }
        sanitized = validator.sanitize_input(data)
        
        # Valid special characters should be preserved (though HTML-escaped)
        assert "example.com" in sanitized["context"]["url"]
        assert "example.com" in sanitized["context"]["email"]
        assert "home" in sanitized["context"]["path"]
    
    def test_validate_and_raise_valid(self):
        """Test validate_and_raise doesn't raise for valid entry."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1"
        )
        validator.validate_and_raise(entry)
    
    def test_validate_and_raise_invalid(self):
        """Test validate_and_raise raises ValidationError for invalid entry."""
        validator = ValidationManager()
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
        
        with pytest.raises(ValidationError):
            validator.validate_and_raise(entry)


class TestTypeValidation:
    """Tests for field type validation."""
    
    def test_id_wrong_type_integer(self):
        """Test validation fails when ID is an integer instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id=12345,
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "ID" in error or "id" in error
        assert "string" in error.lower()
    
    def test_id_wrong_type_list(self):
        """Test validation fails when ID is a list instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id=["test", "id"],
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "string" in error.lower()
    
    def test_timestamp_wrong_type_string(self):
        """Test validation fails when timestamp is a string instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp="2024-01-01",
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "timestamp" in error.lower()
        assert "datetime" in error.lower()
    
    def test_timestamp_wrong_type_integer(self):
        """Test validation fails when timestamp is an integer instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=1234567890,
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "timestamp" in error.lower()
        assert "datetime" in error.lower()
    
    def test_action_wrong_type_integer(self):
        """Test validation fails when action is an integer instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action=123,
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "action" in error.lower()
        assert "string" in error.lower()
    
    def test_action_wrong_type_dict(self):
        """Test validation fails when action is a dict instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action={"type": "action"},
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "action" in error.lower()
        assert "string" in error.lower()
    
    def test_context_wrong_type_string(self):
        """Test validation fails when context is a string instead of dict."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context="not a dict",
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "context" in error.lower()
        assert "dict" in error.lower()
    
    def test_context_wrong_type_list(self):
        """Test validation fails when context is a list instead of dict."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context=["item1", "item2"],
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "context" in error.lower()
        assert "dict" in error.lower()
    
    def test_sensitivity_wrong_type_string(self):
        """Test validation fails when sensitivity is a string instead of enum."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity="public",
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "sensitivity" in error.lower()
        assert "SensitivityLevel" in error
    
    def test_sensitivity_wrong_type_integer(self):
        """Test validation fails when sensitivity is an integer instead of enum."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=1,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "sensitivity" in error.lower()
        assert "SensitivityLevel" in error
    
    def test_device_id_wrong_type_integer(self):
        """Test validation fails when device_id is an integer instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id=12345,
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "device" in error.lower() and "id" in error.lower()
        assert "string" in error.lower()
    
    def test_device_id_wrong_type_dict(self):
        """Test validation fails when device_id is a dict instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id={"device": "1"},
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "device" in error.lower() and "id" in error.lower()
        assert "string" in error.lower()
    
    def test_sync_status_wrong_type_string(self):
        """Test validation fails when sync_status is a string instead of enum."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status="pending",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "sync" in error.lower() and "status" in error.lower()
        assert "SyncStatus" in error
    
    def test_sync_status_wrong_type_integer(self):
        """Test validation fails when sync_status is an integer instead of enum."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=0,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "sync" in error.lower() and "status" in error.lower()
        assert "SyncStatus" in error
    
    def test_tags_wrong_type_string(self):
        """Test validation fails when tags is a string instead of list."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags="tag1,tag2"
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tags" in error.lower()
        assert "list" in error.lower()
    
    def test_tags_wrong_type_dict(self):
        """Test validation fails when tags is a dict instead of list."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags={"tag": "value"}
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tags" in error.lower()
        assert "list" in error.lower()
    
    def test_tags_contains_non_string_integer(self):
        """Test validation fails when tags list contains an integer."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["tag1", 123, "tag3"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tag" in error.lower()
        assert "string" in error.lower()
        assert "1" in error  # Index 1
    
    def test_tags_contains_non_string_dict(self):
        """Test validation fails when tags list contains a dict."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["tag1", {"nested": "tag"}, "tag3"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tag" in error.lower()
        assert "string" in error.lower()
    
    def test_summary_wrong_type_integer(self):
        """Test validation fails when summary is an integer instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=12345
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "summary" in error.lower()
        assert "string" in error.lower()
    
    def test_summary_wrong_type_list(self):
        """Test validation fails when summary is a list instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=["summary", "text"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "summary" in error.lower()
        assert "string" in error.lower()
    
    def test_parent_id_wrong_type_integer(self):
        """Test validation fails when parent_id is an integer instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            parent_id=12345
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() and "id" in error.lower()
        assert "string" in error.lower()
    
    def test_parent_id_wrong_type_list(self):
        """Test validation fails when parent_id is a list instead of string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            parent_id=["parent", "id"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() and "id" in error.lower()
        assert "string" in error.lower()
    
    def test_created_at_wrong_type_string(self):
        """Test validation fails when created_at is a string instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            created_at="2024-01-01"
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "created_at" in error.lower()
        assert "datetime" in error.lower()
    
    def test_created_at_wrong_type_integer(self):
        """Test validation fails when created_at is an integer instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            created_at=1234567890
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "created_at" in error.lower()
        assert "datetime" in error.lower()
    
    def test_updated_at_wrong_type_string(self):
        """Test validation fails when updated_at is a string instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            updated_at="2024-01-01"
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "updated_at" in error.lower()
        assert "datetime" in error.lower()
    
    def test_updated_at_wrong_type_integer(self):
        """Test validation fails when updated_at is an integer instead of datetime."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            updated_at=1234567890
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "updated_at" in error.lower()
        assert "datetime" in error.lower()
    
    def test_all_correct_types(self):
        """Test validation passes when all fields have correct types."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1",
            tags=["tag1", "tag2"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_optional_fields_correct_types(self):
        """Test validation passes when optional fields have correct types."""
        validator = ValidationManager()
        now = datetime.now(UTC)
        entry = MemoryEntry(
            id="test-id",
            timestamp=now,
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["tag1"],
            summary="This is a summary",
            parent_id="parent-123",
            created_at=now,
            updated_at=now
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_optional_fields_none_allowed(self):
        """Test validation passes when optional fields are None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[],
            summary=None,
            parent_id=None,
            created_at=None,
            updated_at=None
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None


class TestRequiredFieldValidation:
    """Tests for required field validation."""
    
    def test_missing_id(self):
        """Test validation fails when ID is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id=None,
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "ID" in error
    
    def test_empty_id(self):
        """Test validation fails when ID is empty string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "ID" in error
        assert "empty" in error.lower()
    
    def test_whitespace_only_id(self):
        """Test validation fails when ID is whitespace only."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="   ",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "ID" in error
        assert "empty" in error.lower()
    
    def test_missing_action(self):
        """Test validation fails when action is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action=None,
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Action" in error or "action" in error
    
    def test_empty_action(self):
        """Test validation fails when action is empty string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "Action" in error
        assert "empty" in error.lower()
    
    def test_whitespace_only_action(self):
        """Test validation fails when action is whitespace only."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="   ",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "Action" in error
        assert "empty" in error.lower()
    
    def test_missing_device_id(self):
        """Test validation fails when device_id is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id=None,
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Device ID" in error or "device_id" in error
    
    def test_empty_device_id(self):
        """Test validation fails when device_id is empty string."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "Device ID" in error
        assert "empty" in error.lower()
    
    def test_whitespace_only_device_id(self):
        """Test validation fails when device_id is whitespace only."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="   ",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "Device ID" in error
        assert "empty" in error.lower()
    
    def test_missing_timestamp(self):
        """Test validation fails when timestamp is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=None,
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Timestamp" in error or "timestamp" in error
    
    def test_missing_context(self):
        """Test validation fails when context is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context=None,
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Context" in error or "context" in error
    
    def test_missing_sensitivity(self):
        """Test validation fails when sensitivity is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=None,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Sensitivity" in error or "sensitivity" in error
    
    def test_missing_sync_status(self):
        """Test validation fails when sync_status is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=None,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Sync status" in error or "sync_status" in error
    
    def test_missing_tags(self):
        """Test validation fails when tags is None."""
        validator = ValidationManager()
        entry = MemoryEntry(
            id="test-id",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": "value"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=None
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert error is not None
        assert "Tags" in error or "tags" in error
    
    def test_all_required_fields_present(self):
        """Test validation passes when all required fields are present."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_empty_tags_list_allowed(self):
        """Test validation passes when tags is an empty list."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_empty_context_dict_allowed(self):
        """Test validation passes when context is an empty dict."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test_action",
            context={},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None


class TestValueValidation:
    """Tests for field value validation (ranges, formats, enum values)."""
    
    def test_id_format_valid(self):
        """Test validation passes for valid ID format."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        # Override with custom ID
        entry.id = "valid-id_123"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_id_format_invalid_special_chars(self):
        """Test validation fails for ID with invalid special characters."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.id = "invalid@id#123"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "alphanumeric" in error.lower() or "format" in error.lower()
    
    def test_id_format_invalid_spaces(self):
        """Test validation fails for ID with spaces."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.id = "invalid id 123"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "alphanumeric" in error.lower() or "format" in error.lower()
    
    def test_action_length_at_max(self):
        """Test validation passes for action at maximum length."""
        validator = ValidationManager()
        max_action = "a" * validator.MAX_ACTION_LENGTH
        entry = create_memory_entry(
            action=max_action,
            context={},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_action_length_exceeds_max(self):
        """Test validation fails for action exceeding maximum length."""
        validator = ValidationManager()
        too_long_action = "a" * (validator.MAX_ACTION_LENGTH + 1)
        entry = create_memory_entry(
            action=too_long_action,
            context={},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "action" in error.lower()
        assert "maximum length" in error.lower()
    
    def test_device_id_length_at_max(self):
        """Test validation passes for device_id at maximum length."""
        validator = ValidationManager()
        max_device_id = "d" * validator.MAX_DEVICE_ID_LENGTH
        entry = create_memory_entry(
            action="test",
            context={},
            device_id=max_device_id,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_device_id_length_exceeds_max(self):
        """Test validation fails for device_id exceeding maximum length."""
        validator = ValidationManager()
        too_long_device_id = "d" * (validator.MAX_DEVICE_ID_LENGTH + 1)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id=too_long_device_id,
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "device" in error.lower() and "id" in error.lower()
        assert "maximum length" in error.lower()
    
    def test_tags_count_at_max(self):
        """Test validation passes for tags at maximum count."""
        validator = ValidationManager()
        max_tags = [f"tag{i}" for i in range(validator.MAX_TAGS_COUNT)]
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=max_tags
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_tags_count_exceeds_max(self):
        """Test validation fails for tags exceeding maximum count."""
        validator = ValidationManager()
        too_many_tags = [f"tag{i}" for i in range(validator.MAX_TAGS_COUNT + 1)]
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=too_many_tags
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tags" in error.lower()
        assert "maximum" in error.lower()
    
    def test_tag_length_at_max(self):
        """Test validation passes for tag at maximum length."""
        validator = ValidationManager()
        max_tag = "t" * validator.MAX_TAG_LENGTH
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[max_tag]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_tag_length_exceeds_max(self):
        """Test validation fails for tag exceeding maximum length."""
        validator = ValidationManager()
        too_long_tag = "t" * (validator.MAX_TAG_LENGTH + 1)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[too_long_tag]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tag" in error.lower()
        assert "maximum length" in error.lower()
    
    def test_tag_empty_string(self):
        """Test validation fails for empty tag."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=["valid", "", "another"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tag" in error.lower()
        assert "empty" in error.lower()
    
    def test_tag_whitespace_only(self):
        """Test validation fails for whitespace-only tag."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=["valid", "   ", "another"]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "tag" in error.lower()
        assert "empty" in error.lower() or "whitespace" in error.lower()
    
    def test_summary_length_at_max(self):
        """Test validation passes for summary at maximum length."""
        validator = ValidationManager()
        max_summary = "s" * validator.MAX_SUMMARY_LENGTH
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.summary = max_summary
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_summary_length_exceeds_max(self):
        """Test validation fails for summary exceeding maximum length."""
        validator = ValidationManager()
        too_long_summary = "s" * (validator.MAX_SUMMARY_LENGTH + 1)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.summary = too_long_summary
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "summary" in error.lower()
        assert "maximum length" in error.lower()
    
    def test_parent_id_format_valid(self):
        """Test validation passes for valid parent_id format."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = "valid-parent_123"
        entry.summary = "Summary for child entry"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_parent_id_format_invalid(self):
        """Test validation fails for invalid parent_id format."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = "invalid@parent#123"
        entry.summary = "Summary for child entry"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() and "id" in error.lower()
        assert "alphanumeric" in error.lower() or "format" in error.lower()
    
    def test_parent_id_empty_string(self):
        """Test validation fails for empty parent_id."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = ""
        entry.summary = "Summary for child entry"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() and "id" in error.lower()
        assert "empty" in error.lower()
    
    def test_parent_id_whitespace_only(self):
        """Test validation fails for whitespace-only parent_id."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = "   "
        entry.summary = "Summary for child entry"
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() and "id" in error.lower()
        assert "empty" in error.lower()
    
    def test_context_size_at_max(self):
        """Test validation passes for context at maximum size."""
        validator = ValidationManager()
        # Create a large context that's just under the limit
        large_value = "x" * (validator.MAX_CONTEXT_SIZE // 2)
        entry = create_memory_entry(
            action="test",
            context={"data": large_value},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_context_size_exceeds_max(self):
        """Test validation fails for context exceeding maximum size."""
        validator = ValidationManager()
        # Create a context that exceeds the limit
        huge_value = "x" * (validator.MAX_CONTEXT_SIZE + 1000)
        entry = create_memory_entry(
            action="test",
            context={"data": huge_value},
            device_id="device-1",
            tags=[]
        )
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "context" in error.lower()
        assert "maximum" in error.lower()
    
    def test_timestamp_in_past_valid(self):
        """Test validation passes for timestamp in the past."""
        validator = ValidationManager()
        past_time = datetime.now(UTC) - timedelta(days=30)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.timestamp = past_time
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_timestamp_recent_past_valid(self):
        """Test validation passes for recent timestamp."""
        validator = ValidationManager()
        recent_time = datetime.now(UTC) - timedelta(minutes=5)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.timestamp = recent_time
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_timestamp_near_future_valid(self):
        """Test validation passes for timestamp slightly in future (within 1 day)."""
        validator = ValidationManager(strict_mode=True)
        near_future = datetime.now(UTC) + timedelta(hours=12)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.timestamp = near_future
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_timestamp_far_future_invalid_strict_mode(self):
        """Test validation fails for timestamp far in future in strict mode."""
        validator = ValidationManager(strict_mode=True)
        far_future = datetime.now(UTC) + timedelta(days=2)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.timestamp = far_future
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "timestamp" in error.lower()
        assert "future" in error.lower()
    
    def test_timestamp_far_future_valid_non_strict_mode(self):
        """Test validation passes for timestamp far in future in non-strict mode."""
        validator = ValidationManager(strict_mode=False)
        far_future = datetime.now(UTC) + timedelta(days=365)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.timestamp = far_future
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sensitivity_enum_valid_public(self):
        """Test validation passes for valid PUBLIC sensitivity."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sensitivity = SensitivityLevel.PUBLIC
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sensitivity_enum_valid_private(self):
        """Test validation passes for valid PRIVATE sensitivity."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sensitivity = SensitivityLevel.PRIVATE
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sensitivity_enum_valid_sensitive(self):
        """Test validation passes for valid SENSITIVE sensitivity."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sensitivity = SensitivityLevel.SENSITIVE
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sync_status_enum_valid_pending(self):
        """Test validation passes for valid PENDING sync status."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sync_status = SyncStatus.PENDING
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sync_status_enum_valid_synced(self):
        """Test validation passes for valid SYNCED sync status."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sync_status = SyncStatus.SYNCED
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_sync_status_enum_valid_conflict(self):
        """Test validation passes for valid CONFLICT sync status."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.sync_status = SyncStatus.CONFLICT
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_business_rule_parent_id_requires_summary(self):
        """Test validation fails when parent_id is set but summary is missing."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = "parent-123"
        entry.summary = None
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() or "summary" in error.lower()
    
    def test_business_rule_parent_id_with_empty_summary(self):
        """Test validation fails when parent_id is set but summary is empty."""
        validator = ValidationManager()
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.parent_id = "parent-123"
        entry.summary = ""
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "parent" in error.lower() or "summary" in error.lower()
    
    def test_business_rule_updated_at_before_created_at(self):
        """Test validation fails when updated_at is before created_at."""
        validator = ValidationManager()
        now = datetime.now(UTC)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.created_at = now
        entry.updated_at = now - timedelta(hours=1)
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is False
        assert "updated_at" in error.lower() or "created_at" in error.lower()
    
    def test_business_rule_updated_at_equals_created_at(self):
        """Test validation passes when updated_at equals created_at."""
        validator = ValidationManager()
        now = datetime.now(UTC)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.created_at = now
        entry.updated_at = now
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None
    
    def test_business_rule_updated_at_after_created_at(self):
        """Test validation passes when updated_at is after created_at."""
        validator = ValidationManager()
        now = datetime.now(UTC)
        entry = create_memory_entry(
            action="test",
            context={},
            device_id="device-1",
            tags=[]
        )
        entry.created_at = now
        entry.updated_at = now + timedelta(hours=1)
        
        is_valid, error = validator.validate_entry(entry)
        assert is_valid is True
        assert error is None

