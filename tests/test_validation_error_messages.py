"""Unit tests for validation error message descriptiveness."""

import pytest
from datetime import datetime, timedelta
from luma_memory.processing.validation import ValidationManager, ValidationError
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus, create_memory_entry


class TestDescriptiveErrorMessages:
    """Tests to verify that error messages are descriptive and helpful."""
    
    @pytest.fixture
    def validator(self):
        """Create a ValidationManager instance for testing."""
        return ValidationManager(strict_mode=True)
    
    @pytest.fixture
    def valid_entry(self):
        """Create a valid memory entry for testing."""
        return create_memory_entry(
            action="test_action",
            context={"key": "value"},
            device_id="test_device",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["test"]
        )
    
    def test_missing_id_error_message(self, validator, valid_entry):
        """Test that missing ID produces a descriptive error message."""
        valid_entry.id = ""
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "ID" in error
        assert "required" in error.lower()
        assert "empty" in error.lower()
    
    def test_missing_action_error_message(self, validator, valid_entry):
        """Test that missing action produces a descriptive error message."""
        valid_entry.action = ""
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Action" in error
        assert "required" in error.lower()
    
    def test_missing_device_id_error_message(self, validator, valid_entry):
        """Test that missing device_id produces a descriptive error message."""
        valid_entry.device_id = ""
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Device ID" in error
        assert "required" in error.lower()
    
    def test_wrong_type_error_includes_actual_type(self, validator, valid_entry):
        """Test that type errors include the actual type received."""
        # Create entry with wrong type for action
        valid_entry.action = 123  # Should be string
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Action" in error
        assert "string" in error.lower()
        assert "int" in error.lower()
    
    def test_wrong_context_type_error_message(self, validator, valid_entry):
        """Test that wrong context type produces descriptive error."""
        valid_entry.context = "not a dict"
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Context" in error
        assert "dictionary" in error.lower()
        assert "str" in error.lower()
    
    def test_invalid_enum_error_lists_valid_values(self, validator):
        """Test that invalid enum errors list all valid values."""
        is_valid, error = validator.validate_enum_string(
            SensitivityLevel, "invalid_value", "sensitivity"
        )
        assert is_valid is False
        assert error is not None
        assert "Invalid sensitivity" in error
        assert "public" in error
        assert "private" in error
        assert "sensitive" in error
    
    def test_action_too_long_error_includes_limit(self, validator, valid_entry):
        """Test that length errors include the maximum allowed length."""
        valid_entry.action = "x" * (ValidationManager.MAX_ACTION_LENGTH + 1)
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Action" in error
        assert "maximum length" in error.lower()
        assert str(ValidationManager.MAX_ACTION_LENGTH) in error
    
    def test_too_many_tags_error_includes_limit(self, validator, valid_entry):
        """Test that tag count errors include the maximum allowed count."""
        valid_entry.tags = [f"tag{i}" for i in range(ValidationManager.MAX_TAGS_COUNT + 1)]
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "tags" in error.lower()
        assert "maximum" in error.lower()
        assert str(ValidationManager.MAX_TAGS_COUNT) in error
    
    def test_tag_too_long_error_includes_index_and_limit(self, validator, valid_entry):
        """Test that tag length errors include the tag index and limit."""
        valid_entry.tags = ["short", "x" * (ValidationManager.MAX_TAG_LENGTH + 1)]
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Tag" in error
        assert "index" in error.lower()
        assert "1" in error  # Index of the long tag
        assert "maximum length" in error.lower()
        assert str(ValidationManager.MAX_TAG_LENGTH) in error
    
    def test_empty_tag_error_includes_index(self, validator, valid_entry):
        """Test that empty tag errors include the tag index."""
        valid_entry.tags = ["valid", "  ", "another"]
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Tag" in error
        assert "index" in error.lower()
        assert "1" in error  # Index of the empty tag
        assert "empty" in error.lower()
    
    def test_invalid_id_format_error_describes_requirements(self, validator, valid_entry):
        """Test that ID format errors describe the required format."""
        valid_entry.id = "invalid@id#with$special%chars"
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "ID" in error
        assert "alphanumeric" in error.lower()
        assert "hyphens" in error.lower() or "underscores" in error.lower()
    
    def test_invalid_parent_id_format_error(self, validator, valid_entry):
        """Test that parent ID format errors are descriptive."""
        valid_entry.parent_id = "invalid@parent"
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Parent ID" in error
        assert "alphanumeric" in error.lower()
    
    def test_context_too_large_error_includes_limit(self, validator, valid_entry):
        """Test that context size errors include the maximum size."""
        valid_entry.context = {"data": "x" * ValidationManager.MAX_CONTEXT_SIZE}
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Context" in error
        assert "size" in error.lower()
        assert "maximum" in error.lower()
        assert str(ValidationManager.MAX_CONTEXT_SIZE) in error
    
    def test_summary_too_long_error_includes_limit(self, validator, valid_entry):
        """Test that summary length errors include the maximum length."""
        valid_entry.summary = "x" * (ValidationManager.MAX_SUMMARY_LENGTH + 1)
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Summary" in error
        assert "maximum length" in error.lower()
        assert str(ValidationManager.MAX_SUMMARY_LENGTH) in error
    
    def test_parent_without_summary_error_is_clear(self, validator, valid_entry):
        """Test that parent_id without summary produces clear error."""
        valid_entry.parent_id = "parent-123"
        valid_entry.summary = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "parent_id" in error.lower()
        assert "summary" in error.lower()
        assert "must have" in error.lower()
    
    def test_timestamp_in_future_error_is_descriptive(self, validator, valid_entry):
        """Test that future timestamp errors are descriptive."""
        future_time = datetime.now() + timedelta(days=2)
        valid_entry.timestamp = future_time
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Timestamp" in error
        assert "future" in error.lower()
        assert "1 day" in error.lower()
    
    def test_updated_before_created_error_is_clear(self, validator, valid_entry):
        """Test that timestamp relationship errors are clear."""
        valid_entry.created_at = datetime(2024, 1, 2)
        valid_entry.updated_at = datetime(2024, 1, 1)
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Updated_at" in error
        assert "created_at" in error.lower()
        assert "earlier" in error.lower()
    
    def test_partial_update_invalid_action_error(self, validator):
        """Test that partial update errors are descriptive."""
        updates = {"action": ""}
        is_valid, error = validator.validate_partial_update(updates)
        assert is_valid is False
        assert error is not None
        assert "Action" in error
        assert "empty" in error.lower()
    
    def test_partial_update_invalid_tags_type_error(self, validator):
        """Test that partial update type errors are descriptive."""
        updates = {"tags": "not a list"}
        is_valid, error = validator.validate_partial_update(updates)
        assert is_valid is False
        assert error is not None
        assert "Tags" in error
        assert "list" in error.lower()
    
    def test_partial_update_invalid_sensitivity_error(self, validator):
        """Test that partial update enum errors are descriptive."""
        updates = {"sensitivity": "invalid_level"}
        is_valid, error = validator.validate_partial_update(updates)
        assert is_valid is False
        assert error is not None
        assert "Invalid sensitivity level" in error
        assert "invalid_level" in error
    
    def test_custom_rule_error_includes_rule_name(self, validator, valid_entry):
        """Test that custom rule errors include the rule name."""
        def failing_rule(entry):
            return False, "This is a custom error"
        
        validator.add_custom_rule(failing_rule, "my_custom_rule")
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "my_custom_rule" in error
        assert "Custom rule" in error
        assert "This is a custom error" in error
    
    def test_custom_rule_exception_error_is_descriptive(self, validator, valid_entry):
        """Test that custom rule exceptions produce descriptive errors."""
        def buggy_rule(entry):
            raise ValueError("Something went wrong in validation")
        
        validator.add_custom_rule(buggy_rule, "buggy_rule")
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "buggy_rule" in error
        assert "exception" in error.lower()
        assert "Something went wrong" in error
    
    def test_validate_and_raise_includes_error_message(self, validator, valid_entry):
        """Test that validate_and_raise includes the error message in exception."""
        valid_entry.action = ""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_and_raise(valid_entry)
        
        error_message = str(exc_info.value)
        assert "Action" in error_message
        assert "required" in error_message.lower()
    
    def test_tag_in_list_type_error_includes_index(self, validator, valid_entry):
        """Test that tag type errors include the specific index."""
        valid_entry.tags = ["valid", 123, "another"]
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Tag" in error
        assert "index" in error.lower()
        assert "1" in error  # Index of the invalid tag
        assert "string" in error.lower()
        assert "int" in error.lower()
    
    def test_error_messages_are_not_generic(self, validator, valid_entry):
        """Test that error messages are specific, not generic."""
        # Test various validation failures to ensure messages are specific
        test_cases = [
            ({"id": ""}, ["ID", "required"]),
            ({"action": ""}, ["Action", "required"]),
            ({"device_id": ""}, ["Device ID", "required"]),
            ({"action": 123}, ["Action", "string"]),
            ({"context": []}, ["Context", "dictionary"]),
        ]
        
        for modifications, expected_terms in test_cases:
            entry = create_memory_entry(
                action="test",
                context={"key": "value"},
                device_id="device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["test"]
            )
            
            # Apply modifications
            for key, value in modifications.items():
                setattr(entry, key, value)
            
            is_valid, error = validator.validate_entry(entry)
            assert is_valid is False
            assert error is not None
            
            # Check that all expected terms are in the error message
            for term in expected_terms:
                assert term in error, f"Expected '{term}' in error message: {error}"
    
    def test_device_id_too_long_error_includes_limit(self, validator, valid_entry):
        """Test that device ID length errors include the maximum length."""
        valid_entry.device_id = "x" * (ValidationManager.MAX_DEVICE_ID_LENGTH + 1)
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Device ID" in error
        assert "maximum length" in error.lower()
        assert str(ValidationManager.MAX_DEVICE_ID_LENGTH) in error
    
    def test_missing_timestamp_error_message(self, validator, valid_entry):
        """Test that missing timestamp produces a descriptive error message."""
        valid_entry.timestamp = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Timestamp" in error
        assert "required" in error.lower()
    
    def test_missing_context_error_message(self, validator, valid_entry):
        """Test that missing context produces a descriptive error message."""
        valid_entry.context = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Context" in error
        assert "required" in error.lower()
    
    def test_missing_sensitivity_error_message(self, validator, valid_entry):
        """Test that missing sensitivity produces a descriptive error message."""
        valid_entry.sensitivity = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Sensitivity" in error
        assert "required" in error.lower()
    
    def test_missing_sync_status_error_message(self, validator, valid_entry):
        """Test that missing sync_status produces a descriptive error message."""
        valid_entry.sync_status = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Sync status" in error
        assert "required" in error.lower()
    
    def test_missing_tags_error_message(self, validator, valid_entry):
        """Test that missing tags field produces a descriptive error message."""
        valid_entry.tags = None
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Tags" in error
        assert "required" in error.lower()
    
    def test_wrong_timestamp_type_error_message(self, validator, valid_entry):
        """Test that wrong timestamp type produces descriptive error."""
        valid_entry.timestamp = "2024-01-01"  # Should be datetime
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Timestamp" in error
        assert "datetime" in error.lower()
        assert "str" in error.lower()
    
    def test_wrong_sensitivity_type_error_message(self, validator, valid_entry):
        """Test that wrong sensitivity type produces descriptive error."""
        valid_entry.sensitivity = "public"  # Should be enum
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Sensitivity" in error
        assert "SensitivityLevel" in error
    
    def test_wrong_sync_status_type_error_message(self, validator, valid_entry):
        """Test that wrong sync_status type produces descriptive error."""
        valid_entry.sync_status = "pending"  # Should be enum
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Sync status" in error
        assert "SyncStatus" in error
    
    def test_wrong_tags_type_error_message(self, validator, valid_entry):
        """Test that wrong tags type produces descriptive error."""
        valid_entry.tags = "tag1,tag2"  # Should be list
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Tags" in error
        assert "list" in error.lower()
        assert "str" in error.lower()
    
    def test_empty_parent_id_error_message(self, validator, valid_entry):
        """Test that empty parent_id produces a descriptive error message."""
        valid_entry.parent_id = "   "
        is_valid, error = validator.validate_entry(valid_entry)
        assert is_valid is False
        assert error is not None
        assert "Parent ID" in error
        assert "empty" in error.lower() or "whitespace" in error.lower()
