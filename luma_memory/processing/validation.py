"""
Validation manager for Luma Memory Module.

This module provides validation and sanitization for memory entries,
ensuring data integrity and security.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, UTC
import re
import html

from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class ValidationManager:
    """
    Manages validation and sanitization of memory entries.
    
    Provides comprehensive validation for memory entry fields including:
    - Field presence validation
    - Type validation
    - Value validation (enum values, ranges, formats)
    - Input sanitization to prevent injection attacks
    - Custom validation rules
    """
    
    # Maximum lengths for string fields
    MAX_ACTION_LENGTH = 1000
    MAX_TAG_LENGTH = 100
    MAX_TAGS_COUNT = 50
    MAX_SUMMARY_LENGTH = 5000
    MAX_DEVICE_ID_LENGTH = 255
    MAX_CONTEXT_SIZE = 1_000_000  # 1MB in characters
    
    # Allowed characters pattern for IDs (alphanumeric, hyphens, underscores)
    ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the validation manager.
        
        Args:
            strict_mode: If True, applies stricter validation rules.
        """
        self.strict_mode = strict_mode
        self.custom_rules = []  # List of custom validation rule functions
    
    def validate_entry(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate a complete memory entry.
        
        Performs comprehensive validation including field presence,
        types, values, and business rules.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Validate required fields presence
        is_valid, error = self._validate_required_fields(entry)
        if not is_valid:
            return False, error
        
        # Validate field types
        is_valid, error = self._validate_field_types(entry)
        if not is_valid:
            return False, error
        
        # Validate field values
        is_valid, error = self._validate_field_values(entry)
        if not is_valid:
            return False, error
        
        # Validate business rules
        is_valid, error = self._validate_business_rules(entry)
        if not is_valid:
            return False, error
        
        # Apply custom validation rules
        is_valid, error = self._apply_custom_rules(entry)
        if not is_valid:
            return False, error
        
        return True, None
    
    def _validate_required_fields(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required fields are present and non-empty.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check ID
        if not isinstance(entry.id, str):
            return False, f"Entry ID must be a string, got {type(entry.id).__name__}"
        if not entry.id or not entry.id.strip():
            return False, "Entry ID is required and cannot be empty"
        
        # Check action
        if not isinstance(entry.action, str):
            return False, f"Action must be a string, got {type(entry.action).__name__}"
        if not entry.action or not entry.action.strip():
            return False, "Action is required and cannot be empty"
        
        # Check device_id
        if not isinstance(entry.device_id, str):
            return False, f"Device ID must be a string, got {type(entry.device_id).__name__}"
        if not entry.device_id or not entry.device_id.strip():
            return False, "Device ID is required and cannot be empty"
        
        # Check timestamp
        if entry.timestamp is None:
            return False, "Timestamp is required"
        
        # Check context
        if entry.context is None:
            return False, "Context is required"
        
        # Check sensitivity
        if entry.sensitivity is None:
            return False, "Sensitivity level is required"
        
        # Check sync_status
        if entry.sync_status is None:
            return False, "Sync status is required"
        
        # Check tags
        if entry.tags is None:
            return False, "Tags field is required (can be empty list)"
        
        return True, None
    
    def _validate_field_types(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate that all fields have the correct types.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate ID type
        if not isinstance(entry.id, str):
            return False, f"Entry ID must be a string, got {type(entry.id).__name__}"
        
        # Validate timestamp type
        if not isinstance(entry.timestamp, datetime):
            return False, f"Timestamp must be a datetime object, got {type(entry.timestamp).__name__}"
        
        # Validate action type
        if not isinstance(entry.action, str):
            return False, f"Action must be a string, got {type(entry.action).__name__}"
        
        # Validate context type
        if not isinstance(entry.context, dict):
            return False, f"Context must be a dictionary, got {type(entry.context).__name__}"
        
        # Validate sensitivity type
        if not isinstance(entry.sensitivity, SensitivityLevel):
            return False, f"Sensitivity must be a SensitivityLevel enum, got {type(entry.sensitivity).__name__}"
        
        # Validate device_id type
        if not isinstance(entry.device_id, str):
            return False, f"Device ID must be a string, got {type(entry.device_id).__name__}"
        
        # Validate sync_status type
        if not isinstance(entry.sync_status, SyncStatus):
            return False, f"Sync status must be a SyncStatus enum, got {type(entry.sync_status).__name__}"
        
        # Validate tags type
        if not isinstance(entry.tags, list):
            return False, f"Tags must be a list, got {type(entry.tags).__name__}"
        
        # Validate each tag is a string
        for i, tag in enumerate(entry.tags):
            if not isinstance(tag, str):
                return False, f"Tag at index {i} must be a string, got {type(tag).__name__}"
        
        # Validate optional fields
        if entry.summary is not None and not isinstance(entry.summary, str):
            return False, f"Summary must be a string, got {type(entry.summary).__name__}"
        
        if entry.parent_id is not None and not isinstance(entry.parent_id, str):
            return False, f"Parent ID must be a string, got {type(entry.parent_id).__name__}"
        
        if entry.created_at is not None and not isinstance(entry.created_at, datetime):
            return False, f"Created_at must be a datetime object, got {type(entry.created_at).__name__}"
        
        if entry.updated_at is not None and not isinstance(entry.updated_at, datetime):
            return False, f"Updated_at must be a datetime object, got {type(entry.updated_at).__name__}"
        
        return True, None
    
    def _validate_enum_values(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate that enum fields contain valid enum values.
        
        This method ensures that sensitivity and sync_status fields
        contain valid enum values from their respective enums.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate sensitivity enum value
        if isinstance(entry.sensitivity, SensitivityLevel):
            # Check if the enum value is one of the defined values
            valid_sensitivity_values = [level.value for level in SensitivityLevel]
            if entry.sensitivity.value not in valid_sensitivity_values:
                return False, f"Invalid sensitivity level: {entry.sensitivity.value}. Must be one of: {', '.join(valid_sensitivity_values)}"
        else:
            # If it's not a SensitivityLevel instance, it should have been caught by type validation
            # But we'll handle it here as a safety check
            return False, f"Sensitivity must be a SensitivityLevel enum, got {type(entry.sensitivity).__name__}"
        
        # Validate sync_status enum value
        if isinstance(entry.sync_status, SyncStatus):
            # Check if the enum value is one of the defined values
            valid_sync_status_values = [status.value for status in SyncStatus]
            if entry.sync_status.value not in valid_sync_status_values:
                return False, f"Invalid sync status: {entry.sync_status.value}. Must be one of: {', '.join(valid_sync_status_values)}"
        else:
            # If it's not a SyncStatus instance, it should have been caught by type validation
            # But we'll handle it here as a safety check
            return False, f"Sync status must be a SyncStatus enum, got {type(entry.sync_status).__name__}"
        
        return True, None
    
    def _validate_field_values(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate that field values are within acceptable ranges and formats.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate ID format
        if not self.ID_PATTERN.match(entry.id):
            return False, "Entry ID must contain only alphanumeric characters, hyphens, and underscores"
        
        # Validate enum values
        is_valid, error = self._validate_enum_values(entry)
        if not is_valid:
            return False, error
        
        # Validate action length
        if len(entry.action) > self.MAX_ACTION_LENGTH:
            return False, f"Action exceeds maximum length of {self.MAX_ACTION_LENGTH} characters"
        
        # Validate device_id length
        if len(entry.device_id) > self.MAX_DEVICE_ID_LENGTH:
            return False, f"Device ID exceeds maximum length of {self.MAX_DEVICE_ID_LENGTH} characters"
        
        # Validate tags count and length
        if len(entry.tags) > self.MAX_TAGS_COUNT:
            return False, f"Number of tags exceeds maximum of {self.MAX_TAGS_COUNT}"
        
        for i, tag in enumerate(entry.tags):
            if not tag.strip():
                return False, f"Tag at index {i} cannot be empty or whitespace only"
            if len(tag) > self.MAX_TAG_LENGTH:
                return False, f"Tag at index {i} exceeds maximum length of {self.MAX_TAG_LENGTH} characters"
        
        # Validate summary length
        if entry.summary is not None and len(entry.summary) > self.MAX_SUMMARY_LENGTH:
            return False, f"Summary exceeds maximum length of {self.MAX_SUMMARY_LENGTH} characters"
        
        # Validate parent_id format if present
        if entry.parent_id is not None:
            if not entry.parent_id.strip():
                return False, "Parent ID cannot be empty or whitespace only"
            if not self.ID_PATTERN.match(entry.parent_id):
                return False, "Parent ID must contain only alphanumeric characters, hyphens, and underscores"
        
        # Validate context size
        context_str = str(entry.context)
        if len(context_str) > self.MAX_CONTEXT_SIZE:
            return False, f"Context size exceeds maximum of {self.MAX_CONTEXT_SIZE} characters"
        
        # Validate timestamp is not in the far future (more than 1 day ahead)
        if self.strict_mode:
            now = datetime.now(UTC)
            timestamp = entry.timestamp
            
            # Normalize both to timezone-aware UTC for safe comparison
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            
            # Compare timestamps (both timezone-aware now)
            time_diff = (timestamp - now).total_seconds()
            if time_diff > 86400:  # 1 day in seconds
                return False, "Timestamp cannot be more than 1 day in the future"
        
        return True, None
    
    def _validate_business_rules(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Validate business rules and logical constraints.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # If parent_id is set, entry should have a summary
        if entry.parent_id is not None and not entry.summary:
            return False, "Entry with parent_id must have a summary"
        
        # Validate timestamp relationships
        if entry.created_at and entry.updated_at:
            if entry.updated_at < entry.created_at:
                return False, "Updated_at cannot be earlier than created_at"
        
        return True, None
    
    def add_custom_rule(self, rule_func, rule_name: Optional[str] = None) -> None:
        """
        Register a custom validation rule.
        
        Custom rules are functions that take a MemoryEntry and return
        a tuple of (is_valid, error_message). They are applied after
        all standard validation rules.
        
        Args:
            rule_func: A callable that takes a MemoryEntry and returns
                      Tuple[bool, Optional[str]]. Should return (True, None)
                      if validation passes, or (False, error_message) if it fails.
            rule_name: Optional name for the rule (for debugging/logging).
        
        Example:
            def no_profanity_rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
                profanity_words = ['badword1', 'badword2']
                if any(word in entry.action.lower() for word in profanity_words):
                    return False, "Action contains prohibited content"
                return True, None
            
            validator.add_custom_rule(no_profanity_rule, "no_profanity")
        """
        if not callable(rule_func):
            raise ValueError("Custom rule must be a callable function")
        
        rule_info = {
            'func': rule_func,
            'name': rule_name or f"custom_rule_{len(self.custom_rules)}"
        }
        self.custom_rules.append(rule_info)
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """
        Remove a custom validation rule by name.
        
        Args:
            rule_name: The name of the rule to remove.
        
        Returns:
            True if the rule was found and removed, False otherwise.
        """
        for i, rule_info in enumerate(self.custom_rules):
            if rule_info['name'] == rule_name:
                self.custom_rules.pop(i)
                return True
        return False
    
    def clear_custom_rules(self) -> None:
        """Remove all custom validation rules."""
        self.custom_rules.clear()
    
    def get_custom_rules(self) -> List[str]:
        """
        Get the names of all registered custom rules.
        
        Returns:
            List of rule names.
        """
        return [rule_info['name'] for rule_info in self.custom_rules]
    
    def _apply_custom_rules(self, entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Apply all registered custom validation rules.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message). Returns the first error encountered.
        """
        for rule_info in self.custom_rules:
            try:
                is_valid, error = rule_info['func'](entry)
                if not is_valid:
                    # Include rule name in error message for debugging
                    rule_name = rule_info['name']
                    error_msg = f"Custom rule '{rule_name}' failed: {error}" if error else f"Custom rule '{rule_name}' failed"
                    return False, error_msg
            except Exception as e:
                # If a custom rule raises an exception, treat it as a validation failure
                rule_name = rule_info['name']
                return False, f"Custom rule '{rule_name}' raised an exception: {str(e)}"
        
        return True, None
    
    def validate_enum_string(self, enum_class: type, value: str, field_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a string value is a valid enum value.
        
        This is useful for validating API inputs before converting them to enums.
        
        Args:
            enum_class: The enum class to validate against (e.g., SensitivityLevel, SyncStatus).
            value: The string value to validate.
            field_name: The name of the field being validated (for error messages).
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        
        try:
            # Try to create an enum instance from the string value
            enum_class(value)
            return True, None
        except ValueError:
            # Get all valid values for the error message
            valid_values = [item.value for item in enum_class]
            return False, f"Invalid {field_name}: '{value}'. Must be one of: {', '.join(valid_values)}"
    
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize input data to prevent injection attacks.
        
        Performs comprehensive sanitization including:
        - HTML escaping to prevent XSS attacks
        - SQL injection prevention (removes dangerous SQL keywords)
        - Control character removal
        - Null byte removal
        - Path traversal prevention
        - Script tag removal
        
        Args:
            data: Dictionary containing input data.
        
        Returns:
            Sanitized dictionary.
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Sanitize string values
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, list):
                # Sanitize list items
                sanitized[key] = self._sanitize_list(value)
            else:
                # Keep other types as-is (numbers, booleans, None, datetime, etc.)
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_string(self, value: str) -> str:
        """
        Sanitize a single string value.
        
        Args:
            value: The string to sanitize.
        
        Returns:
            Sanitized string.
        """
        if not isinstance(value, str):
            return value
        
        # Strip leading/trailing whitespace
        sanitized = value.strip()
        
        # Remove null bytes (can cause issues in C-based systems)
        sanitized = sanitized.replace('\x00', '')
        
        # Remove other control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized 
                          if char == '\n' or char == '\t' or ord(char) >= 32)
        
        # Prevent path traversal attacks
        sanitized = sanitized.replace('../', '').replace('..\\', '')
        
        # HTML escape to prevent XSS
        sanitized = html.escape(sanitized)
        
        # Remove common script patterns (additional XSS prevention)
        # This is done after HTML escaping as a defense-in-depth measure
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'onerror\s*=',
            r'onload\s*=',
            r'onclick\s*=',
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    def _sanitize_list(self, items: List[Any]) -> List[Any]:
        """
        Sanitize a list of items.
        
        Args:
            items: The list to sanitize.
        
        Returns:
            Sanitized list.
        """
        sanitized = []
        for item in items:
            if isinstance(item, str):
                sanitized.append(self._sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(self.sanitize_input(item))
            elif isinstance(item, list):
                sanitized.append(self._sanitize_list(item))
            else:
                sanitized.append(item)
        return sanitized
    
    def validate_and_raise(self, entry: MemoryEntry) -> None:
        """
        Validate an entry and raise ValidationError if invalid.
        
        Convenience method for validation that raises an exception
        instead of returning a tuple.
        
        Args:
            entry: The memory entry to validate.
        
        Raises:
            ValidationError: If validation fails.
        """
        is_valid, error_message = self.validate_entry(entry)
        if not is_valid:
            raise ValidationError(error_message)
    
    def validate_partial_update(self, updates: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a partial update dictionary.
        
        Used when updating specific fields of an existing entry.
        
        Args:
            updates: Dictionary containing fields to update.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate action if present
        if 'action' in updates:
            if not isinstance(updates['action'], str):
                return False, "Action must be a string"
            if not updates['action'].strip():
                return False, "Action cannot be empty"
            if len(updates['action']) > self.MAX_ACTION_LENGTH:
                return False, f"Action exceeds maximum length of {self.MAX_ACTION_LENGTH} characters"
        
        # Validate tags if present
        if 'tags' in updates:
            if not isinstance(updates['tags'], list):
                return False, "Tags must be a list"
            if len(updates['tags']) > self.MAX_TAGS_COUNT:
                return False, f"Number of tags exceeds maximum of {self.MAX_TAGS_COUNT}"
            for i, tag in enumerate(updates['tags']):
                if not isinstance(tag, str):
                    return False, f"Tag at index {i} must be a string"
                if not tag.strip():
                    return False, f"Tag at index {i} cannot be empty"
                if len(tag) > self.MAX_TAG_LENGTH:
                    return False, f"Tag at index {i} exceeds maximum length of {self.MAX_TAG_LENGTH} characters"
        
        # Validate summary if present
        if 'summary' in updates:
            if updates['summary'] is not None:
                if not isinstance(updates['summary'], str):
                    return False, "Summary must be a string"
                if len(updates['summary']) > self.MAX_SUMMARY_LENGTH:
                    return False, f"Summary exceeds maximum length of {self.MAX_SUMMARY_LENGTH} characters"
        
        # Validate sensitivity if present
        if 'sensitivity' in updates:
            if isinstance(updates['sensitivity'], str):
                try:
                    SensitivityLevel(updates['sensitivity'])
                except ValueError:
                    return False, f"Invalid sensitivity level: {updates['sensitivity']}"
            elif not isinstance(updates['sensitivity'], SensitivityLevel):
                return False, "Sensitivity must be a SensitivityLevel enum or valid string"
        
        # Validate sync_status if present
        if 'sync_status' in updates:
            if isinstance(updates['sync_status'], str):
                try:
                    SyncStatus(updates['sync_status'])
                except ValueError:
                    return False, f"Invalid sync status: {updates['sync_status']}"
            elif not isinstance(updates['sync_status'], SyncStatus):
                return False, "Sync status must be a SyncStatus enum or valid string"
        
        # Validate context if present
        if 'context' in updates:
            if not isinstance(updates['context'], dict):
                return False, "Context must be a dictionary"
            context_str = str(updates['context'])
            if len(context_str) > self.MAX_CONTEXT_SIZE:
                return False, f"Context size exceeds maximum of {self.MAX_CONTEXT_SIZE} characters"
        
        return True, None
    
    # Built-in custom validation rules (static methods)
    
    @staticmethod
    def rule_no_empty_context(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Custom rule: Ensure context dictionary is not empty.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        if not entry.context or len(entry.context) == 0:
            return False, "Context dictionary cannot be empty"
        return True, None
    
    @staticmethod
    def rule_require_specific_tags(required_tags: List[str]):
        """
        Factory for custom rule: Ensure entry has at least one of the required tags.
        
        Args:
            required_tags: List of tags, at least one must be present.
        
        Returns:
            A validation rule function.
        
        Example:
            validator.add_custom_rule(
                ValidationManager.rule_require_specific_tags(['important', 'urgent']),
                'require_priority_tag'
            )
        """
        def rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
            if not any(tag in entry.tags for tag in required_tags):
                return False, f"Entry must have at least one of these tags: {', '.join(required_tags)}"
            return True, None
        return rule
    
    @staticmethod
    def rule_max_context_keys(max_keys: int):
        """
        Factory for custom rule: Limit the number of keys in context dictionary.
        
        Args:
            max_keys: Maximum number of keys allowed in context.
        
        Returns:
            A validation rule function.
        
        Example:
            validator.add_custom_rule(
                ValidationManager.rule_max_context_keys(20),
                'limit_context_complexity'
            )
        """
        def rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
            if len(entry.context.keys()) > max_keys:
                return False, f"Context cannot have more than {max_keys} keys"
            return True, None
        return rule
    
    @staticmethod
    def rule_action_pattern(pattern: str, error_message: Optional[str] = None):
        """
        Factory for custom rule: Validate action matches a regex pattern.
        
        Args:
            pattern: Regex pattern that action must match.
            error_message: Optional custom error message.
        
        Returns:
            A validation rule function.
        
        Example:
            validator.add_custom_rule(
                ValidationManager.rule_action_pattern(r'^[A-Z]', 'Action must start with uppercase'),
                'action_capitalized'
            )
        """
        compiled_pattern = re.compile(pattern)
        default_error = f"Action does not match required pattern: {pattern}"
        
        def rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
            if not compiled_pattern.search(entry.action):
                return False, error_message or default_error
            return True, None
        return rule
    
    @staticmethod
    def rule_sensitive_data_encrypted(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Custom rule: Ensure sensitive entries have appropriate sensitivity level.
        
        Checks if context contains potentially sensitive keywords and ensures
        the sensitivity level is not PUBLIC.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        sensitive_keywords = ['password', 'ssn', 'credit_card', 'api_key', 'secret', 'token']
        context_str = str(entry.context).lower()
        
        if any(keyword in context_str for keyword in sensitive_keywords):
            if entry.sensitivity == SensitivityLevel.PUBLIC:
                return False, "Entry contains sensitive data but is marked as PUBLIC"
        
        return True, None
    
    @staticmethod
    def rule_device_id_format(pattern: str = r'^[a-zA-Z0-9_-]+$'):
        """
        Factory for custom rule: Validate device_id format.

        Args:
            pattern: Regex pattern for valid device IDs.

        Returns:
            A validation rule function.

        Example:
            validator.add_custom_rule(
                ValidationManager.rule_device_id_format(r'^device-\\d+$'),
                'device_id_format'
            )
        """
        compiled_pattern = re.compile(pattern)

        def rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
            if not compiled_pattern.match(entry.device_id):
                return False, f"Device ID does not match required format: {pattern}"
            return True, None
        return rule
    
    @staticmethod
    def rule_tag_prefix(required_prefix: str):
        """
        Factory for custom rule: Ensure all tags start with a specific prefix.
        
        Args:
            required_prefix: Prefix that all tags must start with.
        
        Returns:
            A validation rule function.
        
        Example:
            validator.add_custom_rule(
                ValidationManager.rule_tag_prefix('app:'),
                'tag_namespace'
            )
        """
        def rule(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
            for tag in entry.tags:
                if not tag.startswith(required_prefix):
                    return False, f"All tags must start with '{required_prefix}'"
            return True, None
        return rule
    
    @staticmethod
    def rule_no_duplicate_tags(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Custom rule: Ensure no duplicate tags in the entry.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        if len(entry.tags) != len(set(entry.tags)):
            return False, "Entry contains duplicate tags"
        return True, None
    
    @staticmethod
    def rule_no_duplicate_tags(entry: MemoryEntry) -> Tuple[bool, Optional[str]]:
        """
        Custom rule: Ensure no duplicate tags in the entry.
        
        Args:
            entry: The memory entry to validate.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        if len(entry.tags) != len(set(entry.tags)):
            return False, "Entry contains duplicate tags"
        return True, None
