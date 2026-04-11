"""
Data models for Luma Memory Module.

This module defines the core data structures for memory entries,
including enums for sensitivity levels and sync status.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class MemoryType(Enum):
    """
    Types of memory entries that Luma can store.
    
    Note: This enum is maintained for backward compatibility with existing code.
    New implementations should use the action field in MemoryEntry.
    """
    ACTION = "action"
    CONTEXT = "context"
    CONVERSATION = "conversation"
    TASK = "task"
    SYSTEM = "system"


class SensitivityLevel(Enum):
    """
    Sensitivity level for memory entries.
    
    Determines how the data should be handled in terms of privacy and security.
    """
    PUBLIC = "public"
    PRIVATE = "private"
    SENSITIVE = "sensitive"


class SyncStatus(Enum):
    """
    Synchronization status for memory entries.
    
    Tracks whether an entry has been synced across devices.
    """
    PENDING = "pending"
    SYNCED = "synced"
    CONFLICT = "conflict"


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry in the Luma Memory Module.
    
    A memory entry captures user actions, context, and metadata with support
    for encryption, tagging, and cross-device synchronization.
    
    Attributes:
        id: Unique identifier for the entry
        timestamp: When the action occurred
        action: Description of the user action
        context: Dictionary containing contextual information
        sensitivity: Privacy level of the entry
        device_id: Identifier of the device that created the entry
        sync_status: Current synchronization status
        tags: List of tags for categorization and search
        summary: Optional summary for consolidated entries
        parent_id: Reference to parent entry if this is a summary
        created_at: When the entry was created in the system
        updated_at: When the entry was last updated
    """
    id: str
    timestamp: datetime
    action: str
    context: Dict[str, Any]
    sensitivity: SensitivityLevel
    device_id: str
    sync_status: SyncStatus
    tags: list[str]
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        now = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate the memory entry fields.
        
        Ensures all required fields are present and have valid values.
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        # Validate required string fields
        if not self.id or not self.id.strip():
            return False, "Entry ID cannot be empty"
        
        if not self.action or not self.action.strip():
            return False, "Action cannot be empty"
        
        if not self.device_id or not self.device_id.strip():
            return False, "Device ID cannot be empty"
        
        # Validate timestamp
        if not isinstance(self.timestamp, datetime):
            return False, "Timestamp must be a datetime object"
        
        # Validate context
        if not isinstance(self.context, dict):
            return False, "Context must be a dictionary"
        
        # Validate enums
        if not isinstance(self.sensitivity, SensitivityLevel):
            return False, "Sensitivity must be a SensitivityLevel enum"
        
        if not isinstance(self.sync_status, SyncStatus):
            return False, "Sync status must be a SyncStatus enum"
        
        # Validate tags
        if not isinstance(self.tags, list):
            return False, "Tags must be a list"
        
        if not all(isinstance(tag, str) for tag in self.tags):
            return False, "All tags must be strings"
        
        # Validate optional fields
        if self.summary is not None and not isinstance(self.summary, str):
            return False, "Summary must be a string"
        
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id.strip()):
            return False, "Parent ID must be a non-empty string if provided"
        
        return True, None
    
    def is_valid(self) -> bool:
        """
        Check if the memory entry is valid.
        
        Returns:
            True if valid, False otherwise.
        """
        valid, _ = self.validate()
        return valid

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory entry to dictionary for storage or serialization.
        
        Handles conversion of datetime objects to ISO format strings
        and enum values to their string representations.
        
        Returns:
            Dictionary representation of the memory entry.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "action": self.action,
            "context": self.context,
            "sensitivity": self.sensitivity.value,
            "device_id": self.device_id,
            "sync_status": self.sync_status.value,
            "tags": self.tags,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """
        Create a MemoryEntry from a dictionary.
        
        Handles conversion of ISO format strings to datetime objects
        and string values to enum instances.
        
        Args:
            data: Dictionary containing memory entry data.
        
        Returns:
            MemoryEntry instance.
        
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Create a copy to avoid modifying the original
        data = data.copy()
        
        # Convert datetime strings to datetime objects
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        # Convert enum strings to enum instances
        if isinstance(data.get("sensitivity"), str):
            data["sensitivity"] = SensitivityLevel(data["sensitivity"])
        
        if isinstance(data.get("sync_status"), str):
            data["sync_status"] = SyncStatus(data["sync_status"])
        
        # Filter to only include valid MemoryEntry fields
        valid_fields = {
            "id", "timestamp", "action", "context", "sensitivity",
            "device_id", "sync_status", "tags", "summary", "parent_id",
            "created_at", "updated_at"
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)


def create_memory_entry(
    action: str,
    context: Dict[str, Any],
    device_id: str,
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC,
    tags: Optional[list[str]] = None,
    entry_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> MemoryEntry:
    """
    Factory function to create a new MemoryEntry with sensible defaults.
    
    Args:
        action: Description of the user action
        context: Contextual information dictionary
        device_id: Identifier of the device creating the entry
        sensitivity: Privacy level (defaults to PUBLIC)
        tags: Optional list of tags
        entry_id: Optional custom ID (generates UUID if not provided)
        timestamp: Optional timestamp (uses current time if not provided)
    
    Returns:
        New MemoryEntry instance.
    """
    return MemoryEntry(
        id=entry_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(UTC),
        action=action,
        context=context,
        sensitivity=sensitivity,
        device_id=device_id,
        sync_status=SyncStatus.PENDING,
        tags=tags or [],
    )
