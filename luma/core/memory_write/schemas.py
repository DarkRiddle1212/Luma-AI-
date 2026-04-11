"""Data models for the Memory Write Engine.

This module defines the core data structures used throughout the memory write
pipeline: candidate extraction, importance scoring, and memory persistence.
"""

from dataclasses import dataclass
from typing import List, Literal
from datetime import datetime


# Type alias for memory classification
MemoryType = Literal["project_goal", "user_preference", "fact", "statement"]


@dataclass
class MemoryCandidate:
    """A candidate memory before importance scoring.
    
    Represents a potential memory extracted from an interaction that has been
    identified and classified but not yet evaluated for importance.
    
    Attributes:
        text: The memory content text
        type: Classification of memory type
    """
    text: str
    type: MemoryType
    
    def __post_init__(self):
        """Validate memory candidate fields."""
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")
        
        valid_types = ["project_goal", "user_preference", "fact", "statement"]
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")


@dataclass
class ScoredMemory:
    """A candidate memory with importance score.
    
    Represents a memory that has been evaluated for importance and assigned
    a score. Only memories above the configured threshold proceed to storage.
    
    Attributes:
        text: The memory content text
        type: Classification of memory type
        importance: Importance score (0.0 to 1.0)
    """
    text: str
    type: MemoryType
    importance: float
    
    def __post_init__(self):
        """Validate scored memory fields."""
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")
        
        valid_types = ["project_goal", "user_preference", "fact", "statement"]
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")


@dataclass
class StoredMemory:
    """A memory that has been persisted to storage.
    
    Represents a memory that has been successfully stored in the memory store,
    including metadata about the storage operation.
    
    Attributes:
        memory_id: Unique identifier from Memory Store
        text: The memory content text
        type: Classification of memory type
        importance: Importance score (0.0 to 1.0)
        created_at: ISO 8601 formatted timestamp
        is_update: True if existing memory was updated, False if newly created
    """
    memory_id: str
    text: str
    type: MemoryType
    importance: float
    created_at: str
    is_update: bool = False
    
    def __post_init__(self):
        """Validate stored memory fields."""
        if not self.memory_id or not self.memory_id.strip():
            raise ValueError("memory_id must be non-empty")
        
        if not self.text or not self.text.strip():
            raise ValueError("text must be non-empty")
        
        valid_types = ["project_goal", "user_preference", "fact", "statement"]
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")
        
        # Validate ISO 8601 timestamp format
        try:
            datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            raise ValueError("created_at must be valid ISO 8601 timestamp string")


@dataclass
class MemoryWriteResult:
    """Result of memory write operation.
    
    Contains the outcome of processing an interaction through the memory write
    pipeline, including which memories were stored and which were filtered out.
    
    Attributes:
        stored_memories: List of memories that were persisted (above threshold)
        ignored_memories: List of candidates that were filtered out (below threshold)
    """
    stored_memories: List[StoredMemory]
    ignored_memories: List[MemoryCandidate]
    
    def __post_init__(self):
        """Validate memory write result fields."""
        # Ensure lists are not None
        if self.stored_memories is None:
            self.stored_memories = []
        if self.ignored_memories is None:
            self.ignored_memories = []
        
        # Validate list contents
        for memory in self.stored_memories:
            if not isinstance(memory, StoredMemory):
                raise ValueError("stored_memories must contain only StoredMemory objects")
        
        for candidate in self.ignored_memories:
            if not isinstance(candidate, MemoryCandidate):
                raise ValueError("ignored_memories must contain only MemoryCandidate objects")
