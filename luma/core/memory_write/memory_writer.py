"""Memory persistence component for the Memory Write Engine.

This module handles the storage of scored memories to the memory store with
deduplication logic to prevent storing similar memories multiple times.
"""

import uuid
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, List
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters
from .schemas import ScoredMemory, StoredMemory


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher.
    
    Uses Python's difflib.SequenceMatcher to compute similarity ratio
    between two text strings. The texts are normalized (lowercased and
    stripped) before comparison.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0, where 1.0 means identical
    """
    # Normalize texts (lowercase, strip whitespace)
    norm1 = text1.lower().strip()
    norm2 = text2.lower().strip()
    
    # Calculate similarity ratio
    matcher = SequenceMatcher(None, norm1, norm2)
    return matcher.ratio()


class MemoryWriter:
    """Persists memories to storage with deduplication.
    
    The MemoryWriter handles the final stage of the memory write pipeline,
    storing validated memories to the memory store. It checks for similar
    existing memories and updates them instead of creating duplicates when
    similarity exceeds the threshold (0.9).
    
    Attributes:
        memory_store: MemoryInterface implementation for storage operations
        similarity_threshold: Threshold for deduplication (default 0.9)
    """
    
    def __init__(
        self,
        memory_store: MemoryInterface,
        similarity_threshold: float = 0.9
    ):
        """Initialize with memory store interface.
        
        Args:
            memory_store: MemoryInterface implementation for storage
            similarity_threshold: Threshold for deduplication (0.0 to 1.0)
            
        Raises:
            ValueError: If similarity_threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        self.memory_store = memory_store
        self.similarity_threshold = similarity_threshold
    
    def store_memory(self, scored_memory: ScoredMemory) -> StoredMemory:
        """Store a scored memory with deduplication.
        
        Checks for similar existing memories using text similarity. If a
        similar memory exists (similarity > threshold), updates the existing
        memory with the higher importance score. Otherwise, creates a new
        memory entry.
        
        Args:
            scored_memory: ScoredMemory to persist
            
        Returns:
            StoredMemory with storage metadata
            
        Raises:
            ValueError: If scored_memory is None or has invalid fields
            MemoryStorageError: If storage operation fails
        """
        if scored_memory is None:
            raise ValueError("scored_memory cannot be None")
        
        # Check for similar existing memories
        similar_memory = self._find_similar_memory(scored_memory.text)
        
        if similar_memory:
            # Update existing memory
            return self._update_existing_memory(similar_memory, scored_memory)
        else:
            # Create new memory
            return self._create_new_memory(scored_memory)
    
    def _find_similar_memory(self, text: str) -> Optional[MemoryEntry]:
        """Find similar existing memory using text similarity.
        
        Queries the memory store for memories with similar content and
        calculates text similarity. Returns the first memory that exceeds
        the similarity threshold.
        
        Args:
            text: Memory text to search for
            
        Returns:
            MemoryEntry if similar memory found, None otherwise
        """
        try:
            # Query for potentially similar memories
            params: QueryParameters = {
                "query": text,
                "category": "conversation_memory",
                "limit": 5
            }
            
            result = self.memory_store.retrieve(params=params)
            similar_memories = result["memories"]
            
            # Calculate similarity and check threshold
            for existing in similar_memories:
                similarity = calculate_similarity(text, existing["content"])
                if similarity > self.similarity_threshold:
                    return existing
            
            return None
        except Exception:
            # Gracefully handle similarity calculation failures
            # Treat as no similarity found (create new memory)
            return None
    
    def _update_existing_memory(
        self,
        existing: MemoryEntry,
        scored_memory: ScoredMemory
    ) -> StoredMemory:
        """Update an existing memory with new information.
        
        Preserves the original timestamp and updates the updated_at field.
        Uses the maximum importance score between existing and new memory.
        
        Args:
            existing: Existing MemoryEntry to update
            scored_memory: New ScoredMemory with updated information
            
        Returns:
            StoredMemory with is_update=True
        """
        # Get existing importance score (default to 0.0 if not present)
        existing_importance = existing["metadata"].get("importance", 0.0)
        
        # Use maximum importance score
        max_importance = max(existing_importance, scored_memory.importance)
        
        # Update metadata with new importance and updated_at timestamp
        updated_metadata = {
            **existing["metadata"],
            "importance": max_importance,
            "updated_at": datetime.now().isoformat()
        }
        
        # Store updated memory (using same ID updates the existing entry)
        self.memory_store.store(
            content=scored_memory.text,
            metadata={
                "id": existing["id"],
                "importance": max_importance,
                "category": "conversation_memory",
                "tags": [scored_memory.type],
                "updated_at": datetime.now().isoformat()
            }
        )
        
        return StoredMemory(
            memory_id=existing["id"],
            text=scored_memory.text,
            type=scored_memory.type,
            importance=max_importance,
            created_at=existing["timestamp"],  # Preserve original timestamp
            is_update=True
        )
    
    def _create_new_memory(self, scored_memory: ScoredMemory) -> StoredMemory:
        """Create a new memory entry.
        
        Generates a unique ID and stores the memory with current timestamp.
        
        Args:
            scored_memory: ScoredMemory to store
            
        Returns:
            StoredMemory with is_update=False
        """
        # Generate unique ID
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Store memory
        stored_id = self.memory_store.store(
            content=scored_memory.text,
            metadata={
                "importance": scored_memory.importance,
                "category": "conversation_memory",
                "tags": [scored_memory.type],
                "timestamp": timestamp
            }
        )
        
        # Use the ID returned by the store if available, otherwise use generated ID
        final_id = stored_id if stored_id else memory_id
        
        return StoredMemory(
            memory_id=final_id,
            text=scored_memory.text,
            type=scored_memory.type,
            importance=scored_memory.importance,
            created_at=timestamp,
            is_update=False
        )
