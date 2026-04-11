"""
Memory Module - Service Layer

This module implements business logic for memory operations.
Handles validation, orchestration, and delegates to repository for data access.
"""

from typing import Optional, List
from luma.memory.repository import MemoryRepository
from luma.memory.models import Memory


class ValidationError(Exception):
    """Raised when memory data validation fails."""
    pass


class NotFoundError(Exception):
    """Raised when a requested memory is not found."""
    pass


class MemoryService:
    """
    Business logic for memory management.
    
    Handles validation and orchestrates repository operations.
    All validation happens here, not in API routes or repository.
    """
    
    def __init__(self, repository: MemoryRepository):
        """
        Initialize service with repository dependency.
        
        Args:
            repository: MemoryRepository instance for data access
        """
        self.repository = repository
    
    def _validate_content(self, content: str) -> None:
        """
        Validate memory content.
        
        Args:
            content: Content to validate
        
        Raises:
            ValidationError: If content is invalid
        """
        if not content or not content.strip():
            raise ValidationError("Content cannot be empty or whitespace-only")
        
        if len(content.strip()) < 1:
            raise ValidationError("Content must have at least 1 character")
    
    def store_memory(self, content: str, metadata: dict) -> Memory:
        """
        Store a new memory with validation.
        
        Args:
            content: Memory content
            metadata: Additional metadata
        
        Returns:
            Memory: Created memory instance
        
        Raises:
            ValidationError: If content is invalid
        """
        self._validate_content(content)
        return self.repository.create(content, metadata)
    
    def retrieve_memory(self, memory_id: int) -> Memory:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            Memory: Retrieved memory instance
        
        Raises:
            NotFoundError: If memory doesn't exist
        """
        memory = self.repository.get_by_id(memory_id)
        if not memory:
            raise NotFoundError(f"Memory with id {memory_id} not found")
        return memory
    
    def list_memories(self, skip: int = 0, limit: int = 100) -> List[Memory]:
        """
        List all memories with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of Memory instances
        """
        return self.repository.get_all(skip, limit)
    
    def update_memory(self, memory_id: int, content: str, metadata: dict) -> Memory:
        """
        Update an existing memory with validation.
        
        Args:
            memory_id: Memory identifier
            content: New content
            metadata: New metadata
        
        Returns:
            Memory: Updated memory instance
        
        Raises:
            ValidationError: If content is invalid
            NotFoundError: If memory doesn't exist
        """
        self._validate_content(content)
        memory = self.repository.update(memory_id, content, metadata)
        if not memory:
            raise NotFoundError(f"Memory with id {memory_id} not found")
        return memory
    
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            True if deleted
        
        Raises:
            NotFoundError: If memory doesn't exist
        """
        success = self.repository.delete(memory_id)
        if not success:
            raise NotFoundError(f"Memory with id {memory_id} not found")
        return True
