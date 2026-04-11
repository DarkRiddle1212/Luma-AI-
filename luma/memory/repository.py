"""
Memory Module - Repository Layer

This module handles all direct database operations for memory entries.
Repository methods do NOT call commit() - session lifecycle is controlled by the caller.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from luma.memory.models import Memory


class MemoryRepository:
    """
    Data access layer for memory operations.
    
    Handles CRUD operations without managing transaction lifecycle.
    The caller (typically via dependency injection) controls commits and rollbacks.
    """
    
    def __init__(self, session: Session):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session
    
    def create(self, content: str, metadata: dict) -> Memory:
        """
        Create a new memory entry.
        
        Does NOT commit - caller controls transaction.
        
        Args:
            content: Memory content
            metadata: Additional metadata dictionary
        
        Returns:
            Memory: Created memory instance
        """
        memory = Memory(content=content, metadata_=metadata)
        self.session.add(memory)
        self.session.flush()  # Flush to get ID without committing
        return memory
    
    def get_by_id(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            Memory instance if found, None otherwise
        """
        return self.session.query(Memory).filter(Memory.id == memory_id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[Memory]:
        """
        Retrieve all memories with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of Memory instances
        """
        return (
            self.session.query(Memory)
            .order_by(Memory.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update(self, memory_id: int, content: str, metadata: dict) -> Optional[Memory]:
        """
        Update an existing memory.
        
        Does NOT commit - caller controls transaction.
        
        Args:
            memory_id: Memory identifier
            content: New content
            metadata: New metadata
        
        Returns:
            Updated Memory instance if found, None otherwise
        """
        memory = self.get_by_id(memory_id)
        if memory:
            memory.content = content
            memory.metadata_ = metadata
            self.session.flush()
        return memory
    
    def delete(self, memory_id: int) -> bool:
        """
        Delete a memory by ID.
        
        Does NOT commit - caller controls transaction.
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            True if deleted, False if not found
        """
        memory = self.get_by_id(memory_id)
        if memory:
            self.session.delete(memory)
            self.session.flush()
            return True
        return False
