"""
Memory Module - Data Models

This module defines SQLAlchemy ORM models for memory persistence.
Handles memory entries with proper indexing and timestamp management.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Index
from luma.database import Base


class Memory(Base):
    """
    Memory entry model for storing user actions and context.
    
    Attributes:
        id: Primary key (automatically indexed)
        content: The memory content (required)
        metadata: JSON field for flexible metadata storage
        created_at: Timestamp of creation (indexed for time-based queries)
        updated_at: Timestamp of last update (auto-updates on modification)
    """
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)  # Use callable to avoid mutable default, map to 'metadata' column
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional indexes for query performance
    __table_args__ = (
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, content='{self.content[:50]}...', created_at={self.created_at})>"
