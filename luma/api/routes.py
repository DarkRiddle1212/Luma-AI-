"""
API Layer - Route Definitions

This module defines REST API endpoints for the Luma system.
Routes delegate to service layer for business logic - no direct database access.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from luma.database import get_db
from luma.memory.repository import MemoryRepository
from luma.memory.service import MemoryService, ValidationError, NotFoundError
from luma.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter()


# Pydantic models for request/response
class MemoryCreate(BaseModel):
    """Request model for creating a memory."""
    content: str = Field(..., min_length=1, description="Memory content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class MemoryUpdate(BaseModel):
    """Request model for updating a memory."""
    content: str = Field(..., min_length=1, description="Updated memory content")
    metadata: dict = Field(default_factory=dict, description="Updated metadata")


class MemoryResponse(BaseModel):
    """Response model for memory data."""
    id: int
    content: str
    metadata: dict
    created_at: datetime
    updated_at: datetime
    
    model_config = {"from_attributes": True}
    
    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle metadata_ to metadata mapping."""
        if hasattr(obj, 'metadata_'):
            return cls(
                id=obj.id,
                content=obj.content,
                metadata=obj.metadata_,
                created_at=obj.created_at,
                updated_at=obj.updated_at
            )
        return super().model_validate(obj)


# Dependency injection for service layer
def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    """
    Dependency injection for MemoryService.
    
    Args:
        db: Database session from dependency injection
    
    Returns:
        MemoryService: Configured service instance
    """
    repository = MemoryRepository(db)
    return MemoryService(repository)


# API endpoints
@router.post("/memories", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory: MemoryCreate,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Create a new memory entry.
    
    Args:
        memory: Memory data from request body
        service: MemoryService from dependency injection
    
    Returns:
        Created memory with ID and timestamps
    
    Raises:
        HTTPException: 400 if validation fails
    """
    try:
        created_memory = service.store_memory(memory.content, memory.metadata)
        logger.info(f"Created memory {created_memory.id}")
        return MemoryResponse.model_validate(created_memory)
    except ValidationError as e:
        logger.warning(f"Validation error creating memory: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: int,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Retrieve a memory by ID.
    
    Args:
        memory_id: Memory identifier
        service: MemoryService from dependency injection
    
    Returns:
        Memory data
    
    Raises:
        HTTPException: 404 if memory not found
    """
    try:
        memory = service.retrieve_memory(memory_id)
        logger.info(f"Retrieved memory {memory_id}")
        return MemoryResponse.model_validate(memory)
    except NotFoundError as e:
        logger.warning(f"Memory not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/memories", response_model=List[MemoryResponse])
async def list_memories(
    skip: int = 0,
    limit: int = 100,
    service: MemoryService = Depends(get_memory_service)
):
    """
    List all memories with pagination.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        service: MemoryService from dependency injection
    
    Returns:
        List of memories
    """
    memories = service.list_memories(skip, limit)
    logger.info(f"Listed {len(memories)} memories (skip={skip}, limit={limit})")
    return [MemoryResponse.model_validate(m) for m in memories]


@router.put("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: int,
    memory: MemoryUpdate,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Update an existing memory.
    
    Args:
        memory_id: Memory identifier
        memory: Updated memory data
        service: MemoryService from dependency injection
    
    Returns:
        Updated memory data
    
    Raises:
        HTTPException: 400 if validation fails, 404 if not found
    """
    try:
        updated_memory = service.update_memory(memory_id, memory.content, memory.metadata)
        logger.info(f"Updated memory {memory_id}")
        return MemoryResponse.model_validate(updated_memory)
    except ValidationError as e:
        logger.warning(f"Validation error updating memory {memory_id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except NotFoundError as e:
        logger.warning(f"Memory not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: int,
    service: MemoryService = Depends(get_memory_service)
):
    """
    Delete a memory.
    
    Args:
        memory_id: Memory identifier
        service: MemoryService from dependency injection
    
    Raises:
        HTTPException: 404 if memory not found
    """
    try:
        service.delete_memory(memory_id)
        logger.info(f"Deleted memory {memory_id}")
    except NotFoundError as e:
        logger.warning(f"Memory not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
