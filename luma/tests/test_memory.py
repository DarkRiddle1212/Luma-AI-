"""
Unit Tests - Memory Module

Tests for Memory model, MemoryRepository, and MemoryService.
Covers CRUD operations, validation logic, error handling, and pagination.
"""

import pytest
from datetime import datetime
from sqlalchemy.orm import Session

from luma.memory.models import Memory
from luma.memory.repository import MemoryRepository
from luma.memory.service import MemoryService, ValidationError, NotFoundError


# ============================================================================
# 16.1 Test Memory model creation and validation
# ============================================================================

class TestMemoryModel:
    """Tests for Memory SQLAlchemy model."""
    
    def test_memory_creation_with_required_fields(self, test_db: Session):
        """Test creating a memory with only required fields."""
        memory = Memory(content="Test content")
        test_db.add(memory)
        test_db.flush()
        
        assert memory.id is not None
        assert memory.content == "Test content"
        assert memory.metadata_ == {}
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)
    
    def test_memory_creation_with_all_fields(self, test_db: Session):
        """Test creating a memory with all fields including metadata."""
        metadata = {"source": "test", "priority": "high"}
        memory = Memory(content="Full test content", metadata_=metadata)
        test_db.add(memory)
        test_db.flush()
        
        assert memory.id is not None
        assert memory.content == "Full test content"
        assert memory.metadata_ == metadata
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)
    
    def test_memory_timestamps_auto_generated(self, test_db: Session):
        """Test that created_at and updated_at are automatically set."""
        memory = Memory(content="Timestamp test")
        test_db.add(memory)
        test_db.flush()
        
        assert memory.created_at is not None
        assert memory.updated_at is not None
        # Timestamps should be very close (within 1 second)
        time_diff = abs((memory.created_at - memory.updated_at).total_seconds())
        assert time_diff < 1.0
    
    def test_memory_metadata_defaults_to_empty_dict(self, test_db: Session):
        """Test that metadata defaults to empty dict when not provided."""
        memory = Memory(content="Default metadata test")
        test_db.add(memory)
        test_db.flush()
        
        assert memory.metadata_ == {}
        assert isinstance(memory.metadata_, dict)
    
    def test_memory_content_cannot_be_null(self, test_db: Session):
        """Test that content field is required (cannot be null)."""
        memory = Memory()
        test_db.add(memory)
        
        with pytest.raises(Exception):  # SQLAlchemy will raise IntegrityError
            test_db.flush()
        
        # Rollback to clean up the failed transaction
        test_db.rollback()
    
    def test_memory_repr(self, test_db: Session):
        """Test the string representation of Memory model."""
        memory = Memory(content="A" * 100)  # Long content to test truncation
        test_db.add(memory)
        test_db.flush()
        
        repr_str = repr(memory)
        assert "Memory" in repr_str
        assert str(memory.id) in repr_str
        assert "created_at" in repr_str


# ============================================================================
# 16.2 Test MemoryRepository CRUD operations
# ============================================================================

class TestMemoryRepository:
    """Tests for MemoryRepository data access layer."""
    
    def test_create_memory(self, memory_repository: MemoryRepository, test_db: Session):
        """Test creating a memory through repository."""
        content = "Test repository content"
        metadata = {"source": "test"}
        
        memory = memory_repository.create(content, metadata)
        
        assert memory.id is not None
        assert memory.content == content
        assert memory.metadata_ == metadata
        assert memory.created_at is not None
    
    def test_create_memory_does_not_commit(self, memory_repository: MemoryRepository, test_db: Session):
        """Test that create() flushes but does not commit."""
        memory = memory_repository.create("Test content", {})
        
        # Memory should have an ID (flushed)
        assert memory.id is not None
        
        # Rollback should remove the memory since it wasn't committed
        test_db.rollback()
        
        # Query should return None after rollback
        result = test_db.query(Memory).filter(Memory.id == memory.id).first()
        assert result is None
    
    def test_get_by_id_existing(self, memory_repository: MemoryRepository, test_db: Session):
        """Test retrieving an existing memory by ID."""
        # Create a memory
        memory = memory_repository.create("Test content", {})
        test_db.commit()
        
        # Retrieve it
        retrieved = memory_repository.get_by_id(memory.id)
        
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content
    
    def test_get_by_id_nonexistent(self, memory_repository: MemoryRepository):
        """Test retrieving a non-existent memory returns None."""
        result = memory_repository.get_by_id(99999)
        assert result is None
    
    def test_get_all_empty(self, memory_repository: MemoryRepository):
        """Test get_all returns empty list when no memories exist."""
        memories = memory_repository.get_all()
        assert memories == []
    
    def test_get_all_with_memories(self, memory_repository: MemoryRepository, test_db: Session):
        """Test get_all returns all memories."""
        # Create multiple memories
        memory1 = memory_repository.create("Content 1", {})
        memory2 = memory_repository.create("Content 2", {})
        memory3 = memory_repository.create("Content 3", {})
        test_db.commit()
        
        # Retrieve all
        memories = memory_repository.get_all()
        
        assert len(memories) == 3
        # Should be ordered by created_at desc (newest first)
        assert memories[0].id == memory3.id
        assert memories[1].id == memory2.id
        assert memories[2].id == memory1.id
    
    def test_get_all_with_pagination(self, memory_repository: MemoryRepository, test_db: Session):
        """Test get_all pagination with skip and limit."""
        # Create 5 memories
        for i in range(5):
            memory_repository.create(f"Content {i}", {})
        test_db.commit()
        
        # Get first 2
        page1 = memory_repository.get_all(skip=0, limit=2)
        assert len(page1) == 2
        
        # Get next 2
        page2 = memory_repository.get_all(skip=2, limit=2)
        assert len(page2) == 2
        
        # Ensure different memories
        assert page1[0].id != page2[0].id
    
    def test_update_existing_memory(self, memory_repository: MemoryRepository, test_db: Session):
        """Test updating an existing memory."""
        # Create a memory
        memory = memory_repository.create("Original content", {"key": "value"})
        test_db.commit()
        
        # Update it
        new_content = "Updated content"
        new_metadata = {"key": "new_value"}
        updated = memory_repository.update(memory.id, new_content, new_metadata)
        test_db.commit()
        
        assert updated is not None
        assert updated.id == memory.id
        assert updated.content == new_content
        assert updated.metadata_ == new_metadata
    
    def test_update_nonexistent_memory(self, memory_repository: MemoryRepository):
        """Test updating a non-existent memory returns None."""
        result = memory_repository.update(99999, "New content", {})
        assert result is None
    
    def test_update_does_not_commit(self, memory_repository: MemoryRepository, test_db: Session):
        """Test that update() flushes but does not commit."""
        # Create a memory
        memory = memory_repository.create("Original", {})
        test_db.commit()
        
        # Update it
        memory_repository.update(memory.id, "Updated", {})
        
        # Rollback should revert the update
        test_db.rollback()
        
        # Content should still be original
        retrieved = memory_repository.get_by_id(memory.id)
        assert retrieved.content == "Original"
    
    def test_delete_existing_memory(self, memory_repository: MemoryRepository, test_db: Session):
        """Test deleting an existing memory."""
        # Create a memory
        memory = memory_repository.create("To be deleted", {})
        test_db.commit()
        
        # Delete it
        result = memory_repository.delete(memory.id)
        test_db.commit()
        
        assert result is True
        
        # Verify it's gone
        retrieved = memory_repository.get_by_id(memory.id)
        assert retrieved is None
    
    def test_delete_nonexistent_memory(self, memory_repository: MemoryRepository):
        """Test deleting a non-existent memory returns False."""
        result = memory_repository.delete(99999)
        assert result is False
    
    def test_delete_does_not_commit(self, memory_repository: MemoryRepository, test_db: Session):
        """Test that delete() flushes but does not commit."""
        # Create a memory
        memory = memory_repository.create("To be deleted", {})
        test_db.commit()
        
        # Delete it
        memory_repository.delete(memory.id)
        
        # Rollback should restore the memory
        test_db.rollback()
        
        # Memory should still exist
        retrieved = memory_repository.get_by_id(memory.id)
        assert retrieved is not None


# ============================================================================
# 16.3 Test MemoryService validation logic
# ============================================================================

class TestMemoryServiceValidation:
    """Tests for MemoryService validation logic."""
    
    def test_store_memory_with_valid_content(self, memory_service: MemoryService, test_db: Session):
        """Test storing memory with valid content."""
        content = "Valid content"
        metadata = {"key": "value"}
        
        memory = memory_service.store_memory(content, metadata)
        test_db.commit()
        
        assert memory.id is not None
        assert memory.content == content
        assert memory.metadata_ == metadata
    
    def test_store_memory_with_empty_string_raises_error(self, memory_service: MemoryService):
        """Test that empty string content raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            memory_service.store_memory("", {})
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_store_memory_with_whitespace_only_raises_error(self, memory_service: MemoryService):
        """Test that whitespace-only content raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            memory_service.store_memory("   ", {})
        
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()
    
    def test_store_memory_with_tabs_and_newlines_raises_error(self, memory_service: MemoryService):
        """Test that tabs and newlines only raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            memory_service.store_memory("\t\n\r", {})
        
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()
    
    def test_store_memory_with_none_raises_error(self, memory_service: MemoryService):
        """Test that None content raises ValidationError."""
        with pytest.raises(ValidationError):
            memory_service.store_memory(None, {})
    
    def test_update_memory_with_valid_content(self, memory_service: MemoryService, test_db: Session):
        """Test updating memory with valid content."""
        # Create a memory
        memory = memory_service.store_memory("Original", {})
        test_db.commit()
        
        # Update with valid content
        updated = memory_service.update_memory(memory.id, "Updated content", {"new": "data"})
        test_db.commit()
        
        assert updated.content == "Updated content"
        assert updated.metadata_ == {"new": "data"}
    
    def test_update_memory_with_empty_string_raises_error(self, memory_service: MemoryService, test_db: Session):
        """Test that updating with empty string raises ValidationError."""
        # Create a memory
        memory = memory_service.store_memory("Original", {})
        test_db.commit()
        
        # Try to update with empty content
        with pytest.raises(ValidationError) as exc_info:
            memory_service.update_memory(memory.id, "", {})
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_update_memory_with_whitespace_only_raises_error(self, memory_service: MemoryService, test_db: Session):
        """Test that updating with whitespace-only raises ValidationError."""
        # Create a memory
        memory = memory_service.store_memory("Original", {})
        test_db.commit()
        
        # Try to update with whitespace
        with pytest.raises(ValidationError) as exc_info:
            memory_service.update_memory(memory.id, "   ", {})
        
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()
    
    def test_validate_content_with_single_character(self, memory_service: MemoryService, test_db: Session):
        """Test that single character content is valid."""
        memory = memory_service.store_memory("a", {})
        test_db.commit()
        
        assert memory.content == "a"
    
    def test_validate_content_with_leading_trailing_whitespace(self, memory_service: MemoryService, test_db: Session):
        """Test that content with leading/trailing whitespace is valid."""
        content = "  valid content  "
        memory = memory_service.store_memory(content, {})
        test_db.commit()
        
        # Content should be stored as-is (validation checks stripped version)
        assert memory.content == content


# ============================================================================
# 16.4 Test MemoryService error handling
# ============================================================================

class TestMemoryServiceErrorHandling:
    """Tests for MemoryService error handling."""
    
    def test_retrieve_memory_not_found_raises_error(self, memory_service: MemoryService):
        """Test that retrieving non-existent memory raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            memory_service.retrieve_memory(99999)
        
        assert "99999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    
    def test_retrieve_memory_success(self, memory_service: MemoryService, test_db: Session):
        """Test successful memory retrieval."""
        # Create a memory
        memory = memory_service.store_memory("Test content", {})
        test_db.commit()
        
        # Retrieve it
        retrieved = memory_service.retrieve_memory(memory.id)
        
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content
    
    def test_update_memory_not_found_raises_error(self, memory_service: MemoryService):
        """Test that updating non-existent memory raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            memory_service.update_memory(99999, "New content", {})
        
        assert "99999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    
    def test_update_memory_validation_before_not_found(self, memory_service: MemoryService):
        """Test that validation error is raised before NotFoundError."""
        # Try to update non-existent memory with invalid content
        # Validation should fail first
        with pytest.raises(ValidationError):
            memory_service.update_memory(99999, "", {})
    
    def test_delete_memory_not_found_raises_error(self, memory_service: MemoryService):
        """Test that deleting non-existent memory raises NotFoundError."""
        with pytest.raises(NotFoundError) as exc_info:
            memory_service.delete_memory(99999)
        
        assert "99999" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()
    
    def test_delete_memory_success(self, memory_service: MemoryService, test_db: Session):
        """Test successful memory deletion."""
        # Create a memory
        memory = memory_service.store_memory("To be deleted", {})
        test_db.commit()
        
        # Delete it
        result = memory_service.delete_memory(memory.id)
        test_db.commit()
        
        assert result is True
        
        # Verify it's gone
        with pytest.raises(NotFoundError):
            memory_service.retrieve_memory(memory.id)
    
    def test_list_memories_empty(self, memory_service: MemoryService):
        """Test listing memories when none exist."""
        memories = memory_service.list_memories()
        assert memories == []
    
    def test_list_memories_with_data(self, memory_service: MemoryService, test_db: Session):
        """Test listing memories when data exists."""
        # Create multiple memories
        memory1 = memory_service.store_memory("Content 1", {})
        memory2 = memory_service.store_memory("Content 2", {})
        test_db.commit()
        
        # List them
        memories = memory_service.list_memories()
        
        assert len(memories) == 2
        # Should be ordered by created_at desc
        assert memories[0].id == memory2.id
        assert memories[1].id == memory1.id


# ============================================================================
# 16.5 Test pagination in list operations
# ============================================================================

class TestMemoryPagination:
    """Tests for pagination in memory list operations."""
    
    def test_list_memories_default_pagination(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories with default pagination parameters."""
        # Create 3 memories
        for i in range(3):
            memory_service.store_memory(f"Content {i}", {})
        test_db.commit()
        
        # List with defaults (skip=0, limit=100)
        memories = memory_service.list_memories()
        
        assert len(memories) == 3
    
    def test_list_memories_with_limit(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories with custom limit."""
        # Create 5 memories
        for i in range(5):
            memory_service.store_memory(f"Content {i}", {})
        test_db.commit()
        
        # List with limit=3
        memories = memory_service.list_memories(skip=0, limit=3)
        
        assert len(memories) == 3
    
    def test_list_memories_with_skip(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories with skip parameter."""
        # Create 5 memories
        created_memories = []
        for i in range(5):
            mem = memory_service.store_memory(f"Content {i}", {})
            created_memories.append(mem)
        test_db.commit()
        
        # Skip first 2
        memories = memory_service.list_memories(skip=2, limit=100)
        
        assert len(memories) == 3
        # Should get the 3rd, 4th, and 5th memories (in reverse order)
        assert memories[0].id == created_memories[2].id
    
    def test_list_memories_with_skip_and_limit(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories with both skip and limit."""
        # Create 10 memories
        created_memories = []
        for i in range(10):
            mem = memory_service.store_memory(f"Content {i}", {})
            created_memories.append(mem)
        test_db.commit()
        
        # Get page 2 (skip 3, limit 3)
        page2 = memory_service.list_memories(skip=3, limit=3)
        
        assert len(page2) == 3
        # Should get memories 7, 6, 5 (indices 6, 5, 4 in created order)
        assert page2[0].id == created_memories[6].id
        assert page2[1].id == created_memories[5].id
        assert page2[2].id == created_memories[4].id
    
    def test_list_memories_skip_beyond_total(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories when skip exceeds total count."""
        # Create 3 memories
        for i in range(3):
            memory_service.store_memory(f"Content {i}", {})
        test_db.commit()
        
        # Skip beyond total
        memories = memory_service.list_memories(skip=10, limit=100)
        
        assert len(memories) == 0
    
    def test_list_memories_ordered_by_created_at_desc(self, memory_service: MemoryService, test_db: Session):
        """Test that list_memories returns results ordered by created_at descending."""
        # Create memories with slight delay to ensure different timestamps
        import time
        memories_created = []
        for i in range(3):
            mem = memory_service.store_memory(f"Content {i}", {})
            memories_created.append(mem)
            test_db.flush()
            time.sleep(0.01)  # Small delay to ensure different timestamps
        test_db.commit()
        
        # List all
        memories = memory_service.list_memories()
        
        # Should be in reverse order (newest first)
        assert memories[0].id == memories_created[2].id
        assert memories[1].id == memories_created[1].id
        assert memories[2].id == memories_created[0].id
    
    def test_list_memories_pagination_consistency(self, memory_service: MemoryService, test_db: Session):
        """Test that pagination returns consistent, non-overlapping results."""
        # Create 10 memories
        for i in range(10):
            memory_service.store_memory(f"Content {i}", {})
        test_db.commit()
        
        # Get two pages
        page1 = memory_service.list_memories(skip=0, limit=5)
        page2 = memory_service.list_memories(skip=5, limit=5)
        
        assert len(page1) == 5
        assert len(page2) == 5
        
        # Ensure no overlap
        page1_ids = {m.id for m in page1}
        page2_ids = {m.id for m in page2}
        assert len(page1_ids.intersection(page2_ids)) == 0
    
    def test_list_memories_limit_zero(self, memory_service: MemoryService, test_db: Session):
        """Test list_memories with limit=0."""
        # Create 3 memories
        for i in range(3):
            memory_service.store_memory(f"Content {i}", {})
        test_db.commit()
        
        # List with limit=0
        memories = memory_service.list_memories(skip=0, limit=0)
        
        assert len(memories) == 0 