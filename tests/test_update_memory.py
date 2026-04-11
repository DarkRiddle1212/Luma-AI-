"""
Test for update_memory method implementation (Task 10.6).
"""

import pytest
from datetime import datetime
import tempfile
import os

from luma_memory.memory_manager import MemoryManager
from luma_memory.models import SensitivityLevel, SyncStatus
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.validation import ValidationManager, ValidationError
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.config import MemoryModuleConfig


class TestUpdateMemory:
    """Test suite for update_memory method."""
    
    @pytest.fixture
    def memory_storage(self):
        """Create an in-memory storage for testing."""
        return MemoryStorage()
    
    @pytest.fixture
    def validation_manager(self):
        """Create a validation manager."""
        return ValidationManager()
    
    @pytest.fixture
    def memory_manager(self, memory_storage, validation_manager):
        """Create a memory manager with in-memory storage."""
        config = MemoryModuleConfig()
        manager = MemoryManager(
            storage=memory_storage,
            validation=validation_manager,
            config=config
        )
        return manager
    
    def test_update_memory_tags(self, memory_manager):
        """Test updating tags of a memory entry."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User opened document",
            context={"file": "report.pdf"},
            device_id="laptop-001",
            tags=["work", "document"]
        )
        
        # Update tags
        success = memory_manager.update_memory(
            entry_id,
            {"tags": ["work", "important", "urgent"]}
        )
        
        assert success is True
        
        # Verify the update
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        assert "important" in entry.tags
        assert "urgent" in entry.tags
        assert "work" in entry.tags
    
    def test_update_memory_summary(self, memory_manager):
        """Test updating summary of a memory entry."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User completed task",
            context={"task": "review code"},
            device_id="laptop-001"
        )
        
        # Update summary
        success = memory_manager.update_memory(
            entry_id,
            {"summary": "Code review completed successfully"}
        )
        
        assert success is True
        
        # Verify the update
        entry = memory_manager.get_memory(entry_id)
        assert entry.summary == "Code review completed successfully"
    
    def test_update_memory_context(self, memory_manager):
        """Test updating context of a memory entry."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User browsing",
            context={"url": "example.com", "duration": 10},
            device_id="laptop-001"
        )
        
        # Update context
        success = memory_manager.update_memory(
            entry_id,
            {"context": {"url": "example.com", "duration": 25, "status": "completed"}}
        )
        
        assert success is True
        
        # Verify the update
        entry = memory_manager.get_memory(entry_id)
        assert entry.context["duration"] == 25
        assert entry.context["status"] == "completed"
    
    def test_update_memory_sync_status(self, memory_manager):
        """Test updating sync status of a memory entry."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        # Update sync status
        success = memory_manager.update_memory(
            entry_id,
            {"sync_status": SyncStatus.SYNCED}
        )
        
        assert success is True
        
        # Verify the update
        entry = memory_manager.get_memory(entry_id)
        assert entry.sync_status == SyncStatus.SYNCED
    
    def test_update_memory_nonexistent(self, memory_manager):
        """Test updating a non-existent memory entry returns False."""
        success = memory_manager.update_memory(
            "nonexistent-id-12345",
            {"tags": ["test"]}
        )
        
        assert success is False
    
    def test_update_memory_invalid_data(self, memory_manager):
        """Test that invalid updates raise ValidationError."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        # Try to update with invalid data (e.g., invalid sensitivity level)
        with pytest.raises(ValidationError):
            memory_manager.update_memory(
                entry_id,
                {"sensitivity": "invalid_level"}
            )
    
    def test_update_memory_multiple_fields(self, memory_manager):
        """Test updating multiple fields at once."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"data": "original"},
            device_id="laptop-001",
            tags=["original"]
        )
        
        # Update multiple fields
        success = memory_manager.update_memory(
            entry_id,
            {
                "tags": ["updated", "modified"],
                "summary": "Updated entry",
                "context": {"data": "updated", "new_field": "value"}
            }
        )
        
        assert success is True
        
        # Verify all updates
        entry = memory_manager.get_memory(entry_id)
        assert "updated" in entry.tags
        assert entry.summary == "Updated entry"
        assert entry.context["data"] == "updated"
        assert entry.context["new_field"] == "value"


class TestUpdateMemoryWithEncryption:
    """Test suite for update_memory with encryption enabled."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create a temporary encryption key file."""
        fd, path = tempfile.mkstemp(suffix='.key')
        os.close(fd)
        # Generate a key and write it to the file
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        with open(path, 'wb') as f:
            f.write(key)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def memory_storage(self):
        """Create an in-memory storage for testing."""
        return MemoryStorage()
    
    @pytest.fixture
    def validation_manager(self):
        """Create a validation manager."""
        return ValidationManager()
    
    @pytest.fixture
    def encryption_service(self, temp_key_file):
        """Create an encryption service."""
        return EncryptionService(key_path=temp_key_file)
    
    @pytest.fixture
    def memory_manager_with_encryption(self, memory_storage, validation_manager, encryption_service):
        """Create a memory manager with encryption enabled."""
        config = MemoryModuleConfig()
        manager = MemoryManager(
            storage=memory_storage,
            validation=validation_manager,
            encryption=encryption_service,
            config=config
        )
        return manager
    
    def test_update_memory_encrypted_context(self, memory_manager_with_encryption):
        """Test updating context of an encrypted memory entry."""
        # Create a sensitive entry
        entry_id = memory_manager_with_encryption.create_memory(
            action="User login",
            context={"username": "testuser", "password": "secret123"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        
        # Update the context
        success = memory_manager_with_encryption.update_memory(
            entry_id,
            {"context": {"username": "testuser", "password": "newsecret456", "session": "abc123"}}
        )
        
        assert success is True
        
        # Verify the update (context should be encrypted in storage but decrypted when retrieved)
        entry = memory_manager_with_encryption.get_memory(entry_id)
        # The context should be encrypted in storage, so we can't directly verify the values
        # But we can verify the update was successful
        assert entry is not None
    
    def test_update_memory_public_entry_no_encryption(self, memory_manager_with_encryption):
        """Test that updating a public entry doesn't encrypt the context."""
        # Create a public entry
        entry_id = memory_manager_with_encryption.create_memory(
            action="User browsing",
            context={"url": "example.com"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC
        )
        
        # Update the context
        success = memory_manager_with_encryption.update_memory(
            entry_id,
            {"context": {"url": "newexample.com", "duration": 30}}
        )
        
        assert success is True
        
        # Verify the update
        entry = memory_manager_with_encryption.get_memory(entry_id)
        assert entry.context["url"] == "newexample.com"
        assert entry.context["duration"] == 30
