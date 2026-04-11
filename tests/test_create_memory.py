"""
Test for create_memory method implementation (Task 10.3).
"""

import pytest
from datetime import datetime
import tempfile
import os

from luma_memory.memory_manager import MemoryManager
from luma_memory.models import SensitivityLevel
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.config import MemoryModuleConfig


class TestCreateMemory:
    """Test suite for create_memory method."""
    
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
    
    def test_create_memory_basic(self, memory_manager):
        """Test creating a basic memory entry."""
        entry_id = memory_manager.create_memory(
            action="User opened document",
            context={"file": "report.pdf", "page": 1},
            device_id="laptop-001"
        )
        
        assert entry_id is not None
        assert isinstance(entry_id, str)
        
        # Verify the entry was stored
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        assert entry.action == "User opened document"
        assert entry.context["file"] == "report.pdf"
        assert entry.device_id == "laptop-001"
    
    def test_create_memory_with_tags(self, memory_manager):
        """Test creating a memory entry with tags."""
        entry_id = memory_manager.create_memory(
            action="User searched for information",
            context={"query": "Python testing", "results": 10},
            device_id="laptop-001",
            tags=["search", "work"]
        )
        
        entry = memory_manager.get_memory(entry_id)
        assert "search" in entry.tags
        assert "work" in entry.tags
    
    def test_create_memory_with_sensitivity(self, memory_manager):
        """Test creating a memory entry with different sensitivity levels."""
        entry_id = memory_manager.create_memory(
            action="User logged in",
            context={"username": "testuser"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PRIVATE
        )
        
        entry = memory_manager.get_memory(entry_id)
        assert entry.sensitivity == SensitivityLevel.PRIVATE
    
    def test_create_memory_with_custom_id(self, memory_manager):
        """Test creating a memory entry with a custom ID."""
        custom_id = "custom-entry-123"
        entry_id = memory_manager.create_memory(
            action="Custom action",
            context={"data": "test"},
            device_id="laptop-001",
            entry_id=custom_id
        )
        
        assert entry_id == custom_id
        entry = memory_manager.get_memory(custom_id)
        assert entry is not None
    
    def test_create_memory_with_custom_timestamp(self, memory_manager):
        """Test creating a memory entry with a custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        entry_id = memory_manager.create_memory(
            action="Past action",
            context={"data": "test"},
            device_id="laptop-001",
            timestamp=custom_time
        )
        
        entry = memory_manager.get_memory(entry_id)
        assert entry.timestamp == custom_time
    
    def test_create_memory_sanitizes_input(self, memory_manager):
        """Test that create_memory sanitizes input data."""
        entry_id = memory_manager.create_memory(
            action="User input",
            context={"input": "<script>alert('xss')</script>"},
            device_id="laptop-001"
        )
        
        entry = memory_manager.get_memory(entry_id)
        # The sanitization should have cleaned the input
        assert entry is not None
        assert entry.context["input"] != "<script>alert('xss')</script>"
    
    def test_create_memory_validates_entry(self, memory_manager):
        """Test that create_memory validates entries."""
        from luma_memory.processing.validation import ValidationError
        
        # Try to create an entry with empty action (should fail validation)
        with pytest.raises(ValidationError):
            memory_manager.create_memory(
                action="",
                context={"data": "test"},
                device_id="laptop-001"
            )
    
    def test_create_memory_with_encryption(self):
        """Test creating a memory entry with encryption enabled."""
        # Create temporary directory for encryption key
        temp_dir = tempfile.mkdtemp()
        key_path = os.path.join(temp_dir, "encryption.key")
        
        try:
            storage = MemoryStorage()
            validation = ValidationManager()
            # EncryptionService will generate a key if it doesn't exist
            encryption = EncryptionService(key_path)
            config = MemoryModuleConfig()
            
            manager = MemoryManager(
                storage=storage,
                validation=validation,
                encryption=encryption,
                config=config
            )
            
            # Create a sensitive entry
            entry_id = manager.create_memory(
                action="Sensitive action",
                context={"password": "secret123"},
                device_id="laptop-001",
                sensitivity=SensitivityLevel.SENSITIVE
            )
            
            # Verify the entry was created and can be retrieved
            entry = manager.get_memory(entry_id)
            assert entry is not None
            assert entry.action == "Sensitive action"
            
        finally:
            # Clean up
            if os.path.exists(key_path):
                os.unlink(key_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    def test_create_memory_performance(self, memory_manager):
        """Test that create_memory completes within performance requirements."""
        import time
        
        start = time.time()
        entry_id = memory_manager.create_memory(
            action="Performance test",
            context={"data": "test"},
            device_id="laptop-001"
        )
        elapsed_ms = (time.time() - start) * 1000
        
        # Should complete within 100ms (requirement from design doc)
        assert elapsed_ms < 100
        assert entry_id is not None
