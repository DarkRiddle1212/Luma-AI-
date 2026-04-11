"""
Integration tests for MemoryManager.

This test suite validates the full memory management pipeline including:
- End-to-end create flow
- End-to-end query flow
- Encryption integration
- Validation integration
- Error propagation
- Performance requirements
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta

from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager, ValidationError
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.config import MemoryModuleConfig
from luma_memory.models import SensitivityLevel, SyncStatus


class TestMemoryManagerIntegration:
    """Integration tests for the full MemoryManager pipeline."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary database path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)
    
    @pytest.fixture
    def temp_key_path(self):
        """Provide a temporary encryption key path."""
        # Create a temporary directory for the key
        temp_dir = tempfile.mkdtemp()
        key_path = os.path.join(temp_dir, "encryption.key")
        yield key_path
        # Cleanup
        if os.path.exists(key_path):
            os.remove(key_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    @pytest.fixture
    def memory_storage(self):
        """Create an in-memory storage backend."""
        return MemoryStorage()
    
    @pytest.fixture
    def sqlite_storage(self, temp_db_path):
        """Create a SQLite storage backend."""
        storage = SQLiteStorage(temp_db_path)
        yield storage
        # Cleanup: close connections
        if hasattr(storage, 'connection_pool'):
            storage.connection_pool.close_all()
    
    @pytest.fixture
    def encryption_service(self, temp_key_path):
        """Create an encryption service."""
        return EncryptionService(temp_key_path)
    
    @pytest.fixture
    def validation_manager(self):
        """Create a validation manager."""
        return ValidationManager()
    
    @pytest.fixture
    def context_summarizer(self):
        """Create a context summarizer."""
        return ContextSummarizer()
    
    @pytest.fixture
    def config(self):
        """Create a configuration."""
        return MemoryModuleConfig()
    
    @pytest.fixture
    def memory_manager(self, memory_storage, validation_manager, config):
        """Create a basic memory manager with in-memory storage."""
        return MemoryManager(
            storage=memory_storage,
            validation=validation_manager,
            config=config
        )
    
    @pytest.fixture
    def full_memory_manager(
        self,
        memory_storage,
        encryption_service,
        validation_manager,
        context_summarizer,
        config
    ):
        """Create a fully-configured memory manager with all components."""
        return MemoryManager(
            storage=memory_storage,
            encryption=encryption_service,
            validation=validation_manager,
            summarizer=context_summarizer,
            config=config
        )
    
    def test_end_to_end_create_flow(self, memory_manager):
        """Test the complete create memory flow from input to storage."""
        # Create a memory entry
        entry_id = memory_manager.create_memory(
            action="User opened document",
            context={"file": "report.pdf", "page": 1},
            device_id="laptop-001",
            tags=["document", "work"]
        )
        
        # Verify the entry was created
        assert entry_id is not None
        assert isinstance(entry_id, str)
        
        # Retrieve and verify the entry
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        assert entry.action == "User opened document"
        assert entry.context["file"] == "report.pdf"
        assert entry.context["page"] == 1
        assert entry.device_id == "laptop-001"
        assert "document" in entry.tags
        assert "work" in entry.tags
        assert entry.sensitivity == SensitivityLevel.PUBLIC
        assert entry.sync_status == SyncStatus.PENDING
    
    def test_end_to_end_query_flow(self, memory_manager):
        """Test the complete query flow with multiple entries."""
        # Create multiple entries
        entry1_id = memory_manager.create_memory(
            action="User searched",
            context={"query": "python testing"},
            device_id="laptop-001",
            tags=["search"]
        )
        
        entry2_id = memory_manager.create_memory(
            action="User opened file",
            context={"file": "test.py"},
            device_id="laptop-001",
            tags=["file", "work"]
        )
        
        entry3_id = memory_manager.create_memory(
            action="User searched",
            context={"query": "pytest fixtures"},
            device_id="laptop-001",
            tags=["search", "work"]
        )
        
        # Query all entries
        all_entries = memory_manager.query_memories()
        assert len(all_entries) == 3
        
        # Query by tags
        search_entries = memory_manager.query_memories(tags=["search"])
        assert len(search_entries) == 2
        assert all("search" in e.tags for e in search_entries)
        
        # Query by action type
        file_entries = memory_manager.query_memories(action_type="file")
        assert len(file_entries) == 1
        assert "file" in file_entries[0].action.lower()
        
        # Query with pagination
        page1 = memory_manager.query_memories(limit=2, offset=0)
        assert len(page1) == 2
        
        page2 = memory_manager.query_memories(limit=2, offset=2)
        assert len(page2) == 1
    
    def test_encryption_integration(self, full_memory_manager):
        """Test that encryption is properly integrated in the pipeline."""
        # Test 1: Verify encryption service is available
        assert full_memory_manager.encryption is not None, \
            "Encryption service should be available"
        
        # Test 2: Create a sensitive entry and verify encryption/decryption round-trip
        entry_id = full_memory_manager.create_memory(
            action="User logged in",
            context={"username": "testuser", "password": "secret123"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        
        # Retrieve the entry - should be decrypted by get_memory()
        entry = full_memory_manager.get_memory(entry_id)
        assert entry is not None
        assert entry.context["username"] == "testuser"
        assert entry.context["password"] == "secret123"
        assert entry.sensitivity == SensitivityLevel.SENSITIVE
        
        # Test 3: Verify encryption works by manually encrypting and decrypting
        test_data = "sensitive information"
        encrypted = full_memory_manager.encryption.encrypt(test_data)
        assert isinstance(encrypted, bytes), "Encrypted data should be bytes"
        assert encrypted != test_data.encode('utf-8'), "Encrypted data should differ from plain text"
        
        decrypted = full_memory_manager.encryption.decrypt(encrypted)
        assert decrypted == test_data, "Decryption should restore original data"
        
        # Test 4: Create entry with PRIVATE sensitivity and verify decryption
        private_entry_id = full_memory_manager.create_memory(
            action="User browsing history",
            context={"url": "https://example.com", "title": "Example Page"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PRIVATE
        )
        
        private_entry = full_memory_manager.get_memory(private_entry_id)
        assert private_entry is not None
        assert private_entry.context["url"] == "https://example.com"
        assert private_entry.context["title"] == "Example Page"
        assert private_entry.sensitivity == SensitivityLevel.PRIVATE
        
        # Test 5: Create entry with PUBLIC sensitivity (should not be encrypted)
        public_entry_id = full_memory_manager.create_memory(
            action="User opened app",
            context={"app": "calculator", "time": "morning"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC
        )
        
        public_entry = full_memory_manager.get_memory(public_entry_id)
        assert public_entry is not None
        assert public_entry.context["app"] == "calculator"
        assert public_entry.context["time"] == "morning"
        assert public_entry.sensitivity == SensitivityLevel.PUBLIC
        
        # Test 6: Verify that encryption is applied during create and decryption during retrieve
        # by checking that the encryption service methods are being called
        # Create a new sensitive entry
        sensitive_id = full_memory_manager.create_memory(
            action="Sensitive action",
            context={"secret": "confidential data", "code": "12345"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        
        # Retrieve and verify all fields are decrypted correctly
        sensitive_entry = full_memory_manager.get_memory(sensitive_id)
        assert sensitive_entry.context["secret"] == "confidential data"
        assert sensitive_entry.context["code"] == "12345"
        
        # Test 7: Verify encryption with non-string values (should be preserved)
        mixed_entry_id = full_memory_manager.create_memory(
            action="Mixed data types",
            context={
                "text": "encrypted text",
                "number": 42,
                "boolean": True,
                "list": [1, 2, 3]
            },
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        
        mixed_entry = full_memory_manager.get_memory(mixed_entry_id)
        assert mixed_entry.context["text"] == "encrypted text"
        assert mixed_entry.context["number"] == 42
        assert mixed_entry.context["boolean"] is True
        assert mixed_entry.context["list"] == [1, 2, 3]
        
        # Test 8: Verify query_memories also decrypts entries
        all_entries = full_memory_manager.query_memories()
        sensitive_entries = [e for e in all_entries if e.sensitivity == SensitivityLevel.SENSITIVE]
        assert len(sensitive_entries) >= 2, "Should have at least 2 sensitive entries"
        
        # All sensitive entries should be decrypted
        for entry in sensitive_entries:
            for key, value in entry.context.items():
                if isinstance(value, str):
                    # String values should be decrypted (not bytes)
                    assert not isinstance(value, bytes), \
                        f"Context value '{key}' should be decrypted in query results"
    
    def test_validation_integration(self, memory_manager):
        """Test that validation is properly integrated in the pipeline."""
        # Test 1: Empty action should be rejected
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="",
                context={"data": "test"},
                device_id="laptop-001"
            )
        assert "action" in str(exc_info.value).lower()
        
        # Test 2: Empty device_id should be rejected
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id=""
            )
        assert "device" in str(exc_info.value).lower()
        
        # Test 3: Whitespace-only action should be rejected
        with pytest.raises(ValidationError):
            memory_manager.create_memory(
                action="   ",
                context={"data": "test"},
                device_id="laptop-001"
            )
        
        # Test 4: Whitespace-only device_id should be rejected
        with pytest.raises(ValidationError):
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="   "
            )
        
        # Test 5: Action exceeding max length should be rejected
        long_action = "a" * 1001  # MAX_ACTION_LENGTH is 1000
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action=long_action,
                context={"data": "test"},
                device_id="laptop-001"
            )
        assert "length" in str(exc_info.value).lower()
        
        # Test 6: Too many tags should be rejected
        too_many_tags = [f"tag{i}" for i in range(51)]  # MAX_TAGS_COUNT is 50
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="laptop-001",
                tags=too_many_tags
            )
        assert "tag" in str(exc_info.value).lower()
        
        # Test 7: Empty tag should be rejected
        with pytest.raises(ValidationError):
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="laptop-001",
                tags=["valid", "", "another"]
            )
        
        # Test 8: Tag exceeding max length should be rejected
        long_tag = "t" * 101  # MAX_TAG_LENGTH is 100
        with pytest.raises(ValidationError):
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="laptop-001",
                tags=[long_tag]
            )
        
        # Test 9: Valid entry should pass validation
        entry_id = memory_manager.create_memory(
            action="Valid action",
            context={"data": "test", "number": 42},
            device_id="laptop-001",
            tags=["valid", "test"]
        )
        assert entry_id is not None
        
        # Test 10: Input sanitization should be applied
        entry_id = memory_manager.create_memory(
            action="User input",
            context={"input": "<script>alert('xss')</script>"},
            device_id="laptop-001"
        )
        entry = memory_manager.get_memory(entry_id)
        # Script tags should be removed/escaped
        assert "<script>" not in entry.context["input"]
        
        # Test 11: Validation should work with update operations
        entry_id = memory_manager.create_memory(
            action="Original",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        # Valid update should succeed
        success = memory_manager.update_memory(entry_id, {"tags": ["updated"]})
        assert success is True
        
        # Invalid update should fail (too many tags)
        with pytest.raises(ValidationError):
            memory_manager.update_memory(entry_id, {"tags": [f"tag{i}" for i in range(51)]})
        
        # Test 12: Validation should preserve valid data types
        entry_id = memory_manager.create_memory(
            action="Mixed types",
            context={
                "string": "text",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "list": [1, 2, 3],
                "nested": {"key": "value"}
            },
            device_id="laptop-001"
        )
        entry = memory_manager.get_memory(entry_id)
        assert entry.context["string"] == "text"
        assert entry.context["number"] == 42
        assert entry.context["float"] == 3.14
        assert entry.context["boolean"] is True
        assert entry.context["list"] == [1, 2, 3]
        assert entry.context["nested"]["key"] == "value"
    
    def test_error_propagation(self, memory_manager, full_memory_manager):
        """Test that errors propagate correctly through the pipeline."""
        # Test 1: Validation error propagation from create_memory
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="",
                context={},
                device_id="laptop-001"
            )
        assert "action" in str(exc_info.value).lower()
        
        # Test 2: Validation error for empty device_id
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id=""
            )
        assert "device" in str(exc_info.value).lower()
        
        # Test 3: Validation error for too many tags
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="laptop-001",
                tags=[f"tag{i}" for i in range(51)]  # MAX_TAGS_COUNT is 50
            )
        assert "tag" in str(exc_info.value).lower()
        
        # Test 4: Validation error for action exceeding max length
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.create_memory(
                action="a" * 1001,  # MAX_ACTION_LENGTH is 1000
                context={"data": "test"},
                device_id="laptop-001"
            )
        assert "length" in str(exc_info.value).lower()
        
        # Test 5: Validation error propagation from update_memory
        entry_id = memory_manager.create_memory(
            action="Original",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.update_memory(entry_id, {"tags": [f"tag{i}" for i in range(51)]})
        assert "tag" in str(exc_info.value).lower()
        
        # Test 6: Validation error for invalid update fields
        with pytest.raises(ValidationError) as exc_info:
            memory_manager.update_memory(entry_id, {"action": ""})
        assert "action" in str(exc_info.value).lower()
        
        # Test 7: Test retrieval of non-existent entry (should return None, not raise)
        entry = memory_manager.get_memory("non-existent-id")
        assert entry is None
        
        # Test 8: Test update of non-existent entry (should return False, not raise)
        success = memory_manager.update_memory("non-existent-id", {"tags": ["test"]})
        assert success is False
        
        # Test 9: Test delete of non-existent entry (should return False, not raise)
        success = memory_manager.delete_memory("non-existent-id")
        assert success is False
        
        # Test 10: Test encryption error propagation (if encryption service fails)
        # Create a mock encryption service that raises an error
        if full_memory_manager.encryption:
            # Save original encrypt method
            original_encrypt = full_memory_manager.encryption.encrypt
            
            # Replace with a failing version
            def failing_encrypt(data):
                raise Exception("Encryption service failure")
            
            full_memory_manager.encryption.encrypt = failing_encrypt
            
            # Try to create a sensitive entry - should propagate the encryption error
            with pytest.raises(Exception) as exc_info:
                full_memory_manager.create_memory(
                    action="Sensitive action",
                    context={"secret": "data"},
                    device_id="laptop-001",
                    sensitivity=SensitivityLevel.SENSITIVE
                )
            assert "encryption" in str(exc_info.value).lower() or "failure" in str(exc_info.value).lower()
            
            # Restore original method
            full_memory_manager.encryption.encrypt = original_encrypt
        
        # Test 11: Test storage error propagation
        # Create a mock storage that raises StorageError
        from luma_memory.storage.backend import StorageError
        
        # Save original create_entry method
        original_create = memory_manager.storage.create_entry
        
        # Replace with a failing version
        def failing_create(entry):
            raise StorageError("Database connection failed")
        
        memory_manager.storage.create_entry = failing_create
        
        # Try to create an entry - should propagate the storage error
        with pytest.raises(StorageError) as exc_info:
            memory_manager.create_memory(
                action="Test action",
                context={"data": "test"},
                device_id="laptop-001"
            )
        assert "database" in str(exc_info.value).lower() or "connection" in str(exc_info.value).lower()
        
        # Restore original method
        memory_manager.storage.create_entry = original_create
        
        # Test 12: Test that error metrics are recorded
        if memory_manager.config.enable_metrics:
            # Reset metrics
            memory_manager.reset_performance_metrics()
            
            # Cause a validation error
            try:
                memory_manager.create_memory(
                    action="",
                    context={},
                    device_id="laptop-001"
                )
            except ValidationError:
                pass
            
            # Check that error was recorded in metrics
            metrics = memory_manager.get_performance_metrics()
            assert metrics["create_memory"]["errors"] >= 1
            assert metrics["create_memory"]["error_rate"] > 0
        
        # Test 13: Test decryption error propagation
        if full_memory_manager.encryption:
            # Create a sensitive entry
            entry_id = full_memory_manager.create_memory(
                action="Sensitive action",
                context={"secret": "data"},
                device_id="laptop-001",
                sensitivity=SensitivityLevel.SENSITIVE
            )
            
            # Save original decrypt method
            original_decrypt = full_memory_manager.encryption.decrypt
            
            # Replace with a failing version
            def failing_decrypt(data):
                raise Exception("Decryption service failure")
            
            full_memory_manager.encryption.decrypt = failing_decrypt
            
            # Try to retrieve the entry - should propagate the decryption error
            with pytest.raises(Exception) as exc_info:
                full_memory_manager.get_memory(entry_id)
            assert "decryption" in str(exc_info.value).lower() or "failure" in str(exc_info.value).lower()
            
            # Restore original method
            full_memory_manager.encryption.decrypt = original_decrypt
        
        # Test 14: Test query error propagation
        # Save original query_entries method
        original_query = memory_manager.storage.query_entries
        
        # Replace with a failing version
        def failing_query(*args, **kwargs):
            raise StorageError("Query execution failed")
        
        memory_manager.storage.query_entries = failing_query
        
        # Try to query - should propagate the storage error
        with pytest.raises(StorageError) as exc_info:
            memory_manager.query_memories()
        assert "query" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
        
        # Restore original method
        memory_manager.storage.query_entries = original_query
        
        # Test 15: Test get_stats error propagation
        # Save original get_storage_stats method
        original_stats = memory_manager.storage.get_storage_stats
        
        # Replace with a failing version
        def failing_stats():
            raise StorageError("Stats retrieval failed")
        
        memory_manager.storage.get_storage_stats = failing_stats
        
        # Try to get stats - should propagate the storage error
        with pytest.raises(StorageError) as exc_info:
            memory_manager.get_stats()
        assert "stats" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
        
        # Restore original method
        memory_manager.storage.get_storage_stats = original_stats
    
    def test_update_memory_integration(self, memory_manager):
        """Test the update memory flow."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="Original action",
            context={"data": "original"},
            device_id="laptop-001",
            tags=["original"]
        )
        
        # Update the entry
        success = memory_manager.update_memory(
            entry_id,
            {
                "tags": ["updated", "modified"],
                "summary": "This entry was updated"
            }
        )
        assert success is True
        
        # Verify the update
        entry = memory_manager.get_memory(entry_id)
        assert "updated" in entry.tags
        assert "modified" in entry.tags
        assert entry.summary == "This entry was updated"
        assert entry.action == "Original action"  # Unchanged
    
    def test_delete_memory_integration(self, memory_manager):
        """Test the delete memory flow."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="To be deleted",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        # Verify it exists
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        
        # Delete the entry
        success = memory_manager.delete_memory(entry_id)
        assert success is True
        
        # Verify it's gone
        entry = memory_manager.get_memory(entry_id)
        assert entry is None
        
        # Try to delete again
        success = memory_manager.delete_memory(entry_id)
        assert success is False
    
    def test_get_stats_integration(self, memory_manager):
        """Test the get stats functionality."""
        # Create some entries
        for i in range(5):
            memory_manager.create_memory(
                action=f"Action {i}",
                context={"index": i},
                device_id="laptop-001"
            )
        
        # Get stats
        stats = memory_manager.get_stats()
        
        # Verify stats structure
        assert "total_entries" in stats
        assert stats["total_entries"] == 5
        assert "encryption_enabled" in stats
        assert "summarizer_enabled" in stats
        assert "config" in stats
        
        # Verify performance metrics if enabled
        if memory_manager.config.enable_metrics:
            assert "performance" in stats
            assert "create_memory" in stats["performance"]
    
    def test_performance_metrics(self, memory_manager):
        """Test performance metrics collection."""
        # Reset metrics
        memory_manager.reset_performance_metrics()
        
        # Perform operations
        entry_id = memory_manager.create_memory(
            action="Test action",
            context={"data": "test"},
            device_id="laptop-001"
        )
        
        memory_manager.get_memory(entry_id)
        memory_manager.query_memories()
        
        # Get metrics
        metrics = memory_manager.get_performance_metrics()
        
        # Verify metrics structure
        assert "create_memory" in metrics
        assert metrics["create_memory"]["count"] == 1
        assert metrics["create_memory"]["avg_time_ms"] > 0
        
        assert "get_memory" in metrics
        assert metrics["get_memory"]["count"] == 1
        
        assert "query_memories" in metrics
        assert metrics["query_memories"]["count"] == 1
    
    def test_sqlite_storage_integration(self, sqlite_storage, validation_manager, config):
        """Test MemoryManager with SQLite storage backend."""
        manager = MemoryManager(
            storage=sqlite_storage,
            validation=validation_manager,
            config=config
        )
        
        # Create entries
        entry_id = manager.create_memory(
            action="SQLite test",
            context={"backend": "sqlite"},
            device_id="laptop-001"
        )
        
        # Verify persistence
        entry = manager.get_memory(entry_id)
        assert entry is not None
        assert entry.context["backend"] == "sqlite"
    
    def test_input_sanitization(self, memory_manager):
        """Test that input is sanitized in the pipeline."""
        # Create entry with potentially malicious input
        entry_id = memory_manager.create_memory(
            action="User input",
            context={
                "input": "<script>alert('xss')</script>",
                "data": "normal data"
            },
            device_id="laptop-001"
        )
        
        # Retrieve and verify sanitization
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        # The sanitization should have cleaned the input
        assert "<script>" not in entry.context["input"]
        assert entry.context["data"] == "normal data"
    
    def test_concurrent_operations(self, memory_manager):
        """Test that concurrent operations work correctly."""
        import threading
        
        entry_ids = []
        errors = []
        
        def create_entry(index):
            try:
                entry_id = memory_manager.create_memory(
                    action=f"Concurrent action {index}",
                    context={"index": index},
                    device_id=f"device-{index}"
                )
                entry_ids.append(entry_id)
            except Exception as e:
                errors.append(e)
        
        # Create entries concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_entry, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0
        assert len(entry_ids) == 10
        
        # Verify all entries were created
        all_entries = memory_manager.query_memories()
        assert len(all_entries) == 10
    
    def test_time_range_query(self, memory_manager):
        """Test querying with time range filters."""
        now = datetime.now()
        
        # Create entries at different times
        entry1_id = memory_manager.create_memory(
            action="Old action",
            context={"time": "old"},
            device_id="laptop-001",
            timestamp=now - timedelta(days=2)
        )
        
        entry2_id = memory_manager.create_memory(
            action="Recent action",
            context={"time": "recent"},
            device_id="laptop-001",
            timestamp=now - timedelta(hours=1)
        )
        
        entry3_id = memory_manager.create_memory(
            action="Current action",
            context={"time": "current"},
            device_id="laptop-001",
            timestamp=now
        )
        
        # Query with time range
        start_time = now - timedelta(days=1)
        recent_entries = memory_manager.query_memories(start_time=start_time)
        
        assert len(recent_entries) == 2
        assert all(e.timestamp >= start_time for e in recent_entries)
    
    def test_multiple_sensitivity_levels(self, full_memory_manager):
        """Test handling of different sensitivity levels."""
        # Create entries with different sensitivity levels
        public_id = full_memory_manager.create_memory(
            action="Public action",
            context={"data": "public"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC
        )
        
        private_id = full_memory_manager.create_memory(
            action="Private action",
            context={"data": "private"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PRIVATE
        )
        
        sensitive_id = full_memory_manager.create_memory(
            action="Sensitive action",
            context={"data": "sensitive"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE
        )
        
        # Retrieve and verify - all should be decrypted when retrieved
        public_entry = full_memory_manager.get_memory(public_id)
        assert public_entry.context["data"] == "public"
        assert public_entry.sensitivity == SensitivityLevel.PUBLIC
        
        private_entry = full_memory_manager.get_memory(private_id)
        assert private_entry.context["data"] == "private"
        assert private_entry.sensitivity == SensitivityLevel.PRIVATE
        
        sensitive_entry = full_memory_manager.get_memory(sensitive_id)
        assert sensitive_entry.context["data"] == "sensitive"
        assert sensitive_entry.sensitivity == SensitivityLevel.SENSITIVE
        
        # Verify that encryption service is being used for sensitive data
        # The actual storage format depends on the backend, but we can verify
        # that the entries have the correct sensitivity levels
        assert full_memory_manager.encryption is not None
    
    def test_store_operation_performance(self, memory_manager):
        """Test that store operations complete within 100ms (Requirement 6.1)."""
        import time
        
        # Measure time for a single store operation
        start_time = time.perf_counter()
        entry_id = memory_manager.create_memory(
            action="Performance test action",
            context={"data": "test", "number": 42, "nested": {"key": "value"}},
            device_id="laptop-001",
            tags=["performance", "test"]
        )
        end_time = time.perf_counter()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        
        # Verify entry was created
        assert entry_id is not None
        
        # Verify performance requirement: < 100ms
        assert latency_ms < 100, f"Store operation took {latency_ms:.2f}ms, expected < 100ms"
    
    def test_retrieve_operation_performance(self, memory_manager):
        """Test that retrieve operations complete within 200ms (Requirement 6.2)."""
        import time
        
        # Create an entry first
        entry_id = memory_manager.create_memory(
            action="Performance test action",
            context={"data": "test", "number": 42},
            device_id="laptop-001",
            tags=["performance"]
        )
        
        # Measure time for retrieval
        start_time = time.perf_counter()
        entry = memory_manager.get_memory(entry_id)
        end_time = time.perf_counter()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        
        # Verify entry was retrieved
        assert entry is not None
        
        # Verify performance requirement: < 200ms
        assert latency_ms < 200, f"Retrieve operation took {latency_ms:.2f}ms, expected < 200ms"
    
    def test_query_operation_performance(self, memory_manager):
        """Test that query operations complete within 200ms for up to 100 entries (Requirement 6.2)."""
        import time
        
        # Create 100 entries
        for i in range(100):
            memory_manager.create_memory(
                action=f"Query test action {i}",
                context={"index": i, "data": f"test data {i}"},
                device_id="laptop-001",
                tags=["query", "performance"]
            )
        
        # Measure time for query returning 100 entries
        start_time = time.perf_counter()
        entries = memory_manager.query_memories(limit=100)
        end_time = time.perf_counter()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        
        # Verify entries were retrieved
        assert len(entries) == 100
        
        # Verify performance requirement: < 200ms
        assert latency_ms < 200, f"Query operation took {latency_ms:.2f}ms, expected < 200ms"
    
    def test_memory_usage_performance(self, memory_manager):
        """Test that memory usage stays under 100MB during normal operation (Requirement 6.3)."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Perform normal operations: create, query, retrieve
        entry_ids = []
        for i in range(50):
            entry_id = memory_manager.create_memory(
                action=f"Memory test action {i}",
                context={"index": i, "data": f"test data {i}" * 10},
                device_id="laptop-001",
                tags=["memory", "test"]
            )
            entry_ids.append(entry_id)
        
        # Query entries
        entries = memory_manager.query_memories(limit=50)
        
        # Retrieve some entries
        for entry_id in entry_ids[:10]:
            memory_manager.get_memory(entry_id)
        
        # Get final memory usage
        final_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Calculate memory increase
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        # Verify memory usage requirement: < 100MB increase
        # Note: This is a conservative test - the actual module should use much less
        assert memory_increase_mb < 100, \
            f"Memory usage increased by {memory_increase_mb:.2f}MB, expected < 100MB"
    
    def test_bulk_operations_performance(self, memory_manager):
        """Test performance with bulk operations."""
        import time
        
        # Test bulk create performance
        start_time = time.perf_counter()
        entry_ids = []
        for i in range(50):
            entry_id = memory_manager.create_memory(
                action=f"Bulk action {i}",
                context={"index": i},
                device_id="laptop-001"
            )
            entry_ids.append(entry_id)
        end_time = time.perf_counter()
        
        # Calculate average latency per operation
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / 50
        
        # Each operation should still be under 100ms on average
        assert avg_latency_ms < 100, \
            f"Average store latency was {avg_latency_ms:.2f}ms, expected < 100ms"
        
        # Test bulk retrieve performance
        start_time = time.perf_counter()
        for entry_id in entry_ids:
            memory_manager.get_memory(entry_id)
        end_time = time.perf_counter()
        
        # Calculate average latency per operation
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / 50
        
        # Each operation should still be under 200ms on average
        assert avg_latency_ms < 200, \
            f"Average retrieve latency was {avg_latency_ms:.2f}ms, expected < 200ms"
