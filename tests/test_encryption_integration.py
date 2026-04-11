"""Integration tests for encryption service with SQLite storage."""

import pytest
import tempfile
import os
from pathlib import Path
from luma_memory.processing.encryption import EncryptionService
from luma_memory.storage.sqlite_storage import SQLiteStorage


class TestEncryptionIntegration:
    """Integration tests for EncryptionService with SQLiteStorage."""
    
    def test_encryption_with_sqlite_storage(self):
        """Test that encryption service stores keys in SQLite database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "test.key")
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create SQLite storage
            storage = SQLiteStorage(db_path)
            
            # Create encryption service with storage backend
            service = EncryptionService(key_path, storage_backend=storage)
            
            # Verify key metadata was stored
            metadata = service.get_active_key_metadata()
            assert metadata is not None
            assert metadata['is_active'] is True
            assert len(metadata['key_hash']) == 64
            
            # Verify encryption works
            data = "sensitive information"
            encrypted = service.encrypt(data)
            decrypted = service.decrypt(encrypted)
            assert decrypted == data
            
            # Clean up
            storage.close()
    
    def test_key_rotation_with_sqlite_storage(self):
        """Test key rotation with SQLite storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_key_path = os.path.join(tmpdir, "old_key.key")
            new_key_path = os.path.join(tmpdir, "new_key.key")
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create SQLite storage
            storage = SQLiteStorage(db_path)
            
            # Create encryption service with old key
            service = EncryptionService(old_key_path, storage_backend=storage)
            
            # Get old key metadata
            old_metadata = service.get_active_key_metadata()
            old_key_hash = old_metadata['key_hash']
            
            # Rotate to new key
            service.rotate_key(new_key_path, storage_backend=storage)
            
            # Verify new key is active
            new_metadata = service.get_active_key_metadata()
            assert new_metadata is not None
            assert new_metadata['is_active'] is True
            assert new_metadata['key_hash'] != old_key_hash
            
            # Verify encryption works with new key
            data = "new sensitive data"
            encrypted = service.encrypt(data)
            decrypted = service.decrypt(encrypted)
            assert decrypted == data
            
            # Clean up
            storage.close()
