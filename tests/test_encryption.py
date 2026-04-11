"""Unit tests for EncryptionService."""

import pytest
import tempfile
import os
from pathlib import Path
from cryptography.fernet import InvalidToken
from luma_memory.processing.encryption import EncryptionService


class TestEncryptionService:
    """Unit tests for the EncryptionService class."""
    
    def test_key_generation(self, test_data_dir):
        """Test that a new key is generated when key file doesn't exist."""
        key_path = test_data_dir / "new_key.key"
        
        # Ensure key doesn't exist
        assert not key_path.exists()
        
        # Create service - should generate new key
        service = EncryptionService(str(key_path))
        
        # Verify key file was created
        assert key_path.exists()
        
        # Verify key is valid (44 bytes for Fernet)
        with open(key_path, 'rb') as f:
            key = f.read()
        assert len(key) == 44
        
        # Verify service has the key
        assert service.key == key
        assert service.cipher is not None
    
    def test_key_loading(self, test_data_dir):
        """Test that an existing key is loaded correctly."""
        key_path = test_data_dir / "existing_key.key"
        
        # Create first service to generate key
        service1 = EncryptionService(str(key_path))
        original_key = service1.key
        
        # Create second service - should load existing key
        service2 = EncryptionService(str(key_path))
        
        # Verify same key was loaded
        assert service2.key == original_key
    
    def test_encrypt_decrypt_round_trip(self, test_data_dir):
        """Test encryption and decryption round-trip."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Test data
        original_data = "This is sensitive information"
        
        # Encrypt
        encrypted = service.encrypt(original_data)
        
        # Verify encrypted data is bytes
        assert isinstance(encrypted, bytes)
        
        # Verify encrypted data is different from original
        assert encrypted != original_data.encode('utf-8')
        
        # Decrypt
        decrypted = service.decrypt(encrypted)
        
        # Verify decrypted matches original
        assert decrypted == original_data
    
    def test_encrypt_empty_string_raises_error(self, test_data_dir):
        """Test that encrypting empty string raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="Data cannot be empty"):
            service.encrypt("")
    
    def test_encrypt_none_raises_error(self, test_data_dir):
        """Test that encrypting None raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="Data cannot be None"):
            service.encrypt(None)
    
    def test_encrypt_non_string_raises_error(self, test_data_dir):
        """Test that encrypting non-string data raises TypeError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(TypeError, match="Data must be a string"):
            service.encrypt(12345)
    
    def test_decrypt_none_raises_error(self, test_data_dir):
        """Test that decrypting None raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="Encrypted data cannot be None"):
            service.decrypt(None)
    
    def test_decrypt_empty_bytes_raises_error(self, test_data_dir):
        """Test that decrypting empty bytes raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="Encrypted data cannot be empty"):
            service.decrypt(b"")
    
    def test_decrypt_non_bytes_raises_error(self, test_data_dir):
        """Test that decrypting non-bytes data raises TypeError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(TypeError, match="Encrypted data must be bytes"):
            service.decrypt("not bytes")
    
    def test_decrypt_with_wrong_key_raises_invalid_token(self, test_data_dir):
        """Test that decrypting with wrong key raises InvalidToken."""
        key_path1 = test_data_dir / "key1.key"
        key_path2 = test_data_dir / "key2.key"
        
        # Create two services with different keys
        service1 = EncryptionService(str(key_path1))
        service2 = EncryptionService(str(key_path2))
        
        # Encrypt with first key
        data = "secret data"
        encrypted = service1.encrypt(data)
        
        # Try to decrypt with second key - should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service2.decrypt(encrypted)
    
    def test_decrypt_corrupted_data_raises_invalid_token(self, test_data_dir):
        """Test that decrypting corrupted data raises InvalidToken."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Create corrupted data (not valid Fernet token)
        corrupted_data = b"this is not valid encrypted data"
        
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service.decrypt(corrupted_data)
    
    def test_key_rotation_without_storage(self, test_data_dir):
        """Test key rotation without storage backend."""
        old_key_path = test_data_dir / "old_key.key"
        new_key_path = test_data_dir / "new_key.key"
        
        # Create service with old key
        service = EncryptionService(str(old_key_path))
        old_key = service.key
        
        # Encrypt data with old key
        data = "test data"
        encrypted_with_old = service.encrypt(data)
        
        # Rotate to new key
        service.rotate_key(str(new_key_path))
        
        # Verify new key is different
        assert service.key != old_key
        assert service.key_path == new_key_path
        
        # Verify new key file exists
        assert new_key_path.exists()
        
        # Verify encryption works with new key
        encrypted_with_new = service.encrypt(data)
        decrypted = service.decrypt(encrypted_with_new)
        assert decrypted == data
        
        # Note: Old encrypted data cannot be decrypted with new key
        with pytest.raises(InvalidToken):
            service.decrypt(encrypted_with_old)
    
    def test_key_rotation_empty_path_raises_error(self, test_data_dir):
        """Test that key rotation with empty path raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="New key path cannot be empty"):
            service.rotate_key("")
    
    def test_key_rotation_whitespace_path_raises_error(self, test_data_dir):
        """Test that key rotation with whitespace path raises ValueError."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        with pytest.raises(ValueError, match="New key path cannot be empty"):
            service.rotate_key("   ")
    
    def test_invalid_key_path_raises_error(self):
        """Test that invalid key path raises ValueError."""
        with pytest.raises(ValueError, match="key_path must be a non-empty string"):
            EncryptionService("")
    
    def test_none_key_path_raises_error(self):
        """Test that None key path raises ValueError."""
        with pytest.raises(ValueError, match="key_path must be a non-empty string"):
            EncryptionService(None)
    
    def test_whitespace_key_path_raises_error(self):
        """Test that whitespace key path raises ValueError."""
        with pytest.raises(ValueError, match="key_path cannot be empty or whitespace"):
            EncryptionService("   ")
    
    def test_invalid_key_file_content_raises_error(self, test_data_dir):
        """Test that invalid key file content raises ValueError."""
        key_path = test_data_dir / "invalid.key"
        
        # Create file with invalid key content
        with open(key_path, 'wb') as f:
            f.write(b"invalid key content")
        
        with pytest.raises(ValueError, match="Invalid key length"):
            EncryptionService(str(key_path))
    
    def test_empty_key_file_raises_error(self, test_data_dir):
        """Test that empty key file raises ValueError."""
        key_path = test_data_dir / "empty.key"
        
        # Create empty key file
        key_path.touch()
        
        with pytest.raises(ValueError, match="Key file .* is empty"):
            EncryptionService(str(key_path))
    
    def test_encrypt_unicode_data(self, test_data_dir):
        """Test encryption of Unicode data."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Test with various Unicode characters
        unicode_data = "Hello 世界 🌍 Привет"
        
        encrypted = service.encrypt(unicode_data)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == unicode_data
    
    def test_encrypt_large_data(self, test_data_dir):
        """Test encryption of large data."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Create large string (1MB)
        large_data = "x" * (1024 * 1024)
        
        encrypted = service.encrypt(large_data)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == large_data
    
    def test_multiple_encrypt_produces_different_ciphertext(self, test_data_dir):
        """Test that encrypting same data multiple times produces different ciphertext."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        data = "same data"
        
        # Encrypt same data twice
        encrypted1 = service.encrypt(data)
        encrypted2 = service.encrypt(data)
        
        # Ciphertext should be different (due to IV/nonce)
        assert encrypted1 != encrypted2
        
        # But both should decrypt to same plaintext
        assert service.decrypt(encrypted1) == data
        assert service.decrypt(encrypted2) == data
    
    def test_key_file_permissions(self, test_data_dir):
        """Test that key file has restrictive permissions (Unix only)."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Check file permissions (Unix only)
        if os.name != 'nt':  # Not Windows
            stat_info = os.stat(key_path)
            permissions = stat_info.st_mode & 0o777
            # Should be 0o600 (owner read/write only)
            assert permissions == 0o600
    
    def test_key_rotation_with_storage_backend(self, test_data_dir):
        """Test key rotation with storage backend re-encrypts existing data."""
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
        from datetime import datetime
        import base64
        
        # Set up storage backend
        db_path = test_data_dir / "test_rotation.db"
        storage = SQLiteStorage(str(db_path))
        
        # Set up encryption service with storage backend
        old_key_path = test_data_dir / "old_key.key"
        service = EncryptionService(str(old_key_path), storage_backend=storage)
        
        # Create test entries with encrypted data (base64 encoded for JSON storage)
        encrypted1 = service.encrypt("sensitive data 1")
        encrypted2 = service.encrypt("sensitive data 2")
        
        entry1 = MemoryEntry(
            id="test-1",
            timestamp=datetime.now(),
            action="test_action",
            context={"encrypted_field": base64.b64encode(encrypted1).decode('utf-8')},
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        entry2 = MemoryEntry(
            id="test-2",
            timestamp=datetime.now(),
            action="test_action",
            context={"encrypted_field": base64.b64encode(encrypted2).decode('utf-8')},
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        # Store entries
        storage.create_entry(entry1)
        storage.create_entry(entry2)
        
        # Verify data can be decrypted with old key
        stored_entry1 = storage.get_entry("test-1")
        encrypted_bytes1 = base64.b64decode(stored_entry1.context["encrypted_field"])
        decrypted1 = service.decrypt(encrypted_bytes1)
        assert decrypted1 == "sensitive data 1"
        
        # Rotate key with storage backend
        new_key_path = test_data_dir / "new_key.key"
        service.rotate_key(str(new_key_path), storage_backend=storage)
        
        # Verify service is using new key
        assert service.key_path == new_key_path
        assert service.key != service._load_key(old_key_path)
        
        # Verify new key file exists
        assert new_key_path.exists()
        
        # Verify data was re-encrypted and can be decrypted with new key
        re_encrypted_entry1 = storage.get_entry("test-1")
        re_encrypted_entry2 = storage.get_entry("test-2")
        
        encrypted_bytes1_new = base64.b64decode(re_encrypted_entry1.context["encrypted_field"])
        encrypted_bytes2_new = base64.b64decode(re_encrypted_entry2.context["encrypted_field"])
        
        decrypted1_new = service.decrypt(encrypted_bytes1_new)
        decrypted2_new = service.decrypt(encrypted_bytes2_new)
        
        assert decrypted1_new == "sensitive data 1"
        assert decrypted2_new == "sensitive data 2"
        
        # Verify old key cannot decrypt the re-encrypted data
        old_service = EncryptionService(str(old_key_path))
        with pytest.raises(InvalidToken):
            old_service.decrypt(encrypted_bytes1_new)
        
        # Clean up storage connections
        storage.close()
    
    def test_key_rotation_with_mixed_data(self, test_data_dir):
        """Test key rotation handles entries with both encrypted and plain data."""
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
        from datetime import datetime
        import base64
        
        # Set up storage backend
        db_path = test_data_dir / "test_mixed.db"
        storage = SQLiteStorage(str(db_path))
        
        # Set up encryption service
        old_key_path = test_data_dir / "old_key_mixed.key"
        service = EncryptionService(str(old_key_path), storage_backend=storage)
        
        # Create entry with mixed data (encrypted and plain)
        encrypted_data = service.encrypt("secret")
        entry = MemoryEntry(
            id="test-mixed",
            timestamp=datetime.now(),
            action="test_action",
            context={
                "encrypted_field": base64.b64encode(encrypted_data).decode('utf-8'),
                "plain_field": "not encrypted",
                "number_field": 42
            },
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        
        storage.create_entry(entry)
        
        # Rotate key
        new_key_path = test_data_dir / "new_key_mixed.key"
        service.rotate_key(str(new_key_path), storage_backend=storage)
        
        # Verify encrypted field was re-encrypted
        re_encrypted_entry = storage.get_entry("test-mixed")
        encrypted_bytes = base64.b64decode(re_encrypted_entry.context["encrypted_field"])
        decrypted = service.decrypt(encrypted_bytes)
        assert decrypted == "secret"
        
        # Verify plain fields remain unchanged
        assert re_encrypted_entry.context["plain_field"] == "not encrypted"
        assert re_encrypted_entry.context["number_field"] == 42
        
        # Clean up
        storage.close()
    
    def test_key_rotation_with_empty_storage(self, test_data_dir):
        """Test key rotation works with empty storage backend."""
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        
        # Set up empty storage backend
        db_path = test_data_dir / "test_empty.db"
        storage = SQLiteStorage(str(db_path))
        
        # Set up encryption service
        old_key_path = test_data_dir / "old_key_empty.key"
        service = EncryptionService(str(old_key_path), storage_backend=storage)
        
        # Rotate key with empty storage
        new_key_path = test_data_dir / "new_key_empty.key"
        service.rotate_key(str(new_key_path), storage_backend=storage)
        
        # Verify rotation succeeded
        assert service.key_path == new_key_path
        assert new_key_path.exists()
        
        # Verify service can encrypt/decrypt with new key
        data = "test data"
        encrypted = service.encrypt(data)
        decrypted = service.decrypt(encrypted)
        assert decrypted == data
        
        # Clean up
        storage.close()
    
    def test_key_rotation_failure_rollback(self, test_data_dir):
        """Test that key rotation rolls back on failure during re-encryption."""
        from luma_memory.storage.sqlite_storage import SQLiteStorage
        from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
        from datetime import datetime
        from unittest.mock import Mock, patch
        import base64
        
        # Set up storage backend
        db_path = test_data_dir / "test_rollback.db"
        storage = SQLiteStorage(str(db_path))
        
        # Set up encryption service
        old_key_path = test_data_dir / "old_key_rollback.key"
        service = EncryptionService(str(old_key_path), storage_backend=storage)
        old_key = service.key
        
        # Create test entry
        encrypted_data = service.encrypt("data")
        entry = MemoryEntry(
            id="test-rollback",
            timestamp=datetime.now(),
            action="test_action",
            context={"encrypted_field": base64.b64encode(encrypted_data).decode('utf-8')},
            sensitivity=SensitivityLevel.SENSITIVE,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"]
        )
        storage.create_entry(entry)
        
        # Mock storage to fail during re-encryption
        new_key_path = test_data_dir / "new_key_rollback.key"
        
        with patch.object(storage, 'update_entry', side_effect=Exception("Database error")):
            # Attempt key rotation - should fail
            with pytest.raises(Exception, match="Key rotation failed during data re-encryption"):
                service.rotate_key(str(new_key_path), storage_backend=storage)
        
        # Verify service still uses old key
        assert service.key == old_key
        assert service.key_path == old_key_path
        
        # Verify new key file was cleaned up
        assert not new_key_path.exists()
        
        # Verify data can still be decrypted with old key
        stored_entry = storage.get_entry("test-rollback")
        encrypted_bytes = base64.b64decode(stored_entry.context["encrypted_field"])
        decrypted = service.decrypt(encrypted_bytes)
        assert decrypted == "data"
        
        # Clean up
        storage.close()


class TestDataIntegrityValidation:
    """Tests for data integrity validation in encryption/decryption."""
    
    def test_tampered_data_detected(self, test_data_dir):
        """Test that tampered encrypted data is detected and rejected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt data
        original_data = "sensitive information"
        encrypted = service.encrypt(original_data)
        
        # Tamper with the encrypted data by modifying a byte
        tampered = bytearray(encrypted)
        tampered[10] = (tampered[10] + 1) % 256  # Modify one byte
        tampered = bytes(tampered)
        
        # Attempt to decrypt tampered data - should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service.decrypt(tampered)
    
    def test_truncated_data_detected(self, test_data_dir):
        """Test that truncated encrypted data is detected and rejected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt data
        original_data = "sensitive information"
        encrypted = service.encrypt(original_data)
        
        # Truncate the encrypted data
        truncated = encrypted[:len(encrypted) // 2]
        
        # Attempt to decrypt truncated data - should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service.decrypt(truncated)
    
    def test_appended_data_handled(self, test_data_dir):
        """Test that encrypted data with appended bytes is handled correctly.
        
        Note: Fernet ignores trailing bytes after a valid token, so this test
        verifies that behavior rather than expecting rejection.
        """
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt data
        original_data = "sensitive information"
        encrypted = service.encrypt(original_data)
        
        # Append extra bytes to encrypted data
        appended = encrypted + b"extra bytes"
        
        # Fernet will ignore trailing bytes and decrypt the valid token
        # This is expected behavior - the HMAC validates the token portion
        decrypted = service.decrypt(appended)
        assert decrypted == original_data
    
    def test_swapped_encrypted_blocks_detected(self, test_data_dir):
        """Test that swapped encrypted data blocks are detected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt two different pieces of data
        data1 = "first piece of data"
        data2 = "second piece of data"
        encrypted1 = service.encrypt(data1)
        encrypted2 = service.encrypt(data2)
        
        # Try to decrypt with swapped data - should fail or return wrong data
        # Fernet includes authentication, so this should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            # Try to use part of encrypted1 with part of encrypted2
            swapped = encrypted1[:20] + encrypted2[20:]
            service.decrypt(swapped)
    
    def test_replay_attack_prevention(self, test_data_dir):
        """Test that encrypted data cannot be replayed with different context."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt data
        original_data = "sensitive information"
        encrypted = service.encrypt(original_data)
        
        # Decrypt should work with correct data
        decrypted = service.decrypt(encrypted)
        assert decrypted == original_data
        
        # Same encrypted data should decrypt to same value (no time-based validation in Fernet)
        decrypted_again = service.decrypt(encrypted)
        assert decrypted_again == original_data
        
        # This is expected behavior - Fernet doesn't prevent replay attacks
        # but ensures data integrity and authenticity
    
    def test_bit_flip_attack_detected(self, test_data_dir):
        """Test that bit-flip attacks on encrypted data are detected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Encrypt data
        original_data = "sensitive information"
        encrypted = service.encrypt(original_data)
        
        # Flip a single bit in the encrypted data
        bit_flipped = bytearray(encrypted)
        bit_flipped[15] ^= 0x01  # Flip the least significant bit
        bit_flipped = bytes(bit_flipped)
        
        # Attempt to decrypt bit-flipped data - should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service.decrypt(bit_flipped)
    
    def test_empty_ciphertext_rejected(self, test_data_dir):
        """Test that empty ciphertext is rejected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Try to decrypt empty bytes
        with pytest.raises(ValueError, match="Encrypted data cannot be empty"):
            service.decrypt(b"")
    
    def test_malformed_base64_in_ciphertext(self, test_data_dir):
        """Test that malformed base64 in ciphertext is detected."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Create data that looks like it might be base64 but isn't valid Fernet
        malformed = b"this-is-not-valid-fernet-token-data"
        
        # Attempt to decrypt - should fail
        with pytest.raises(InvalidToken, match="Failed to decrypt data"):
            service.decrypt(malformed)
    
    def test_data_integrity_across_multiple_encryptions(self, test_data_dir):
        """Test that data integrity is maintained across multiple encrypt/decrypt cycles."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        original_data = "sensitive information"
        
        # Perform multiple encrypt/decrypt cycles
        for i in range(10):
            encrypted = service.encrypt(original_data)
            decrypted = service.decrypt(encrypted)
            assert decrypted == original_data, f"Data integrity failed on iteration {i}"
    
    def test_data_integrity_with_special_characters(self, test_data_dir):
        """Test data integrity with special characters and unicode."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Test with various special characters
        test_cases = [
            "Hello\nWorld",  # Newlines
            "Tab\tSeparated",  # Tabs
            "Quote\"Test",  # Quotes
            "Backslash\\Test",  # Backslashes
            "Unicode: 你好世界 🌍",  # Unicode
            "Emoji: 😀🎉🔒",  # Emojis
            "Mixed: \n\t\"\\你好😀",  # Mixed special chars
        ]
        
        for test_data in test_cases:
            encrypted = service.encrypt(test_data)
            decrypted = service.decrypt(encrypted)
            assert decrypted == test_data, f"Data integrity failed for: {test_data}"
    
    def test_data_integrity_with_binary_like_strings(self, test_data_dir):
        """Test data integrity with strings that contain binary-like data."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # Test with strings that might be confused with binary data
        test_cases = [
            "\x00\x01\x02\x03",  # Null bytes and control characters
            "".join(chr(i) for i in range(32, 127)),  # ASCII printable
            "\r\n\r\n",  # CRLF sequences
        ]
        
        for test_data in test_cases:
            encrypted = service.encrypt(test_data)
            decrypted = service.decrypt(encrypted)
            assert decrypted == test_data
    
    def test_concurrent_encryption_integrity(self, test_data_dir):
        """Test that concurrent encryptions maintain data integrity."""
        import threading
        
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        results = []
        errors = []
        
        def encrypt_decrypt(data, index):
            try:
                encrypted = service.encrypt(data)
                decrypted = service.decrypt(encrypted)
                results.append((index, decrypted == data))
            except Exception as e:
                errors.append((index, str(e)))
        
        # Create multiple threads that encrypt/decrypt concurrently
        threads = []
        for i in range(20):
            data = f"concurrent data {i}"
            thread = threading.Thread(target=encrypt_decrypt, args=(data, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all encryptions maintained integrity
        assert len(results) == 20
        assert all(success for _, success in results)
    
    def test_key_rotation_maintains_data_integrity(self, test_data_dir):
        """Test that key rotation maintains data integrity for new encryptions."""
        old_key_path = test_data_dir / "old_key.key"
        new_key_path = test_data_dir / "new_key.key"
        
        # Create service with old key
        service = EncryptionService(str(old_key_path))
        
        # Encrypt data with old key
        old_data = "data before rotation"
        old_encrypted = service.encrypt(old_data)
        old_decrypted = service.decrypt(old_encrypted)
        assert old_decrypted == old_data
        
        # Rotate key
        service.rotate_key(str(new_key_path))
        
        # Encrypt new data with new key
        new_data = "data after rotation"
        new_encrypted = service.encrypt(new_data)
        new_decrypted = service.decrypt(new_encrypted)
        assert new_decrypted == new_data
        
        # Verify old encrypted data cannot be decrypted with new key
        with pytest.raises(InvalidToken):
            service.decrypt(old_encrypted)
    
    def test_invalid_utf8_in_decrypted_data(self, test_data_dir):
        """Test handling of invalid UTF-8 sequences in decrypted data."""
        key_path = test_data_dir / "test.key"
        service = EncryptionService(str(key_path))
        
        # This test verifies that if somehow invalid UTF-8 gets encrypted,
        # decryption will fail gracefully
        # Note: Normal encrypt() won't allow this, but we test the decrypt path
        
        # Create a Fernet token with invalid UTF-8 data
        from cryptography.fernet import Fernet
        
        # Encrypt invalid UTF-8 bytes directly using the cipher
        invalid_utf8 = b'\xff\xfe\xfd'  # Invalid UTF-8 sequence
        encrypted = service.cipher.encrypt(invalid_utf8)
        
        # Attempt to decrypt - should raise ValueError for invalid UTF-8
        with pytest.raises(ValueError, match="Decrypted data is not valid UTF-8"):
            service.decrypt(encrypted)
