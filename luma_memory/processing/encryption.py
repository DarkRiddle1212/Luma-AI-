"""
Encryption service for Luma Memory Module.

This module provides AES-256 encryption/decryption capabilities for sensitive
memory entry data using the Fernet symmetric encryption scheme from the
cryptography library.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime, UTC
from cryptography.fernet import Fernet, InvalidToken
import logging

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Handles encryption and decryption of sensitive data using AES-256.
    
    Uses Fernet symmetric encryption which provides:
    - AES-256 encryption in CBC mode
    - HMAC for authentication
    - Timestamp for key rotation support
    
    The encryption key is stored securely on the filesystem and can be rotated
    when needed.
    
    Attributes:
        key_path: Path to the encryption key file
        key: The encryption key bytes
        cipher: Fernet cipher instance for encryption/decryption
    
    Example:
        >>> service = EncryptionService("./keys/encryption.key")
        >>> encrypted = service.encrypt("sensitive data")
        >>> decrypted = service.decrypt(encrypted)
        >>> assert decrypted == "sensitive data"
    """
    
    def __init__(self, key_path: str, storage_backend=None):
        """
        Initialize the encryption service.
        
        Args:
            key_path: Path to the encryption key file. If the file doesn't exist,
                     a new key will be generated and stored at this path.
            storage_backend: Optional storage backend for storing key metadata in database.
                           Must have a connection_pool attribute with get_connection() method.
        
        Raises:
            OSError: If the key file cannot be read or created
            ValueError: If the key file contains invalid data or key_path is invalid
        """
        if not key_path or not isinstance(key_path, str):
            raise ValueError("key_path must be a non-empty string")
        
        if not key_path.strip():
            raise ValueError("key_path cannot be empty or whitespace")
        
        self.key_path = Path(key_path)
        self.storage_backend = storage_backend
        self.key = self._load_or_generate_key(self.key_path)
        self.cipher = Fernet(self.key)
        logger.info(f"EncryptionService initialized with key at {self.key_path}")
    
    def encrypt(self, data: str) -> bytes:
        """
        Encrypt data using AES-256.
        
        Args:
            data: Plain text string to encrypt
        
        Returns:
            Encrypted data as bytes
        
        Raises:
            TypeError: If data is not a string
            ValueError: If data is None or empty
            Exception: If encryption fails
        
        Example:
            >>> service = EncryptionService("./keys/test.key")
            >>> encrypted = service.encrypt("my secret")
            >>> isinstance(encrypted, bytes)
            True
        """
        if data is None:
            raise ValueError("Data cannot be None")
        
        if not isinstance(data, str):
            raise TypeError(f"Data must be a string, got {type(data)}")
        
        if not data:
            raise ValueError("Data cannot be empty")
        
        try:
            encrypted_data = self.cipher.encrypt(data.encode('utf-8'))
            logger.debug(f"Successfully encrypted {len(data)} characters")
            return encrypted_data
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise Exception(f"Encryption failed: {str(e)}")
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """
        Decrypt data and validate integrity.
        
        Args:
            encrypted_data: Encrypted data as bytes
        
        Returns:
            Decrypted plain text string
        
        Raises:
            TypeError: If encrypted_data is not bytes
            ValueError: If encrypted_data is None or empty
            InvalidToken: If the data is corrupted or the key is wrong
            Exception: If decryption fails
        
        Example:
            >>> service = EncryptionService("./keys/test.key")
            >>> encrypted = service.encrypt("my secret")
            >>> decrypted = service.decrypt(encrypted)
            >>> decrypted == "my secret"
            True
        """
        if encrypted_data is None:
            raise ValueError("Encrypted data cannot be None")
        
        if not isinstance(encrypted_data, bytes):
            raise TypeError(f"Encrypted data must be bytes, got {type(encrypted_data)}")
        
        if not encrypted_data:
            raise ValueError("Encrypted data cannot be empty")
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            result = decrypted_data.decode('utf-8')
            logger.debug(f"Successfully decrypted {len(result)} characters")
            return result
        except InvalidToken as e:
            logger.error("Decryption failed: Invalid token or corrupted data")
            raise InvalidToken("Failed to decrypt data. The data may be corrupted or the key is incorrect.")
        except UnicodeDecodeError as e:
            logger.error(f"Decryption failed: Invalid UTF-8 encoding: {e}")
            raise ValueError(f"Decrypted data is not valid UTF-8: {str(e)}")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise Exception(f"Decryption failed: {str(e)}")
    
    def rotate_key(self, new_key_path: str, storage_backend=None) -> None:
        """
        Rotate encryption key and optionally re-encrypt existing data.
        
        This method generates a new encryption key and updates the service to use it.
        If a storage_backend is provided, it will re-encrypt all encrypted entries
        in the storage using the new key.
        
        Args:
            new_key_path: Path where the new key should be stored
            storage_backend: Optional storage backend to re-encrypt existing data.
                           Must have methods: query_entries() and update_entry()
        
        Raises:
            OSError: If the new key file cannot be created
            ValueError: If the new key path is invalid
            Exception: If re-encryption of existing data fails
        
        Example:
            >>> service = EncryptionService("./keys/old.key")
            >>> service.rotate_key("./keys/new.key")
            >>> # Now service uses the new key
            
            >>> # With storage backend for re-encryption
            >>> service.rotate_key("./keys/new.key", storage_backend=storage)
            >>> # All encrypted data is now re-encrypted with new key
        """
        if not new_key_path or not new_key_path.strip():
            raise ValueError("New key path cannot be empty")
        
        new_path = Path(new_key_path)
        
        # Store old cipher for re-encryption
        old_cipher = self.cipher
        old_key_path = self.key_path
        old_key = self.key
        
        # Generate and save new key
        new_key = self._generate_key()
        self._save_key(new_path, new_key)
        
        # Create new cipher
        new_cipher = Fernet(new_key)
        
        # Re-encrypt existing data if storage backend is provided
        if storage_backend is not None:
            logger.info("Re-encrypting existing data with new key")
            try:
                self._re_encrypt_storage_data(old_cipher, new_cipher, storage_backend)
            except Exception as e:
                logger.error(f"Failed to re-encrypt existing data: {e}")
                # Clean up the new key file since rotation failed
                try:
                    new_path.unlink()
                except:
                    pass
                raise Exception(f"Key rotation failed during data re-encryption: {e}")
        
        # Mark old key as inactive and store new key metadata in database
        backend = storage_backend or self.storage_backend
        if backend:
            try:
                # Mark old key as inactive
                self._mark_key_inactive(old_key)
                # Store new key metadata
                self._store_key_metadata(new_key, is_active=True)
            except Exception as e:
                logger.warning(f"Failed to update key metadata in database: {e}")
                # Don't fail the rotation if metadata update fails
        
        # Update service to use new key
        self.key_path = new_path
        self.key = new_key
        self.cipher = new_cipher
        
        logger.info(f"Key rotated from {old_key_path} to {new_path}")
    
    def _re_encrypt_storage_data(self, old_cipher, new_cipher, storage_backend) -> None:
        """
        Re-encrypt all encrypted data in storage with a new key.
        
        Args:
            old_cipher: The old Fernet cipher for decryption
            new_cipher: The new Fernet cipher for encryption
            storage_backend: Storage backend with query_entries() and update_entry() methods
        
        Raises:
            Exception: If re-encryption fails
        """
        import base64
        
        try:
            # Query all entries (in batches to handle large datasets)
            offset = 0
            batch_size = 100
            total_re_encrypted = 0
            
            while True:
                # Get a batch of entries
                entries = storage_backend.query_entries(limit=batch_size, offset=offset)
                
                if not entries:
                    break
                
                for entry in entries:
                    # Check if entry has encrypted context data
                    # The context field may contain encrypted data as base64-encoded strings
                    if isinstance(entry.context, dict):
                        re_encrypted = False
                        new_context = {}
                        
                        for key, value in entry.context.items():
                            # Check if value is base64-encoded encrypted data (string that decodes to bytes)
                            if isinstance(value, str):
                                try:
                                    # Try to decode as base64
                                    encrypted_bytes = base64.b64decode(value)
                                    # Try to decrypt with old key
                                    decrypted = old_cipher.decrypt(encrypted_bytes)
                                    # Re-encrypt with new key
                                    re_encrypted_bytes = new_cipher.encrypt(decrypted)
                                    # Store as base64-encoded string
                                    new_context[key] = base64.b64encode(re_encrypted_bytes).decode('utf-8')
                                    re_encrypted = True
                                except Exception:
                                    # Not encrypted data or decryption failed, keep original value
                                    new_context[key] = value
                            else:
                                new_context[key] = value
                        
                        # Update entry if any fields were re-encrypted
                        if re_encrypted:
                            storage_backend.update_entry(entry.id, {'context': new_context})
                            total_re_encrypted += 1
                            logger.debug(f"Re-encrypted entry {entry.id}")
                
                offset += batch_size
            
            logger.info(f"Successfully re-encrypted {total_re_encrypted} entries")
            
        except Exception as e:
            logger.error(f"Error during data re-encryption: {e}")
            raise
    
    def _load_or_generate_key(self, key_path: Path) -> bytes:
        """
        Load existing key or generate new one.
        
        Args:
            key_path: Path to the key file
        
        Returns:
            Encryption key as bytes
        
        Raises:
            OSError: If the key file cannot be read or created
            ValueError: If the key file contains invalid data
        """
        if key_path.exists():
            return self._load_key(key_path)
        else:
            logger.info(f"Key file not found at {key_path}, generating new key")
            key = self._generate_key()
            self._save_key(key_path, key)
            # Store key metadata in database if storage backend is available
            if self.storage_backend:
                self._store_key_metadata(key, is_active=True)
            return key
    
    def _generate_key(self) -> bytes:
        """
        Generate a new Fernet encryption key.
        
        Returns:
            New encryption key as bytes
        """
        key = Fernet.generate_key()
        logger.debug("Generated new encryption key")
        return key
    
    def _load_key(self, key_path: Path) -> bytes:
        """
        Load encryption key from file.
        
        Args:
            key_path: Path to the key file
        
        Returns:
            Encryption key as bytes
        
        Raises:
            OSError: If the key file cannot be read
            ValueError: If the key file contains invalid data or wrong key format
        """
        try:
            with open(key_path, 'rb') as key_file:
                key = key_file.read()
            
            # Validate key is not empty
            if not key:
                raise ValueError(f"Key file at {key_path} is empty")
            
            # Validate key length (Fernet keys are 44 bytes when base64 encoded)
            if len(key) != 44:
                raise ValueError(f"Invalid key length in {key_path}. Expected 44 bytes, got {len(key)} bytes")
            
            # Validate that it's a valid Fernet key by trying to create a cipher
            try:
                Fernet(key)
            except Exception as e:
                raise ValueError(f"Invalid Fernet key format in {key_path}: {str(e)}")
            
            logger.debug(f"Loaded encryption key from {key_path}")
            return key
        except FileNotFoundError:
            raise OSError(f"Key file not found at {key_path}")
        except ValueError as e:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            raise OSError(f"Failed to load key from {key_path}: {e}")
    
    def _save_key(self, key_path: Path, key: bytes) -> None:
        """
        Save encryption key to file with secure permissions.
        
        Args:
            key_path: Path where the key should be saved
            key: Encryption key as bytes
        
        Raises:
            OSError: If the key file cannot be created or written
        """
        try:
            # Create parent directory if it doesn't exist
            key_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write key to file
            with open(key_path, 'wb') as key_file:
                key_file.write(key)
            
            # Set restrictive permissions (owner read/write only)
            # This is Unix-specific; on Windows it will be ignored
            try:
                os.chmod(key_path, 0o600)
            except (OSError, AttributeError):
                # Windows doesn't support chmod in the same way
                logger.warning(f"Could not set file permissions on {key_path}")
            
            logger.info(f"Saved encryption key to {key_path}")
        except Exception as e:
            raise OSError(f"Failed to save key to {key_path}: {e}")
    
    def _compute_key_hash(self, key: bytes) -> str:
        """
        Compute SHA-256 hash of encryption key for storage.
        
        Args:
            key: Encryption key as bytes
        
        Returns:
            Hexadecimal string representation of the key hash
        """
        return hashlib.sha256(key).hexdigest()
    
    def _store_key_metadata(self, key: bytes, is_active: bool = True) -> None:
        """
        Store encryption key metadata in database.
        
        Args:
            key: Encryption key as bytes
            is_active: Whether this key is currently active
        
        Raises:
            Exception: If database operation fails
        """
        if not self.storage_backend:
            logger.debug("No storage backend available, skipping key metadata storage")
            return
        
        try:
            key_hash = self._compute_key_hash(key)
            now = datetime.now(UTC) if hasattr(datetime, 'UTC') else datetime.now(UTC)
            
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO encryption_keys (key_hash, created_at, is_active)
                    VALUES (?, ?, ?)
                """, (key_hash, now.isoformat(), 1 if is_active else 0))
                conn.commit()
                
            logger.info(f"Stored key metadata in database (hash: {key_hash[:16]}..., active: {is_active})")
        except Exception as e:
            logger.error(f"Failed to store key metadata: {e}")
            raise
    
    def _mark_key_inactive(self, key: bytes) -> None:
        """
        Mark an encryption key as inactive in the database.
        
        Args:
            key: Encryption key as bytes
        
        Raises:
            Exception: If database operation fails
        """
        if not self.storage_backend:
            logger.debug("No storage backend available, skipping key deactivation")
            return
        
        try:
            key_hash = self._compute_key_hash(key)
            now = datetime.now(UTC) if hasattr(datetime, 'UTC') else datetime.now(UTC)
            
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE encryption_keys
                    SET is_active = 0, rotated_at = ?
                    WHERE key_hash = ? AND is_active = 1
                """, (now.isoformat(), key_hash))
                conn.commit()
                
            logger.info(f"Marked key as inactive in database (hash: {key_hash[:16]}...)")
        except Exception as e:
            logger.error(f"Failed to mark key as inactive: {e}")
            raise
    
    def get_active_key_metadata(self) -> Optional[dict]:
        """
        Get metadata for the currently active encryption key.
        
        Returns:
            Dictionary with key metadata or None if not found
        
        Raises:
            Exception: If database operation fails
        """
        if not self.storage_backend:
            logger.debug("No storage backend available, cannot retrieve key metadata")
            return None
        
        try:
            key_hash = self._compute_key_hash(self.key)
            
            with self.storage_backend.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, key_hash, created_at, rotated_at, is_active
                    FROM encryption_keys
                    WHERE key_hash = ? AND is_active = 1
                """, (key_hash,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    'id': row['id'],
                    'key_hash': row['key_hash'],
                    'created_at': row['created_at'],
                    'rotated_at': row['rotated_at'],
                    'is_active': bool(row['is_active'])
                }
        except Exception as e:
            logger.error(f"Failed to retrieve key metadata: {e}")
            raise
