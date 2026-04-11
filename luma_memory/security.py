"""
Security utilities for encryption and data protection.
"""

from cryptography.fernet import Fernet
from typing import Optional
import base64
import hashlib


class SecurityManager:
    """Handles encryption and security for sensitive memory data."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            encryption_key: Base encryption key (will be hashed to proper format)
        """
        if encryption_key:
            # Hash the key to ensure it's the right length for Fernet
            key_bytes = hashlib.sha256(encryption_key.encode()).digest()
            self.key = base64.urlsafe_b64encode(key_bytes)
            self.cipher = Fernet(self.key)
        else:
            # Generate a new key if none provided
            self.key = Fernet.generate_key()
            self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt a string.
        
        Args:
            data: Plain text to encrypt
        
        Returns:
            Encrypted string (base64 encoded)
        """
        encrypted_bytes = self.cipher.encrypt(data.encode())
        return encrypted_bytes.decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt a string.
        
        Args:
            encrypted_data: Encrypted string (base64 encoded)
        
        Returns:
            Decrypted plain text
        """
        decrypted_bytes = self.cipher.decrypt(encrypted_data.encode())
        return decrypted_bytes.decode()
    
    def get_key(self) -> str:
        """Get the encryption key (for backup/storage)."""
        return self.key.decode()
    
    @staticmethod
    def sanitize_log_content(content: str, max_length: int = 100) -> str:
        """
        Sanitize content for logging (truncate and remove sensitive info).
        
        Args:
            content: Content to sanitize
            max_length: Maximum length to keep
        
        Returns:
            Sanitized content
        """
        # Truncate
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        # Remove potential sensitive patterns (basic implementation)
        # In production, use more sophisticated PII detection
        sensitive_patterns = [
            "password", "token", "api_key", "secret",
            "ssn", "credit_card", "email"
        ]
        
        lower_content = content.lower()
        for pattern in sensitive_patterns:
            if pattern in lower_content:
                return "[REDACTED - Sensitive Content]"
        
        return content
