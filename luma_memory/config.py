"""
Configuration management for Luma Memory Module.

This module provides configuration settings using Pydantic Settings,
supporting environment variables and .env file loading.

For detailed configuration documentation, see CONFIG_GUIDE.md
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Optional


class MemoryModuleConfig(BaseSettings):
    """
    Configuration settings for the Luma Memory Module.
    
    Settings can be loaded from:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)
    
    All settings have sensible defaults for local development.
    
    Environment variables should be prefixed with the setting name in uppercase.
    For example: DB_PATH, CACHE_SIZE, API_PORT, etc.
    """
    
    # Storage settings
    db_path: str = "./data/luma_memory.db"
    cache_size: int = 1000
    max_storage_size_mb: int = 1000
    
    # Summarization settings
    summarization_threshold: int = 1000  # Number of entries before triggering summarization
    similarity_threshold: float = 0.8
    retention_days_raw: int = 30
    retention_days_summary: int = 365
    
    # Encryption settings
    encryption_key_path: str = "./keys/encryption.key"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Performance settings
    query_timeout_ms: int = 200
    connection_pool_size: int = 10
    
    # Monitoring settings
    enable_metrics: bool = True
    log_level: str = "INFO"
    log_format: str = "human"  # "json" for structured JSON, "human" for readable
    log_file: Optional[str] = None  # Optional path to log file
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB default for log rotation
    log_backup_count: int = 5  # Number of backup log files to keep
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("cache_size", "max_storage_size_mb", "summarization_threshold", 
                     "retention_days_raw", "retention_days_summary", "api_port", 
                     "api_workers", "query_timeout_ms", "connection_pool_size",
                     "log_max_bytes", "log_backup_count")
    @classmethod
    def validate_positive_integers(cls, v: int, info) -> int:
        """Validate that integer settings are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer, got {v}")
        return v
    
    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate that similarity threshold is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {v}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is a valid option."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got '{v}'. "
                f"Valid options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        return v_upper
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate that log format is a valid option."""
        valid_formats = ["json", "human"]
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(
                f"log_format must be one of {valid_formats}, got '{v}'. "
                f"Valid options: json (structured JSON), human (readable)"
            )
        return v_lower
    
    @field_validator("api_host")
    @classmethod
    def validate_api_host(cls, v: str) -> str:
        """Validate that API host is not empty."""
        if not v or not v.strip():
            raise ValueError("api_host cannot be empty")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_retention_days(self) -> "MemoryModuleConfig":
        """Validate that summary retention is longer than raw retention."""
        if self.retention_days_summary < self.retention_days_raw:
            raise ValueError(
                f"retention_days_summary ({self.retention_days_summary}) must be >= "
                f"retention_days_raw ({self.retention_days_raw}). "
                f"Summaries should be retained longer than raw entries."
            )
        return self
    
    @classmethod
    def load_config(cls, env_file: Optional[str] = None) -> "MemoryModuleConfig":
        """
        Load configuration from environment variables and .env file.
        
        Args:
            env_file: Optional path to .env file. If not provided, looks for .env in current directory.
        
        Returns:
            MemoryModuleConfig: Loaded configuration instance
        
        Raises:
            ValueError: If configuration values are invalid
        
        Example:
            >>> config = MemoryModuleConfig.load_config()
            >>> config = MemoryModuleConfig.load_config(env_file=".env.production")
        """
        if env_file:
            # Create a new config with custom env_file
            class CustomConfig(cls):
                model_config = SettingsConfigDict(
                    env_file=env_file,
                    env_file_encoding="utf-8",
                    case_sensitive=False,
                    extra="ignore"
                )
            return CustomConfig()
        
        return cls()
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get an environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
        
        Returns:
            Environment variable value or default
        """
        return os.environ.get(key.upper(), default)
    
    def is_env_loaded(self) -> bool:
        """
        Check if .env file exists and can be loaded.
        
        Returns:
            bool: True if .env file exists, False otherwise
        """
        env_file = self.model_config.get("env_file", ".env")
        return Path(env_file).exists() if env_file else False
