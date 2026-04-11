"""Tests for configuration management."""

import os
import pytest
from pathlib import Path
from luma_memory.config import MemoryModuleConfig


class TestMemoryModuleConfig:
    """Test suite for MemoryModuleConfig."""
    
    def test_default_config_values(self):
        """Test that default configuration values are set correctly."""
        config = MemoryModuleConfig()
        
        # Storage settings
        assert config.db_path == "./data/luma_memory.db"
        assert config.cache_size == 1000
        assert config.max_storage_size_mb == 1000
        
        # API settings
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.api_workers == 4
        
        # Monitoring settings
        assert config.enable_metrics is True
        assert config.log_level == "INFO"
    
    def test_load_config_from_env_variables(self, monkeypatch):
        """Test loading configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("DB_PATH", "/custom/path/db.sqlite")
        monkeypatch.setenv("CACHE_SIZE", "2000")
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        config = MemoryModuleConfig()
        
        assert config.db_path == "/custom/path/db.sqlite"
        assert config.cache_size == 2000
        assert config.api_port == 9000
        assert config.log_level == "DEBUG"
    
    def test_load_config_method(self):
        """Test the load_config class method."""
        config = MemoryModuleConfig.load_config()
        
        assert isinstance(config, MemoryModuleConfig)
        assert config.db_path is not None
    
    def test_load_config_with_custom_env_file(self, tmp_path):
        """Test loading configuration from a custom .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env.test"
        env_file.write_text(
            "DB_PATH=/test/path/db.sqlite\n"
            "CACHE_SIZE=3000\n"
            "API_PORT=7000\n"
        )
        
        config = MemoryModuleConfig.load_config(env_file=str(env_file))
        
        assert config.db_path == "/test/path/db.sqlite"
        assert config.cache_size == 3000
        assert config.api_port == 7000
    
    def test_get_env_var(self, monkeypatch):
        """Test the get_env_var helper method."""
        config = MemoryModuleConfig()
        
        monkeypatch.setenv("TEST_VAR", "test_value")
        
        assert config.get_env_var("TEST_VAR") == "test_value"
        assert config.get_env_var("test_var") == "test_value"  # Case insensitive
        assert config.get_env_var("NONEXISTENT", "default") == "default"
    
    def test_is_env_loaded(self, tmp_path, monkeypatch):
        """Test checking if .env file exists."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        config = MemoryModuleConfig()
        
        # No .env file exists
        assert config.is_env_loaded() is False
        
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("DB_PATH=/test/path\n")
        
        assert config.is_env_loaded() is True
    
    def test_env_priority_over_defaults(self, monkeypatch):
        """Test that environment variables take priority over defaults."""
        monkeypatch.setenv("CACHE_SIZE", "5000")
        
        config = MemoryModuleConfig()
        
        # Environment variable should override default
        assert config.cache_size == 5000
        assert config.cache_size != 1000  # Not the default
    
    def test_type_conversion(self, monkeypatch):
        """Test that environment variables are converted to correct types."""
        monkeypatch.setenv("CACHE_SIZE", "2500")
        monkeypatch.setenv("ENABLE_METRICS", "false")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.75")
        
        config = MemoryModuleConfig()
        
        assert isinstance(config.cache_size, int)
        assert config.cache_size == 2500
        
        assert isinstance(config.enable_metrics, bool)
        assert config.enable_metrics is False
        
        assert isinstance(config.similarity_threshold, float)
        assert config.similarity_threshold == 0.75
    
    def test_invalid_cache_size_raises_error(self, monkeypatch):
        """Test that invalid cache_size raises a clear error."""
        monkeypatch.setenv("CACHE_SIZE", "-100")
        
        with pytest.raises(ValueError, match="cache_size must be a positive integer"):
            MemoryModuleConfig()
    
    def test_invalid_similarity_threshold_raises_error(self, monkeypatch):
        """Test that invalid similarity_threshold raises a clear error."""
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "1.5")
        
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            MemoryModuleConfig()
    
    def test_invalid_log_level_raises_error(self, monkeypatch):
        """Test that invalid log_level raises a clear error."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        
        with pytest.raises(ValueError, match="log_level must be one of"):
            MemoryModuleConfig()
    
    def test_invalid_retention_days_raises_error(self, monkeypatch):
        """Test that invalid retention days configuration raises a clear error."""
        monkeypatch.setenv("RETENTION_DAYS_RAW", "100")
        monkeypatch.setenv("RETENTION_DAYS_SUMMARY", "50")
        
        with pytest.raises(ValueError, match="retention_days_summary.*must be >=.*retention_days_raw"):
            MemoryModuleConfig()
    
    def test_empty_api_host_raises_error(self, monkeypatch):
        """Test that empty api_host raises a clear error."""
        monkeypatch.setenv("API_HOST", "")
        
        with pytest.raises(ValueError, match="api_host cannot be empty"):
            MemoryModuleConfig()
    
    def test_valid_log_levels(self, monkeypatch):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            monkeypatch.setenv("LOG_LEVEL", level)
            config = MemoryModuleConfig()
            assert config.log_level == level
    
    def test_log_level_case_insensitive(self, monkeypatch):
        """Test that log level is case insensitive."""
        monkeypatch.setenv("LOG_LEVEL", "debug")
        config = MemoryModuleConfig()
        assert config.log_level == "DEBUG"

    def test_valid_log_formats(self, monkeypatch):
        """Test that all valid log formats are accepted."""
        valid_formats = ["json", "human"]
        
        for fmt in valid_formats:
            monkeypatch.setenv("LOG_FORMAT", fmt)
            config = MemoryModuleConfig()
            assert config.log_format == fmt
    
    def test_log_format_case_insensitive(self, monkeypatch):
        """Test that log format is case insensitive."""
        monkeypatch.setenv("LOG_FORMAT", "JSON")
        config = MemoryModuleConfig()
        assert config.log_format == "json"
    
    def test_invalid_log_format_raises_error(self, monkeypatch):
        """Test that invalid log format raises validation error."""
        monkeypatch.setenv("LOG_FORMAT", "xml")
        
        with pytest.raises(ValueError, match="log_format must be one of"):
            MemoryModuleConfig()
    
    def test_log_file_optional(self, monkeypatch):
        """Test that log_file is optional and defaults to None."""
        config = MemoryModuleConfig()
        assert config.log_file is None
    
    def test_log_file_can_be_set(self, monkeypatch):
        """Test that log_file can be configured."""
        monkeypatch.setenv("LOG_FILE", "./logs/test.log")
        config = MemoryModuleConfig()
        assert config.log_file == "./logs/test.log"
