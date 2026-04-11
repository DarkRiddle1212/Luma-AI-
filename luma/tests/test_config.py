"""
Unit tests for configuration module.

Tests Settings validation, environment variable loading, and configuration error handling.
"""

import pytest
import os
from pydantic import ValidationError
from luma.config import Settings


class TestSettingsValidation:
    """Test Settings class validation with valid values."""
    
    def test_settings_with_default_values(self):
        """Test that Settings can be created with default values."""
        settings = Settings()
        
        assert settings.database_url == "sqlite:///./luma.db"
        assert settings.api_prefix == "/api/v1"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.log_level == "INFO"
        assert settings.environment == "development"
    
    def test_settings_with_valid_log_levels(self):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level.upper()
    
    def test_settings_with_lowercase_log_level(self):
        """Test that lowercase log levels are converted to uppercase."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
        
        settings = Settings(log_level="info")
        assert settings.log_level == "INFO"
    
    def test_settings_with_valid_port_range(self):
        """Test that valid port numbers are accepted."""
        # Test minimum valid port
        settings = Settings(api_port=1)
        assert settings.api_port == 1
        
        # Test common port
        settings = Settings(api_port=8080)
        assert settings.api_port == 8080
        
        # Test maximum valid port
        settings = Settings(api_port=65535)
        assert settings.api_port == 65535
    
    def test_settings_with_custom_values(self):
        """Test that Settings accepts custom valid values."""
        settings = Settings(
            database_url="sqlite:///./test.db",
            api_prefix="/api/v2",
            api_host="127.0.0.1",
            api_port=9000,
            log_level="DEBUG",
            environment="production"
        )
        
        assert settings.database_url == "sqlite:///./test.db"
        assert settings.api_prefix == "/api/v2"
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 9000
        assert settings.log_level == "DEBUG"
        assert settings.environment == "production"


class TestSettingsInvalidLogLevel:
    """Test Settings validation with invalid log_level."""
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")
        
        # Verify error message contains expected information
        error_str = str(exc_info.value)
        assert "log_level" in error_str.lower()
    
    def test_invalid_log_level_error_message(self):
        """Test that error message lists valid log levels."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="TRACE")
        
        error_str = str(exc_info.value)
        # Check that valid levels are mentioned in error
        assert any(level in error_str for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    def test_empty_log_level_raises_error(self):
        """Test that empty log level raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(log_level="")
    
    def test_numeric_log_level_raises_error(self):
        """Test that numeric log level raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(log_level="123")


class TestSettingsInvalidPort:
    """Test Settings validation with invalid api_port."""
    
    def test_port_zero_raises_error(self):
        """Test that port 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(api_port=0)
        
        error_str = str(exc_info.value)
        assert "api_port" in error_str.lower()
    
    def test_negative_port_raises_error(self):
        """Test that negative port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(api_port=-1)
        
        error_str = str(exc_info.value)
        assert "api_port" in error_str.lower()
    
    def test_port_above_max_raises_error(self):
        """Test that port above 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(api_port=65536)
        
        error_str = str(exc_info.value)
        assert "api_port" in error_str.lower()
    
    def test_port_far_above_max_raises_error(self):
        """Test that port far above maximum raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(api_port=100000)


class TestEnvironmentVariableLoading:
    """Test Settings loading from environment variables."""
    
    def test_load_database_url_from_env(self, monkeypatch):
        """Test that database_url can be loaded from environment variable."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./custom.db")
        settings = Settings()
        assert settings.database_url == "sqlite:///./custom.db"
    
    def test_load_api_port_from_env(self, monkeypatch):
        """Test that api_port can be loaded from environment variable."""
        monkeypatch.setenv("API_PORT", "9000")
        settings = Settings()
        assert settings.api_port == 9000
    
    def test_load_log_level_from_env(self, monkeypatch):
        """Test that log_level can be loaded from environment variable."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"
    
    def test_load_environment_from_env(self, monkeypatch):
        """Test that environment can be loaded from environment variable."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings()
        assert settings.environment == "production"
    
    def test_load_multiple_settings_from_env(self, monkeypatch):
        """Test that multiple settings can be loaded from environment variables."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./prod.db")
        monkeypatch.setenv("API_PORT", "8080")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("ENVIRONMENT", "production")
        
        settings = Settings()
        
        assert settings.database_url == "sqlite:///./prod.db"
        assert settings.api_port == 8080
        assert settings.log_level == "WARNING"
        assert settings.environment == "production"
    
    def test_env_validation_still_applies(self, monkeypatch):
        """Test that validation still applies to environment variables."""
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        
        with pytest.raises(ValidationError):
            Settings()
    
    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case-insensitive."""
        monkeypatch.setenv("log_level", "error")
        settings = Settings()
        assert settings.log_level == "ERROR"
