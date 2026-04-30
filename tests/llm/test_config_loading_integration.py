"""
Integration tests for configuration loading from environment variables.

Tests environment variable loading, validation, and provider instantiation.

**Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.7**

Feature: gemini-provider-integration
"""

import os
import pytest
from unittest.mock import patch

from luma.core.llm.config import load_llm_config_from_env, LLMConfig
from luma.core.llm.providers.provider_factory import ProviderFactory


# ---------------------------------------------------------------------------
# Test: Environment variable loading for Gemini
# ---------------------------------------------------------------------------

class TestGeminiConfigLoading:
    """Test configuration loading for Gemini provider."""
    
    def test_load_gemini_config_with_all_env_vars(self):
        """
        Load Gemini configuration with all environment variables set.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
        """
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-api-key-12345",
            "GEMINI_MODEL": "gemini-pro",
            "GEMINI_TIMEOUT": "45.0",
            "GEMINI_MAX_TOKENS": "2048",
            "GEMINI_TEMPERATURE": "0.8",
            "GEMINI_LOG_PROMPTS": "true"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_llm_config_from_env()
        
        # Verify LLMConfig fields
        assert config.provider_name == "gemini"
        assert config.api_key == "test-api-key-12345"
        assert config.model == "gemini-pro"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.timeout_seconds == 45.0
        
        # Verify provider_config
        assert config.provider_config["api_key"] == "test-api-key-12345"
        assert config.provider_config["model"] == "gemini-pro"
        assert config.provider_config["timeout"] == 45.0
        assert config.provider_config["max_tokens"] == 2048
        assert config.provider_config["temperature"] == 0.8
        assert config.provider_config["log_prompts"] is True
    
    def test_load_gemini_config_with_defaults(self):
        """
        Load Gemini configuration with only required env vars (use defaults).
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
        """
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-api-key-67890"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_llm_config_from_env()
        
        # Verify defaults are applied
        assert config.provider_name == "gemini"
        assert config.api_key == "test-api-key-67890"
        assert config.model == "gemini-2.5-flash"  # Default
        assert config.temperature == 0.4  # Default
        assert config.max_tokens == 1024  # Default
        assert config.timeout_seconds == 30.0  # Default
        
        # Verify provider_config defaults
        assert config.provider_config["model"] == "gemini-2.5-flash"
        assert config.provider_config["timeout"] == 30.0
        assert config.provider_config["max_tokens"] == 1024
        assert config.provider_config["temperature"] == 0.4
        assert config.provider_config["log_prompts"] is False  # Default
    
    def test_load_gemini_config_missing_api_key(self):
        """
        Loading Gemini config without API key raises ValueError.
        
        **Validates: Requirement 15.7**
        """
        env_vars = {
            "LLM_PROVIDER": "gemini"
            # GEMINI_API_KEY is missing
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                load_llm_config_from_env()
        
        assert "GEMINI_API_KEY" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()
    
    def test_load_config_default_provider(self):
        """
        Loading config without LLM_PROVIDER defaults to 'gemini'.
        
        **Validates: Requirement 15.1**
        """
        env_vars = {
            # LLM_PROVIDER not set (should default to "gemini")
            "GEMINI_API_KEY": "test-default-key"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_llm_config_from_env()
        
        assert config.provider_name == "gemini"
        assert config.api_key == "test-default-key"


# ---------------------------------------------------------------------------
# Test: Environment variable loading for Mock provider
# ---------------------------------------------------------------------------

class TestMockConfigLoading:
    """Test configuration loading for Mock provider."""
    
    def test_load_mock_config(self):
        """
        Load Mock provider configuration.
        
        **Validates: Requirements 15.1, 15.2, 15.3**
        """
        env_vars = {
            "LLM_PROVIDER": "mock"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_llm_config_from_env()
        
        # Verify mock provider config
        assert config.provider_name == "mock"
        assert config.api_key == "mock-key"
        assert config.model == "default-model"
        
        # Verify provider_config for mock
        assert config.provider_config["responses"] == []
        assert config.provider_config["delay"] == 0.0
        assert config.provider_config["error_mode"] is None


# ---------------------------------------------------------------------------
# Test: Provider instantiation from config
# ---------------------------------------------------------------------------

class TestProviderInstantiation:
    """Test provider instantiation from loaded configuration."""
    
    def test_instantiate_gemini_provider_from_config(self):
        """
        Instantiate Gemini provider from loaded configuration.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.5**
        """
        from unittest.mock import MagicMock
        from luma.core.structured_logger import StructuredLogger
        
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-instantiate-key"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_llm_config_from_env()
        
        # Create mock logger
        logger = MagicMock(spec=StructuredLogger)
        
        # Instantiate provider using factory
        provider = ProviderFactory.create(
            provider_name=config.provider_name,
            config=config.provider_config,
            logger=logger
        )
        
        # Verify provider is correct type
        from luma.core.llm.providers.gemini_provider import GeminiProvider
        assert isinstance(provider, GeminiProvider)
    
    def test_instantiate_mock_provider_from_config(self):
        """
        Instantiate Mock provider from loaded configuration.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.5**
        """
        env_vars = {
            "LLM_PROVIDER": "mock"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_llm_config_from_env()
        
        # Instantiate provider using factory
        provider = ProviderFactory.create(
            provider_name=config.provider_name,
            config=config.provider_config
        )
        
        # Verify provider is correct type
        from luma.core.llm.providers.mock_provider import MockProvider
        assert isinstance(provider, MockProvider)
    
    def test_provider_instantiation_with_logger(self):
        """
        Instantiate provider with logger from configuration.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.5, 15.6**
        """
        from unittest.mock import MagicMock
        from luma.core.structured_logger import StructuredLogger
        
        env_vars = {
            "LLM_PROVIDER": "mock"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_llm_config_from_env()
        
        # Create mock logger
        logger = MagicMock(spec=StructuredLogger)
        
        # Instantiate provider with logger
        provider = ProviderFactory.create(
            provider_name=config.provider_name,
            config=config.provider_config,
            logger=logger
        )
        
        # Verify provider was created
        assert provider is not None


# ---------------------------------------------------------------------------
# Test: Configuration validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_provider_name_raises_error(self):
        """
        Loading config with invalid provider name raises ValueError.
        
        **Validates: Requirement 15.7**
        """
        env_vars = {
            "LLM_PROVIDER": "invalid-provider-name"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                load_llm_config_from_env()
        
        assert "Unknown provider" in str(exc_info.value)
        assert "invalid-provider-name" in str(exc_info.value)
    
    def test_numeric_env_var_parsing(self):
        """
        Numeric environment variables are parsed correctly.
        
        **Validates: Requirements 15.3, 15.4**
        """
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_TIMEOUT": "60.5",
            "GEMINI_MAX_TOKENS": "4096",
            "GEMINI_TEMPERATURE": "0.9"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_llm_config_from_env()
        
        # Verify numeric values are parsed correctly
        assert isinstance(config.provider_config["timeout"], float)
        assert config.provider_config["timeout"] == 60.5
        
        assert isinstance(config.provider_config["max_tokens"], int)
        assert config.provider_config["max_tokens"] == 4096
        
        assert isinstance(config.provider_config["temperature"], float)
        assert config.provider_config["temperature"] == 0.9
    
    def test_boolean_env_var_parsing(self):
        """
        Boolean environment variables are parsed correctly.
        
        **Validates: Requirements 15.3, 15.4**
        """
        # Test "true" value
        env_vars_true = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_LOG_PROMPTS": "true"
        }
        
        with patch.dict(os.environ, env_vars_true, clear=False):
            config_true = load_llm_config_from_env()
        
        assert config_true.provider_config["log_prompts"] is True
        
        # Test "false" value
        env_vars_false = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_LOG_PROMPTS": "false"
        }
        
        with patch.dict(os.environ, env_vars_false, clear=False):
            config_false = load_llm_config_from_env()
        
        assert config_false.provider_config["log_prompts"] is False
        
        # Test other values (should be false)
        env_vars_other = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_LOG_PROMPTS": "yes"
        }
        
        with patch.dict(os.environ, env_vars_other, clear=False):
            config_other = load_llm_config_from_env()
        
        assert config_other.provider_config["log_prompts"] is False


# ---------------------------------------------------------------------------
# Test: End-to-end configuration flow
# ---------------------------------------------------------------------------

class TestEndToEndConfigFlow:
    """Test complete configuration flow from env vars to provider."""
    
    def test_complete_flow_gemini(self):
        """
        Complete flow: env vars → config → provider → client.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.5, 15.6**
        """
        from unittest.mock import MagicMock
        from luma.core.llm.llm_client import ProviderLLMClient
        from luma.core.structured_logger import StructuredLogger
        
        env_vars = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-flow-key",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_TEMPERATURE": "0.5"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            # Step 1: Load config from environment
            config = load_llm_config_from_env()
            
            # Step 2: Create provider from config
            logger = MagicMock(spec=StructuredLogger)
            provider = ProviderFactory.create(
                provider_name=config.provider_name,
                config=config.provider_config,
                logger=logger
            )
            
            # Step 3: Create LLM client with provider
            client = ProviderLLMClient(
                provider=provider,
                config=config,
                logger=logger
            )
            
            # Verify client was created successfully
            assert client is not None
            assert client._provider is provider
            assert client._config is config
    
    def test_complete_flow_mock(self):
        """
        Complete flow with Mock provider.
        
        **Validates: Requirements 15.1, 15.2, 15.3, 15.5, 15.6**
        """
        from unittest.mock import MagicMock
        from luma.core.llm.llm_client import ProviderLLMClient
        from luma.core.structured_logger import StructuredLogger
        
        env_vars = {
            "LLM_PROVIDER": "mock"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Step 1: Load config from environment
            config = load_llm_config_from_env()
            
            # Step 2: Create provider from config
            provider = ProviderFactory.create(
                provider_name=config.provider_name,
                config=config.provider_config
            )
            
            # Step 3: Create LLM client with provider
            logger = MagicMock(spec=StructuredLogger)
            client = ProviderLLMClient(
                provider=provider,
                config=config,
                logger=logger
            )
            
            # Verify client was created successfully
            assert client is not None
            assert client._provider is provider
            assert client._config is config
