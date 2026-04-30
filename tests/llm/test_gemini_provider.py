"""
Unit tests for GeminiProvider.

Tests the GeminiProvider class structure, initialization, configuration validation,
and helper methods. The generate() method will be tested in subsequent tasks.
"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch

# Mock google.generativeai before importing GeminiProvider
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError
from luma.core.structured_logger import StructuredLogger


class TestGeminiProviderStructure:
    """Test GeminiProvider class structure and inheritance."""
    
    def test_gemini_provider_inherits_from_llm_provider(self):
        """Test that GeminiProvider is a subclass of LLMProvider."""
        assert issubclass(GeminiProvider, LLMProvider)
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_gemini_provider_can_be_instantiated_with_valid_config(
        self, mock_model, mock_configure
    ):
        """Test GeminiProvider can be instantiated with valid configuration."""
        config = {
            "api_key": "test-api-key-12345",
            "model": "gemini-2.5-flash",
            "timeout": 30.0
        }
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        
        assert provider is not None
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, LLMProvider)


class TestGeminiProviderInitialization:
    """Test GeminiProvider initialization and configuration."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_initialization_with_minimal_config(self, mock_model, mock_configure):
        """Test initialization with only required fields uses defaults."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        
        assert provider._api_key == "test-key"
        assert provider._model_name == "gemini-2.5-flash"  # default
        assert provider._timeout == 30.0  # default
        assert provider._log_prompts is False  # default
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_initialization_with_full_config(self, mock_model, mock_configure):
        """Test initialization with all configuration fields."""
        config = {
            "api_key": "test-key-67890",
            "model": "gemini-pro",
            "timeout": 60.0,
            "max_tokens": 2048,
            "temperature": 0.7,
            "log_prompts": True
        }
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        
        assert provider._api_key == "test-key-67890"
        assert provider._model_name == "gemini-pro"
        assert provider._timeout == 60.0
        assert provider._log_prompts is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_initialization_configures_gemini_sdk(self, mock_model, mock_configure):
        """Test that initialization configures Gemini SDK with API key."""
        config = {"api_key": "test-api-key"}
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        
        # Verify genai.configure was called with the API key
        mock_configure.assert_called_once_with(api_key="test-api-key")
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_initialization_creates_generative_model(self, mock_model, mock_configure):
        """Test that initialization creates a GenerativeModel instance."""
        config = {"api_key": "test-key", "model": "gemini-2.5-flash"}
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        
        # Verify GenerativeModel was instantiated with the model name
        mock_model.assert_called_once_with("gemini-2.5-flash")
        assert provider._model == mock_model.return_value


class TestGeminiProviderConfigValidation:
    """Test GeminiProvider configuration validation."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_validate_config_accepts_valid_config(self, mock_model, mock_configure):
        """Test validate_config returns True for valid configuration."""
        config = {"api_key": "valid-key"}
        logger = Mock(spec=StructuredLogger)
        
        provider = GeminiProvider(config, logger)
        assert provider.validate_config(config) is True
    
    def test_validate_config_raises_on_missing_api_key(self):
        """Test validate_config raises ValueError when api_key is missing."""
        config = {"model": "gemini-pro"}
        
        with pytest.raises(ValueError, match="Gemini provider requires 'api_key'"):
            # validate_config is called in __init__
            logger = Mock(spec=StructuredLogger)
            with patch('luma.core.llm.providers.gemini_provider.genai.configure'):
                with patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel'):
                    GeminiProvider(config, logger)
    
    def test_validate_config_raises_on_empty_api_key(self):
        """Test validate_config raises ValueError when api_key is empty."""
        config = {"api_key": ""}
        
        with pytest.raises(ValueError, match="Gemini provider requires 'api_key'"):
            logger = Mock(spec=StructuredLogger)
            with patch('luma.core.llm.providers.gemini_provider.genai.configure'):
                with patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel'):
                    GeminiProvider(config, logger)
    
    def test_validate_config_raises_on_whitespace_only_api_key(self):
        """Test validate_config raises ValueError when api_key is whitespace."""
        config = {"api_key": "   "}
        
        with pytest.raises(ValueError, match="non-empty, non-whitespace string"):
            logger = Mock(spec=StructuredLogger)
            with patch('luma.core.llm.providers.gemini_provider.genai.configure'):
                with patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel'):
                    GeminiProvider(config, logger)


class TestGeminiProviderHelperMethods:
    """Test GeminiProvider helper methods."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_masks_long_key(self, mock_model, mock_configure):
        """Test _mask_api_key masks all but last 4 characters."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("abcdefghijklmnop")
        
        assert masked == "************mnop"
        assert len(masked) == 16
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_masks_short_key(self, mock_model, mock_configure):
        """Test _mask_api_key returns **** for keys <= 4 characters."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        assert provider._mask_api_key("abc") == "****"
        assert provider._mask_api_key("abcd") == "****"
        assert provider._mask_api_key("a") == "****"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_shows_last_4_chars(self, mock_model, mock_configure):
        """Test _mask_api_key shows last 4 characters for keys > 4 chars."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("my-secret-key-1234")
        
        assert masked.endswith("1234")
        assert "secret" not in masked


class TestGeminiProviderGenerate:
    """Test GeminiProvider generate method."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_returns_dict_with_required_keys(self, mock_model_class, mock_configure):
        """Test generate() returns a dict with all required keys."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)

        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "hello"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 10
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance

        result = provider.generate("test prompt", {})
        assert "text" in result
        assert "model" in result
        assert "provider" in result
        assert result["provider"] == "gemini"
