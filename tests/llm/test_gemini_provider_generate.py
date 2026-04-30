"""
Unit tests for GeminiProvider generate() method.

Tests the generate() method implementation including option extraction,
response normalization, error handling, and logging.
"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch

# Mock google.generativeai before importing GeminiProvider
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.llm.providers.provider_interface import ProviderError
from luma.core.structured_logger import StructuredLogger


class TestGeminiProviderGenerate:
    """Test GeminiProvider generate method."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_extracts_options_with_fallbacks(self, mock_model_class, mock_configure):
        """Test generate() extracts options with fallbacks to config defaults."""
        config = {
            "api_key": "test-key",
            "model": "gemini-2.5-flash",
            "temperature": 0.5,
            "max_tokens": 2048
        }
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance and its response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        # Call generate with minimal options
        result = provider.generate("test prompt", {})
        
        # Verify generate_content was called with config defaults
        call_args = mock_model_instance.generate_content.call_args
        assert call_args[0][0] == "test prompt"
        # Verify the result uses config defaults
        assert result["model"] == "gemini-2.5-flash"
        assert result["provider"] == "gemini"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_returns_normalized_response(self, mock_model_class, mock_configure):
        """Test generate() returns normalized response dictionary."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance and its response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        result = provider.generate("test prompt", {})
        
        # Verify normalized response format
        assert result["text"] == "Generated text"
        assert result["model"] == "gemini-2.5-flash"
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["provider"] == "gemini"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_logs_request_lifecycle(self, mock_model_class, mock_configure):
        """Test generate() logs request start, success, and error events."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance and its response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        result = provider.generate("test prompt", {"request_id": "req-123"})
        
        # Verify logging calls
        assert logger.log.call_count == 2  # start and success
        
        # Check start log
        start_call = logger.log.call_args_list[0]
        assert start_call[0][0] == "provider_request_start"
        assert start_call[0][1]["request_id"] == "req-123"
        assert start_call[0][1]["provider"] == "gemini"
        
        # Check success log
        success_call = logger.log.call_args_list[1]
        assert success_call[0][0] == "provider_request_success"
        assert success_call[0][1]["request_id"] == "req-123"
        assert success_call[0][1]["prompt_tokens"] == 10
        assert success_call[0][1]["completion_tokens"] == 20
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_handles_api_errors(self, mock_model_class, mock_configure):
        """Test generate() maps API errors to ProviderError."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance to raise an error
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API Error 500")
        provider._model = mock_model_instance
        
        # Verify ProviderError is raised
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("test prompt", {})
        
        assert "server error" in str(exc_info.value)
        assert exc_info.value.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_handles_empty_response(self, mock_model_class, mock_configure):
        """Test generate() raises ProviderError for empty responses."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance with empty candidates
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = []
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        # Verify ProviderError is raised
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("test prompt", {})
        
        assert "no candidates returned" in str(exc_info.value)
        assert exc_info.value.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_generate_uses_model_override(self, mock_model_class, mock_configure):
        """Test generate() uses model from options when provided."""
        config = {"api_key": "test-key", "model": "gemini-2.5-flash"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance and its response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        
        # Mock GenerativeModel to return our mock instance
        mock_model_class.return_value = mock_model_instance
        mock_model_instance.generate_content.return_value = mock_response
        
        # Call generate with model override
        result = provider.generate("test prompt", {"model": "gemini-pro"})
        
        # Verify new model was created with override name
        assert mock_model_class.call_count == 2  # once in __init__, once for override
        assert mock_model_class.call_args_list[1][0][0] == "gemini-pro"
        
        # Verify response uses override model name
        assert result["model"] == "gemini-pro"


class TestGeminiProviderResponseNormalization:
    """Test response normalization with various Gemini response structures."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_normalize_response_with_complete_metadata(self, mock_model_class, mock_configure):
        """Test normalization with complete usage metadata."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock response with complete metadata
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Complete response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 100
        
        result = provider._normalize_response(mock_response, "gemini-2.5-flash", "req-1")
        
        assert result["text"] == "Complete response"
        assert result["model"] == "gemini-2.5-flash"
        assert result["prompt_tokens"] == 50
        assert result["completion_tokens"] == 100
        assert result["provider"] == "gemini"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_normalize_response_without_usage_metadata(self, mock_model_class, mock_configure):
        """Test normalization when usage metadata is missing."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock response without usage metadata
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Response text"
        mock_response.usage_metadata = None
        
        result = provider._normalize_response(mock_response, "gemini-2.5-flash", "req-2")
        
        assert result["text"] == "Response text"
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_normalize_response_empty_candidates(self, mock_model_class, mock_configure):
        """Test normalization raises error for empty candidates."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock response with empty candidates
        mock_response = Mock()
        mock_response.candidates = []
        
        with pytest.raises(ProviderError) as exc_info:
            provider._normalize_response(mock_response, "gemini-2.5-flash", "req-3")
        
        assert "no candidates returned" in str(exc_info.value)
        assert exc_info.value.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_normalize_response_missing_content(self, mock_model_class, mock_configure):
        """Test normalization raises error when content is missing."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock response with candidate but no content
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = None
        
        with pytest.raises(ProviderError) as exc_info:
            provider._normalize_response(mock_response, "gemini-2.5-flash", "req-4")
        
        assert "no content in candidate" in str(exc_info.value)
        assert exc_info.value.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_normalize_response_empty_parts(self, mock_model_class, mock_configure):
        """Test normalization raises error when content parts are empty."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock response with content but empty parts
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = []
        
        with pytest.raises(ProviderError) as exc_info:
            provider._normalize_response(mock_response, "gemini-2.5-flash", "req-5")
        
        assert "no content in candidate" in str(exc_info.value)
        assert exc_info.value.is_transient is False


class TestGeminiProviderErrorMapping:
    """Test error mapping for specific HTTP status codes."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_400_bad_request(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 400 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 400: Invalid request parameters")
        result = provider._map_error(error, "req-1")
        
        assert "bad request" in str(result)
        assert result.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_401_unauthorized(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 401 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 401: Unauthorized")
        result = provider._map_error(error, "req-2")
        
        assert "authentication error" in str(result)
        assert result.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_403_forbidden(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 403 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 403: Forbidden")
        result = provider._map_error(error, "req-3")
        
        assert "authentication error" in str(result)
        assert result.is_transient is False
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_429_rate_limit(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 429 rate limit errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 429: Rate limit exceeded")
        result = provider._map_error(error, "req-4")
        
        assert "rate limit exceeded" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_500_server_error(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 500 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 500: Internal server error")
        result = provider._map_error(error, "req-5")
        
        assert "server error" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_502_bad_gateway(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 502 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 502: Bad gateway")
        result = provider._map_error(error, "req-6")
        
        assert "server error" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_503_service_unavailable(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 503 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 503: Service unavailable")
        result = provider._map_error(error, "req-7")
        
        assert "server error" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_504_gateway_timeout(self, mock_model_class, mock_configure):
        """Test mapping of HTTP 504 errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("HTTP 504: Gateway timeout")
        result = provider._map_error(error, "req-8")
        
        assert "server error" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_timeout(self, mock_model_class, mock_configure):
        """Test mapping of timeout errors."""
        config = {"api_key": "test-key", "timeout": 30.0}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("Request timeout after 30 seconds")
        result = provider._map_error(error, "req-9")
        
        assert "timeout" in str(result)
        assert "30.0s" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_quota_exceeded(self, mock_model_class, mock_configure):
        """Test mapping of quota exceeded errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("Quota exceeded for this API key")
        result = provider._map_error(error, "req-10")
        
        assert "rate limit exceeded" in str(result)
        assert result.is_transient is True
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_map_error_unexpected(self, mock_model_class, mock_configure):
        """Test mapping of unexpected errors."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = Exception("Something unexpected happened")
        result = provider._map_error(error, "req-11")
        
        assert "unexpected error" in str(result)
        assert result.is_transient is False


class TestGeminiProviderAPIKeyMasking:
    """Test API key masking with various key lengths."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_standard_length(self, mock_model_class, mock_configure):
        """Test masking of standard length API key."""
        config = {"api_key": "test-key-1234567890"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("test-key-1234567890")
        
        assert masked == "***************7890"
        assert len(masked) == len("test-key-1234567890")
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_short_key(self, mock_model_class, mock_configure):
        """Test masking of short API key (4 characters or less)."""
        config = {"api_key": "test"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("test")
        
        assert masked == "****"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_very_short(self, mock_model_class, mock_configure):
        """Test masking of very short API key."""
        config = {"api_key": "test"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("ab")
        
        assert masked == "****"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_exactly_five_chars(self, mock_model_class, mock_configure):
        """Test masking of 5-character API key."""
        config = {"api_key": "test"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        masked = provider._mask_api_key("abcde")
        
        assert masked == "*bcde"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_mask_api_key_long_key(self, mock_model_class, mock_configure):
        """Test masking of long API key."""
        config = {"api_key": "test"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        long_key = "a" * 50 + "wxyz"
        masked = provider._mask_api_key(long_key)
        
        assert masked.endswith("wxyz")
        assert masked.startswith("*" * 50)
        assert len(masked) == 54


class TestGeminiProviderLogging:
    """Test logging behavior for start, success, and error events."""
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_log_request_start(self, mock_model_class, mock_configure):
        """Test logging of request start event."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        provider._log_request_start("req-123", "gemini-2.5-flash", 100)
        
        logger.log.assert_called_once_with(
            "provider_request_start",
            {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "prompt_length": 100,
                "request_id": "req-123"
            }
        )
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_log_request_success(self, mock_model_class, mock_configure):
        """Test logging of successful request."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        result = {
            "text": "Generated text",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "provider": "gemini"
        }
        
        provider._log_request_success("req-456", result)
        
        logger.log.assert_called_once_with(
            "provider_request_success",
            {
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "request_id": "req-456"
            }
        )
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_log_request_error(self, mock_model_class, mock_configure):
        """Test logging of request error."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        error = ProviderError("Rate limit exceeded", is_transient=True)
        
        provider._log_request_error("req-789", error)
        
        logger.log.assert_called_once_with(
            "provider_request_error",
            {
                "provider": "gemini",
                "error_message": "Rate limit exceeded",
                "is_transient": True,
                "request_id": "req-789"
            }
        )
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_logging_full_lifecycle(self, mock_model_class, mock_configure):
        """Test logging throughout full request lifecycle."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock successful response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 25
        mock_response.usage_metadata.candidates_token_count = 75
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        result = provider.generate("test prompt", {"request_id": "req-full"})
        
        # Verify both start and success logs
        assert logger.log.call_count == 2
        
        start_call = logger.log.call_args_list[0]
        assert start_call[0][0] == "provider_request_start"
        assert start_call[0][1]["request_id"] == "req-full"
        
        success_call = logger.log.call_args_list[1]
        assert success_call[0][0] == "provider_request_success"
        assert success_call[0][1]["request_id"] == "req-full"
    
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    def test_logging_error_lifecycle(self, mock_model_class, mock_configure):
        """Test logging when request fails."""
        config = {"api_key": "test-key"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock error response
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("HTTP 500: Server error")
        provider._model = mock_model_instance
        
        with pytest.raises(ProviderError):
            provider.generate("test prompt", {"request_id": "req-error"})
        
        # Verify start and error logs
        assert logger.log.call_count == 2
        
        start_call = logger.log.call_args_list[0]
        assert start_call[0][0] == "provider_request_start"
        
        error_call = logger.log.call_args_list[1]
        assert error_call[0][0] == "provider_request_error"
        assert error_call[0][1]["request_id"] == "req-error"
        assert error_call[0][1]["is_transient"] is True
