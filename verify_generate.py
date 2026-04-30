"""
Simple verification script for GeminiProvider.generate() implementation.
"""

import sys
from unittest.mock import Mock, MagicMock

# Mock google.generativeai before importing
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.llm.providers.provider_interface import ProviderError
from luma.core.structured_logger import StructuredLogger

def test_generate_basic():
    """Test basic generate() functionality."""
    print("Testing basic generate() functionality...")
    
    # Create provider
    config = {"api_key": "test-key", "temperature": 0.5, "max_tokens": 2048}
    logger = Mock(spec=StructuredLogger)
    
    # Mock genai module
    import google.generativeai as genai
    genai.configure = Mock()
    genai.GenerativeModel = Mock()
    genai.types = Mock()
    
    # Create mock generation config
    mock_gen_config = Mock()
    genai.types.GenerationConfig = Mock(return_value=mock_gen_config)
    
    provider = GeminiProvider(config, logger)
    
    # Mock the model and response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = "Generated text"
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_model.generate_content.return_value = mock_response
    provider._model = mock_model
    
    # Call generate
    result = provider.generate("test prompt", {})
    
    # Verify result
    assert result["text"] == "Generated text"
    assert result["model"] == "gemini-2.5-flash"
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 20
    assert result["provider"] == "gemini"
    
    print("✓ Basic generate() test passed")

def test_generate_error_handling():
    """Test error handling in generate()."""
    print("Testing error handling...")
    
    # Create provider
    config = {"api_key": "test-key"}
    logger = Mock(spec=StructuredLogger)
    
    # Mock genai module
    import google.generativeai as genai
    genai.configure = Mock()
    genai.GenerativeModel = Mock()
    genai.types = Mock()
    
    # Create mock generation config
    mock_gen_config = Mock()
    genai.types.GenerationConfig = Mock(return_value=mock_gen_config)
    
    provider = GeminiProvider(config, logger)
    
    # Mock the model to raise an error
    mock_model = Mock()
    mock_model.generate_content.side_effect = Exception("API Error 500")
    provider._model = mock_model
    
    # Call generate and expect ProviderError
    try:
        provider.generate("test prompt", {})
        assert False, "Should have raised ProviderError"
    except ProviderError as e:
        assert "server error" in str(e)
        assert e.is_transient is True
        print("✓ Error handling test passed")

def test_generate_logging():
    """Test logging in generate()."""
    print("Testing logging...")
    
    # Create provider
    config = {"api_key": "test-key"}
    logger = Mock(spec=StructuredLogger)
    
    # Mock genai module
    import google.generativeai as genai
    genai.configure = Mock()
    genai.GenerativeModel = Mock()
    genai.types = Mock()
    
    # Create mock generation config
    mock_gen_config = Mock()
    genai.types.GenerationConfig = Mock(return_value=mock_gen_config)
    
    provider = GeminiProvider(config, logger)
    
    # Mock the model and response
    mock_model = Mock()
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = "Generated text"
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_model.generate_content.return_value = mock_response
    provider._model = mock_model
    
    # Call generate
    result = provider.generate("test prompt", {"request_id": "req-123"})
    
    # Verify logging
    assert logger.log.call_count == 2  # start and success
    
    # Check start log
    start_call = logger.log.call_args_list[0]
    assert start_call[0][0] == "provider_request_start"
    assert start_call[0][1]["request_id"] == "req-123"
    
    # Check success log
    success_call = logger.log.call_args_list[1]
    assert success_call[0][0] == "provider_request_success"
    assert success_call[0][1]["request_id"] == "req-123"
    
    print("✓ Logging test passed")

if __name__ == "__main__":
    test_generate_basic()
    test_generate_error_handling()
    test_generate_logging()
    print("\n✅ All verification tests passed!")
