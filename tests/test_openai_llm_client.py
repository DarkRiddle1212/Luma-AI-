"""
Unit tests for OpenAI_LLM_Client implementation.

Tests the OpenAI_LLM_Client class to ensure it correctly implements
the LLM_Client_Interface and handles API key configuration, error
handling, and response generation.

**Validates: Requirements 3.3, 3.4, 3.5, 10.4**
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from luma.core.reasoning.openai_llm_client import OpenAI_LLM_Client
from luma.core.reasoning.llm_client_interface import LLM_Client_Interface


@pytest.fixture(autouse=True)
def mock_openai_module():
    """Mock the openai module for all tests."""
    mock_openai = MagicMock()
    sys.modules['openai'] = mock_openai
    yield mock_openai
    if 'openai' in sys.modules:
        del sys.modules['openai']


class TestOpenAILLMClientInitialization:
    """Test OpenAI_LLM_Client initialization and configuration."""
    
    def test_client_implements_interface(self, mock_openai_module):
        """Test that OpenAI_LLM_Client implements LLM_Client_Interface."""
        client = OpenAI_LLM_Client(api_key="test-key")
        assert isinstance(client, LLM_Client_Interface)
    
    def test_initialization_with_api_key(self, mock_openai_module):
        """Test client initialization with explicit API key."""
        client = OpenAI_LLM_Client(api_key="test-api-key")
        
        assert client.api_key == "test-api-key"
        assert client.model == "gpt-3.5-turbo"
        assert client.temperature == 0.7
        assert client.max_tokens is None
        mock_openai_module.OpenAI.assert_called_once_with(api_key="test-api-key")
    
    def test_initialization_with_environment_variable(self, mock_openai_module):
        """Test client initialization using OPENAI_API_KEY environment variable."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-api-key'}):
            client = OpenAI_LLM_Client()
            
            assert client.api_key == "env-api-key"
            mock_openai_module.OpenAI.assert_called_once_with(api_key="env-api-key")
    
    def test_initialization_without_api_key_raises_error(self, mock_openai_module):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OpenAI_LLM_Client()
            
            assert "API key must be provided" in str(exc_info.value)
            assert "OPENAI_API_KEY" in str(exc_info.value)
    
    def test_initialization_with_custom_parameters(self, mock_openai_module):
        """Test client initialization with custom model and parameters."""
        client = OpenAI_LLM_Client(
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert client.model == "gpt-4"
        assert client.temperature == 0.5
        assert client.max_tokens == 1000
    
    def test_initialization_without_openai_package_raises_error(self):
        """Test that missing openai package raises ImportError."""
        # Remove the mock temporarily
        if 'openai' in sys.modules:
            del sys.modules['openai']
        
        with pytest.raises(ImportError) as exc_info:
            OpenAI_LLM_Client(api_key="test-key")
        
        assert "openai" in str(exc_info.value).lower()
        assert "pip install openai" in str(exc_info.value)


class TestOpenAILLMClientGenerate:
    """Test OpenAI_LLM_Client generate method."""
    
    def test_generate_returns_string(self, mock_openai_module):
        """Test that generate method returns a string response."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client
        
        client = OpenAI_LLM_Client(api_key="test-key")
        response = client.generate("Test prompt")
        
        assert isinstance(response, str)
        assert response == "This is a test response"
    
    def test_generate_sends_prompt_to_api(self, mock_openai_module):
        """Test that generate sends the prompt to OpenAI API correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client
        
        client = OpenAI_LLM_Client(api_key="test-key")
        client.generate("What is Python?")
        
        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        
        assert call_args.kwargs['model'] == "gpt-3.5-turbo"
        assert call_args.kwargs['messages'] == [
            {"role": "user", "content": "What is Python?"}
        ]
        assert call_args.kwargs['temperature'] == 0.7
        assert call_args.kwargs['max_tokens'] is None
    
    def test_generate_with_custom_parameters(self, mock_openai_module):
        """Test that generate uses custom model and parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client
        
        client = OpenAI_LLM_Client(
            api_key="test-key",
            model="gpt-4",
            temperature=0.3,
            max_tokens=500
        )
        client.generate("Test")
        
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == "gpt-4"
        assert call_args.kwargs['temperature'] == 0.3
        assert call_args.kwargs['max_tokens'] == 500
    
    def test_generate_with_empty_prompt_raises_error(self, mock_openai_module):
        """Test that generate raises ValueError for empty prompt."""
        client = OpenAI_LLM_Client(api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            client.generate("")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_generate_with_whitespace_only_prompt_raises_error(self, mock_openai_module):
        """Test that generate raises ValueError for whitespace-only prompt."""
        client = OpenAI_LLM_Client(api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            client.generate("   \n\t  ")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_generate_handles_api_errors(self, mock_openai_module):
        """Test that generate properly handles and re-raises API errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error: Rate limit exceeded")
        mock_openai_module.OpenAI.return_value = mock_client
        
        client = OpenAI_LLM_Client(api_key="test-key")
        
        with pytest.raises(Exception) as exc_info:
            client.generate("Test prompt")
        
        assert "OpenAI API request failed" in str(exc_info.value)
        assert "Rate limit exceeded" in str(exc_info.value)


class TestOpenAILLMClientIntegration:
    """Integration tests for OpenAI_LLM_Client."""
    
    def test_client_can_be_used_as_interface(self, mock_openai_module):
        """Test that OpenAI_LLM_Client can be used through the interface."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Interface response"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_module.OpenAI.return_value = mock_client
        
        # Use through interface type
        client: LLM_Client_Interface = OpenAI_LLM_Client(api_key="test-key")
        response = client.generate("Test")
        
        assert response == "Interface response"
    
    def test_multiple_generate_calls(self, mock_openai_module):
        """Test that client can handle multiple generate calls."""
        mock_client = MagicMock()
        
        # Setup different responses for each call
        responses = []
        for i in range(3):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f"Response {i+1}"
            responses.append(mock_response)
        
        mock_client.chat.completions.create.side_effect = responses
        mock_openai_module.OpenAI.return_value = mock_client
        
        client = OpenAI_LLM_Client(api_key="test-key")
        
        result1 = client.generate("Prompt 1")
        result2 = client.generate("Prompt 2")
        result3 = client.generate("Prompt 3")
        
        assert result1 == "Response 1"
        assert result2 == "Response 2"
        assert result3 == "Response 3"
        assert mock_client.chat.completions.create.call_count == 3
