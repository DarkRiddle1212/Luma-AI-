"""
Unit tests for LLM Provider Interface.

Tests the ProviderError exception class and the LLMProvider abstract base class.
"""

import pytest
from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError


class TestProviderError:
    """Test ProviderError exception class."""
    
    def test_provider_error_initialization_with_defaults(self):
        """Test ProviderError initializes with default is_transient=False."""
        error = ProviderError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.is_transient is False
    
    def test_provider_error_initialization_with_transient_true(self):
        """Test ProviderError initializes with is_transient=True."""
        error = ProviderError("Network timeout", is_transient=True)
        
        assert str(error) == "Network timeout"
        assert error.is_transient is True
    
    def test_provider_error_initialization_with_transient_false(self):
        """Test ProviderError initializes with is_transient=False explicitly."""
        error = ProviderError("Invalid API key", is_transient=False)
        
        assert str(error) == "Invalid API key"
        assert error.is_transient is False
    
    def test_provider_error_is_exception_subclass(self):
        """Test ProviderError is a subclass of Exception."""
        error = ProviderError("Test")
        
        assert isinstance(error, Exception)
    
    def test_provider_error_can_be_raised_and_caught(self):
        """Test ProviderError can be raised and caught."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Test error", is_transient=True)
        
        assert str(exc_info.value) == "Test error"
        assert exc_info.value.is_transient is True


class TestLLMProviderInterface:
    """Test LLMProvider abstract base class."""
    
    def test_cannot_instantiate_abstract_provider(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMProvider()
    
    def test_subclass_must_implement_generate(self):
        """Test that subclass must implement generate() method."""
        class IncompleteProvider(LLMProvider):
            def validate_config(self, config):
                return True
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider()
    
    def test_subclass_must_implement_validate_config(self):
        """Test that subclass must implement validate_config() method."""
        class IncompleteProvider(LLMProvider):
            def generate(self, prompt, options):
                return {}
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProvider()
    
    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete subclass implementing all methods can be instantiated."""
        class CompleteProvider(LLMProvider):
            def generate(self, prompt, options):
                return {
                    "text": "Generated text",
                    "model": "test-model",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "provider": "test"
                }
            
            def validate_config(self, config):
                return True
        
        provider = CompleteProvider()
        assert provider is not None
        assert isinstance(provider, LLMProvider)
    
    def test_complete_subclass_generate_returns_expected_format(self):
        """Test that a complete subclass generate() returns expected dictionary format."""
        class CompleteProvider(LLMProvider):
            def generate(self, prompt, options):
                return {
                    "text": f"Response to: {prompt}",
                    "model": options.get("model", "default-model"),
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": 15,
                    "provider": "test"
                }
            
            def validate_config(self, config):
                return "api_key" in config
        
        provider = CompleteProvider()
        result = provider.generate("Hello world", {"model": "test-model"})
        
        assert result["text"] == "Response to: Hello world"
        assert result["model"] == "test-model"
        assert result["prompt_tokens"] == 2
        assert result["completion_tokens"] == 15
        assert result["provider"] == "test"
    
    def test_complete_subclass_validate_config_works(self):
        """Test that a complete subclass validate_config() works correctly."""
        class CompleteProvider(LLMProvider):
            def generate(self, prompt, options):
                return {
                    "text": "test",
                    "model": "test",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "provider": "test"
                }
            
            def validate_config(self, config):
                if "api_key" not in config:
                    raise ValueError("api_key is required")
                return True
        
        provider = CompleteProvider()
        
        # Valid config
        assert provider.validate_config({"api_key": "test-key"}) is True
        
        # Invalid config
        with pytest.raises(ValueError, match="api_key is required"):
            provider.validate_config({})
