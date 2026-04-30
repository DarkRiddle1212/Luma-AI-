"""
End-to-end integration test with real Gemini API.

Tests successful generation, authentication, and timeout handling with the actual Gemini API.
Requires GEMINI_API_KEY environment variable to be set.

**Validates: Requirements 2.2, 2.3, 9.1, 9.2**

Feature: gemini-provider-integration
"""

import os
import pytest

from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.llm.providers.provider_interface import ProviderError
from luma.core.structured_logger import StructuredLogger
from unittest.mock import MagicMock


# Skip all tests in this module if GEMINI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set - skipping real API tests"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gemini_config():
    """Create Gemini configuration from environment variables."""
    return {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
        "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
        "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.4")),
        "log_prompts": False
    }


@pytest.fixture
def mock_logger():
    """Create a mock StructuredLogger."""
    return MagicMock(spec=StructuredLogger)


@pytest.fixture
def gemini_provider(gemini_config, mock_logger):
    """Create a GeminiProvider instance with real API key."""
    return GeminiProvider(config=gemini_config, logger=mock_logger)


# ---------------------------------------------------------------------------
# Test: Successful generation with real Gemini API
# ---------------------------------------------------------------------------

class TestGeminiE2EIntegration:
    """End-to-end integration tests with real Gemini API."""
    
    @pytest.mark.slow
    def test_successful_generation(self, gemini_provider):
        """
        Test successful text generation with real Gemini API.
        
        **Validates: Requirements 2.2, 2.3**
        """
        prompt = "What is 2 + 2? Answer with just the number."
        options = {
            "model": "gemini-2.5-flash",
            "temperature": 0.0,  # Use 0 for deterministic response
            "max_tokens": 10,
            "request_id": "e2e-test-001"
        }
        
        # Call generate
        result = gemini_provider.generate(prompt, options)
        
        # Verify response structure
        assert isinstance(result, dict)
        assert "text" in result
        assert "model" in result
        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert "provider" in result
        
        # Verify response content
        assert len(result["text"]) > 0
        assert result["provider"] == "gemini"
        assert result["model"] == "gemini-2.5-flash"
        
        # Verify token counts are positive
        assert result["prompt_tokens"] > 0
        assert result["completion_tokens"] > 0
        
        # Verify response contains expected answer
        assert "4" in result["text"]
    
    @pytest.mark.slow
    def test_authentication_with_valid_key(self, gemini_config, mock_logger):
        """
        Test authentication with valid API key.
        
        **Validates: Requirement 9.2**
        """
        # Create provider with valid key
        provider = GeminiProvider(config=gemini_config, logger=mock_logger)
        
        # Should be able to make a request
        prompt = "Say 'hello'"
        options = {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 10,
            "request_id": "auth-test-001"
        }
        
        result = provider.generate(prompt, options)
        
        # Should succeed without raising authentication error
        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 0
    
    @pytest.mark.slow
    def test_timeout_handling(self, gemini_config, mock_logger):
        """
        Test timeout handling with very short timeout.
        
        **Validates: Requirement 9.1**
        
        Note: This test may be flaky depending on network conditions.
        A very short timeout (0.001s) should trigger a timeout error.
        """
        # Create config with very short timeout
        short_timeout_config = {
            **gemini_config,
            "timeout": 0.001  # 1ms - should timeout
        }
        
        provider = GeminiProvider(config=short_timeout_config, logger=mock_logger)
        
        prompt = "Write a long essay about the history of computing."
        options = {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 1000,
            "request_id": "timeout-test-001"
        }
        
        # Should raise ProviderError with is_transient=True
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(prompt, options)
        
        # Verify error is transient
        assert exc_info.value.is_transient is True
        
        # Verify error message mentions timeout
        assert "timeout" in str(exc_info.value).lower()
    
    @pytest.mark.slow
    def test_multiple_sequential_requests(self, gemini_provider):
        """
        Test multiple sequential requests to verify provider stability.
        
        **Validates: Requirements 2.2, 2.3**
        """
        prompts = [
            "What is 1 + 1?",
            "What is 2 + 2?",
            "What is 3 + 3?"
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            options = {
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
                "max_tokens": 10,
                "request_id": f"multi-test-{i:03d}"
            }
            
            result = gemini_provider.generate(prompt, options)
            results.append(result)
        
        # Verify all requests succeeded
        assert len(results) == 3
        
        # Verify each result has valid structure
        for result in results:
            assert "text" in result
            assert "provider" in result
            assert result["provider"] == "gemini"
            assert len(result["text"]) > 0
    
    @pytest.mark.slow
    def test_response_normalization(self, gemini_provider):
        """
        Test that response normalization works correctly with real API.
        
        **Validates: Requirement 2.3**
        """
        prompt = "Say 'test'"
        options = {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 50,
            "request_id": "norm-test-001"
        }
        
        result = gemini_provider.generate(prompt, options)
        
        # Verify normalized response format
        assert isinstance(result["text"], str)
        assert isinstance(result["model"], str)
        assert isinstance(result["prompt_tokens"], int)
        assert isinstance(result["completion_tokens"], int)
        assert isinstance(result["provider"], str)
        
        # Verify values are reasonable
        assert result["prompt_tokens"] >= 0
        assert result["completion_tokens"] >= 0
        assert result["provider"] == "gemini"
    
    @pytest.mark.slow
    def test_different_temperature_values(self, gemini_provider):
        """
        Test generation with different temperature values.
        
        **Validates: Requirement 2.2**
        """
        prompt = "Write a creative sentence."
        temperatures = [0.0, 0.5, 1.0]
        
        for temp in temperatures:
            options = {
                "model": "gemini-2.5-flash",
                "temperature": temp,
                "max_tokens": 50,
                "request_id": f"temp-test-{temp}"
            }
            
            result = gemini_provider.generate(prompt, options)
            
            # Should succeed with any valid temperature
            assert result is not None
            assert "text" in result
            assert len(result["text"]) > 0
    
    @pytest.mark.slow
    def test_different_max_tokens_values(self, gemini_provider):
        """
        Test generation with different max_tokens values.
        
        **Validates: Requirement 2.2**
        """
        prompt = "Count from 1 to 10."
        max_tokens_values = [10, 50, 100]
        
        for max_tokens in max_tokens_values:
            options = {
                "model": "gemini-2.5-flash",
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "request_id": f"tokens-test-{max_tokens}"
            }
            
            result = gemini_provider.generate(prompt, options)
            
            # Should succeed with any valid max_tokens
            assert result is not None
            assert "text" in result
            assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Test: Authentication errors (requires invalid key)
# ---------------------------------------------------------------------------

class TestGeminiAuthenticationErrors:
    """Test authentication error handling."""
    
    @pytest.mark.slow
    def test_invalid_api_key(self, mock_logger):
        """
        Test that invalid API key raises non-transient error.
        
        **Validates: Requirement 9.2**
        
        Note: This test uses an obviously invalid API key.
        """
        invalid_config = {
            "api_key": "invalid-key-12345",
            "model": "gemini-2.5-flash",
            "timeout": 30.0,
            "max_tokens": 1024,
            "temperature": 0.7,
            "log_prompts": False
        }
        
        provider = GeminiProvider(config=invalid_config, logger=mock_logger)
        
        prompt = "Test prompt"
        options = {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_tokens": 10,
            "request_id": "invalid-key-test"
        }
        
        # Should raise ProviderError with is_transient=False
        with pytest.raises(ProviderError) as exc_info:
            provider.generate(prompt, options)
        
        # Verify error is non-transient (authentication errors shouldn't be retried)
        assert exc_info.value.is_transient is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

# These tests require a valid GEMINI_API_KEY environment variable.
# To run these tests:
#   1. Set GEMINI_API_KEY environment variable
#   2. Run: pytest tests/llm/test_gemini_e2e_integration.py -v -m slow
#
# These tests are marked as "slow" because they make real API calls.
# To skip slow tests: pytest -m "not slow"
