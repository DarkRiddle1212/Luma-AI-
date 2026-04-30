"""
Property-based test for GeminiProvider field consistency.

**Validates: Requirements 2.4**

Property 2: Provider Field Consistency
For any generation request to GeminiProvider, the returned dictionary SHALL have
`provider` field equal to "gemini".
"""

import sys
import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, MagicMock, patch

# Mock google.generativeai before importing GeminiProvider
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating random prompts
prompt_strategy = st.text(min_size=1, max_size=1000)

# Strategy for generating random options dictionaries
options_strategy = st.fixed_dictionaries(
    {},
    optional={
        "model": st.sampled_from(["gemini-2.5-flash", "gemini-pro", "gemini-1.5-pro"]),
        "temperature": st.floats(min_value=0.0, max_value=2.0),
        "max_tokens": st.integers(min_value=1, max_value=4096),
        "request_id": st.text(min_size=1, max_size=50),
    }
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

class TestGeminiProviderFieldConsistency:
    """
    Property 2: Provider Field Consistency

    **Validates: Requirements 2.4**
    
    For any generation request to GeminiProvider, the returned dictionary SHALL
    have `provider` field equal to "gemini".
    """

    @pytest.mark.property_test
    @patch('luma.core.llm.providers.gemini_provider.genai.configure')
    @patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel')
    @given(prompt=prompt_strategy, options=options_strategy)
    @settings(max_examples=100, deadline=None)
    def test_provider_field_always_gemini(
        self,
        mock_model_class,
        mock_configure,
        prompt: str,
        options: dict
    ):
        """
        For any valid prompt and options, GeminiProvider.generate() returns
        a dictionary with provider="gemini".

        **Validates: Requirements 2.4**
        
        Feature: gemini-provider-integration, Property 2: Provider field consistency
        """
        # Setup provider
        config = {
            "api_key": "test-key-12345678",
            "model": "gemini-2.5-flash",
            "temperature": 0.4,
            "max_tokens": 1024
        }
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)
        
        # Mock the model instance and its response
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Generated response text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        
        # Configure the mock to return our response
        mock_model_instance.generate_content.return_value = mock_response
        provider._model = mock_model_instance
        
        # If model override is provided, mock GenerativeModel to return our instance
        if "model" in options:
            mock_model_class.return_value = mock_model_instance
        
        # Call generate
        result = provider.generate(prompt, options)
        
        # Property assertion: provider field must always be "gemini"
        assert "provider" in result, (
            "Response dictionary must contain 'provider' field"
        )
        assert result["provider"] == "gemini", (
            f"GeminiProvider must always set provider='gemini', got '{result['provider']}'"
        )
