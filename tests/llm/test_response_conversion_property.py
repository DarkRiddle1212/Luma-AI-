"""
Property-based test for LLMClient response conversion.

**Validates: Requirements 5.3**

Property: For any valid provider response dictionary, LLMClient converts it to 
LLMResponse preserving all field values.

Feature: gemini-provider-integration, Property 8: Response Conversion
"""

import pytest
from hypothesis import given, settings, strategies as st
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from luma.core.llm.llm_client import ProviderLLMClient
from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.config import LLMConfig
from luma.core.llm.schemas import LLMRequest, LLMResponse, PromptContext
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating valid provider response dictionaries
provider_response_strategy = st.fixed_dictionaries({
    "text": st.text(min_size=0, max_size=1000),
    "model": st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    "prompt_tokens": st.integers(min_value=0, max_value=100000),
    "completion_tokens": st.integers(min_value=0, max_value=100000),
    "provider": st.sampled_from(["gemini", "openai", "anthropic", "mock"])
})

# Strategy for generating request IDs
request_id_strategy = st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",)))

# Strategy for generating LLMRequest objects
@st.composite
def llm_request_strategy(draw):
    """Generate LLMRequest objects for testing."""
    request_id = draw(request_id_strategy)
    model = draw(st.text(min_size=1, max_size=50))
    
    ctx = PromptContext(
        system_instructions="Test system instructions",
        user_profile="test_user",
        relevant_memories=[],
        current_input="Test input",
        output_constraints="Be concise"
    )
    
    return LLMRequest(
        prompt_context=ctx,
        model=model,
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.integers(min_value=1, max_value=4000)),
        request_id=request_id
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_config(max_retries: int = 2) -> LLMConfig:
    """Create a test LLMConfig."""
    return LLMConfig(
        api_key="test-key",
        model="test-model",
        max_retries=max_retries,
        provider_name="mock",
    )


def make_logger() -> StructuredLogger:
    """Create a mock logger."""
    return MagicMock(spec=StructuredLogger)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestResponseConversionProperty:
    """
    Property: Response dictionary to LLMResponse conversion.
    
    **Validates: Requirements 5.3**
    
    Feature: gemini-provider-integration, Property 8: Response Conversion
    """
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_preserves_all_fields(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        For any valid provider response dictionary and LLMRequest,
        LLMClient converts the response to LLMResponse preserving all field values.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Create a mock provider that returns the generated response
        provider = MockProvider(config={"responses": [response_dict]})
        
        # Create LLMClient with the mock provider
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify result is an LLMResponse
        assert isinstance(result, LLMResponse)
        
        # Verify all fields are preserved
        assert result.raw_text == response_dict["text"]
        assert result.model == response_dict["model"]
        assert result.prompt_tokens == response_dict["prompt_tokens"]
        assert result.completion_tokens == response_dict["completion_tokens"]
        assert result.provider == response_dict["provider"]
        assert result.request_id == request.request_id
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_with_extra_fields(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        Provider response dictionaries may contain extra fields beyond the required ones.
        LLMClient should still convert successfully, ignoring extra fields.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Add extra fields to the response dictionary
        response_with_extras = {
            **response_dict,
            "extra_field": "extra_value",
            "metadata": {"key": "value"},
            "timestamp": 1234567890
        }
        
        # Create a mock provider that returns the response with extras
        provider = MockProvider(config={"responses": [response_with_extras]})
        
        # Create LLMClient with the mock provider
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify result is an LLMResponse
        assert isinstance(result, LLMResponse)
        
        # Verify all required fields are preserved (extra fields ignored)
        assert result.raw_text == response_dict["text"]
        assert result.model == response_dict["model"]
        assert result.prompt_tokens == response_dict["prompt_tokens"]
        assert result.completion_tokens == response_dict["completion_tokens"]
        assert result.provider == response_dict["provider"]
        assert result.request_id == request.request_id
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_token_counts_non_negative(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        Token counts in the converted LLMResponse should be non-negative integers.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Create a mock provider
        provider = MockProvider(config={"responses": [response_dict]})
        
        # Create LLMClient
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify token counts are non-negative
        assert result.prompt_tokens >= 0
        assert result.completion_tokens >= 0
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_provider_field_preserved(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        The provider field should be preserved in the conversion.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Create a mock provider
        provider = MockProvider(config={"responses": [response_dict]})
        
        # Create LLMClient
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify provider field is preserved
        assert result.provider == response_dict["provider"]
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_model_field_preserved(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        The model field should be preserved in the conversion, regardless of request model.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Create a mock provider
        provider = MockProvider(config={"responses": [response_dict]})
        
        # Create LLMClient
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify model field is preserved (not overridden by request model)
        assert result.model == response_dict["model"]
    
    @pytest.mark.property_test
    @given(
        response_dict=provider_response_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_response_conversion_text_field_preserved(self, response_dict: Dict[str, Any], request: LLMRequest):
        """
        The text field should be preserved exactly as provided.
        
        **Validates: Requirements 5.3**
        
        Feature: gemini-provider-integration, Property 8: Response Conversion
        """
        # Create a mock provider
        provider = MockProvider(config={"responses": [response_dict]})
        
        # Create LLMClient
        client = ProviderLLMClient(
            provider=provider,
            config=make_config(),
            logger=make_logger()
        )
        
        # Call complete() and get the result
        result = client.complete(request)
        
        # Verify text field is preserved exactly
        assert result.raw_text == response_dict["text"]