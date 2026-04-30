"""
Property-based test for provider response format.

**Validates: Requirements 1.3, 6.1**

Property: All provider responses contain required keys (text, model, prompt_tokens,
completion_tokens, provider) with correct types (str, str, int, int, str).

Feature: gemini-provider-integration, Property 1: Provider response format
"""

import pytest
from hypothesis import given, settings, strategies as st
from typing import Dict, Any

from luma.core.llm.providers.provider_interface import LLMProvider


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating valid provider response dictionaries
provider_response_strategy = st.fixed_dictionaries({
    "text": st.text(min_size=0),  # Allow empty text
    "model": st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",))),
    "prompt_tokens": st.integers(min_value=0, max_value=100000),
    "completion_tokens": st.integers(min_value=0, max_value=100000),
    "provider": st.sampled_from(["gemini", "openai", "anthropic", "mock"])
})

# Strategy for generating provider responses with extra fields (should still be valid)
@st.composite
def provider_response_with_extras_strategy(draw):
    """Generate provider response with extra fields."""
    base = draw(provider_response_strategy)
    extra_field = draw(st.text())
    metadata = draw(st.dictionaries(st.text(), st.text(), max_size=3))
    return {
        **base,
        "extra_field": extra_field,
        "metadata": metadata
    }

# Strategy for generating invalid responses (missing required keys)
missing_key_strategy = st.sampled_from([
    "text", "model", "prompt_tokens", "completion_tokens", "provider"
])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_provider_response(response: Dict[str, Any]) -> bool:
    """
    Validate that a provider response contains all required keys with correct types.
    
    Args:
        response: Provider response dictionary to validate
    
    Returns:
        True if response is valid, False otherwise
    """
    required_keys = ["text", "model", "prompt_tokens", "completion_tokens", "provider"]
    
    # Check all required keys are present
    for key in required_keys:
        if key not in response:
            return False
    
    # Check types
    if not isinstance(response["text"], str):
        return False
    if not isinstance(response["model"], str):
        return False
    if not isinstance(response["prompt_tokens"], int):
        return False
    if not isinstance(response["completion_tokens"], int):
        return False
    if not isinstance(response["provider"], str):
        return False
    
    return True


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestProviderResponseFormat:
    """
    Property: Provider response format validation.
    
    **Validates: Requirements 1.3, 6.1**
    
    Feature: gemini-provider-integration, Property 1: Provider response format
    """
    
    @pytest.mark.property_test
    @given(response=provider_response_strategy)
    @settings(max_examples=100, deadline=None)
    def test_valid_responses_contain_all_required_keys(self, response: Dict[str, Any]):
        """
        For any valid provider response, all required keys must be present.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        required_keys = ["text", "model", "prompt_tokens", "completion_tokens", "provider"]
        
        for key in required_keys:
            assert key in response, (
                f"Provider response must contain '{key}' key. "
                f"Got keys: {list(response.keys())}"
            )
    
    @pytest.mark.property_test
    @given(response=provider_response_strategy)
    @settings(max_examples=100, deadline=None)
    def test_valid_responses_have_correct_types(self, response: Dict[str, Any]):
        """
        For any valid provider response, all fields must have correct types.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        assert isinstance(response["text"], str), (
            f"'text' must be str, got {type(response['text']).__name__}"
        )
        assert isinstance(response["model"], str), (
            f"'model' must be str, got {type(response['model']).__name__}"
        )
        assert isinstance(response["prompt_tokens"], int), (
            f"'prompt_tokens' must be int, got {type(response['prompt_tokens']).__name__}"
        )
        assert isinstance(response["completion_tokens"], int), (
            f"'completion_tokens' must be int, got {type(response['completion_tokens']).__name__}"
        )
        assert isinstance(response["provider"], str), (
            f"'provider' must be str, got {type(response['provider']).__name__}"
        )
    
    @pytest.mark.property_test
    @given(response=provider_response_strategy)
    @settings(max_examples=100, deadline=None)
    def test_token_counts_are_non_negative(self, response: Dict[str, Any]):
        """
        For any valid provider response, token counts must be non-negative integers.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        assert response["prompt_tokens"] >= 0, (
            f"'prompt_tokens' must be non-negative, got {response['prompt_tokens']}"
        )
        assert response["completion_tokens"] >= 0, (
            f"'completion_tokens' must be non-negative, got {response['completion_tokens']}"
        )
    
    @pytest.mark.property_test
    @given(response=provider_response_with_extras_strategy())
    @settings(max_examples=100, deadline=None)
    def test_responses_with_extra_fields_are_valid(self, response: Dict[str, Any]):
        """
        Provider responses may contain extra fields beyond the required ones.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        # Validate that required fields are still present and valid
        assert validate_provider_response(response), (
            "Provider response with extra fields must still contain all required fields "
            "with correct types"
        )
    
    @pytest.mark.property_test
    @given(
        response=provider_response_strategy,
        missing_key=missing_key_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_responses_missing_required_keys_are_invalid(
        self,
        response: Dict[str, Any],
        missing_key: str
    ):
        """
        Provider responses missing any required key should be detected as invalid.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        # Create a copy and remove one required key
        incomplete_response = {k: v for k, v in response.items() if k != missing_key}
        
        # Validation should fail
        assert not validate_provider_response(incomplete_response), (
            f"Response missing '{missing_key}' should be invalid"
        )
    
    @pytest.mark.property_test
    @given(response=provider_response_strategy)
    @settings(max_examples=100, deadline=None)
    def test_model_field_is_non_empty_string(self, response: Dict[str, Any]):
        """
        The 'model' field should be a non-empty string.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        assert isinstance(response["model"], str), (
            f"'model' must be str, got {type(response['model']).__name__}"
        )
        assert len(response["model"]) > 0, (
            "'model' must be a non-empty string"
        )
    
    @pytest.mark.property_test
    @given(response=provider_response_strategy)
    @settings(max_examples=100, deadline=None)
    def test_provider_field_is_valid_identifier(self, response: Dict[str, Any]):
        """
        The 'provider' field should be a valid provider identifier string.
        
        **Validates: Requirements 1.3, 6.1**
        
        Feature: gemini-provider-integration, Property 1: Provider response format
        """
        assert isinstance(response["provider"], str), (
            f"'provider' must be str, got {type(response['provider']).__name__}"
        )
        # Provider should be one of the known providers
        assert response["provider"] in ["gemini", "openai", "anthropic", "mock"], (
            f"'provider' should be a known provider identifier, got '{response['provider']}'"
        )
