"""
Property-based test for GeminiProvider token extraction.

**Validates: Requirements 2.8, 6.3**

Property 5: Token Extraction
For any Gemini API response containing usage metadata, the normalized response
dictionary SHALL contain `prompt_tokens` and `completion_tokens` values extracted
from the metadata. When usage_metadata is None/missing, both SHALL default to 0.
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
# Helpers
# ---------------------------------------------------------------------------

def _make_provider():
    """Create a GeminiProvider with mocked Gemini SDK."""
    with patch('luma.core.llm.providers.gemini_provider.genai.configure'), \
         patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel'):
        config = {
            "api_key": "test-key-12345678",
            "model": "gemini-2.5-flash",
        }
        logger = Mock(spec=StructuredLogger)
        return GeminiProvider(config, logger)


def _make_response(prompt_token_count, candidates_token_count, usage_metadata_present=True):
    """Build a mock Gemini response object."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = "some generated text"

    if usage_metadata_present:
        mock_usage = Mock()
        mock_usage.prompt_token_count = prompt_token_count
        mock_usage.candidates_token_count = candidates_token_count
        mock_response.usage_metadata = mock_usage
    else:
        mock_response.usage_metadata = None

    return mock_response


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-negative token counts (realistic range)
token_count_strategy = st.integers(min_value=0, max_value=100_000)

# Strategy for responses with usage metadata present
usage_metadata_strategy = st.tuples(token_count_strategy, token_count_strategy)

# Strategy for model names
model_name_strategy = st.sampled_from([
    "gemini-2.5-flash",
    "gemini-pro",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestTokenExtractionProperty:
    """
    Property 5: Token Extraction

    **Validates: Requirements 2.8, 6.3**

    For any Gemini API response containing usage metadata, the normalized response
    dictionary SHALL contain `prompt_tokens` and `completion_tokens` values that
    exactly match the metadata values. When usage_metadata is None, both SHALL be 0.

    Feature: gemini-provider-integration, Property 5: Token extraction
    """

    @pytest.mark.property_test
    @given(
        prompt_tokens=token_count_strategy,
        completion_tokens=token_count_strategy,
        model_name=model_name_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_token_counts_extracted_from_usage_metadata(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
    ):
        """
        For any response with usage_metadata present, _normalize_response() SHALL
        extract prompt_tokens and completion_tokens exactly from the metadata.

        **Validates: Requirements 2.8, 6.3**

        Feature: gemini-provider-integration, Property 5: Token extraction
        """
        provider = _make_provider()
        mock_response = _make_response(prompt_tokens, completion_tokens, usage_metadata_present=True)

        result = provider._normalize_response(mock_response, model_name, "req-test")

        assert result["prompt_tokens"] == prompt_tokens, (
            f"Expected prompt_tokens={prompt_tokens}, got {result['prompt_tokens']}"
        )
        assert result["completion_tokens"] == completion_tokens, (
            f"Expected completion_tokens={completion_tokens}, got {result['completion_tokens']}"
        )

    @pytest.mark.property_test
    @given(model_name=model_name_strategy)
    @settings(max_examples=100, deadline=None)
    def test_token_counts_default_to_zero_when_usage_metadata_is_none(
        self,
        model_name: str,
    ):
        """
        When usage_metadata is None, _normalize_response() SHALL set both
        prompt_tokens and completion_tokens to 0.

        **Validates: Requirements 2.8, 6.3**

        Feature: gemini-provider-integration, Property 5: Token extraction
        """
        provider = _make_provider()
        mock_response = _make_response(0, 0, usage_metadata_present=False)

        result = provider._normalize_response(mock_response, model_name, "req-test")

        assert result["prompt_tokens"] == 0, (
            f"Expected prompt_tokens=0 when usage_metadata is None, got {result['prompt_tokens']}"
        )
        assert result["completion_tokens"] == 0, (
            f"Expected completion_tokens=0 when usage_metadata is None, got {result['completion_tokens']}"
        )

    @pytest.mark.property_test
    @given(
        prompt_tokens=token_count_strategy,
        completion_tokens=token_count_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_token_counts_are_non_negative_integers(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        """
        For any response, the extracted token counts SHALL be non-negative integers.

        **Validates: Requirements 2.8, 6.3**

        Feature: gemini-provider-integration, Property 5: Token extraction
        """
        provider = _make_provider()
        mock_response = _make_response(prompt_tokens, completion_tokens, usage_metadata_present=True)

        result = provider._normalize_response(mock_response, "gemini-2.5-flash", "req-test")

        assert isinstance(result["prompt_tokens"], int), (
            f"prompt_tokens must be int, got {type(result['prompt_tokens'])}"
        )
        assert isinstance(result["completion_tokens"], int), (
            f"completion_tokens must be int, got {type(result['completion_tokens'])}"
        )
        assert result["prompt_tokens"] >= 0, (
            f"prompt_tokens must be >= 0, got {result['prompt_tokens']}"
        )
        assert result["completion_tokens"] >= 0, (
            f"completion_tokens must be >= 0, got {result['completion_tokens']}"
        )
