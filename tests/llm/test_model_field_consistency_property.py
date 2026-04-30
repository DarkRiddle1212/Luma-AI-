"""
Property-based test for GeminiProvider model field consistency.

**Validates: Requirements 6.5**

Property 10: The `model` field in the response dictionary always equals
the model name used for the request.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from hypothesis import given, settings, strategies as st

# Mock google.generativeai before importing GeminiProvider
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider  # noqa: E402
from luma.core.structured_logger import StructuredLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate plausible model name strings: non-empty, printable, no whitespace-only
model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
    min_size=1,
    max_size=64,
).filter(lambda s: s.strip())


def _make_mock_response(text: str = "hello") -> Mock:
    """Build a minimal mock Gemini API response."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = text
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 5
    mock_response.usage_metadata.candidates_token_count = 10
    return mock_response


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestModelFieldConsistency:
    """
    Property 10: Model field consistency.

    **Validates: Requirements 6.5**

    For any model name passed in the options dict, the response `model` field
    must equal that model name.
    """

    @pytest.mark.property_test
    @given(model_name=model_name_strategy)
    @settings(max_examples=100, deadline=None)
    @patch("luma.core.llm.providers.gemini_provider.genai.configure")
    @patch("luma.core.llm.providers.gemini_provider.genai.GenerativeModel")
    def test_response_model_matches_requested_model(
        self, mock_model_class, mock_configure, model_name: str
    ):
        """
        For any model name string, generate() returns a response whose `model`
        field equals the requested model name.

        **Validates: Requirements 6.5**

        Feature: gemini-provider-integration, Property 10: Model field consistency
        """
        config = {"api_key": "test-api-key", "model": "gemini-2.5-flash"}
        logger = Mock(spec=StructuredLogger)
        provider = GeminiProvider(config, logger)

        # Wire the mock so any GenerativeModel() call returns a model that
        # produces a successful response.
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = _make_mock_response()
        mock_model_class.return_value = mock_model_instance

        # Also patch the provider's default model instance so the same mock
        # is used when the model name matches the default.
        provider._model = mock_model_instance

        result = provider.generate("test prompt", {"model": model_name})

        assert result["model"] == model_name, (
            f"Expected response model '{model_name}' but got '{result['model']}'"
        )
