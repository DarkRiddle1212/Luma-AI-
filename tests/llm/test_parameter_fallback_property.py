"""
Property-based test for GeminiProvider parameter fallback behavior.

**Validates: Requirements 2.6, 2.7**

Property 4: Parameter Fallback Behavior
For any generation request, if the options dictionary contains a `model`
parameter, it SHALL be used; if the options dictionary does not contain a
`model` parameter, the provider's configured default model SHALL be used.

Feature: gemini-provider-integration, Property 4: Parameter fallback behavior
"""

import sys
from unittest.mock import MagicMock, Mock

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

# Valid model names to use in tests
VALID_MODEL_NAMES = [
    "gemini-2.5-flash",
    "gemini-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-ultra",
]

# Strategy for config model names
config_model_strategy = st.sampled_from(VALID_MODEL_NAMES)

# Strategy for options model names (different pool to allow overlap/divergence)
options_model_strategy = st.sampled_from(VALID_MODEL_NAMES)

# Strategy for options dicts that include a "model" key
options_with_model_strategy = st.fixed_dictionaries(
    {"model": options_model_strategy},
    optional={
        "temperature": st.floats(min_value=0.0, max_value=2.0),
        "max_tokens": st.integers(min_value=1, max_value=4096),
        "request_id": st.text(min_size=1, max_size=50),
    },
)

# Strategy for options dicts that do NOT include a "model" key
options_without_model_strategy = st.fixed_dictionaries(
    {},
    optional={
        "temperature": st.floats(min_value=0.0, max_value=2.0),
        "max_tokens": st.integers(min_value=1, max_value=4096),
        "request_id": st.text(min_size=1, max_size=50),
    },
).filter(lambda d: "model" not in d)

# Simple prompt strategy
prompt_strategy = st.text(min_size=1, max_size=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(config_model: str) -> GeminiProvider:
    """Create a GeminiProvider with a specific default model and mocked SDK."""
    genai_mock = sys.modules["google.generativeai"]
    genai_mock.configure = MagicMock()
    genai_mock.GenerativeModel = MagicMock()

    config = {
        "api_key": "test-api-key-for-property-test",
        "model": config_model,
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    logger = Mock(spec=StructuredLogger)
    return GeminiProvider(config, logger)


def _mock_successful_response(provider: GeminiProvider) -> None:
    """Attach a mock model instance that returns a successful response."""
    genai_mock = sys.modules["google.generativeai"]

    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Generated response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_model_instance.generate_content.return_value = mock_response

    # Patch both the default model and any dynamically created model
    provider._model = mock_model_instance
    genai_mock.GenerativeModel.return_value = mock_model_instance


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestParameterFallbackBehavior:
    """
    Property 4: Parameter Fallback Behavior

    **Validates: Requirements 2.6, 2.7**

    When options contains "model", that model SHALL be used in the response.
    When options does not contain "model", the config's default model SHALL be used.
    """

    @pytest.mark.property_test
    @given(
        prompt=prompt_strategy,
        config_model=config_model_strategy,
        options=options_with_model_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_options_model_overrides_config_model(
        self,
        prompt: str,
        config_model: str,
        options: dict,
    ):
        """
        When options dict contains a "model" key, the response "model" field
        SHALL equal the value from options, not the config default.

        **Validates: Requirements 2.6, 2.7**

        Feature: gemini-provider-integration, Property 4: Parameter fallback behavior
        """
        provider = _make_provider(config_model)
        _mock_successful_response(provider)

        result = provider.generate(prompt, options)

        expected_model = options["model"]
        assert result["model"] == expected_model, (
            f"When options contains model='{expected_model}', "
            f"response model must be '{expected_model}', "
            f"but got '{result['model']}' (config default was '{config_model}')"
        )

    @pytest.mark.property_test
    @given(
        prompt=prompt_strategy,
        config_model=config_model_strategy,
        options=options_without_model_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_config_model_used_when_options_has_no_model(
        self,
        prompt: str,
        config_model: str,
        options: dict,
    ):
        """
        When options dict does NOT contain a "model" key, the response "model"
        field SHALL equal the provider's configured default model.

        **Validates: Requirements 2.6, 2.7**

        Feature: gemini-provider-integration, Property 4: Parameter fallback behavior
        """
        provider = _make_provider(config_model)
        _mock_successful_response(provider)

        result = provider.generate(prompt, options)

        assert result["model"] == config_model, (
            f"When options has no 'model' key, response model must fall back to "
            f"config default '{config_model}', but got '{result['model']}'"
        )
