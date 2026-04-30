"""
Property-based test for error status code inclusion in ProviderError messages.

**Validates: Requirements 2.5, 7.2**

Property 3: For any HTTP error response from the Gemini API, the raised
ProviderError message SHALL contain the HTTP status code as a substring.

Feature: gemini-provider-integration
Property 3: Error status code inclusion
"""

import sys
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st

# Mock google.generativeai before importing GeminiProvider
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

from luma.core.llm.providers.gemini_provider import GeminiProvider  # noqa: E402
from luma.core.llm.providers.provider_interface import ProviderError  # noqa: E402


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The set of HTTP error status codes that GeminiProvider._map_error() handles
HTTP_ERROR_CODES = [400, 401, 403, 429, 500, 502, 503, 504]

http_status_code_strategy = st.sampled_from(HTTP_ERROR_CODES)


def _make_provider() -> GeminiProvider:
    """Create a GeminiProvider instance with mocked SDK dependencies."""
    genai_mock = sys.modules["google.generativeai"]
    genai_mock.configure = MagicMock()
    genai_mock.GenerativeModel = MagicMock()

    config = {"api_key": "test-api-key-for-property-test"}
    logger_mock = MagicMock()
    return GeminiProvider(config, logger_mock)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestErrorStatusCodeInclusion:
    """
    Property 3: Error Status Code Inclusion.

    For any HTTP error response from the Gemini API, the raised ProviderError
    message SHALL contain the HTTP status code as a substring.

    **Validates: Requirements 2.5, 7.2**
    """

    @pytest.mark.property_test
    @given(status_code=http_status_code_strategy)
    @settings(max_examples=100, deadline=None)
    def test_provider_error_message_contains_status_code(self, status_code: int):
        """
        For any HTTP error status code in the supported set, calling
        GeminiProvider._map_error() with an exception whose string
        representation contains that status code must produce a ProviderError
        whose message also contains the status code as a substring.

        **Validates: Requirements 2.5, 7.2**
        """
        provider = _make_provider()

        # Simulate an HTTP error exception whose message contains the status code
        error = Exception(f"HTTP {status_code}: error response from API")

        result = provider._map_error(error, "req-property-test")

        assert isinstance(result, ProviderError), (
            f"_map_error must return a ProviderError, got {type(result)}"
        )
        assert str(status_code) in str(result), (
            f"ProviderError message must contain the HTTP status code '{status_code}'. "
            f"Got message: '{str(result)}'"
        )
