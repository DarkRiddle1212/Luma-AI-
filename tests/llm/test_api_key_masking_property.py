"""
Property-based test for GeminiProvider API key masking.

**Validates: Requirements 8.3**

Property 11: API Key Masking
For any API key string:
- If length > 4: all but the last 4 characters are asterisks, last 4 are preserved.
- If length <= 4: result is exactly "****".

Feature: gemini-provider-integration
Property 11: API key masking
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st

# ---------------------------------------------------------------------------
# Patch google.generativeai before importing GeminiProvider
# ---------------------------------------------------------------------------

_genai_mock = MagicMock()
_genai_mock.configure = MagicMock()
_genai_mock.GenerativeModel = MagicMock(return_value=MagicMock())
_genai_mock.types = MagicMock()

sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.generativeai", _genai_mock)

from luma.core.llm.providers.gemini_provider import GeminiProvider  # noqa: E402
from luma.core.structured_logger import StructuredLogger  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider() -> GeminiProvider:
    """Create a minimal GeminiProvider instance for testing _mask_api_key."""
    logger = MagicMock(spec=StructuredLogger)
    config = {"api_key": "test-key-1234"}
    return GeminiProvider(config=config, logger=logger)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keys of length <= 4 (edge case)
short_key_strategy = st.text(min_size=0, max_size=4)

# Keys of length > 4
long_key_strategy = st.text(min_size=5)

# All lengths combined
any_key_strategy = st.text(min_size=0)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestApiKeyMaskingProperty:
    """
    Property 11: API Key Masking

    **Validates: Requirements 8.3**
    """

    @pytest.mark.property_test
    @given(key=long_key_strategy)
    @settings(max_examples=100, deadline=None)
    def test_long_key_last_four_preserved(self, key: str):
        """
        For keys with length > 4: last 4 characters are preserved unchanged.

        **Validates: Requirements 8.3**
        """
        provider = _make_provider()
        result = provider._mask_api_key(key)
        assert result[-4:] == key[-4:], (
            f"Last 4 chars must be preserved. key={key!r}, result={result!r}"
        )

    @pytest.mark.property_test
    @given(key=long_key_strategy)
    @settings(max_examples=100, deadline=None)
    def test_long_key_prefix_is_all_asterisks(self, key: str):
        """
        For keys with length > 4: all characters except the last 4 are asterisks.

        **Validates: Requirements 8.3**
        """
        provider = _make_provider()
        result = provider._mask_api_key(key)
        prefix = result[:-4]
        assert all(c == "*" for c in prefix), (
            f"All chars before last 4 must be '*'. key={key!r}, result={result!r}"
        )

    @pytest.mark.property_test
    @given(key=long_key_strategy)
    @settings(max_examples=100, deadline=None)
    def test_long_key_result_length_matches_input(self, key: str):
        """
        For keys with length > 4: masked result has the same length as the input.

        **Validates: Requirements 8.3**
        """
        provider = _make_provider()
        result = provider._mask_api_key(key)
        assert len(result) == len(key), (
            f"Masked key length must equal input length. key={key!r}, result={result!r}"
        )

    @pytest.mark.property_test
    @given(key=short_key_strategy)
    @settings(max_examples=100, deadline=None)
    def test_short_key_returns_four_asterisks(self, key: str):
        """
        For keys with length <= 4: result is exactly "****".

        **Validates: Requirements 8.3**
        """
        provider = _make_provider()
        result = provider._mask_api_key(key)
        assert result == "****", (
            f"Short key must mask to '****'. key={key!r}, result={result!r}"
        )
