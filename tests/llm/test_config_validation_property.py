"""
Property-based test for LLMConfig configuration validation.

**Validates: Requirements 3.4**

Property 6: Configuration Validation
For any string that is empty or contains only whitespace, setting it as
`provider_name` in `LLMConfig` SHALL raise a `ValueError`.

Feature: gemini-provider-integration
Property 6: Configuration validation
"""

import pytest
from hypothesis import given, settings, strategies as st

from luma.core.llm.config import LLMConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate empty string or whitespace-only strings (spaces, tabs, newlines)
whitespace_chars = st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"])

invalid_provider_name_strategy = st.one_of(
    # Empty string
    st.just(""),
    # Whitespace-only strings of various lengths (1–20 chars)
    st.text(alphabet=whitespace_chars, min_size=1, max_size=20),
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

class TestConfigValidationProperty:
    """
    Property 6: Configuration Validation

    **Validates: Requirements 3.4**

    For any empty or whitespace-only string used as provider_name,
    LLMConfig must raise a ValueError.
    """

    @pytest.mark.property_test
    @given(invalid_name=invalid_provider_name_strategy)
    @settings(max_examples=100, deadline=None)
    def test_empty_or_whitespace_provider_name_raises_value_error(self, invalid_name: str):
        """
        For any empty or whitespace-only provider_name, LLMConfig raises ValueError.

        **Validates: Requirements 3.4**
        """
        with pytest.raises(ValueError):
            LLMConfig(
                api_key="test-api-key",
                model="gemini-2.5-flash",
                provider_name=invalid_name,
            )
