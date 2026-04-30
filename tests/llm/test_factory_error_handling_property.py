"""
Property-based test for ProviderFactory error handling.

**Validates: Requirements 4.4**

Property 7: Factory Error Handling
For any provider name string that is not in the factory's registry,
calling ProviderFactory.create() SHALL raise a ValueError with a message
listing the supported provider names.
"""

import sys
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st

# Mock google.generativeai before importing ProviderFactory (which imports GeminiProvider)
sys.modules.setdefault('google', MagicMock())
sys.modules.setdefault('google.generativeai', MagicMock())

from luma.core.llm.providers.provider_factory import ProviderFactory  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Known provider names that ARE in the registry — we must exclude these
_KNOWN_PROVIDERS = frozenset(ProviderFactory._REGISTRY.keys())  # {"gemini", "mock"}

# Generate arbitrary text strings that are NOT known provider names
unknown_provider_name = st.text(min_size=1).filter(
    lambda s: s not in _KNOWN_PROVIDERS
)

# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@pytest.mark.property_test
class TestFactoryErrorHandling:
    """
    Property 7: Factory Error Handling

    **Validates: Requirements 4.4**
    """

    @given(provider_name=unknown_provider_name)
    @settings(max_examples=100, deadline=None)
    def test_unknown_provider_raises_value_error(self, provider_name: str):
        """
        For any provider name not in the registry, ProviderFactory.create()
        SHALL raise a ValueError.

        **Validates: Requirements 4.4**
        """
        with pytest.raises(ValueError):
            ProviderFactory.create(provider_name, {})

    @given(provider_name=unknown_provider_name)
    @settings(max_examples=100, deadline=None)
    def test_error_message_contains_supported_providers(self, provider_name: str):
        """
        The ValueError message SHALL list all supported provider names so
        developers can quickly identify valid options.

        **Validates: Requirements 4.4**
        """
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create(provider_name, {})

        error_message = str(exc_info.value)
        for supported_name in ProviderFactory._REGISTRY.keys():
            assert supported_name in error_message, (
                f"Expected supported provider '{supported_name}' to appear in "
                f"error message, but got: {error_message!r}"
            )
