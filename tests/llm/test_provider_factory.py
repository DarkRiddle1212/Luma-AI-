"""
Unit tests for ProviderFactory.

Tests provider instantiation, registry lookup, config validation,
error handling for unknown providers, and extensibility via register().
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock google.generativeai before importing GeminiProvider (via ProviderFactory)
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from luma.core.llm.providers.provider_factory import ProviderFactory
from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.providers.provider_interface import LLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_logger():
    return MagicMock()


# ---------------------------------------------------------------------------
# create() — known providers
# ---------------------------------------------------------------------------

class TestProviderFactoryCreate:
    def test_create_mock_provider_returns_mock_instance(self):
        config = {"responses": []}
        provider = ProviderFactory.create("mock", config)
        assert isinstance(provider, MockProvider)

    def test_create_mock_provider_with_logger(self):
        config = {"responses": []}
        logger = _mock_logger()
        provider = ProviderFactory.create("mock", config, logger=logger)
        assert isinstance(provider, MockProvider)

    def test_create_gemini_provider_returns_gemini_instance(self):
        with patch('luma.core.llm.providers.gemini_provider.genai.configure'), \
             patch('luma.core.llm.providers.gemini_provider.genai.GenerativeModel', return_value=MagicMock()):
            config = {"api_key": "test-key-1234"}
            logger = _mock_logger()
            provider = ProviderFactory.create("gemini", config, logger=logger)
            assert isinstance(provider, GeminiProvider)

    def test_create_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create("openai", {})
        assert "openai" in str(exc_info.value)

    def test_create_unknown_provider_error_lists_supported(self):
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create("nonexistent", {})
        msg = str(exc_info.value)
        assert "gemini" in msg
        assert "mock" in msg

    def test_create_validates_config_before_instantiation(self):
        # Gemini requires api_key — missing it should raise ValueError
        with pytest.raises(ValueError):
            ProviderFactory.create("gemini", {}, logger=_mock_logger())

    def test_create_mock_no_logger_still_works(self):
        provider = ProviderFactory.create("mock", {})
        assert isinstance(provider, MockProvider)


# ---------------------------------------------------------------------------
# register() — extensibility
# ---------------------------------------------------------------------------

class TestProviderFactoryRegister:
    def setup_method(self):
        """Snapshot registry before each test so we can restore it."""
        self._original_registry = dict(ProviderFactory._REGISTRY)

    def teardown_method(self):
        """Restore registry after each test."""
        ProviderFactory._REGISTRY.clear()
        ProviderFactory._REGISTRY.update(self._original_registry)

    def test_register_new_provider_makes_it_creatable(self):
        class DummyProvider(LLMProvider):
            def __init__(self, config, logger=None):
                pass

            def generate(self, prompt, options):
                return {"text": "dummy", "model": "dummy", "prompt_tokens": 0,
                        "completion_tokens": 0, "provider": "dummy"}

            def validate_config(self, config):
                return True

        ProviderFactory.register("dummy", DummyProvider)
        provider = ProviderFactory.create("dummy", {})
        assert isinstance(provider, DummyProvider)

    def test_register_overwrites_existing_provider(self):
        class AltMock(LLMProvider):
            def __init__(self, config, logger=None):
                pass

            def generate(self, prompt, options):
                return {}

            def validate_config(self, config):
                return True

        ProviderFactory.register("mock", AltMock)
        provider = ProviderFactory.create("mock", {})
        assert isinstance(provider, AltMock)

    def test_registry_contains_gemini_and_mock_by_default(self):
        assert "gemini" in ProviderFactory._REGISTRY
        assert "mock" in ProviderFactory._REGISTRY
