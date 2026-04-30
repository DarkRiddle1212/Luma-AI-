"""
Unit tests for MockProvider.

Covers Requirements: 11.2, 11.4, 11.5, 11.6, 11.7
"""

import time
import pytest

from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.providers.provider_interface import ProviderError


SAMPLE_RESPONSE = {"text": "hello", "model": "mock", "prompt_tokens": 5, "completion_tokens": 3, "provider": "mock"}
SAMPLE_RESPONSE_2 = {"text": "world", "model": "mock", "prompt_tokens": 4, "completion_tokens": 2, "provider": "mock"}


def make_provider(responses=None, delay=0.0, error_mode=None):
    config = {"responses": responses or [], "delay": delay}
    if error_mode is not None:
        config["error_mode"] = error_mode
    return MockProvider(config)


class TestSequentialResponseReturn:
    """Requirement 11.2 — responses returned in order."""

    def test_returns_first_response(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE])
        result = provider.generate("prompt", {})
        assert result == SAMPLE_RESPONSE

    def test_returns_responses_in_order(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE, SAMPLE_RESPONSE_2])
        assert provider.generate("p", {}) == SAMPLE_RESPONSE
        assert provider.generate("p", {}) == SAMPLE_RESPONSE_2

    def test_each_call_advances_index(self):
        responses = [{"text": str(i)} for i in range(5)]
        provider = make_provider(responses=responses)
        for i in range(5):
            assert provider.generate("p", {})["text"] == str(i)


class TestExhaustedResponses:
    """Requirement 11.4 — raises ProviderError when responses are exhausted."""

    def test_raises_when_no_responses_configured(self):
        provider = make_provider(responses=[])
        with pytest.raises(ProviderError, match="no more mock responses available"):
            provider.generate("p", {})

    def test_raises_after_all_responses_consumed(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE])
        provider.generate("p", {})
        with pytest.raises(ProviderError, match="no more mock responses available"):
            provider.generate("p", {})

    def test_exhausted_error_is_not_transient(self):
        provider = make_provider(responses=[])
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("p", {})
        assert exc_info.value.is_transient is False


class TestDelaySimulation:
    """Requirement 11.5 — delay > 0 causes sleep before returning."""

    def test_no_delay_by_default(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE])
        start = time.monotonic()
        provider.generate("p", {})
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_delay_is_applied(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE], delay=0.1)
        start = time.monotonic()
        provider.generate("p", {})
        elapsed = time.monotonic() - start
        assert elapsed >= 0.1

    def test_zero_delay_does_not_sleep(self):
        provider = make_provider(responses=[SAMPLE_RESPONSE], delay=0.0)
        start = time.monotonic()
        provider.generate("p", {})
        elapsed = time.monotonic() - start
        assert elapsed < 0.1


class TestErrorModeInjection:
    """Requirements 11.6, 11.7 — error_mode raises ProviderError with correct message and transience."""

    def test_error_mode_raises_provider_error(self):
        provider = make_provider(error_mode="something went wrong")
        with pytest.raises(ProviderError, match="something went wrong"):
            provider.generate("p", {})

    def test_error_mode_message_is_preserved(self):
        msg = "custom error message"
        provider = make_provider(error_mode=msg)
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("p", {})
        assert str(exc_info.value) == msg

    def test_transient_error_mode_sets_is_transient_true(self):
        provider = make_provider(error_mode="transient network failure")
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("p", {})
        assert exc_info.value.is_transient is True

    def test_non_transient_error_mode_sets_is_transient_false(self):
        provider = make_provider(error_mode="permanent auth failure")
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("p", {})
        assert exc_info.value.is_transient is False

    def test_error_mode_takes_priority_over_responses(self):
        """error_mode should raise even when responses are available."""
        provider = make_provider(responses=[SAMPLE_RESPONSE], error_mode="forced error")
        with pytest.raises(ProviderError, match="forced error"):
            provider.generate("p", {})

    def test_transient_keyword_case_insensitive(self):
        provider = make_provider(error_mode="TRANSIENT_TIMEOUT")
        with pytest.raises(ProviderError) as exc_info:
            provider.generate("p", {})
        assert exc_info.value.is_transient is True


class TestValidateConfig:
    """Requirement 11.2 — validate_config always returns True."""

    def test_validate_config_returns_true_for_empty_config(self):
        provider = make_provider()
        assert provider.validate_config({}) is True

    def test_validate_config_returns_true_for_any_config(self):
        provider = make_provider()
        assert provider.validate_config({"api_key": "abc", "model": "x"}) is True
