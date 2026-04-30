"""
Unit tests for luma/core/llm/config.py.

Covers:
- Field validators raise ValueError for invalid inputs (Requirements 6.2–6.6)
- from_dict round-trips correctly (Requirement 6.7)
- Default values are applied correctly (Requirement 6.1)
"""

import pytest
from luma.core.llm.config import LLMConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> LLMConfig:
    defaults = dict(api_key="test-api-key", model="gpt-4o")
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ---------------------------------------------------------------------------
# Default values — Requirement 6.1
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_temperature_default(self):
        cfg = make_config()
        assert cfg.temperature == 0.7

    def test_max_tokens_default(self):
        cfg = make_config()
        assert cfg.max_tokens == 1024

    def test_timeout_seconds_default(self):
        cfg = make_config()
        assert cfg.timeout_seconds == 30.0

    def test_max_retries_default(self):
        cfg = make_config()
        assert cfg.max_retries == 3

    def test_max_response_chars_default(self):
        cfg = make_config()
        assert cfg.max_response_chars == 4000

    def test_base_url_default_is_none(self):
        cfg = make_config()
        assert cfg.base_url is None

    def test_fallback_response_default_non_empty(self):
        cfg = make_config()
        assert cfg.fallback_response and cfg.fallback_response.strip()


# ---------------------------------------------------------------------------
# api_key validation — Requirement 6.6
# ---------------------------------------------------------------------------

class TestApiKeyValidation:
    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError):
            make_config(api_key="")

    def test_whitespace_only_api_key_raises(self):
        with pytest.raises(ValueError):
            make_config(api_key="   ")

    def test_valid_api_key_accepted(self):
        cfg = make_config(api_key="sk-abc123")
        assert cfg.api_key == "sk-abc123"


# ---------------------------------------------------------------------------
# temperature validation — Requirement 6.2
# ---------------------------------------------------------------------------

class TestTemperatureValidation:
    def test_temperature_below_range_raises(self):
        with pytest.raises(ValueError):
            make_config(temperature=-0.1)

    def test_temperature_above_range_raises(self):
        with pytest.raises(ValueError):
            make_config(temperature=2.1)

    def test_temperature_at_lower_bound_valid(self):
        cfg = make_config(temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_at_upper_bound_valid(self):
        cfg = make_config(temperature=2.0)
        assert cfg.temperature == 2.0


# ---------------------------------------------------------------------------
# max_tokens validation — Requirement 6.3
# ---------------------------------------------------------------------------

class TestMaxTokensValidation:
    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError):
            make_config(max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValueError):
            make_config(max_tokens=-1)

    def test_max_tokens_one_valid(self):
        cfg = make_config(max_tokens=1)
        assert cfg.max_tokens == 1


# ---------------------------------------------------------------------------
# timeout_seconds validation — Requirement 6.4
# ---------------------------------------------------------------------------

class TestTimeoutValidation:
    def test_timeout_zero_raises(self):
        with pytest.raises(ValueError):
            make_config(timeout_seconds=0.0)

    def test_timeout_negative_raises(self):
        with pytest.raises(ValueError):
            make_config(timeout_seconds=-1.0)

    def test_timeout_positive_valid(self):
        cfg = make_config(timeout_seconds=60.0)
        assert cfg.timeout_seconds == 60.0


# ---------------------------------------------------------------------------
# max_retries validation — Requirement 6.5
# ---------------------------------------------------------------------------

class TestMaxRetriesValidation:
    def test_max_retries_negative_raises(self):
        with pytest.raises(ValueError):
            make_config(max_retries=-1)

    def test_max_retries_zero_valid(self):
        cfg = make_config(max_retries=0)
        assert cfg.max_retries == 0

    def test_max_retries_positive_valid(self):
        cfg = make_config(max_retries=5)
        assert cfg.max_retries == 5


# ---------------------------------------------------------------------------
# from_dict — Requirement 6.7
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_from_dict_minimal(self):
        cfg = LLMConfig.from_dict({"api_key": "key-xyz", "model": "gpt-3.5-turbo"})
        assert cfg.api_key == "key-xyz"
        assert cfg.model == "gpt-3.5-turbo"
        assert cfg.temperature == 0.7

    def test_from_dict_full_round_trip(self):
        data = {
            "api_key": "sk-test",
            "model": "gpt-4o",
            "temperature": 1.2,
            "max_tokens": 512,
            "timeout_seconds": 15.0,
            "max_retries": 2,
            "max_response_chars": 2000,
            "fallback_response": "Sorry, unavailable.",
            "base_url": "http://localhost:11434",
        }
        cfg = LLMConfig.from_dict(data)
        assert cfg.api_key == data["api_key"]
        assert cfg.model == data["model"]
        assert cfg.temperature == data["temperature"]
        assert cfg.max_tokens == data["max_tokens"]
        assert cfg.timeout_seconds == data["timeout_seconds"]
        assert cfg.max_retries == data["max_retries"]
        assert cfg.max_response_chars == data["max_response_chars"]
        assert cfg.fallback_response == data["fallback_response"]
        assert cfg.base_url == data["base_url"]

    def test_from_dict_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            LLMConfig.from_dict({"api_key": "", "model": "gpt-4o"})
