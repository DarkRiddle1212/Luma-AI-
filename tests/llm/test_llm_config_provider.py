"""
Unit tests for provider-related configuration in luma/core/llm/config.py.

Covers:
- LLMConfig provider_name and provider_config fields (Requirements 3.1, 3.2)
- provider_name validation (Requirement 3.4)
- load_llm_config_from_env() environment variable loading (Requirements 3.6, 3.7)
- Default value application (Requirement 3.3)
- Missing GEMINI_API_KEY raises ValueError (Requirement 3.5)
"""

import pytest
from unittest.mock import patch
from luma.core.llm.config import LLMConfig, load_llm_config_from_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> LLMConfig:
    defaults = dict(api_key="test-api-key", model="gemini-2.5-flash")
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ---------------------------------------------------------------------------
# provider_name and provider_config fields — Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

class TestProviderFields:
    def test_provider_name_default_is_gemini(self):
        cfg = make_config()
        assert cfg.provider_name == "gemini"

    def test_provider_config_default_is_empty_dict(self):
        cfg = make_config()
        assert cfg.provider_config == {}

    def test_provider_name_can_be_set(self):
        cfg = make_config(provider_name="mock")
        assert cfg.provider_name == "mock"

    def test_provider_config_can_be_set(self):
        pc = {"api_key": "abc", "model": "gemini-2.5-flash"}
        cfg = make_config(provider_config=pc)
        assert cfg.provider_config == pc

    def test_provider_config_accepts_arbitrary_keys(self):
        pc = {"api_key": "key", "timeout": 60.0, "log_prompts": True}
        cfg = make_config(provider_config=pc)
        assert cfg.provider_config["timeout"] == 60.0
        assert cfg.provider_config["log_prompts"] is True


# ---------------------------------------------------------------------------
# provider_name validation — Requirement 3.4
# ---------------------------------------------------------------------------

class TestProviderNameValidation:
    def test_empty_provider_name_raises(self):
        with pytest.raises(ValueError):
            make_config(provider_name="")

    def test_whitespace_only_provider_name_raises(self):
        with pytest.raises(ValueError):
            make_config(provider_name="   ")

    def test_valid_provider_name_accepted(self):
        cfg = make_config(provider_name="openai")
        assert cfg.provider_name == "openai"

    def test_single_char_provider_name_accepted(self):
        cfg = make_config(provider_name="x")
        assert cfg.provider_name == "x"


# ---------------------------------------------------------------------------
# load_llm_config_from_env — Requirements 3.5, 3.6, 3.7
# ---------------------------------------------------------------------------

class TestLoadLLMConfigFromEnv:
    def test_gemini_provider_with_api_key(self):
        env = {"GEMINI_API_KEY": "my-gemini-key"}
        with patch.dict("os.environ", env, clear=False):
            # Remove any existing LLM_PROVIDER to use default
            with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}, clear=False):
                cfg = load_llm_config_from_env()
        assert cfg.provider_name == "gemini"
        assert cfg.api_key == "my-gemini-key"
        assert cfg.provider_config["api_key"] == "my-gemini-key"

    def test_missing_gemini_api_key_raises(self):
        env = {"LLM_PROVIDER": "gemini"}
        # Ensure GEMINI_API_KEY is not set
        with patch.dict("os.environ", env, clear=False):
            with patch("os.environ.get", side_effect=lambda k, d=None: env.get(k, d) if k == "LLM_PROVIDER" else None):
                # Use a cleaner approach: remove GEMINI_API_KEY from env
                import os
                original = os.environ.pop("GEMINI_API_KEY", None)
                try:
                    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                        load_llm_config_from_env()
                finally:
                    if original is not None:
                        os.environ["GEMINI_API_KEY"] = original

    def test_mock_provider_does_not_require_api_key(self):
        env = {"LLM_PROVIDER": "mock"}
        with patch.dict("os.environ", env, clear=False):
            cfg = load_llm_config_from_env()
        assert cfg.provider_name == "mock"

    def test_unknown_provider_raises(self):
        env = {"LLM_PROVIDER": "unknown-provider-xyz"}
        with patch.dict("os.environ", env, clear=False):
            with pytest.raises(ValueError):
                load_llm_config_from_env()

    def test_default_provider_is_gemini(self):
        import os
        original = os.environ.pop("LLM_PROVIDER", None)
        try:
            env = {"GEMINI_API_KEY": "key-for-default-test"}
            with patch.dict("os.environ", env, clear=False):
                cfg = load_llm_config_from_env()
            assert cfg.provider_name == "gemini"
        finally:
            if original is not None:
                os.environ["LLM_PROVIDER"] = original


# ---------------------------------------------------------------------------
# Default values from env — Requirement 3.3, 3.7
# ---------------------------------------------------------------------------

class TestGeminiDefaultValues:
    def _load_with_only_api_key(self) -> LLMConfig:
        """Load config with only GEMINI_API_KEY set, no other GEMINI_* vars."""
        import os
        keys_to_remove = [
            "GEMINI_MODEL", "GEMINI_TIMEOUT", "GEMINI_MAX_TOKENS",
            "GEMINI_TEMPERATURE", "GEMINI_LOG_PROMPTS",
        ]
        saved = {k: os.environ.pop(k, None) for k in keys_to_remove}
        try:
            with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}, clear=False):
                return load_llm_config_from_env()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_default_model_is_gemini_flash(self):
        cfg = self._load_with_only_api_key()
        assert cfg.provider_config["model"] == "gemini-2.5-flash"

    def test_default_timeout_is_30(self):
        cfg = self._load_with_only_api_key()
        assert cfg.provider_config["timeout"] == 30.0

    def test_default_max_tokens_is_1024(self):
        cfg = self._load_with_only_api_key()
        assert cfg.provider_config["max_tokens"] == 1024

    def test_default_temperature_is_0_4(self):
        cfg = self._load_with_only_api_key()
        assert cfg.provider_config["temperature"] == 0.4

    def test_default_log_prompts_is_false(self):
        cfg = self._load_with_only_api_key()
        assert cfg.provider_config["log_prompts"] is False

    def test_custom_model_from_env(self):
        import os
        saved = os.environ.pop("GEMINI_MODEL", None)
        try:
            with patch.dict("os.environ", {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_MODEL": "gemini-1.5-pro",
            }, clear=False):
                cfg = load_llm_config_from_env()
            assert cfg.provider_config["model"] == "gemini-1.5-pro"
        finally:
            if saved is not None:
                os.environ["GEMINI_MODEL"] = saved

    def test_custom_timeout_from_env(self):
        import os
        saved = os.environ.pop("GEMINI_TIMEOUT", None)
        try:
            with patch.dict("os.environ", {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_TIMEOUT": "60.0",
            }, clear=False):
                cfg = load_llm_config_from_env()
            assert cfg.provider_config["timeout"] == 60.0
        finally:
            if saved is not None:
                os.environ["GEMINI_TIMEOUT"] = saved

    def test_log_prompts_true_from_env(self):
        import os
        saved = os.environ.pop("GEMINI_LOG_PROMPTS", None)
        try:
            with patch.dict("os.environ", {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "GEMINI_LOG_PROMPTS": "true",
            }, clear=False):
                cfg = load_llm_config_from_env()
            assert cfg.provider_config["log_prompts"] is True
        finally:
            if saved is not None:
                os.environ["GEMINI_LOG_PROMPTS"] = saved
