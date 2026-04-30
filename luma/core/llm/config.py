"""
LLM Integration Configuration.

Defines LLMConfig carrying API key, model name, and generation parameters.
Uses Pydantic if available, otherwise dataclasses with __post_init__ validation.
Follows the dual-path pattern from luma/core/personalization/schemas.py.
"""

import os
from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False

_DEFAULT_FALLBACK = (
    "I'm having trouble generating a response right now. "
    "Please try again in a moment."
)

if _USE_PYDANTIC:
    class LLMConfig(BaseModel):
        """Configuration object for the LLM integration layer."""

        api_key: str
        model: str
        temperature: float = 0.7
        max_tokens: int = 1024
        timeout_seconds: float = 30.0
        max_retries: int = 3
        max_response_chars: int = 4000
        fallback_response: str = _DEFAULT_FALLBACK
        base_url: Optional[str] = None
        provider_name: str = "gemini"
        provider_config: Dict[str, Any] = Field(default_factory=dict)

        @field_validator("api_key")
        @classmethod
        def api_key_non_empty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("api_key must be a non-empty, non-whitespace string")
            return v

        @field_validator("temperature")
        @classmethod
        def temperature_in_range(cls, v: float) -> float:
            if not 0.0 <= v <= 2.0:
                raise ValueError(f"temperature must be in [0.0, 2.0], got {v}")
            return v

        @field_validator("max_tokens")
        @classmethod
        def max_tokens_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError(f"max_tokens must be a positive integer, got {v}")
            return v

        @field_validator("timeout_seconds")
        @classmethod
        def timeout_positive(cls, v: float) -> float:
            if v <= 0:
                raise ValueError(f"timeout_seconds must be a positive float, got {v}")
            return v

        @field_validator("max_retries")
        @classmethod
        def max_retries_non_negative(cls, v: int) -> int:
            if v < 0:
                raise ValueError(f"max_retries must be a non-negative integer, got {v}")
            return v

        @field_validator("provider_name")
        @classmethod
        def provider_name_non_empty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("provider_name must be a non-empty, non-whitespace string")
            return v

        @classmethod
        def from_dict(cls, data: dict) -> "LLMConfig":
            return cls(**data)

else:
    from dataclasses import dataclass, field

    @dataclass
    class LLMConfig:
        """Configuration object for the LLM integration layer."""

        api_key: str
        model: str
        temperature: float = 0.7
        max_tokens: int = 1024
        timeout_seconds: float = 30.0
        max_retries: int = 3
        max_response_chars: int = 4000
        fallback_response: str = _DEFAULT_FALLBACK
        base_url: Optional[str] = None
        provider_name: str = "gemini"
        provider_config: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if not self.api_key or not self.api_key.strip():
                raise ValueError("api_key must be a non-empty, non-whitespace string")
            if not 0.0 <= self.temperature <= 2.0:
                raise ValueError(
                    f"temperature must be in [0.0, 2.0], got {self.temperature}"
                )
            if self.max_tokens <= 0:
                raise ValueError(
                    f"max_tokens must be a positive integer, got {self.max_tokens}"
                )
            if self.timeout_seconds <= 0:
                raise ValueError(
                    f"timeout_seconds must be a positive float, got {self.timeout_seconds}"
                )
            if self.max_retries < 0:
                raise ValueError(
                    f"max_retries must be a non-negative integer, got {self.max_retries}"
                )
            if not self.provider_name or not self.provider_name.strip():
                raise ValueError("provider_name must be a non-empty, non-whitespace string")

        @classmethod
        def from_dict(cls, data: dict) -> "LLMConfig":
            return cls(**data)


def load_llm_config_from_env() -> LLMConfig:
    """Load LLMConfig from environment variables.

    Reads LLM_PROVIDER (default "gemini") and provider-specific env vars.
    Raises ValueError if required variables are missing.
    """
    provider_name = os.getenv("LLM_PROVIDER", "gemini")

    if provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required when LLM_PROVIDER=gemini"
            )
        provider_config: Dict[str, Any] = {
            "api_key": api_key,
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
            "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.4")),
            "log_prompts": os.getenv("GEMINI_LOG_PROMPTS", "false").lower() == "true",
        }
    elif provider_name == "mock":
        provider_config = {"responses": [], "delay": 0.0, "error_mode": None}
        api_key = "mock-key"
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    return LLMConfig(
        api_key=provider_config.get("api_key", "mock-key"),
        model=provider_config.get("model", "default-model"),
        temperature=provider_config.get("temperature", 0.7),
        max_tokens=provider_config.get("max_tokens", 1024),
        timeout_seconds=provider_config.get("timeout", 30.0),
        provider_name=provider_name,
        provider_config=provider_config,
    )
