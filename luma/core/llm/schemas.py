"""
LLM Integration Data Schemas.

Defines PromptContext, LLMRequest, LLMResponse, and ParsedResponse models,
plus exception classes for the LLM module.

Uses Pydantic if available, otherwise dataclasses with __post_init__ validation.
Follows the dual-path pattern from luma/core/personalization/schemas.py.
"""

from typing import Dict, List

try:
    from pydantic import BaseModel, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class LLMGenerationError(Exception):
    """Raised by LLMEngine when generation fails."""


class LLMClientError(Exception):
    """Raised by LLMClient when an API call fails."""


class PromptBuildError(Exception):
    """Raised by PromptBuilder when a prompt cannot be constructed."""


class ResponseParseError(Exception):
    """Raised by ResponseParser when a response cannot be parsed."""


# ---------------------------------------------------------------------------
# Data models — Pydantic path
# ---------------------------------------------------------------------------

if _USE_PYDANTIC:
    class PromptContext(BaseModel):
        """Context used by PromptBuilder to construct a structured prompt."""

        system_instructions: str
        user_profile: str
        relevant_memories: List[str]
        current_input: str
        output_constraints: str

    class LLMRequest(BaseModel):
        """Input schema for a single LLM generation request."""

        prompt_context: PromptContext
        model: str
        temperature: float
        max_tokens: int
        request_id: str

        @field_validator("temperature")
        @classmethod
        def temperature_in_range(cls, v: float) -> float:
            if not 0.0 <= v <= 2.0:
                raise ValueError(
                    f"temperature must be in [0.0, 2.0], got {v}"
                )
            return v

        @field_validator("max_tokens")
        @classmethod
        def max_tokens_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError(
                    f"max_tokens must be a positive integer, got {v}"
                )
            return v

    class LLMResponse(BaseModel):
        """Raw output schema carrying the provider's response and metadata."""

        request_id: str
        raw_text: str
        model: str
        prompt_tokens: int
        completion_tokens: int
        provider: str

        @field_validator("prompt_tokens", "completion_tokens")
        @classmethod
        def tokens_non_negative(cls, v: int) -> int:
            if v < 0:
                raise ValueError(
                    f"token counts must be non-negative, got {v}"
                )
            return v

    class ParsedResponse(BaseModel):
        """Cleaned and validated response returned to callers."""

        request_id: str
        text: str
        is_valid: bool
        validation_notes: List[str]
        token_usage: Dict[str, int]
        truncated: bool

# ---------------------------------------------------------------------------
# Data models — dataclasses fallback path
# ---------------------------------------------------------------------------

else:
    from dataclasses import dataclass, field

    @dataclass
    class PromptContext:
        """Context used by PromptBuilder to construct a structured prompt."""

        system_instructions: str
        user_profile: str
        relevant_memories: List[str]
        current_input: str
        output_constraints: str

    @dataclass
    class LLMRequest:
        """Input schema for a single LLM generation request."""

        prompt_context: "PromptContext"
        model: str
        temperature: float
        max_tokens: int
        request_id: str

        def __post_init__(self) -> None:
            if not 0.0 <= self.temperature <= 2.0:
                raise ValueError(
                    f"temperature must be in [0.0, 2.0], got {self.temperature}"
                )
            if self.max_tokens <= 0:
                raise ValueError(
                    f"max_tokens must be a positive integer, got {self.max_tokens}"
                )

    @dataclass
    class LLMResponse:
        """Raw output schema carrying the provider's response and metadata."""

        request_id: str
        raw_text: str
        model: str
        prompt_tokens: int
        completion_tokens: int
        provider: str

        def __post_init__(self) -> None:
            if self.prompt_tokens < 0:
                raise ValueError(
                    f"prompt_tokens must be non-negative, got {self.prompt_tokens}"
                )
            if self.completion_tokens < 0:
                raise ValueError(
                    f"completion_tokens must be non-negative, got {self.completion_tokens}"
                )

    @dataclass
    class ParsedResponse:
        """Cleaned and validated response returned to callers."""

        request_id: str
        text: str
        is_valid: bool
        validation_notes: List[str]
        token_usage: Dict[str, int]
        truncated: bool
