"""
Unit tests for luma/core/llm/schemas.py.

Covers:
- LLMRequest temperature validation (Requirements 1.5, 1.6)
- LLMRequest max_tokens validation (Requirements 1.6)
- Exception class hierarchy (Requirements 1.7–1.10)
"""

import pytest
from luma.core.llm.schemas import (
    PromptContext,
    LLMRequest,
    LLMGenerationError,
    LLMClientError,
    PromptBuildError,
    ResponseParseError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prompt_context() -> PromptContext:
    return PromptContext(
        system_instructions="You are a helpful assistant.",
        user_profile="User prefers concise answers.",
        relevant_memories=[],
        current_input="Hello",
        output_constraints="Be brief.",
    )


def make_llm_request(**overrides) -> LLMRequest:
    defaults = dict(
        prompt_context=make_prompt_context(),
        model="gpt-4o",
        temperature=1.0,
        max_tokens=256,
        request_id="test-req-001",
    )
    defaults.update(overrides)
    return LLMRequest(**defaults)


# ---------------------------------------------------------------------------
# Temperature validation — Requirement 1.5
# ---------------------------------------------------------------------------

class TestTemperatureValidation:
    def test_temperature_below_range_raises(self):
        with pytest.raises(ValueError):
            make_llm_request(temperature=-0.1)

    def test_temperature_above_range_raises(self):
        with pytest.raises(ValueError):
            make_llm_request(temperature=2.1)

    def test_temperature_at_lower_bound_is_valid(self):
        req = make_llm_request(temperature=0.0)
        assert req.temperature == 0.0

    def test_temperature_at_upper_bound_is_valid(self):
        req = make_llm_request(temperature=2.0)
        assert req.temperature == 2.0

    def test_temperature_in_middle_is_valid(self):
        req = make_llm_request(temperature=1.0)
        assert req.temperature == 1.0


# ---------------------------------------------------------------------------
# max_tokens validation — Requirement 1.6
# ---------------------------------------------------------------------------

class TestMaxTokensValidation:
    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError):
            make_llm_request(max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValueError):
            make_llm_request(max_tokens=-1)

    def test_max_tokens_one_is_valid(self):
        req = make_llm_request(max_tokens=1)
        assert req.max_tokens == 1


# ---------------------------------------------------------------------------
# Exception class hierarchy — Requirements 1.7–1.10
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    def test_llm_generation_error_is_exception(self):
        assert issubclass(LLMGenerationError, Exception)

    def test_llm_client_error_is_exception(self):
        assert issubclass(LLMClientError, Exception)

    def test_prompt_build_error_is_exception(self):
        assert issubclass(PromptBuildError, Exception)

    def test_response_parse_error_is_exception(self):
        assert issubclass(ResponseParseError, Exception)
