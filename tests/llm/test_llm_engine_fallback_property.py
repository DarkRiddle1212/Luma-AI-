"""
Property-based test for LLMEngine fallback determinism.

**Validates: Requirements 5.4, 9.1, 12.7**

Property: when MockLLMClient always raises LLMClientError, generate() always
returns a ParsedResponse with is_valid=False and "llm_client_error" in
validation_notes, regardless of the request parameters.
"""

import pytest
from hypothesis import given, settings, strategies as st

from luma.core.llm.llm_engine import LLMEngine
from luma.core.llm.llm_client import LLMClient
from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.response_parser import ResponseParser
from luma.core.llm.schemas import (
    LLMClientError,
    LLMRequest,
    LLMResponse,
    ParsedResponse,
    PromptContext,
)
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class AlwaysFailingLLMClient(LLMClient):
    """MockLLMClient that always raises LLMClientError."""

    def __init__(self, error_message: str = "simulated provider failure") -> None:
        self._error_message = error_message

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMClientError(self._error_message)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

prompt_context_strategy = st.builds(
    PromptContext,
    system_instructions=non_empty_text,
    user_profile=st.text(),
    relevant_memories=st.lists(st.text(), max_size=5),
    current_input=non_empty_text,
    output_constraints=st.text(),
)

llm_request_strategy = st.builds(
    LLMRequest,
    prompt_context=prompt_context_strategy,
    model=st.sampled_from(["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]),
    temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    max_tokens=st.integers(min_value=1, max_value=4096),
    request_id=st.uuids().map(str),
)

error_message_strategy = st.text(min_size=1, max_size=200)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestLLMEngineFallbackDeterminism:
    """
    Property: LLMEngine always returns a fallback ParsedResponse when
    LLMClient raises LLMClientError.

    **Validates: Requirements 5.4, 9.1, 12.7**
    """

    def _make_engine(self, error_message: str = "simulated failure") -> LLMEngine:
        return LLMEngine(
            prompt_builder=PromptBuilder(),
            llm_client=AlwaysFailingLLMClient(error_message),
            response_parser=ResponseParser(),
            logger=StructuredLogger("test_fallback_property"),
            fallback_response="Fallback text for testing.",
        )

    @pytest.mark.property_test
    @given(request=llm_request_strategy)
    @settings(max_examples=50, deadline=None)
    def test_fallback_is_valid_false(self, request: LLMRequest):
        """
        For any valid LLMRequest, when LLMClient always raises LLMClientError,
        generate() returns a ParsedResponse with is_valid=False.

        **Validates: Requirements 5.4, 9.1, 12.7**
        """
        engine = self._make_engine()
        result = engine.generate(request)
        assert isinstance(result, ParsedResponse)
        assert result.is_valid is False

    @pytest.mark.property_test
    @given(request=llm_request_strategy)
    @settings(max_examples=50, deadline=None)
    def test_fallback_has_llm_client_error_note(self, request: LLMRequest):
        """
        For any valid LLMRequest, when LLMClient always raises LLMClientError,
        generate() returns a ParsedResponse with "llm_client_error" in validation_notes.

        **Validates: Requirements 5.4, 9.1, 12.7**
        """
        engine = self._make_engine()
        result = engine.generate(request)
        assert "llm_client_error" in result.validation_notes

    @pytest.mark.property_test
    @given(request=llm_request_strategy, error_msg=error_message_strategy)
    @settings(max_examples=50, deadline=None)
    def test_fallback_does_not_raise(self, request: LLMRequest, error_msg: str):
        """
        For any valid LLMRequest and any LLMClientError message,
        generate() never raises an exception.

        **Validates: Requirements 5.4, 9.1, 12.7**
        """
        engine = self._make_engine(error_message=error_msg)
        # Should not raise
        result = engine.generate(request)
        assert result is not None

    @pytest.mark.property_test
    @given(request=llm_request_strategy)
    @settings(max_examples=50, deadline=None)
    def test_fallback_request_id_matches(self, request: LLMRequest):
        """
        The fallback ParsedResponse carries the same request_id as the input request.

        **Validates: Requirements 5.4, 9.1**
        """
        engine = self._make_engine()
        result = engine.generate(request)
        assert result.request_id == request.request_id
