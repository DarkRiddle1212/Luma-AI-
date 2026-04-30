"""
Unit tests for luma.core.llm.llm_engine.LLMEngine.

**Validates: Requirements 5.3, 5.4, 5.5, 5.6, 5.7, 5.10, 11.3, 11.4**
"""

import pytest

from luma.core.llm.llm_engine import LLMEngine
from luma.core.llm.llm_client import LLMClient
from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.response_parser import ResponseParser
from luma.core.llm.schemas import (
    LLMClientError,
    LLMGenerationError,
    LLMRequest,
    LLMResponse,
    ParsedResponse,
    PromptBuildError,
    PromptContext,
    ResponseParseError,
)
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Configurable mock LLMClient — no real HTTP calls."""

    def __init__(self, response: LLMResponse = None, raises: Exception = None):
        self._response = response
        self._raises = raises
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        # Return a copy with the request's request_id so the pipeline stays consistent
        r = self._response
        return LLMResponse(
            request_id=request.request_id,
            raw_text=r.raw_text,
            model=r.model,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            provider=r.provider,
        )


class SpyPromptBuilder(PromptBuilder):
    """PromptBuilder that records calls and optionally raises."""

    def __init__(self, raises: Exception = None):
        self._raises = raises
        self.call_count = 0
        self.last_context = None

    def build(self, context: PromptContext) -> str:
        self.call_count += 1
        self.last_context = context
        if self._raises is not None:
            raise self._raises
        return super().build(context)


class SpyResponseParser(ResponseParser):
    """ResponseParser that records calls and optionally raises."""

    def __init__(self, raises: Exception = None):
        super().__init__()
        self._raises = raises
        self.call_count = 0
        self.last_response = None

    def parse(self, response: LLMResponse) -> ParsedResponse:
        self.call_count += 1
        self.last_response = response
        if self._raises is not None:
            raise self._raises
        return super().parse(response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_context(**kwargs) -> PromptContext:
    defaults = dict(
        system_instructions="You are a helpful assistant.",
        user_profile="Prefers concise answers.",
        relevant_memories=[],
        current_input="What is the capital of France?",
        output_constraints="Be brief.",
    )
    defaults.update(kwargs)
    return PromptContext(**defaults)


def make_request(**kwargs) -> LLMRequest:
    defaults = dict(
        prompt_context=make_context(),
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=512,
        request_id="test-req-001",
    )
    defaults.update(kwargs)
    return LLMRequest(**defaults)


def make_llm_response(request_id: str = "test-req-001", text: str = "Paris.") -> LLMResponse:
    return LLMResponse(
        request_id=request_id,
        raw_text=text,
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
        provider="openai",
    )


def make_engine(
    prompt_builder=None,
    llm_client=None,
    response_parser=None,
    fallback_response="Fallback text.",
) -> LLMEngine:
    return LLMEngine(
        prompt_builder=prompt_builder or PromptBuilder(),
        llm_client=llm_client or MockLLMClient(response=make_llm_response()),
        response_parser=response_parser or ResponseParser(),
        logger=StructuredLogger("test_llm_engine"),
        fallback_response=fallback_response,
    )


# ---------------------------------------------------------------------------
# Tests: successful pipeline
# ---------------------------------------------------------------------------

class TestSuccessfulPipeline:
    """Requirement 5.3 — pipeline invokes all three components in order."""

    def test_successful_generate_returns_parsed_response(self):
        engine = make_engine()
        request = make_request()
        result = engine.generate(request)
        assert isinstance(result, ParsedResponse)
        assert result.is_valid is True
        assert result.text == "Paris."

    def test_pipeline_invokes_prompt_builder(self):
        spy_builder = SpyPromptBuilder()
        engine = make_engine(prompt_builder=spy_builder)
        engine.generate(make_request())
        assert spy_builder.call_count == 1

    def test_pipeline_invokes_llm_client(self):
        mock_client = MockLLMClient(response=make_llm_response())
        engine = make_engine(llm_client=mock_client)
        engine.generate(make_request())
        assert mock_client.call_count == 1

    def test_pipeline_invokes_response_parser(self):
        spy_parser = SpyResponseParser()
        engine = make_engine(response_parser=spy_parser)
        engine.generate(make_request())
        assert spy_parser.call_count == 1

    def test_pipeline_order_builder_before_client_before_parser(self):
        """Verify ordering by checking that parser received the client's response text."""
        llm_resp = make_llm_response(text="Hello world.")
        mock_client = MockLLMClient(response=llm_resp)
        spy_parser = SpyResponseParser()
        engine = make_engine(llm_client=mock_client, response_parser=spy_parser)
        engine.generate(make_request())
        assert spy_parser.last_response is not None
        assert spy_parser.last_response.raw_text == "Hello world."

    def test_result_request_id_matches_request(self):
        engine = make_engine()
        request = make_request(request_id="my-unique-id")
        result = engine.generate(request)
        assert result.request_id == "my-unique-id"

    def test_token_usage_populated(self):
        engine = make_engine()
        result = engine.generate(make_request())
        assert "prompt" in result.token_usage
        assert "completion" in result.token_usage


# ---------------------------------------------------------------------------
# Tests: LLMClientError → fallback
# ---------------------------------------------------------------------------

class TestLLMClientErrorFallback:
    """Requirements 5.4, 9.1, 12.7 — LLMClientError returns fallback without raising."""

    def test_llm_client_error_does_not_raise(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client)
        # Must not raise
        result = engine.generate(make_request())
        assert result is not None

    def test_llm_client_error_returns_parsed_response(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client)
        result = engine.generate(make_request())
        assert isinstance(result, ParsedResponse)

    def test_fallback_is_valid_false(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client)
        result = engine.generate(make_request())
        assert result.is_valid is False

    def test_fallback_has_llm_client_error_note(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client)
        result = engine.generate(make_request())
        assert "llm_client_error" in result.validation_notes

    def test_fallback_text_is_configured_message(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client, fallback_response="Custom fallback.")
        result = engine.generate(make_request())
        assert result.text == "Custom fallback."

    def test_fallback_request_id_matches(self):
        mock_client = MockLLMClient(raises=LLMClientError("provider down"))
        engine = make_engine(llm_client=mock_client)
        request = make_request(request_id="fallback-req-id")
        result = engine.generate(request)
        assert result.request_id == "fallback-req-id"


# ---------------------------------------------------------------------------
# Tests: PromptBuildError → LLMGenerationError
# ---------------------------------------------------------------------------

class TestPromptBuildErrorHandling:
    """Requirement 5.5, 12.6 — PromptBuildError raises LLMGenerationError."""

    def test_prompt_build_error_raises_llm_generation_error(self):
        spy_builder = SpyPromptBuilder(raises=PromptBuildError("bad context"))
        engine = make_engine(prompt_builder=spy_builder)
        with pytest.raises(LLMGenerationError):
            engine.generate(make_request())

    def test_prompt_build_error_chained_as_cause(self):
        original = PromptBuildError("bad context")
        spy_builder = SpyPromptBuilder(raises=original)
        engine = make_engine(prompt_builder=spy_builder)
        with pytest.raises(LLMGenerationError) as exc_info:
            engine.generate(make_request())
        assert exc_info.value.__cause__ is original

    def test_prompt_build_error_does_not_call_llm_client(self):
        spy_builder = SpyPromptBuilder(raises=PromptBuildError("bad context"))
        mock_client = MockLLMClient(response=make_llm_response())
        engine = make_engine(prompt_builder=spy_builder, llm_client=mock_client)
        with pytest.raises(LLMGenerationError):
            engine.generate(make_request())
        assert mock_client.call_count == 0


# ---------------------------------------------------------------------------
# Tests: ResponseParseError → LLMGenerationError
# ---------------------------------------------------------------------------

class TestResponseParseErrorHandling:
    """Requirement 5.6, 12.6 — ResponseParseError raises LLMGenerationError."""

    def test_response_parse_error_raises_llm_generation_error(self):
        spy_parser = SpyResponseParser(raises=ResponseParseError("parse failed"))
        engine = make_engine(response_parser=spy_parser)
        with pytest.raises(LLMGenerationError):
            engine.generate(make_request())

    def test_response_parse_error_chained_as_cause(self):
        original = ResponseParseError("parse failed")
        spy_parser = SpyResponseParser(raises=original)
        engine = make_engine(response_parser=spy_parser)
        with pytest.raises(LLMGenerationError) as exc_info:
            engine.generate(make_request())
        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: no real HTTP calls
# ---------------------------------------------------------------------------

class TestNoRealHTTPCalls:
    """Requirements 5.10, 10.5, 11.4 — MockLLMClient prevents real HTTP calls."""

    def test_mock_client_used_no_real_calls(self):
        """MockLLMClient is injected; no real network calls are made."""
        mock_client = MockLLMClient(response=make_llm_response(text="Mocked response."))
        engine = make_engine(llm_client=mock_client)
        result = engine.generate(make_request())
        assert result.text == "Mocked response."
        assert mock_client.call_count == 1

    def test_llm_client_called_exactly_once_per_generate(self):
        """Requirement 5.10 — LLMClient called at most once per generate()."""
        mock_client = MockLLMClient(response=make_llm_response())
        engine = make_engine(llm_client=mock_client)
        engine.generate(make_request())
        assert mock_client.call_count == 1

    def test_llm_client_not_called_on_prompt_build_error(self):
        """LLMClient is never called when PromptBuilder raises."""
        spy_builder = SpyPromptBuilder(raises=PromptBuildError("empty input"))
        mock_client = MockLLMClient(response=make_llm_response())
        engine = make_engine(prompt_builder=spy_builder, llm_client=mock_client)
        with pytest.raises(LLMGenerationError):
            engine.generate(make_request())
        assert mock_client.call_count == 0


# ---------------------------------------------------------------------------
# Tests: generate_response backward-compat
# ---------------------------------------------------------------------------

class TestGenerateResponseBackwardCompat:
    """Requirements 5.1, 5.2 — generate_response delegates to generate()."""

    def test_generate_response_returns_string(self):
        engine = make_engine()
        result = engine.generate_response("Hello", {})
        assert isinstance(result, str)

    def test_generate_response_uses_prompt_as_current_input(self):
        spy_builder = SpyPromptBuilder()
        engine = make_engine(prompt_builder=spy_builder)
        engine.generate_response("What is Python?", {})
        assert spy_builder.last_context.current_input == "What is Python?"

    def test_generate_response_returns_text_from_generate(self):
        mock_client = MockLLMClient(response=make_llm_response(text="42."))
        engine = make_engine(llm_client=mock_client)
        result = engine.generate_response("What is the answer?", {})
        assert result == "42."

    def test_generate_response_uses_context_system_instructions(self):
        spy_builder = SpyPromptBuilder()
        engine = make_engine(prompt_builder=spy_builder)
        engine.generate_response("Hello", {"system_instructions": "Act as a pirate."})
        assert spy_builder.last_context.system_instructions == "Act as a pirate."

    def test_generate_response_fallback_on_client_error(self):
        mock_client = MockLLMClient(raises=LLMClientError("down"))
        engine = make_engine(llm_client=mock_client, fallback_response="Sorry, try later.")
        result = engine.generate_response("Hello", {})
        assert result == "Sorry, try later."

    def test_generate_response_uses_default_system_instructions_when_not_provided(self):
        spy_builder = SpyPromptBuilder()
        engine = make_engine(prompt_builder=spy_builder)
        engine.generate_response("Hello", {})
        assert spy_builder.last_context.system_instructions  # non-empty default
