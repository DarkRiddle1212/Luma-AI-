"""
Unit tests for LumaService chat flow with LLM integration.

Tests:
- process_chat calls LLMEngine.generate() with correctly constructed PromptContext
- Fallback path when ParsedResponse.is_valid = False
- Memories passed as strings, not raw MemoryEntry objects

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import asyncio
from unittest.mock import MagicMock, call

import pytest

from luma.api.services.luma_service import LumaService
from luma.core.llm.schemas import ParsedResponse, PromptContext, LLMRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def _make_parsed_response(
    text="Hello, how can I help?",
    is_valid=True,
    validation_notes=None,
) -> ParsedResponse:
    return ParsedResponse(
        request_id="req-001",
        text=text,
        is_valid=is_valid,
        validation_notes=validation_notes or [],
        token_usage={"prompt": 10, "completion": 20},
        truncated=False,
    )


def _make_service(llm_response: ParsedResponse = None, llm_raises=None):
    """Build a LumaService with all dependencies mocked, including LLMEngine."""
    memory = MagicMock()
    insight_engine = MagicMock()
    insight_moments_engine = MagicMock()
    personalization_engine = MagicMock()
    teacher_mode = MagicMock()
    llm_engine = MagicMock()
    logger = MagicMock()

    # Memory returns two entries
    memory.retrieve.return_value = {
        "memories": [
            {
                "content": "User likes Python",
                "id": "1",
                "metadata": {},
                "timestamp": "2024-01-01T00:00:00",
                "category": "chat",
                "tags": [],
            },
            {
                "content": "User prefers concise answers",
                "id": "2",
                "metadata": {},
                "timestamp": "2024-01-01T00:00:00",
                "category": "chat",
                "tags": [],
            },
        ],
        "total_count": 2,
        "query_metadata": {},
    }
    memory.store.return_value = "mem_id_1"

    adaptation_ctx = MagicMock()
    adaptation_ctx.tone = "casual"
    adaptation_ctx.style = "concise"
    adaptation_ctx.focus = "high-level"
    adaptation_ctx.reasons = {}
    adaptation_ctx.model_dump.return_value = {
        "tone": "casual",
        "style": "concise",
        "focus": "high-level",
        "reasons": {},
    }

    personalization_result = MagicMock()
    personalization_result.adaptation = adaptation_ctx
    personalization_engine.personalize.return_value = personalization_result

    insight_moments_engine.generate_moments.return_value = []

    # Configure LLMEngine mock
    if llm_raises is not None:
        llm_engine.generate.side_effect = llm_raises
    else:
        llm_engine.generate.return_value = (
            llm_response if llm_response is not None else _make_parsed_response()
        )

    service = LumaService(
        memory_interface=memory,
        insight_engine=insight_engine,
        insight_moments_engine=insight_moments_engine,
        personalization_engine=personalization_engine,
        teacher_mode=teacher_mode,
        llm_engine=llm_engine,
        logger=logger,
    )
    return service, memory, llm_engine, logger, adaptation_ctx


# ---------------------------------------------------------------------------
# Tests: LLMEngine.generate() is called (Requirement 7.1)
# ---------------------------------------------------------------------------

class TestProcessChatCallsLLMEngine:
    """Validates: Requirements 7.1"""

    def test_llm_engine_generate_is_called(self):
        service, _, llm_engine, _, _ = _make_service()
        run(service.process_chat("user1", "Hello"))
        llm_engine.generate.assert_called_once()

    def test_llm_engine_generate_receives_llm_request(self):
        service, _, llm_engine, _, _ = _make_service()
        run(service.process_chat("user1", "Hello"))
        args, _ = llm_engine.generate.call_args
        assert isinstance(args[0], LLMRequest)

    def test_response_comes_from_llm_engine(self):
        parsed = _make_parsed_response(text="LLM generated reply")
        service, _, _, _, _ = _make_service(llm_response=parsed)
        result = run(service.process_chat("user1", "Hello"))
        assert result["response"] == "LLM generated reply"


# ---------------------------------------------------------------------------
# Tests: PromptContext is correctly constructed (Requirement 7.2)
# ---------------------------------------------------------------------------

class TestPromptContextConstruction:
    """Validates: Requirements 7.2"""

    def _get_prompt_context(self, message="Tell me about Python") -> PromptContext:
        service, _, llm_engine, _, _ = _make_service()
        run(service.process_chat("user1", message))
        args, _ = llm_engine.generate.call_args
        return args[0].prompt_context

    def test_current_input_is_user_message(self):
        ctx = self._get_prompt_context("Tell me about Python")
        assert ctx.current_input == "Tell me about Python"

    def test_system_instructions_is_non_empty_string(self):
        ctx = self._get_prompt_context()
        assert isinstance(ctx.system_instructions, str)
        assert len(ctx.system_instructions.strip()) > 0

    def test_user_profile_contains_tone(self):
        ctx = self._get_prompt_context()
        assert "casual" in ctx.user_profile

    def test_user_profile_contains_style(self):
        ctx = self._get_prompt_context()
        assert "concise" in ctx.user_profile

    def test_output_constraints_contains_tone(self):
        ctx = self._get_prompt_context()
        assert "casual" in ctx.output_constraints

    def test_output_constraints_contains_style(self):
        ctx = self._get_prompt_context()
        assert "concise" in ctx.output_constraints

    def test_relevant_memories_is_list(self):
        ctx = self._get_prompt_context()
        assert isinstance(ctx.relevant_memories, list)

    def test_relevant_memories_count_matches_retrieved(self):
        ctx = self._get_prompt_context()
        # Two memories were configured in _make_service
        assert len(ctx.relevant_memories) == 2


# ---------------------------------------------------------------------------
# Tests: Memories passed as strings, not raw objects (Requirement 7.3)
# ---------------------------------------------------------------------------

class TestMemoriesPassedAsStrings:
    """Validates: Requirements 7.3"""

    def _get_relevant_memories(self):
        service, _, llm_engine, _, _ = _make_service()
        run(service.process_chat("user1", "Hello"))
        args, _ = llm_engine.generate.call_args
        return args[0].prompt_context.relevant_memories

    def test_memories_are_strings(self):
        memories = self._get_relevant_memories()
        for m in memories:
            assert isinstance(m, str), f"Expected str, got {type(m)}"

    def test_memories_contain_content_text(self):
        memories = self._get_relevant_memories()
        assert "User likes Python" in memories
        assert "User prefers concise answers" in memories

    def test_no_dict_objects_in_memories(self):
        memories = self._get_relevant_memories()
        for m in memories:
            assert not isinstance(m, dict), "Memory should not be a raw dict/MemoryEntry"


# ---------------------------------------------------------------------------
# Tests: Fallback path when ParsedResponse.is_valid = False (Requirement 7.4)
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """Validates: Requirements 7.4"""

    def test_fallback_text_used_as_response(self):
        fallback = _make_parsed_response(
            text="I'm having trouble right now.",
            is_valid=False,
            validation_notes=["llm_client_error"],
        )
        service, _, _, _, _ = _make_service(llm_response=fallback)
        result = run(service.process_chat("user1", "Hello"))
        assert result["response"] == "I'm having trouble right now."

    def test_warning_logged_on_invalid_response(self):
        fallback = _make_parsed_response(
            text="Fallback text.",
            is_valid=False,
            validation_notes=["llm_client_error"],
        )
        service, _, _, logger, _ = _make_service(llm_response=fallback)
        run(service.process_chat("user1", "Hello"))
        logger.log.assert_called()

    def test_warning_log_event_name_indicates_fallback(self):
        fallback = _make_parsed_response(
            text="Fallback text.",
            is_valid=False,
            validation_notes=["empty response"],
        )
        service, _, _, logger, _ = _make_service(llm_response=fallback)
        run(service.process_chat("user1", "Hello"))
        # Check that at least one log call used a fallback-related event name
        log_events = [c.args[0] for c in logger.log.call_args_list]
        assert any("fallback" in ev for ev in log_events)

    def test_valid_response_does_not_log_fallback_warning(self):
        valid = _make_parsed_response(text="Good response.", is_valid=True)
        service, _, _, logger, _ = _make_service(llm_response=valid)
        run(service.process_chat("user1", "Hello"))
        log_events = [c.args[0] for c in logger.log.call_args_list]
        assert not any("fallback" in ev for ev in log_events)

    def test_process_chat_still_returns_dict_on_fallback(self):
        fallback = _make_parsed_response(
            text="Fallback.",
            is_valid=False,
            validation_notes=["llm_client_error"],
        )
        service, _, _, _, _ = _make_service(llm_response=fallback)
        result = run(service.process_chat("user1", "Hello"))
        assert "response" in result
        assert "insight_moments" in result
        assert "personalization" in result


# ---------------------------------------------------------------------------
# Tests: LLMEngine is optional (backward compat)
# ---------------------------------------------------------------------------

class TestLLMEngineOptional:
    """LumaService still works when llm_engine=None (backward compat)."""

    def test_process_chat_works_without_llm_engine(self):
        memory = MagicMock()
        memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {},
        }
        memory.store.return_value = "id"

        adaptation_ctx = MagicMock()
        adaptation_ctx.tone = "casual"
        adaptation_ctx.style = "concise"
        adaptation_ctx.focus = "high-level"
        adaptation_ctx.reasons = {}
        adaptation_ctx.model_dump.return_value = {
            "tone": "casual", "style": "concise", "focus": "high-level", "reasons": {}
        }
        personalization_result = MagicMock()
        personalization_result.adaptation = adaptation_ctx

        personalization_engine = MagicMock()
        personalization_engine.personalize.return_value = personalization_result

        insight_moments_engine = MagicMock()
        insight_moments_engine.generate_moments.return_value = []

        service = LumaService(
            memory_interface=memory,
            insight_engine=MagicMock(),
            insight_moments_engine=insight_moments_engine,
            personalization_engine=personalization_engine,
            teacher_mode=MagicMock(),
            llm_engine=None,
        )
        result = run(service.process_chat("user1", "Hello"))
        assert "response" in result
