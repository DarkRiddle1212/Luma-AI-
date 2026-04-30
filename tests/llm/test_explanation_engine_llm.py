"""
Unit tests for ExplanationEngine with LLM integration.

Tests:
- explain() calls LLMEngine.generate() with teacher-mode system instructions
- output_constraints reflect AdaptationContext tone/style/focus
- TeachingSessionError raised when ParsedResponse.is_valid = False
- No MemoryInterface or InsightEngine calls made

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from luma.core.teacher.explanation_engine import ExplanationEngine, _TEACHER_SYSTEM_INSTRUCTIONS
from luma.core.teacher.schemas import Explanation, TeachingSessionError
from luma.core.llm.schemas import LLMRequest, ParsedResponse, PromptContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockCtx:
    tone: str = "casual"
    style: str = "concise"
    focus: str = "high-level"


def _make_lesson(
    id="lesson-1",
    topic="Python",
    title="Introduction to Python",
    difficulty="beginner",
    content="Python is a high-level programming language.",
    order=1,
):
    """Return a minimal Lesson-like object."""
    from luma.core.teacher.schemas import Lesson
    return Lesson(
        id=id,
        topic=topic,
        title=title,
        difficulty=difficulty,
        content=content,
        order=order,
    )


def _make_parsed_response(
    text="Python is a versatile language used for many purposes.",
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


def _make_engine_with_mock_llm(llm_response: ParsedResponse = None, llm_raises=None):
    """Build an ExplanationEngine with a mocked LLMEngine."""
    llm_engine = MagicMock()
    if llm_raises is not None:
        llm_engine.generate.side_effect = llm_raises
    else:
        llm_engine.generate.return_value = (
            llm_response if llm_response is not None else _make_parsed_response()
        )
    return ExplanationEngine(llm_engine=llm_engine), llm_engine


# ---------------------------------------------------------------------------
# Tests: LLMEngine.generate() is called (Requirement 8.1)
# ---------------------------------------------------------------------------

class TestExplainCallsLLMEngine:
    """Validates: Requirements 8.1"""

    def test_llm_engine_generate_is_called(self):
        engine, llm_engine = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx()
        engine.explain(lesson, ctx)
        llm_engine.generate.assert_called_once()

    def test_llm_engine_generate_receives_llm_request(self):
        engine, llm_engine = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx()
        engine.explain(lesson, ctx)
        args, _ = llm_engine.generate.call_args
        assert isinstance(args[0], LLMRequest)

    def test_explanation_content_comes_from_llm(self):
        parsed = _make_parsed_response(text="LLM-generated explanation text.")
        engine, _ = _make_engine_with_mock_llm(llm_response=parsed)
        lesson = _make_lesson()
        ctx = _MockCtx()
        result = engine.explain(lesson, ctx)
        assert result.content == "LLM-generated explanation text."

    def test_explanation_lesson_id_matches(self):
        engine, _ = _make_engine_with_mock_llm()
        lesson = _make_lesson(id="lesson-42")
        ctx = _MockCtx()
        result = engine.explain(lesson, ctx)
        assert result.lesson_id == "lesson-42"

    def test_returns_explanation_instance(self):
        engine, _ = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx()
        result = engine.explain(lesson, ctx)
        assert isinstance(result, Explanation)


# ---------------------------------------------------------------------------
# Tests: Teacher-mode system instructions (Requirement 8.2)
# ---------------------------------------------------------------------------

class TestTeacherModeSystemInstructions:
    """Validates: Requirements 8.2"""

    def _get_prompt_context(self, lesson=None, ctx=None) -> PromptContext:
        engine, llm_engine = _make_engine_with_mock_llm()
        if lesson is None:
            lesson = _make_lesson()
        if ctx is None:
            ctx = _MockCtx()
        engine.explain(lesson, ctx)
        args, _ = llm_engine.generate.call_args
        return args[0].prompt_context

    def test_system_instructions_is_non_empty(self):
        pc = self._get_prompt_context()
        assert isinstance(pc.system_instructions, str)
        assert len(pc.system_instructions.strip()) > 0

    def test_system_instructions_mentions_tutor(self):
        pc = self._get_prompt_context()
        assert "tutor" in pc.system_instructions.lower()

    def test_system_instructions_prohibits_invented_facts(self):
        pc = self._get_prompt_context()
        # Should instruct LLM not to invent facts
        instructions_lower = pc.system_instructions.lower()
        assert "invent" in instructions_lower or "beyond" in instructions_lower

    def test_system_instructions_matches_constant(self):
        pc = self._get_prompt_context()
        assert pc.system_instructions == _TEACHER_SYSTEM_INSTRUCTIONS

    def test_current_input_is_lesson_content(self):
        lesson = _make_lesson(content="Loops allow repeated execution of code.")
        pc = self._get_prompt_context(lesson=lesson)
        assert pc.current_input == "Loops allow repeated execution of code."


# ---------------------------------------------------------------------------
# Tests: output_constraints reflect AdaptationContext (Requirement 8.3)
# ---------------------------------------------------------------------------

class TestOutputConstraintsReflectAdaptationContext:
    """Validates: Requirements 8.3"""

    def _get_prompt_context(self, tone="technical", style="detailed", focus="deep-technical") -> PromptContext:
        engine, llm_engine = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx(tone=tone, style=style, focus=focus)
        engine.explain(lesson, ctx)
        args, _ = llm_engine.generate.call_args
        return args[0].prompt_context

    def test_output_constraints_contains_tone(self):
        pc = self._get_prompt_context(tone="technical")
        assert "technical" in pc.output_constraints

    def test_output_constraints_contains_style(self):
        pc = self._get_prompt_context(style="step-by-step")
        assert "step-by-step" in pc.output_constraints

    def test_output_constraints_contains_focus(self):
        pc = self._get_prompt_context(focus="deep-technical")
        assert "deep-technical" in pc.output_constraints

    def test_output_constraints_casual_tone(self):
        pc = self._get_prompt_context(tone="casual")
        assert "casual" in pc.output_constraints

    def test_output_constraints_high_level_focus(self):
        pc = self._get_prompt_context(focus="high-level")
        assert "high-level" in pc.output_constraints

    def test_output_constraints_concise_style(self):
        pc = self._get_prompt_context(style="concise")
        assert "concise" in pc.output_constraints


# ---------------------------------------------------------------------------
# Tests: TeachingSessionError on invalid ParsedResponse (Requirement 8.4)
# ---------------------------------------------------------------------------

class TestTeachingSessionErrorOnInvalidResponse:
    """Validates: Requirements 8.4"""

    def test_raises_teaching_session_error_when_is_valid_false(self):
        invalid = _make_parsed_response(
            text="",
            is_valid=False,
            validation_notes=["empty response"],
        )
        engine, _ = _make_engine_with_mock_llm(llm_response=invalid)
        lesson = _make_lesson()
        ctx = _MockCtx()
        with pytest.raises(TeachingSessionError):
            engine.explain(lesson, ctx)

    def test_error_message_is_descriptive(self):
        invalid = _make_parsed_response(
            text="",
            is_valid=False,
            validation_notes=["llm_client_error"],
        )
        engine, _ = _make_engine_with_mock_llm(llm_response=invalid)
        lesson = _make_lesson(id="lesson-99")
        ctx = _MockCtx()
        with pytest.raises(TeachingSessionError) as exc_info:
            engine.explain(lesson, ctx)
        assert "lesson-99" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()

    def test_no_error_when_is_valid_true(self):
        valid = _make_parsed_response(text="Good explanation.", is_valid=True)
        engine, _ = _make_engine_with_mock_llm(llm_response=valid)
        lesson = _make_lesson()
        ctx = _MockCtx()
        result = engine.explain(lesson, ctx)
        assert result.content == "Good explanation."

    def test_validation_notes_included_in_error_message(self):
        invalid = _make_parsed_response(
            text="",
            is_valid=False,
            validation_notes=["empty response"],
        )
        engine, _ = _make_engine_with_mock_llm(llm_response=invalid)
        lesson = _make_lesson()
        ctx = _MockCtx()
        with pytest.raises(TeachingSessionError) as exc_info:
            engine.explain(lesson, ctx)
        assert "empty response" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: No MemoryInterface or InsightEngine calls (Requirement 8.5)
# ---------------------------------------------------------------------------

class TestNoMemoryOrInsightCalls:
    """Validates: Requirements 8.5"""

    def test_no_memory_interface_attribute(self):
        engine, _ = _make_engine_with_mock_llm()
        # ExplanationEngine should not hold a memory_interface
        assert not hasattr(engine, "_memory_interface")
        assert not hasattr(engine, "memory_interface")

    def test_no_insight_engine_attribute(self):
        engine, _ = _make_engine_with_mock_llm()
        # ExplanationEngine should not hold an insight_engine
        assert not hasattr(engine, "_insight_engine")
        assert not hasattr(engine, "insight_engine")

    def test_llm_engine_called_exactly_once(self):
        engine, llm_engine = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx()
        engine.explain(lesson, ctx)
        assert llm_engine.generate.call_count == 1

    def test_relevant_memories_is_empty_list(self):
        """ExplanationEngine must not pass memories — only lesson + AdaptationContext."""
        engine, llm_engine = _make_engine_with_mock_llm()
        lesson = _make_lesson()
        ctx = _MockCtx()
        engine.explain(lesson, ctx)
        args, _ = llm_engine.generate.call_args
        assert args[0].prompt_context.relevant_memories == []


# ---------------------------------------------------------------------------
# Tests: Backward compatibility — no LLMEngine (template fallback)
# ---------------------------------------------------------------------------

class TestBackwardCompatibilityNoLLM:
    """ExplanationEngine without llm_engine uses template-based generation."""

    def test_explain_works_without_llm_engine(self):
        engine = ExplanationEngine()  # no llm_engine
        lesson = _make_lesson()
        ctx = _MockCtx(tone="casual", style="concise", focus="high-level")
        result = engine.explain(lesson, ctx)
        assert isinstance(result, Explanation)
        assert result.content

    def test_template_fallback_contains_lesson_title(self):
        engine = ExplanationEngine()
        lesson = _make_lesson(title="Variables in Python")
        ctx = _MockCtx(tone="technical", style="balanced", focus="high-level")
        result = engine.explain(lesson, ctx)
        assert "Variables in Python" in result.content


# ---------------------------------------------------------------------------
# Tests: Input validation still works with LLMEngine
# ---------------------------------------------------------------------------

class TestInputValidationWithLLM:
    """Lesson validation still applies when LLMEngine is present."""

    def test_empty_content_raises_value_error(self):
        engine, _ = _make_engine_with_mock_llm()

        @dataclass
        class _BadLesson:
            id: str = "l1"
            topic: str = "Python"
            title: str = "Intro"
            difficulty: str = "beginner"
            content: str = ""
            order: int = 0

        ctx = _MockCtx()
        with pytest.raises(ValueError):
            engine.explain(_BadLesson(), ctx)

    def test_missing_field_raises_value_error(self):
        engine, _ = _make_engine_with_mock_llm()

        @dataclass
        class _NoContent:
            id: str = "l2"
            topic: str = "Python"
            title: str = "Intro"
            difficulty: str = "beginner"
            order: int = 0

        ctx = _MockCtx()
        with pytest.raises(ValueError):
            engine.explain(_NoContent(), ctx)
