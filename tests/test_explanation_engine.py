"""
Unit tests for ExplanationEngine.

Covers tone vocabulary signals, style structure signals, focus depth signals,
rationale content, ValueError on invalid lesson, and input immutability.
"""

import pytest
from dataclasses import dataclass

from luma.core.teacher.explanation_engine import ExplanationEngine
from luma.core.teacher.lesson_generator import LessonGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockCtx:
    tone: str
    style: str
    focus: str


def _make_lesson(topic: str = "Python", user_level: str = "beginner"):
    """Return the first lesson from LessonGenerator for the given topic/level."""
    return LessonGenerator().generate(topic=topic, user_level=user_level)[0]


def _explain(tone: str, style: str, focus: str, lesson=None):
    if lesson is None:
        lesson = _make_lesson()
    ctx = _MockCtx(tone=tone, style=style, focus=focus)
    return ExplanationEngine().explain(lesson, ctx)


# ---------------------------------------------------------------------------
# Tone vocabulary signals
# ---------------------------------------------------------------------------

def test_tone_technical_contains_domain_specific():
    explanation = _explain(tone="technical", style="balanced", focus="high-level")
    assert "domain-specific terminology" in explanation.content or \
           "Using domain-specific" in explanation.content


def test_tone_casual_contains_plain_language():
    explanation = _explain(tone="casual", style="balanced", focus="high-level")
    assert "plain language" in explanation.content or \
           "In plain language" in explanation.content


def test_tone_formal_contains_formal_academic():
    explanation = _explain(tone="formal", style="balanced", focus="high-level")
    assert "formal academic" in explanation.content or \
           "In formal academic" in explanation.content


# ---------------------------------------------------------------------------
# Style structure signals
# ---------------------------------------------------------------------------

def test_style_step_by_step_contains_step_1():
    explanation = _explain(tone="technical", style="step-by-step", focus="high-level")
    assert "Step 1" in explanation.content


def test_style_concise_contains_summary():
    explanation = _explain(tone="technical", style="concise", focus="high-level")
    assert "Summary:" in explanation.content


def test_style_detailed_contains_overview_and_details():
    explanation = _explain(tone="technical", style="detailed", focus="high-level")
    assert "Overview:" in explanation.content
    assert "Details:" in explanation.content


def test_style_balanced_contains_lesson_title():
    lesson = _make_lesson(topic="Python")
    explanation = _explain(tone="technical", style="balanced", focus="high-level", lesson=lesson)
    assert lesson.title in explanation.content


# ---------------------------------------------------------------------------
# Focus depth signals
# ---------------------------------------------------------------------------

def test_focus_deep_technical_contains_implementation_details():
    explanation = _explain(tone="technical", style="balanced", focus="deep-technical")
    assert "Implementation details" in explanation.content


def test_focus_high_level_contains_high_level_overview():
    explanation = _explain(tone="technical", style="balanced", focus="high-level")
    assert "High-level overview" in explanation.content


# ---------------------------------------------------------------------------
# Rationale field
# ---------------------------------------------------------------------------

def test_rationale_is_non_empty():
    explanation = _explain(tone="casual", style="concise", focus="high-level")
    assert explanation.rationale


def test_rationale_references_adaptation_context_fields():
    ctx = _MockCtx(tone="formal", style="detailed", focus="deep-technical")
    lesson = _make_lesson()
    explanation = ExplanationEngine().explain(lesson, ctx)
    assert "tone=" in explanation.rationale
    assert "style=" in explanation.rationale
    assert "focus=" in explanation.rationale


# ---------------------------------------------------------------------------
# ValueError on missing/empty lesson field
# ---------------------------------------------------------------------------

@dataclass
class _BadLesson:
    """Lesson-like object with an empty content field."""
    id: str = "l1"
    topic: str = "Python"
    title: str = "Intro"
    difficulty: str = "beginner"
    content: str = ""   # empty — should trigger ValueError
    order: int = 0


def test_empty_lesson_content_raises_value_error():
    ctx = _MockCtx(tone="technical", style="balanced", focus="high-level")
    with pytest.raises(ValueError):
        ExplanationEngine().explain(_BadLesson(), ctx)


@dataclass
class _MissingFieldLesson:
    """Lesson-like object missing the 'content' attribute entirely."""
    id: str = "l2"
    topic: str = "Python"
    title: str = "Intro"
    difficulty: str = "beginner"
    order: int = 0
    # 'content' intentionally omitted


def test_missing_lesson_field_raises_value_error():
    ctx = _MockCtx(tone="technical", style="balanced", focus="high-level")
    with pytest.raises(ValueError):
        ExplanationEngine().explain(_MissingFieldLesson(), ctx)


# ---------------------------------------------------------------------------
# Input arguments are not mutated
# ---------------------------------------------------------------------------

def test_input_lesson_not_mutated():
    lesson = _make_lesson(topic="Algorithms")
    original_id = lesson.id
    original_title = lesson.title
    original_content = lesson.content
    original_difficulty = lesson.difficulty
    original_order = lesson.order

    ctx = _MockCtx(tone="casual", style="step-by-step", focus="deep-technical")
    ExplanationEngine().explain(lesson, ctx)

    assert lesson.id == original_id
    assert lesson.title == original_title
    assert lesson.content == original_content
    assert lesson.difficulty == original_difficulty
    assert lesson.order == original_order


def test_input_ctx_not_mutated():
    lesson = _make_lesson()
    ctx = _MockCtx(tone="formal", style="detailed", focus="high-level")
    original_tone = ctx.tone
    original_style = ctx.style
    original_focus = ctx.focus

    ExplanationEngine().explain(lesson, ctx)

    assert ctx.tone == original_tone
    assert ctx.style == original_style
    assert ctx.focus == original_focus
