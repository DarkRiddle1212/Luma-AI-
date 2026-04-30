"""
ExplanationEngine — produces adapted explanations via LLMEngine (Teacher Mode).

Accepts a Lesson and an AdaptationContext, constructs a PromptContext with
teacher-mode system instructions, and delegates text generation to LLMEngine.
Falls back to deterministic template-based generation when no LLMEngine is
provided (backward compatibility).
"""

import uuid
from typing import Optional

from luma.core.teacher.schemas import Explanation, Lesson, TeachingSessionError

# Required fields that must be non-empty strings on every Lesson.
_REQUIRED_LESSON_FIELDS = ("id", "topic", "title", "difficulty", "content")

# Teacher-mode system instructions for the LLM.
_TEACHER_SYSTEM_INSTRUCTIONS = (
    "You are an expert tutor helping a student learn. "
    "Explain the provided lesson content clearly and accurately. "
    "Do not invent facts, examples, or concepts beyond what is contained in the lesson content. "
    "Adapt your explanation to the student's level and the specified tone, style, and focus."
)

# Tone → prefix mapping (used in fallback / output_constraints)
_TONE_PREFIX = {
    "technical": "Using domain-specific terminology: ",
    "casual": "In plain language: ",
    "formal": "In formal academic terms: ",
}

# Focus → suffix mapping
_FOCUS_SUFFIX = {
    "deep-technical": " [Implementation details and edge cases included.]",
    "high-level": " [High-level overview only.]",
}


def _build_style_body(lesson: Lesson, style: str) -> str:
    """Return the structural body of the explanation based on *style*."""
    if style == "step-by-step":
        return (
            f"Step 1: Understand {lesson.title}. "
            f"Step 2: Apply concepts from {lesson.content}."
        )
    if style == "concise":
        return f"Summary: {lesson.title} — {lesson.content}"
    if style == "detailed":
        return (
            f"Overview: {lesson.title}\n\n"
            f"Details: {lesson.content}\n\n"
            "Conclusion: This covers the key aspects."
        )
    # "balanced" (default)
    return f"{lesson.title}: {lesson.content}"


def _build_output_constraints(adaptation_ctx) -> str:
    """Build output_constraints string from AdaptationContext."""
    return (
        f"Tone: {adaptation_ctx.tone}. "
        f"Style: {adaptation_ctx.style}. "
        f"Focus: {adaptation_ctx.focus}."
    )


class ExplanationEngine:
    """Engine that adapts lesson content to a user's context via LLMEngine."""

    def __init__(self, llm_engine=None) -> None:
        """
        Parameters
        ----------
        llm_engine:
            An LLMEngine instance used to generate explanation text.
            When None, falls back to deterministic template-based generation
            (backward compatibility).
        """
        self._llm_engine = llm_engine

    def explain(self, lesson: Lesson, adaptation_ctx) -> Explanation:
        """
        Produce an Explanation for *lesson* adapted to *adaptation_ctx*.

        When an LLMEngine is available, constructs a PromptContext with
        teacher-mode system instructions and delegates to LLMEngine.generate().
        Otherwise falls back to deterministic template generation.

        Parameters
        ----------
        lesson:
            A Lesson instance.  All required fields (id, topic, title,
            difficulty, content) must be non-empty strings.
        adaptation_ctx:
            Any object exposing .tone, .style, and .focus string attributes.

        Returns
        -------
        Explanation
            A new Explanation object.  Input arguments are never mutated.

        Raises
        ------
        ValueError
            If any required Lesson field is missing or empty.
        TeachingSessionError
            If LLMEngine returns an invalid ParsedResponse.
        """
        # --- Validate lesson fields ---
        for field_name in _REQUIRED_LESSON_FIELDS:
            value = getattr(lesson, field_name, None)
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Lesson is missing required field: {field_name}")

        if self._llm_engine is not None:
            return self._explain_with_llm(lesson, adaptation_ctx)
        return self._explain_with_template(lesson, adaptation_ctx)

    def _explain_with_llm(self, lesson: Lesson, adaptation_ctx) -> Explanation:
        """Generate explanation using LLMEngine."""
        from luma.core.llm.schemas import LLMRequest, PromptContext

        prompt_context = PromptContext(
            system_instructions=_TEACHER_SYSTEM_INSTRUCTIONS,
            user_profile=_build_output_constraints(adaptation_ctx),
            relevant_memories=[],
            current_input=lesson.content,
            output_constraints=_build_output_constraints(adaptation_ctx),
        )

        request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id=str(uuid.uuid4()),
        )

        parsed = self._llm_engine.generate(request)

        if not parsed.is_valid:
            raise TeachingSessionError(
                f"LLM returned an invalid response for lesson '{lesson.id}': "
                f"{', '.join(parsed.validation_notes) or 'unknown error'}"
            )

        rationale = (
            f"Adapted using tone={adaptation_ctx.tone!r}, "
            f"style={adaptation_ctx.style!r}, "
            f"focus={adaptation_ctx.focus!r}"
        )

        return Explanation(
            lesson_id=lesson.id,
            content=parsed.text,
            rationale=rationale,
        )

    def _explain_with_template(self, lesson: Lesson, adaptation_ctx) -> Explanation:
        """Deterministic template-based fallback (no LLMEngine)."""
        tone = adaptation_ctx.tone
        style = adaptation_ctx.style
        focus = adaptation_ctx.focus

        prefix = _TONE_PREFIX.get(tone, "")
        body = _build_style_body(lesson, style)
        suffix = _FOCUS_SUFFIX.get(focus, "")

        content = prefix + body + suffix

        rationale = (
            f"Adapted using tone={tone!r}, style={style!r}, focus={focus!r}"
        )

        return Explanation(
            lesson_id=lesson.id,
            content=content,
            rationale=rationale,
        )
