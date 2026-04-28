"""
ExplanationEngine — stateless component that produces adapted explanations.

Accepts a Lesson and an AdaptationContext (plain data object with .tone,
.style, .focus attributes) and returns an Explanation.  No imports from
luma.core.personalization; the caller is responsible for supplying a
compatible context object.
"""

from luma.core.teacher.schemas import Explanation, Lesson

# Required fields that must be non-empty strings on every Lesson.
_REQUIRED_LESSON_FIELDS = ("id", "topic", "title", "difficulty", "content")

# Tone → prefix mapping
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


class ExplanationEngine:
    """Stateless engine that adapts lesson content to a user's context."""

    def explain(self, lesson: Lesson, adaptation_ctx) -> Explanation:
        """
        Produce an Explanation for *lesson* adapted to *adaptation_ctx*.

        Parameters
        ----------
        lesson:
            A Lesson instance.  All required fields (id, topic, title,
            difficulty, content) must be non-empty strings.
        adaptation_ctx:
            Any object exposing .tone, .style, and .focus string attributes.
            Valid values mirror those defined in AdaptationContext.

        Returns
        -------
        Explanation
            A new Explanation object.  Input arguments are never mutated.

        Raises
        ------
        ValueError
            If any required Lesson field is missing or empty.
        """
        # --- Validate lesson fields ---
        for field_name in _REQUIRED_LESSON_FIELDS:
            value = getattr(lesson, field_name, None)
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Lesson is missing required field: {field_name}")

        tone = adaptation_ctx.tone
        style = adaptation_ctx.style
        focus = adaptation_ctx.focus

        # --- Build content deterministically ---
        prefix = _TONE_PREFIX.get(tone, "")
        body = _build_style_body(lesson, style)
        suffix = _FOCUS_SUFFIX.get(focus, "")

        content = prefix + body + suffix

        # --- Build rationale ---
        rationale = (
            f"Adapted using tone={tone!r}, style={style!r}, focus={focus!r}"
        )

        return Explanation(
            lesson_id=lesson.id,
            content=content,
            rationale=rationale,
        )
