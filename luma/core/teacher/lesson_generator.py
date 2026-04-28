"""
LessonGenerator: deterministic lesson sequence builder for Teacher Mode.

Generates an ordered list of Lesson objects based on topic and user level.
No randomness — identical inputs always produce identical outputs.
"""

import re
from typing import List

from luma.core.teacher.schemas import Lesson

# Difficulty ordering used for sorting and level comparisons
_DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced"]

# Per-level lesson templates: (title_template, content_template, difficulty)
_LESSON_TEMPLATES = [
    # beginner (indices 0-1)
    (
        "Introduction to {topic}",
        "This lesson covers the basics of {topic}.",
        "beginner",
    ),
    (
        "{topic} Fundamentals",
        "In this lesson you will learn the fundamental concepts of {topic}.",
        "beginner",
    ),
    # intermediate (indices 2-3)
    (
        "Intermediate {topic}",
        "This lesson explores intermediate techniques in {topic}.",
        "intermediate",
    ),
    (
        "Applying {topic}",
        "Learn how to apply {topic} concepts to real-world problems.",
        "intermediate",
    ),
    # advanced (indices 4-5)
    (
        "Advanced {topic}",
        "This lesson dives deep into advanced aspects of {topic}.",
        "advanced",
    ),
    (
        "Mastering {topic}",
        "Achieve mastery of {topic} through expert-level techniques and patterns.",
        "advanced",
    ),
]

# How many lessons to include per user_level
_LEVEL_SLICE = {
    "beginner": 2,       # lessons 0-1
    "intermediate": 4,   # lessons 0-3
    "advanced": 6,       # lessons 0-5
}


def _topic_slug(topic: str) -> str:
    """Convert topic to a slug: lowercase, spaces → underscores, strip non-alphanumeric."""
    slug = topic.lower().replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    return slug


class LessonGenerator:
    """Generates a deterministic, ordered sequence of Lesson objects."""

    def generate(self, topic: str, user_level: str) -> List[Lesson]:
        """
        Generate lessons for *topic* appropriate for *user_level*.

        Parameters
        ----------
        topic:
            The subject to teach. Must be a non-empty, non-whitespace string.
        user_level:
            One of ``"beginner"``, ``"intermediate"``, or ``"advanced"``.

        Returns
        -------
        List[Lesson]
            Non-empty list ordered beginner → intermediate → advanced.

        Raises
        ------
        ValueError
            If *topic* is empty or whitespace-only.
        """
        if not topic or not topic.strip():
            raise ValueError(
                f"topic must be a non-empty string, got {topic!r}"
            )

        slug = _topic_slug(topic)
        count = _LEVEL_SLICE.get(user_level, len(_LESSON_TEMPLATES))
        templates = _LESSON_TEMPLATES[:count]

        lessons: List[Lesson] = []
        for order, (title_tpl, content_tpl, difficulty) in enumerate(templates):
            lessons.append(
                Lesson(
                    id=f"{slug}_{order}",
                    topic=topic,
                    title=title_tpl.format(topic=topic),
                    difficulty=difficulty,
                    content=content_tpl.format(topic=topic),
                    order=order,
                )
            )

        return lessons
