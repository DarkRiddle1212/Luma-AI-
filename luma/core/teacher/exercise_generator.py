"""
ExerciseGenerator: deterministic exercise builder for Teacher Mode.

Generates exactly 3 Exercise objects per lesson (conceptual, practical,
mini-project) based on the lesson content and user level.
No randomness — identical inputs always produce identical outputs.
"""

from typing import List

from luma.core.teacher.schemas import Exercise, Lesson

# Required fields that every Lesson must have as non-empty strings
_REQUIRED_LESSON_FIELDS = ("id", "topic", "title", "difficulty", "content")

# Exercise types generated in order
_EXERCISE_TYPES = ("conceptual", "practical", "mini-project")

# Difficulty assigned per user_level and exercise index
_DIFFICULTY_MAP = {
    "beginner":     ("beginner", "beginner", "intermediate"),
    "intermediate": ("intermediate", "intermediate", "advanced"),
    "advanced":     ("advanced", "advanced", "advanced"),
}


class ExerciseGenerator:
    """Generates a deterministic set of exercises for a given lesson."""

    def generate(self, lesson: Lesson, user_level: str) -> List[Exercise]:
        """
        Generate exercises for *lesson* appropriate for *user_level*.

        Parameters
        ----------
        lesson:
            A fully-populated Lesson object. Must have non-empty id, topic,
            title, difficulty, and content fields.
        user_level:
            One of ``"beginner"``, ``"intermediate"``, or ``"advanced"``.

        Returns
        -------
        List[Exercise]
            Exactly 3 exercises: conceptual, practical, mini-project.

        Raises
        ------
        ValueError
            If *lesson* is missing any required field.
        """
        # Validate required lesson fields
        for field_name in _REQUIRED_LESSON_FIELDS:
            value = getattr(lesson, field_name, None)
            if not value or not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Lesson is missing required field: {field_name}"
                )

        difficulties = _DIFFICULTY_MAP.get(
            user_level,
            ("beginner", "beginner", "intermediate"),
        )

        exercises: List[Exercise] = []
        for i, ex_type in enumerate(_EXERCISE_TYPES):
            ex_id = f"{lesson.id}_ex_{i}"
            difficulty = difficulties[i]

            if ex_type == "conceptual":
                prompt = f"Explain the concept of {lesson.title} in your own words."
                explanation = (
                    f"This conceptual exercise reinforces understanding of "
                    f"{lesson.topic} at {user_level} level."
                )
            elif ex_type == "practical":
                prompt = f"Apply {lesson.title} to solve a real-world problem."
                explanation = (
                    f"This practical exercise applies {lesson.topic} concepts "
                    f"at {user_level} level."
                )
            else:  # mini-project
                prompt = f"Build a small project demonstrating {lesson.title}."
                explanation = (
                    f"This mini-project integrates {lesson.topic} skills "
                    f"at {user_level} level."
                )

            exercises.append(
                Exercise(
                    id=ex_id,
                    lesson_id=lesson.id,
                    type=ex_type,
                    difficulty=difficulty,
                    prompt=prompt,
                    explanation=explanation,
                )
            )

        return exercises
