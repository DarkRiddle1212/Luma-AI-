"""
Unit tests for ExerciseGenerator.

Verifies difficulty ceilings, conceptual exercise presence, non-empty
explanations, ValueError on invalid lesson, and input immutability.
"""

import pytest

from luma.core.teacher.exercise_generator import ExerciseGenerator
from luma.core.teacher.lesson_generator import LessonGenerator

_DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_lesson(topic: str = "Python", user_level: str = "beginner"):
    """Return the first lesson from LessonGenerator for the given inputs."""
    gen = LessonGenerator()
    lessons = gen.generate(topic=topic, user_level=user_level)
    assert lessons, "LessonGenerator returned no lessons"
    return lessons[0]


# ---------------------------------------------------------------------------
# Difficulty ceiling tests
# ---------------------------------------------------------------------------


class TestDifficultyCeiling:
    """Every exercise difficulty must not exceed one level above user_level."""

    def test_beginner_no_advanced_exercises(self):
        lesson = _get_lesson(user_level="beginner")
        gen = ExerciseGenerator()
        exercises = gen.generate(lesson=lesson, user_level="beginner")
        for ex in exercises:
            assert ex.difficulty != "advanced", (
                f"beginner user got advanced exercise: {ex!r}"
            )

    def test_intermediate_all_within_ceiling(self):
        lesson = _get_lesson(user_level="intermediate")
        gen = ExerciseGenerator()
        exercises = gen.generate(lesson=lesson, user_level="intermediate")
        max_idx = _DIFFICULTY_ORDER.index("advanced")
        for ex in exercises:
            assert _DIFFICULTY_ORDER.index(ex.difficulty) <= max_idx, (
                f"intermediate user got exercise above ceiling: {ex!r}"
            )

    def test_advanced_all_valid_difficulties(self):
        lesson = _get_lesson(user_level="advanced")
        gen = ExerciseGenerator()
        exercises = gen.generate(lesson=lesson, user_level="advanced")
        valid = {"beginner", "intermediate", "advanced"}
        for ex in exercises:
            assert ex.difficulty in valid, (
                f"advanced user got invalid difficulty: {ex.difficulty!r}"
            )


# ---------------------------------------------------------------------------
# Conceptual exercise presence
# ---------------------------------------------------------------------------


class TestConceptualPresence:
    """At least one conceptual exercise must always be present."""

    @pytest.mark.parametrize("user_level", ["beginner", "intermediate", "advanced"])
    def test_at_least_one_conceptual_per_level(self, user_level):
        lesson = _get_lesson(user_level=user_level)
        gen = ExerciseGenerator()
        exercises = gen.generate(lesson=lesson, user_level=user_level)
        conceptual = [e for e in exercises if e.type == "conceptual"]
        assert len(conceptual) >= 1, (
            f"No conceptual exercise for user_level={user_level!r}"
        )


# ---------------------------------------------------------------------------
# Non-empty explanation
# ---------------------------------------------------------------------------


class TestNonEmptyExplanation:
    """Every exercise must have a non-empty explanation."""

    @pytest.mark.parametrize("user_level", ["beginner", "intermediate", "advanced"])
    def test_all_exercises_have_explanation(self, user_level):
        lesson = _get_lesson(user_level=user_level)
        gen = ExerciseGenerator()
        exercises = gen.generate(lesson=lesson, user_level=user_level)
        for ex in exercises:
            assert ex.explanation and ex.explanation.strip(), (
                f"Exercise {ex.id!r} has empty explanation"
            )


# ---------------------------------------------------------------------------
# ValueError on missing/empty lesson field
# ---------------------------------------------------------------------------


class TestInvalidLessonRaisesValueError:
    """Lesson with empty content field must raise ValueError."""

    def test_empty_content_raises_value_error(self):
        """Create a lesson-like object with empty content and verify ValueError."""
        # Build a valid lesson first, then create one with empty content
        # using the schema directly
        from luma.core.teacher.schemas import Lesson

        # We need a lesson with empty content — bypass schema validation
        # by using a simple namespace object
        class _FakeLesson:
            id = "fake_id"
            topic = "Python"
            title = "Intro"
            difficulty = "beginner"
            content = ""  # empty — should trigger ValueError in ExerciseGenerator
            order = 0

        gen = ExerciseGenerator()
        with pytest.raises(ValueError):
            gen.generate(lesson=_FakeLesson(), user_level="beginner")


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


class TestInputNotMutated:
    """generate() must not mutate the lesson or user_level arguments."""

    def test_lesson_not_mutated(self):
        lesson = _get_lesson(user_level="intermediate")
        original_id = lesson.id
        original_topic = lesson.topic
        original_title = lesson.title
        original_difficulty = lesson.difficulty
        original_content = lesson.content
        original_order = lesson.order

        gen = ExerciseGenerator()
        gen.generate(lesson=lesson, user_level="intermediate")

        assert lesson.id == original_id
        assert lesson.topic == original_topic
        assert lesson.title == original_title
        assert lesson.difficulty == original_difficulty
        assert lesson.content == original_content
        assert lesson.order == original_order
