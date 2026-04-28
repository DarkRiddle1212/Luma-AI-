"""
Unit tests for LessonGenerator.

Tests cover: non-empty output, ValueError on bad input, ordering,
immutability, and per-level difficulty filtering.
"""

import pytest

from luma.core.teacher.lesson_generator import LessonGenerator

_DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced"]


# ---------------------------------------------------------------------------
# 1. Known topic produces a non-empty lesson list
# ---------------------------------------------------------------------------

def test_generate_returns_non_empty_list():
    generator = LessonGenerator()
    lessons = generator.generate(topic="Python", user_level="beginner")
    assert len(lessons) > 0


# ---------------------------------------------------------------------------
# 2. Empty string raises ValueError
# ---------------------------------------------------------------------------

def test_generate_raises_on_empty_string():
    generator = LessonGenerator()
    with pytest.raises(ValueError):
        generator.generate(topic="", user_level="beginner")


# ---------------------------------------------------------------------------
# 3. Whitespace-only string raises ValueError
# ---------------------------------------------------------------------------

def test_generate_raises_on_whitespace_only():
    generator = LessonGenerator()
    with pytest.raises(ValueError):
        generator.generate(topic="   ", user_level="beginner")


# ---------------------------------------------------------------------------
# 4. Lesson ordering is correct for each user_level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_level", ["beginner", "intermediate", "advanced"])
def test_lesson_ordering_is_non_decreasing(user_level):
    generator = LessonGenerator()
    lessons = generator.generate(topic="Algebra", user_level=user_level)
    indices = [_DIFFICULTY_ORDER.index(l.difficulty) for l in lessons]
    assert indices == sorted(indices), (
        f"Lessons not ordered for user_level={user_level!r}: "
        f"{[l.difficulty for l in lessons]}"
    )


# ---------------------------------------------------------------------------
# 5. Input arguments are not mutated
# ---------------------------------------------------------------------------

def test_generate_does_not_mutate_topic():
    generator = LessonGenerator()
    topic = "Machine Learning"
    original = topic
    generator.generate(topic=topic, user_level="advanced")
    assert topic == original


# ---------------------------------------------------------------------------
# 6. user_level="beginner" produces only beginner lessons
# ---------------------------------------------------------------------------

def test_beginner_level_produces_only_beginner_lessons():
    generator = LessonGenerator()
    lessons = generator.generate(topic="Chemistry", user_level="beginner")
    difficulties = {l.difficulty for l in lessons}
    assert difficulties == {"beginner"}, (
        f"Expected only beginner lessons, got: {difficulties}"
    )


# ---------------------------------------------------------------------------
# 7. user_level="intermediate" produces beginner and intermediate (no advanced)
# ---------------------------------------------------------------------------

def test_intermediate_level_produces_no_advanced_lessons():
    generator = LessonGenerator()
    lessons = generator.generate(topic="Chemistry", user_level="intermediate")
    difficulties = {l.difficulty for l in lessons}
    assert "advanced" not in difficulties, (
        f"Expected no advanced lessons, got: {difficulties}"
    )
    assert "beginner" in difficulties
    assert "intermediate" in difficulties


# ---------------------------------------------------------------------------
# 8. user_level="advanced" produces all three difficulty levels
# ---------------------------------------------------------------------------

def test_advanced_level_produces_all_difficulty_levels():
    generator = LessonGenerator()
    lessons = generator.generate(topic="Chemistry", user_level="advanced")
    difficulties = {l.difficulty for l in lessons}
    assert difficulties == {"beginner", "intermediate", "advanced"}, (
        f"Expected all difficulty levels, got: {difficulties}"
    )
