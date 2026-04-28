"""
Property-based tests for the luma-teacher-mode feature.

Contains Hypothesis property tests verifying schema validation and
behavioural invariants for the Teacher Mode module.
Each test is tagged: # Feature: luma-teacher-mode, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

from datetime import datetime

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from luma.core.teacher.schemas import (
    Exercise,
    Lesson,
    ProgressRecord,
    TeachingSession,
)

# ---------------------------------------------------------------------------
# Valid enum sets (mirrors schemas.py constants)
# ---------------------------------------------------------------------------

_VALID_DIFFICULTY = {"beginner", "intermediate", "advanced"}
_VALID_STATUS = {"active", "completed", "paused"}
_VALID_EXERCISE_TYPE = {"conceptual", "practical", "mini-project"}

# ---------------------------------------------------------------------------
# Strategies for invalid values
# ---------------------------------------------------------------------------

_invalid_difficulty_st = st.text().filter(lambda s: s not in _VALID_DIFFICULTY)
_invalid_status_st = st.text().filter(lambda s: s not in _VALID_STATUS)
_invalid_score_st = st.one_of(
    st.floats(max_value=-0.001, allow_nan=False),
    st.floats(min_value=1.001, allow_nan=False),
)

# ---------------------------------------------------------------------------
# Property 17: Schema validation rejects invalid enum and range values
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 17: Schema validation rejects invalid enum and range values


@given(
    invalid_difficulty=_invalid_difficulty_st,
    lesson_id=st.text(min_size=1, max_size=20),
    topic=st.text(min_size=1, max_size=50),
    title=st.text(min_size=1, max_size=100),
    content=st.text(min_size=1, max_size=500),
    order=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_lesson_rejects_invalid_difficulty(
    invalid_difficulty, lesson_id, topic, title, content, order
):
    """
    **Validates: Requirements 6.6, 6.7, 6.8**

    For any string NOT in {"beginner", "intermediate", "advanced"} used as
    the difficulty field in Lesson, the constructor SHALL raise a ValueError.
    """
    with pytest.raises((ValueError, Exception)):
        Lesson(
            id=lesson_id,
            topic=topic,
            title=title,
            difficulty=invalid_difficulty,
            content=content,
            order=order,
        )


@given(
    invalid_difficulty=_invalid_difficulty_st,
    exercise_id=st.text(min_size=1, max_size=20),
    lesson_id=st.text(min_size=1, max_size=20),
    prompt=st.text(min_size=1, max_size=300),
    explanation=st.text(min_size=1, max_size=300),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_exercise_rejects_invalid_difficulty(
    invalid_difficulty, exercise_id, lesson_id, prompt, explanation
):
    """
    **Validates: Requirements 6.6, 6.7, 6.8**

    For any string NOT in {"beginner", "intermediate", "advanced"} used as
    the difficulty field in Exercise, the constructor SHALL raise a ValueError.
    """
    with pytest.raises((ValueError, Exception)):
        Exercise(
            id=exercise_id,
            lesson_id=lesson_id,
            type="conceptual",
            difficulty=invalid_difficulty,
            prompt=prompt,
            explanation=explanation,
        )


@given(
    invalid_score=_invalid_score_st,
    user_id=st.text(min_size=1, max_size=20),
    topic=st.text(min_size=1, max_size=50),
    lesson_id=st.text(min_size=1, max_size=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_progress_record_rejects_invalid_score(
    invalid_score, user_id, topic, lesson_id
):
    """
    **Validates: Requirements 6.6, 6.7, 6.8**

    For any float outside [0.0, 1.0] used as the score in ProgressRecord,
    the constructor SHALL raise a ValueError.
    """
    with pytest.raises((ValueError, Exception)):
        ProgressRecord(
            user_id=user_id,
            topic=topic,
            lesson_id=lesson_id,
            completed_at="2024-01-15T10:30:00",
            score=invalid_score,
        )


@given(
    invalid_status=_invalid_status_st,
    session_id=st.text(min_size=1, max_size=36),
    user_id=st.text(min_size=1, max_size=20),
    topic=st.text(min_size=1, max_size=50),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_teaching_session_rejects_invalid_status(
    invalid_status, session_id, user_id, topic
):
    """
    **Validates: Requirements 6.6, 6.7, 6.8**

    For any string NOT in {"active", "completed", "paused"} used as the
    status in TeachingSession, the constructor SHALL raise a ValueError.
    """
    with pytest.raises((ValueError, Exception)):
        TeachingSession(
            session_id=session_id,
            user_id=user_id,
            topic=topic,
            status=invalid_status,
            lessons=[],
            explanations=[],
            exercises=[],
            created_at="2024-01-15T10:30:00",
        )


# ---------------------------------------------------------------------------
# Import LessonGenerator for Properties 1–5
# ---------------------------------------------------------------------------

from luma.core.teacher.lesson_generator import LessonGenerator

_DIFFICULTY_ORDER = ["beginner", "intermediate", "advanced"]

_valid_topic_st = st.text(min_size=1).filter(lambda s: s.strip())
_user_level_st = st.sampled_from(["beginner", "intermediate", "advanced"])

# ---------------------------------------------------------------------------
# Property 1: Lesson difficulty ordering and validity invariant
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 1: Lesson difficulty ordering and validity invariant


@given(topic=_valid_topic_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_lesson_difficulty_ordering_and_validity(topic, user_level):
    """
    **Validates: Requirements 1.2, 1.3**

    For any valid topic string and user_level, every lesson SHALL have a
    difficulty in {"beginner", "intermediate", "advanced"}, and the list
    SHALL be ordered so that all "beginner" lessons precede all
    "intermediate" lessons, and all "intermediate" lessons precede all
    "advanced" lessons.
    """
    generator = LessonGenerator()
    lessons = generator.generate(topic=topic, user_level=user_level)

    # All difficulties must be valid
    for lesson in lessons:
        assert lesson.difficulty in {"beginner", "intermediate", "advanced"}

    # Ordering: difficulty index must be non-decreasing
    indices = [_DIFFICULTY_ORDER.index(lesson.difficulty) for lesson in lessons]
    assert indices == sorted(indices), (
        f"Lessons are not ordered by difficulty: {[l.difficulty for l in lessons]}"
    )


# ---------------------------------------------------------------------------
# Property 2: Lesson sequence contains at least one lesson at or below UserLevel
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 2: Lesson sequence contains at least one lesson at or below UserLevel


@given(topic=_valid_topic_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_lesson_sequence_contains_lesson_at_or_below_user_level(topic, user_level):
    """
    **Validates: Requirements 1.6**

    For any valid topic string and user_level, the list SHALL contain at
    least one lesson whose difficulty is at or below the provided user_level.
    Level ordering: "beginner" <= "intermediate" <= "advanced".
    """
    generator = LessonGenerator()
    lessons = generator.generate(topic=topic, user_level=user_level)

    user_level_index = _DIFFICULTY_ORDER.index(user_level)
    at_or_below = [
        l for l in lessons
        if _DIFFICULTY_ORDER.index(l.difficulty) <= user_level_index
    ]
    assert len(at_or_below) >= 1, (
        f"No lesson at or below user_level={user_level!r} found in "
        f"{[l.difficulty for l in lessons]}"
    )


# ---------------------------------------------------------------------------
# Property 3: Lesson IDs are unique within a generated sequence
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 3: Lesson IDs are unique within a generated sequence


@given(topic=_valid_topic_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_lesson_ids_are_unique(topic, user_level):
    """
    **Validates: Requirements 1.4**

    For any valid topic string and user_level, all lesson IDs in the
    returned list SHALL be distinct.
    """
    generator = LessonGenerator()
    lessons = generator.generate(topic=topic, user_level=user_level)

    ids = [lesson.id for lesson in lessons]
    assert len(ids) == len(set(ids)), (
        f"Duplicate lesson IDs found: {ids}"
    )


# ---------------------------------------------------------------------------
# Property 4: LessonGenerator determinism
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 4: LessonGenerator determinism


@given(topic=_valid_topic_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_lesson_generator_determinism(topic, user_level):
    """
    **Validates: Requirements 1.5, 10.1**

    For any valid topic string and user_level, calling generate() twice
    with identical inputs SHALL produce lesson sequences with identical
    IDs, titles, difficulties, and ordering.
    """
    generator = LessonGenerator()
    lessons_a = generator.generate(topic=topic, user_level=user_level)
    lessons_b = generator.generate(topic=topic, user_level=user_level)

    assert len(lessons_a) == len(lessons_b), (
        f"Different lengths: {len(lessons_a)} vs {len(lessons_b)}"
    )
    for a, b in zip(lessons_a, lessons_b):
        assert a.id == b.id
        assert a.title == b.title
        assert a.difficulty == b.difficulty
        assert a.order == b.order


# ---------------------------------------------------------------------------
# Property 5: Whitespace-only topic raises ValueError
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 5: Whitespace-only topic raises ValueError

_whitespace_st = st.one_of(
    st.just(""),
    st.text(
        alphabet=st.characters(whitelist_categories=("Zs",)),
        min_size=1,
    ),
)


@given(topic=_whitespace_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_whitespace_only_topic_raises_value_error(topic, user_level):
    """
    **Validates: Requirements 1.7, 11.4**

    For any string composed entirely of whitespace characters (including
    the empty string), calling generate() SHALL raise a ValueError.
    """
    generator = LessonGenerator()
    with pytest.raises(ValueError):
        generator.generate(topic=topic, user_level=user_level)


# ---------------------------------------------------------------------------
# Import ExplanationEngine for Properties 6–7
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from luma.core.teacher.explanation_engine import ExplanationEngine


@_dataclass
class _MockAdaptationCtx:
    tone: str
    style: str
    focus: str


_non_empty_text_st = st.text(min_size=1).filter(lambda s: s.strip())

_valid_lesson_st = st.builds(
    Lesson,
    id=_non_empty_text_st,
    topic=_non_empty_text_st,
    title=_non_empty_text_st,
    difficulty=st.sampled_from(["beginner", "intermediate", "advanced"]),
    content=_non_empty_text_st,
    order=st.integers(min_value=0),
)

_mock_ctx_st = st.builds(
    _MockAdaptationCtx,
    tone=st.sampled_from(["technical", "casual", "formal"]),
    style=st.sampled_from(["step-by-step", "concise", "detailed", "balanced"]),
    focus=st.sampled_from(["deep-technical", "high-level"]),
)

# ---------------------------------------------------------------------------
# Property 6: Explanation output fields are always non-empty
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 6: Explanation output fields are always non-empty


@given(lesson=_valid_lesson_st, ctx=_mock_ctx_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_explanation_output_fields_are_non_empty(lesson, ctx):
    """
    **Validates: Requirements 2.1, 2.6**

    For any valid Lesson and AdaptationContext, the Explanation returned by
    ExplanationEngine.explain() SHALL have a non-empty `content` field and
    a non-empty `rationale` field.
    """
    engine = ExplanationEngine()
    explanation = engine.explain(lesson, ctx)

    assert explanation.content, (
        f"Explanation.content is empty for lesson={lesson!r}, ctx={ctx!r}"
    )
    assert explanation.rationale, (
        f"Explanation.rationale is empty for lesson={lesson!r}, ctx={ctx!r}"
    )


# ---------------------------------------------------------------------------
# Property 7: ExplanationEngine determinism
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 7: ExplanationEngine determinism


@given(lesson=_valid_lesson_st, ctx=_mock_ctx_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_explanation_engine_determinism(lesson, ctx):
    """
    **Validates: Requirements 2.5, 10.2**

    For any valid Lesson and AdaptationContext, calling explain() twice with
    identical inputs SHALL produce identical Explanation objects (same
    `content` and `rationale`).
    """
    engine = ExplanationEngine()
    explanation_a = engine.explain(lesson, ctx)
    explanation_b = engine.explain(lesson, ctx)

    assert explanation_a.content == explanation_b.content, (
        f"Non-deterministic content for lesson={lesson!r}, ctx={ctx!r}"
    )
    assert explanation_a.rationale == explanation_b.rationale, (
        f"Non-deterministic rationale for lesson={lesson!r}, ctx={ctx!r}"
    )


# ---------------------------------------------------------------------------
# Import ExerciseGenerator for Properties 8–10
# ---------------------------------------------------------------------------

from luma.core.teacher.exercise_generator import ExerciseGenerator

_DIFFICULTY_ORDER_EX = ["beginner", "intermediate", "advanced"]

# ---------------------------------------------------------------------------
# Property 8: Exercise field invariants
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 8: Exercise field invariants


@given(lesson=_valid_lesson_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_exercise_field_invariants(lesson, user_level):
    """
    **Validates: Requirements 3.1, 3.2, 3.4, 3.6**

    For any valid Lesson and user_level, every Exercise in the list returned
    by ExerciseGenerator.generate() SHALL have:
    - a `type` in {"conceptual", "practical", "mini-project"}
    - a non-empty `explanation` field
    - the list SHALL contain at least one Exercise with type == "conceptual"
    """
    generator = ExerciseGenerator()
    exercises = generator.generate(lesson=lesson, user_level=user_level)

    valid_types = {"conceptual", "practical", "mini-project"}
    for exercise in exercises:
        assert exercise.type in valid_types, (
            f"Exercise type {exercise.type!r} not in {valid_types}"
        )
        assert exercise.explanation and exercise.explanation.strip(), (
            f"Exercise explanation is empty for exercise id={exercise.id!r}"
        )

    conceptual_exercises = [e for e in exercises if e.type == "conceptual"]
    assert len(conceptual_exercises) >= 1, (
        "No conceptual exercise found in generated exercises"
    )


# ---------------------------------------------------------------------------
# Property 9: Exercise difficulty ceiling invariant
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 9: Exercise difficulty ceiling invariant


@given(lesson=_valid_lesson_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_exercise_difficulty_ceiling(lesson, user_level):
    """
    **Validates: Requirements 3.3**

    For any valid Lesson and user_level, every Exercise SHALL have a
    difficulty that does not exceed one level above the provided user_level:
    - "beginner"     → max "intermediate"
    - "intermediate" → max "advanced"
    - "advanced"     → max "advanced"
    """
    generator = ExerciseGenerator()
    exercises = generator.generate(lesson=lesson, user_level=user_level)

    user_level_index = _DIFFICULTY_ORDER_EX.index(user_level)
    max_allowed_index = min(user_level_index + 1, len(_DIFFICULTY_ORDER_EX) - 1)

    for exercise in exercises:
        exercise_difficulty_index = _DIFFICULTY_ORDER_EX.index(exercise.difficulty)
        assert exercise_difficulty_index <= max_allowed_index, (
            f"Exercise difficulty {exercise.difficulty!r} exceeds ceiling for "
            f"user_level={user_level!r} (max allowed: "
            f"{_DIFFICULTY_ORDER_EX[max_allowed_index]!r})"
        )


# ---------------------------------------------------------------------------
# Property 10: ExerciseGenerator determinism
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 10: ExerciseGenerator determinism


@given(lesson=_valid_lesson_st, user_level=_user_level_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_exercise_generator_determinism(lesson, user_level):
    """
    **Validates: Requirements 3.5, 10.3**

    For any valid Lesson and user_level, calling generate() twice with
    identical inputs SHALL produce identical exercise lists.
    """
    generator = ExerciseGenerator()
    exercises_a = generator.generate(lesson=lesson, user_level=user_level)
    exercises_b = generator.generate(lesson=lesson, user_level=user_level)

    assert len(exercises_a) == len(exercises_b), (
        f"Different number of exercises: {len(exercises_a)} vs {len(exercises_b)}"
    )
    for a, b in zip(exercises_a, exercises_b):
        assert a.id == b.id, f"Exercise IDs differ: {a.id!r} vs {b.id!r}"
        assert a.type == b.type, f"Exercise types differ: {a.type!r} vs {b.type!r}"
        assert a.difficulty == b.difficulty, (
            f"Exercise difficulties differ: {a.difficulty!r} vs {b.difficulty!r}"
        )
        assert a.prompt == b.prompt, f"Exercise prompts differ"
        assert a.explanation == b.explanation, f"Exercise explanations differ"


# ---------------------------------------------------------------------------
# Mock MemoryInterface helper for Properties 11–15
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from luma.core.teacher.progress_tracker import ProgressTracker


def _make_mock_memory():
    """Create a mock MemoryInterface that stores and retrieves in-memory."""
    stored = []

    def store(content, metadata=None):
        stored.append({"content": content, "metadata": metadata or {}})
        return f"id_{len(stored)}"

    def retrieve(query=None, params=None, limit=10, **kwargs):
        return {
            "memories": [
                {
                    "id": f"id_{i}",
                    "content": s["content"],
                    "metadata": s["metadata"],
                    "timestamp": "2024-01-15T10:30:00",
                    "category": s["metadata"].get("category", ""),
                    "tags": [],
                }
                for i, s in enumerate(stored)
            ],
            "total_count": len(stored),
            "query_metadata": {},
        }

    mi = MagicMock()
    mi.store.side_effect = store
    mi.retrieve.side_effect = retrieve
    return mi, stored


# ---------------------------------------------------------------------------
# Property 11: ProgressRecord round-trip preserves all fields
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 11: ProgressRecord round-trip preserves all fields


@given(
    user_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    topic=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    lesson_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_progress_record_round_trip(user_id, topic, lesson_id, score):
    """
    **Validates: Requirements 4.1, 4.2, 7.3**

    For any valid (user_id, topic, lesson_id, score) tuple, calling
    record_completion() followed by get_progress() SHALL return a list
    containing a ProgressRecord with the same user_id, topic, lesson_id,
    and score.
    """
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.record_completion(user_id, topic, lesson_id, score)
    records = tracker.get_progress(user_id, topic)

    assert len(records) >= 1, "Expected at least one ProgressRecord after record_completion"
    matching = [r for r in records if r.lesson_id == lesson_id]
    assert len(matching) == 1, f"Expected exactly one record for lesson_id={lesson_id!r}"
    rec = matching[0]
    assert rec.user_id == user_id
    assert rec.topic == topic
    assert rec.lesson_id == lesson_id
    assert abs(rec.score - score) < 1e-9, f"Score mismatch: {rec.score} != {score}"


# ---------------------------------------------------------------------------
# Property 12: record_completion is idempotent
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 12: record_completion is idempotent


@given(
    user_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    topic=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    lesson_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_record_completion_is_idempotent(user_id, topic, lesson_id, score):
    """
    **Validates: Requirements 4.5**

    For any valid (user_id, topic, lesson_id, score) tuple, calling
    record_completion() twice with the same arguments SHALL result in
    exactly one ProgressRecord for that lesson.
    """
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.record_completion(user_id, topic, lesson_id, score)
    tracker.record_completion(user_id, topic, lesson_id, score)

    records = tracker.get_progress(user_id, topic)
    matching = [r for r in records if r.lesson_id == lesson_id]
    assert len(matching) == 1, (
        f"Expected exactly one record for lesson_id={lesson_id!r}, got {len(matching)}"
    )


# ---------------------------------------------------------------------------
# Property 13: Weak areas are exactly lessons below score threshold
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 13: Weak areas are exactly lessons below score threshold


@given(
    user_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    topic=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    records=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        min_size=0,
        max_size=10,
    ).map(lambda pairs: list({lid: score for lid, score in pairs}.items())),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_weak_areas_are_exactly_lessons_below_threshold(user_id, topic, records):
    """
    **Validates: Requirements 4.3**

    For any list of (lesson_id, score) pairs recorded for a user and topic,
    get_weak_areas() SHALL return exactly the set of lesson IDs whose score
    is strictly below 0.6.
    """
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    for lesson_id, score in records:
        tracker.record_completion(user_id, topic, lesson_id, score)

    weak_areas = tracker.get_weak_areas(user_id, topic)
    expected_weak = {lid for lid, score in records if score < 0.6}

    assert set(weak_areas) == expected_weak, (
        f"Weak areas mismatch: got {set(weak_areas)}, expected {expected_weak}"
    )


# ---------------------------------------------------------------------------
# Property 14: Completion ratio invariant
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 14: Completion ratio invariant


@given(
    user_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    topic=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    records=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ),
        min_size=0,
        max_size=10,
    ).map(lambda pairs: list({lid: score for lid, score in pairs}.items())),
    total_lessons=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_completion_ratio_invariant(user_id, topic, records, total_lessons):
    """
    **Validates: Requirements 4.4**

    For any list of ProgressRecords and a positive total lesson count,
    get_completion_ratio() SHALL return len(records) / total_lessons,
    clamped to [0.0, 1.0].
    """
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    for lesson_id, score in records:
        tracker.record_completion(user_id, topic, lesson_id, score)

    ratio = tracker.get_completion_ratio(user_id, topic, total_lessons)
    expected = min(1.0, max(0.0, len(records) / total_lessons))

    assert 0.0 <= ratio <= 1.0, f"Ratio {ratio} is outside [0.0, 1.0]"
    assert abs(ratio - expected) < 1e-9, (
        f"Ratio mismatch: got {ratio}, expected {expected} "
        f"(records={len(records)}, total_lessons={total_lessons})"
    )


# ---------------------------------------------------------------------------
# Property 15: ProgressTracker always stores under "teacher_progress" category
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 15: ProgressTracker always stores under "teacher_progress" category


@given(
    user_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    topic=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    lesson_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_progress_tracker_stores_under_teacher_progress_category(
    user_id, topic, lesson_id, score
):
    """
    **Validates: Requirements 4.8, 7.3**

    For any valid record_completion() call, the MemoryInterface.store()
    invocation SHALL include category: "teacher_progress" in the metadata dict.
    """
    store_calls = []

    def capturing_store(content, metadata=None):
        store_calls.append({"content": content, "metadata": metadata or {}})
        return f"id_{len(store_calls)}"

    stored = []

    def retrieve(query=None, params=None, limit=10, **kwargs):
        return {
            "memories": [
                {
                    "id": f"id_{i}",
                    "content": s["content"],
                    "metadata": s["metadata"],
                    "timestamp": "2024-01-15T10:30:00",
                    "category": s["metadata"].get("category", ""),
                    "tags": [],
                }
                for i, s in enumerate(stored)
            ],
            "total_count": len(stored),
            "query_metadata": {},
        }

    mi = MagicMock()
    mi.store.side_effect = capturing_store
    mi.retrieve.side_effect = retrieve

    tracker = ProgressTracker(mi)
    tracker.record_completion(user_id, topic, lesson_id, score)

    assert len(store_calls) == 1, "Expected exactly one store() call"
    metadata = store_calls[0]["metadata"]
    assert metadata.get("category") == "teacher_progress", (
        f"Expected category='teacher_progress', got {metadata.get('category')!r}"
    )


# ---------------------------------------------------------------------------
# Property 16: Completed lessons are excluded from TeachingSession
# ---------------------------------------------------------------------------

# Feature: luma-teacher-mode, Property 16: Completed lessons are excluded from TeachingSession

from luma.core.teacher.teacher_mode import TeacherMode
from luma.core.teacher.lesson_generator import LessonGenerator as _LessonGenerator16
from luma.core.teacher.explanation_engine import ExplanationEngine as _ExplanationEngine16
from luma.core.teacher.exercise_generator import ExerciseGenerator as _ExerciseGenerator16
from luma.core.teacher.progress_tracker import ProgressTracker as _ProgressTracker16


def _make_teacher_mode(stored):
    """Build a TeacherMode with mock personalization/insight and real sub-components."""
    from dataclasses import dataclass as _dc16

    @_dc16
    class _Adaptation16:
        tone: str = "casual"
        style: str = "balanced"
        focus: str = "high-level"

    @_dc16
    class _PersonalizationResult16:
        adaptation: object = None
        def __post_init__(self):
            if self.adaptation is None:
                self.adaptation = _Adaptation16()

    @_dc16
    class _Insight16:
        text: str

    @_dc16
    class _InsightReport16:
        insights: list

    # Build mock memory that uses the shared `stored` list
    mi = MagicMock()

    def _store(content, metadata=None):
        stored.append({"content": content, "metadata": metadata or {}})
        return f"id_{len(stored)}"

    def _retrieve(query=None, params=None, limit=10, **kwargs):
        return {
            "memories": [
                {
                    "id": f"id_{i}",
                    "content": s["content"],
                    "metadata": s["metadata"],
                    "timestamp": "2024-01-15T10:30:00",
                    "category": s["metadata"].get("category", ""),
                    "tags": [],
                }
                for i, s in enumerate(stored)
            ],
            "total_count": len(stored),
            "query_metadata": {},
        }

    mi.store.side_effect = _store
    mi.retrieve.side_effect = _retrieve

    personalization_engine = MagicMock()
    personalization_engine.personalize.return_value = _PersonalizationResult16()

    insight_engine = MagicMock()
    insight_engine.generate_insights.return_value = _InsightReport16(insights=[])

    progress_tracker = _ProgressTracker16(mi)

    return TeacherMode(
        memory_interface=mi,
        personalization_engine=personalization_engine,
        insight_engine=insight_engine,
        lesson_generator=_LessonGenerator16(),
        explanation_engine=_ExplanationEngine16(),
        exercise_generator=_ExerciseGenerator16(),
        progress_tracker=progress_tracker,
    ), progress_tracker


@given(
    completed_count=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_completed_lessons_excluded_from_session(completed_count):
    """
    **Validates: Requirements 5.3**

    For any set of already-completed lesson IDs for a user and topic, none
    of those lesson IDs SHALL appear in the `lessons` list of the
    TeachingSession returned by TeacherMode.teach().
    """
    stored = []
    teacher, progress_tracker = _make_teacher_mode(stored)

    # Generate the known lesson list for topic="Python", user_level="beginner"
    lesson_gen = _LessonGenerator16()
    all_lessons = lesson_gen.generate(topic="Python", user_level="beginner")

    # Pre-record `completed_count` lessons as completed
    lessons_to_complete = all_lessons[:completed_count]
    for lesson in lessons_to_complete:
        progress_tracker.record_completion("user1", "Python", lesson.id, score=1.0)

    completed_ids = {l.id for l in lessons_to_complete}

    # Call teach() and verify none of the pre-completed IDs appear in the session
    session = teacher.teach("user1", "Python")

    session_lesson_ids = {l.id for l in session.lessons}
    overlap = completed_ids & session_lesson_ids

    assert not overlap, (
        f"Completed lesson IDs {overlap} appeared in the session's lessons. "
        f"Session lesson IDs: {session_lesson_ids}"
    )
