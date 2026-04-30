"""
Unit tests for TeacherMode optional repository injection.

Covers:
- In-memory mode (no repository): TeacherMode works without error, repository never called
- Persistence mode (injected repository): get_progress() called at session start,
  upsert_progress() called after each lesson
- upsert_progress() receives correct user_id, topic, progress, weak_areas
- get_progress() receives correct user_id and topic
- No import from luma.storage in the teacher_mode module

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from luma.core.memory_interface import (
    MemoryEntry,
    MemoryInterface,
    RetrievalResult,
)
from luma.core.teacher.teacher_mode import TeacherMode
from luma.core.teacher.lesson_generator import LessonGenerator
from luma.core.teacher.explanation_engine import ExplanationEngine
from luma.core.teacher.exercise_generator import ExerciseGenerator
from luma.core.teacher.progress_tracker import ProgressTracker
from luma.core.teacher.schemas import (
    Lesson,
    Explanation,
    Exercise,
    ProgressRecord,
    TeachingSession,
)


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------


class FakeMemoryInterface(MemoryInterface):
    """Minimal in-memory store for testing."""

    def __init__(self) -> None:
        self._store: List[dict] = []

    def store(self, content: str, metadata: Optional[dict] = None) -> str:
        entry = {"content": content, "metadata": metadata or {}}
        self._store.append(entry)
        return str(len(self._store))

    def retrieve(self, query=None, params=None, limit=10, **kwargs) -> RetrievalResult:
        category = (params or {}).get("category")
        memories: List[MemoryEntry] = []
        for i, item in enumerate(self._store):
            meta = item.get("metadata") or {}
            if category is None or meta.get("category") == category:
                memories.append(
                    {
                        "id": str(i),
                        "content": item["content"],
                        "metadata": meta,
                        "timestamp": "2024-01-01T00:00:00",
                        "category": meta.get("category", ""),
                        "tags": [],
                    }
                )
        return {
            "memories": memories,
            "total_count": len(memories),
            "query_metadata": {},
        }


def _make_lesson(lesson_id: str, topic: str = "python", order: int = 1) -> Lesson:
    return Lesson(
        id=lesson_id,
        topic=topic,
        title=f"Lesson {lesson_id}",
        difficulty="beginner",
        content=f"Content for {lesson_id}",
        order=order,
    )


class FakeLessonGenerator:
    """Returns a fixed list of lessons."""

    def __init__(self, lessons: List[Lesson]) -> None:
        self._lessons = lessons

    def generate(self, topic: str, user_level: str) -> List[Lesson]:
        return list(self._lessons)


class FakeExplanationEngine:
    def explain(self, lesson: Lesson, adaptation_ctx) -> Explanation:
        return Explanation(
            lesson_id=lesson.id,
            content=f"Explanation for {lesson.id}",
            rationale="test",
        )


class FakeExerciseGenerator:
    def generate(self, lesson: Lesson, user_level: str) -> List[Exercise]:
        return [
            Exercise(
                id=f"ex-{lesson.id}",
                lesson_id=lesson.id,
                type="conceptual",
                difficulty="beginner",
                prompt="What is it?",
                explanation="Because.",
            )
        ]


class FakePersonalizationEngine:
    def personalize(self, *args, **kwargs):
        result = MagicMock()
        result.adaptation = MagicMock()
        return result


class FakeInsightEngine:
    def generate_insights(self, *args, **kwargs):
        report = MagicMock()
        report.insights = []
        return report


def _build_teacher_mode(
    lessons: List[Lesson],
    teacher_repository: Optional[Any] = None,
    memory_interface: Optional[MemoryInterface] = None,
) -> TeacherMode:
    mem = memory_interface or FakeMemoryInterface()
    progress_tracker = ProgressTracker(mem)
    return TeacherMode(
        memory_interface=mem,
        personalization_engine=FakePersonalizationEngine(),
        insight_engine=FakeInsightEngine(),
        lesson_generator=FakeLessonGenerator(lessons),
        explanation_engine=FakeExplanationEngine(),
        exercise_generator=FakeExerciseGenerator(),
        progress_tracker=progress_tracker,
        teacher_repository=teacher_repository,
    )


# ---------------------------------------------------------------------------
# Tests: in-memory mode (no repository) — Requirement 13.4
# ---------------------------------------------------------------------------


class TestInMemoryMode:
    def test_teach_works_without_repository(self):
        """TeacherMode operates normally when teacher_repository=None."""
        lessons = [_make_lesson("l1"), _make_lesson("l2")]
        mode = _build_teacher_mode(lessons)
        session = mode.teach(user_id="alice", topic="python")
        assert session is not None
        assert session.user_id == "alice"
        assert session.topic == "python"

    def test_repository_attribute_is_none_by_default(self):
        """teacher_repository defaults to None."""
        mode = _build_teacher_mode([_make_lesson("l1")])
        assert mode._teacher_repository is None

    def test_session_status_active_when_lessons_remain(self):
        """Session status is 'active' when lessons are processed."""
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons)
        session = mode.teach(user_id="u1", topic="python")
        assert session.status == "active"
        assert len(session.lessons) == 1

    def test_session_status_completed_when_all_done(self):
        """Session status is 'completed' when all lessons already finished."""
        mem = FakeMemoryInterface()
        lessons = [_make_lesson("l1")]
        # Pre-record completion so no lessons remain
        tracker = ProgressTracker(mem)
        tracker.record_completion("u1", "python", "l1", score=1.0)

        mode = TeacherMode(
            memory_interface=mem,
            personalization_engine=FakePersonalizationEngine(),
            insight_engine=FakeInsightEngine(),
            lesson_generator=FakeLessonGenerator(lessons),
            explanation_engine=FakeExplanationEngine(),
            exercise_generator=FakeExerciseGenerator(),
            progress_tracker=tracker,
        )
        session = mode.teach(user_id="u1", topic="python")
        assert session.status == "completed"
        assert session.lessons == []


# ---------------------------------------------------------------------------
# Tests: persistence mode — Requirements 13.1, 13.2, 13.3
# ---------------------------------------------------------------------------


class TestPersistenceMode:
    def test_get_progress_called_at_session_start(self):
        """get_progress() is called once at the start of teach()."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="alice", topic="python")

        mock_repo.get_progress.assert_called_once_with("alice", "python")

    def test_upsert_progress_called_after_each_lesson(self):
        """upsert_progress() is called once per lesson processed."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1"), _make_lesson("l2"), _make_lesson("l3")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="alice", topic="python")

        assert mock_repo.upsert_progress.call_count == 3

    def test_get_progress_called_before_upsert(self):
        """get_progress() is called before any upsert_progress() call."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="alice", topic="python")

        call_names = [c[0] for c in mock_repo.mock_calls]
        get_idx = call_names.index("get_progress")
        upsert_idx = call_names.index("upsert_progress")
        assert get_idx < upsert_idx, "get_progress() must be called before upsert_progress()"

    def test_upsert_progress_receives_correct_user_id_and_topic(self):
        """upsert_progress() receives the correct user_id and topic."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="bob", topic="rust")

        for c in mock_repo.upsert_progress.call_args_list:
            assert c.kwargs.get("user_id") == "bob" or c.args[0] == "bob"
            assert c.kwargs.get("topic") == "rust" or c.args[1] == "rust"

    def test_upsert_progress_receives_float_progress(self):
        """upsert_progress() receives a float progress value."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="u1", topic="python")

        _, kwargs = mock_repo.upsert_progress.call_args
        progress = kwargs.get("progress")
        assert isinstance(progress, float)
        assert 0.0 <= progress <= 1.0

    def test_upsert_progress_receives_weak_areas_list(self):
        """upsert_progress() receives weak_areas as a list."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="u1", topic="python")

        _, kwargs = mock_repo.upsert_progress.call_args
        assert isinstance(kwargs.get("weak_areas"), list)

    def test_session_still_returned_with_repository(self):
        """TeachingSession is returned correctly even when repository is injected."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1"), _make_lesson("l2")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        session = mode.teach(user_id="u1", topic="python")

        assert session is not None
        assert session.status == "active"
        assert len(session.lessons) == 2

    def test_upsert_not_called_when_no_lessons_processed(self):
        """upsert_progress() is not called when all lessons are already completed."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        mem = FakeMemoryInterface()
        lessons = [_make_lesson("l1")]
        tracker = ProgressTracker(mem)
        tracker.record_completion("u1", "python", "l1", score=1.0)

        mode = TeacherMode(
            memory_interface=mem,
            personalization_engine=FakePersonalizationEngine(),
            insight_engine=FakeInsightEngine(),
            lesson_generator=FakeLessonGenerator(lessons),
            explanation_engine=FakeExplanationEngine(),
            exercise_generator=FakeExerciseGenerator(),
            progress_tracker=tracker,
            teacher_repository=mock_repo,
        )

        session = mode.teach(user_id="u1", topic="python")

        assert session.status == "completed"
        mock_repo.upsert_progress.assert_not_called()
        # get_progress is still called at session start
        mock_repo.get_progress.assert_called_once_with("u1", "python")

    def test_progress_value_increases_with_each_lesson(self):
        """Progress value passed to upsert_progress() increases as lessons complete."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1"), _make_lesson("l2"), _make_lesson("l3")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        mode.teach(user_id="u1", topic="python")

        progress_values = [
            c.kwargs.get("progress") for c in mock_repo.upsert_progress.call_args_list
        ]
        # Each call should have a non-decreasing progress value
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]

    def test_default_user_id_forwarded_to_repository(self):
        """Default user_id='default' is forwarded to get_progress() and upsert_progress()."""
        mock_repo = MagicMock()
        mock_repo.get_progress.return_value = None
        lessons = [_make_lesson("l1")]
        mode = _build_teacher_mode(lessons, teacher_repository=mock_repo)

        # TeacherMode.teach() requires explicit user_id; use "default"
        mode.teach(user_id="default", topic="python")

        mock_repo.get_progress.assert_called_once_with("default", "python")
        for c in mock_repo.upsert_progress.call_args_list:
            assert c.kwargs.get("user_id") == "default" or c.args[0] == "default"


# ---------------------------------------------------------------------------
# Tests: no import from luma.storage — Requirement 13.5
# ---------------------------------------------------------------------------


class TestNoStorageImport:
    def test_teacher_mode_does_not_import_luma_storage(self):
        """TeacherMode module must not import from luma.storage."""
        import luma.core.teacher.teacher_mode as module

        source_file = module.__file__
        with open(source_file) as f:
            source = f.read()

        assert "from luma.storage" not in source
        assert "import luma.storage" not in source
