"""
Unit tests for TeacherMode orchestrator.

Tests the full orchestration flow, dependency call order, error handling,
and session properties.
"""

import uuid
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

from luma.core.teacher.teacher_mode import TeacherMode
from luma.core.teacher.lesson_generator import LessonGenerator
from luma.core.teacher.explanation_engine import ExplanationEngine
from luma.core.teacher.exercise_generator import ExerciseGenerator
from luma.core.teacher.progress_tracker import ProgressTracker
from luma.core.teacher.schemas import TeachingSession, TeachingSessionError


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockAdaptation:
    tone: str = "casual"
    style: str = "balanced"
    focus: str = "high-level"


@dataclass
class _MockPersonalizationResult:
    adaptation: object = None

    def __post_init__(self):
        if self.adaptation is None:
            self.adaptation = _MockAdaptation()


@dataclass
class _MockInsight:
    text: str


@dataclass
class _MockInsightReport:
    insights: list


# ---------------------------------------------------------------------------
# Shared mock memory factory
# ---------------------------------------------------------------------------

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


def _make_teacher_mode(mi=None):
    """Build a TeacherMode with real sub-components and mock engines."""
    if mi is None:
        mi, _ = _make_mock_memory()

    personalization_engine = MagicMock()
    personalization_engine.personalize.return_value = _MockPersonalizationResult()

    insight_engine = MagicMock()
    insight_engine.generate_insights.return_value = _MockInsightReport(insights=[])

    progress_tracker = ProgressTracker(mi)

    teacher = TeacherMode(
        memory_interface=mi,
        personalization_engine=personalization_engine,
        insight_engine=insight_engine,
        lesson_generator=LessonGenerator(),
        explanation_engine=ExplanationEngine(),
        exercise_generator=ExerciseGenerator(),
        progress_tracker=progress_tracker,
    )
    return teacher, personalization_engine, insight_engine, progress_tracker, mi


# ---------------------------------------------------------------------------
# Test 1: Full orchestration — verify call order
# ---------------------------------------------------------------------------

def test_full_orchestration_call_order():
    """
    Test full TeacherMode.teach() flow with all dependencies mocked,
    verifying that each component is called in the correct order.
    """
    mi, stored = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    call_order = []

    original_personalize = pe.personalize.side_effect
    pe.personalize.side_effect = lambda *a, **kw: (
        call_order.append("personalize") or _MockPersonalizationResult()
    )
    ie.generate_insights.side_effect = lambda *a, **kw: (
        call_order.append("generate_insights") or _MockInsightReport(insights=[])
    )

    session = teacher.teach("user1", "Python")

    assert "personalize" in call_order
    assert "generate_insights" in call_order

    # personalize must come before generate_insights
    assert call_order.index("personalize") < call_order.index("generate_insights")

    # Session must be a valid TeachingSession
    assert isinstance(session, TeachingSession)
    assert session.user_id == "user1"
    assert session.topic == "Python"

    # personalization_engine.personalize() was called
    pe.personalize.assert_called_once_with("user1", "Python")

    # insight_engine.generate_insights() was called
    ie.generate_insights.assert_called_once_with(namespace="Python")

    # memory.store() was called with category="teacher_sessions"
    teacher_session_calls = [
        c for c in mi.store.call_args_list
        if c[0][1].get("category") == "teacher_sessions"
        if len(c[0]) > 1
    ]
    # Check via kwargs or positional
    session_store_calls = []
    for c in mi.store.call_args_list:
        args, kwargs = c
        meta = args[1] if len(args) > 1 else kwargs.get("metadata", {})
        if isinstance(meta, dict) and meta.get("category") == "teacher_sessions":
            session_store_calls.append(c)
    assert len(session_store_calls) >= 1, (
        "Expected at least one store() call with category='teacher_sessions'"
    )


# ---------------------------------------------------------------------------
# Test 2: AdaptationContext retrieved before explanation generation
# ---------------------------------------------------------------------------

def test_personalize_called_before_explain():
    """
    Test that personalization_engine.personalize() is called before any
    explanation is generated (AdaptationContext is available first).
    """
    mi, _ = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    call_order = []

    pe.personalize.side_effect = lambda *a, **kw: (
        call_order.append("personalize") or _MockPersonalizationResult()
    )
    ie.generate_insights.side_effect = lambda *a, **kw: (
        call_order.append("generate_insights") or _MockInsightReport(insights=[])
    )

    # Wrap ExplanationEngine.explain to track calls
    original_explain = teacher._explanation_engine.explain

    def tracked_explain(lesson, ctx):
        call_order.append("explain")
        return original_explain(lesson, ctx)

    teacher._explanation_engine.explain = tracked_explain

    session = teacher.teach("user1", "Python")

    assert "personalize" in call_order
    if "explain" in call_order:
        assert call_order.index("personalize") < call_order.index("explain"), (
            "personalize() must be called before explain()"
        )


# ---------------------------------------------------------------------------
# Test 3: Completed lessons are excluded from session
# ---------------------------------------------------------------------------

def test_completed_lessons_excluded():
    """
    Test that already-completed lessons are excluded from the session lesson list.
    """
    mi, _ = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    # Generate lessons to know their IDs
    lesson_gen = LessonGenerator()
    all_lessons = lesson_gen.generate(topic="Python", user_level="beginner")

    # Pre-complete the first lesson
    first_lesson = all_lessons[0]
    pt.record_completion("user1", "Python", first_lesson.id, score=1.0)

    session = teacher.teach("user1", "Python")

    session_lesson_ids = {l.id for l in session.lessons}
    assert first_lesson.id not in session_lesson_ids, (
        f"Completed lesson {first_lesson.id!r} should not appear in session lessons"
    )


# ---------------------------------------------------------------------------
# Test 4: status="completed" when no incomplete lessons remain
# ---------------------------------------------------------------------------

def test_status_completed_when_all_lessons_done():
    """
    Test that status="completed" is returned when no incomplete lessons remain.

    TeacherMode infers user_level from progress history (avg score >= 0.8 → "advanced"),
    so we must complete all lessons at the inferred level to get status="completed".
    We use score=0.0 to keep user_level at "beginner" (no history → beginner),
    then complete all beginner lessons.
    """
    mi, _ = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    # With no prior history, user_level defaults to "beginner" → 2 lessons.
    # Complete all beginner lessons with score=0.0 so avg stays below 0.5
    # (keeps user_level at "beginner" on the next call too).
    lesson_gen = LessonGenerator()
    all_lessons = lesson_gen.generate(topic="Python", user_level="beginner")
    for lesson in all_lessons:
        pt.record_completion("user1", "Python", lesson.id, score=0.0)

    # Now avg score = 0.0 < 0.5 → user_level stays "beginner" → same 2 lessons
    # → all already completed → status="completed"
    session = teacher.teach("user1", "Python")

    assert session.status == "completed", (
        f"Expected status='completed', got {session.status!r}"
    )
    assert session.lessons == [], "Expected empty lessons list when all completed"


# ---------------------------------------------------------------------------
# Test 5: session_id is a valid UUID
# ---------------------------------------------------------------------------

def test_session_id_is_valid_uuid():
    """
    Test that session_id is a valid UUID in every returned TeachingSession.
    """
    mi, _ = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    session = teacher.teach("user1", "Python")

    # Validate that session_id parses as a UUID without raising
    parsed = uuid.UUID(session.session_id)
    assert str(parsed) == session.session_id, (
        f"session_id {session.session_id!r} is not a canonical UUID string"
    )


# ---------------------------------------------------------------------------
# Test 6: TeachingSessionError raised and logged on sub-component failure
# ---------------------------------------------------------------------------

def test_teaching_session_error_raised_and_logged():
    """
    Test that TeachingSessionError is raised and logged when any sub-component raises.
    """
    mi, _ = _make_mock_memory()

    mock_logger = MagicMock()

    personalization_engine = MagicMock()
    personalization_engine.personalize.side_effect = RuntimeError("personalize failed")

    insight_engine = MagicMock()
    insight_engine.generate_insights.return_value = _MockInsightReport(insights=[])

    progress_tracker = ProgressTracker(mi)

    teacher = TeacherMode(
        memory_interface=mi,
        personalization_engine=personalization_engine,
        insight_engine=insight_engine,
        lesson_generator=LessonGenerator(),
        explanation_engine=ExplanationEngine(),
        exercise_generator=ExerciseGenerator(),
        progress_tracker=progress_tracker,
        logger=mock_logger,
    )

    with pytest.raises(TeachingSessionError):
        teacher.teach("user1", "Python")

    # Logger should have been called
    mock_logger.log.assert_called_once()
    log_args = mock_logger.log.call_args
    assert "teacher_mode_error" in log_args[0] or "teacher_mode_error" in str(log_args)


# ---------------------------------------------------------------------------
# Test 7: MemoryInterface.store() called with category="teacher_sessions"
# ---------------------------------------------------------------------------

def test_memory_store_called_with_teacher_sessions_category():
    """
    Test that MemoryInterface.store() is called with category="teacher_sessions"
    after the session concludes.
    """
    mi, stored = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    teacher.teach("user1", "Python")

    # Find store calls with category="teacher_sessions"
    session_store_calls = []
    for c in mi.store.call_args_list:
        args, kwargs = c
        meta = args[1] if len(args) > 1 else kwargs.get("metadata", {})
        if isinstance(meta, dict) and meta.get("category") == "teacher_sessions":
            session_store_calls.append(c)

    assert len(session_store_calls) >= 1, (
        "Expected at least one MemoryInterface.store() call with "
        "category='teacher_sessions'"
    )


# ---------------------------------------------------------------------------
# Test 8: ProgressTracker delegation — no direct MemoryInterface calls for progress
# ---------------------------------------------------------------------------

def test_progress_tracker_delegation():
    """
    Test that ProgressTracker is used for all progress data and TeacherMode
    does not call MemoryInterface directly for progress operations.
    """
    mi, stored = _make_mock_memory()
    teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

    teacher.teach("user1", "Python")

    # All store() calls should be either "teacher_progress" (from ProgressTracker)
    # or "teacher_sessions" (from TeacherMode). No raw progress calls from TeacherMode.
    for c in mi.store.call_args_list:
        args, kwargs = c
        meta = args[1] if len(args) > 1 else kwargs.get("metadata", {})
        if isinstance(meta, dict):
            category = meta.get("category", "")
            assert category in ("teacher_progress", "teacher_sessions"), (
                f"Unexpected category in store() call: {category!r}. "
                "TeacherMode should only store under 'teacher_sessions'; "
                "progress should go through ProgressTracker under 'teacher_progress'."
            )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestTeacherModeIntegration:
    """Integration tests using mock MemoryInterface."""

    def test_full_teach_flow_produces_valid_session(self):
        """Full TeacherMode.teach() flow produces a valid TeachingSession."""
        mi, _ = _make_mock_memory()
        teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

        session = teacher.teach("user1", "Python")

        assert isinstance(session, TeachingSession)
        assert session.user_id == "user1"
        assert session.topic == "Python"
        assert session.status in {"active", "completed"}
        assert session.session_id  # non-empty
        assert session.created_at  # non-empty
        # UUID validation
        import uuid
        uuid.UUID(session.session_id)

    def test_completed_lessons_excluded_from_subsequent_sessions(self):
        """Completed lessons are excluded from subsequent sessions."""
        mi, _ = _make_mock_memory()
        teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

        # First session
        session1 = teacher.teach("user1", "Python")
        first_session_lesson_ids = {l.id for l in session1.lessons}

        # Second session — lessons from first session should be excluded
        session2 = teacher.teach("user1", "Python")
        second_session_lesson_ids = {l.id for l in session2.lessons}

        # No overlap between first and second session lessons
        overlap = first_session_lesson_ids & second_session_lesson_ids
        assert not overlap, (
            f"Lessons {overlap} appeared in both sessions"
        )

    def test_teaching_session_error_raised_on_component_failure(self):
        """TeachingSessionError is raised and logged when a sub-component raises."""
        mi, _ = _make_mock_memory()

        mock_logger = MagicMock()
        personalization_engine = MagicMock()
        personalization_engine.personalize.side_effect = RuntimeError("component failed")

        insight_engine = MagicMock()
        insight_engine.generate_insights.return_value = _MockInsightReport(insights=[])

        progress_tracker = ProgressTracker(mi)

        teacher = TeacherMode(
            memory_interface=mi,
            personalization_engine=personalization_engine,
            insight_engine=insight_engine,
            lesson_generator=LessonGenerator(),
            explanation_engine=ExplanationEngine(),
            exercise_generator=ExerciseGenerator(),
            progress_tracker=progress_tracker,
            logger=mock_logger,
        )

        with pytest.raises(TeachingSessionError):
            teacher.teach("user1", "Python")

        mock_logger.log.assert_called_once()

    def test_status_completed_when_no_incomplete_lessons_remain(self):
        """status="completed" is returned when no incomplete lessons remain."""
        mi, _ = _make_mock_memory()
        teacher, pe, ie, pt, mi = _make_teacher_mode(mi)

        # Complete all beginner lessons with score=0.0 to keep user_level at beginner
        lesson_gen = LessonGenerator()
        all_lessons = lesson_gen.generate(topic="Python", user_level="beginner")
        for lesson in all_lessons:
            pt.record_completion("user1", "Python", lesson.id, score=0.0)

        session = teacher.teach("user1", "Python")

        assert session.status == "completed"
        assert session.lessons == []
