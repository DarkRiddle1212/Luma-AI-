"""
TeacherMode: top-level orchestrator for the Teacher Mode subsystem.

Coordinates PersonalizationEngine, InsightEngine, LessonGenerator,
ExplanationEngine, ExerciseGenerator, ProgressTracker, and MemoryInterface
to produce a single TeachingSession per call to teach().
"""

import uuid
from datetime import datetime, UTC
from typing import List, Optional

from luma.core.memory_interface import MemoryInterface
from luma.core.structured_logger import StructuredLogger
from luma.core.teacher.schemas import TeachingSession, TeachingSessionError
from luma.core.teacher.lesson_generator import LessonGenerator
from luma.core.teacher.explanation_engine import ExplanationEngine
from luma.core.teacher.exercise_generator import ExerciseGenerator
from luma.core.teacher.progress_tracker import ProgressTracker


class TeacherMode:
    """
    Orchestrates a full teaching session for a given user and topic.

    Parameters
    ----------
    memory_interface : MemoryInterface
        Used to persist the TeachingSession summary.
    personalization_engine :
        Provides AdaptationContext via .personalize(user_id, topic).
    insight_engine :
        Provides InsightReport via .generate_insights(namespace=topic).
    lesson_generator : LessonGenerator
        Generates the ordered lesson sequence.
    explanation_engine : ExplanationEngine
        Produces adapted explanations per lesson.
    exercise_generator : ExerciseGenerator
        Produces exercises per lesson.
    progress_tracker : ProgressTracker
        Retrieves and records lesson completion.
    logger : Optional[StructuredLogger]
        When provided, errors are logged before re-raising.
    """

    def __init__(
        self,
        memory_interface: MemoryInterface,
        personalization_engine,
        insight_engine,
        lesson_generator: LessonGenerator,
        explanation_engine: ExplanationEngine,
        exercise_generator: ExerciseGenerator,
        progress_tracker: ProgressTracker,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        self._memory = memory_interface
        self._personalization_engine = personalization_engine
        self._insight_engine = insight_engine
        self._lesson_generator = lesson_generator
        self._explanation_engine = explanation_engine
        self._exercise_generator = exercise_generator
        self._progress_tracker = progress_tracker
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def teach(self, user_id: str, topic: str) -> TeachingSession:
        """
        Run a full teaching session for *user_id* on *topic*.

        Parameters
        ----------
        user_id : str
            Identifier of the learner.
        topic : str
            Subject to teach.

        Returns
        -------
        TeachingSession
            Populated session object.  status is ``"completed"`` when all
            lessons were already finished, otherwise ``"active"``.

        Raises
        ------
        TeachingSessionError
            If any sub-component raises an exception.
        """
        try:
            return self._run(user_id, topic)
        except TeachingSessionError:
            raise
        except Exception as exc:
            if self._logger is not None:
                self._logger.log(
                    "teacher_mode_error",
                    {"error": str(exc), "user_id": user_id, "topic": topic},
                )
            raise TeachingSessionError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _run(self, user_id: str, topic: str) -> TeachingSession:
        session_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()

        # Step 1 — Personalization
        personalization_result = self._personalization_engine.personalize(user_id, topic)
        adaptation_ctx = personalization_result.adaptation

        # Step 2 — Insights
        insight_report = self._insight_engine.generate_insights(namespace=topic)
        insight_texts: List[str] = [insight.text for insight in insight_report.insights]

        # Step 3 — Progress history
        progress_records = self._progress_tracker.get_progress(user_id, topic)
        completed_ids = {record.lesson_id for record in progress_records}

        # Step 4 — Infer user level from progress history
        if not progress_records:
            user_level = "beginner"
        else:
            avg_score = sum(r.score for r in progress_records) / len(progress_records)
            if avg_score >= 0.8:
                user_level = "advanced"
            elif avg_score >= 0.5:
                user_level = "intermediate"
            else:
                user_level = "beginner"

        # Step 5 — Generate full lesson sequence
        all_lessons = self._lesson_generator.generate(topic, user_level)

        # Step 6 — Filter completed lessons, then prioritize weak-area lessons
        remaining = [l for l in all_lessons if l.id not in completed_ids]

        if insight_texts:
            # Lessons whose title or content contains any insight keyword go first
            priority: List = []
            normal: List = []
            for lesson in remaining:
                if any(
                    kw.lower() in lesson.title.lower() or kw.lower() in lesson.content.lower()
                    for kw in insight_texts
                ):
                    priority.append(lesson)
                else:
                    normal.append(lesson)
            remaining = priority + normal

        # Step 7 — Early return when nothing left to teach
        if not remaining:
            return TeachingSession(
                session_id=session_id,
                user_id=user_id,
                topic=topic,
                status="completed",
                lessons=[],
                explanations=[],
                exercises=[],
                created_at=created_at,
            )

        # Steps 8 & 9 — Explain, generate exercises, record completion
        lessons_processed = []
        explanations = []
        all_exercises = []

        for lesson in remaining:
            explanation = self._explanation_engine.explain(lesson, adaptation_ctx)
            exercises = self._exercise_generator.generate(lesson, user_level)

            lessons_processed.append(lesson)
            explanations.append(explanation)
            all_exercises.extend(exercises)

            self._progress_tracker.record_completion(
                user_id, topic, lesson.id, score=0.0
            )

        # Step 10 — Persist session summary
        content = (
            f"TeachingSession for user={user_id}, topic={topic}, "
            f"lessons={len(lessons_processed)}"
        )
        metadata = {
            "user_id": user_id,
            "topic": topic,
            "category": "teacher_sessions",
            "session_id": session_id,
        }
        self._memory.store(content, metadata)

        # Step 11 — Return populated session
        return TeachingSession(
            session_id=session_id,
            user_id=user_id,
            topic=topic,
            status="active",
            lessons=lessons_processed,
            explanations=explanations,
            exercises=all_exercises,
            created_at=created_at,
        )
