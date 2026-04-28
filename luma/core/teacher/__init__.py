"""
luma.core.teacher — Teacher Mode subsystem

Public exports for the teacher module.
"""

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
    TeachingSessionError,
    ProgressStorageError,
    ProgressRetrievalError,
)

__all__ = [
    "TeacherMode",
    "LessonGenerator",
    "ExplanationEngine",
    "ExerciseGenerator",
    "ProgressTracker",
    "Lesson",
    "Explanation",
    "Exercise",
    "ProgressRecord",
    "TeachingSession",
    "TeachingSessionError",
    "ProgressStorageError",
    "ProgressRetrievalError",
]
