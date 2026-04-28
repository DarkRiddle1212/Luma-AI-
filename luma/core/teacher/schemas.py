"""
Teacher Mode Data Schemas.

Defines Lesson, Explanation, Exercise, ProgressRecord, and TeachingSession models,
plus TeachingSessionError, ProgressStorageError, and ProgressRetrievalError exceptions.
Uses Pydantic if available, otherwise dataclasses with __post_init__ validation.
Follows the dual-path pattern in luma/core/personalization/schemas.py.
"""

from typing import List

try:
    from pydantic import BaseModel, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


# ---------------------------------------------------------------------------
# Exception classes (plain Python — not models)
# ---------------------------------------------------------------------------

class TeachingSessionError(Exception):
    """Raised by TeacherMode when orchestration fails."""


class ProgressStorageError(Exception):
    """Raised by ProgressTracker when a storage operation fails."""


class ProgressRetrievalError(Exception):
    """Raised by ProgressTracker when a retrieval operation fails."""


# ---------------------------------------------------------------------------
# Data models — dual-path: Pydantic or dataclasses
# ---------------------------------------------------------------------------

_DIFFICULTY_VALUES = {"beginner", "intermediate", "advanced"}
_STATUS_VALUES = {"active", "completed", "paused"}
_EXERCISE_TYPE_VALUES = {"conceptual", "practical", "mini-project"}

if _USE_PYDANTIC:
    class Lesson(BaseModel):
        """A structured unit of learning content covering a single concept."""

        id: str
        topic: str
        title: str
        difficulty: str
        content: str
        order: int

        @field_validator("difficulty")
        @classmethod
        def difficulty_valid(cls, v: str) -> str:
            if v not in _DIFFICULTY_VALUES:
                raise ValueError(
                    f"difficulty must be one of {_DIFFICULTY_VALUES}, got {v!r}"
                )
            return v

    class Explanation(BaseModel):
        """A user-adapted textual breakdown of a lesson's concept."""

        lesson_id: str
        content: str
        rationale: str

    class Exercise(BaseModel):
        """A task or question assigned to reinforce lesson content."""

        id: str
        lesson_id: str
        type: str
        difficulty: str
        prompt: str
        explanation: str

        @field_validator("type")
        @classmethod
        def type_valid(cls, v: str) -> str:
            if v not in _EXERCISE_TYPE_VALUES:
                raise ValueError(
                    f"type must be one of {_EXERCISE_TYPE_VALUES}, got {v!r}"
                )
            return v

        @field_validator("difficulty")
        @classmethod
        def difficulty_valid(cls, v: str) -> str:
            if v not in _DIFFICULTY_VALUES:
                raise ValueError(
                    f"difficulty must be one of {_DIFFICULTY_VALUES}, got {v!r}"
                )
            return v

    class ProgressRecord(BaseModel):
        """A persisted record of a user's completed lesson and exercise score."""

        user_id: str
        topic: str
        lesson_id: str
        completed_at: str
        score: float

        @field_validator("score")
        @classmethod
        def score_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"score must be in [0.0, 1.0], got {v}")
            return v

    class TeachingSession(BaseModel):
        """A single end-to-end interaction in which a user studies a topic."""

        session_id: str
        user_id: str
        topic: str
        status: str
        lessons: List[Lesson]
        explanations: List[Explanation]
        exercises: List[Exercise]
        created_at: str

        @field_validator("status")
        @classmethod
        def status_valid(cls, v: str) -> str:
            if v not in _STATUS_VALUES:
                raise ValueError(
                    f"status must be one of {_STATUS_VALUES}, got {v!r}"
                )
            return v

else:
    from dataclasses import dataclass, field

    @dataclass
    class Lesson:
        """A structured unit of learning content covering a single concept."""

        id: str
        topic: str
        title: str
        difficulty: str
        content: str
        order: int

        def __post_init__(self) -> None:
            if self.difficulty not in _DIFFICULTY_VALUES:
                raise ValueError(
                    f"difficulty must be one of {_DIFFICULTY_VALUES}, got {self.difficulty!r}"
                )

    @dataclass
    class Explanation:
        """A user-adapted textual breakdown of a lesson's concept."""

        lesson_id: str
        content: str
        rationale: str

    @dataclass
    class Exercise:
        """A task or question assigned to reinforce lesson content."""

        id: str
        lesson_id: str
        type: str
        difficulty: str
        prompt: str
        explanation: str

        def __post_init__(self) -> None:
            if self.type not in _EXERCISE_TYPE_VALUES:
                raise ValueError(
                    f"type must be one of {_EXERCISE_TYPE_VALUES}, got {self.type!r}"
                )
            if self.difficulty not in _DIFFICULTY_VALUES:
                raise ValueError(
                    f"difficulty must be one of {_DIFFICULTY_VALUES}, got {self.difficulty!r}"
                )

    @dataclass
    class ProgressRecord:
        """A persisted record of a user's completed lesson and exercise score."""

        user_id: str
        topic: str
        lesson_id: str
        completed_at: str
        score: float

        def __post_init__(self) -> None:
            if not 0.0 <= self.score <= 1.0:
                raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")

    @dataclass
    class TeachingSession:
        """A single end-to-end interaction in which a user studies a topic."""

        session_id: str
        user_id: str
        topic: str
        status: str
        lessons: List[Lesson]
        explanations: List[Explanation]
        exercises: List[Exercise]
        created_at: str

        def __post_init__(self) -> None:
            if self.status not in _STATUS_VALUES:
                raise ValueError(
                    f"status must be one of {_STATUS_VALUES}, got {self.status!r}"
                )
