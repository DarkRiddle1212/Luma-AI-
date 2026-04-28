"""
ProgressTracker — persists and queries lesson completion records.

Stores progress via MemoryInterface using the "teacher_progress" category,
and provides helpers for weak-area detection and completion ratio calculation.
"""

import json
from datetime import datetime, UTC
from typing import List

from luma.core.memory_interface import MemoryInterface
from luma.core.teacher.schemas import (
    ProgressRecord,
    ProgressStorageError,
    ProgressRetrievalError,
)


class ProgressTracker:
    """Tracks per-user lesson completion and scores via MemoryInterface."""

    def __init__(self, memory_interface: MemoryInterface) -> None:
        self._memory = memory_interface

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_completion(
        self,
        user_id: str,
        topic: str,
        lesson_id: str,
        score: float,
    ) -> None:
        """Persist a lesson completion record (idempotent by lesson_id)."""
        existing = self.get_progress(user_id, topic)
        if any(r.lesson_id == lesson_id for r in existing):
            return  # already recorded — skip silently

        content = json.dumps(
            {
                "user_id": user_id,
                "topic": topic,
                "lesson_id": lesson_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "score": score,
            }
        )
        metadata = {
            "user_id": user_id,
            "topic": topic,
            "lesson_id": lesson_id,
            "category": "teacher_progress",
        }

        try:
            self._memory.store(content, metadata)
        except Exception as exc:
            raise ProgressStorageError(str(exc)) from exc

    def get_progress(self, user_id: str, topic: str) -> List[ProgressRecord]:
        """Return all completion records for *user_id* / *topic*."""
        try:
            result = self._memory.retrieve(
                params={"category": "teacher_progress", "limit": 1000}
            )
        except Exception as exc:
            raise ProgressRetrievalError(str(exc)) from exc

        records: List[ProgressRecord] = []
        for entry in result["memories"]:
            meta = entry.get("metadata") or {}
            if (
                meta.get("category") == "teacher_progress"
                and meta.get("user_id") == user_id
                and meta.get("topic") == topic
            ):
                data = json.loads(entry["content"])
                records.append(
                    ProgressRecord(
                        user_id=data["user_id"],
                        topic=data["topic"],
                        lesson_id=data["lesson_id"],
                        completed_at=data["completed_at"],
                        score=data["score"],
                    )
                )
        return records

    def get_weak_areas(self, user_id: str, topic: str) -> List[str]:
        """Return lesson IDs where the recorded score is below 0.6."""
        records = self.get_progress(user_id, topic)
        return [r.lesson_id for r in records if r.score < 0.6]

    def get_completion_ratio(
        self,
        user_id: str,
        topic: str,
        total_lessons: int,
    ) -> float:
        """Return completed / total_lessons, clamped to [0.0, 1.0]."""
        if total_lessons == 0:
            return 0.0
        records = self.get_progress(user_id, topic)
        ratio = len(records) / total_lessons
        return max(0.0, min(1.0, ratio))
