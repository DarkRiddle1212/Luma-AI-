"""
luma/storage/repositories/teacher_repository.py

Repository for learning progress persistence operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from luma.storage import LearningProgressRecord, RepositoryError
from luma.storage.models import LearningProgressModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TeacherRepository:
    """Repository for upsert/get/delete operations on the ``learning_progress`` table.

    Parameters
    ----------
    session:
        A SQLAlchemy ``Session`` managed externally (e.g. via
        ``DatabaseManager.get_session()``).  This repository never calls
        ``session.commit()``; transaction control belongs to the caller.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_progress(
        self,
        user_id: str,
        topic: str,
        progress: float,
        weak_areas: Optional[list] = None,
    ) -> LearningProgressRecord:
        """Insert or update a learning progress record.

        ``weak_areas=None`` defaults to ``[]`` for new records and preserves
        the existing value on updates.  Uses a dialect-aware
        ``INSERT ... ON CONFLICT DO UPDATE`` keyed on ``(user_id, topic)``.
        """
        try:
            dialect = self._session.get_bind().dialect.name

            now = _utcnow()

            insert_values = {
                "user_id": user_id,
                "topic": topic,
                "progress": progress,
                "weak_areas": weak_areas if weak_areas is not None else [],
                "last_updated": now,
            }

            # On update, always overwrite progress and last_updated;
            # only overwrite weak_areas when explicitly provided.
            update_set: dict = {"progress": progress, "last_updated": now}
            if weak_areas is not None:
                update_set["weak_areas"] = weak_areas

            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as postgresql_insert

                stmt = postgresql_insert(LearningProgressModel).values(**insert_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id", "topic"],
                    set_=update_set,
                )
            else:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                stmt = sqlite_insert(LearningProgressModel).values(**insert_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id", "topic"],
                    set_=update_set,
                )

            self._session.execute(stmt)
            self._session.flush()

            # Fetch the persisted record and return as a domain dataclass.
            model = (
                self._session.query(LearningProgressModel)
                .filter_by(user_id=user_id, topic=topic)
                .one()
            )
            self._session.refresh(model)
            return self._to_record(model)
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to upsert progress for user '{user_id}', topic '{topic}': {exc}",
                cause=exc,
            ) from exc

    def get_progress(
        self, user_id: str, topic: str
    ) -> Optional[LearningProgressRecord]:
        """Return the progress record for ``(user_id, topic)``, or ``None``."""
        try:
            model = (
                self._session.query(LearningProgressModel)
                .filter_by(user_id=user_id, topic=topic)
                .one_or_none()
            )
            return self._to_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve progress for user '{user_id}', topic '{topic}': {exc}",
                cause=exc,
            ) from exc

    def get_all_progress(self, user_id: str) -> List[LearningProgressRecord]:
        """Return all progress records for *user_id*."""
        try:
            models = (
                self._session.query(LearningProgressModel)
                .filter_by(user_id=user_id)
                .all()
            )
            return [self._to_record(m) for m in models]
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve all progress for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def delete_progress(self, user_id: str, topic: str) -> bool:
        """Delete the progress record for ``(user_id, topic)``.

        Returns
        -------
        bool
            ``True`` if a record was deleted, ``False`` if it did not exist.
        """
        try:
            model = (
                self._session.query(LearningProgressModel)
                .filter_by(user_id=user_id, topic=topic)
                .one_or_none()
            )
            if model is None:
                return False
            self._session.delete(model)
            self._session.flush()
            return True
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to delete progress for user '{user_id}', topic '{topic}': {exc}",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_record(self, model: LearningProgressModel) -> LearningProgressRecord:
        """Convert an ORM model instance to a domain dataclass."""
        return LearningProgressRecord(
            id=model.id,
            user_id=model.user_id,
            topic=model.topic,
            progress=model.progress,
            weak_areas=model.weak_areas,
            last_updated=model.last_updated,
        )
