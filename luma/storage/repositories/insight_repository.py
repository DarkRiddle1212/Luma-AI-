"""
luma/storage/repositories/insight_repository.py

Repository for insight persistence operations.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from luma.storage import InsightRecord, RepositoryError
from luma.storage.models import InsightModel


class InsightRepository:
    """Repository for CRUD operations on the ``insights`` table.

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

    def create(
        self,
        user_id: str,
        message: str,
        confidence: float,
        evidence: Optional[dict] = None,
    ) -> InsightRecord:
        """Persist a new insight and return its domain record."""
        try:
            model = InsightModel(
                user_id=user_id,
                message=message,
                confidence=confidence,
                evidence=evidence,
            )
            self._session.add(model)
            self._session.flush()  # populate auto-generated id + created_at
            self._session.refresh(model)
            return self._to_record(model)
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to create insight for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def get_by_id(self, insight_id: int) -> Optional[InsightRecord]:
        """Return the insight with the given id, or ``None`` if not found."""
        try:
            model = self._session.get(InsightModel, insight_id)
            return self._to_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve insight id={insight_id}: {exc}",
                cause=exc,
            ) from exc

    def get_by_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[InsightRecord]:
        """Return insights for *user_id*, newest first.

        Parameters
        ----------
        user_id:
            Filter to this user.
        limit:
            Maximum number of records to return.
        """
        try:
            stmt = (
                select(InsightModel)
                .where(InsightModel.user_id == user_id)
                .order_by(desc(InsightModel.created_at))
                .limit(limit)
            )
            rows = self._session.execute(stmt).scalars().all()
            return [self._to_record(row) for row in rows]
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve insights for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def delete(self, insight_id: int) -> bool:
        """Delete the insight with the given id.

        Returns
        -------
        bool
            ``True`` if a record was deleted, ``False`` if it did not exist.
        """
        try:
            model = self._session.get(InsightModel, insight_id)
            if model is None:
                return False
            self._session.delete(model)
            self._session.flush()
            return True
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to delete insight id={insight_id}: {exc}",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_record(self, model: InsightModel) -> InsightRecord:
        """Convert an ORM model instance to a domain dataclass."""
        return InsightRecord(
            id=model.id,
            user_id=model.user_id,
            message=model.message,
            confidence=model.confidence,
            evidence=model.evidence,
            created_at=model.created_at,
        )
