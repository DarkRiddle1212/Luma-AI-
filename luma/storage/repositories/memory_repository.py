"""
luma/storage/repositories/memory_repository.py

Repository for memory persistence operations.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from luma.storage import MemoryRecord, RepositoryError
from luma.storage.models import MemoryModel


class MemoryRepository:
    """Repository for CRUD operations on the ``memories`` table.

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
        namespace: Optional[str],
        content: str,
        importance_score: float,
        final_score: float,
    ) -> MemoryRecord:
        """Persist a new memory and return its domain record."""
        try:
            model = MemoryModel(
                user_id=user_id,
                namespace=namespace,
                content=content,
                importance_score=importance_score,
                final_score=final_score,
            )
            self._session.add(model)
            self._session.flush()  # populate auto-generated id + created_at
            self._session.refresh(model)
            return self._to_record(model)
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to create memory for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def get_by_id(self, memory_id: int) -> Optional[MemoryRecord]:
        """Return the memory with the given id, or ``None`` if not found."""
        try:
            model = self._session.get(MemoryModel, memory_id)
            return self._to_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve memory id={memory_id}: {exc}",
                cause=exc,
            ) from exc

    def get_by_user(
        self,
        user_id: str,
        namespace: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """Return memories for *user_id*, newest first.

        Parameters
        ----------
        user_id:
            Filter to this user.
        namespace:
            When provided, further filter to this namespace only.
        limit:
            Maximum number of records to return.
        """
        try:
            stmt = (
                select(MemoryModel)
                .where(MemoryModel.user_id == user_id)
                .order_by(desc(MemoryModel.created_at))
                .limit(limit)
            )
            if namespace is not None:
                stmt = stmt.where(MemoryModel.namespace == namespace)
            rows = self._session.execute(stmt).scalars().all()
            return [self._to_record(row) for row in rows]
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve memories for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def update(
        self,
        memory_id: int,
        importance_score: Optional[float] = None,
        final_score: Optional[float] = None,
    ) -> Optional[MemoryRecord]:
        """Update scores on an existing memory.

        Returns the updated ``MemoryRecord``, or ``None`` if *memory_id* does
        not exist.  Fields passed as ``None`` are left unchanged.
        """
        try:
            model = self._session.get(MemoryModel, memory_id)
            if model is None:
                return None
            if importance_score is not None:
                model.importance_score = importance_score
            if final_score is not None:
                model.final_score = final_score
            self._session.flush()
            self._session.refresh(model)
            return self._to_record(model)
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to update memory id={memory_id}: {exc}",
                cause=exc,
            ) from exc

    def delete(self, memory_id: int) -> bool:
        """Delete the memory with the given id.

        Returns
        -------
        bool
            ``True`` if a record was deleted, ``False`` if it did not exist.
        """
        try:
            model = self._session.get(MemoryModel, memory_id)
            if model is None:
                return False
            self._session.delete(model)
            self._session.flush()
            return True
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to delete memory id={memory_id}: {exc}",
                cause=exc,
            ) from exc

    def count_by_user(self, user_id: str) -> int:
        """Return the total number of memories stored for *user_id*."""
        try:
            stmt = select(func.count()).select_from(MemoryModel).where(
                MemoryModel.user_id == user_id
            )
            return self._session.execute(stmt).scalar_one()
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to count memories for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_record(self, model: MemoryModel) -> MemoryRecord:
        """Convert an ORM model instance to a domain dataclass."""
        return MemoryRecord(
            id=model.id,
            user_id=model.user_id,
            namespace=model.namespace,
            content=model.content,
            importance_score=model.importance_score,
            final_score=model.final_score,
            created_at=model.created_at,
        )
