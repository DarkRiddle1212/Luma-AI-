"""
luma/storage/repositories/personalization_repository.py

Repository for user profile persistence operations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from luma.storage import RepositoryError, UserProfileRecord
from luma.storage.models import UserProfileModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PersonalizationRepository:
    """Repository for upsert/get/delete operations on the ``user_profiles`` table.

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

    def upsert(
        self,
        user_id: str,
        interests: Optional[list] = None,
        preferences: Optional[dict] = None,
        strengths: Optional[list] = None,
    ) -> UserProfileRecord:
        """Insert or update a user profile.

        Fields passed as ``None`` preserve the existing value in the database.
        For a brand-new record, ``None`` fields default to empty collections.

        Uses a dialect-aware ``INSERT ... ON CONFLICT DO UPDATE`` so the
        operation is atomic at the database level.
        """
        try:
            dialect = self._session.get_bind().dialect.name

            now = _utcnow()

            # Build the "insert" values — use defaults for a new row.
            insert_values = {
                "user_id": user_id,
                "interests": interests if interests is not None else [],
                "preferences": preferences if preferences is not None else {},
                "strengths": strengths if strengths is not None else [],
                "updated_at": now,
            }

            # Build the "update" set — only overwrite fields that were provided.
            update_set: dict = {"updated_at": now}
            if interests is not None:
                update_set["interests"] = interests
            if preferences is not None:
                update_set["preferences"] = preferences
            if strengths is not None:
                update_set["strengths"] = strengths

            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as postgresql_insert

                stmt = postgresql_insert(UserProfileModel).values(**insert_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id"],
                    set_=update_set,
                )
            else:
                # Default to SQLite dialect (also works for other dialects that
                # support the same syntax, e.g. MySQL via on_conflict_do_update).
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                stmt = sqlite_insert(UserProfileModel).values(**insert_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id"],
                    set_=update_set,
                )

            self._session.execute(stmt)
            self._session.flush()

            # Fetch the persisted record and return as a domain dataclass.
            model = self._session.get(UserProfileModel, user_id)
            # Refresh to pick up any server-side defaults / triggers.
            self._session.refresh(model)
            return self._to_record(model)
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to upsert profile for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def get_by_user(self, user_id: str) -> Optional[UserProfileRecord]:
        """Return the profile for *user_id*, or ``None`` if not found."""
        try:
            model = self._session.get(UserProfileModel, user_id)
            return self._to_record(model) if model is not None else None
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to retrieve profile for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    def delete(self, user_id: str) -> bool:
        """Delete the profile for *user_id*.

        Returns
        -------
        bool
            ``True`` if a record was deleted, ``False`` if it did not exist.
        """
        try:
            model = self._session.get(UserProfileModel, user_id)
            if model is None:
                return False
            self._session.delete(model)
            self._session.flush()
            return True
        except SQLAlchemyError as exc:
            raise RepositoryError(
                f"Failed to delete profile for user '{user_id}': {exc}",
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_record(self, model: UserProfileModel) -> UserProfileRecord:
        """Convert an ORM model instance to a domain dataclass."""
        return UserProfileRecord(
            user_id=model.user_id,
            interests=model.interests,
            preferences=model.preferences,
            strengths=model.strengths,
            updated_at=model.updated_at,
        )
