"""
luma/storage/migrations/__init__.py

MigrationRunner: versioned schema migration management.

Maintains a ``schema_version`` table with a single row containing the current
integer version. Applies registered migrations in ascending order, each inside
its own transaction. Raises ``MigrationError`` on failure and never re-applies
an already-recorded version.

Requirements: 9.1, 9.2, 9.5, 9.6, 9.7, 9.8, 16.3
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from luma.storage import MigrationError

# Import the initial migration upgrade function
from luma.storage.migrations.v001_initial_schema import upgrade as v001_upgrade

logger = logging.getLogger(__name__)

# Registry of all migrations: list of (version, upgrade_fn) in ascending order.
# New migrations are appended here.
_MIGRATIONS: List[Tuple[int, Callable[[Session], None]]] = [
    (1, v001_upgrade),
]


class MigrationRunner:
    """
    Applies pending schema migrations in ascending version order.

    Parameters
    ----------
    database_manager:
        A ``DatabaseManager`` instance used to obtain sessions for reading
        the current version and applying each migration.
    """

    def __init__(self, database_manager) -> None:
        self._db = database_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_version(self) -> int:
        """
        Return the current schema version.

        Returns 0 if the ``schema_version`` table does not exist yet.
        """
        with self._db.get_session() as session:
            return self._read_version(session)

    def run_pending(self) -> None:
        """
        Apply all registered migrations whose version is greater than the
        current ``schema_version``, in ascending order.

        Each migration runs inside its own transaction (provided by
        ``DatabaseManager.get_session()``). On success the ``schema_version``
        row is updated and the applied version is logged. On failure the
        transaction is rolled back and a ``MigrationError`` is raised; no
        further migrations are attempted.
        """
        current = self.get_current_version()
        pending = [(v, fn) for v, fn in _MIGRATIONS if v > current]

        for version, upgrade_fn in sorted(pending, key=lambda t: t[0]):
            logger.info("Applying migration v%03d (current version: %d)", version, current)
            try:
                with self._db.get_session() as session:
                    upgrade_fn(session)
                    self._write_version(session, version)
                logger.info("Migration v%03d applied successfully", version)
                current = version
            except Exception as exc:
                raise MigrationError(
                    version=version,
                    message=str(exc),
                    cause=exc,
                ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_version(self, session: Session) -> int:
        """Read the current version from ``schema_version``, or 0 if absent."""
        try:
            result = session.execute(
                text("SELECT version FROM schema_version LIMIT 1")
            ).fetchone()
            return result[0] if result is not None else 0
        except OperationalError:
            # Table does not exist yet — treat as version 0.
            return 0

    def _write_version(self, session: Session, version: int) -> None:
        """
        Upsert the version row in ``schema_version``.

        Uses a DELETE + INSERT pattern so it works on both SQLite and
        PostgreSQL without dialect-specific syntax.
        """
        session.execute(text("DELETE FROM schema_version"))
        session.execute(
            text("INSERT INTO schema_version (version) VALUES (:v)"),
            {"v": version},
        )
