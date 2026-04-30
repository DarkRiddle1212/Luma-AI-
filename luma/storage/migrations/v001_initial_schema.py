"""
luma/storage/migrations/v001_initial_schema.py

Initial schema migration (version 1).

Creates all four domain tables and the ``schema_version`` table using
SQLAlchemy DDL constructs (``CREATE TABLE IF NOT EXISTS``) so that the
migration is idempotent and dialect-independent.

Requirements: 9.3, 9.4, 17.4
"""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.orm import Session

# Import the ORM models so that their table definitions are registered on
# ``Base.metadata`` before we call ``create_all``.
from luma.storage.models import (  # noqa: F401 – side-effect import
    InsightModel,
    LearningProgressModel,
    MemoryModel,
    UserProfileModel,
)
from luma.storage.database import Base


def upgrade(session: Session) -> None:
    """
    Apply the initial schema migration.

    Creates the following tables if they do not already exist:

    - ``schema_version``  – single-row version tracker
    - ``memories``        – memory entries
    - ``insights``        – insight entries
    - ``user_profiles``   – user profile data
    - ``learning_progress`` – per-user topic progress

    Uses ``Base.metadata.create_all(checkfirst=True)`` for the four ORM-mapped
    tables (dialect-independent DDL, equivalent to ``CREATE TABLE IF NOT
    EXISTS``) and a standalone ``MetaData`` table definition for
    ``schema_version``, which is not an ORM model.

    Parameters
    ----------
    session:
        The active SQLAlchemy ``Session`` provided by ``MigrationRunner``.
        The caller is responsible for committing or rolling back the
        surrounding transaction.
    """
    bind = session.get_bind()

    # ------------------------------------------------------------------
    # 1. Create the four ORM-mapped domain tables (IF NOT EXISTS).
    # ------------------------------------------------------------------
    # ``checkfirst=True`` emits ``CREATE TABLE IF NOT EXISTS`` on SQLite and
    # an equivalent conditional DDL on PostgreSQL, satisfying requirement 17.4.
    Base.metadata.create_all(bind=bind, checkfirst=True)

    # ------------------------------------------------------------------
    # 2. Create the ``schema_version`` table (IF NOT EXISTS).
    # ------------------------------------------------------------------
    # ``schema_version`` is not an ORM model (it is managed exclusively by
    # ``MigrationRunner``), so we define it inline with a standalone
    # ``MetaData`` instance and use ``checkfirst=True`` for idempotency.
    _schema_version_meta = MetaData()
    _schema_version_table = Table(
        "schema_version",
        _schema_version_meta,
        Column("version", Integer, nullable=False),
    )
    _schema_version_meta.create_all(bind=bind, checkfirst=True)
