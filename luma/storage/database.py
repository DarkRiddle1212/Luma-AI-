"""
luma/storage/database.py

Database engine creation, session lifecycle, and connection pooling for the
Luma Persistence & Storage Layer.

Provides:
- ``Base``: SQLAlchemy declarative base that all ORM models inherit from.
- ``DatabaseManager``: Centralised engine + session management.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from luma.storage import StorageConfigurationError


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models in the storage layer."""


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """
    Centralises SQLAlchemy engine creation, session lifecycle, and connection
    pooling.

    Parameters
    ----------
    database_url:
        SQLAlchemy-compatible database URL (e.g. ``"sqlite:///:memory:"`` or
        ``"postgresql://user:pass@host/db"``).
    pool_size:
        Number of persistent connections to keep in the pool (non-SQLite only).
    max_overflow:
        Maximum extra connections above ``pool_size`` (non-SQLite only).
    pool_timeout:
        Seconds to wait for a connection from the pool before raising a
        ``TimeoutError`` (non-SQLite only).
    echo_sql:
        When ``True``, SQLAlchemy logs every SQL statement it executes.

    Raises
    ------
    StorageConfigurationError
        If the engine cannot be created due to an invalid ``database_url``.
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        echo_sql: bool = False,
    ) -> None:
        self._database_url = database_url

        try:
            if "sqlite" in database_url:
                connect_args = {"check_same_thread": False}
                if ":memory:" in database_url:
                    # In-memory SQLite: use StaticPool so all connections share
                    # the same in-memory database (required for tests).
                    self._engine = create_engine(
                        database_url,
                        connect_args=connect_args,
                        poolclass=StaticPool,
                        echo=echo_sql,
                    )
                else:
                    # File-based SQLite: use NullPool to avoid cross-thread
                    # connection reuse issues.
                    self._engine = create_engine(
                        database_url,
                        connect_args=connect_args,
                        poolclass=NullPool,
                        echo=echo_sql,
                    )
            else:
                # Non-SQLite (e.g. PostgreSQL): use the standard connection pool.
                self._engine = create_engine(
                    database_url,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    echo=echo_sql,
                )
        except ArgumentError as exc:
            raise StorageConfigurationError(
                f"Invalid database URL '{database_url}': {exc}",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise StorageConfigurationError(
                f"Failed to create database engine for URL '{database_url}': {exc}",
                cause=exc,
            ) from exc

        self._Session = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager that yields a ``Session``.

        - Commits the transaction on successful exit.
        - Rolls back the transaction if any exception is raised.
        - Always closes the session in the ``finally`` block.

        Yields
        ------
        Session
            A SQLAlchemy ORM session scoped to this unit of work.
        """
        session: Session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_all_tables(self) -> None:
        """Create all ORM-mapped tables that do not already exist."""
        Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Close all connections in the pool, enabling clean shutdown."""
        self._engine.dispose()
