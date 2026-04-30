"""
luma/storage/config.py

Storage configuration for the Luma Persistence & Storage Layer.

All settings are loaded from environment variables with the ``LUMA_STORAGE_``
prefix, following the same Pydantic Settings pattern used in ``luma/config.py``.

Example environment variables::

    LUMA_STORAGE_DATABASE_URL=postgresql://user:pass@localhost/luma
    LUMA_STORAGE_ENVIRONMENT=production
    LUMA_STORAGE_POOL_SIZE=10
    LUMA_STORAGE_MAX_OVERFLOW=20
    LUMA_STORAGE_POOL_TIMEOUT=60.0
    LUMA_STORAGE_ECHO_SQL=false
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    """
    Storage layer configuration.

    Loaded from environment variables with the ``LUMA_STORAGE_`` prefix.
    All fields have sensible defaults for local development with SQLite.

    Fields
    ------
    database_url:
        SQLAlchemy-compatible database URL.
        Default: ``"sqlite:///./luma.db"`` (file-based SQLite in the current
        working directory).

    environment:
        Deployment environment. Accepted values: ``"development"``,
        ``"staging"``, ``"production"``.
        Default: ``"development"``.

    pool_size:
        Number of persistent connections to keep in the connection pool.
        Ignored for SQLite (which uses ``StaticPool`` or ``NullPool``).
        Default: ``5``.

    max_overflow:
        Maximum number of connections that can be created above ``pool_size``
        when the pool is exhausted.
        Ignored for SQLite.
        Default: ``10``.

    pool_timeout:
        Seconds to wait for a connection from the pool before raising a
        ``TimeoutError``.
        Default: ``30.0``.

    echo_sql:
        When ``True``, SQLAlchemy logs every SQL statement it executes.
        Useful for debugging; should be ``False`` in production.
        Default: ``False``.
    """

    # ------------------------------------------------------------------ #
    # Fields                                                               #
    # ------------------------------------------------------------------ #

    database_url: str = "sqlite:///./luma.db"
    """SQLAlchemy database URL. Defaults to a local SQLite file."""

    environment: Literal["development", "staging", "production"] = "development"
    """Deployment environment. Controls validation rules (e.g. no SQLite in production)."""

    pool_size: int = 5
    """Number of persistent connections in the connection pool (non-SQLite only)."""

    max_overflow: int = 10
    """Maximum extra connections above ``pool_size`` (non-SQLite only)."""

    pool_timeout: float = 30.0
    """Seconds to wait for a pool connection before raising ``TimeoutError``."""

    echo_sql: bool = False
    """Log all SQL statements when ``True``. Disable in production."""

    # ------------------------------------------------------------------ #
    # Pydantic Settings configuration                                      #
    # ------------------------------------------------------------------ #

    model_config = SettingsConfigDict(
        env_prefix="LUMA_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def validate_production_sqlite(self) -> "StorageConfig":
        """
        Reject SQLite as the database backend when running in production.

        SQLite is not suitable for production workloads due to its limited
        concurrency model. Use PostgreSQL (or another server-grade database)
        for production deployments.

        Raises
        ------
        ValueError
            When ``environment`` is ``"production"`` and ``database_url``
            starts with ``"sqlite"``.
        """
        if self.environment == "production" and self.database_url.startswith("sqlite"):
            raise ValueError(
                "SQLite is not recommended for production use. "
                "Set LUMA_STORAGE_DATABASE_URL to a PostgreSQL (or other "
                "server-grade) database URL when LUMA_STORAGE_ENVIRONMENT=production."
            )
        return self
