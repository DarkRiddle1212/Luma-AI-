"""
tests/storage/test_storage_config.py

Unit tests for luma/storage/config.py (StorageConfig).

Covers:
- Default field values (Requirements 4.1–4.6)
- LUMA_STORAGE_ env-var prefix loading (Requirement 4.8)
- Production + SQLite raises ValueError (Requirement 4.7)
- Valid environment values
- Non-SQLite production URL is accepted
"""

from __future__ import annotations

import pytest

from luma.storage.config import StorageConfig


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestStorageConfigDefaults:
    """StorageConfig should expose documented defaults when no env vars are set."""

    def test_default_database_url(self):
        cfg = StorageConfig()
        assert cfg.database_url == "sqlite:///./luma.db"

    def test_default_environment(self):
        cfg = StorageConfig()
        assert cfg.environment == "development"

    def test_default_pool_size(self):
        cfg = StorageConfig()
        assert cfg.pool_size == 5

    def test_default_max_overflow(self):
        cfg = StorageConfig()
        assert cfg.max_overflow == 10

    def test_default_pool_timeout(self):
        cfg = StorageConfig()
        assert cfg.pool_timeout == 30.0

    def test_default_echo_sql(self):
        cfg = StorageConfig()
        assert cfg.echo_sql is False


# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------


class TestStorageConfigEnvPrefix:
    """Settings should be overridable via LUMA_STORAGE_* environment variables."""

    def test_database_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_DATABASE_URL", "sqlite:///./test.db")
        cfg = StorageConfig()
        assert cfg.database_url == "sqlite:///./test.db"

    def test_environment_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_ENVIRONMENT", "staging")
        cfg = StorageConfig()
        assert cfg.environment == "staging"

    def test_pool_size_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_POOL_SIZE", "20")
        cfg = StorageConfig()
        assert cfg.pool_size == 20

    def test_max_overflow_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_MAX_OVERFLOW", "50")
        cfg = StorageConfig()
        assert cfg.max_overflow == 50

    def test_pool_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_POOL_TIMEOUT", "60.0")
        cfg = StorageConfig()
        assert cfg.pool_timeout == 60.0

    def test_echo_sql_from_env(self, monkeypatch):
        monkeypatch.setenv("LUMA_STORAGE_ECHO_SQL", "true")
        cfg = StorageConfig()
        assert cfg.echo_sql is True

    def test_unprefixed_env_var_is_ignored(self, monkeypatch):
        """Variables without the LUMA_STORAGE_ prefix must not affect StorageConfig."""
        monkeypatch.setenv("POOL_SIZE", "999")
        cfg = StorageConfig()
        assert cfg.pool_size == 5  # default unchanged


# ---------------------------------------------------------------------------
# Production + SQLite validation
# ---------------------------------------------------------------------------


class TestProductionSQLiteValidation:
    """StorageConfig must reject SQLite when environment == 'production'."""

    def test_production_sqlite_raises_value_error(self):
        with pytest.raises(ValueError, match="SQLite is not recommended for production"):
            StorageConfig(
                environment="production",
                database_url="sqlite:///./luma.db",
            )

    def test_production_sqlite_memory_raises_value_error(self):
        with pytest.raises(ValueError, match="SQLite is not recommended for production"):
            StorageConfig(
                environment="production",
                database_url="sqlite:///:memory:",
            )

    def test_production_postgres_is_valid(self):
        """A PostgreSQL URL in production should not raise."""
        cfg = StorageConfig(
            environment="production",
            database_url="postgresql://user:pass@localhost/luma",
        )
        assert cfg.environment == "production"
        assert cfg.database_url.startswith("postgresql")

    def test_staging_sqlite_is_valid(self):
        """SQLite in staging should not raise."""
        cfg = StorageConfig(
            environment="staging",
            database_url="sqlite:///./luma.db",
        )
        assert cfg.environment == "staging"

    def test_development_sqlite_is_valid(self):
        """SQLite in development (the default) should not raise."""
        cfg = StorageConfig(
            environment="development",
            database_url="sqlite:///./luma.db",
        )
        assert cfg.environment == "development"


# ---------------------------------------------------------------------------
# Valid environment values
# ---------------------------------------------------------------------------


class TestValidEnvironmentValues:
    """Only 'development', 'staging', and 'production' are accepted."""

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_valid_environment(self, env):
        if env == "production":
            cfg = StorageConfig(
                environment=env,
                database_url="postgresql://user:pass@localhost/luma",
            )
        else:
            cfg = StorageConfig(environment=env)
        assert cfg.environment == env

    def test_invalid_environment_raises(self):
        with pytest.raises(Exception):
            StorageConfig(environment="test")  # type: ignore[arg-type]
