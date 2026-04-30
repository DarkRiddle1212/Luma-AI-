"""
tests/storage/test_migrations.py

Unit tests for the MigrationRunner and initial migration.

Requirements: 9.1–9.8
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from luma.storage import MigrationError
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner, _MIGRATIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "memories",
    "insights",
    "user_profiles",
    "learning_progress",
    "schema_version",
}


def make_db() -> DatabaseManager:
    """Return a fresh in-memory SQLite DatabaseManager for each test."""
    return DatabaseManager("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Test 1: Initial migration creates all tables (Req 9.3, 9.4)
# ---------------------------------------------------------------------------


def test_run_pending_creates_all_tables():
    """After run_pending() on a fresh DB, all 5 expected tables exist."""
    db = make_db()
    runner = MigrationRunner(db)
    runner.run_pending()

    inspector = inspect(db._engine)
    actual_tables = set(inspector.get_table_names())
    assert EXPECTED_TABLES.issubset(actual_tables), (
        f"Missing tables: {EXPECTED_TABLES - actual_tables}"
    )


# ---------------------------------------------------------------------------
# Test 2: schema_version is updated after migration (Req 9.2, 9.5)
# ---------------------------------------------------------------------------


def test_run_pending_updates_schema_version():
    """After run_pending(), get_current_version() returns the latest migration version."""
    db = make_db()
    runner = MigrationRunner(db)

    # Before any migration, version should be 0
    assert runner.get_current_version() == 0

    runner.run_pending()

    # Latest registered migration version
    latest_version = max(v for v, _ in _MIGRATIONS)
    assert runner.get_current_version() == latest_version


# ---------------------------------------------------------------------------
# Test 3: Migrations are not re-applied (Req 9.6, 9.8)
# ---------------------------------------------------------------------------


def test_run_pending_idempotent():
    """Calling run_pending() twice does NOT re-apply migrations; version stays stable."""
    db = make_db()
    runner = MigrationRunner(db)

    runner.run_pending()
    version_after_first = runner.get_current_version()

    # Second call should be a no-op
    runner.run_pending()
    version_after_second = runner.get_current_version()

    assert version_after_first == version_after_second
    assert version_after_second == max(v for v, _ in _MIGRATIONS)


# ---------------------------------------------------------------------------
# Test 4: Failed migration raises MigrationError with correct version (Req 9.7)
# ---------------------------------------------------------------------------


def test_failed_migration_raises_migration_error(monkeypatch):
    """Injecting a bad upgrade function causes MigrationError with the correct version."""
    db = make_db()

    def bad_upgrade(session):
        raise RuntimeError("intentional failure")

    # Patch the migration registry to include a bad migration at version 2
    patched = list(_MIGRATIONS) + [(2, bad_upgrade)]
    monkeypatch.setattr(
        "luma.storage.migrations._MIGRATIONS",
        patched,
    )

    runner = MigrationRunner(db)

    # Apply v1 first so only v2 (the bad one) is pending
    # We do this by running with only the good migration first
    good_only = [(v, fn) for v, fn in patched if v == 1]
    monkeypatch.setattr("luma.storage.migrations._MIGRATIONS", good_only)
    runner.run_pending()
    assert runner.get_current_version() == 1

    # Now restore the patched list with the bad migration
    monkeypatch.setattr("luma.storage.migrations._MIGRATIONS", patched)

    with pytest.raises(MigrationError) as exc_info:
        runner.run_pending()

    assert exc_info.value.version == 2


# ---------------------------------------------------------------------------
# Test 5: Failed migration does NOT update schema_version (rollback) (Req 9.7)
# ---------------------------------------------------------------------------


def test_failed_migration_does_not_update_schema_version(monkeypatch):
    """After a failed migration, schema_version is NOT updated (rollback worked)."""
    db = make_db()

    def bad_upgrade(session):
        raise RuntimeError("intentional failure")

    patched = list(_MIGRATIONS) + [(2, bad_upgrade)]

    # Apply v1 first
    good_only = [(v, fn) for v, fn in patched if v == 1]
    monkeypatch.setattr("luma.storage.migrations._MIGRATIONS", good_only)
    runner = MigrationRunner(db)
    runner.run_pending()
    assert runner.get_current_version() == 1

    # Now attempt v2 (bad)
    monkeypatch.setattr("luma.storage.migrations._MIGRATIONS", patched)
    with pytest.raises(MigrationError):
        runner.run_pending()

    # Version must still be 1 — the failed migration must not have been recorded
    assert runner.get_current_version() == 1


# ---------------------------------------------------------------------------
# Test 6: get_current_version returns 0 on a fresh DB (Req 9.1)
# ---------------------------------------------------------------------------


def test_get_current_version_returns_zero_on_fresh_db():
    """Before any migration, get_current_version() returns 0."""
    db = make_db()
    runner = MigrationRunner(db)
    assert runner.get_current_version() == 0
