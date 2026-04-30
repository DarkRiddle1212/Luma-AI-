"""
tests/storage/test_teacher_repository.py

Unit tests for TeacherRepository.

Uses an in-memory SQLite database with MigrationRunner to set up the schema,
providing integration-style unit tests for all upsert/get/delete operations.

Requirements: 8.1–8.8
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from luma.storage import LearningProgressRecord, RepositoryError
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner
from luma.storage.repositories.teacher_repository import TeacherRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """Fresh in-memory SQLite database with migrations applied."""
    manager = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(manager).run_pending()
    yield manager
    manager.dispose()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _upsert(db: DatabaseManager, **kwargs) -> LearningProgressRecord:
    """Upsert a learning progress record using a managed session."""
    defaults = dict(
        user_id="user1",
        topic="algebra",
        progress=0.5,
        weak_areas=["fractions"],
    )
    defaults.update(kwargs)
    with db.get_session() as session:
        repo = TeacherRepository(session)
        return repo.upsert_progress(**defaults)


# ---------------------------------------------------------------------------
# 1. upsert_progress() creates a new progress record with correct fields
# ---------------------------------------------------------------------------


def test_upsert_creates_new_progress_with_correct_fields(db):
    record = _upsert(
        db,
        user_id="alice",
        topic="geometry",
        progress=0.3,
        weak_areas=["angles", "proofs"],
    )

    assert isinstance(record, LearningProgressRecord)
    assert record.user_id == "alice"
    assert record.topic == "geometry"
    assert record.progress == 0.3
    assert record.weak_areas == ["angles", "proofs"]
    assert record.last_updated is not None
    assert record.id is not None


# ---------------------------------------------------------------------------
# 2. upsert_progress() updates an existing progress record
# ---------------------------------------------------------------------------


def test_upsert_updates_existing_progress(db):
    _upsert(db, user_id="bob", topic="calculus", progress=0.2, weak_areas=["limits"])

    updated = _upsert(
        db,
        user_id="bob",
        topic="calculus",
        progress=0.8,
        weak_areas=["integrals", "derivatives"],
    )

    assert updated.user_id == "bob"
    assert updated.topic == "calculus"
    assert updated.progress == 0.8
    assert updated.weak_areas == ["integrals", "derivatives"]


# ---------------------------------------------------------------------------
# 3. upsert_progress() with weak_areas=None preserves existing weak_areas on update
# ---------------------------------------------------------------------------


def test_upsert_none_weak_areas_preserves_existing(db):
    _upsert(db, user_id="carol", topic="physics", progress=0.4, weak_areas=["momentum", "energy"])

    with db.get_session() as session:
        repo = TeacherRepository(session)
        result = repo.upsert_progress(user_id="carol", topic="physics", progress=0.7, weak_areas=None)

    assert result.progress == 0.7
    assert result.weak_areas == ["momentum", "energy"]


# ---------------------------------------------------------------------------
# 4. get_progress() returns the record after upsert
# ---------------------------------------------------------------------------


def test_get_progress_returns_record_after_upsert(db):
    _upsert(db, user_id="dave", topic="chemistry", progress=0.6, weak_areas=["bonding"])

    with db.get_session() as session:
        fetched = TeacherRepository(session).get_progress("dave", "chemistry")

    assert fetched is not None
    assert isinstance(fetched, LearningProgressRecord)
    assert fetched.user_id == "dave"
    assert fetched.topic == "chemistry"
    assert fetched.progress == 0.6
    assert fetched.weak_areas == ["bonding"]


# ---------------------------------------------------------------------------
# 5. get_progress() returns None for non-existent (user_id, topic)
# ---------------------------------------------------------------------------


def test_get_progress_returns_none_for_nonexistent(db):
    with db.get_session() as session:
        result = TeacherRepository(session).get_progress("ghost", "nonexistent_topic")

    assert result is None


# ---------------------------------------------------------------------------
# 6. get_all_progress() returns all records for a user
# ---------------------------------------------------------------------------


def test_get_all_progress_returns_all_records_for_user(db):
    _upsert(db, user_id="eve", topic="math", progress=0.5, weak_areas=[])
    _upsert(db, user_id="eve", topic="science", progress=0.7, weak_areas=["cells"])
    _upsert(db, user_id="eve", topic="history", progress=0.9, weak_areas=[])

    with db.get_session() as session:
        records = TeacherRepository(session).get_all_progress("eve")

    assert len(records) == 3
    topics = {r.topic for r in records}
    assert topics == {"math", "science", "history"}
    for r in records:
        assert r.user_id == "eve"


# ---------------------------------------------------------------------------
# 7. get_all_progress() returns empty list for user with no records
# ---------------------------------------------------------------------------


def test_get_all_progress_returns_empty_list_for_user_with_no_records(db):
    with db.get_session() as session:
        records = TeacherRepository(session).get_all_progress("nobody")

    assert records == []


# ---------------------------------------------------------------------------
# 8. delete_progress() returns True and removes the record
# ---------------------------------------------------------------------------


def test_delete_returns_true_and_removes_record(db):
    _upsert(db, user_id="frank", topic="biology", progress=0.5, weak_areas=[])

    with db.get_session() as session:
        deleted = TeacherRepository(session).delete_progress("frank", "biology")

    assert deleted is True

    with db.get_session() as session:
        fetched = TeacherRepository(session).get_progress("frank", "biology")

    assert fetched is None


# ---------------------------------------------------------------------------
# 9. delete_progress() returns False for non-existent (user_id, topic)
# ---------------------------------------------------------------------------


def test_delete_returns_false_for_nonexistent(db):
    with db.get_session() as session:
        result = TeacherRepository(session).delete_progress("nobody", "nothing")

    assert result is False


# ---------------------------------------------------------------------------
# 10. Error wrapping: SQLAlchemyError → RepositoryError
# ---------------------------------------------------------------------------


def test_upsert_sqlalchemy_error_wrapped_as_repository_error():
    """Mock session that raises SQLAlchemyError should produce RepositoryError."""
    mock_session = MagicMock()
    mock_session.get_bind.return_value.dialect.name = "sqlite"
    mock_session.execute.side_effect = SQLAlchemyError("db exploded")

    repo = TeacherRepository(mock_session)

    with pytest.raises(RepositoryError) as exc_info:
        repo.upsert_progress(user_id="test", topic="math", progress=0.5)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


def test_get_progress_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.query.side_effect = SQLAlchemyError("connection lost")

    repo = TeacherRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.get_progress("test", "math")


def test_get_all_progress_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.query.side_effect = SQLAlchemyError("timeout")

    repo = TeacherRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.get_all_progress("test")


def test_delete_progress_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.query.side_effect = SQLAlchemyError("disk full")

    repo = TeacherRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.delete_progress("test", "math")
