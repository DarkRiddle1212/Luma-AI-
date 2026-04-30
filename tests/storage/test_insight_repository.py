"""
tests/storage/test_insight_repository.py

Unit tests for InsightRepository.

Uses an in-memory SQLite database with MigrationRunner to set up the schema,
providing integration-style unit tests for all CRUD operations.

Requirements: 6.1–6.8
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from luma.storage import InsightRecord, RepositoryError
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner
from luma.storage.repositories.insight_repository import InsightRepository


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


def _create(db: DatabaseManager, **kwargs) -> InsightRecord:
    """Create an insight record using a managed session."""
    defaults = dict(
        user_id="user1",
        message="test insight",
        confidence=0.8,
        evidence=None,
    )
    defaults.update(kwargs)
    with db.get_session() as session:
        r = InsightRepository(session)
        return r.create(**defaults)


# ---------------------------------------------------------------------------
# 1. create() returns InsightRecord with correct fields (evidence=None)
# ---------------------------------------------------------------------------


def test_create_returns_insight_record_with_no_evidence(db):
    record = _create(
        db,
        user_id="alice",
        message="user prefers short answers",
        confidence=0.9,
        evidence=None,
    )

    assert isinstance(record, InsightRecord)
    assert record.id is not None
    assert record.user_id == "alice"
    assert record.message == "user prefers short answers"
    assert record.confidence == pytest.approx(0.9)
    assert record.evidence is None
    assert record.created_at is not None


def test_create_returns_insight_record_with_evidence_dict(db):
    evidence = {"source": "session_42", "examples": ["q1", "q2"]}
    record = _create(
        db,
        user_id="bob",
        message="user struggles with recursion",
        confidence=0.75,
        evidence=evidence,
    )

    assert isinstance(record, InsightRecord)
    assert record.id is not None
    assert record.user_id == "bob"
    assert record.message == "user struggles with recursion"
    assert record.confidence == pytest.approx(0.75)
    assert record.evidence == evidence
    assert record.created_at is not None


# ---------------------------------------------------------------------------
# 2. get_by_id() returns the record after create
# ---------------------------------------------------------------------------


def test_get_by_id_returns_record_after_create(db):
    created = _create(db, user_id="alice", message="remember this insight")

    with db.get_session() as session:
        fetched = InsightRepository(session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.message == "remember this insight"
    assert fetched.user_id == "alice"


# ---------------------------------------------------------------------------
# 3. get_by_id() returns None for non-existent id
# ---------------------------------------------------------------------------


def test_get_by_id_returns_none_for_nonexistent_id(db):
    with db.get_session() as session:
        result = InsightRepository(session).get_by_id(99999)

    assert result is None


# ---------------------------------------------------------------------------
# 4. get_by_user() returns all insights ordered by created_at descending
# ---------------------------------------------------------------------------


def test_get_by_user_returns_all_insights_ordered_descending(db):
    r1 = _create(db, user_id="carol", message="first insight")
    time.sleep(0.01)
    r2 = _create(db, user_id="carol", message="second insight")
    time.sleep(0.01)
    r3 = _create(db, user_id="carol", message="third insight")

    with db.get_session() as session:
        records = InsightRepository(session).get_by_user("carol")

    assert len(records) == 3
    # Newest first
    assert records[0].id == r3.id
    assert records[1].id == r2.id
    assert records[2].id == r1.id


# ---------------------------------------------------------------------------
# 5. get_by_user() respects the limit parameter
# ---------------------------------------------------------------------------


def test_get_by_user_respects_limit(db):
    for i in range(5):
        _create(db, user_id="dave", message=f"insight {i}")

    with db.get_session() as session:
        records = InsightRepository(session).get_by_user("dave", limit=3)

    assert len(records) == 3


# ---------------------------------------------------------------------------
# 6. delete() returns True and removes the record
# ---------------------------------------------------------------------------


def test_delete_returns_true_and_removes_record(db):
    created = _create(db, user_id="eve")

    with db.get_session() as session:
        deleted = InsightRepository(session).delete(created.id)

    assert deleted is True

    with db.get_session() as session:
        fetched = InsightRepository(session).get_by_id(created.id)

    assert fetched is None


# ---------------------------------------------------------------------------
# 7. delete() returns False for non-existent id
# ---------------------------------------------------------------------------


def test_delete_returns_false_for_nonexistent_id(db):
    with db.get_session() as session:
        result = InsightRepository(session).delete(99999)

    assert result is False


# ---------------------------------------------------------------------------
# 8. Error wrapping: SQLAlchemyError → RepositoryError
# ---------------------------------------------------------------------------


def test_create_sqlalchemy_error_wrapped_as_repository_error():
    """Mock session that raises SQLAlchemyError on add should produce RepositoryError."""
    mock_session = MagicMock()
    mock_session.add.side_effect = SQLAlchemyError("db exploded")

    repo = InsightRepository(mock_session)

    with pytest.raises(RepositoryError) as exc_info:
        repo.create(
            user_id="test",
            message="some insight",
            confidence=0.5,
            evidence=None,
        )

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


def test_get_by_id_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("connection lost")

    repo = InsightRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.get_by_id(1)


def test_delete_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("disk full")

    repo = InsightRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.delete(1)
