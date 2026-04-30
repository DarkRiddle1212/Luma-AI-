"""
tests/storage/test_memory_repository.py

Unit tests for MemoryRepository.

Uses an in-memory SQLite database with MigrationRunner to set up the schema,
providing integration-style unit tests for all CRUD operations.

Requirements: 5.1–5.10
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from luma.storage import MemoryRecord, RepositoryError
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner
from luma.storage.repositories.memory_repository import MemoryRepository


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


@pytest.fixture()
def repo(db):
    """MemoryRepository backed by a real session (auto-committed via context manager)."""
    return db


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _create(db: DatabaseManager, **kwargs) -> MemoryRecord:
    """Create a memory record using a managed session."""
    defaults = dict(
        user_id="user1",
        namespace=None,
        content="test content",
        importance_score=0.5,
        final_score=0.6,
    )
    defaults.update(kwargs)
    with db.get_session() as session:
        r = MemoryRepository(session)
        return r.create(**defaults)


# ---------------------------------------------------------------------------
# 1. create() returns a MemoryRecord with correct fields
# ---------------------------------------------------------------------------


def test_create_returns_memory_record_with_correct_fields(db):
    record = _create(
        db,
        user_id="alice",
        namespace="work",
        content="hello world",
        importance_score=0.8,
        final_score=0.9,
    )

    assert isinstance(record, MemoryRecord)
    assert record.id is not None
    assert record.user_id == "alice"
    assert record.namespace == "work"
    assert record.content == "hello world"
    assert record.importance_score == pytest.approx(0.8)
    assert record.final_score == pytest.approx(0.9)
    assert record.created_at is not None


# ---------------------------------------------------------------------------
# 2. get_by_id() returns the record after create
# ---------------------------------------------------------------------------


def test_get_by_id_returns_record_after_create(db):
    created = _create(db, user_id="alice", content="remember this")

    with db.get_session() as session:
        fetched = MemoryRepository(session).get_by_id(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.content == "remember this"
    assert fetched.user_id == "alice"


# ---------------------------------------------------------------------------
# 3. get_by_id() returns None for non-existent id
# ---------------------------------------------------------------------------


def test_get_by_id_returns_none_for_nonexistent_id(db):
    with db.get_session() as session:
        result = MemoryRepository(session).get_by_id(99999)

    assert result is None


# ---------------------------------------------------------------------------
# 4. get_by_user() returns all memories ordered by created_at descending
# ---------------------------------------------------------------------------


def test_get_by_user_returns_all_memories_ordered_descending(db):
    # Create three records with a small delay to ensure distinct timestamps
    r1 = _create(db, user_id="bob", content="first")
    time.sleep(0.01)
    r2 = _create(db, user_id="bob", content="second")
    time.sleep(0.01)
    r3 = _create(db, user_id="bob", content="third")

    with db.get_session() as session:
        records = MemoryRepository(session).get_by_user("bob")

    assert len(records) == 3
    # Newest first
    assert records[0].id == r3.id
    assert records[1].id == r2.id
    assert records[2].id == r1.id


# ---------------------------------------------------------------------------
# 5. get_by_user() with namespace filter returns only matching records
# ---------------------------------------------------------------------------


def test_get_by_user_namespace_filter(db):
    _create(db, user_id="carol", namespace="work", content="work memory")
    _create(db, user_id="carol", namespace="personal", content="personal memory")
    _create(db, user_id="carol", namespace="work", content="another work memory")

    with db.get_session() as session:
        work_records = MemoryRepository(session).get_by_user("carol", namespace="work")

    assert len(work_records) == 2
    assert all(r.namespace == "work" for r in work_records)


# ---------------------------------------------------------------------------
# 6. get_by_user() respects the limit parameter
# ---------------------------------------------------------------------------


def test_get_by_user_respects_limit(db):
    for i in range(5):
        _create(db, user_id="dave", content=f"memory {i}")

    with db.get_session() as session:
        records = MemoryRepository(session).get_by_user("dave", limit=3)

    assert len(records) == 3


# ---------------------------------------------------------------------------
# 7. update() updates importance_score and final_score
# ---------------------------------------------------------------------------


def test_update_changes_importance_and_final_score(db):
    created = _create(db, user_id="eve", importance_score=0.1, final_score=0.2)

    with db.get_session() as session:
        updated = MemoryRepository(session).update(
            created.id, importance_score=0.7, final_score=0.8
        )

    assert updated is not None
    assert updated.importance_score == pytest.approx(0.7)
    assert updated.final_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 8. update() returns None for non-existent id
# ---------------------------------------------------------------------------


def test_update_returns_none_for_nonexistent_id(db):
    with db.get_session() as session:
        result = MemoryRepository(session).update(99999, importance_score=0.5)

    assert result is None


# ---------------------------------------------------------------------------
# 9. update() with None fields preserves existing values
# ---------------------------------------------------------------------------


def test_update_with_none_fields_preserves_existing_values(db):
    created = _create(db, user_id="frank", importance_score=0.3, final_score=0.4)

    with db.get_session() as session:
        updated = MemoryRepository(session).update(created.id, importance_score=0.9)

    assert updated is not None
    assert updated.importance_score == pytest.approx(0.9)
    # final_score was not passed → should remain unchanged
    assert updated.final_score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 10. delete() returns True and removes the record
# ---------------------------------------------------------------------------


def test_delete_returns_true_and_removes_record(db):
    created = _create(db, user_id="grace")

    with db.get_session() as session:
        deleted = MemoryRepository(session).delete(created.id)

    assert deleted is True

    with db.get_session() as session:
        fetched = MemoryRepository(session).get_by_id(created.id)

    assert fetched is None


# ---------------------------------------------------------------------------
# 11. delete() returns False for non-existent id
# ---------------------------------------------------------------------------


def test_delete_returns_false_for_nonexistent_id(db):
    with db.get_session() as session:
        result = MemoryRepository(session).delete(99999)

    assert result is False


# ---------------------------------------------------------------------------
# 12. count_by_user() returns correct count
# ---------------------------------------------------------------------------


def test_count_by_user_returns_correct_count(db):
    _create(db, user_id="henry")
    _create(db, user_id="henry")
    _create(db, user_id="henry")
    _create(db, user_id="other_user")

    with db.get_session() as session:
        count = MemoryRepository(session).count_by_user("henry")

    assert count == 3


# ---------------------------------------------------------------------------
# 13. Error wrapping: SQLAlchemyError → RepositoryError
# ---------------------------------------------------------------------------


def test_sqlalchemy_error_is_wrapped_as_repository_error():
    """Mock session that raises SQLAlchemyError should produce RepositoryError."""
    mock_session = MagicMock()
    mock_session.add.side_effect = SQLAlchemyError("db exploded")

    repo = MemoryRepository(mock_session)

    with pytest.raises(RepositoryError) as exc_info:
        repo.create(
            user_id="test",
            namespace=None,
            content="content",
            importance_score=0.5,
            final_score=0.5,
        )

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


def test_get_by_id_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("connection lost")

    repo = MemoryRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.get_by_id(1)


def test_delete_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("disk full")

    repo = MemoryRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.delete(1)
