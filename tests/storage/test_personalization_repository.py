"""
tests/storage/test_personalization_repository.py

Unit tests for PersonalizationRepository.

Uses an in-memory SQLite database with MigrationRunner to set up the schema,
providing integration-style unit tests for all upsert/get/delete operations.

Requirements: 7.1–7.8
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from luma.storage import RepositoryError, UserProfileRecord
from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner
from luma.storage.repositories.personalization_repository import PersonalizationRepository


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


def _upsert(db: DatabaseManager, **kwargs) -> UserProfileRecord:
    """Upsert a user profile using a managed session."""
    defaults = dict(
        user_id="user1",
        interests=["math", "science"],
        preferences={"theme": "dark"},
        strengths=["problem-solving"],
    )
    defaults.update(kwargs)
    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        return repo.upsert(**defaults)


# ---------------------------------------------------------------------------
# 1. upsert() creates a new profile with correct fields
# ---------------------------------------------------------------------------


def test_upsert_creates_new_profile_with_correct_fields(db):
    record = _upsert(
        db,
        user_id="alice",
        interests=["art", "music"],
        preferences={"language": "en"},
        strengths=["creativity"],
    )

    assert isinstance(record, UserProfileRecord)
    assert record.user_id == "alice"
    assert record.interests == ["art", "music"]
    assert record.preferences == {"language": "en"}
    assert record.strengths == ["creativity"]
    assert record.updated_at is not None


# ---------------------------------------------------------------------------
# 2. upsert() updates an existing profile
# ---------------------------------------------------------------------------


def test_upsert_updates_existing_profile(db):
    _upsert(db, user_id="bob", interests=["chess"], preferences={}, strengths=[])

    updated = _upsert(
        db,
        user_id="bob",
        interests=["chess", "go"],
        preferences={"difficulty": "hard"},
        strengths=["strategy"],
    )

    assert updated.user_id == "bob"
    assert updated.interests == ["chess", "go"]
    assert updated.preferences == {"difficulty": "hard"}
    assert updated.strengths == ["strategy"]


# ---------------------------------------------------------------------------
# 3. Partial upsert with interests=None preserves existing interests
# ---------------------------------------------------------------------------


def test_partial_upsert_none_interests_preserves_existing(db):
    _upsert(db, user_id="carol", interests=["history"], preferences={}, strengths=[])

    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        result = repo.upsert(user_id="carol", interests=None, preferences={"mode": "easy"})

    assert result.interests == ["history"]
    assert result.preferences == {"mode": "easy"}


# ---------------------------------------------------------------------------
# 4. Partial upsert with preferences=None preserves existing preferences
# ---------------------------------------------------------------------------


def test_partial_upsert_none_preferences_preserves_existing(db):
    _upsert(db, user_id="dave", interests=[], preferences={"font": "large"}, strengths=[])

    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        result = repo.upsert(user_id="dave", preferences=None, strengths=["reading"])

    assert result.preferences == {"font": "large"}
    assert result.strengths == ["reading"]


# ---------------------------------------------------------------------------
# 5. Partial upsert with strengths=None preserves existing strengths
# ---------------------------------------------------------------------------


def test_partial_upsert_none_strengths_preserves_existing(db):
    _upsert(db, user_id="eve", interests=[], preferences={}, strengths=["logic", "math"])

    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        result = repo.upsert(user_id="eve", strengths=None, interests=["coding"])

    assert result.strengths == ["logic", "math"]
    assert result.interests == ["coding"]


# ---------------------------------------------------------------------------
# 6. get_by_user() returns the profile after upsert
# ---------------------------------------------------------------------------


def test_get_by_user_returns_profile_after_upsert(db):
    _upsert(db, user_id="frank", interests=["biology"], preferences={"speed": "slow"}, strengths=["patience"])

    with db.get_session() as session:
        fetched = PersonalizationRepository(session).get_by_user("frank")

    assert fetched is not None
    assert isinstance(fetched, UserProfileRecord)
    assert fetched.user_id == "frank"
    assert fetched.interests == ["biology"]
    assert fetched.preferences == {"speed": "slow"}
    assert fetched.strengths == ["patience"]


# ---------------------------------------------------------------------------
# 7. get_by_user() returns None for non-existent user
# ---------------------------------------------------------------------------


def test_get_by_user_returns_none_for_nonexistent_user(db):
    with db.get_session() as session:
        result = PersonalizationRepository(session).get_by_user("ghost_user")

    assert result is None


# ---------------------------------------------------------------------------
# 8. delete() returns True and removes the profile
# ---------------------------------------------------------------------------


def test_delete_returns_true_and_removes_profile(db):
    _upsert(db, user_id="grace")

    with db.get_session() as session:
        deleted = PersonalizationRepository(session).delete("grace")

    assert deleted is True

    with db.get_session() as session:
        fetched = PersonalizationRepository(session).get_by_user("grace")

    assert fetched is None


# ---------------------------------------------------------------------------
# 9. delete() returns False for non-existent user
# ---------------------------------------------------------------------------


def test_delete_returns_false_for_nonexistent_user(db):
    with db.get_session() as session:
        result = PersonalizationRepository(session).delete("nobody")

    assert result is False


# ---------------------------------------------------------------------------
# 10. Error wrapping: SQLAlchemyError → RepositoryError
# ---------------------------------------------------------------------------


def test_upsert_sqlalchemy_error_wrapped_as_repository_error():
    """Mock session that raises SQLAlchemyError should produce RepositoryError."""
    mock_session = MagicMock()
    mock_session.get_bind.return_value.dialect.name = "sqlite"
    mock_session.execute.side_effect = SQLAlchemyError("db exploded")

    repo = PersonalizationRepository(mock_session)

    with pytest.raises(RepositoryError) as exc_info:
        repo.upsert(user_id="test", interests=["x"])

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)


def test_get_by_user_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("connection lost")

    repo = PersonalizationRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.get_by_user("test")


def test_delete_sqlalchemy_error_wrapped():
    mock_session = MagicMock()
    mock_session.get.side_effect = SQLAlchemyError("disk full")

    repo = PersonalizationRepository(mock_session)

    with pytest.raises(RepositoryError):
        repo.delete("test")
