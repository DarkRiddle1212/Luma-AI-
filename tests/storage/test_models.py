"""
tests/storage/test_models.py

Unit tests for luma/storage/models.py (ORM models).

Verifies column types, nullability, defaults, and index definitions via
SQLAlchemy inspection for all four ORM models.

Requirements: 3.1–3.8
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.types import DateTime, Float, Integer, JSON, String, Text

from luma.storage.database import Base, DatabaseManager
from luma.storage.models import (
    InsightModel,
    LearningProgressModel,
    MemoryModel,
    UserProfileModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite engine with all tables created."""
    db = DatabaseManager("sqlite:///:memory:")
    db.create_all_tables()
    yield db._engine
    db.dispose()


@pytest.fixture(scope="module")
def inspector(engine):
    """Return a SQLAlchemy inspector for the in-memory database."""
    return inspect(engine)


def _col(inspector, table_name: str, col_name: str) -> dict:
    """Return the column info dict for a given table and column name."""
    cols = {c["name"]: c for c in inspector.get_columns(table_name)}
    assert col_name in cols, f"Column '{col_name}' not found in table '{table_name}'"
    return cols[col_name]


def _index_names(inspector, table_name: str) -> set[str]:
    """Return the set of index names for a given table."""
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def _index_columns(inspector, table_name: str, index_name: str) -> list[str]:
    """Return the column list for a named index."""
    for idx in inspector.get_indexes(table_name):
        if idx["name"] == index_name:
            return idx["column_names"]
    raise AssertionError(f"Index '{index_name}' not found on table '{table_name}'")


# ---------------------------------------------------------------------------
# MemoryModel tests (Requirement 3.1, 3.6, 3.7, 3.8)
# ---------------------------------------------------------------------------


class TestMemoryModel:
    """Verify MemoryModel schema matches Requirement 3.1."""

    def test_table_name(self):
        assert MemoryModel.__tablename__ == "memories"

    def test_id_column(self, inspector):
        col = _col(inspector, "memories", "id")
        assert isinstance(col["type"], Integer)
        assert col["nullable"] is False  # PK is non-nullable

    def test_id_is_primary_key(self):
        pk_cols = [c.name for c in MemoryModel.__table__.primary_key.columns]
        assert "id" in pk_cols

    def test_id_autoincrement(self):
        col = MemoryModel.__table__.c["id"]
        assert col.autoincrement is True or col.autoincrement == "auto"

    def test_user_id_column(self, inspector):
        col = _col(inspector, "memories", "user_id")
        assert isinstance(col["type"], String)
        assert col["nullable"] is False

    def test_namespace_column(self, inspector):
        col = _col(inspector, "memories", "namespace")
        assert isinstance(col["type"], String)
        assert col["nullable"] is True

    def test_content_column(self, inspector):
        col = _col(inspector, "memories", "content")
        assert isinstance(col["type"], Text)
        assert col["nullable"] is False

    def test_importance_score_column(self, inspector):
        col = _col(inspector, "memories", "importance_score")
        assert isinstance(col["type"], Float)
        assert col["nullable"] is False

    def test_importance_score_default(self):
        col = MemoryModel.__table__.c["importance_score"]
        assert col.default is not None
        assert col.default.arg == 0.0

    def test_final_score_column(self, inspector):
        col = _col(inspector, "memories", "final_score")
        assert isinstance(col["type"], Float)
        assert col["nullable"] is False

    def test_final_score_default(self):
        col = MemoryModel.__table__.c["final_score"]
        assert col.default is not None
        assert col.default.arg == 0.0

    def test_created_at_column(self, inspector):
        col = _col(inspector, "memories", "created_at")
        assert isinstance(col["type"], DateTime)
        assert col["nullable"] is False

    def test_created_at_default_is_callable(self):
        col = MemoryModel.__table__.c["created_at"]
        assert col.default is not None
        assert callable(col.default.arg)

    def test_user_id_has_individual_index(self, inspector):
        """user_id should have a standalone index (Requirement 3.1)."""
        indexes = inspector.get_indexes("memories")
        indexed_cols = [
            idx["column_names"] for idx in indexes if not idx.get("unique", False)
        ]
        assert any("user_id" in cols for cols in indexed_cols)

    def test_created_at_has_individual_index(self, inspector):
        """created_at should have a standalone index (Requirement 3.1)."""
        indexes = inspector.get_indexes("memories")
        indexed_cols = [idx["column_names"] for idx in indexes]
        assert any("created_at" in cols for cols in indexed_cols)

    def test_composite_index_user_id_namespace(self, inspector):
        """Composite index on (user_id, namespace) must exist (Requirement 3.6)."""
        assert "ix_memories_user_id_namespace" in _index_names(inspector, "memories")
        cols = _index_columns(inspector, "memories", "ix_memories_user_id_namespace")
        assert cols == ["user_id", "namespace"]

    def test_inherits_from_base(self):
        """MemoryModel must inherit from Base (Requirement 3.7)."""
        assert issubclass(MemoryModel, Base)


# ---------------------------------------------------------------------------
# InsightModel tests (Requirement 3.2, 3.7, 3.8)
# ---------------------------------------------------------------------------


class TestInsightModel:
    """Verify InsightModel schema matches Requirement 3.2."""

    def test_table_name(self):
        assert InsightModel.__tablename__ == "insights"

    def test_id_column(self, inspector):
        col = _col(inspector, "insights", "id")
        assert isinstance(col["type"], Integer)
        assert col["nullable"] is False

    def test_id_is_primary_key(self):
        pk_cols = [c.name for c in InsightModel.__table__.primary_key.columns]
        assert "id" in pk_cols

    def test_id_autoincrement(self):
        col = InsightModel.__table__.c["id"]
        assert col.autoincrement is True or col.autoincrement == "auto"

    def test_user_id_column(self, inspector):
        col = _col(inspector, "insights", "user_id")
        assert isinstance(col["type"], String)
        assert col["nullable"] is False

    def test_message_column(self, inspector):
        col = _col(inspector, "insights", "message")
        assert isinstance(col["type"], Text)
        assert col["nullable"] is False

    def test_confidence_column(self, inspector):
        col = _col(inspector, "insights", "confidence")
        assert isinstance(col["type"], Float)
        assert col["nullable"] is False

    def test_evidence_column(self, inspector):
        col = _col(inspector, "insights", "evidence")
        assert isinstance(col["type"], (JSON,))
        assert col["nullable"] is True

    def test_created_at_column(self, inspector):
        col = _col(inspector, "insights", "created_at")
        assert isinstance(col["type"], DateTime)
        assert col["nullable"] is False

    def test_created_at_default_is_callable(self):
        col = InsightModel.__table__.c["created_at"]
        assert col.default is not None
        assert callable(col.default.arg)

    def test_user_id_has_index(self, inspector):
        """user_id should be indexed (Requirement 3.2)."""
        indexes = inspector.get_indexes("insights")
        indexed_cols = [idx["column_names"] for idx in indexes]
        assert any("user_id" in cols for cols in indexed_cols)

    def test_created_at_has_index(self, inspector):
        """created_at should be indexed (Requirement 3.2)."""
        indexes = inspector.get_indexes("insights")
        indexed_cols = [idx["column_names"] for idx in indexes]
        assert any("created_at" in cols for cols in indexed_cols)

    def test_inherits_from_base(self):
        """InsightModel must inherit from Base (Requirement 3.7)."""
        assert issubclass(InsightModel, Base)


# ---------------------------------------------------------------------------
# UserProfileModel tests (Requirement 3.3, 3.7, 3.8)
# ---------------------------------------------------------------------------


class TestUserProfileModel:
    """Verify UserProfileModel schema matches Requirement 3.3."""

    def test_table_name(self):
        assert UserProfileModel.__tablename__ == "user_profiles"

    def test_user_id_is_primary_key(self):
        pk_cols = [c.name for c in UserProfileModel.__table__.primary_key.columns]
        assert "user_id" in pk_cols

    def test_user_id_column(self, inspector):
        col = _col(inspector, "user_profiles", "user_id")
        assert isinstance(col["type"], String)
        assert col["nullable"] is False

    def test_interests_column(self, inspector):
        col = _col(inspector, "user_profiles", "interests")
        assert isinstance(col["type"], JSON)
        assert col["nullable"] is False

    def test_interests_default_is_list(self):
        col = UserProfileModel.__table__.c["interests"]
        assert col.default is not None
        # The default callable should produce an empty list
        fn = col.default.arg
        assert callable(fn) and fn.__name__ == "list"

    def test_preferences_column(self, inspector):
        col = _col(inspector, "user_profiles", "preferences")
        assert isinstance(col["type"], JSON)
        assert col["nullable"] is False

    def test_preferences_default_is_dict(self):
        col = UserProfileModel.__table__.c["preferences"]
        assert col.default is not None
        fn = col.default.arg
        assert callable(fn) and fn.__name__ == "dict"

    def test_strengths_column(self, inspector):
        col = _col(inspector, "user_profiles", "strengths")
        assert isinstance(col["type"], JSON)
        assert col["nullable"] is False

    def test_strengths_default_is_list(self):
        col = UserProfileModel.__table__.c["strengths"]
        assert col.default is not None
        fn = col.default.arg
        assert callable(fn) and fn.__name__ == "list"

    def test_updated_at_column(self, inspector):
        col = _col(inspector, "user_profiles", "updated_at")
        assert isinstance(col["type"], DateTime)
        assert col["nullable"] is False

    def test_updated_at_default_is_callable(self):
        col = UserProfileModel.__table__.c["updated_at"]
        assert col.default is not None
        assert callable(col.default.arg)

    def test_updated_at_onupdate_is_callable(self):
        col = UserProfileModel.__table__.c["updated_at"]
        assert col.onupdate is not None
        assert callable(col.onupdate.arg)

    def test_inherits_from_base(self):
        """UserProfileModel must inherit from Base (Requirement 3.7)."""
        assert issubclass(UserProfileModel, Base)


# ---------------------------------------------------------------------------
# LearningProgressModel tests (Requirement 3.4, 3.5, 3.7, 3.8)
# ---------------------------------------------------------------------------


class TestLearningProgressModel:
    """Verify LearningProgressModel schema matches Requirements 3.4 and 3.5."""

    def test_table_name(self):
        assert LearningProgressModel.__tablename__ == "learning_progress"

    def test_id_column(self, inspector):
        col = _col(inspector, "learning_progress", "id")
        assert isinstance(col["type"], Integer)
        assert col["nullable"] is False

    def test_id_is_primary_key(self):
        pk_cols = [c.name for c in LearningProgressModel.__table__.primary_key.columns]
        assert "id" in pk_cols

    def test_id_autoincrement(self):
        col = LearningProgressModel.__table__.c["id"]
        assert col.autoincrement is True or col.autoincrement == "auto"

    def test_user_id_column(self, inspector):
        col = _col(inspector, "learning_progress", "user_id")
        assert isinstance(col["type"], String)
        assert col["nullable"] is False

    def test_topic_column(self, inspector):
        col = _col(inspector, "learning_progress", "topic")
        assert isinstance(col["type"], String)
        assert col["nullable"] is False

    def test_progress_column(self, inspector):
        col = _col(inspector, "learning_progress", "progress")
        assert isinstance(col["type"], Float)
        assert col["nullable"] is False

    def test_progress_default(self):
        col = LearningProgressModel.__table__.c["progress"]
        assert col.default is not None
        assert col.default.arg == 0.0

    def test_weak_areas_column(self, inspector):
        col = _col(inspector, "learning_progress", "weak_areas")
        assert isinstance(col["type"], JSON)
        assert col["nullable"] is False

    def test_weak_areas_default_is_list(self):
        col = LearningProgressModel.__table__.c["weak_areas"]
        assert col.default is not None
        fn = col.default.arg
        assert callable(fn) and fn.__name__ == "list"

    def test_last_updated_column(self, inspector):
        col = _col(inspector, "learning_progress", "last_updated")
        assert isinstance(col["type"], DateTime)
        assert col["nullable"] is False

    def test_last_updated_default_is_callable(self):
        col = LearningProgressModel.__table__.c["last_updated"]
        assert col.default is not None
        assert callable(col.default.arg)

    def test_last_updated_onupdate_is_callable(self):
        col = LearningProgressModel.__table__.c["last_updated"]
        assert col.onupdate is not None
        assert callable(col.onupdate.arg)

    def test_user_id_has_individual_index(self, inspector):
        """user_id should have a standalone index (Requirement 3.4)."""
        indexes = inspector.get_indexes("learning_progress")
        indexed_cols = [idx["column_names"] for idx in indexes]
        assert any("user_id" in cols for cols in indexed_cols)

    def test_composite_index_user_id_topic(self, inspector):
        """Composite index on (user_id, topic) must exist (Requirement 3.5)."""
        assert "ix_learning_progress_user_id_topic" in _index_names(
            inspector, "learning_progress"
        )
        cols = _index_columns(
            inspector, "learning_progress", "ix_learning_progress_user_id_topic"
        )
        assert cols == ["user_id", "topic"]

    def test_inherits_from_base(self):
        """LearningProgressModel must inherit from Base (Requirement 3.7)."""
        assert issubclass(LearningProgressModel, Base)


# ---------------------------------------------------------------------------
# DateTime UTC tests (Requirement 3.8)
# ---------------------------------------------------------------------------


class TestDateTimeUTC:
    """All DateTime columns should store UTC values (Requirement 3.8)."""

    def test_memory_created_at_default_returns_utc(self):
        from datetime import timezone
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_insight_created_at_default_returns_utc(self):
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_user_profile_updated_at_default_returns_utc(self):
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_user_profile_updated_at_onupdate_returns_utc(self):
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_learning_progress_last_updated_default_returns_utc(self):
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_learning_progress_last_updated_onupdate_returns_utc(self):
        from luma.storage.models import _utcnow

        dt = _utcnow()
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_all_datetime_defaults_reference_utcnow(self):
        """All DateTime column defaults/onupdates should reference the _utcnow helper."""
        datetime_cols = [
            MemoryModel.__table__.c["created_at"],
            InsightModel.__table__.c["created_at"],
            UserProfileModel.__table__.c["updated_at"],
            LearningProgressModel.__table__.c["last_updated"],
        ]
        for col in datetime_cols:
            assert col.default is not None
            fn = col.default.arg
            assert callable(fn) and fn.__name__ == "_utcnow", (
                f"Column {col.name} default should reference _utcnow"
            )

    def test_onupdate_columns_reference_utcnow(self):
        """Columns with onupdate should reference _utcnow."""
        onupdate_cols = [
            UserProfileModel.__table__.c["updated_at"],
            LearningProgressModel.__table__.c["last_updated"],
        ]
        for col in onupdate_cols:
            assert col.onupdate is not None
            fn = col.onupdate.arg
            assert callable(fn) and fn.__name__ == "_utcnow", (
                f"Column {col.name} onupdate should reference _utcnow"
            )


# ---------------------------------------------------------------------------
# All tables present in schema (Requirement 3.7)
# ---------------------------------------------------------------------------


class TestAllTablesPresent:
    """All four ORM tables must be created by create_all_tables() (Requirement 3.7)."""

    def test_all_tables_exist(self, inspector):
        table_names = set(inspector.get_table_names())
        assert "memories" in table_names
        assert "insights" in table_names
        assert "user_profiles" in table_names
        assert "learning_progress" in table_names
