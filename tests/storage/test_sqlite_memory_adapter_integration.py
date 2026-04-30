"""
Integration tests for SQLiteMemoryAdapter.

Uses a real in-memory SQLite database with migrations applied.
Tests metadata mapping, category→namespace mapping, and error wrapping.

Requirements: 10.1–10.6, 15.5
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from luma.storage import DatabaseManager, MigrationRunner, MemoryRepository, RepositoryError
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import MemoryStorageError, MemoryRetrievalError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """Fresh in-memory SQLite database with migrations applied."""
    database_manager = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(database_manager).run_pending()
    yield database_manager
    database_manager.dispose()


@pytest.fixture()
def adapter(db):
    """SQLiteMemoryAdapter wired to the in-memory database."""
    with db.get_session() as session:
        repo = MemoryRepository(session)
    # repo is used only as a reference; adapter creates fresh repos per operation
    return SQLiteMemoryAdapter(repo, db)


# ---------------------------------------------------------------------------
# store() — metadata mapping
# ---------------------------------------------------------------------------


class TestStoreMetadataMapping:
    def test_store_with_full_metadata(self, adapter, db):
        """store() with user_id, namespace, importance, final_score stores correctly."""
        metadata = {
            "user_id": "alice",
            "namespace": "work",
            "importance": 0.8,
            "final_score": 0.9,
        }
        memory_id = adapter.store("Full metadata content", metadata=metadata)

        assert memory_id is not None
        assert isinstance(memory_id, str)

        # Verify the record was persisted with correct field values
        with db.get_session() as session:
            repo = MemoryRepository(session)
            record = repo.get_by_id(int(memory_id))

        assert record is not None
        assert record.user_id == "alice"
        assert record.namespace == "work"
        assert record.content == "Full metadata content"
        assert record.importance_score == pytest.approx(0.8)
        assert record.final_score == pytest.approx(0.9)

    def test_store_with_category_maps_to_namespace(self, adapter, db):
        """store() with category in metadata uses it as namespace."""
        metadata = {
            "user_id": "bob",
            "category": "science",
        }
        memory_id = adapter.store("Category content", metadata=metadata)

        with db.get_session() as session:
            repo = MemoryRepository(session)
            record = repo.get_by_id(int(memory_id))

        assert record is not None
        assert record.namespace == "science"

    def test_store_with_no_metadata_uses_defaults(self, adapter, db):
        """store() with no metadata defaults to user_id='default', namespace=None."""
        memory_id = adapter.store("No metadata content")

        with db.get_session() as session:
            repo = MemoryRepository(session)
            record = repo.get_by_id(int(memory_id))

        assert record is not None
        assert record.user_id == "default"
        assert record.namespace is None
        assert record.importance_score == pytest.approx(0.0)
        assert record.final_score == pytest.approx(0.0)

    def test_store_namespace_takes_precedence_over_category(self, adapter, db):
        """When both namespace and category are in metadata, namespace wins."""
        metadata = {
            "user_id": "carol",
            "namespace": "explicit_ns",
            "category": "fallback_cat",
        }
        memory_id = adapter.store("Namespace wins", metadata=metadata)

        with db.get_session() as session:
            repo = MemoryRepository(session)
            record = repo.get_by_id(int(memory_id))

        assert record.namespace == "explicit_ns"


# ---------------------------------------------------------------------------
# retrieve() — category→namespace mapping and filtering
# ---------------------------------------------------------------------------


class TestRetrieveCategoryNamespaceMapping:
    def test_retrieve_with_category_filter_maps_to_namespace(self, adapter, db):
        """retrieve() with category param filters by namespace in get_by_user()."""
        adapter.store("Memory A", metadata={"user_id": "dave", "namespace": "sports"})
        adapter.store("Memory B", metadata={"user_id": "dave", "namespace": "music"})

        result = adapter.retrieve(params={"user_id": "dave", "category": "sports"})

        contents = [m["content"] for m in result["memories"]]
        assert "Memory A" in contents
        assert "Memory B" not in contents

    def test_retrieve_returns_memories_for_correct_user(self, adapter, db):
        """retrieve() only returns memories belonging to the requested user."""
        adapter.store("Eve memory", metadata={"user_id": "eve"})
        adapter.store("Frank memory", metadata={"user_id": "frank"})

        result = adapter.retrieve(params={"user_id": "eve"})

        assert result["total_count"] == 1
        assert result["memories"][0]["content"] == "Eve memory"

    def test_retrieve_respects_limit_parameter(self, adapter, db):
        """retrieve() returns at most `limit` memories."""
        for i in range(5):
            adapter.store(f"Memory {i}", metadata={"user_id": "grace"})

        result = adapter.retrieve(params={"user_id": "grace", "limit": 3})

        assert len(result["memories"]) == 3
        assert result["total_count"] == 3

    def test_retrieve_returns_result_structure(self, adapter, db):
        """retrieve() returns a properly structured RetrievalResult."""
        adapter.store("Structured content", metadata={"user_id": "henry"})

        result = adapter.retrieve(params={"user_id": "henry"})

        assert "memories" in result
        assert "total_count" in result
        assert "query_metadata" in result
        assert result["total_count"] == 1
        memory = result["memories"][0]
        assert "id" in memory
        assert "content" in memory
        assert memory["content"] == "Structured content"


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


class TestErrorWrapping:
    def test_repository_error_from_store_wrapped_as_memory_storage_error(self):
        """RepositoryError raised during store() is wrapped as MemoryStorageError."""
        mock_db = MagicMock()
        mock_db.get_session.side_effect = RepositoryError("DB failure")

        repo = MagicMock()
        adapter = SQLiteMemoryAdapter(repo, mock_db)

        with pytest.raises(MemoryStorageError):
            adapter.store("content", metadata={"user_id": "test"})

    def test_repository_error_from_retrieve_wrapped_as_memory_retrieval_error(self):
        """RepositoryError raised during retrieve() is wrapped as MemoryRetrievalError."""
        mock_db = MagicMock()
        mock_db.get_session.side_effect = RepositoryError("DB failure")

        repo = MagicMock()
        adapter = SQLiteMemoryAdapter(repo, mock_db)

        with pytest.raises(MemoryRetrievalError):
            adapter.retrieve(params={"user_id": "test"})
