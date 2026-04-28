"""
Unit tests for ProgressTracker.

Tests cover:
- record_completion stores with correct metadata
- record_completion idempotency (skips duplicate lesson IDs)
- get_progress retrieves with correct query parameters
- ProgressStorageError wraps original exception on store failure
- ProgressRetrievalError wraps original exception on retrieve failure
- get_weak_areas returns only lesson IDs with score < 0.6
- get_completion_ratio returns 0.0 when total_lessons == 0
"""

import json
import pytest
from unittest.mock import MagicMock, call

from luma.core.teacher.progress_tracker import ProgressTracker
from luma.core.teacher.schemas import ProgressStorageError, ProgressRetrievalError


# ---------------------------------------------------------------------------
# Helper: in-memory mock MemoryInterface
# ---------------------------------------------------------------------------


def _make_mock_memory():
    """Create a mock MemoryInterface that stores and retrieves in-memory."""
    stored = []

    def store(content, metadata=None):
        stored.append({"content": content, "metadata": metadata or {}})
        return f"id_{len(stored)}"

    def retrieve(query=None, params=None, limit=10, **kwargs):
        return {
            "memories": [
                {
                    "id": f"id_{i}",
                    "content": s["content"],
                    "metadata": s["metadata"],
                    "timestamp": "2024-01-15T10:30:00",
                    "category": s["metadata"].get("category", ""),
                    "tags": [],
                }
                for i, s in enumerate(stored)
            ],
            "total_count": len(stored),
            "query_metadata": {},
        }

    mi = MagicMock()
    mi.store.side_effect = store
    mi.retrieve.side_effect = retrieve
    return mi, stored


# ---------------------------------------------------------------------------
# Test 1: record_completion calls store() with correct metadata
# ---------------------------------------------------------------------------


def test_record_completion_stores_correct_metadata():
    """record_completion() calls MemoryInterface.store() with correct metadata fields."""
    mi, stored = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.record_completion(
        user_id="user1",
        topic="python",
        lesson_id="lesson_01",
        score=0.85,
    )

    assert mi.store.call_count == 1
    _, kwargs = mi.store.call_args
    # store is called positionally: store(content, metadata)
    args = mi.store.call_args[0]
    metadata = args[1] if len(args) > 1 else mi.store.call_args[1].get("metadata", {})

    assert metadata["user_id"] == "user1"
    assert metadata["topic"] == "python"
    assert metadata["lesson_id"] == "lesson_01"
    assert metadata["category"] == "teacher_progress"


# ---------------------------------------------------------------------------
# Test 2: record_completion skips duplicate lesson IDs (idempotency)
# ---------------------------------------------------------------------------


def test_record_completion_skips_duplicate_lesson_id():
    """record_completion() called twice with same args only calls store() once."""
    mi, stored = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.record_completion("user1", "python", "lesson_01", 0.9)
    tracker.record_completion("user1", "python", "lesson_01", 0.9)

    assert mi.store.call_count == 1, (
        f"Expected store() called once, got {mi.store.call_count}"
    )


# ---------------------------------------------------------------------------
# Test 3: get_progress calls retrieve() with correct query parameters
# ---------------------------------------------------------------------------


def test_get_progress_retrieves_with_correct_params():
    """get_progress() calls MemoryInterface.retrieve() with category='teacher_progress'."""
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.get_progress("user1", "python")

    assert mi.retrieve.call_count >= 1
    call_kwargs = mi.retrieve.call_args[1] if mi.retrieve.call_args[1] else {}
    call_args = mi.retrieve.call_args[0] if mi.retrieve.call_args[0] else ()

    # retrieve is called with params= keyword argument
    params = call_kwargs.get("params") or (call_args[1] if len(call_args) > 1 else None)
    assert params is not None, "Expected retrieve() to be called with params="
    assert params.get("category") == "teacher_progress", (
        f"Expected category='teacher_progress', got {params.get('category')!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: ProgressStorageError wraps original exception on store failure
# ---------------------------------------------------------------------------


def test_progress_storage_error_wraps_original_exception():
    """ProgressStorageError is raised with __cause__ set when store() fails."""
    mi = MagicMock()
    original_exc = Exception("storage failed")
    mi.store.side_effect = original_exc
    # retrieve must return a valid empty result so idempotency check passes
    mi.retrieve.return_value = {
        "memories": [],
        "total_count": 0,
        "query_metadata": {},
    }

    tracker = ProgressTracker(mi)

    with pytest.raises(ProgressStorageError) as exc_info:
        tracker.record_completion("user1", "python", "lesson_01", 0.5)

    assert exc_info.value.__cause__ is original_exc


# ---------------------------------------------------------------------------
# Test 5: ProgressRetrievalError wraps original exception on retrieve failure
# ---------------------------------------------------------------------------


def test_progress_retrieval_error_wraps_original_exception():
    """ProgressRetrievalError is raised with __cause__ set when retrieve() fails."""
    mi = MagicMock()
    original_exc = Exception("retrieval failed")
    mi.retrieve.side_effect = original_exc

    tracker = ProgressTracker(mi)

    with pytest.raises(ProgressRetrievalError) as exc_info:
        tracker.get_progress("user1", "python")

    assert exc_info.value.__cause__ is original_exc


# ---------------------------------------------------------------------------
# Test 6: get_weak_areas returns only lesson IDs with score < 0.6
# ---------------------------------------------------------------------------


def test_get_weak_areas_returns_only_below_threshold():
    """get_weak_areas() returns only lesson IDs where score < 0.6."""
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    tracker.record_completion("user1", "python", "lesson_low", 0.3)
    tracker.record_completion("user1", "python", "lesson_boundary", 0.6)
    tracker.record_completion("user1", "python", "lesson_high", 0.9)

    weak_areas = tracker.get_weak_areas("user1", "python")

    assert "lesson_low" in weak_areas, "lesson with score 0.3 should be a weak area"
    assert "lesson_boundary" not in weak_areas, "lesson with score 0.6 should NOT be a weak area"
    assert "lesson_high" not in weak_areas, "lesson with score 0.9 should NOT be a weak area"
    assert len(weak_areas) == 1


# ---------------------------------------------------------------------------
# Test 7: get_completion_ratio returns 0.0 when total_lessons == 0
# ---------------------------------------------------------------------------


def test_get_completion_ratio_returns_zero_when_total_lessons_is_zero():
    """get_completion_ratio() returns 0.0 when total_lessons == 0."""
    mi, _ = _make_mock_memory()
    tracker = ProgressTracker(mi)

    # Even with a recorded lesson, total_lessons=0 should return 0.0
    tracker.record_completion("user1", "python", "lesson_01", 0.8)
    ratio = tracker.get_completion_ratio("user1", "python", total_lessons=0)

    assert ratio == 0.0, f"Expected 0.0, got {ratio}"
