"""
Unit tests for InsightEngine optional repository injection.

Covers:
- In-memory mode (no repository): engine works without error, create() never called
- Persistence mode (injected repository): create() called once per generated insight
- Repository create() receives correct user_id, message, confidence, evidence
- Empty memory list: no create() calls even with repository injected
- user_id parameter is forwarded to repository.create()

Requirements: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

import pytest

from luma.core.insight.insight_engine import InsightEngine
from luma.core.insight.insight_generator import InsightGenerator
from luma.core.insight.pattern_detector import PatternDetector
from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.memory_interface import (
    MemoryEntry,
    MemoryInterface,
    MemoryRetrievalError,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMemoryInterface(MemoryInterface):
    def __init__(self, memories: List[MemoryEntry]) -> None:
        self._memories = memories

    def store(self, content: str, metadata: Optional[dict] = None) -> str:
        raise AssertionError("InsightEngine must never call store()")

    def retrieve(self, query=None, params=None, limit=10, **kwargs) -> RetrievalResult:
        return {
            "memories": self._memories,
            "total_count": len(self._memories),
            "query_metadata": {},
        }


def _make_memories(n: int) -> List[MemoryEntry]:
    """Create n MemoryEntry dicts that will produce detectable patterns."""
    words = ["python", "java", "rust", "python", "java", "python"]
    return [
        {
            "id": f"m{i}",
            "content": f"{words[i % len(words)]} programming language",
            "metadata": {},
            "timestamp": f"2024-0{(i % 9) + 1}-01T00:00:00",
            "category": "tech",
            "tags": [],
        }
        for i in range(n)
    ]


def _build_engine(
    memories: List[MemoryEntry],
    insight_repository: Optional[Any] = None,
) -> InsightEngine:
    """Build an InsightEngine with low thresholds for easy pattern detection."""
    return InsightEngine(
        memory_interface=MockMemoryInterface(memories),
        pattern_detector=PatternDetector(min_frequency=1, min_confidence=0.0),
        trend_analyzer=TrendAnalyzer(trend_ratio_threshold=1.5),
        insight_generator=InsightGenerator(min_confidence=0.0),
        insight_repository=insight_repository,
    )


# ---------------------------------------------------------------------------
# Tests: in-memory mode (no repository) — Requirement 11.3
# ---------------------------------------------------------------------------


class TestInMemoryMode:
    def test_engine_works_without_repository(self):
        """Engine operates normally when insight_repository=None."""
        engine = _build_engine(_make_memories(6))
        report = engine.generate_insights()
        assert report is not None
        assert report.memory_count == 6

    def test_no_repository_attribute_is_none_by_default(self):
        """insight_repository defaults to None."""
        engine = _build_engine(_make_memories(3))
        assert engine._insight_repository is None

    def test_empty_memories_no_error_without_repository(self):
        """Empty memory list with no repository returns empty report without error."""
        engine = _build_engine([])
        report = engine.generate_insights()
        assert report.insights == []
        assert report.memory_count == 0


# ---------------------------------------------------------------------------
# Tests: persistence mode (injected repository) — Requirements 11.1, 11.2
# ---------------------------------------------------------------------------


class TestPersistenceMode:
    def test_create_called_for_each_insight(self):
        """create() is called once per generated insight."""
        mock_repo = MagicMock()
        memories = _make_memories(6)
        engine = _build_engine(memories, insight_repository=mock_repo)

        report = engine.generate_insights(user_id="user-42")

        assert mock_repo.create.call_count == len(report.insights)

    def test_create_called_with_correct_user_id(self):
        """create() receives the user_id passed to generate_insights()."""
        mock_repo = MagicMock()
        engine = _build_engine(_make_memories(6), insight_repository=mock_repo)

        engine.generate_insights(user_id="alice")

        for c in mock_repo.create.call_args_list:
            assert c.kwargs["user_id"] == "alice"

    def test_create_called_with_correct_message(self):
        """create() receives insight.text as message."""
        mock_repo = MagicMock()
        memories = _make_memories(6)
        engine = _build_engine(memories, insight_repository=mock_repo)

        report = engine.generate_insights(user_id="u1")

        called_messages = {c.kwargs["message"] for c in mock_repo.create.call_args_list}
        expected_messages = {i.text for i in report.insights}
        assert called_messages == expected_messages

    def test_create_called_with_correct_confidence(self):
        """create() receives insight.confidence as confidence."""
        mock_repo = MagicMock()
        memories = _make_memories(6)
        engine = _build_engine(memories, insight_repository=mock_repo)

        report = engine.generate_insights(user_id="u1")

        called_confidences = [c.kwargs["confidence"] for c in mock_repo.create.call_args_list]
        expected_confidences = [i.confidence for i in report.insights]
        assert sorted(called_confidences) == sorted(expected_confidences)

    def test_create_called_with_evidence_dict(self):
        """create() receives evidence as a dict (not a list)."""
        mock_repo = MagicMock()
        engine = _build_engine(_make_memories(6), insight_repository=mock_repo)

        engine.generate_insights(user_id="u1")

        for c in mock_repo.create.call_args_list:
            evidence = c.kwargs["evidence"]
            # evidence must be None or a dict — never a raw list
            assert evidence is None or isinstance(evidence, dict)

    def test_no_create_calls_for_empty_memories(self):
        """create() is never called when no memories are retrieved."""
        mock_repo = MagicMock()
        engine = _build_engine([], insight_repository=mock_repo)

        engine.generate_insights(user_id="u1")

        mock_repo.create.assert_not_called()

    def test_default_user_id_used_when_not_specified(self):
        """Default user_id='default' is forwarded to create() when not provided."""
        mock_repo = MagicMock()
        engine = _build_engine(_make_memories(6), insight_repository=mock_repo)

        engine.generate_insights()

        for c in mock_repo.create.call_args_list:
            assert c.kwargs["user_id"] == "default"

    def test_report_still_returned_with_repository(self):
        """InsightReport is returned correctly even when repository is injected."""
        mock_repo = MagicMock()
        memories = _make_memories(6)
        engine = _build_engine(memories, insight_repository=mock_repo)

        report = engine.generate_insights(user_id="u1")

        assert report.memory_count == 6
        assert len(report.insights) > 0


# ---------------------------------------------------------------------------
# Tests: no import from luma.storage — Requirement 11.4
# ---------------------------------------------------------------------------


class TestNoStorageImport:
    def test_insight_engine_does_not_import_luma_storage(self):
        """InsightEngine module must not import from luma.storage."""
        import luma.core.insight.insight_engine as module
        import sys

        # Verify luma.storage is not in the module's direct imports
        source_file = module.__file__
        with open(source_file) as f:
            source = f.read()

        assert "from luma.storage" not in source
        assert "import luma.storage" not in source
