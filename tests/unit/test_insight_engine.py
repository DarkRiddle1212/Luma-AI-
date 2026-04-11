"""
Unit tests for luma.core.insight.insight_engine.InsightEngine.

Covers:
- retrieve() called with namespace as category filter when provided
- retrieve() called without category filter when namespace is None
- InsightReport has correct memory_count
- InsightReport has correct pattern_count
- InsightReport has correct trend_count
- Empty InsightReport returned when no memories retrieved
- store() is never called on MemoryInterface
- MemoryRetrievalError is propagated
"""

import pytest
from typing import List, Optional, Dict, Any

from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult,
    MemoryRetrievalError,
)
from luma.core.insight.insight_engine import InsightEngine
from luma.core.insight.pattern_detector import PatternDetector
from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.insight.insight_generator import InsightGenerator
from luma.core.insight.schemas import InsightReport


# ---------------------------------------------------------------------------
# MockMemoryInterface
# ---------------------------------------------------------------------------

class MockMemoryInterface(MemoryInterface):
    def __init__(self, memories: List[MemoryEntry]):
        self._memories = memories
        self.retrieve_calls: List[Dict[str, Any]] = []
        self.store_called = False

    def store(self, content: str, metadata=None) -> str:
        raise AssertionError("InsightEngine must never call store()")

    def retrieve(self, query=None, params=None, limit=10, **kwargs) -> RetrievalResult:
        self.retrieve_calls.append({"query": query, "params": params})
        return {
            "memories": self._memories,
            "total_count": len(self._memories),
            "query_metadata": {},
        }


class ErrorMemoryInterface(MemoryInterface):
    """Always raises MemoryRetrievalError on retrieve()."""

    def store(self, content: str, metadata=None) -> str:
        raise AssertionError("Should not be called")

    def retrieve(self, query=None, params=None, limit=10, **kwargs) -> RetrievalResult:
        raise MemoryRetrievalError("Simulated retrieval failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memories(n: int, category: str = "tech") -> List[MemoryEntry]:
    """Create n distinct MemoryEntry dicts with varied content."""
    words = ["python", "java", "rust", "python", "java", "python"]
    return [
        {
            "id": f"m{i}",
            "content": f"{words[i % len(words)]} programming language",
            "metadata": {},
            "timestamp": f"2024-0{(i % 9) + 1}-01T00:00:00",
            "category": category,
            "tags": [],
        }
        for i in range(n)
    ]


def _build_engine(memories: List[MemoryEntry]) -> tuple:
    """Return (engine, mock_interface) with low thresholds for easy pattern detection."""
    mock = MockMemoryInterface(memories)
    engine = InsightEngine(
        memory_interface=mock,
        pattern_detector=PatternDetector(min_frequency=1, min_confidence=0.0),
        trend_analyzer=TrendAnalyzer(trend_ratio_threshold=1.5),
        insight_generator=InsightGenerator(min_confidence=0.0),
    )
    return engine, mock


# ---------------------------------------------------------------------------
# Tests: namespace / category filter
# ---------------------------------------------------------------------------

class TestNamespaceFilter:
    def test_retrieve_called_with_namespace_when_provided(self):
        memories = _make_memories(3)
        engine, mock = _build_engine(memories)

        engine.generate_insights(namespace="work")

        assert len(mock.retrieve_calls) == 1
        params = mock.retrieve_calls[0]["params"]
        assert params is not None
        assert params.get("category") == "work"

    def test_retrieve_called_without_namespace_filter_when_not_provided(self):
        memories = _make_memories(3)
        engine, mock = _build_engine(memories)

        engine.generate_insights(namespace=None)

        assert len(mock.retrieve_calls) == 1
        params = mock.retrieve_calls[0]["params"]
        assert params is not None
        assert "category" not in params

    def test_retrieve_called_with_correct_limit(self):
        memories = _make_memories(3)
        engine, mock = _build_engine(memories)

        engine.generate_insights(limit=42)

        params = mock.retrieve_calls[0]["params"]
        assert params["limit"] == 42


# ---------------------------------------------------------------------------
# Tests: InsightReport counts
# ---------------------------------------------------------------------------

class TestInsightReportCounts:
    def test_insight_report_has_correct_memory_count(self):
        memories = _make_memories(5)
        engine, _ = _build_engine(memories)

        report = engine.generate_insights()

        assert report.memory_count == 5

    def test_insight_report_has_correct_pattern_count(self):
        memories = _make_memories(6)
        engine, _ = _build_engine(memories)

        report = engine.generate_insights()

        # Verify pattern_count matches what PatternDetector would produce
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        expected_patterns = detector.detect(memories)
        assert report.pattern_count == len(expected_patterns)

    def test_insight_report_has_correct_trend_count(self):
        memories = _make_memories(6)
        engine, _ = _build_engine(memories)

        report = engine.generate_insights()

        # Verify trend_count matches what TrendAnalyzer would produce
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        patterns = detector.detect(memories)
        expected_trends = analyzer.analyze(patterns, memories)
        assert report.trend_count == len(expected_trends)


# ---------------------------------------------------------------------------
# Tests: empty memory retrieval
# ---------------------------------------------------------------------------

class TestEmptyMemoryRetrieval:
    def test_empty_report_returned_for_empty_memory_retrieval(self):
        engine, _ = _build_engine([])

        report = engine.generate_insights()

        assert isinstance(report, InsightReport)
        assert report.insights == []
        assert report.pattern_count == 0
        assert report.trend_count == 0
        assert report.memory_count == 0


# ---------------------------------------------------------------------------
# Tests: store() never called
# ---------------------------------------------------------------------------

class TestStoreNeverCalled:
    def test_store_never_called(self):
        memories = _make_memories(3)
        engine, mock = _build_engine(memories)

        # Should not raise AssertionError from MockMemoryInterface.store()
        report = engine.generate_insights()

        assert report is not None
        # If store() had been called, MockMemoryInterface would have raised AssertionError


# ---------------------------------------------------------------------------
# Tests: error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    def test_memory_retrieval_error_propagated(self):
        error_interface = ErrorMemoryInterface()
        engine = InsightEngine(
            memory_interface=error_interface,
            pattern_detector=PatternDetector(min_frequency=1, min_confidence=0.0),
            trend_analyzer=TrendAnalyzer(trend_ratio_threshold=1.5),
            insight_generator=InsightGenerator(min_confidence=0.0),
        )

        with pytest.raises(MemoryRetrievalError):
            engine.generate_insights()
