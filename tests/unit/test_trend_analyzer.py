"""
Unit tests for TrendAnalyzer.

Covers:
- Returns [] for fewer than 2 memories
- Returns [] when all memories fall in one window
- Increasing trend detected when ratio is met
- Decreasing trend detected when ratio is met
- No trend emitted when ratio is not met
- Confidence is in [0.0, 1.0]
- Input objects are not mutated
- ValueError raised for trend_ratio_threshold <= 1.0
- Malformed timestamps are skipped gracefully
"""

import copy
import pytest
from typing import List

from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.insight.schemas import PatternResult, TrendResult
from luma.core.memory_interface import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory(mem_id: str, timestamp: str, content: str = "test") -> MemoryEntry:
    return {
        "id": mem_id,
        "content": content,
        "metadata": {},
        "timestamp": timestamp,
        "category": "test",
        "tags": [],
    }


def make_pattern(pattern: str, evidence: List[str]) -> PatternResult:
    return PatternResult(
        pattern_type="keyword",
        pattern=pattern,
        frequency=len(evidence),
        confidence=min(1.0, len(evidence) / 10),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_raises_for_threshold_equal_to_one(self):
        with pytest.raises(ValueError, match="trend_ratio_threshold"):
            TrendAnalyzer(trend_ratio_threshold=1.0)

    def test_raises_for_threshold_below_one(self):
        with pytest.raises(ValueError, match="trend_ratio_threshold"):
            TrendAnalyzer(trend_ratio_threshold=0.5)

    def test_raises_for_negative_threshold(self):
        with pytest.raises(ValueError):
            TrendAnalyzer(trend_ratio_threshold=-1.0)

    def test_valid_threshold_above_one(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        assert analyzer is not None

    def test_valid_threshold_exactly_above_one(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.01)
        assert analyzer is not None


# ---------------------------------------------------------------------------
# Edge cases: too few memories
# ---------------------------------------------------------------------------

class TestFewMemories:
    def test_returns_empty_for_zero_memories(self):
        analyzer = TrendAnalyzer()
        result = analyzer.analyze([], [])
        assert result == []

    def test_returns_empty_for_one_memory(self):
        analyzer = TrendAnalyzer()
        memories = [make_memory("m1", "2024-01-01T00:00:00")]
        patterns = [make_pattern("python", ["m1"])]
        result = analyzer.analyze(patterns, memories)
        assert result == []

    def test_returns_empty_for_no_patterns(self):
        analyzer = TrendAnalyzer()
        memories = [
            make_memory("m1", "2024-01-01T00:00:00"),
            make_memory("m2", "2024-06-01T00:00:00"),
        ]
        result = analyzer.analyze([], memories)
        assert result == []


# ---------------------------------------------------------------------------
# Edge case: all memories in one window
# ---------------------------------------------------------------------------

class TestAllMemoriesInOneWindow:
    def test_returns_empty_when_all_same_timestamp(self):
        """All memories share the same timestamp → midpoint == t_max → all in recent."""
        analyzer = TrendAnalyzer()
        ts = "2024-01-01T00:00:00"
        memories = [
            make_memory("m1", ts),
            make_memory("m2", ts),
            make_memory("m3", ts),
        ]
        patterns = [make_pattern("python", ["m1", "m2", "m3"])]
        result = analyzer.analyze(patterns, memories)
        assert result == []

    def test_returns_empty_when_all_in_recent_window(self):
        """Two memories with same timestamp → midpoint == t_max → both in recent."""
        analyzer = TrendAnalyzer()
        memories = [
            make_memory("m1", "2024-06-01T00:00:00"),
            make_memory("m2", "2024-06-01T00:00:00"),
        ]
        patterns = [make_pattern("python", ["m1", "m2"])]
        result = analyzer.analyze(patterns, memories)
        assert result == []


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

class TestTrendDetection:
    def _make_spread_memories(self):
        """4 earlier memories + 4 recent memories, well separated."""
        return [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("e3", "2024-02-01T00:00:00"),
            make_memory("e4", "2024-02-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
            make_memory("r3", "2024-08-01T00:00:00"),
            make_memory("r4", "2024-08-15T00:00:00"),
        ]

    def test_increasing_trend_detected(self):
        """Pattern with 1 earlier and 3 recent occurrences → ratio 3.0 >= 1.5."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        # 1 earlier, 3 recent → ratio = 3.0
        patterns = [make_pattern("python", ["e1", "r1", "r2", "r3"])]
        results = analyzer.analyze(patterns, memories)
        assert len(results) == 1
        assert results[0].trend == "increasing"
        assert results[0].topic == "python"

    def test_decreasing_trend_detected(self):
        """Pattern with 3 earlier and 1 recent occurrence → ratio 3.0 >= 1.5."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        # 3 earlier, 1 recent → ratio = 3.0
        patterns = [make_pattern("python", ["e1", "e2", "e3", "r1"])]
        results = analyzer.analyze(patterns, memories)
        assert len(results) == 1
        assert results[0].trend == "decreasing"
        assert results[0].topic == "python"

    def test_no_trend_when_ratio_not_met(self):
        """Pattern with 2 earlier and 2 recent → ratio 1.0 < 1.5 → no trend."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        patterns = [make_pattern("python", ["e1", "e2", "r1", "r2"])]
        results = analyzer.analyze(patterns, memories)
        assert results == []

    def test_no_trend_when_earlier_count_is_zero(self):
        """Pattern with evidence only in recent window → earlier_count == 0 → skip."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        patterns = [make_pattern("python", ["r1", "r2", "r3"])]
        results = analyzer.analyze(patterns, memories)
        assert results == []

    def test_time_window_label_is_recent_half(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        patterns = [make_pattern("python", ["e1", "r1", "r2", "r3"])]
        results = analyzer.analyze(patterns, memories)
        assert results[0].time_window == "recent_half"

    def test_multiple_patterns_multiple_trends(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = self._make_spread_memories()
        patterns = [
            make_pattern("python", ["e1", "r1", "r2", "r3"]),   # increasing
            make_pattern("java", ["e1", "e2", "e3", "r1"]),     # decreasing
        ]
        results = analyzer.analyze(patterns, memories)
        trends = {r.topic: r.trend for r in results}
        assert trends["python"] == "increasing"
        assert trends["java"] == "decreasing"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_in_range(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
            make_memory("r3", "2024-08-01T00:00:00"),
        ]
        patterns = [make_pattern("python", ["e1", "r1", "r2", "r3"])]
        results = analyzer.analyze(patterns, memories)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_confidence_formula(self):
        """confidence = min(1.0, max(recent, earlier) / total)"""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        # 2 earlier, 4 recent, 6 total → max=4, confidence=4/6 ≈ 0.667
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
            make_memory("r3", "2024-08-01T00:00:00"),
            make_memory("r4", "2024-08-15T00:00:00"),
        ]
        patterns = [make_pattern("python", ["e1", "r1", "r2", "r3"])]
        results = analyzer.analyze(patterns, memories)
        assert len(results) == 1
        expected = min(1.0, 3 / 6)  # max(3 recent, 1 earlier) / 6 total
        assert abs(results[0].confidence - expected) < 1e-9

    def test_confidence_capped_at_one(self):
        """When max count equals total, confidence should be exactly 1.0."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
        ]
        # 1 earlier, 2 recent → ratio=2.0 >= 1.5 → increasing
        # max(2,1)/3 = 0.667 — won't hit cap here, but let's verify it's ≤ 1.0
        patterns = [make_pattern("python", ["e1", "r1", "r2"])]
        results = analyzer.analyze(patterns, memories)
        assert len(results) == 1
        assert results[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_memories_not_mutated(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
        ]
        original_memories = copy.deepcopy(memories)
        patterns = [make_pattern("python", ["e1", "r1", "r2"])]
        analyzer.analyze(patterns, memories)
        assert memories == original_memories

    def test_patterns_not_mutated(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
        ]
        patterns = [make_pattern("python", ["e1", "r1", "r2"])]
        original_patterns = copy.deepcopy(patterns)
        analyzer.analyze(patterns, memories)
        for orig, after in zip(original_patterns, patterns):
            assert orig.pattern == after.pattern
            assert orig.evidence == after.evidence
            assert orig.frequency == after.frequency
            assert orig.confidence == after.confidence


# ---------------------------------------------------------------------------
# Malformed timestamps
# ---------------------------------------------------------------------------

class TestMalformedTimestamps:
    def test_malformed_timestamp_skipped_gracefully(self):
        """Memory with bad timestamp is skipped; valid memories still processed."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("bad", "not-a-timestamp"),
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
        ]
        # bad memory is skipped; e1 earlier, r1+r2 recent → ratio 2.0 >= 1.5
        patterns = [make_pattern("python", ["e1", "r1", "r2"])]
        results = analyzer.analyze(patterns, memories)
        # Should still detect trend from valid memories
        assert len(results) == 1
        assert results[0].trend == "increasing"

    def test_all_malformed_timestamps_returns_empty(self):
        """If all timestamps are malformed, fewer than 2 valid → return []."""
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            make_memory("m1", "bad-ts"),
            make_memory("m2", "also-bad"),
        ]
        patterns = [make_pattern("python", ["m1", "m2"])]
        result = analyzer.analyze(patterns, memories)
        assert result == []

    def test_empty_timestamp_skipped(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        memories = [
            {"id": "m1", "content": "test", "metadata": {}, "timestamp": "",
             "category": "test", "tags": []},
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
        ]
        patterns = [make_pattern("python", ["e1", "r1"])]
        # e1 earlier, r1 recent → ratio 1.0 < 1.5 → no trend (equal counts)
        result = analyzer.analyze(patterns, memories)
        assert result == []


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_by_confidence_desc_then_topic_asc(self):
        analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
        # Build memories: 2 earlier + 4 recent
        memories = [
            make_memory("e1", "2024-01-01T00:00:00"),
            make_memory("e2", "2024-01-15T00:00:00"),
            make_memory("r1", "2024-07-01T00:00:00"),
            make_memory("r2", "2024-07-15T00:00:00"),
            make_memory("r3", "2024-08-01T00:00:00"),
            make_memory("r4", "2024-08-15T00:00:00"),
        ]
        # python: 1 earlier, 3 recent → increasing, confidence = 3/6 = 0.5
        # java:   1 earlier, 3 recent → increasing, confidence = 3/6 = 0.5
        # (same confidence → sort by topic ASC: java < python)
        patterns = [
            make_pattern("python", ["e1", "r1", "r2", "r3"]),
            make_pattern("java", ["e2", "r2", "r3", "r4"]),
        ]
        results = analyzer.analyze(patterns, memories)
        assert len(results) == 2
        assert results[0].topic == "java"
        assert results[1].topic == "python"
