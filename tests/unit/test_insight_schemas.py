"""
Unit tests for luma.core.insight.schemas.

Tests validation logic for PatternResult, TrendResult, Insight, and InsightReport.
Works regardless of whether Pydantic or dataclasses are used.
"""

import pytest
from luma.core.insight.schemas import (
    PatternResult,
    TrendResult,
    Insight,
    InsightReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern_result(**overrides):
    defaults = dict(
        pattern_type="keyword",
        pattern="python",
        frequency=3,
        confidence=0.8,
        evidence=["mem-1", "mem-2", "mem-3"],
    )
    defaults.update(overrides)
    return PatternResult(**defaults)


def _make_trend_result(**overrides):
    defaults = dict(
        trend="increasing",
        topic="python",
        confidence=0.7,
        time_window="recent_half",
    )
    defaults.update(overrides)
    return TrendResult(**defaults)


def _make_insight(**overrides):
    defaults = dict(
        text="You frequently engage with 'python'.",
        confidence=0.8,
        evidence=["mem-1"],
    )
    defaults.update(overrides)
    return Insight(**defaults)


# ---------------------------------------------------------------------------
# PatternResult
# ---------------------------------------------------------------------------

class TestPatternResult:
    def test_valid_construction(self):
        pr = _make_pattern_result()
        assert pr.pattern_type == "keyword"
        assert pr.pattern == "python"
        assert pr.frequency == 3
        assert pr.confidence == 0.8
        assert pr.evidence == ["mem-1", "mem-2", "mem-3"]

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_pattern_result(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_pattern_result(confidence=1.1)

    def test_confidence_at_zero_is_valid(self):
        pr = _make_pattern_result(confidence=0.0)
        assert pr.confidence == 0.0

    def test_confidence_at_one_is_valid(self):
        pr = _make_pattern_result(confidence=1.0)
        assert pr.confidence == 1.0

    def test_frequency_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_pattern_result(frequency=0)

    def test_frequency_negative_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_pattern_result(frequency=-1)

    def test_frequency_one_is_valid(self):
        pr = _make_pattern_result(frequency=1, evidence=["mem-1"])
        assert pr.frequency == 1

    def test_empty_evidence_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_pattern_result(evidence=[])


# ---------------------------------------------------------------------------
# TrendResult
# ---------------------------------------------------------------------------

class TestTrendResult:
    def test_valid_construction(self):
        tr = _make_trend_result()
        assert tr.trend == "increasing"
        assert tr.topic == "python"
        assert tr.confidence == 0.7
        assert tr.time_window == "recent_half"

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_trend_result(confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_trend_result(confidence=1.01)

    def test_confidence_at_boundaries_valid(self):
        tr_low = _make_trend_result(confidence=0.0)
        tr_high = _make_trend_result(confidence=1.0)
        assert tr_low.confidence == 0.0
        assert tr_high.confidence == 1.0


# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------

class TestInsight:
    def test_valid_construction(self):
        ins = _make_insight()
        assert ins.text == "You frequently engage with 'python'."
        assert ins.confidence == 0.8
        assert ins.evidence == ["mem-1"]

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_insight(confidence=-0.5)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_insight(confidence=2.0)

    def test_confidence_at_boundaries_valid(self):
        ins_low = _make_insight(confidence=0.0)
        ins_high = _make_insight(confidence=1.0)
        assert ins_low.confidence == 0.0
        assert ins_high.confidence == 1.0

    def test_empty_evidence_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_insight(evidence=[])


# ---------------------------------------------------------------------------
# InsightReport
# ---------------------------------------------------------------------------

class TestInsightReport:
    def test_valid_construction(self):
        insights = [_make_insight()]
        report = InsightReport(
            insights=insights,
            pattern_count=5,
            trend_count=2,
            memory_count=100,
        )
        assert len(report.insights) == 1
        assert report.pattern_count == 5
        assert report.trend_count == 2
        assert report.memory_count == 100

    def test_empty_insights_is_valid(self):
        report = InsightReport(
            insights=[],
            pattern_count=0,
            trend_count=0,
            memory_count=0,
        )
        assert report.insights == []
