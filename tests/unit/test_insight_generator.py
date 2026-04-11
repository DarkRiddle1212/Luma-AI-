"""
Unit tests for luma.core.insight.insight_generator.InsightGenerator.

Covers:
- Empty inputs return []
- Each pattern above threshold produces at least one insight
- Each trend above threshold produces at least one insight
- Pattern + trend for same topic are combined into a single insight
- All insights have non-empty evidence
- Insights with no derivable evidence are dropped
- Output is sorted deterministically (confidence DESC, text ASC)
- ValueError raised for invalid constructor args
"""

import pytest
from typing import List

from luma.core.insight.insight_generator import InsightGenerator
from luma.core.insight.schemas import Insight, PatternResult, TrendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pattern(
    pattern: str,
    frequency: int = 3,
    confidence: float = 0.8,
    evidence: List[str] = None,
    pattern_type: str = "keyword",
) -> PatternResult:
    return PatternResult(
        pattern_type=pattern_type,
        pattern=pattern,
        frequency=frequency,
        confidence=confidence,
        evidence=evidence or ["m1", "m2", "m3"],
    )


def make_trend(
    topic: str,
    trend: str = "increasing",
    confidence: float = 0.8,
    time_window: str = "recent_half",
) -> TrendResult:
    return TrendResult(
        trend=trend,
        topic=topic,
        confidence=confidence,
        time_window=time_window,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_min_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            InsightGenerator(min_confidence=-0.1)

    def test_min_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            InsightGenerator(min_confidence=1.1)

    def test_valid_default_does_not_raise(self):
        gen = InsightGenerator()
        assert gen is not None

    def test_boundary_zero_valid(self):
        gen = InsightGenerator(min_confidence=0.0)
        assert gen is not None

    def test_boundary_one_valid(self):
        gen = InsightGenerator(min_confidence=1.0)
        assert gen is not None


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_both_empty_returns_empty(self):
        gen = InsightGenerator(min_confidence=0.5)
        result = gen.generate([], [])
        assert result == []

    def test_empty_patterns_empty_trends_returns_empty(self):
        gen = InsightGenerator(min_confidence=0.0)
        result = gen.generate([], [])
        assert result == []


# ---------------------------------------------------------------------------
# Pattern-only insights
# ---------------------------------------------------------------------------

class TestPatternOnlyInsights:
    def test_pattern_above_threshold_produces_insight(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.8)]
        result = gen.generate(patterns, [])
        assert len(result) == 1
        assert "python" in result[0].text

    def test_pattern_at_threshold_produces_insight(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.5)]
        result = gen.generate(patterns, [])
        assert len(result) == 1

    def test_pattern_below_threshold_excluded(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.4)]
        result = gen.generate(patterns, [])
        assert result == []

    def test_pattern_only_text_template(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", frequency=5, confidence=0.8)]
        result = gen.generate(patterns, [])
        assert len(result) == 1
        assert result[0].text == "You frequently engage with 'python' (mentioned 5 times)."

    def test_multiple_patterns_produce_multiple_insights(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [
            make_pattern("python", confidence=0.8),
            make_pattern("java", confidence=0.7),
        ]
        result = gen.generate(patterns, [])
        assert len(result) == 2
        topics = {i.text for i in result}
        assert any("python" in t for t in topics)
        assert any("java" in t for t in topics)

    def test_pattern_confidence_used_for_insight(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", confidence=0.75)]
        result = gen.generate(patterns, [])
        assert result[0].confidence == 0.75

    def test_pattern_evidence_used_for_insight(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", evidence=["m1", "m2"])]
        result = gen.generate(patterns, [])
        assert result[0].evidence == ["m1", "m2"]


# ---------------------------------------------------------------------------
# Trend-only insights
# ---------------------------------------------------------------------------

class TestTrendOnlyInsights:
    def test_trend_above_threshold_with_matching_pattern_produces_insight(self):
        gen = InsightGenerator(min_confidence=0.5)
        # Trend with no matching pattern in patterns list → should be skipped
        # Trend with matching pattern → should produce trend-only insight
        patterns = [make_pattern("python", confidence=0.3)]  # below threshold
        trends = [make_trend("python", confidence=0.8)]
        result = gen.generate(patterns, trends)
        # pattern below threshold → not combined; trend above threshold → trend-only
        assert len(result) == 1
        assert "python" in result[0].text
        assert "increasing" in result[0].text

    def test_trend_below_threshold_excluded(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.3)]  # below threshold
        trends = [make_trend("python", confidence=0.4)]  # below threshold
        result = gen.generate(patterns, trends)
        assert result == []

    def test_trend_only_text_template(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", confidence=0.8)]
        # Make pattern below threshold so trend is not combined
        gen2 = InsightGenerator(min_confidence=0.9)
        trends = [make_trend("python", trend="decreasing", time_window="recent_half", confidence=0.95)]
        result = gen2.generate(patterns, trends)
        # pattern confidence 0.8 < 0.9 → not combined; trend 0.95 >= 0.9 → trend-only
        assert len(result) == 1
        assert result[0].text == "Your interest in 'python' is decreasing (recent_half)."

    def test_trend_without_matching_pattern_is_dropped(self):
        gen = InsightGenerator(min_confidence=0.5)
        # No patterns at all → trend has no evidence source → dropped
        trends = [make_trend("python", confidence=0.8)]
        result = gen.generate([], trends)
        assert result == []

    def test_trend_evidence_derived_from_pattern(self):
        gen = InsightGenerator(min_confidence=0.9)  # high threshold
        patterns = [make_pattern("python", confidence=0.8, evidence=["m1", "m2"])]
        trends = [make_trend("python", confidence=0.95)]
        result = gen.generate(patterns, trends)
        # pattern 0.8 < 0.9 → not combined; trend 0.95 >= 0.9 → trend-only
        assert len(result) == 1
        assert result[0].evidence == ["m1", "m2"]


# ---------------------------------------------------------------------------
# Combined insights
# ---------------------------------------------------------------------------

class TestCombinedInsights:
    def test_pattern_and_trend_same_topic_combined(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("python", confidence=0.9)]
        result = gen.generate(patterns, trends)
        assert len(result) == 1
        assert "frequently engage" in result[0].text
        assert "interest" in result[0].text

    def test_combined_text_template(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("python", trend="increasing", time_window="recent_half", confidence=0.9)]
        result = gen.generate(patterns, trends)
        assert len(result) == 1
        assert result[0].text == (
            "You frequently engage with 'python' and your interest is increasing (recent_half)."
        )

    def test_combined_confidence_is_average(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", confidence=0.6)]
        trends = [make_trend("python", confidence=0.8)]
        result = gen.generate(patterns, trends)
        assert len(result) == 1
        assert abs(result[0].confidence - 0.7) < 1e-9

    def test_combined_uses_pattern_evidence(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", evidence=["m1", "m2", "m3"])]
        trends = [make_trend("python")]
        result = gen.generate(patterns, trends)
        assert result[0].evidence == ["m1", "m2", "m3"]

    def test_combined_trend_not_emitted_separately(self):
        """When pattern+trend are combined, no separate trend-only insight is emitted."""
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("python", confidence=0.9)]
        result = gen.generate(patterns, trends)
        # Only one combined insight, not two
        assert len(result) == 1

    def test_trend_below_threshold_not_combined(self):
        """If trend confidence is below threshold, pattern-only insight is produced."""
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("python", confidence=0.3)]  # below threshold
        result = gen.generate(patterns, trends)
        assert len(result) == 1
        # Should be pattern-only
        assert "mentioned" in result[0].text
        assert "interest" not in result[0].text


# ---------------------------------------------------------------------------
# Evidence requirements
# ---------------------------------------------------------------------------

class TestEvidenceRequirements:
    def test_all_insights_have_non_empty_evidence(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [
            make_pattern("python", evidence=["m1", "m2"]),
            make_pattern("java", evidence=["m3"]),
        ]
        trends = [make_trend("python")]
        result = gen.generate(patterns, trends)
        for insight in result:
            assert len(insight.evidence) > 0, f"Insight has empty evidence: {insight.text}"

    def test_insight_with_empty_evidence_dropped(self):
        """Insights that would have empty evidence are dropped."""
        gen = InsightGenerator(min_confidence=0.5)
        # Trend with no matching pattern → no evidence → dropped
        trends = [make_trend("orphan_topic", confidence=0.9)]
        result = gen.generate([], trends)
        assert result == []


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------

class TestSortingAndDeterminism:
    def test_sorted_by_confidence_desc(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [
            make_pattern("python", confidence=0.6),
            make_pattern("java", confidence=0.9),
            make_pattern("rust", confidence=0.75),
        ]
        result = gen.generate(patterns, [])
        confidences = [i.confidence for i in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_sorted_by_text_asc_for_equal_confidence(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [
            make_pattern("zzz", confidence=0.8, frequency=3),
            make_pattern("aaa", confidence=0.8, frequency=3),
        ]
        result = gen.generate(patterns, [])
        assert len(result) == 2
        assert result[0].text < result[1].text

    def test_same_input_produces_same_output(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [
            make_pattern("python", confidence=0.8),
            make_pattern("java", confidence=0.7),
        ]
        trends = [make_trend("python", confidence=0.9)]
        result1 = gen.generate(patterns, trends)
        result2 = gen.generate(patterns, trends)
        assert len(result1) == len(result2)
        for i1, i2 in zip(result1, result2):
            assert i1.text == i2.text
            assert i1.confidence == i2.confidence
            assert i1.evidence == i2.evidence

    def test_multiple_patterns_and_trends_sorted_correctly(self):
        gen = InsightGenerator(min_confidence=0.0)
        patterns = [
            make_pattern("python", confidence=0.9),
            make_pattern("java", confidence=0.7),
        ]
        trends = [
            make_trend("python", confidence=0.9),
            make_trend("java", confidence=0.7),
        ]
        result = gen.generate(patterns, trends)
        # Both combined: python avg=(0.9+0.9)/2=0.9, java avg=(0.7+0.7)/2=0.7
        assert len(result) == 2
        assert result[0].confidence >= result[1].confidence


# ---------------------------------------------------------------------------
# Mixed scenarios
# ---------------------------------------------------------------------------

class TestMixedScenarios:
    def test_some_patterns_above_some_below_threshold(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [
            make_pattern("python", confidence=0.8),
            make_pattern("java", confidence=0.3),  # below threshold
        ]
        result = gen.generate(patterns, [])
        assert len(result) == 1
        assert "python" in result[0].text

    def test_pattern_and_unrelated_trend(self):
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("java", confidence=0.8)]  # different topic, no matching pattern
        result = gen.generate(patterns, trends)
        # python → pattern-only; java trend → no matching pattern → dropped
        assert len(result) == 1
        assert "python" in result[0].text

    def test_trend_confidence_below_threshold_not_combined(self):
        """Pattern above threshold, trend below threshold → pattern-only insight."""
        gen = InsightGenerator(min_confidence=0.5)
        patterns = [make_pattern("python", confidence=0.8)]
        trends = [make_trend("python", confidence=0.4)]
        result = gen.generate(patterns, trends)
        assert len(result) == 1
        assert "mentioned" in result[0].text  # pattern-only template
