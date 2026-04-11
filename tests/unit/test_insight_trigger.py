"""
Unit tests for InsightTrigger.

Requirements: 2.7, 2.8, 2.9, 6.4, 10.5, 10.6
"""

import pytest
from luma.core.insight.schemas import Insight
from luma.core.insight_moments.insight_trigger import InsightTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_insight(text: str, confidence: float) -> Insight:
    """Create a minimal valid Insight."""
    return Insight(text=text, confidence=confidence, evidence=["ev1"])


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

class TestInsightTriggerConstruction:
    def test_default_threshold_is_0_7(self):
        trigger = InsightTrigger()
        # Insight at exactly 0.7 should pass
        insight = make_insight("hello", 0.7)
        assert trigger.filter_insights([insight]) == [insight]

    def test_surfaced_history_defaults_to_empty_set(self):
        """When surfaced_history is not provided, no insight is excluded by history."""
        trigger = InsightTrigger(confidence_threshold=0.0)
        insight = make_insight("any text", 0.5)
        assert trigger.filter_insights([insight]) == [insight]

    def test_surfaced_history_none_treated_as_empty(self):
        """Passing None explicitly should behave the same as omitting the argument."""
        trigger = InsightTrigger(confidence_threshold=0.0, surfaced_history=None)
        insight = make_insight("any text", 0.5)
        assert trigger.filter_insights([insight]) == [insight]

    @pytest.mark.parametrize("bad_threshold", [-0.001, -1.0, 1.001, 2.0, float("inf")])
    def test_raises_value_error_for_threshold_below_zero(self, bad_threshold):
        with pytest.raises(ValueError):
            InsightTrigger(confidence_threshold=bad_threshold)

    def test_threshold_exactly_0_is_valid(self):
        trigger = InsightTrigger(confidence_threshold=0.0)
        assert trigger is not None

    def test_threshold_exactly_1_is_valid(self):
        trigger = InsightTrigger(confidence_threshold=1.0)
        assert trigger is not None


# ---------------------------------------------------------------------------
# filter_insights — basic behaviour
# ---------------------------------------------------------------------------

class TestFilterInsights:
    def test_empty_input_returns_empty_list(self):
        trigger = InsightTrigger()
        assert trigger.filter_insights([]) == []

    def test_insight_at_exact_threshold_is_included(self):
        """Boundary: confidence == threshold must be included (>= comparison)."""
        threshold = 0.6
        trigger = InsightTrigger(confidence_threshold=threshold)
        insight = make_insight("boundary", threshold)
        result = trigger.filter_insights([insight])
        assert result == [insight]

    def test_insight_below_threshold_is_excluded(self):
        threshold = 0.6
        trigger = InsightTrigger(confidence_threshold=threshold)
        insight = make_insight("low confidence", threshold - 0.001)
        result = trigger.filter_insights([insight])
        assert result == []

    def test_insight_above_threshold_is_included(self):
        trigger = InsightTrigger(confidence_threshold=0.5)
        insight = make_insight("high confidence", 0.9)
        result = trigger.filter_insights([insight])
        assert result == [insight]

    # ------------------------------------------------------------------
    # Surfaced history
    # ------------------------------------------------------------------

    def test_insight_in_surfaced_history_is_excluded(self):
        history = {"already seen"}
        trigger = InsightTrigger(confidence_threshold=0.0, surfaced_history=history)
        insight = make_insight("already seen", 1.0)
        result = trigger.filter_insights([insight])
        assert result == []

    def test_history_exclusion_is_case_sensitive(self):
        """An insight whose text differs only in case from a history entry is NOT excluded."""
        history = {"Already Seen"}
        trigger = InsightTrigger(confidence_threshold=0.0, surfaced_history=history)
        insight = make_insight("already seen", 1.0)  # lowercase — different from history
        result = trigger.filter_insights([insight])
        assert result == [insight]

    def test_history_exclusion_exact_match_only(self):
        """Partial matches do not trigger exclusion."""
        history = {"already"}
        trigger = InsightTrigger(confidence_threshold=0.0, surfaced_history=history)
        insight = make_insight("already seen", 1.0)
        result = trigger.filter_insights([insight])
        assert result == [insight]

    # ------------------------------------------------------------------
    # Order preservation
    # ------------------------------------------------------------------

    def test_output_preserves_input_order(self):
        trigger = InsightTrigger(confidence_threshold=0.0)
        insights = [
            make_insight("first", 0.9),
            make_insight("second", 0.8),
            make_insight("third", 0.7),
        ]
        result = trigger.filter_insights(insights)
        assert result == insights

    def test_output_preserves_order_with_some_excluded(self):
        trigger = InsightTrigger(confidence_threshold=0.5)
        a = make_insight("a", 0.9)
        b = make_insight("b", 0.3)  # excluded
        c = make_insight("c", 0.8)
        result = trigger.filter_insights([a, b, c])
        assert result == [a, c]

    # ------------------------------------------------------------------
    # Object identity (same references, not copies)
    # ------------------------------------------------------------------

    def test_returned_objects_are_same_references(self):
        trigger = InsightTrigger(confidence_threshold=0.0)
        insight = make_insight("ref check", 0.8)
        result = trigger.filter_insights([insight])
        assert result[0] is insight

    # ------------------------------------------------------------------
    # Combined filters
    # ------------------------------------------------------------------

    def test_both_filters_applied_independently(self):
        """An insight can be excluded by confidence OR by history."""
        history = {"seen"}
        trigger = InsightTrigger(confidence_threshold=0.5, surfaced_history=history)

        low_conf = make_insight("new text", 0.3)       # excluded by confidence
        in_history = make_insight("seen", 0.9)          # excluded by history
        passes = make_insight("fresh", 0.8)             # passes both

        result = trigger.filter_insights([low_conf, in_history, passes])
        assert result == [passes]
