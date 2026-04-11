"""
Unit tests for InsightMomentsEngine.

Tests the orchestration logic using mock sub-components to verify
the pipeline behavior in isolation.

Requirements: 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 10.4, 10.10
"""

from unittest.mock import MagicMock, call
import pytest

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.insight_moments_engine import InsightMomentsEngine
from luma.core.insight_moments.schemas import DeliveryPayload, TimingContext


def make_insight(text: str, confidence: float = 0.8) -> Insight:
    """Helper to create a valid Insight."""
    return Insight(text=text, confidence=confidence, evidence=["evidence1"])


def make_payload(message: str, confidence: float = 0.8) -> DeliveryPayload:
    """Helper to create a valid DeliveryPayload."""
    return DeliveryPayload(
        payload_type="insight_moment",
        message=message,
        type="inline_suggestion",
        confidence=confidence,
    )


def make_context(session_ended: bool = True) -> TimingContext:
    """Helper to create a TimingContext."""
    return TimingContext(
        session_ended=session_ended,
        repeated_behavior=False,
        current_timestamp=1000.0,
    )


def make_engine(trigger=None, strategy=None, manager=None):
    """Helper to create an InsightMomentsEngine with mock sub-components."""
    trigger = trigger or MagicMock()
    strategy = strategy or MagicMock()
    manager = manager or MagicMock()
    return InsightMomentsEngine(trigger, strategy, manager), trigger, strategy, manager


# ---------------------------------------------------------------------------
# Test: empty input returns [] immediately
# ---------------------------------------------------------------------------

def test_empty_insights_returns_empty_list():
    """Returns [] for empty insight list without calling any sub-components."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    result = engine.generate_moments([], context)

    assert result == []
    trigger.filter_insights.assert_not_called()
    strategy.should_deliver.assert_not_called()
    manager.format_delivery.assert_not_called()


# ---------------------------------------------------------------------------
# Test: filter_insights is called with the full input list
# ---------------------------------------------------------------------------

def test_filter_insights_called_with_full_input():
    """filter_insights receives the complete input list."""
    engine, trigger, strategy, manager = make_engine()
    trigger.filter_insights.return_value = []

    insights = [make_insight("a"), make_insight("b"), make_insight("c")]
    context = make_context()

    engine.generate_moments(insights, context)

    trigger.filter_insights.assert_called_once_with(insights)


# ---------------------------------------------------------------------------
# Test: should_deliver is called for each filtered insight
# ---------------------------------------------------------------------------

def test_should_deliver_called_for_each_filtered_insight():
    """should_deliver is called once per insight returned by filter_insights."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insight_a = make_insight("a")
    insight_b = make_insight("b")
    filtered = [insight_a, insight_b]

    trigger.filter_insights.return_value = filtered
    strategy.should_deliver.return_value = False

    engine.generate_moments([insight_a, insight_b], context)

    assert strategy.should_deliver.call_count == 2
    strategy.should_deliver.assert_any_call(insight_a, context)
    strategy.should_deliver.assert_any_call(insight_b, context)


# ---------------------------------------------------------------------------
# Test: format_delivery is called only for approved insights
# ---------------------------------------------------------------------------

def test_format_delivery_called_only_for_approved_insights():
    """format_delivery is called only when should_deliver returns True."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insight_a = make_insight("a")
    insight_b = make_insight("b")
    insight_c = make_insight("c")
    filtered = [insight_a, insight_b, insight_c]

    trigger.filter_insights.return_value = filtered
    # Only insight_a and insight_c are approved
    strategy.should_deliver.side_effect = lambda insight, ctx: insight.text in ("a", "c")

    payload_a = make_payload("a")
    payload_c = make_payload("c")
    manager.format_delivery.side_effect = lambda insight: (
        payload_a if insight.text == "a" else payload_c
    )

    result = engine.generate_moments([insight_a, insight_b, insight_c], context)

    assert manager.format_delivery.call_count == 2
    manager.format_delivery.assert_any_call(insight_a)
    manager.format_delivery.assert_any_call(insight_c)
    assert result == [payload_a, payload_c]


# ---------------------------------------------------------------------------
# Test: format_delivery is NOT called for rejected insights
# ---------------------------------------------------------------------------

def test_format_delivery_not_called_for_rejected_insights():
    """format_delivery is never called for insights where should_deliver returns False."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insight_a = make_insight("a")
    insight_b = make_insight("b")  # will be rejected

    trigger.filter_insights.return_value = [insight_a, insight_b]
    strategy.should_deliver.side_effect = lambda insight, ctx: insight.text == "a"

    payload_a = make_payload("a")
    manager.format_delivery.return_value = payload_a

    engine.generate_moments([insight_a, insight_b], context)

    # format_delivery should only be called for insight_a, not insight_b
    for mock_call in manager.format_delivery.call_args_list:
        assert mock_call != call(insight_b), (
            "format_delivery was called for rejected insight_b"
        )
    manager.format_delivery.assert_called_once_with(insight_a)


# ---------------------------------------------------------------------------
# Test: payloads are returned in the order insights were approved
# ---------------------------------------------------------------------------

def test_payloads_returned_in_approval_order():
    """Returned payloads preserve the order in which insights were approved."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insight_a = make_insight("a")
    insight_b = make_insight("b")
    insight_c = make_insight("c")

    trigger.filter_insights.return_value = [insight_a, insight_b, insight_c]
    strategy.should_deliver.return_value = True

    payload_a = make_payload("a")
    payload_b = make_payload("b")
    payload_c = make_payload("c")

    def format_side_effect(insight):
        return {"a": payload_a, "b": payload_b, "c": payload_c}[insight.text]

    manager.format_delivery.side_effect = format_side_effect

    result = engine.generate_moments([insight_a, insight_b, insight_c], context)

    assert result == [payload_a, payload_b, payload_c]


# ---------------------------------------------------------------------------
# Test: input insight fields are not modified after the call
# ---------------------------------------------------------------------------

def test_input_insight_fields_not_modified():
    """generate_moments never modifies text, confidence, or evidence of input insights."""
    insight = make_insight("original text", confidence=0.75)
    original_text = insight.text
    original_confidence = insight.confidence
    original_evidence = list(insight.evidence)

    trigger = MagicMock()
    strategy = MagicMock()
    manager = MagicMock()

    trigger.filter_insights.return_value = [insight]
    strategy.should_deliver.return_value = True
    manager.format_delivery.return_value = make_payload("original text", 0.75)

    engine = InsightMomentsEngine(trigger, strategy, manager)
    engine.generate_moments([insight], make_context())

    assert insight.text == original_text
    assert insight.confidence == original_confidence
    assert list(insight.evidence) == original_evidence


# ---------------------------------------------------------------------------
# Test: all insights rejected → empty list returned
# ---------------------------------------------------------------------------

def test_all_insights_rejected_returns_empty():
    """Returns [] when all filtered insights are rejected by should_deliver."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insights = [make_insight("a"), make_insight("b")]
    trigger.filter_insights.return_value = insights
    strategy.should_deliver.return_value = False

    result = engine.generate_moments(insights, context)

    assert result == []
    manager.format_delivery.assert_not_called()


# ---------------------------------------------------------------------------
# Test: filter returns empty → should_deliver and format_delivery not called
# ---------------------------------------------------------------------------

def test_filter_returns_empty_skips_remaining_steps():
    """When filter_insights returns [], should_deliver and format_delivery are not called."""
    engine, trigger, strategy, manager = make_engine()
    context = make_context()

    insights = [make_insight("a")]
    trigger.filter_insights.return_value = []

    result = engine.generate_moments(insights, context)

    assert result == []
    strategy.should_deliver.assert_not_called()
    manager.format_delivery.assert_not_called()
