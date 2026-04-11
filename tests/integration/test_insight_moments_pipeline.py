"""
Integration tests for the full Insight Moments Engine pipeline.

Uses real components (no mocks) to verify end-to-end behavior.

Requirements: 1.1, 1.5, 6.3, 10.10
"""

import pytest

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.delivery_manager import DeliveryManager
from luma.core.insight_moments.insight_moments_engine import InsightMomentsEngine
from luma.core.insight_moments.insight_trigger import InsightTrigger
from luma.core.insight_moments.schemas import TimingContext
from luma.core.insight_moments.timing_strategy import TimingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_insight(text: str, confidence: float = 0.8) -> Insight:
    return Insight(text=text, confidence=confidence, evidence=["evidence1"])


def make_engine(
    confidence_threshold: float = 0.7,
    surfaced_history=None,
    cooldown_seconds: float = 3600.0,
    delivery_timestamps=None,
) -> InsightMomentsEngine:
    trigger = InsightTrigger(
        confidence_threshold=confidence_threshold,
        surfaced_history=surfaced_history,
    )
    strategy = TimingStrategy(
        cooldown_seconds=cooldown_seconds,
        delivery_timestamps=delivery_timestamps,
    )
    manager = DeliveryManager()
    return InsightMomentsEngine(trigger, strategy, manager)


def make_context(
    session_ended: bool = True,
    repeated_behavior: bool = False,
    current_timestamp: float = 1000.0,
) -> TimingContext:
    return TimingContext(
        session_ended=session_ended,
        repeated_behavior=repeated_behavior,
        current_timestamp=current_timestamp,
    )


# ---------------------------------------------------------------------------
# Test: full pipeline — insights above threshold, session_ended=True → payloads returned
# ---------------------------------------------------------------------------

def test_full_pipeline_above_threshold_session_ended():
    """
    Insights above confidence threshold with session_ended=True produce payloads.
    """
    engine = make_engine(confidence_threshold=0.7)
    context = make_context(session_ended=True)

    insights = [
        make_insight("You tend to work best in the morning.", confidence=0.9),
        make_insight("You often revisit tasks after breaks.", confidence=0.75),
    ]

    result = engine.generate_moments(insights, context)

    assert len(result) == 2
    messages = [p.message for p in result]
    assert "You tend to work best in the morning." in messages
    assert "You often revisit tasks after breaks." in messages
    for payload in result:
        assert payload.payload_type == "insight_moment"
        assert payload.confidence > 0.0


# ---------------------------------------------------------------------------
# Test: full pipeline — insights below confidence threshold → empty output
# ---------------------------------------------------------------------------

def test_full_pipeline_below_confidence_threshold():
    """
    Insights below the confidence threshold are filtered out, producing no payloads.
    """
    engine = make_engine(confidence_threshold=0.7)
    context = make_context(session_ended=True)

    insights = [
        make_insight("Low confidence insight.", confidence=0.3),
        make_insight("Another low confidence insight.", confidence=0.5),
    ]

    result = engine.generate_moments(insights, context)

    assert result == []


# ---------------------------------------------------------------------------
# Test: full pipeline — insights in surfaced_history → empty output
# ---------------------------------------------------------------------------

def test_full_pipeline_insights_in_surfaced_history():
    """
    Insights whose text is in surfaced_history are excluded, producing no payloads.
    """
    insight_text = "You work best in the morning."
    engine = make_engine(
        confidence_threshold=0.0,
        surfaced_history={insight_text},
    )
    context = make_context(session_ended=True)

    insights = [make_insight(insight_text, confidence=0.9)]

    result = engine.generate_moments(insights, context)

    assert result == []


# ---------------------------------------------------------------------------
# Test: full pipeline — insights in cooldown → empty output
# ---------------------------------------------------------------------------

def test_full_pipeline_insights_in_cooldown():
    """
    Insights delivered recently (within cooldown window) are not re-delivered.
    """
    insight_text = "You work best in the morning."
    now = 1000.0
    # Delivered 60 seconds ago, cooldown is 3600 seconds
    delivery_timestamps = {insight_text: now - 60}

    engine = make_engine(
        confidence_threshold=0.0,
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)

    insights = [make_insight(insight_text, confidence=0.9)]

    result = engine.generate_moments(insights, context)

    assert result == []


# ---------------------------------------------------------------------------
# Test: deduplication round-trip
# ---------------------------------------------------------------------------

def test_deduplication_round_trip():
    """
    Running the pipeline twice with the first run's messages added to surfaced_history
    produces an empty result on the second run.
    """
    insights = [
        make_insight("You work best in the morning.", confidence=0.9),
        make_insight("You often revisit tasks after breaks.", confidence=0.8),
    ]
    context = make_context(session_ended=True)

    # First run — no history
    engine_first = make_engine(confidence_threshold=0.7)
    first_result = engine_first.generate_moments(insights, context)

    assert len(first_result) > 0, "First run should produce payloads"

    # Collect messages from first run and use as surfaced_history for second run
    surfaced = {payload.message for payload in first_result}

    engine_second = make_engine(
        confidence_threshold=0.7,
        surfaced_history=surfaced,
    )
    second_result = engine_second.generate_moments(insights, context)

    assert second_result == [], (
        "Second run with first run's messages in surfaced_history should return []"
    )
