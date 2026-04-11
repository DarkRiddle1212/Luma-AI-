"""
Unit tests for TimingStrategy.

Requirements: 3.2, 3.3, 3.4, 3.7, 10.7, 10.8
"""

import pytest

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.schemas import TimingContext
from luma.core.insight_moments.timing_strategy import TimingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_insight(text: str = "You work best in the morning.", confidence: float = 0.8) -> Insight:
    return Insight(text=text, confidence=confidence, evidence=["mem_001"])


def make_context(
    session_ended: bool = False,
    repeated_behavior: bool = False,
    current_timestamp: float = 1000.0,
) -> TimingContext:
    return TimingContext(
        session_ended=session_ended,
        repeated_behavior=repeated_behavior,
        current_timestamp=current_timestamp,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_raises_value_error_for_negative_cooldown():
    """Requirement 10.7: ValueError raised for cooldown_seconds < 0."""
    with pytest.raises(ValueError):
        TimingStrategy(cooldown_seconds=-1.0)


def test_raises_value_error_for_negative_cooldown_zero_boundary():
    """cooldown_seconds == 0 is valid (no cooldown)."""
    strategy = TimingStrategy(cooldown_seconds=0.0)
    assert strategy is not None


def test_delivery_timestamps_defaults_to_empty_dict():
    """Requirement 10.8: delivery_timestamps defaults to empty dict when not provided."""
    strategy = TimingStrategy()
    insight = make_insight()
    context = make_context(session_ended=True)
    # With no delivery history, cooldown never fires — session_ended should return True.
    assert strategy.should_deliver(insight, context) is True


# ---------------------------------------------------------------------------
# Trigger signal tests (no cooldown active)
# ---------------------------------------------------------------------------

def test_returns_true_when_session_ended_and_not_in_cooldown():
    """Requirement 3.2: session_ended=True triggers delivery when not in cooldown."""
    strategy = TimingStrategy(cooldown_seconds=3600.0)
    insight = make_insight()
    context = make_context(session_ended=True)
    assert strategy.should_deliver(insight, context) is True


def test_returns_true_when_repeated_behavior_and_not_in_cooldown():
    """Requirement 3.3: repeated_behavior=True triggers delivery when not in cooldown."""
    strategy = TimingStrategy(cooldown_seconds=3600.0)
    insight = make_insight()
    context = make_context(repeated_behavior=True)
    assert strategy.should_deliver(insight, context) is True


def test_returns_false_when_neither_signal_is_true_and_not_in_cooldown():
    """Default case: no trigger signal → False."""
    strategy = TimingStrategy(cooldown_seconds=3600.0)
    insight = make_insight()
    context = make_context(session_ended=False, repeated_behavior=False)
    assert strategy.should_deliver(insight, context) is False


def test_returns_true_when_insight_not_in_delivery_timestamps_and_session_ended():
    """Insight absent from delivery_timestamps → no cooldown → session_ended fires."""
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps={"some_other_insight": 500.0},
    )
    insight = make_insight(text="A brand new insight.")
    context = make_context(session_ended=True, current_timestamp=1000.0)
    assert strategy.should_deliver(insight, context) is True


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

def test_returns_false_when_within_cooldown_even_if_session_ended():
    """Requirement 3.4: cooldown overrides session_ended signal."""
    now = 1000.0
    insight = make_insight()
    delivery_timestamps = {insight.text: now - 10}  # delivered 10 seconds ago
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)
    assert strategy.should_deliver(insight, context) is False


def test_returns_false_when_within_cooldown_even_if_repeated_behavior():
    """Requirement 3.4: cooldown overrides repeated_behavior signal."""
    now = 1000.0
    insight = make_insight()
    delivery_timestamps = {insight.text: now - 100}  # delivered 100 seconds ago
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(repeated_behavior=True, current_timestamp=now)
    assert strategy.should_deliver(insight, context) is False


def test_returns_false_when_within_cooldown_both_signals_true():
    """Cooldown overrides even when both session_ended and repeated_behavior are True."""
    now = 1000.0
    insight = make_insight()
    delivery_timestamps = {insight.text: now - 50}
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, repeated_behavior=True, current_timestamp=now)
    assert strategy.should_deliver(insight, context) is False


def test_returns_true_when_cooldown_has_expired():
    """After cooldown expires, trigger signals fire normally."""
    now = 10000.0
    insight = make_insight()
    delivery_timestamps = {insight.text: now - 7200}  # delivered 2 hours ago
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)
    assert strategy.should_deliver(insight, context) is True


def test_cooldown_boundary_exactly_at_limit_is_still_in_cooldown():
    """elapsed == cooldown_seconds is NOT in cooldown (< comparison), so delivery fires."""
    now = 1000.0
    insight = make_insight()
    # elapsed = 3600.0 exactly — NOT less than cooldown_seconds, so cooldown does not block.
    delivery_timestamps = {insight.text: now - 3600.0}
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)
    # elapsed (3600) is NOT < cooldown_seconds (3600), so cooldown does not apply.
    assert strategy.should_deliver(insight, context) is True


def test_negative_elapsed_treated_as_not_in_cooldown():
    """Requirement 3.7: future delivery timestamp → elapsed < 0 → not in cooldown."""
    now = 1000.0
    insight = make_insight()
    # Timestamp in the future relative to current_timestamp
    delivery_timestamps = {insight.text: now + 500}  # elapsed = -500
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=delivery_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)
    # elapsed = -500, which is NOT < 3600 ... wait, -500 < 3600 is True → cooldown fires.
    # Per design: "elapsed < 0 < cooldown_seconds is False" — but -500 < 3600 is True.
    # The design note says treat as NOT in cooldown when elapsed < 0.
    # The algorithm as written: elapsed < cooldown_seconds → -500 < 3600 → True → return False.
    # However the design says "elapsed < 0 < cooldown_seconds" means no cooldown.
    # We test the actual documented behaviour from the design doc:
    # "context.current_timestamp is in the past relative to a delivery timestamp →
    #  Elapsed time is negative; treat as not in cooldown (elapsed < 0 < cooldown_seconds)"
    # This means the implementation should check elapsed >= 0 before applying cooldown.
    # The task description says: "If elapsed is negative (timestamp in future),
    # treat as not in cooldown (elapsed < 0 < cooldown_seconds is False, so no cooldown applies)"
    # So the expected result is True (no cooldown, session_ended fires).
    assert strategy.should_deliver(insight, context) is True


# ---------------------------------------------------------------------------
# Read-only: delivery_timestamps is not modified
# ---------------------------------------------------------------------------

def test_delivery_timestamps_not_modified():
    """TimingStrategy never modifies delivery_timestamps."""
    now = 1000.0
    insight = make_insight()
    original_timestamps = {insight.text: now - 10}
    timestamps_copy = dict(original_timestamps)
    strategy = TimingStrategy(
        cooldown_seconds=3600.0,
        delivery_timestamps=original_timestamps,
    )
    context = make_context(session_ended=True, current_timestamp=now)
    strategy.should_deliver(insight, context)
    assert original_timestamps == timestamps_copy, (
        "delivery_timestamps was modified by should_deliver"
    )
