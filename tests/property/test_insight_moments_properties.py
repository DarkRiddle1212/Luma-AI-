"""
Property-based tests for the Insight Moments Engine — InsightTrigger properties.

Properties 1–4 cover InsightTrigger behaviour.
Each test is tagged: Feature: insight-moments-engine, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

from hypothesis import given, settings, strategies as st, HealthCheck

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.insight_trigger import InsightTrigger
from luma.core.insight_moments.schemas import TimingContext


# ---------------------------------------------------------------------------
# Shared strategy
# ---------------------------------------------------------------------------

def insight_strategy():
    """Generate valid Insight objects with arbitrary text, confidence, and evidence."""
    return st.builds(
        Insight,
        text=st.text(min_size=1, max_size=200),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        evidence=st.lists(st.text(min_size=1), min_size=1, max_size=10),
    )


# ---------------------------------------------------------------------------
# Property 1: Trigger output is a subset of input
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 1: Trigger output is a subset of input
@given(insights=st.lists(insight_strategy(), max_size=20))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_trigger_output_is_subset_of_input(insights):
    """
    **Validates: Requirements 2.4, 11.2**

    For any list of Insight objects, every insight in the returned list SHALL
    also be present in the input list (by object identity).
    """
    trigger = InsightTrigger(confidence_threshold=0.0)
    result = trigger.filter_insights(insights)
    for r in result:
        assert r in insights, (
            f"Insight {r.text!r} in output but not found in input list"
        )


# ---------------------------------------------------------------------------
# Property 2: Trigger respects confidence threshold
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 2: Trigger respects confidence threshold
@given(
    insights=st.lists(insight_strategy(), max_size=20),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_trigger_respects_confidence_threshold(insights, threshold):
    """
    **Validates: Requirements 2.2, 2.5, 11.3**

    For any list of Insight objects and any configured confidence_threshold,
    every insight returned by filter_insights() SHALL have
    confidence >= confidence_threshold, and no insight with
    confidence < confidence_threshold SHALL appear in the output.
    """
    trigger = InsightTrigger(confidence_threshold=threshold)
    result = trigger.filter_insights(insights)
    for r in result:
        assert r.confidence >= threshold, (
            f"Insight with confidence={r.confidence} passed threshold={threshold}"
        )
    # Also verify that no excluded insight had confidence >= threshold
    # (i.e., the only reason an insight is absent is low confidence or history)
    result_ids = {id(r) for r in result}
    for insight in insights:
        if insight.confidence >= threshold and id(insight) not in result_ids:
            # It must have been excluded by history — but we used no history here,
            # so this should never happen.
            assert False, (
                f"Insight with confidence={insight.confidence} >= threshold={threshold} "
                f"was unexpectedly excluded (no surfaced_history configured)"
            )


# ---------------------------------------------------------------------------
# Property 3: Trigger excludes surfaced history
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 3: Trigger excludes surfaced history
@given(
    insights=st.lists(insight_strategy(), min_size=1, max_size=20),
    extra_history=st.sets(st.text(min_size=1, max_size=50), max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_trigger_excludes_surfaced_history(insights, extra_history):
    """
    **Validates: Requirements 2.3, 6.1, 11.4**

    For any list of Insight objects and any surfaced_history set, no insight
    returned by filter_insights() SHALL have a text field present in
    surfaced_history.
    """
    # Build history from the first insight's text plus any extra strings
    history = {insights[0].text} | extra_history
    trigger = InsightTrigger(confidence_threshold=0.0, surfaced_history=history)
    result = trigger.filter_insights(insights)
    for r in result:
        assert r.text not in history, (
            f"Insight with text={r.text!r} appeared in output despite being in surfaced_history"
        )


# ---------------------------------------------------------------------------
# Property 4: Trigger determinism
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 4: Trigger determinism — same inputs produce same outputs
@given(
    insights=st.lists(insight_strategy(), max_size=20),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_trigger_determinism(insights, threshold):
    """
    **Validates: Requirements 2.9, 11.5**

    Calling filter_insights() twice with identical inputs SHALL produce
    identical output lists (same elements in the same order).
    """
    trigger = InsightTrigger(confidence_threshold=threshold)
    result1 = trigger.filter_insights(insights)
    result2 = trigger.filter_insights(insights)

    assert len(result1) == len(result2), (
        f"Non-deterministic result count: {len(result1)} vs {len(result2)}"
    )
    for r1, r2 in zip(result1, result2):
        assert r1 is r2, (
            f"Non-deterministic result: got different object references on second call"
        )


# ---------------------------------------------------------------------------
# Property 7: Cooldown always overrides trigger signals
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 7: Cooldown always overrides trigger signals
@given(insight=insight_strategy())
@settings(max_examples=100)
def test_cooldown_overrides_signals(insight):
    """
    **Validates: Requirements 3.4, 6.2**

    For any Insight whose text was last delivered within cooldown_seconds of
    context.current_timestamp, TimingStrategy.should_deliver() SHALL return False
    regardless of the values of context.session_ended and context.repeated_behavior.
    """
    from luma.core.insight_moments.timing_strategy import TimingStrategy
    from luma.core.insight_moments.schemas import TimingContext

    now = 1000.0
    delivery_timestamps = {insight.text: now - 10}  # delivered 10s ago
    strategy = TimingStrategy(cooldown_seconds=3600.0, delivery_timestamps=delivery_timestamps)
    context = TimingContext(session_ended=True, repeated_behavior=True, current_timestamp=now)
    assert strategy.should_deliver(insight, context) is False


# ---------------------------------------------------------------------------
# Property 5: Delivery confidence round-trip
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 5: Delivery confidence round-trip
@given(insight=insight_strategy())
@settings(max_examples=100)
def test_delivery_confidence_roundtrip(insight):
    """
    **Validates: Requirements 4.2, 4.8, 12.2**

    For any Insight object passed to DeliveryManager.format_delivery(), the
    confidence field of the resulting DeliveryPayload SHALL equal the confidence
    field of the input Insight exactly (no rounding, no transformation).
    """
    from luma.core.insight_moments.delivery_manager import DeliveryManager

    manager = DeliveryManager()
    payload = manager.format_delivery(insight)
    assert payload.confidence == insight.confidence, (
        f"Confidence round-trip failed: input={insight.confidence}, "
        f"output={payload.confidence}"
    )


# ---------------------------------------------------------------------------
# Property 6: Delivery type classification is exhaustive and correct
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 6: Delivery type classification is exhaustive and correct
@given(insight=insight_strategy())
@settings(max_examples=100)
def test_delivery_type_classification(insight):
    """
    **Validates: Requirements 4.4, 4.5, 4.6, 12.3, 12.4, 12.5**

    For any Insight object passed to DeliveryManager.format_delivery():
    - confidence >= 0.85 → "highlighted_insight"
    - 0.70 <= confidence < 0.85 → "inline_suggestion"
    - confidence < 0.70 → "subtle_notification"

    Every valid confidence value maps to exactly one delivery type.
    """
    from luma.core.insight_moments.delivery_manager import DeliveryManager

    manager = DeliveryManager(highlighted_threshold=0.85, inline_threshold=0.70)
    payload = manager.format_delivery(insight)

    if insight.confidence >= 0.85:
        assert payload.type == "highlighted_insight", (
            f"Expected 'highlighted_insight' for confidence={insight.confidence}, "
            f"got {payload.type!r}"
        )
    elif insight.confidence >= 0.70:
        assert payload.type == "inline_suggestion", (
            f"Expected 'inline_suggestion' for confidence={insight.confidence}, "
            f"got {payload.type!r}"
        )
    else:
        assert payload.type == "subtle_notification", (
            f"Expected 'subtle_notification' for confidence={insight.confidence}, "
            f"got {payload.type!r}"
        )


# ---------------------------------------------------------------------------
# Property 8: Insight immutability through the full pipeline
# ---------------------------------------------------------------------------

# Feature: insight-moments-engine, Property 8: Insight immutability through the full pipeline
@given(insights=st.lists(insight_strategy(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_insight_immutability(insights):
    """
    **Validates: Requirements 1.7, 9.1, 9.2, 9.3, 9.4**

    For any list of Insight objects passed to generate_moments(), the text,
    confidence, and evidence fields of every input Insight SHALL be identical
    before and after the call.
    """
    import copy
    from luma.core.insight_moments.insight_moments_engine import InsightMomentsEngine
    from luma.core.insight_moments.timing_strategy import TimingStrategy
    from luma.core.insight_moments.delivery_manager import DeliveryManager

    snapshots = [(i.text, i.confidence, list(i.evidence)) for i in insights]
    # Build engine with permissive settings so some insights pass
    trigger = InsightTrigger(confidence_threshold=0.0)
    strategy = TimingStrategy(cooldown_seconds=0.0)
    manager = DeliveryManager()
    engine = InsightMomentsEngine(trigger, strategy, manager)
    context = TimingContext(session_ended=True, repeated_behavior=False, current_timestamp=9999.0)
    engine.generate_moments(insights, context)
    for i, (text, conf, evidence) in zip(insights, snapshots):
        assert i.text == text
        assert i.confidence == conf
        assert list(i.evidence) == evidence
