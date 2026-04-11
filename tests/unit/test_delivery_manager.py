"""
Unit tests for DeliveryManager.

Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 10.9
"""

import pytest

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.delivery_manager import DeliveryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_insight(text: str = "Test insight", confidence: float = 0.75) -> Insight:
    return Insight(text=text, confidence=confidence, evidence=["mem-1"])


# ---------------------------------------------------------------------------
# payload_type is always "insight_moment"
# ---------------------------------------------------------------------------

def test_payload_type_is_insight_moment():
    """payload_type must always be 'insight_moment' regardless of confidence."""
    manager = DeliveryManager()
    for confidence in [0.0, 0.5, 0.70, 0.85, 1.0]:
        insight = make_insight(confidence=confidence)
        payload = manager.format_delivery(insight)
        assert payload.payload_type == "insight_moment", (
            f"Expected payload_type='insight_moment', got {payload.payload_type!r} "
            f"for confidence={confidence}"
        )


# ---------------------------------------------------------------------------
# message equals insight's text
# ---------------------------------------------------------------------------

def test_message_equals_insight_text():
    """message field must equal the insight's text field exactly."""
    manager = DeliveryManager()
    insight = make_insight(text="You have been very productive this week.")
    payload = manager.format_delivery(insight)
    assert payload.message == insight.text


# ---------------------------------------------------------------------------
# Delivery type classification
# ---------------------------------------------------------------------------

def test_confidence_0_90_is_highlighted_insight():
    """confidence=0.90 (above highlighted_threshold=0.85) → highlighted_insight."""
    manager = DeliveryManager()
    payload = manager.format_delivery(make_insight(confidence=0.90))
    assert payload.type == "highlighted_insight"


def test_confidence_0_85_is_highlighted_insight_boundary():
    """confidence=0.85 exactly equals highlighted_threshold → highlighted_insight (>= comparison)."""
    manager = DeliveryManager()
    payload = manager.format_delivery(make_insight(confidence=0.85))
    assert payload.type == "highlighted_insight"


def test_confidence_0_80_is_inline_suggestion():
    """confidence=0.80 (between inline and highlighted thresholds) → inline_suggestion."""
    manager = DeliveryManager()
    payload = manager.format_delivery(make_insight(confidence=0.80))
    assert payload.type == "inline_suggestion"


def test_confidence_0_70_is_inline_suggestion_boundary():
    """confidence=0.70 exactly equals inline_threshold → inline_suggestion (>= comparison)."""
    manager = DeliveryManager()
    payload = manager.format_delivery(make_insight(confidence=0.70))
    assert payload.type == "inline_suggestion"


def test_confidence_0_50_is_subtle_notification():
    """confidence=0.50 (below inline_threshold=0.70) → subtle_notification."""
    manager = DeliveryManager()
    payload = manager.format_delivery(make_insight(confidence=0.50))
    assert payload.type == "subtle_notification"


# ---------------------------------------------------------------------------
# Confidence is preserved exactly
# ---------------------------------------------------------------------------

def test_confidence_preserved_exactly():
    """The confidence field of the payload must equal the insight's confidence exactly."""
    manager = DeliveryManager()
    for confidence in [0.0, 0.123456789, 0.70, 0.85, 1.0]:
        insight = make_insight(confidence=confidence)
        payload = manager.format_delivery(insight)
        assert payload.confidence == confidence, (
            f"Expected confidence={confidence}, got {payload.confidence}"
        )


# ---------------------------------------------------------------------------
# Input Insight is not modified
# ---------------------------------------------------------------------------

def test_input_insight_not_modified():
    """format_delivery must not modify any field of the input Insight."""
    manager = DeliveryManager()
    original_text = "Original insight text"
    original_confidence = 0.75
    original_evidence = ["mem-1", "mem-2"]
    insight = Insight(
        text=original_text,
        confidence=original_confidence,
        evidence=original_evidence,
    )
    manager.format_delivery(insight)
    assert insight.text == original_text
    assert insight.confidence == original_confidence
    assert list(insight.evidence) == original_evidence


# ---------------------------------------------------------------------------
# ValueError when highlighted_threshold <= inline_threshold
# ---------------------------------------------------------------------------

def test_value_error_when_highlighted_equals_inline():
    """Raise ValueError when highlighted_threshold == inline_threshold."""
    with pytest.raises(ValueError):
        DeliveryManager(highlighted_threshold=0.70, inline_threshold=0.70)


def test_value_error_when_highlighted_less_than_inline():
    """Raise ValueError when highlighted_threshold < inline_threshold."""
    with pytest.raises(ValueError):
        DeliveryManager(highlighted_threshold=0.60, inline_threshold=0.70)


def test_valid_construction_with_custom_thresholds():
    """No error when highlighted_threshold > inline_threshold."""
    manager = DeliveryManager(highlighted_threshold=0.90, inline_threshold=0.60)
    assert manager is not None
