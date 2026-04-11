"""
Unit tests for luma/core/insight_moments/schemas.py.

Tests validation for DeliveryPayload and InsightMoment models,
covering both the Pydantic and dataclass paths.

Requirements: 5.5, 5.6, 5.7
"""

import importlib
import sys
import types
import pytest

from luma.core.insight_moments.schemas import (
    DeliveryPayload,
    InsightMoment,
    TimingContext,
    TriggerDecision,
    _USE_PYDANTIC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_delivery_payload(**kwargs):
    defaults = dict(
        payload_type="insight_moment",
        message="Test message",
        type="subtle_notification",
        confidence=0.5,
    )
    defaults.update(kwargs)
    return DeliveryPayload(**defaults)


def make_insight_moment(**kwargs):
    defaults = dict(
        payload_type="insight_moment",
        message="Test message",
        delivery_type="subtle_notification",
        confidence=0.5,
        evidence=["mem_001"],
    )
    defaults.update(kwargs)
    return InsightMoment(**defaults)


# ---------------------------------------------------------------------------
# DeliveryPayload — confidence validation (Requirement 5.6)
# ---------------------------------------------------------------------------

class TestDeliveryPayloadConfidenceValidation:
    def test_valid_confidence_lower_bound(self):
        payload = make_delivery_payload(confidence=0.0)
        assert payload.confidence == 0.0

    def test_valid_confidence_upper_bound(self):
        payload = make_delivery_payload(confidence=1.0)
        assert payload.confidence == 1.0

    def test_valid_confidence_midpoint(self):
        payload = make_delivery_payload(confidence=0.5)
        assert payload.confidence == 0.5

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(confidence=1.01)

    def test_confidence_negative_large_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(confidence=-5.0)

    def test_confidence_large_positive_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(confidence=2.0)


# ---------------------------------------------------------------------------
# DeliveryPayload — type validation (Requirement 5.7)
# ---------------------------------------------------------------------------

class TestDeliveryPayloadTypeValidation:
    @pytest.mark.parametrize("valid_type", [
        "subtle_notification",
        "inline_suggestion",
        "highlighted_insight",
    ])
    def test_valid_type_accepted(self, valid_type):
        payload = make_delivery_payload(type=valid_type)
        assert payload.type == valid_type

    def test_invalid_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(type="unknown_type")

    def test_empty_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(type="")

    def test_mixed_case_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(type="Subtle_Notification")

    def test_partial_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_delivery_payload(type="subtle")


# ---------------------------------------------------------------------------
# InsightMoment — confidence validation (Requirement 5.5)
# ---------------------------------------------------------------------------

class TestInsightMomentConfidenceValidation:
    def test_valid_confidence_lower_bound(self):
        moment = make_insight_moment(confidence=0.0)
        assert moment.confidence == 0.0

    def test_valid_confidence_upper_bound(self):
        moment = make_insight_moment(confidence=1.0)
        assert moment.confidence == 1.0

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_insight_moment(confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_insight_moment(confidence=1.01)


# ---------------------------------------------------------------------------
# InsightMoment — delivery_type validation (Requirement 5.7)
# ---------------------------------------------------------------------------

class TestInsightMomentDeliveryTypeValidation:
    @pytest.mark.parametrize("valid_type", [
        "subtle_notification",
        "inline_suggestion",
        "highlighted_insight",
    ])
    def test_valid_delivery_type_accepted(self, valid_type):
        moment = make_insight_moment(delivery_type=valid_type)
        assert moment.delivery_type == valid_type

    def test_invalid_delivery_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_insight_moment(delivery_type="bad_type")

    def test_empty_delivery_type_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_insight_moment(delivery_type="")


# ---------------------------------------------------------------------------
# InsightMoment — evidence validation (Requirement 5.5)
# ---------------------------------------------------------------------------

class TestInsightMomentEvidenceValidation:
    def test_non_empty_evidence_accepted(self):
        moment = make_insight_moment(evidence=["mem_001", "mem_002"])
        assert moment.evidence == ["mem_001", "mem_002"]

    def test_empty_evidence_raises(self):
        with pytest.raises((ValueError, Exception)):
            make_insight_moment(evidence=[])


# ---------------------------------------------------------------------------
# TimingContext and TriggerDecision — no validation, just construction
# ---------------------------------------------------------------------------

class TestTimingContextConstruction:
    def test_creates_successfully(self):
        ctx = TimingContext(
            session_ended=True,
            repeated_behavior=False,
            current_timestamp=1234567890.0,
        )
        assert ctx.session_ended is True
        assert ctx.repeated_behavior is False
        assert ctx.current_timestamp == 1234567890.0


class TestTriggerDecisionConstruction:
    def test_creates_successfully(self):
        decision = TriggerDecision(
            insight_text="Some insight",
            passed=True,
            reason="Confidence above threshold",
        )
        assert decision.insight_text == "Some insight"
        assert decision.passed is True
        assert decision.reason == "Confidence above threshold"


# ---------------------------------------------------------------------------
# Dual-path: test the dataclass path explicitly by monkey-patching
# ---------------------------------------------------------------------------

class TestDataclassPath:
    """
    Force the dataclass code path by temporarily removing pydantic from sys.modules
    and reimporting the schemas module. This ensures both paths are exercised.
    """

    def _get_dataclass_module(self):
        """Return a fresh import of schemas with Pydantic disabled."""
        # Save originals
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if "pydantic" in key or "insight_moments.schemas" in key or "insight_moments" == key:
                saved_modules[key] = sys.modules.pop(key)

        # Insert a fake pydantic that raises ImportError on import
        fake_pydantic = types.ModuleType("pydantic")

        def _raise(*args, **kwargs):
            raise ImportError("pydantic not available")

        fake_pydantic.BaseModel = None
        fake_pydantic.field_validator = None

        # Remove pydantic so the try/except in schemas falls to the except branch
        pydantic_keys = [k for k in sys.modules if "pydantic" in k]
        for k in pydantic_keys:
            sys.modules.pop(k, None)

        # Temporarily block pydantic import
        sys.modules["pydantic"] = None  # type: ignore[assignment]

        try:
            import importlib
            mod = importlib.import_module("luma.core.insight_moments.schemas")
            # Force reload to pick up the patched pydantic state
            mod = importlib.reload(mod)
            return mod
        finally:
            # Restore everything
            sys.modules.pop("pydantic", None)
            sys.modules.pop("luma.core.insight_moments.schemas", None)
            for key, val in saved_modules.items():
                sys.modules[key] = val
            # Reload the original module back
            importlib.import_module("luma.core.insight_moments.schemas")

    def test_dataclass_delivery_payload_confidence_invalid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        with pytest.raises(ValueError):
            mod.DeliveryPayload(
                payload_type="insight_moment",
                message="msg",
                type="subtle_notification",
                confidence=1.5,
            )

    def test_dataclass_delivery_payload_type_invalid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        with pytest.raises(ValueError):
            mod.DeliveryPayload(
                payload_type="insight_moment",
                message="msg",
                type="not_valid",
                confidence=0.5,
            )

    def test_dataclass_insight_moment_confidence_invalid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        with pytest.raises(ValueError):
            mod.InsightMoment(
                payload_type="insight_moment",
                message="msg",
                delivery_type="subtle_notification",
                confidence=-0.1,
                evidence=["e1"],
            )

    def test_dataclass_insight_moment_delivery_type_invalid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        with pytest.raises(ValueError):
            mod.InsightMoment(
                payload_type="insight_moment",
                message="msg",
                delivery_type="bad_type",
                confidence=0.5,
                evidence=["e1"],
            )

    def test_dataclass_insight_moment_empty_evidence_invalid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        with pytest.raises(ValueError):
            mod.InsightMoment(
                payload_type="insight_moment",
                message="msg",
                delivery_type="subtle_notification",
                confidence=0.5,
                evidence=[],
            )

    def test_dataclass_delivery_payload_valid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        payload = mod.DeliveryPayload(
            payload_type="insight_moment",
            message="msg",
            type="highlighted_insight",
            confidence=0.9,
        )
        assert payload.confidence == 0.9
        assert payload.type == "highlighted_insight"

    def test_dataclass_insight_moment_valid(self):
        mod = self._get_dataclass_module()
        if mod._USE_PYDANTIC:
            pytest.skip("Could not force dataclass path in this environment")
        moment = mod.InsightMoment(
            payload_type="insight_moment",
            message="msg",
            delivery_type="inline_suggestion",
            confidence=0.75,
            evidence=["e1"],
        )
        assert moment.confidence == 0.75
        assert moment.delivery_type == "inline_suggestion"
