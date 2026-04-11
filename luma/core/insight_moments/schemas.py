"""
Insight Moments Engine Data Schemas.

Defines TimingContext, TriggerDecision, InsightMoment, and DeliveryPayload models.
Uses Pydantic if available, otherwise dataclasses with __post_init__ validation —
matching the dual-path pattern in luma/core/insight/schemas.py.
"""

from typing import List

try:
    from pydantic import BaseModel, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False

_VALID_DELIVERY_TYPES = {"subtle_notification", "inline_suggestion", "highlighted_insight"}

if _USE_PYDANTIC:
    class TimingContext(BaseModel):
        """Captures the current session state for timing evaluation."""

        session_ended: bool
        repeated_behavior: bool
        current_timestamp: float

    class TriggerDecision(BaseModel):
        """Records the InsightTrigger's evaluation result for a given insight."""

        insight_text: str
        passed: bool
        reason: str

    class InsightMoment(BaseModel):
        """A structured output model representing a single user-facing insight event."""

        payload_type: str
        message: str
        delivery_type: str
        confidence: float
        evidence: List[str]

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

        @field_validator("delivery_type")
        @classmethod
        def delivery_type_valid(cls, v: str) -> str:
            if v not in _VALID_DELIVERY_TYPES:
                raise ValueError(
                    f"delivery_type must be one of {_VALID_DELIVERY_TYPES}, got {v!r}"
                )
            return v

        @field_validator("evidence")
        @classmethod
        def evidence_non_empty(cls, v: List[str]) -> List[str]:
            if not v:
                raise ValueError("evidence must be a non-empty list")
            return v

    class DeliveryPayload(BaseModel):
        """The final structured output produced by the DeliveryManager."""

        payload_type: str
        message: str
        type: str
        confidence: float

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

        @field_validator("type")
        @classmethod
        def type_valid(cls, v: str) -> str:
            if v not in _VALID_DELIVERY_TYPES:
                raise ValueError(
                    f"type must be one of {_VALID_DELIVERY_TYPES}, got {v!r}"
                )
            return v

else:
    from dataclasses import dataclass

    @dataclass
    class TimingContext:
        """Captures the current session state for timing evaluation."""

        session_ended: bool
        repeated_behavior: bool
        current_timestamp: float

    @dataclass
    class TriggerDecision:
        """Records the InsightTrigger's evaluation result for a given insight."""

        insight_text: str
        passed: bool
        reason: str

    @dataclass
    class InsightMoment:
        """A structured output model representing a single user-facing insight event."""

        payload_type: str
        message: str
        delivery_type: str
        confidence: float
        evidence: List[str]

        def __post_init__(self) -> None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"confidence must be in [0.0, 1.0], got {self.confidence}"
                )
            if self.delivery_type not in _VALID_DELIVERY_TYPES:
                raise ValueError(
                    f"delivery_type must be one of {_VALID_DELIVERY_TYPES}, got {self.delivery_type!r}"
                )
            if not self.evidence:
                raise ValueError("evidence must be a non-empty list")

    @dataclass
    class DeliveryPayload:
        """The final structured output produced by the DeliveryManager."""

        payload_type: str
        message: str
        type: str
        confidence: float

        def __post_init__(self) -> None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"confidence must be in [0.0, 1.0], got {self.confidence}"
                )
            if self.type not in _VALID_DELIVERY_TYPES:
                raise ValueError(
                    f"type must be one of {_VALID_DELIVERY_TYPES}, got {self.type!r}"
                )
