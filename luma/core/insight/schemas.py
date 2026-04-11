"""
Insight Engine Data Schemas.

Defines PatternResult, TrendResult, Insight, and InsightReport models.
Uses Pydantic if available, otherwise dataclasses with __post_init__ validation.
"""

from typing import List

try:
    from pydantic import BaseModel, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


if _USE_PYDANTIC:
    class PatternResult(BaseModel):
        """A detected pattern with type, value, frequency, confidence, and evidence."""

        pattern_type: str
        pattern: str
        frequency: int
        confidence: float
        evidence: List[str]

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

        @field_validator("frequency")
        @classmethod
        def frequency_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError(f"frequency must be a positive integer, got {v}")
            return v

        @field_validator("evidence")
        @classmethod
        def evidence_non_empty(cls, v: List[str]) -> List[str]:
            if not v:
                raise ValueError("evidence must be a non-empty list")
            return v

    class TrendResult(BaseModel):
        """A temporal trend with direction, topic, confidence, and time window."""

        trend: str
        topic: str
        confidence: float
        time_window: str

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

    class Insight(BaseModel):
        """A human-readable observation with text, confidence, and supporting evidence."""

        text: str
        confidence: float
        evidence: List[str]

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

        @field_validator("evidence")
        @classmethod
        def evidence_non_empty(cls, v: List[str]) -> List[str]:
            if not v:
                raise ValueError("evidence must be a non-empty list")
            return v

    class InsightReport(BaseModel):
        """Top-level output containing insights and associated metadata."""

        insights: List[Insight]
        pattern_count: int
        trend_count: int
        memory_count: int

else:
    from dataclasses import dataclass, field

    @dataclass
    class PatternResult:
        """A detected pattern with type, value, frequency, confidence, and evidence."""

        pattern_type: str
        pattern: str
        frequency: int
        confidence: float
        evidence: List[str]

        def __post_init__(self) -> None:
            if self.frequency <= 0:
                raise ValueError(f"frequency must be a positive integer, got {self.frequency}")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
            if not self.evidence:
                raise ValueError("evidence must be a non-empty list")

    @dataclass
    class TrendResult:
        """A temporal trend with direction, topic, confidence, and time window."""

        trend: str
        topic: str
        confidence: float
        time_window: str

        def __post_init__(self) -> None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    @dataclass
    class Insight:
        """A human-readable observation with text, confidence, and supporting evidence."""

        text: str
        confidence: float
        evidence: List[str]

        def __post_init__(self) -> None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
            if not self.evidence:
                raise ValueError("evidence must be a non-empty list")

    @dataclass
    class InsightReport:
        """Top-level output containing insights and associated metadata."""

        insights: List[Insight]
        pattern_count: int
        trend_count: int
        memory_count: int
