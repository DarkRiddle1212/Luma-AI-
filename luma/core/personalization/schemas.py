"""
Personalization Engine Data Schemas.

Defines UserProfile, Preference, AdaptationContext, and PersonalizationResult models.
Uses Pydantic if available, otherwise dataclasses with __post_init__ validation.
Matches the dual-path pattern in luma/core/insight/schemas.py.
"""

from typing import Dict, List

try:
    from pydantic import BaseModel, field_validator
    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


if _USE_PYDANTIC:
    class UserProfile(BaseModel):
        """Aggregated user profile built from memories and insights."""

        interests: List[str]
        behavior_patterns: List[str]
        interaction_style: str
        strengths: List[str]
        evidence: Dict[str, List[str]]

        @field_validator("interaction_style")
        @classmethod
        def interaction_style_valid(cls, v: str) -> str:
            valid = {"concise", "detailed", "balanced"}
            if v not in valid:
                raise ValueError(f"interaction_style must be one of {valid}, got {v!r}")
            return v

        @field_validator("interests", "behavior_patterns")
        @classmethod
        def strings_non_empty(cls, v: List[str]) -> List[str]:
            for item in v:
                if not item or not item.strip():
                    raise ValueError(
                        "interests and behavior_patterns must contain non-empty strings"
                    )
            return v

    class Preference(BaseModel):
        """A detected user preference with confidence and explanation."""

        preference: str
        confidence: float
        reason: str

        @field_validator("confidence")
        @classmethod
        def confidence_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
            return v

        @field_validator("reason")
        @classmethod
        def reason_non_empty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("reason must be non-empty")
            return v

    class AdaptationContext(BaseModel):
        """Concrete adaptation instructions derived from profile and preferences."""

        tone: str
        style: str
        focus: str
        reasons: Dict[str, str]

        @field_validator("tone")
        @classmethod
        def tone_valid(cls, v: str) -> str:
            valid = {"technical", "casual", "formal"}
            if v not in valid:
                raise ValueError(f"tone must be one of {valid}, got {v!r}")
            return v

        @field_validator("style")
        @classmethod
        def style_valid(cls, v: str) -> str:
            valid = {"concise", "detailed", "step-by-step", "balanced"}
            if v not in valid:
                raise ValueError(f"style must be one of {valid}, got {v!r}")
            return v

        @field_validator("focus")
        @classmethod
        def focus_valid(cls, v: str) -> str:
            valid = {"high-level", "deep-technical"}
            if v not in valid:
                raise ValueError(f"focus must be one of {valid}, got {v!r}")
            return v

    class PersonalizationResult(BaseModel):
        """Top-level output of the personalization pipeline."""

        profile: UserProfile
        preferences: List[Preference]
        adaptation: AdaptationContext

else:
    from dataclasses import dataclass

    @dataclass
    class UserProfile:
        """Aggregated user profile built from memories and insights."""

        interests: List[str]
        behavior_patterns: List[str]
        interaction_style: str
        strengths: List[str]
        evidence: Dict[str, List[str]]

        def __post_init__(self) -> None:
            valid = {"concise", "detailed", "balanced"}
            if self.interaction_style not in valid:
                raise ValueError(
                    f"interaction_style must be one of {valid}, got {self.interaction_style!r}"
                )
            for s in self.interests:
                if not s or not s.strip():
                    raise ValueError("interests must contain non-empty strings")
            for s in self.behavior_patterns:
                if not s or not s.strip():
                    raise ValueError("behavior_patterns must contain non-empty strings")

    @dataclass
    class Preference:
        """A detected user preference with confidence and explanation."""

        preference: str
        confidence: float
        reason: str

        def __post_init__(self) -> None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    f"confidence must be in [0.0, 1.0], got {self.confidence}"
                )
            if not self.reason or not self.reason.strip():
                raise ValueError("reason must be non-empty")

    @dataclass
    class AdaptationContext:
        """Concrete adaptation instructions derived from profile and preferences."""

        tone: str
        style: str
        focus: str
        reasons: Dict[str, str]

        def __post_init__(self) -> None:
            if self.tone not in {"technical", "casual", "formal"}:
                raise ValueError(
                    f"tone must be one of technical/casual/formal, got {self.tone!r}"
                )
            if self.style not in {"concise", "detailed", "step-by-step", "balanced"}:
                raise ValueError(
                    f"style must be one of concise/detailed/step-by-step/balanced, got {self.style!r}"
                )
            if self.focus not in {"high-level", "deep-technical"}:
                raise ValueError(
                    f"focus must be one of high-level/deep-technical, got {self.focus!r}"
                )

    @dataclass
    class PersonalizationResult:
        """Top-level output of the personalization pipeline."""

        profile: UserProfile
        preferences: List[Preference]
        adaptation: AdaptationContext
