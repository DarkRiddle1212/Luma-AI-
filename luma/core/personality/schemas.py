"""
Personality Layer Data Schemas.

Defines PersonalityProfile, ToneSelection, StylePreference, PromptInstructions,
and GuardrailResult models. Uses Pydantic if available, otherwise dataclasses
with __post_init__ validation. Matches the dual-path pattern in
luma/core/personalization/schemas.py.
"""

from typing import Any, Dict, List

try:
    from pydantic import BaseModel, field_validator

    _USE_PYDANTIC = True
except ImportError:
    _USE_PYDANTIC = False


# Valid enum values
VALID_TONES = {
    "professional",
    "friendly",
    "concise",
    "technical",
    "teacher",
    "motivational",
    "analytical",
}

VALID_STYLES = {
    "short_answers",
    "step_by_step",
    "detailed_explanations",
    "high_signal_low_noise",
    "motivational_style",
    "technical_depth",
}


class PersonalityError(Exception):
    """Exception raised by PersonalityEngine for errors in personality layer."""

    pass


if _USE_PYDANTIC:

    class PersonalityProfile(BaseModel):
        """Aggregated personality profile for a user."""

        base_identity: str
        preferred_tone: str
        preferred_style: str
        output_constraints: List[str]

    class ToneSelection(BaseModel):
        """Selected tone with rationale and context signals."""

        tone: str
        rationale: str
        context_signals: Dict[str, Any]

        @field_validator("tone")
        @classmethod
        def tone_valid(cls, v: str) -> str:
            if v not in VALID_TONES:
                raise ValueError(f"tone must be one of {VALID_TONES}, got {v!r}")
            return v

    class StylePreference(BaseModel):
        """User's preferred communication style."""

        style: str
        description: str
        active: bool

        @field_validator("style")
        @classmethod
        def style_valid(cls, v: str) -> str:
            if v not in VALID_STYLES:
                raise ValueError(f"style must be one of {VALID_STYLES}, got {v!r}")
            return v

    class PromptInstructions(BaseModel):
        """Final augmented prompt instructions for LLM layer."""

        system_identity: str
        tone_guidance: str
        style_constraints: str
        output_rules: List[str]
        metadata: Dict[str, Any]

    class GuardrailResult(BaseModel):
        """Validation result of a response against quality constraints."""

        passed: bool
        violations: List[str]
        score: float
        notes: str

        @field_validator("score")
        @classmethod
        def score_range(cls, v: float) -> float:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"score must be in [0.0, 1.0], got {v}")
            return v

else:
    from dataclasses import dataclass

    @dataclass
    class PersonalityProfile:
        """Aggregated personality profile for a user."""

        base_identity: str
        preferred_tone: str
        preferred_style: str
        output_constraints: List[str]

    @dataclass
    class ToneSelection:
        """Selected tone with rationale and context signals."""

        tone: str
        rationale: str
        context_signals: Dict[str, Any]

        def __post_init__(self) -> None:
            if self.tone not in VALID_TONES:
                raise ValueError(
                    f"tone must be one of {VALID_TONES}, got {self.tone!r}"
                )

    @dataclass
    class StylePreference:
        """User's preferred communication style."""

        style: str
        description: str
        active: bool

        def __post_init__(self) -> None:
            if self.style not in VALID_STYLES:
                raise ValueError(
                    f"style must be one of {VALID_STYLES}, got {self.style!r}"
                )

    @dataclass
    class PromptInstructions:
        """Final augmented prompt instructions for LLM layer."""

        system_identity: str
        tone_guidance: str
        style_constraints: str
        output_rules: List[str]
        metadata: Dict[str, Any]

    @dataclass
    class GuardrailResult:
        """Validation result of a response against quality constraints."""

        passed: bool
        violations: List[str]
        score: float
        notes: str

        def __post_init__(self) -> None:
            if not 0.0 <= self.score <= 1.0:
                raise ValueError(
                    f"score must be in [0.0, 1.0], got {self.score}"
                )
