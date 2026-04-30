"""
Unit tests for luma.core.personality.schemas.

Tests validation logic for PersonalityProfile, ToneSelection, StylePreference,
PromptInstructions, and GuardrailResult. Works regardless of whether Pydantic
or dataclasses are used.
"""

import pytest
from luma.core.personality.schemas import (
    PersonalityProfile,
    ToneSelection,
    StylePreference,
    PromptInstructions,
    GuardrailResult,
    PersonalityError,
    VALID_TONES,
    VALID_STYLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_personality_profile(**overrides):
    defaults = dict(
        base_identity="You are Luma, an AI assistant.",
        preferred_tone="friendly",
        preferred_style="high_signal_low_noise",
        output_constraints=["no rambling", "no repetition"],
    )
    defaults.update(overrides)
    return PersonalityProfile(**defaults)


def _make_tone_selection(**overrides):
    defaults = dict(
        tone="friendly",
        rationale="user preference",
        context_signals={"mode": "chat"},
    )
    defaults.update(overrides)
    return ToneSelection(**defaults)


def _make_style_preference(**overrides):
    defaults = dict(
        style="high_signal_low_noise",
        description="Balance detail and brevity",
        active=True,
    )
    defaults.update(overrides)
    return StylePreference(**defaults)


def _make_prompt_instructions(**overrides):
    defaults = dict(
        system_identity="You are Luma, an AI assistant.",
        tone_guidance="Use conversational language",
        style_constraints="Balance detail and brevity",
        output_rules=["no rambling", "no repetition"],
        metadata={"tone": "friendly", "style": "high_signal_low_noise"},
    )
    defaults.update(overrides)
    return PromptInstructions(**defaults)


def _make_guardrail_result(**overrides):
    defaults = dict(
        passed=True,
        violations=[],
        score=1.0,
        notes="No violations detected",
    )
    defaults.update(overrides)
    return GuardrailResult(**defaults)


# ---------------------------------------------------------------------------
# PersonalityProfile
# ---------------------------------------------------------------------------

class TestPersonalityProfile:
    def test_valid_construction(self):
        profile = _make_personality_profile()
        assert profile.base_identity == "You are Luma, an AI assistant."
        assert profile.preferred_tone == "friendly"
        assert profile.preferred_style == "high_signal_low_noise"
        assert profile.output_constraints == ["no rambling", "no repetition"]

    def test_empty_output_constraints_is_valid(self):
        profile = _make_personality_profile(output_constraints=[])
        assert profile.output_constraints == []

    def test_multiple_output_constraints(self):
        constraints = ["no rambling", "no repetition", "no contradiction", "no vague filler"]
        profile = _make_personality_profile(output_constraints=constraints)
        assert len(profile.output_constraints) == 4


# ---------------------------------------------------------------------------
# ToneSelection
# ---------------------------------------------------------------------------

class TestToneSelection:
    def test_valid_construction_friendly(self):
        selection = _make_tone_selection(tone="friendly")
        assert selection.tone == "friendly"
        assert selection.rationale == "user preference"
        assert selection.context_signals == {"mode": "chat"}

    def test_valid_construction_professional(self):
        selection = _make_tone_selection(tone="professional")
        assert selection.tone == "professional"

    def test_valid_construction_concise(self):
        selection = _make_tone_selection(tone="concise")
        assert selection.tone == "concise"

    def test_valid_construction_technical(self):
        selection = _make_tone_selection(tone="technical")
        assert selection.tone == "technical"

    def test_valid_construction_teacher(self):
        selection = _make_tone_selection(tone="teacher")
        assert selection.tone == "teacher"

    def test_valid_construction_motivational(self):
        selection = _make_tone_selection(tone="motivational")
        assert selection.tone == "motivational"

    def test_valid_construction_analytical(self):
        selection = _make_tone_selection(tone="analytical")
        assert selection.tone == "analytical"

    def test_invalid_tone_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            _make_tone_selection(tone="invalid_tone")
        assert "tone must be one of" in str(exc_info.value)
        assert "invalid_tone" in str(exc_info.value)

    def test_empty_tone_raises_value_error(self):
        with pytest.raises(ValueError):
            _make_tone_selection(tone="")

    def test_random_invalid_tone_raises(self):
        with pytest.raises(ValueError):
            _make_tone_selection(tone="super_friendly")

    def test_empty_context_signals_is_valid(self):
        selection = _make_tone_selection(context_signals={})
        assert selection.context_signals == {}


# ---------------------------------------------------------------------------
# StylePreference
# ---------------------------------------------------------------------------

class TestStylePreference:
    def test_valid_construction_high_signal_low_noise(self):
        pref = _make_style_preference(style="high_signal_low_noise")
        assert pref.style == "high_signal_low_noise"
        assert pref.description == "Balance detail and brevity"
        assert pref.active is True

    def test_valid_construction_short_answers(self):
        pref = _make_style_preference(style="short_answers")
        assert pref.style == "short_answers"

    def test_valid_construction_step_by_step(self):
        pref = _make_style_preference(style="step_by_step")
        assert pref.style == "step_by_step"

    def test_valid_construction_detailed_explanations(self):
        pref = _make_style_preference(style="detailed_explanations")
        assert pref.style == "detailed_explanations"

    def test_valid_construction_motivational_style(self):
        pref = _make_style_preference(style="motivational_style")
        assert pref.style == "motivational_style"

    def test_valid_construction_technical_depth(self):
        pref = _make_style_preference(style="technical_depth")
        assert pref.style == "technical_depth"

    def test_invalid_style_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            _make_style_preference(style="invalid_style")
        assert "style must be one of" in str(exc_info.value)
        assert "invalid_style" in str(exc_info.value)

    def test_empty_style_raises_value_error(self):
        with pytest.raises(ValueError):
            _make_style_preference(style="")

    def test_random_invalid_style_raises(self):
        with pytest.raises(ValueError):
            _make_style_preference(style="verbose")

    def test_active_false_is_valid(self):
        pref = _make_style_preference(active=False)
        assert pref.active is False


# ---------------------------------------------------------------------------
# PromptInstructions
# ---------------------------------------------------------------------------

class TestPromptInstructions:
    def test_valid_construction(self):
        instructions = _make_prompt_instructions()
        assert instructions.system_identity == "You are Luma, an AI assistant."
        assert instructions.tone_guidance == "Use conversational language"
        assert instructions.style_constraints == "Balance detail and brevity"
        assert instructions.output_rules == ["no rambling", "no repetition"]
        assert instructions.metadata == {"tone": "friendly", "style": "high_signal_low_noise"}

    def test_empty_output_rules_is_valid(self):
        instructions = _make_prompt_instructions(output_rules=[])
        assert instructions.output_rules == []

    def test_empty_metadata_is_valid(self):
        instructions = _make_prompt_instructions(metadata={})
        assert instructions.metadata == {}

    def test_multiple_output_rules(self):
        rules = ["no rambling", "no repetition", "no contradiction", "no vague filler", "respect length"]
        instructions = _make_prompt_instructions(output_rules=rules)
        assert len(instructions.output_rules) == 5


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------

class TestGuardrailResult:
    def test_valid_construction_passed(self):
        result = _make_guardrail_result()
        assert result.passed is True
        assert result.violations == []
        assert result.score == 1.0
        assert result.notes == "No violations detected"

    def test_valid_construction_failed(self):
        result = _make_guardrail_result(
            passed=False,
            violations=["rambling", "repetition"],
            score=0.5,
            notes="Multiple violations detected",
        )
        assert result.passed is False
        assert result.violations == ["rambling", "repetition"]
        assert result.score == 0.5
        assert result.notes == "Multiple violations detected"

    def test_score_at_zero_is_valid(self):
        result = _make_guardrail_result(score=0.0)
        assert result.score == 0.0

    def test_score_at_one_is_valid(self):
        result = _make_guardrail_result(score=1.0)
        assert result.score == 1.0

    def test_score_below_zero_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            _make_guardrail_result(score=-0.1)
        assert "score must be in [0.0, 1.0]" in str(exc_info.value)

    def test_score_above_one_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            _make_guardrail_result(score=1.1)
        assert "score must be in [0.0, 1.0]" in str(exc_info.value)

    def test_score_negative_large_raises(self):
        with pytest.raises(ValueError):
            _make_guardrail_result(score=-5.0)

    def test_score_large_positive_raises(self):
        with pytest.raises(ValueError):
            _make_guardrail_result(score=10.0)

    def test_score_mid_range_is_valid(self):
        result = _make_guardrail_result(score=0.75)
        assert result.score == 0.75

    def test_multiple_violations(self):
        violations = ["rambling", "repetition", "contradiction", "vague filler"]
        result = _make_guardrail_result(violations=violations)
        assert len(result.violations) == 4


# ---------------------------------------------------------------------------
# PersonalityError Exception
# ---------------------------------------------------------------------------

class TestPersonalityError:
    def test_personality_error_is_exception(self):
        assert issubclass(PersonalityError, Exception)

    def test_personality_error_can_be_raised(self):
        with pytest.raises(PersonalityError):
            raise PersonalityError("Test error")

    def test_personality_error_message(self):
        with pytest.raises(PersonalityError) as exc_info:
            raise PersonalityError("Custom error message")
        assert "Custom error message" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Serialization and Deserialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_personality_profile_dict_conversion(self):
        """Test that PersonalityProfile can be converted to/from dict."""
        profile = _make_personality_profile()
        # Both Pydantic and dataclasses support dict conversion
        if hasattr(profile, 'model_dump'):
            # Pydantic v2
            data = profile.model_dump()
        elif hasattr(profile, 'dict'):
            # Pydantic v1
            data = profile.dict()
        else:
            # dataclasses
            from dataclasses import asdict
            data = asdict(profile)
        
        assert data["base_identity"] == "You are Luma, an AI assistant."
        assert data["preferred_tone"] == "friendly"

    def test_tone_selection_dict_conversion(self):
        """Test that ToneSelection can be converted to/from dict."""
        selection = _make_tone_selection()
        if hasattr(selection, 'model_dump'):
            data = selection.model_dump()
        elif hasattr(selection, 'dict'):
            data = selection.dict()
        else:
            from dataclasses import asdict
            data = asdict(selection)
        
        assert data["tone"] == "friendly"
        assert data["rationale"] == "user preference"

    def test_guardrail_result_dict_conversion(self):
        """Test that GuardrailResult can be converted to/from dict."""
        result = _make_guardrail_result()
        if hasattr(result, 'model_dump'):
            data = result.model_dump()
        elif hasattr(result, 'dict'):
            data = result.dict()
        else:
            from dataclasses import asdict
            data = asdict(result)
        
        assert data["passed"] is True
        assert data["score"] == 1.0
