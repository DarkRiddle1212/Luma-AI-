"""
Unit tests for luma.core.personalization.schemas.

Tests validation logic for UserProfile, Preference, AdaptationContext,
and PersonalizationResult. Works regardless of whether Pydantic or
dataclasses are used.
"""

import pytest
from luma.core.personalization.schemas import (
    UserProfile,
    Preference,
    AdaptationContext,
    PersonalizationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_profile(**overrides):
    defaults = dict(
        interests=["python", "machine learning"],
        behavior_patterns=["I prefer concise answers"],
        interaction_style="balanced",
        strengths=["problem solving"],
        evidence={"interests": ["mem-1"], "behavior_patterns": ["mem-2"]},
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


def _make_preference(**overrides):
    defaults = dict(
        preference="technical",
        confidence=0.8,
        reason="Detected in 3 memories: mem-1, mem-2, mem-3",
    )
    defaults.update(overrides)
    return Preference(**defaults)


def _make_adaptation_context(**overrides):
    defaults = dict(
        tone="technical",
        style="balanced",
        focus="high-level",
        reasons={
            "tone": "technical preference detected",
            "style": "balanced interaction style",
            "focus": "insufficient depth signals",
        },
    )
    defaults.update(overrides)
    return AdaptationContext(**defaults)


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

class TestUserProfile:
    def test_valid_construction_balanced(self):
        profile = _make_user_profile()
        assert profile.interaction_style == "balanced"
        assert profile.interests == ["python", "machine learning"]

    def test_valid_construction_concise(self):
        profile = _make_user_profile(interaction_style="concise")
        assert profile.interaction_style == "concise"

    def test_valid_construction_detailed(self):
        profile = _make_user_profile(interaction_style="detailed")
        assert profile.interaction_style == "detailed"

    def test_invalid_interaction_style_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(interaction_style="verbose")

    def test_invalid_interaction_style_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(interaction_style="")

    def test_invalid_interaction_style_random_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(interaction_style="fast")

    def test_empty_interests_is_valid(self):
        profile = _make_user_profile(interests=[])
        assert profile.interests == []

    def test_empty_string_in_interests_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(interests=["python", ""])

    def test_whitespace_only_string_in_interests_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(interests=["python", "   "])

    def test_empty_string_in_behavior_patterns_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(behavior_patterns=[""])

    def test_whitespace_only_in_behavior_patterns_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_user_profile(behavior_patterns=["  "])

    def test_empty_behavior_patterns_is_valid(self):
        profile = _make_user_profile(behavior_patterns=[])
        assert profile.behavior_patterns == []

    def test_empty_evidence_is_valid(self):
        profile = _make_user_profile(evidence={})
        assert profile.evidence == {}


# ---------------------------------------------------------------------------
# Preference
# ---------------------------------------------------------------------------

class TestPreference:
    def test_valid_construction(self):
        pref = _make_preference()
        assert pref.preference == "technical"
        assert pref.confidence == 0.8
        assert pref.reason == "Detected in 3 memories: mem-1, mem-2, mem-3"

    def test_confidence_at_zero_is_valid(self):
        pref = _make_preference(confidence=0.0)
        assert pref.confidence == 0.0

    def test_confidence_at_one_is_valid(self):
        pref = _make_preference(confidence=1.0)
        assert pref.confidence == 1.0

    def test_confidence_below_zero_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_preference(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_preference(confidence=1.1)

    def test_confidence_negative_large_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_preference(confidence=-5.0)

    def test_empty_reason_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_preference(reason="")

    def test_whitespace_only_reason_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_preference(reason="   ")

    def test_valid_reason_accepted(self):
        pref = _make_preference(reason="Some valid reason")
        assert pref.reason == "Some valid reason"


# ---------------------------------------------------------------------------
# AdaptationContext
# ---------------------------------------------------------------------------

class TestAdaptationContext:
    def test_valid_construction(self):
        ctx = _make_adaptation_context()
        assert ctx.tone == "technical"
        assert ctx.style == "balanced"
        assert ctx.focus == "high-level"

    # Tone validation
    def test_tone_technical_valid(self):
        ctx = _make_adaptation_context(tone="technical")
        assert ctx.tone == "technical"

    def test_tone_casual_valid(self):
        ctx = _make_adaptation_context(tone="casual")
        assert ctx.tone == "casual"

    def test_tone_formal_valid(self):
        ctx = _make_adaptation_context(tone="formal")
        assert ctx.tone == "formal"

    def test_tone_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(tone="friendly")

    def test_tone_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(tone="")

    # Style validation
    def test_style_concise_valid(self):
        ctx = _make_adaptation_context(style="concise")
        assert ctx.style == "concise"

    def test_style_detailed_valid(self):
        ctx = _make_adaptation_context(style="detailed")
        assert ctx.style == "detailed"

    def test_style_step_by_step_valid(self):
        ctx = _make_adaptation_context(style="step-by-step")
        assert ctx.style == "step-by-step"

    def test_style_balanced_valid(self):
        ctx = _make_adaptation_context(style="balanced")
        assert ctx.style == "balanced"

    def test_style_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(style="verbose")

    def test_style_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(style="")

    # Focus validation
    def test_focus_high_level_valid(self):
        ctx = _make_adaptation_context(focus="high-level")
        assert ctx.focus == "high-level"

    def test_focus_deep_technical_valid(self):
        ctx = _make_adaptation_context(focus="deep-technical")
        assert ctx.focus == "deep-technical"

    def test_focus_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(focus="medium")

    def test_focus_empty_raises(self):
        with pytest.raises((ValueError, Exception)):
            _make_adaptation_context(focus="")


# ---------------------------------------------------------------------------
# PersonalizationResult
# ---------------------------------------------------------------------------

class TestPersonalizationResult:
    def test_valid_construction(self):
        profile = _make_user_profile()
        pref = _make_preference()
        ctx = _make_adaptation_context()
        result = PersonalizationResult(
            profile=profile,
            preferences=[pref],
            adaptation=ctx,
        )
        assert result.profile is profile
        assert result.preferences == [pref]
        assert result.adaptation is ctx

    def test_empty_preferences_is_valid(self):
        profile = _make_user_profile()
        ctx = _make_adaptation_context()
        result = PersonalizationResult(
            profile=profile,
            preferences=[],
            adaptation=ctx,
        )
        assert result.preferences == []

    def test_multiple_preferences(self):
        profile = _make_user_profile()
        prefs = [_make_preference(preference=f"pref-{i}") for i in range(3)]
        ctx = _make_adaptation_context()
        result = PersonalizationResult(
            profile=profile,
            preferences=prefs,
            adaptation=ctx,
        )
        assert len(result.preferences) == 3
