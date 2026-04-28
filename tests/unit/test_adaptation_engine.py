"""
Unit tests for AdaptationEngine.

Covers tone, style, focus determination, default/edge cases, and immutability.
"""

import copy
import pytest

from luma.core.personalization.adaptation_engine import AdaptationEngine
from luma.core.personalization.schemas import AdaptationContext, Preference, UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(
    interests=None,
    behavior_patterns=None,
    interaction_style="balanced",
    strengths=None,
    evidence=None,
) -> UserProfile:
    return UserProfile(
        interests=interests or [],
        behavior_patterns=behavior_patterns or [],
        interaction_style=interaction_style,
        strengths=strengths or [],
        evidence=evidence or {},
    )


def _pref(label: str, confidence: float = 0.9) -> Preference:
    return Preference(preference=label, confidence=confidence, reason="test reason")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdaptationEngineDefaults:
    def test_returns_default_context_for_empty_profile_and_preferences(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [])
        assert isinstance(ctx, AdaptationContext)
        assert ctx.tone == "casual"
        assert ctx.style == "balanced"
        assert ctx.focus == "high-level"


class TestToneDetermination:
    def test_tone_is_technical_when_technical_preference_present(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [_pref("technical")])
        assert ctx.tone == "technical"

    def test_tone_is_technical_when_interest_matches_technical_keyword(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interests=["python"]), [])
        assert ctx.tone == "technical"

    def test_tone_is_formal_when_formal_preference_present(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [_pref("formal")])
        assert ctx.tone == "formal"

    def test_tone_is_casual_when_no_tone_signal_present(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interests=["cooking"]), [])
        assert ctx.tone == "casual"

    def test_technical_preference_takes_priority_over_formal(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [_pref("technical"), _pref("formal")])
        assert ctx.tone == "technical"


class TestStyleDetermination:
    def test_style_is_step_by_step_when_preference_present(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [_pref("step-by-step")])
        assert ctx.style == "step-by-step"

    def test_style_is_concise_when_interaction_style_is_concise(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interaction_style="concise"), [])
        assert ctx.style == "concise"

    def test_style_is_detailed_when_interaction_style_is_detailed(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interaction_style="detailed"), [])
        assert ctx.style == "detailed"

    def test_style_is_balanced_when_no_style_signal(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interaction_style="balanced"), [])
        assert ctx.style == "balanced"

    def test_step_by_step_preference_overrides_interaction_style(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(interaction_style="concise"), [_pref("step-by-step")])
        assert ctx.style == "step-by-step"


class TestFocusDetermination:
    def test_focus_is_deep_technical_when_preference_present(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [_pref("deep-technical")])
        assert ctx.focus == "deep-technical"

    def test_focus_is_deep_technical_when_profile_has_three_or_more_strengths(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(strengths=["a", "b", "c"]), [])
        assert ctx.focus == "deep-technical"

    def test_focus_is_deep_technical_when_profile_has_more_than_three_strengths(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(strengths=["a", "b", "c", "d"]), [])
        assert ctx.focus == "deep-technical"

    def test_focus_is_high_level_when_no_depth_signal(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(strengths=["a", "b"]), [])
        assert ctx.focus == "high-level"

    def test_focus_is_high_level_when_strengths_empty(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [])
        assert ctx.focus == "high-level"


class TestReasons:
    def test_all_three_reasons_are_non_empty_strings(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), [])
        assert isinstance(ctx.reasons["tone"], str) and ctx.reasons["tone"].strip()
        assert isinstance(ctx.reasons["style"], str) and ctx.reasons["style"].strip()
        assert isinstance(ctx.reasons["focus"], str) and ctx.reasons["focus"].strip()

    def test_reasons_non_empty_for_all_explicit_signals(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(
            _profile(interaction_style="detailed", strengths=["x", "y", "z"]),
            [_pref("technical"), _pref("step-by-step"), _pref("deep-technical")],
        )
        for key in ("tone", "style", "focus"):
            assert ctx.reasons[key], f"reasons[{key!r}] is empty"
            assert ctx.reasons[key].strip(), f"reasons[{key!r}] is whitespace-only"


class TestImmutability:
    def test_input_profile_is_not_modified(self):
        engine = AdaptationEngine()
        profile = _profile(
            interests=["python"],
            behavior_patterns=["reads docs"],
            interaction_style="concise",
            strengths=["debugging"],
        )
        original_profile = copy.deepcopy(profile)
        engine.adapt(profile, [_pref("technical")])
        assert profile.interests == original_profile.interests
        assert profile.behavior_patterns == original_profile.behavior_patterns
        assert profile.interaction_style == original_profile.interaction_style
        assert profile.strengths == original_profile.strengths
        assert profile.evidence == original_profile.evidence

    def test_input_preferences_are_not_modified(self):
        engine = AdaptationEngine()
        prefs = [_pref("technical"), _pref("step-by-step")]
        original_prefs = copy.deepcopy(prefs)
        engine.adapt(_profile(), prefs)
        for orig, after in zip(original_prefs, prefs):
            assert orig.preference == after.preference
            assert orig.confidence == after.confidence
            assert orig.reason == after.reason


class TestNonePreferences:
    def test_none_preferences_treated_as_empty_list(self):
        engine = AdaptationEngine()
        ctx = engine.adapt(_profile(), None)
        assert ctx.tone == "casual"
        assert ctx.style == "balanced"
        assert ctx.focus == "high-level"
