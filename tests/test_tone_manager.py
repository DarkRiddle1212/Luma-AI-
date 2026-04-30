"""
Unit tests for ToneManager component.

Tests tone selection priority, context-based selection, and graceful handling
of invalid user preferences.
"""

import pytest

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.tone_manager import ToneManager
from luma.core.personality.schemas import ToneSelection


@pytest.fixture
def tone_manager():
    """Create a ToneManager instance."""
    return ToneManager()


@pytest.fixture
def default_context():
    """Create a default AdaptationContext."""
    return AdaptationContext(
        tone="casual",
        style="balanced",
        focus="high-level",
        reasons={},
    )


class TestToneManagerUserPreference:
    """Test user preference priority (highest priority)."""

    def test_valid_user_preference_overrides_all(self, tone_manager, default_context):
        """Valid user preference should override mode and context."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="teacher",
            user_preference="analytical",
        )
        assert result.tone == "analytical"
        assert result.rationale == "user preference"
        assert result.context_signals["user_preference"] == "analytical"

    def test_all_valid_tones_as_user_preference(self, tone_manager, default_context):
        """All valid tones should be accepted as user preference."""
        valid_tones = [
            "professional",
            "friendly",
            "concise",
            "technical",
            "teacher",
            "motivational",
            "analytical",
        ]
        for tone in valid_tones:
            result = tone_manager.select_tone(
                context=default_context,
                mode="chat",
                user_preference=tone,
            )
            assert result.tone == tone
            assert result.rationale == "user preference"

    def test_invalid_user_preference_falls_back(self, tone_manager, default_context):
        """Invalid user preference should fall back to context-based selection."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference="invalid_tone",
        )
        # Should fall back to casual context → friendly
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"


class TestToneManagerModeSelection:
    """Test mode-based selection (second priority)."""

    def test_teacher_mode_selects_teacher_tone(self, tone_manager, default_context):
        """Teacher mode should select teacher tone."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="teacher",
            user_preference=None,
        )
        assert result.tone == "teacher"
        assert result.rationale == "teacher mode active"

    def test_teacher_mode_overrides_context(self, tone_manager):
        """Teacher mode should override context tone."""
        technical_context = AdaptationContext(
            tone="technical",
            style="balanced",
            focus="deep-technical",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=technical_context,
            mode="teacher",
            user_preference=None,
        )
        assert result.tone == "teacher"
        assert result.rationale == "teacher mode active"


class TestToneManagerContextSelection:
    """Test context-based selection (third, fourth, fifth priority)."""

    def test_technical_context_selects_technical_tone(self, tone_manager):
        """Technical context should select technical tone."""
        technical_context = AdaptationContext(
            tone="technical",
            style="balanced",
            focus="deep-technical",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=technical_context,
            mode="chat",
            user_preference=None,
        )
        assert result.tone == "technical"
        assert result.rationale == "technical context detected"

    def test_formal_context_selects_professional_tone(self, tone_manager):
        """Formal context should select professional tone."""
        formal_context = AdaptationContext(
            tone="formal",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=formal_context,
            mode="chat",
            user_preference=None,
        )
        assert result.tone == "professional"
        assert result.rationale == "formal context detected"

    def test_casual_context_selects_friendly_tone(self, tone_manager):
        """Casual context should select friendly tone."""
        casual_context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=casual_context,
            mode="chat",
            user_preference=None,
        )
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"


class TestToneManagerDefault:
    """Test default tone selection (lowest priority)."""

    def test_casual_context_triggers_friendly_as_default(self, tone_manager):
        """Casual context should trigger friendly tone (which is also the default)."""
        casual_context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=casual_context,
            mode="chat",
            user_preference=None,
        )
        # Casual context triggers friendly
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"

    def test_default_fallback_with_non_matching_mode(self, tone_manager):
        """Should fall back to default when mode doesn't match any rule."""
        # Use formal context with a non-teacher mode
        # This tests that when no specific rule matches, we get the appropriate fallback
        formal_context = AdaptationContext(
            tone="formal",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=formal_context,
            mode="chat",  # Not "teacher", so mode rule doesn't apply
            user_preference=None,
        )
        # Should match formal context → professional
        assert result.tone == "professional"
        assert result.rationale == "formal context detected"


class TestToneManagerPriorityOrder:
    """Test explicit priority order verification."""

    def test_user_preference_overrides_teacher_mode(self, tone_manager, default_context):
        """User preference (priority 1) should override teacher mode (priority 2)."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="teacher",
            user_preference="analytical",
        )
        assert result.tone == "analytical"
        assert result.rationale == "user preference"

    def test_user_preference_overrides_technical_context(self, tone_manager):
        """User preference (priority 1) should override technical context (priority 3)."""
        technical_context = AdaptationContext(
            tone="technical",
            style="balanced",
            focus="deep-technical",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=technical_context,
            mode="chat",
            user_preference="friendly",
        )
        assert result.tone == "friendly"
        assert result.rationale == "user preference"

    def test_teacher_mode_overrides_formal_context(self, tone_manager):
        """Teacher mode (priority 2) should override formal context (priority 4)."""
        formal_context = AdaptationContext(
            tone="formal",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=formal_context,
            mode="teacher",
            user_preference=None,
        )
        assert result.tone == "teacher"
        assert result.rationale == "teacher mode active"

    def test_technical_context_overrides_default(self, tone_manager):
        """Technical context (priority 3) should override default (priority 6)."""
        technical_context = AdaptationContext(
            tone="technical",
            style="balanced",
            focus="deep-technical",
            reasons={},
        )
        result = tone_manager.select_tone(
            context=technical_context,
            mode="chat",
            user_preference=None,
        )
        assert result.tone == "technical"
        assert result.rationale == "technical context detected"


class TestToneManagerDeterminism:
    """Test deterministic behavior."""

    def test_identical_inputs_produce_identical_outputs(self, tone_manager):
        """Calling select_tone twice with identical inputs should produce identical results."""
        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )

        result1 = tone_manager.select_tone(
            context=context,
            mode="chat",
            user_preference="analytical",
        )
        result2 = tone_manager.select_tone(
            context=context,
            mode="chat",
            user_preference="analytical",
        )

        assert result1.tone == result2.tone
        assert result1.rationale == result2.rationale
        assert result1.context_signals == result2.context_signals


class TestToneManagerContextSignals:
    """Test context signals in ToneSelection."""

    def test_context_signals_include_all_inputs(self, tone_manager, default_context):
        """Context signals should include context tone, mode, and user preference."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="teacher",
            user_preference="analytical",
        )
        assert result.context_signals["context_tone"] == "casual"
        assert result.context_signals["mode"] == "teacher"
        assert result.context_signals["user_preference"] == "analytical"

    def test_context_signals_with_none_preference(self, tone_manager, default_context):
        """Context signals should handle None user preference."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference=None,
        )
        assert result.context_signals["user_preference"] is None


class TestToneManagerGuidanceMapping:
    """Test tone → guidance mapping."""

    def test_all_valid_tones_have_guidance(self, tone_manager):
        """All valid tones should have corresponding guidance strings."""
        from luma.core.personality.schemas import VALID_TONES

        for tone in VALID_TONES:
            assert tone in tone_manager.TONE_GUIDANCE
            assert isinstance(tone_manager.TONE_GUIDANCE[tone], str)
            assert len(tone_manager.TONE_GUIDANCE[tone]) > 0

    def test_guidance_mapping_completeness(self, tone_manager):
        """Verify specific guidance strings for each tone."""
        expected_keywords = {
            "professional": ["formal", "structured"],
            "friendly": ["conversational", "warm"],
            "concise": ["brevity", "brief"],
            "technical": ["technical", "terminology"],
            "teacher": ["explanatory", "learning"],
            "motivational": ["encouraging", "positive"],
            "analytical": ["logical", "reasoning"],
        }

        for tone, keywords in expected_keywords.items():
            guidance = tone_manager.TONE_GUIDANCE[tone].lower()
            # At least one keyword should be present
            assert any(keyword in guidance for keyword in keywords), \
                f"Tone '{tone}' guidance should contain at least one of {keywords}"


class TestToneManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_user_preference(self, tone_manager, default_context):
        """Empty string user preference should fall back to context-based selection."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference="",
        )
        # Should fall back to casual context → friendly
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"

    def test_whitespace_user_preference(self, tone_manager, default_context):
        """Whitespace user preference should fall back to context-based selection."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference="   ",
        )
        # Should fall back to casual context → friendly
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"

    def test_case_sensitive_user_preference(self, tone_manager, default_context):
        """User preference should be case-sensitive (uppercase should not match)."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference="ANALYTICAL",
        )
        # Should fall back because "ANALYTICAL" != "analytical"
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"

    def test_empty_mode_string(self, tone_manager, default_context):
        """Empty mode string should not trigger teacher mode."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="",
            user_preference=None,
        )
        # Should fall back to casual context → friendly
        assert result.tone == "friendly"
        assert result.rationale == "casual context detected"

    def test_multiple_calls_with_different_contexts(self, tone_manager):
        """Multiple calls with different contexts should produce different results."""
        casual_context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        formal_context = AdaptationContext(
            tone="formal",
            style="balanced",
            focus="high-level",
            reasons={},
        )

        result1 = tone_manager.select_tone(
            context=casual_context,
            mode="chat",
            user_preference=None,
        )
        result2 = tone_manager.select_tone(
            context=formal_context,
            mode="chat",
            user_preference=None,
        )

        assert result1.tone == "friendly"
        assert result2.tone == "professional"
        assert result1.tone != result2.tone

    def test_context_signals_preserved_across_fallback(self, tone_manager, default_context):
        """Context signals should be preserved even when falling back from invalid preference."""
        result = tone_manager.select_tone(
            context=default_context,
            mode="chat",
            user_preference="invalid_tone",
        )
        # Should still record the invalid preference in context signals
        assert result.context_signals["user_preference"] == "invalid_tone"
        assert result.context_signals["mode"] == "chat"
        assert result.context_signals["context_tone"] == "casual"

