"""
Unit tests for PersonalizationEngine.

Tests cover:
- retrieve() called with {"limit": 500} params
- store() is never called
- profile_builder.build() called with retrieved memories and insights
- preference_detector.detect() called with memories and profile
- adaptation_engine.adapt() called with profile and preferences
- Returns PersonalizationResult with all three components
- Returns empty profile and preferences when memory store is empty
- MemoryRetrievalError is propagated
- Optional insights passed to profile_builder when provided
- TypeError raised for None input_data
- TypeError raised for None context
"""

import pytest
from unittest.mock import MagicMock, call

from luma.core.memory_interface import MemoryRetrievalError
from luma.core.personalization.personalization_engine import PersonalizationEngine
from luma.core.personalization.schemas import (
    AdaptationContext,
    PersonalizationResult,
    Preference,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(mem_id: str, content: str = "some content") -> dict:
    return {
        "id": mem_id,
        "content": content,
        "metadata": {},
        "timestamp": "2024-01-15T10:30:00",
        "category": "general",
        "tags": [],
    }


def _make_retrieval_result(memories: list) -> dict:
    return {
        "memories": memories,
        "total_count": len(memories),
        "query_metadata": {},
    }


def _make_empty_profile() -> UserProfile:
    return UserProfile(
        interests=[],
        behavior_patterns=[],
        interaction_style="balanced",
        strengths=[],
        evidence={},
    )


def _make_adaptation_context() -> AdaptationContext:
    return AdaptationContext(
        tone="casual",
        style="balanced",
        focus="high-level",
        reasons={"tone": "default", "style": "default", "focus": "default"},
    )


def _make_engine_with_mocks(memories=None, profile=None, preferences=None, adaptation=None):
    """Build a PersonalizationEngine with fully mocked dependencies."""
    if memories is None:
        memories = [_make_memory("m1")]
    if profile is None:
        profile = _make_empty_profile()
    if preferences is None:
        preferences = []
    if adaptation is None:
        adaptation = _make_adaptation_context()

    memory_interface = MagicMock()
    # store() raises AssertionError if called — it must never be called
    memory_interface.store.side_effect = AssertionError(
        "PersonalizationEngine must never call memory_interface.store()"
    )
    memory_interface.retrieve.return_value = _make_retrieval_result(memories)

    profile_builder = MagicMock()
    profile_builder.build.return_value = profile

    preference_detector = MagicMock()
    preference_detector.detect.return_value = preferences

    adaptation_engine = MagicMock()
    adaptation_engine.adapt.return_value = adaptation

    engine = PersonalizationEngine(
        memory_interface=memory_interface,
        profile_builder=profile_builder,
        preference_detector=preference_detector,
        adaptation_engine=adaptation_engine,
    )

    return engine, memory_interface, profile_builder, preference_detector, adaptation_engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPersonalizationEngineRetrieve:
    def test_retrieve_called_with_limit_500(self):
        engine, memory_interface, _, _, _ = _make_engine_with_mocks()
        engine.personalize("input", "context")
        memory_interface.retrieve.assert_called_once_with(params={"limit": 500})

    def test_store_is_never_called(self):
        engine, memory_interface, _, _, _ = _make_engine_with_mocks()
        engine.personalize("input", "context")
        memory_interface.store.assert_not_called()

    def test_store_raises_if_called(self):
        """Verify the mock is wired to raise AssertionError on store()."""
        engine, memory_interface, _, _, _ = _make_engine_with_mocks()
        with pytest.raises(AssertionError):
            memory_interface.store("should not be called")


class TestPersonalizationEnginePipelineCalls:
    def test_profile_builder_called_with_memories_and_empty_insights(self):
        memories = [_make_memory("m1"), _make_memory("m2")]
        engine, _, profile_builder, _, _ = _make_engine_with_mocks(memories=memories)
        engine.personalize("input", "context")
        profile_builder.build.assert_called_once_with(memories, [])

    def test_profile_builder_called_with_provided_insights(self):
        memories = [_make_memory("m1")]
        insights = [{"text": "insight text", "confidence": 0.9, "evidence": ["m1"]}]
        engine, _, profile_builder, _, _ = _make_engine_with_mocks(memories=memories)
        engine.personalize("input", "context", insights=insights)
        profile_builder.build.assert_called_once_with(memories, insights)

    def test_preference_detector_called_with_memories_and_profile(self):
        memories = [_make_memory("m1")]
        profile = _make_empty_profile()
        engine, _, _, preference_detector, _ = _make_engine_with_mocks(
            memories=memories, profile=profile
        )
        engine.personalize("input", "context")
        preference_detector.detect.assert_called_once_with(memories, profile)

    def test_adaptation_engine_called_with_profile_and_preferences(self):
        profile = _make_empty_profile()
        preferences = [
            Preference(preference="python", confidence=0.8, reason="Detected in 2 memories: m1, m2")
        ]
        engine, _, _, _, adaptation_engine = _make_engine_with_mocks(
            profile=profile, preferences=preferences
        )
        engine.personalize("input", "context")
        adaptation_engine.adapt.assert_called_once_with(profile, preferences)


class TestPersonalizationEngineResult:
    def test_returns_personalization_result_with_all_three_components(self):
        profile = _make_empty_profile()
        preferences = []
        adaptation = _make_adaptation_context()
        engine, _, _, _, _ = _make_engine_with_mocks(
            profile=profile, preferences=preferences, adaptation=adaptation
        )
        result = engine.personalize("input", "context")
        assert isinstance(result, PersonalizationResult)
        assert result.profile == profile
        assert result.preferences == preferences
        assert result.adaptation == adaptation

    def test_returns_empty_profile_and_preferences_when_memory_store_empty(self):
        empty_profile = _make_empty_profile()
        empty_preferences = []
        adaptation = _make_adaptation_context()

        engine, _, profile_builder, preference_detector, _ = _make_engine_with_mocks(
            memories=[],
            profile=empty_profile,
            preferences=empty_preferences,
            adaptation=adaptation,
        )
        result = engine.personalize("input", "context")

        # profile_builder.build() should be called with empty memories
        profile_builder.build.assert_called_once_with([], [])
        # preference_detector.detect() should be called with empty memories
        preference_detector.detect.assert_called_once_with([], empty_profile)

        assert result.profile.interests == []
        assert result.profile.behavior_patterns == []
        assert result.preferences == []


class TestPersonalizationEngineErrors:
    def test_memory_retrieval_error_is_propagated(self):
        engine, memory_interface, _, _, _ = _make_engine_with_mocks()
        memory_interface.retrieve.side_effect = MemoryRetrievalError("DB connection failed")

        with pytest.raises(MemoryRetrievalError) as exc_info:
            engine.personalize("input", "context")

        assert "DB connection failed" in str(exc_info.value)

    def test_type_error_raised_for_none_input_data(self):
        engine, _, _, _, _ = _make_engine_with_mocks()
        with pytest.raises(TypeError):
            engine.personalize(None, "context")

    def test_type_error_raised_for_none_context(self):
        engine, _, _, _, _ = _make_engine_with_mocks()
        with pytest.raises(TypeError):
            engine.personalize("input", None)

    def test_type_error_raised_for_both_none(self):
        engine, _, _, _, _ = _make_engine_with_mocks()
        with pytest.raises(TypeError):
            engine.personalize(None, None)


class TestPersonalizationEngineInsights:
    def test_insights_none_defaults_to_empty_list(self):
        memories = [_make_memory("m1")]
        engine, _, profile_builder, _, _ = _make_engine_with_mocks(memories=memories)
        engine.personalize("input", "context", insights=None)
        profile_builder.build.assert_called_once_with(memories, [])

    def test_insights_passed_through_to_profile_builder(self):
        memories = [_make_memory("m1")]
        insights = [
            {"text": "User is skilled at Python", "confidence": 0.85, "evidence": ["m1"]},
            {"text": "User prefers concise answers", "confidence": 0.75, "evidence": ["m1"]},
        ]
        engine, _, profile_builder, _, _ = _make_engine_with_mocks(memories=memories)
        engine.personalize("input", "context", insights=insights)
        profile_builder.build.assert_called_once_with(memories, insights)
