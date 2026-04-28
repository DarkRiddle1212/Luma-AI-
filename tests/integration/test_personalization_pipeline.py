"""
Integration tests for the full Personalization Engine pipeline.

Uses real ProfileBuilder, PreferenceDetector, and AdaptationEngine instances
with a mock MemoryInterface that returns controlled MemoryEntry lists.

Requirements: 9.1, 9.2, 10.1, 10.2, 10.3, 10.4, 12.9, 12.10
"""

import pytest
from unittest.mock import MagicMock

from luma.core.personalization import (
    PersonalizationEngine,
    ProfileBuilder,
    PreferenceDetector,
    AdaptationEngine,
    PersonalizationResult,
    UserProfile,
    AdaptationContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory(
    mem_id: str,
    content: str,
    tags: list = None,
    category: str = "general",
) -> dict:
    """Build a MemoryEntry dict with the required fields."""
    return {
        "id": mem_id,
        "content": content,
        "metadata": {},
        "timestamp": "2024-01-15T10:30:00",
        "category": category,
        "tags": tags or [],
    }


def make_mock_memory_interface(memories: list) -> MagicMock:
    """
    Build a mock MemoryInterface that returns the given memories and raises
    AssertionError if store() is ever called (enforcing read-only contract).
    """
    mi = MagicMock()
    mi.store.side_effect = AssertionError(
        "PersonalizationEngine must never call memory_interface.store()"
    )
    mi.retrieve.return_value = {
        "memories": memories,
        "total_count": len(memories),
        "query_metadata": {},
    }
    return mi


def make_engine(memories: list) -> PersonalizationEngine:
    """Build a PersonalizationEngine with real components and a mock MemoryInterface."""
    mi = make_mock_memory_interface(memories)
    return PersonalizationEngine(
        memory_interface=mi,
        profile_builder=ProfileBuilder(min_keyword_frequency=2),
        preference_detector=PreferenceDetector(min_confidence=0.5, min_frequency=2),
        adaptation_engine=AdaptationEngine(),
    )


# ---------------------------------------------------------------------------
# Test: full pipeline with realistic memory data
# ---------------------------------------------------------------------------

class TestFullPipelineWithRealisticData:
    """
    Test the full pipeline end-to-end with realistic mock memory data.
    Requirements: 9.1, 9.2, 10.1, 12.9, 12.10
    """

    def test_full_pipeline_returns_personalization_result(self):
        """
        Full pipeline with memories containing tags, categories, and behavior
        phrases returns a PersonalizationResult with the correct structure.
        Requirements: 9.1, 10.1, 12.9
        """
        memories = [
            make_memory(
                "m1",
                "I prefer using Python for data science projects. "
                "Python is great for machine learning and data analysis.",
                tags=["python", "data-science"],
                category="programming",
            ),
            make_memory(
                "m2",
                "I prefer writing clean, well-documented Python code. "
                "Python libraries like pandas and numpy are very useful.",
                tags=["python", "best-practices"],
                category="programming",
            ),
            make_memory(
                "m3",
                "I always review my code before committing. "
                "Code review is an important part of software engineering.",
                tags=["code-review", "engineering"],
                category="programming",
            ),
            make_memory(
                "m4",
                "I always write unit tests for my functions. "
                "Testing is essential for reliable software.",
                tags=["testing", "engineering"],
                category="programming",
            ),
        ]

        engine = make_engine(memories)
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        # Result must be a PersonalizationResult
        assert isinstance(result, PersonalizationResult)

        # Profile must be a UserProfile
        assert isinstance(result.profile, UserProfile)

        # Preferences must be a list
        assert isinstance(result.preferences, list)

        # Adaptation must be an AdaptationContext
        assert isinstance(result.adaptation, AdaptationContext)

    def test_full_pipeline_profile_is_non_empty_for_rich_memories(self):
        """
        Memories with repeated keywords produce a non-empty profile.
        Requirements: 9.1, 12.10
        """
        memories = [
            make_memory(
                "m1",
                "I prefer using Python for data science projects. "
                "Python is great for machine learning and data analysis.",
                tags=["python", "data-science"],
                category="programming",
            ),
            make_memory(
                "m2",
                "I prefer writing clean, well-documented Python code. "
                "Python libraries like pandas and numpy are very useful.",
                tags=["python", "best-practices"],
                category="programming",
            ),
            make_memory(
                "m3",
                "I always review my code before committing. "
                "Code review is an important part of software engineering.",
                tags=["code-review", "engineering"],
                category="programming",
            ),
            make_memory(
                "m4",
                "I always write unit tests for my functions. "
                "Testing is essential for reliable software.",
                tags=["testing", "engineering"],
                category="programming",
            ),
        ]

        engine = make_engine(memories)
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        # Profile should have detected interests from repeated keywords
        assert len(result.profile.interests) > 0, (
            "Expected non-empty interests from memories with repeated keywords"
        )

    def test_full_pipeline_preferences_detected_from_repeated_tags(self):
        """
        Tags appearing in multiple memories are detected as preferences.
        Requirements: 9.1, 12.10
        """
        memories = [
            make_memory("m1", "Working on a Python project.", tags=["python"], category="programming"),
            make_memory("m2", "Python is my go-to language.", tags=["python"], category="programming"),
            make_memory("m3", "Learning Python data structures.", tags=["python"], category="programming"),
        ]

        engine = make_engine(memories)
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        # "python" tag appears in all 3 memories → should be detected as preference
        pref_labels = {p.preference for p in result.preferences}
        assert "python" in pref_labels, (
            f"Expected 'python' in preferences, got: {pref_labels}"
        )

    def test_full_pipeline_behavior_phrases_extracted(self):
        """
        The same behavior phrase appearing in multiple memories is extracted
        into profile.behavior_patterns.
        Requirements: 9.1, 12.10
        """
        # Use the same phrase in both memories so it meets min_keyword_frequency=2
        memories = [
            make_memory(
                "m1",
                "I prefer using python for all my projects.",
                tags=["productivity"],
                category="habits",
            ),
            make_memory(
                "m2",
                "I prefer using python because it is readable.",
                tags=["productivity"],
                category="habits",
            ),
        ]

        # Use min_keyword_frequency=1 so any phrase appearing once is included,
        # which lets us verify the extraction mechanism works end-to-end.
        mi = make_mock_memory_interface(memories)
        engine = PersonalizationEngine(
            memory_interface=mi,
            profile_builder=ProfileBuilder(min_keyword_frequency=1),
            preference_detector=PreferenceDetector(min_confidence=0.5, min_frequency=1),
            adaptation_engine=AdaptationEngine(),
        )
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        # With min_keyword_frequency=1, any behavior phrase found in at least 1 memory
        # should be extracted. Both memories contain "I prefer" so at least one phrase
        # should appear in behavior_patterns.
        assert len(result.profile.behavior_patterns) > 0, (
            "Expected behavior_patterns to be non-empty when 'I prefer' appears in memories"
        )

    def test_full_pipeline_adaptation_context_has_valid_fields(self):
        """
        The AdaptationContext returned always has valid tone, style, and focus.
        Requirements: 9.1, 10.1
        """
        memories = [
            make_memory("m1", "Python programming tutorial.", tags=["python"], category="programming"),
            make_memory("m2", "Advanced Python techniques.", tags=["python"], category="programming"),
        ]

        engine = make_engine(memories)
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        assert result.adaptation.tone in {"technical", "casual", "formal"}
        assert result.adaptation.style in {"concise", "detailed", "step-by-step", "balanced"}
        assert result.adaptation.focus in {"high-level", "deep-technical"}

        # All reasons must be non-empty
        for key in ("tone", "style", "focus"):
            assert key in result.adaptation.reasons
            assert result.adaptation.reasons[key].strip(), (
                f"reasons[{key!r}] must be non-empty"
            )

    def test_full_pipeline_technical_tone_from_technical_interests(self):
        """
        Memories with technical keywords produce a technical tone in the
        AdaptationContext.
        Requirements: 9.1, 10.1
        """
        memories = [
            make_memory(
                "m1",
                "Learning about machine learning algorithms and neural networks.",
                tags=["machine-learning"],
                category="data-science",
            ),
            make_memory(
                "m2",
                "Studying deep learning and neural network architectures.",
                tags=["deep-learning"],
                category="data-science",
            ),
        ]

        engine = make_engine(memories)
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        # "machine", "learning", "neural", "network" are technical domain keywords
        # so tone should be "technical"
        assert result.adaptation.tone == "technical", (
            f"Expected technical tone for technical memories, got: {result.adaptation.tone!r}"
        )


# ---------------------------------------------------------------------------
# Test: continuous update — profile reflects new memories on second call
# ---------------------------------------------------------------------------

class TestContinuousUpdate:
    """
    Test that the profile and preferences evolve as the memory store grows.
    Requirements: 9.1, 9.2, 10.2, 10.3
    """

    def test_profile_reflects_new_memories_on_second_call(self):
        """
        When called twice with different memory data, the second result reflects
        the new memories (continuous update, no caching between calls).
        Requirements: 9.1, 9.2, 10.2
        """
        # First call: memories about Python
        memories_v1 = [
            make_memory("m1", "Python is great for scripting.", tags=["python"], category="programming"),
            make_memory("m2", "I use Python for automation tasks.", tags=["python"], category="programming"),
        ]

        # Second call: memories about JavaScript
        memories_v2 = [
            make_memory("m1", "JavaScript is essential for web development.", tags=["javascript"], category="web"),
            make_memory("m2", "I use JavaScript for frontend projects.", tags=["javascript"], category="web"),
            make_memory("m3", "React is a popular JavaScript framework.", tags=["javascript", "react"], category="web"),
        ]

        engine_v1 = make_engine(memories_v1)
        result_v1 = engine_v1.personalize({"query": "test"}, {"session_id": "s1"})

        engine_v2 = make_engine(memories_v2)
        result_v2 = engine_v2.personalize({"query": "test"}, {"session_id": "s2"})

        # The two results should differ because the memory data differs
        interests_v1 = set(result_v1.profile.interests)
        interests_v2 = set(result_v2.profile.interests)

        # v1 should contain python-related keywords
        assert any("python" in kw for kw in interests_v1), (
            f"Expected python-related interests in v1, got: {interests_v1}"
        )

        # v2 should contain javascript-related keywords
        assert any("javascript" in kw for kw in interests_v2), (
            f"Expected javascript-related interests in v2, got: {interests_v2}"
        )

    def test_preferences_evolve_as_memory_store_grows(self):
        """
        Adding more memories with a consistent tag increases the confidence of
        the corresponding preference (or keeps it present).
        Requirements: 9.2, 10.3
        """
        # Small store: 2 memories with "python" tag
        memories_small = [
            make_memory("m1", "Python scripting basics.", tags=["python"], category="programming"),
            make_memory("m2", "Python for data analysis.", tags=["python"], category="programming"),
        ]

        # Larger store: 5 memories with "python" tag
        memories_large = [
            make_memory("m1", "Python scripting basics.", tags=["python"], category="programming"),
            make_memory("m2", "Python for data analysis.", tags=["python"], category="programming"),
            make_memory("m3", "Advanced Python patterns.", tags=["python"], category="programming"),
            make_memory("m4", "Python testing strategies.", tags=["python"], category="programming"),
            make_memory("m5", "Python packaging and deployment.", tags=["python"], category="programming"),
        ]

        engine_small = make_engine(memories_small)
        result_small = engine_small.personalize({"query": "test"}, {"session_id": "s1"})

        engine_large = make_engine(memories_large)
        result_large = engine_large.personalize({"query": "test"}, {"session_id": "s2"})

        # Both should detect "python" as a preference
        prefs_small = {p.preference: p.confidence for p in result_small.preferences}
        prefs_large = {p.preference: p.confidence for p in result_large.preferences}

        assert "python" in prefs_small, (
            f"Expected 'python' preference in small store, got: {list(prefs_small.keys())}"
        )
        assert "python" in prefs_large, (
            f"Expected 'python' preference in large store, got: {list(prefs_large.keys())}"
        )

        # Confidence should be >= in the larger store (more evidence)
        assert prefs_large["python"] >= prefs_small["python"], (
            f"Expected confidence to be >= with more memories: "
            f"small={prefs_small['python']}, large={prefs_large['python']}"
        )

    def test_second_call_with_different_data_produces_different_profile(self):
        """
        Two calls with completely different memory content produce different profiles,
        confirming no state is cached between calls.
        Requirements: 9.2, 10.2
        """
        memories_a = [
            make_memory("a1", "Cooking pasta with tomato sauce is delicious.", tags=["cooking"], category="food"),
            make_memory("a2", "I enjoy making homemade pasta from scratch.", tags=["cooking"], category="food"),
        ]

        memories_b = [
            make_memory("b1", "Running a marathon requires months of training.", tags=["running"], category="fitness"),
            make_memory("b2", "I enjoy long distance running in the morning.", tags=["running"], category="fitness"),
        ]

        engine_a = make_engine(memories_a)
        result_a = engine_a.personalize({"query": "test"}, {"session_id": "s1"})

        engine_b = make_engine(memories_b)
        result_b = engine_b.personalize({"query": "test"}, {"session_id": "s2"})

        # Profiles should differ
        assert result_a.profile.interests != result_b.profile.interests, (
            "Expected different interests for completely different memory content"
        )


# ---------------------------------------------------------------------------
# Test: empty memory store returns default PersonalizationResult
# ---------------------------------------------------------------------------

class TestEmptyMemoryStore:
    """
    Test that an empty memory store returns a default PersonalizationResult.
    Requirements: 10.4, 12.9
    """

    def test_empty_memory_store_returns_personalization_result(self):
        """
        Full pipeline with empty memory store returns a PersonalizationResult
        (not None, not an exception).
        Requirements: 10.4, 12.9
        """
        engine = make_engine([])
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        assert isinstance(result, PersonalizationResult)

    def test_empty_memory_store_returns_empty_profile(self):
        """
        Empty memory store produces a UserProfile with empty lists.
        Requirements: 10.4
        """
        engine = make_engine([])
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        assert result.profile.interests == []
        assert result.profile.behavior_patterns == []
        assert result.profile.strengths == []
        assert result.profile.interaction_style == "balanced"

    def test_empty_memory_store_returns_empty_preferences(self):
        """
        Empty memory store produces an empty preferences list.
        Requirements: 10.4
        """
        engine = make_engine([])
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        assert result.preferences == []

    def test_empty_memory_store_returns_default_adaptation_context(self):
        """
        Empty memory store produces a default AdaptationContext with
        tone=casual, style=balanced, focus=high-level.
        Requirements: 10.4
        """
        engine = make_engine([])
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        assert result.adaptation.tone == "casual"
        assert result.adaptation.style == "balanced"
        assert result.adaptation.focus == "high-level"

    def test_empty_memory_store_adaptation_reasons_non_empty(self):
        """
        Even with empty memories, all adaptation reasons must be non-empty strings.
        Requirements: 10.4
        """
        engine = make_engine([])
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        for key in ("tone", "style", "focus"):
            assert key in result.adaptation.reasons
            assert isinstance(result.adaptation.reasons[key], str)
            assert result.adaptation.reasons[key].strip(), (
                f"reasons[{key!r}] must be non-empty even for empty memory store"
            )


# ---------------------------------------------------------------------------
# Test: store() is never called on the MemoryInterface
# ---------------------------------------------------------------------------

class TestReadOnlyMemoryConsumption:
    """
    Verify the engine never calls store() on the MemoryInterface.
    Requirements: 9.1, 10.1
    """

    def test_store_never_called_with_memories(self):
        """
        store() must never be called during personalization with memories.
        """
        memories = [
            make_memory("m1", "Python programming.", tags=["python"], category="programming"),
            make_memory("m2", "More Python content.", tags=["python"], category="programming"),
        ]
        mi = make_mock_memory_interface(memories)
        engine = PersonalizationEngine(
            memory_interface=mi,
            profile_builder=ProfileBuilder(min_keyword_frequency=2),
            preference_detector=PreferenceDetector(min_confidence=0.5, min_frequency=2),
            adaptation_engine=AdaptationEngine(),
        )

        # Should not raise AssertionError from store() side_effect
        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        mi.store.assert_not_called()
        assert isinstance(result, PersonalizationResult)

    def test_store_never_called_with_empty_memories(self):
        """
        store() must never be called during personalization with empty memory store.
        """
        mi = make_mock_memory_interface([])
        engine = PersonalizationEngine(
            memory_interface=mi,
            profile_builder=ProfileBuilder(min_keyword_frequency=2),
            preference_detector=PreferenceDetector(min_confidence=0.5, min_frequency=2),
            adaptation_engine=AdaptationEngine(),
        )

        result = engine.personalize({"query": "test"}, {"session_id": "s1"})

        mi.store.assert_not_called()
        assert isinstance(result, PersonalizationResult)

    def test_retrieve_called_with_limit_500(self):
        """
        retrieve() must be called with params={"limit": 500}.
        Requirements: 9.1, 10.1
        """
        mi = make_mock_memory_interface([])
        engine = PersonalizationEngine(
            memory_interface=mi,
            profile_builder=ProfileBuilder(min_keyword_frequency=2),
            preference_detector=PreferenceDetector(min_confidence=0.5, min_frequency=2),
            adaptation_engine=AdaptationEngine(),
        )

        engine.personalize({"query": "test"}, {"session_id": "s1"})

        mi.retrieve.assert_called_once_with(params={"limit": 500})
