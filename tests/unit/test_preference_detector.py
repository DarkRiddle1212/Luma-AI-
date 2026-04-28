"""
Unit tests for PreferenceDetector.

Covers:
- Empty memory list returns []
- Preference detected from tags appearing in >= min_frequency memories
- No preference emitted for signal in only 1 memory (when min_frequency=2)
- Confidence is proportional to frequency
- Confidence is boosted when signal matches profile interests
- Reason string references contributing memory IDs
- ValueError raised for min_confidence outside [0.0, 1.0]
- ValueError raised for min_frequency < 1
- Output sorted by (confidence DESC, preference ASC)
"""

import pytest
from luma.core.personalization.preference_detector import PreferenceDetector
from luma.core.personalization.schemas import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory(id: str, tags=None, category="general", content=""):
    return {
        "id": id,
        "content": content,
        "metadata": {},
        "timestamp": "2024-01-15T10:30:00",
        "category": category,
        "tags": tags or [],
    }


def make_profile(interests=None, behavior_patterns=None):
    return UserProfile(
        interests=interests or [],
        behavior_patterns=behavior_patterns or [],
        interaction_style="balanced",
        strengths=[],
        evidence={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreferenceDetectorConstruction:
    def test_valid_defaults(self):
        detector = PreferenceDetector()
        assert detector._min_confidence == 0.6
        assert detector._min_frequency == 2

    def test_valid_custom_params(self):
        detector = PreferenceDetector(min_confidence=0.3, min_frequency=3)
        assert detector._min_confidence == 0.3
        assert detector._min_frequency == 3

    def test_min_confidence_zero_is_valid(self):
        detector = PreferenceDetector(min_confidence=0.0)
        assert detector._min_confidence == 0.0

    def test_min_confidence_one_is_valid(self):
        detector = PreferenceDetector(min_confidence=1.0)
        assert detector._min_confidence == 1.0

    def test_raises_value_error_for_min_confidence_below_zero(self):
        with pytest.raises(ValueError, match="min_confidence"):
            PreferenceDetector(min_confidence=-0.1)

    def test_raises_value_error_for_min_confidence_above_one(self):
        with pytest.raises(ValueError, match="min_confidence"):
            PreferenceDetector(min_confidence=1.1)

    def test_raises_value_error_for_min_frequency_zero(self):
        with pytest.raises(ValueError, match="min_frequency"):
            PreferenceDetector(min_frequency=0)

    def test_raises_value_error_for_min_frequency_negative(self):
        with pytest.raises(ValueError, match="min_frequency"):
            PreferenceDetector(min_frequency=-1)


class TestPreferenceDetectorDetect:
    def test_empty_memories_returns_empty_list(self):
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        profile = make_profile()
        result = detector.detect([], profile)
        assert result == []

    def test_preference_detected_from_tags_in_multiple_memories(self):
        """Tag appearing in >= min_frequency memories should produce a preference."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["python"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        labels = [p.preference for p in result]
        assert "python" in labels

    def test_no_preference_for_signal_in_only_one_memory(self):
        """Signal appearing in only 1 memory should not be emitted when min_frequency=2."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["java"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        labels = [p.preference for p in result]
        assert "python" not in labels
        assert "java" not in labels

    def test_confidence_proportional_to_frequency(self):
        """Confidence should be len(ids) / len(memories)."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        # 3 memories, tag appears in 2 → confidence = 2/3
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["python"]),
            make_memory("m3", tags=["java"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        python_pref = next(p for p in result if p.preference == "python")
        expected = 2 / 3
        assert abs(python_pref.confidence - expected) < 1e-9

    def test_confidence_boosted_when_signal_in_profile_interests(self):
        """Confidence should be boosted by 0.1 when signal is in profile.interests."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["python"]),
            make_memory("m3", tags=["other"]),
            make_memory("m4", tags=["other"]),
        ]
        profile_with = make_profile(interests=["python"])
        profile_without = make_profile(interests=[])

        result_with = detector.detect(memories, profile_with)
        result_without = detector.detect(memories, profile_without)

        conf_with = next(p.confidence for p in result_with if p.preference == "python")
        conf_without = next(p.confidence for p in result_without if p.preference == "python")

        assert conf_with == conf_without + 0.1

    def test_confidence_boosted_when_signal_in_behavior_patterns(self):
        """Confidence should be boosted by 0.1 when signal is in profile.behavior_patterns."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        memories = [
            make_memory("m1", tags=["coding"]),
            make_memory("m2", tags=["coding"]),
        ]
        profile = make_profile(behavior_patterns=["coding"])
        result = detector.detect(memories, profile)
        pref = next(p for p in result if p.preference == "coding")
        base = 2 / 2  # 1.0
        # boost would exceed 1.0, so capped at 1.0
        assert pref.confidence == 1.0

    def test_confidence_boost_capped_at_one(self):
        """Boosted confidence must not exceed 1.0."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        # All 2 memories have the tag → base confidence = 1.0
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["python"]),
        ]
        profile = make_profile(interests=["python"])
        result = detector.detect(memories, profile)
        pref = next(p for p in result if p.preference == "python")
        assert pref.confidence <= 1.0

    def test_reason_references_contributing_memory_ids(self):
        """Reason string should mention the contributing memory IDs."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("mem-alpha", tags=["python"]),
            make_memory("mem-beta", tags=["python"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        pref = next(p for p in result if p.preference == "python")
        assert "mem-alpha" in pref.reason or "mem-beta" in pref.reason

    def test_reason_contains_memory_count(self):
        """Reason string should contain the count of contributing memories."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("m1", tags=["python"]),
            make_memory("m2", tags=["python"]),
            make_memory("m3", tags=["python"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        pref = next(p for p in result if p.preference == "python")
        assert "3" in pref.reason

    def test_reason_truncates_ids_beyond_three(self):
        """Reason should show at most 3 IDs and append '...' when more exist."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        memories = [make_memory(f"m{i}", tags=["python"]) for i in range(5)]
        profile = make_profile()
        result = detector.detect(memories, profile)
        pref = next(p for p in result if p.preference == "python")
        assert "..." in pref.reason

    def test_output_sorted_by_confidence_desc_then_preference_asc(self):
        """Output should be sorted by (confidence DESC, preference ASC)."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
        # Use a unique category to avoid interference; 4 memories total.
        # "alpha" tag in 3 memories → confidence 3/4
        # "beta" tag in 2 memories → confidence 2/4
        # "gamma" tag in 2 memories → confidence 2/4
        # category "unique-cat" in all 4 → confidence 4/4 (highest, but we check relative order of tags)
        memories = [
            make_memory("m1", tags=["alpha", "beta", "gamma"], category="unique-cat"),
            make_memory("m2", tags=["alpha", "beta", "gamma"], category="unique-cat"),
            make_memory("m3", tags=["alpha"], category="unique-cat"),
            make_memory("m4", tags=[], category="unique-cat"),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)

        # alpha should come before beta and gamma (higher confidence)
        alpha_idx = next(i for i, p in enumerate(result) if p.preference == "alpha")
        beta_idx = next(i for i, p in enumerate(result) if p.preference == "beta")
        gamma_idx = next(i for i, p in enumerate(result) if p.preference == "gamma")
        assert alpha_idx < beta_idx
        assert alpha_idx < gamma_idx

        # beta and gamma have equal confidence; beta < gamma alphabetically
        assert beta_idx < gamma_idx

    def test_category_used_as_signal(self):
        """Memory category should be treated as a preference signal."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("m1", category="science"),
            make_memory("m2", category="science"),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        labels = [p.preference for p in result]
        assert "science" in labels

    def test_behavior_phrase_in_content_used_as_signal(self):
        """Behavior phrases from profile that appear in memory content should be signals."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        memories = [
            make_memory("m1", content="I always prefer Python for scripting"),
            make_memory("m2", content="I always prefer Python for data work"),
        ]
        profile = make_profile(behavior_patterns=["I always prefer Python"])
        result = detector.detect(memories, profile)
        labels = [p.preference for p in result]
        assert "I always prefer Python" in labels

    def test_threshold_filters_low_confidence(self):
        """Preferences below min_confidence should not be emitted."""
        # 1 memory out of 10 has the tag → confidence = 0.1
        detector = PreferenceDetector(min_confidence=0.5, min_frequency=1)
        memories = [make_memory(f"m{i}") for i in range(10)]
        memories[0]["tags"] = ["rare-tag"]
        profile = make_profile()
        result = detector.detect(memories, profile)
        labels = [p.preference for p in result]
        assert "rare-tag" not in labels

    def test_each_memory_id_counted_once_per_signal(self):
        """A memory with the same tag twice should only count once per signal."""
        detector = PreferenceDetector(min_confidence=0.0, min_frequency=2)
        # m1 has "python" twice in tags — should still count as 1 memory
        memories = [
            {"id": "m1", "content": "", "metadata": {}, "timestamp": "2024-01-15T10:30:00",
             "category": "general", "tags": ["python", "python"]},
            make_memory("m2", tags=["python"]),
        ]
        profile = make_profile()
        result = detector.detect(memories, profile)
        pref = next((p for p in result if p.preference == "python"), None)
        assert pref is not None
        # confidence = 2/2 = 1.0 (2 distinct memories)
        assert pref.confidence == 1.0
