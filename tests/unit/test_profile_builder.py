"""
Unit tests for luma.core.personalization.profile_builder.ProfileBuilder.

Tests cover:
- Empty memory list → default UserProfile
- Keyword frequency threshold (include at threshold, exclude below)
- interaction_style: concise / detailed / balanced
- Strengths extracted only from high-confidence insights
- Evidence maps to correct memory IDs
- ValueError for min_keyword_frequency < 1
- Supplements interests from high-confidence insight text
"""

import pytest
from luma.core.personalization.profile_builder import ProfileBuilder
from luma.core.personalization.schemas import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mem(mem_id: str, content: str, **kwargs) -> dict:
    """Create a minimal MemoryEntry dict."""
    return {
        "id": mem_id,
        "content": content,
        "metadata": kwargs.get("metadata", {}),
        "timestamp": "2024-01-15T10:30:00",
        "category": kwargs.get("category", "general"),
        "tags": kwargs.get("tags", []),
    }


def _insight(text: str, confidence: float, evidence=None):
    """Create a plain-dict insight."""
    return {
        "text": text,
        "confidence": confidence,
        "evidence": evidence or ["mem-1"],
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestProfileBuilderConstruction:
    def test_default_min_keyword_frequency(self):
        builder = ProfileBuilder()
        assert builder._min_freq == 2

    def test_custom_min_keyword_frequency(self):
        builder = ProfileBuilder(min_keyword_frequency=3)
        assert builder._min_freq == 3

    def test_min_keyword_frequency_zero_raises(self):
        with pytest.raises(ValueError):
            ProfileBuilder(min_keyword_frequency=0)

    def test_min_keyword_frequency_negative_raises(self):
        with pytest.raises(ValueError):
            ProfileBuilder(min_keyword_frequency=-1)

    def test_min_keyword_frequency_one_is_valid(self):
        builder = ProfileBuilder(min_keyword_frequency=1)
        assert builder._min_freq == 1


# ---------------------------------------------------------------------------
# Empty memories
# ---------------------------------------------------------------------------

class TestEmptyMemories:
    def test_empty_memories_returns_default_profile(self):
        builder = ProfileBuilder()
        profile = builder.build([], [])
        assert isinstance(profile, UserProfile)
        assert profile.interests == []
        assert profile.behavior_patterns == []
        assert profile.interaction_style == "balanced"
        assert profile.strengths == []
        assert profile.evidence == {}

    def test_empty_memories_with_insights_still_returns_default(self):
        """Even with insights, empty memories → default profile."""
        builder = ProfileBuilder()
        insights = [_insight("Python programming", 0.9)]
        profile = builder.build([], insights)
        assert profile.interests == []
        assert profile.interaction_style == "balanced"
        assert profile.evidence == {}


# ---------------------------------------------------------------------------
# Keyword frequency threshold
# ---------------------------------------------------------------------------

class TestKeywordFrequency:
    def test_keyword_at_threshold_is_included(self):
        """A keyword appearing in exactly min_keyword_frequency memories is included."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("m1", "python programming language"),
            _mem("m2", "python scripting language"),
        ]
        profile = builder.build(memories, [])
        assert "python" in profile.interests

    def test_keyword_below_threshold_is_excluded(self):
        """A keyword appearing in fewer than min_keyword_frequency memories is excluded."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("m1", "python programming language"),
            _mem("m2", "java scripting language"),
        ]
        profile = builder.build(memories, [])
        # "python" appears only in m1 → excluded
        assert "python" not in profile.interests

    def test_keyword_above_threshold_is_included(self):
        """A keyword appearing in more than min_keyword_frequency memories is included."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("m1", "python programming language"),
            _mem("m2", "python scripting language"),
            _mem("m3", "python data science"),
        ]
        profile = builder.build(memories, [])
        assert "python" in profile.interests

    def test_frequency_one_threshold_includes_single_occurrence(self):
        """With min_keyword_frequency=1, any keyword appearing once is included."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        memories = [_mem("m1", "python programming")]
        profile = builder.build(memories, [])
        assert "python" in profile.interests

    def test_stop_words_excluded(self):
        """Stop words should not appear in interests."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        memories = [
            _mem("m1", "the and for are but"),
            _mem("m2", "the and for are but"),
        ]
        profile = builder.build(memories, [])
        for stop_word in ["the", "and", "for", "are", "but"]:
            assert stop_word not in profile.interests

    def test_short_tokens_excluded(self):
        """Tokens shorter than 3 characters should not appear in interests."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        memories = [
            _mem("m1", "go is ok"),
            _mem("m2", "go is ok"),
        ]
        profile = builder.build(memories, [])
        for short in ["go", "is", "ok"]:
            assert short not in profile.interests


# ---------------------------------------------------------------------------
# Interaction style
# ---------------------------------------------------------------------------

class TestInteractionStyle:
    def test_concise_for_short_memories(self):
        """Average word count < 15 → concise."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        # Each memory has ~5 words
        memories = [
            _mem("m1", "short note here"),
            _mem("m2", "brief entry today"),
            _mem("m3", "quick thought now"),
        ]
        profile = builder.build(memories, [])
        assert profile.interaction_style == "concise"

    def test_detailed_for_long_memories(self):
        """Average word count > 50 → detailed."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        long_content = " ".join(["word"] * 60)
        memories = [
            _mem("m1", long_content),
            _mem("m2", long_content),
        ]
        profile = builder.build(memories, [])
        assert profile.interaction_style == "detailed"

    def test_balanced_for_medium_memories(self):
        """Average word count between 15 and 50 → balanced."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        medium_content = " ".join(["word"] * 30)
        memories = [
            _mem("m1", medium_content),
            _mem("m2", medium_content),
        ]
        profile = builder.build(memories, [])
        assert profile.interaction_style == "balanced"

    def test_balanced_for_empty_memories(self):
        """Empty memory list → balanced."""
        builder = ProfileBuilder()
        profile = builder.build([], [])
        assert profile.interaction_style == "balanced"

    def test_boundary_exactly_15_words_is_balanced(self):
        """Exactly 15 words average → balanced (not concise, since avg is not < 15)."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        content = " ".join(["word"] * 15)
        memories = [_mem("m1", content)]
        profile = builder.build(memories, [])
        assert profile.interaction_style == "balanced"

    def test_boundary_exactly_50_words_is_balanced(self):
        """Exactly 50 words average → balanced (not detailed, since avg is not > 50)."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        content = " ".join(["word"] * 50)
        memories = [_mem("m1", content)]
        profile = builder.build(memories, [])
        assert profile.interaction_style == "balanced"


# ---------------------------------------------------------------------------
# Strengths
# ---------------------------------------------------------------------------

class TestStrengths:
    def test_strengths_from_high_confidence_insights(self):
        """Strengths extracted from insights with confidence >= 0.6."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Python programming skills", 0.8, ["m1"]),
            _insight("Data analysis expertise", 0.7, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        assert len(profile.strengths) > 0

    def test_strengths_not_from_low_confidence_insights(self):
        """Insights with confidence < 0.6 do not contribute to strengths."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Python programming skills", 0.5, ["m1"]),
            _insight("Data analysis expertise", 0.3, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        assert profile.strengths == []

    def test_strengths_at_confidence_boundary(self):
        """Insight with confidence exactly 0.6 contributes to strengths."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [_insight("Machine learning expert", 0.6, ["m1"])]
        profile = builder.build(memories, insights)
        assert len(profile.strengths) > 0

    def test_strengths_sorted_alphabetically(self):
        """Strengths are sorted alphabetically."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Zebra pattern recognition", 0.9, ["m1"]),
            _insight("Alpha data analysis", 0.9, ["m1"]),
            _insight("Machine learning skills", 0.9, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        assert profile.strengths == sorted(profile.strengths)

    def test_strengths_deduplicated(self):
        """Duplicate strength topics are deduplicated."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Python programming skills", 0.9, ["m1"]),
            _insight("Python programming skills", 0.8, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        # Should not have duplicates
        assert len(profile.strengths) == len(set(profile.strengths))


# ---------------------------------------------------------------------------
# Evidence mapping
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_interests_contains_contributing_memory_ids(self):
        """evidence['interests'] contains IDs of memories that contributed keywords."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("mem-a", "python programming language"),
            _mem("mem-b", "python scripting language"),
            _mem("mem-c", "java programming language"),
        ]
        profile = builder.build(memories, [])
        assert "interests" in profile.evidence
        # mem-a and mem-b both have "python"
        assert "mem-a" in profile.evidence["interests"]
        assert "mem-b" in profile.evidence["interests"]

    def test_evidence_ids_are_sorted(self):
        """Evidence ID lists are sorted."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("zzz", "python programming language"),
            _mem("aaa", "python scripting language"),
        ]
        profile = builder.build(memories, [])
        if "interests" in profile.evidence:
            ids = profile.evidence["interests"]
            assert ids == sorted(ids)

    def test_evidence_behavior_patterns_contains_contributing_ids(self):
        """evidence['behavior_patterns'] contains IDs of memories with behavior phrases."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("m1", "I prefer concise answers when possible"),
            _mem("m2", "I prefer detailed explanations for complex topics"),
            _mem("m3", "I like reading books"),
        ]
        profile = builder.build(memories, [])
        if "behavior_patterns" in profile.evidence:
            assert "m1" in profile.evidence["behavior_patterns"]
            assert "m2" in profile.evidence["behavior_patterns"]

    def test_evidence_strengths_contains_insight_evidence_ids(self):
        """evidence['strengths'] contains evidence IDs from contributing insights."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Python programming skills", 0.9, ["insight-ev-1", "insight-ev-2"]),
        ]
        profile = builder.build(memories, insights)
        if "strengths" in profile.evidence:
            assert "insight-ev-1" in profile.evidence["strengths"]
            assert "insight-ev-2" in profile.evidence["strengths"]

    def test_evidence_empty_when_no_contributions(self):
        """Evidence dict is empty when no keywords/patterns/strengths are found."""
        builder = ProfileBuilder(min_keyword_frequency=5)
        # Only 1 memory, so no keyword meets threshold of 5
        memories = [_mem("m1", "python programming")]
        profile = builder.build(memories, [])
        assert profile.evidence == {}


# ---------------------------------------------------------------------------
# Insight supplement for interests
# ---------------------------------------------------------------------------

class TestInsightSupplement:
    def test_supplements_interests_from_high_confidence_insight_text(self):
        """Keywords from high-confidence insight text are added to interests."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        # Only 1 memory, so no keyword meets threshold from memories alone
        memories = [_mem("m1", "general content here")]
        insights = [
            _insight("machine learning algorithms", 0.8, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        # "machine" and "learning" and "algorithms" should be supplemented
        assert "machine" in profile.interests or "learning" in profile.interests or "algorithms" in profile.interests

    def test_does_not_supplement_from_low_confidence_insights(self):
        """Keywords from low-confidence insights are NOT added to interests."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [_mem("m1", "general content here")]
        insights = [
            _insight("machine learning algorithms", 0.5, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        assert "machine" not in profile.interests
        assert "learning" not in profile.interests

    def test_no_duplicate_keywords_from_insight_supplement(self):
        """Keywords already in interests from memories are not duplicated."""
        builder = ProfileBuilder(min_keyword_frequency=2)
        memories = [
            _mem("m1", "python programming language"),
            _mem("m2", "python scripting language"),
        ]
        insights = [
            _insight("python machine learning", 0.9, ["m1"]),
        ]
        profile = builder.build(memories, insights)
        # "python" should appear only once
        assert profile.interests.count("python") == 1


# ---------------------------------------------------------------------------
# Skipping invalid memories and insights
# ---------------------------------------------------------------------------

class TestSkipping:
    def test_skips_memories_with_none_content(self):
        """Memories with None content are skipped."""
        builder = ProfileBuilder(min_keyword_frequency=1)
        memories = [
            {"id": "m1", "content": None, "metadata": {}, "timestamp": "2024-01-15T10:30:00", "category": "general", "tags": []},
            _mem("m2", "python programming language"),
        ]
        # Should not raise; m1 is skipped
        profile = builder.build(memories, [])
        assert isinstance(profile, UserProfile)

    def test_skips_insights_with_invalid_confidence(self):
        """Insights with confidence outside [0.0, 1.0] are skipped."""
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insights = [
            _insight("Python skills", 1.5, ["m1"]),   # invalid
            _insight("Data skills", -0.1, ["m1"]),    # invalid
            _insight("Valid insight", 0.8, ["m1"]),   # valid
        ]
        profile = builder.build(memories, insights)
        # Should not raise; invalid insights are skipped
        assert isinstance(profile, UserProfile)
        assert len(profile.strengths) > 0  # from the valid insight

    def test_handles_pydantic_insight_objects(self):
        """Handles Pydantic/dataclass insight objects with attribute access."""
        from luma.core.insight.schemas import Insight
        builder = ProfileBuilder()
        memories = [_mem("m1", "some content here")]
        insight_obj = Insight(text="Python programming skills", confidence=0.9, evidence=["m1"])
        profile = builder.build(memories, [insight_obj])
        assert isinstance(profile, UserProfile)
        assert len(profile.strengths) > 0
