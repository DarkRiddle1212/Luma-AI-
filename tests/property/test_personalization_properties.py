"""
Property-based tests for the Personalization Engine.

Properties 3-6 cover ProfileBuilder behaviour.
Each test is tagged: Feature: personalization-engine, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

import copy
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import Any, List

from luma.core.personalization.profile_builder import ProfileBuilder


# ---------------------------------------------------------------------------
# Strategies (from design doc)
# ---------------------------------------------------------------------------

def memory_entry_strategy():
    return st.fixed_dictionaries({
        "id": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
        "content": st.text(min_size=0, max_size=300),
        "metadata": st.dictionaries(st.text(min_size=1), st.text()),
        "timestamp": st.just("2024-01-15T10:30:00"),
        "category": st.text(min_size=1, max_size=30),
        "tags": st.lists(st.text(min_size=1, max_size=20), max_size=5),
    })


def insight_strategy():
    """Generate plain-dict insights with valid confidence values."""
    return st.fixed_dictionaries({
        "text": st.text(min_size=0, max_size=200),
        "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        "evidence": st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
            min_size=1,
            max_size=5,
        ),
    })


# ---------------------------------------------------------------------------
# Property 3: ProfileBuilder interaction_style is always valid
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 3: ProfileBuilder interaction_style is always valid
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_interaction_style_always_valid(memories):
    """
    **Validates: Requirements 2.4, 5.6, 13.2**

    For any list of memory entries, ProfileBuilder.build() must return a
    UserProfile whose interaction_style is one of {"concise", "detailed", "balanced"}.
    """
    builder = ProfileBuilder(min_keyword_frequency=1)
    profile = builder.build(memories, [])
    assert profile.interaction_style in {"concise", "detailed", "balanced"}, (
        f"interaction_style {profile.interaction_style!r} is not a valid value"
    )


# ---------------------------------------------------------------------------
# Property 4: ProfileBuilder string lists contain only non-empty strings
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 4: ProfileBuilder string lists contain only non-empty strings
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_string_lists_contain_only_non_empty_strings(memories):
    """
    **Validates: Requirements 2.2, 2.3, 13.3, 13.4**

    For any list of memory entries, every string in profile.interests and
    profile.behavior_patterns must be non-empty and not whitespace-only.
    """
    builder = ProfileBuilder(min_keyword_frequency=1)
    profile = builder.build(memories, [])

    for item in profile.interests:
        assert isinstance(item, str), (
            f"interests contains non-string item: {item!r}"
        )
        assert item.strip() != "", (
            f"interests contains empty/whitespace-only string: {item!r}"
        )

    for item in profile.behavior_patterns:
        assert isinstance(item, str), (
            f"behavior_patterns contains non-string item: {item!r}"
        )
        assert item.strip() != "", (
            f"behavior_patterns contains empty/whitespace-only string: {item!r}"
        )


# ---------------------------------------------------------------------------
# Property 5: ProfileBuilder evidence IDs are subset of input memory IDs
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 5: ProfileBuilder evidence IDs are subset of input memory IDs
@given(memories=st.lists(memory_entry_strategy(), min_size=1, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_evidence_ids_subset_of_input_memory_ids(memories):
    """
    **Validates: Requirements 2.10, 7.5, 13.6**

    For any non-empty list of memory entries, every ID in evidence values
    (for "interests" and "behavior_patterns") must be present in the input
    memory IDs.

    Note: evidence["strengths"] contains insight evidence IDs, not memory IDs,
    so it is excluded from this check.
    """
    builder = ProfileBuilder(min_keyword_frequency=1)
    profile = builder.build(memories, [])

    input_ids = {m["id"] for m in memories}

    for key in ("interests", "behavior_patterns"):
        if key in profile.evidence:
            for eid in profile.evidence[key]:
                assert eid in input_ids, (
                    f"evidence[{key!r}] contains ID {eid!r} not in input memory IDs"
                )


# ---------------------------------------------------------------------------
# Property 6: ProfileBuilder determinism
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 6: ProfileBuilder determinism
@given(
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50),
    insights=st.lists(insight_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_profile_builder_is_deterministic(memories, insights):
    """
    **Validates: Requirements 2.8, 9.3, 13.5**

    Calling ProfileBuilder.build() twice with identical inputs must produce
    identical outputs (same field values in same order).
    """
    builder = ProfileBuilder(min_keyword_frequency=1)

    # Deep copy inputs to ensure they are not mutated between calls
    memories_copy = copy.deepcopy(memories)
    insights_copy = copy.deepcopy(insights)

    profile1 = builder.build(memories, insights)
    profile2 = builder.build(memories_copy, insights_copy)

    assert profile1.interests == profile2.interests, (
        f"Non-deterministic interests: {profile1.interests!r} vs {profile2.interests!r}"
    )
    assert profile1.behavior_patterns == profile2.behavior_patterns, (
        f"Non-deterministic behavior_patterns: "
        f"{profile1.behavior_patterns!r} vs {profile2.behavior_patterns!r}"
    )
    assert profile1.interaction_style == profile2.interaction_style, (
        f"Non-deterministic interaction_style: "
        f"{profile1.interaction_style!r} vs {profile2.interaction_style!r}"
    )
    assert profile1.strengths == profile2.strengths, (
        f"Non-deterministic strengths: {profile1.strengths!r} vs {profile2.strengths!r}"
    )
    assert profile1.evidence == profile2.evidence, (
        f"Non-deterministic evidence: {profile1.evidence!r} vs {profile2.evidence!r}"
    )


# ---------------------------------------------------------------------------
# Additional strategies for PreferenceDetector properties
# ---------------------------------------------------------------------------

from luma.core.personalization.preference_detector import PreferenceDetector
from luma.core.personalization.schemas import UserProfile, Preference


def _non_empty_printable_text(min_size=1, max_size=30):
    """Generate text that is non-empty and not whitespace-only (satisfies UserProfile validation)."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Po", "Sm")),
        min_size=min_size,
        max_size=max_size,
    )


def user_profile_strategy():
    return st.builds(
        UserProfile,
        interests=st.lists(_non_empty_printable_text(1, 30), max_size=10),
        behavior_patterns=st.lists(_non_empty_printable_text(1, 50), max_size=5),
        interaction_style=st.sampled_from(["concise", "detailed", "balanced"]),
        strengths=st.lists(_non_empty_printable_text(1, 30), max_size=5),
        evidence=st.just({}),
    )


def preference_strategy():
    return st.builds(
        Preference,
        preference=_non_empty_printable_text(1, 30),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        reason=_non_empty_printable_text(1, 100),
    )


# ---------------------------------------------------------------------------
# Property 7: PreferenceDetector confidence is always in range and above threshold
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 7: PreferenceDetector confidence is always in range and above threshold
@given(
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50),
    profile=user_profile_strategy(),
    threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_preference_detector_confidence_in_range_and_above_threshold(memories, profile, threshold):
    """
    **Validates: Requirements 3.1, 3.2, 6.3, 8.1, 8.2, 9.4, 12.2, 12.7, 14.2**

    For any list of memories, UserProfile, and configured min_confidence threshold,
    every Preference returned has confidence in [0.0, 1.0] and confidence >= min_confidence.
    """
    detector = PreferenceDetector(min_confidence=threshold, min_frequency=1)
    preferences = detector.detect(memories, profile)
    for pref in preferences:
        assert 0.0 <= pref.confidence <= 1.0, (
            f"confidence {pref.confidence} is outside [0.0, 1.0]"
        )
        assert pref.confidence >= threshold, (
            f"confidence {pref.confidence} is below threshold {threshold}"
        )


# ---------------------------------------------------------------------------
# Property 8: PreferenceDetector reason is always non-empty
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 8: PreferenceDetector reason is always non-empty
@given(
    memories=st.lists(memory_entry_strategy(), min_size=1, max_size=50),
    profile=user_profile_strategy(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_preference_detector_reason_always_non_empty(memories, profile):
    """
    **Validates: Requirements 3.3, 3.5, 14.3**

    For any non-empty list of memories and UserProfile, every Preference returned
    has a non-empty reason string.
    """
    detector = PreferenceDetector(min_confidence=0.0, min_frequency=1)
    preferences = detector.detect(memories, profile)
    for pref in preferences:
        assert pref.reason, (
            f"Preference {pref.preference!r} has an empty reason"
        )
        assert pref.reason.strip(), (
            f"Preference {pref.preference!r} has a whitespace-only reason"
        )


# ---------------------------------------------------------------------------
# Property 9: PreferenceDetector determinism
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 9: PreferenceDetector determinism
@given(
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50),
    profile=user_profile_strategy(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_preference_detector_is_deterministic(memories, profile):
    """
    **Validates: Requirements 3.6, 9.4, 14.4**

    Calling detect() twice with identical inputs must produce identical output.
    """
    detector = PreferenceDetector(min_confidence=0.5, min_frequency=2)

    memories_copy = copy.deepcopy(memories)
    profile_copy = copy.deepcopy(profile)

    result1 = detector.detect(memories, profile)
    result2 = detector.detect(memories_copy, profile_copy)

    assert len(result1) == len(result2), (
        f"Non-deterministic result length: {len(result1)} vs {len(result2)}"
    )
    for p1, p2 in zip(result1, result2):
        assert p1.preference == p2.preference, (
            f"Non-deterministic preference: {p1.preference!r} vs {p2.preference!r}"
        )
        assert p1.confidence == p2.confidence, (
            f"Non-deterministic confidence: {p1.confidence} vs {p2.confidence}"
        )
        assert p1.reason == p2.reason, (
            f"Non-deterministic reason: {p1.reason!r} vs {p2.reason!r}"
        )


# ---------------------------------------------------------------------------
# Property 10: PreferenceDetector minimum frequency invariant
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 10: PreferenceDetector minimum frequency invariant
@given(
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50),
    profile=user_profile_strategy(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_preference_detector_minimum_frequency_invariant(memories, profile):
    """
    **Validates: Requirements 3.4, 3.7, 14.5**

    For any list of memories, no Preference is emitted for signals appearing in
    fewer than min_frequency memories. Verified by counting memory IDs in the
    reason string.
    """
    min_frequency = 2
    detector = PreferenceDetector(min_confidence=0.0, min_frequency=min_frequency)
    preferences = detector.detect(memories, profile)

    for pref in preferences:
        # The reason format is: "Detected in N memories: id1, id2, id3..."
        # Extract N from the reason string
        reason = pref.reason
        # Parse the count from "Detected in N memories:"
        import re
        match = re.match(r"Detected in (\d+) memories:", reason)
        assert match is not None, (
            f"Reason {reason!r} does not match expected format"
        )
        count = int(match.group(1))
        assert count >= min_frequency, (
            f"Preference {pref.preference!r} has only {count} supporting memories, "
            f"which is below min_frequency={min_frequency}"
        )


# ---------------------------------------------------------------------------
# AdaptationEngine strategies and imports
# ---------------------------------------------------------------------------

from luma.core.personalization.adaptation_engine import AdaptationEngine


# ---------------------------------------------------------------------------
# Property 11: AdaptationEngine output fields are always valid enum values
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 11: AdaptationEngine output fields are always valid enum values
@given(
    profile=user_profile_strategy(),
    preferences=st.lists(preference_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_adaptation_engine_output_fields_are_valid_enum_values(profile, preferences):
    """
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 6.4, 7.2, 9.5, 12.3, 12.8, 15.2, 15.3, 15.4, 15.5, 15.6**

    For any UserProfile and list of Preferences, the returned AdaptationContext
    must have tone, style, and focus values drawn from their respective valid sets.
    """
    engine = AdaptationEngine()
    ctx = engine.adapt(profile, preferences)

    assert ctx.tone in {"technical", "casual", "formal"}, (
        f"tone {ctx.tone!r} is not a valid value"
    )
    assert ctx.style in {"concise", "detailed", "step-by-step", "balanced"}, (
        f"style {ctx.style!r} is not a valid value"
    )
    assert ctx.focus in {"high-level", "deep-technical"}, (
        f"focus {ctx.focus!r} is not a valid value"
    )


# ---------------------------------------------------------------------------
# Property 12: AdaptationEngine reasons are always non-empty
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 12: AdaptationEngine reasons are always non-empty
@given(
    profile=user_profile_strategy(),
    preferences=st.lists(preference_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_adaptation_engine_reasons_always_non_empty(profile, preferences):
    """
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 6.4, 7.2, 9.5, 12.3, 12.8, 15.2, 15.3, 15.4, 15.5, 15.6**

    For any UserProfile and list of Preferences, reasons["tone"], reasons["style"],
    and reasons["focus"] must all be non-empty strings.
    """
    engine = AdaptationEngine()
    ctx = engine.adapt(profile, preferences)

    for key in ("tone", "style", "focus"):
        assert key in ctx.reasons, f"reasons is missing key {key!r}"
        assert isinstance(ctx.reasons[key], str), (
            f"reasons[{key!r}] is not a string: {ctx.reasons[key]!r}"
        )
        assert ctx.reasons[key].strip(), (
            f"reasons[{key!r}] is empty or whitespace-only"
        )


# ---------------------------------------------------------------------------
# Property 13: AdaptationEngine determinism
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 13: AdaptationEngine determinism
@given(
    profile=user_profile_strategy(),
    preferences=st.lists(preference_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_adaptation_engine_is_deterministic(profile, preferences):
    """
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 6.4, 7.2, 9.5, 12.3, 12.8, 15.2, 15.3, 15.4, 15.5, 15.6**

    Calling adapt() twice with identical inputs must produce identical output.
    """
    engine = AdaptationEngine()

    profile_copy = copy.deepcopy(profile)
    preferences_copy = copy.deepcopy(preferences)

    ctx1 = engine.adapt(profile, preferences)
    ctx2 = engine.adapt(profile_copy, preferences_copy)

    assert ctx1.tone == ctx2.tone, (
        f"Non-deterministic tone: {ctx1.tone!r} vs {ctx2.tone!r}"
    )
    assert ctx1.style == ctx2.style, (
        f"Non-deterministic style: {ctx1.style!r} vs {ctx2.style!r}"
    )
    assert ctx1.focus == ctx2.focus, (
        f"Non-deterministic focus: {ctx1.focus!r} vs {ctx2.focus!r}"
    )
    assert ctx1.reasons == ctx2.reasons, (
        f"Non-deterministic reasons: {ctx1.reasons!r} vs {ctx2.reasons!r}"
    )


# ---------------------------------------------------------------------------
# PersonalizationEngine imports
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from luma.core.personalization.personalization_engine import PersonalizationEngine
from luma.core.personalization.schemas import PersonalizationResult


def _make_mock_memory_interface(memories):
    """Build a mock MemoryInterface that returns the given memories and raises on store()."""
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


def _make_real_engine(memories):
    """Build a PersonalizationEngine with real components and a mock MemoryInterface."""
    from luma.core.personalization.profile_builder import ProfileBuilder
    from luma.core.personalization.preference_detector import PreferenceDetector
    from luma.core.personalization.adaptation_engine import AdaptationEngine

    mi = _make_mock_memory_interface(memories)
    engine = PersonalizationEngine(
        memory_interface=mi,
        profile_builder=ProfileBuilder(min_keyword_frequency=1),
        preference_detector=PreferenceDetector(min_confidence=0.0, min_frequency=1),
        adaptation_engine=AdaptationEngine(),
    )
    return engine


# ---------------------------------------------------------------------------
# Property 1: PersonalizationResult always contains all three components
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 1: PersonalizationResult always contains all three components
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_personalization_result_always_contains_all_three_components(memories):
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 6.1, 6.5, 6.6, 9.1, 9.2, 10.1, 12.4, 12.5, 12.9**

    For any list of memories (via mock MemoryInterface), the returned
    PersonalizationResult must have:
    - profile: a UserProfile instance
    - preferences: a list
    - adaptation: an AdaptationContext instance
    """
    from luma.core.personalization.schemas import UserProfile, AdaptationContext

    engine = _make_real_engine(memories)
    result = engine.personalize("input", "context")

    assert isinstance(result, PersonalizationResult), (
        f"Expected PersonalizationResult, got {type(result)}"
    )
    assert isinstance(result.profile, UserProfile), (
        f"Expected UserProfile for result.profile, got {type(result.profile)}"
    )
    assert isinstance(result.preferences, list), (
        f"Expected list for result.preferences, got {type(result.preferences)}"
    )
    assert isinstance(result.adaptation, AdaptationContext), (
        f"Expected AdaptationContext for result.adaptation, got {type(result.adaptation)}"
    )


# ---------------------------------------------------------------------------
# Property 2: Input immutability through the full pipeline
# ---------------------------------------------------------------------------

# Feature: personalization-engine, Property 2: Input immutability through the full pipeline
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_input_immutability_through_full_pipeline(memories):
    """
    **Validates: Requirements 1.8, 4.8, 6.5, 6.6**

    For any list of memories, the id, content, category, and tags fields of
    every input memory are identical before and after calling personalize().
    The pipeline must not mutate any MemoryEntry object.
    """
    import copy

    # Capture deep copies of the fields we care about before calling personalize()
    before = [
        {
            "id": m["id"],
            "content": m["content"],
            "category": m["category"],
            "tags": copy.deepcopy(m["tags"]),
        }
        for m in memories
    ]

    engine = _make_real_engine(memories)
    engine.personalize("input", "context")

    # Verify each memory's tracked fields are unchanged
    for i, (mem, snap) in enumerate(zip(memories, before)):
        assert mem["id"] == snap["id"], (
            f"Memory[{i}].id was mutated: {snap['id']!r} → {mem['id']!r}"
        )
        assert mem["content"] == snap["content"], (
            f"Memory[{i}].content was mutated: {snap['content']!r} → {mem['content']!r}"
        )
        assert mem["category"] == snap["category"], (
            f"Memory[{i}].category was mutated: {snap['category']!r} → {mem['category']!r}"
        )
        assert mem["tags"] == snap["tags"], (
            f"Memory[{i}].tags was mutated: {snap['tags']!r} → {mem['tags']!r}"
        )
