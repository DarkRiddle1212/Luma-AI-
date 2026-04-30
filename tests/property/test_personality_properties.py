"""
Property-based tests for the Personality Layer.

Property 1 covers schema validation for enum fields.
Each test is tagged: Feature: luma-personality-layer, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

from hypothesis import given, settings, strategies as st, HealthCheck
import pytest

from luma.core.personality.schemas import (
    ToneSelection,
    StylePreference,
    GuardrailResult,
    VALID_TONES,
    VALID_STYLES,
)
from luma.core.personality.system_prompt import SystemPrompt
from luma.core.personalization.schemas import AdaptationContext


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def invalid_tone_strategy():
    """Generate strings that are NOT in VALID_TONES."""
    # Generate random strings that don't match any valid tone
    return st.text(min_size=1, max_size=30).filter(lambda x: x not in VALID_TONES)


def invalid_style_strategy():
    """Generate strings that are NOT in VALID_STYLES."""
    # Generate random strings that don't match any valid style
    return st.text(min_size=1, max_size=30).filter(lambda x: x not in VALID_STYLES)


def out_of_range_score_strategy():
    """Generate floats outside the range [0.0, 1.0]."""
    # Generate floats that are either < 0.0 or > 1.0
    return st.one_of(
        st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),  # Below 0.0
        st.floats(min_value=1.001, allow_nan=False, allow_infinity=False),   # Above 1.0
    )


def personality_id_strategy():
    """Generate random personality_id strings for testing SystemPrompt."""
    # Generate a mix of known personality IDs and random strings
    return st.one_of(
        st.sampled_from(["default", "ceo", "developer", "tutor"]),  # Known IDs
        st.text(min_size=0, max_size=50),  # Random strings including empty
    )


# ---------------------------------------------------------------------------
# Property 1: Schema validation rejects invalid enum values
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 1: Schema validation rejects invalid enum values
@given(invalid_tone=invalid_tone_strategy())
@settings(max_examples=100)
def test_tone_selection_rejects_invalid_tone(invalid_tone):
    """
    **Validates: Requirements 1.6, 1.7, 4.4, 11.2**

    For any string not in VALID_TONES, constructing a ToneSelection with that
    tone SHALL raise a ValueError with a descriptive message.
    """
    with pytest.raises(ValueError) as exc_info:
        ToneSelection(
            tone=invalid_tone,
            rationale="test rationale",
            context_signals={},
        )
    
    # Verify the error message is descriptive
    error_message = str(exc_info.value)
    assert "tone must be one of" in error_message, (
        f"Error message should mention valid tones, got: {error_message}"
    )
    assert invalid_tone in error_message or repr(invalid_tone) in error_message, (
        f"Error message should mention the invalid value {invalid_tone!r}, got: {error_message}"
    )


# Feature: luma-personality-layer, Property 1: Schema validation rejects invalid enum values
@given(invalid_style=invalid_style_strategy())
@settings(max_examples=100)
def test_style_preference_rejects_invalid_style(invalid_style):
    """
    **Validates: Requirements 1.6, 1.7, 4.4, 11.2**

    For any string not in VALID_STYLES, constructing a StylePreference with that
    style SHALL raise a ValueError with a descriptive message.
    """
    with pytest.raises(ValueError) as exc_info:
        StylePreference(
            style=invalid_style,
            description="test description",
            active=True,
        )
    
    # Verify the error message is descriptive
    error_message = str(exc_info.value)
    assert "style must be one of" in error_message, (
        f"Error message should mention valid styles, got: {error_message}"
    )
    assert invalid_style in error_message or repr(invalid_style) in error_message, (
        f"Error message should mention the invalid value {invalid_style!r}, got: {error_message}"
    )


# ---------------------------------------------------------------------------
# Property 2: Schema validation rejects out-of-range score values
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 2: Schema validation rejects out-of-range score values
@given(out_of_range_score=out_of_range_score_strategy())
@settings(max_examples=100)
def test_guardrail_result_rejects_out_of_range_score(out_of_range_score):
    """
    **Validates: Requirements 1.8**

    For any float value outside the range [0.0, 1.0], constructing a GuardrailResult
    with that score SHALL raise a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        GuardrailResult(
            passed=True,
            violations=[],
            score=out_of_range_score,
            notes="test notes",
        )
    
    # Verify the error message mentions the score range
    error_message = str(exc_info.value)
    assert "score must be in [0.0, 1.0]" in error_message or "0.0" in error_message and "1.0" in error_message, (
        f"Error message should mention valid score range [0.0, 1.0], got: {error_message}"
    )
    assert str(out_of_range_score) in error_message or f"{out_of_range_score}" in error_message, (
        f"Error message should mention the invalid value {out_of_range_score}, got: {error_message}"
    )


# ---------------------------------------------------------------------------
# Property 3: SystemPrompt identity is always non-empty
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 3: SystemPrompt identity is always non-empty
@given(personality_id=personality_id_strategy())
@settings(max_examples=100)
def test_system_prompt_identity_always_non_empty(personality_id):
    """
    **Validates: Requirements 2.1**

    For any personality_id string, calling SystemPrompt.get_identity(personality_id)
    SHALL return a non-empty string.
    """
    system_prompt = SystemPrompt()
    identity = system_prompt.get_identity(personality_id)
    
    # Verify the identity is a string
    assert isinstance(identity, str), (
        f"get_identity() should return a string, got {type(identity).__name__}"
    )
    
    # Verify the identity is non-empty
    assert len(identity) > 0, (
        f"get_identity() should return a non-empty string for personality_id={personality_id!r}, "
        f"got empty string"
    )
    
    # Verify the identity is not just whitespace
    assert identity.strip(), (
        f"get_identity() should return a non-whitespace string for personality_id={personality_id!r}, "
        f"got only whitespace: {identity!r}"
    )


# ---------------------------------------------------------------------------
# Property 4: SystemPrompt determinism
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 4: SystemPrompt determinism
@given(personality_id=personality_id_strategy())
@settings(max_examples=100)
def test_system_prompt_determinism(personality_id):
    """
    **Validates: Requirements 2.5, 10.1**

    For any personality_id string, calling SystemPrompt.get_identity(personality_id)
    twice SHALL produce identical output strings.
    """
    system_prompt = SystemPrompt()
    
    # Call get_identity() twice with the same input
    identity_first_call = system_prompt.get_identity(personality_id)
    identity_second_call = system_prompt.get_identity(personality_id)
    
    # Verify both calls return strings
    assert isinstance(identity_first_call, str), (
        f"First call to get_identity() should return a string, got {type(identity_first_call).__name__}"
    )
    assert isinstance(identity_second_call, str), (
        f"Second call to get_identity() should return a string, got {type(identity_second_call).__name__}"
    )
    
    # Verify the outputs are identical
    assert identity_first_call == identity_second_call, (
        f"get_identity() should be deterministic for personality_id={personality_id!r}. "
        f"First call returned: {identity_first_call!r}, "
        f"Second call returned: {identity_second_call!r}"
    )


# ---------------------------------------------------------------------------
# Property 5: ToneManager respects valid user preferences
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 5: ToneManager respects valid user preferences
@given(valid_tone=st.sampled_from(sorted(VALID_TONES)))
@settings(max_examples=100)
def test_tone_manager_respects_user_preference(valid_tone):
    """
    **Validates: Requirements 3.2**

    For any valid tone string, when passed as user_preference to ToneManager.select_tone(),
    the returned ToneSelection SHALL have that tone and a rationale containing "user preference".
    """
    from luma.core.personality.tone_manager import ToneManager
    from luma.core.personalization.schemas import AdaptationContext
    
    # Create a ToneManager instance
    tone_manager = ToneManager()
    
    # Create a minimal AdaptationContext (context should be ignored when user_preference is set)
    context = AdaptationContext(
        tone="casual",
        style="balanced",
        focus="high-level",
        reasons={},
    )
    
    # Call select_tone with the valid user preference
    result = tone_manager.select_tone(
        context=context,
        mode="chat",
        user_preference=valid_tone,
    )
    
    # Verify the result is a ToneSelection
    assert isinstance(result, ToneSelection), (
        f"select_tone() should return a ToneSelection, got {type(result).__name__}"
    )
    
    # Verify the selected tone matches the user preference
    assert result.tone == valid_tone, (
        f"select_tone() should respect user_preference={valid_tone!r}, "
        f"but returned tone={result.tone!r}"
    )
    
    # Verify the rationale contains "user preference"
    assert "user preference" in result.rationale.lower(), (
        f"select_tone() rationale should contain 'user preference' when user_preference is set, "
        f"but got rationale={result.rationale!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: ToneManager determinism
# ---------------------------------------------------------------------------

def adaptation_context_strategy():
    """Generate random AdaptationContext objects for testing ToneManager."""
    return st.builds(
        lambda tone, style, focus, reasons: AdaptationContext(
            tone=tone,
            style=style,
            focus=focus,
            reasons=reasons,
        ),
        tone=st.sampled_from(["technical", "casual", "formal"]),
        style=st.sampled_from(["concise", "detailed", "step-by-step", "balanced"]),
        focus=st.sampled_from(["high-level", "deep-technical"]),
        reasons=st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=50), max_size=5),
    )


def mode_strategy():
    """Generate random mode strings for testing ToneManager."""
    # Include common modes and random strings
    return st.one_of(
        st.sampled_from(["chat", "teacher", "assistant", "debug"]),
        st.text(min_size=0, max_size=30),
    )


def user_preference_strategy():
    """Generate random user_preference values (valid tones, invalid tones, or None)."""
    return st.one_of(
        st.none(),
        st.sampled_from(sorted(VALID_TONES)),
        st.text(min_size=1, max_size=30).filter(lambda x: x not in VALID_TONES),
    )


# Feature: luma-personality-layer, Property 6: ToneManager determinism
@given(
    context=adaptation_context_strategy(),
    mode=mode_strategy(),
    user_preference=user_preference_strategy(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_tone_manager_determinism(context, mode, user_preference):
    """
    **Validates: Requirements 3.7, 10.2**

    For any combination of AdaptationContext, mode, and user_preference,
    calling ToneManager.select_tone() twice with identical inputs SHALL
    produce identical ToneSelection objects (same tone and rationale).
    """
    from luma.core.personality.tone_manager import ToneManager
    from luma.core.personalization.schemas import AdaptationContext
    
    # Create a ToneManager instance
    tone_manager = ToneManager()
    
    # Call select_tone twice with identical inputs
    result_first_call = tone_manager.select_tone(
        context=context,
        mode=mode,
        user_preference=user_preference,
    )
    result_second_call = tone_manager.select_tone(
        context=context,
        mode=mode,
        user_preference=user_preference,
    )
    
    # Verify both calls return ToneSelection objects
    assert isinstance(result_first_call, ToneSelection), (
        f"First call to select_tone() should return a ToneSelection, got {type(result_first_call).__name__}"
    )
    assert isinstance(result_second_call, ToneSelection), (
        f"Second call to select_tone() should return a ToneSelection, got {type(result_second_call).__name__}"
    )
    
    # Verify the selected tone is identical
    assert result_first_call.tone == result_second_call.tone, (
        f"select_tone() should be deterministic for context={context}, mode={mode!r}, user_preference={user_preference!r}. "
        f"First call returned tone={result_first_call.tone!r}, "
        f"Second call returned tone={result_second_call.tone!r}"
    )
    
    # Verify the rationale is identical
    assert result_first_call.rationale == result_second_call.rationale, (
        f"select_tone() should be deterministic for context={context}, mode={mode!r}, user_preference={user_preference!r}. "
        f"First call returned rationale={result_first_call.rationale!r}, "
        f"Second call returned rationale={result_second_call.rationale!r}"
    )
    
    # Verify the context_signals are identical
    assert result_first_call.context_signals == result_second_call.context_signals, (
        f"select_tone() should be deterministic for context={context}, mode={mode!r}, user_preference={user_preference!r}. "
        f"First call returned context_signals={result_first_call.context_signals!r}, "
        f"Second call returned context_signals={result_second_call.context_signals!r}"
    )


# ---------------------------------------------------------------------------
# Property 7: ToneManager gracefully handles invalid user preferences
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 7: ToneManager gracefully handles invalid user preferences
@given(invalid_tone=invalid_tone_strategy())
@settings(max_examples=100)
def test_tone_manager_handles_invalid_user_preference(invalid_tone):
    """
    **Validates: Requirements 11.1**

    For any invalid tone string (not in the valid set), when passed as user_preference
    to ToneManager.select_tone(), the method SHALL NOT raise an exception and SHALL
    fall back to context-based tone selection.
    """
    from luma.core.personality.tone_manager import ToneManager
    from luma.core.personalization.schemas import AdaptationContext
    
    # Create a ToneManager instance
    tone_manager = ToneManager()
    
    # Create an AdaptationContext with a known tone for fallback verification
    context = AdaptationContext(
        tone="technical",
        style="balanced",
        focus="high-level",
        reasons={},
    )
    
    # Call select_tone with the invalid user preference - should NOT raise an exception
    result = tone_manager.select_tone(
        context=context,
        mode="chat",
        user_preference=invalid_tone,
    )
    
    # Verify the result is a ToneSelection
    assert isinstance(result, ToneSelection), (
        f"select_tone() should return a ToneSelection even with invalid user_preference, "
        f"got {type(result).__name__}"
    )
    
    # Verify the selected tone is valid (from the valid set)
    assert result.tone in VALID_TONES, (
        f"select_tone() should fall back to a valid tone when user_preference={invalid_tone!r} is invalid, "
        f"but returned tone={result.tone!r} which is not in VALID_TONES"
    )
    
    # Verify the rationale does NOT contain "user preference" (since invalid preference was ignored)
    assert "user preference" not in result.rationale.lower(), (
        f"select_tone() rationale should NOT contain 'user preference' when user_preference={invalid_tone!r} is invalid, "
        f"but got rationale={result.rationale!r}"
    )
    
    # Verify fallback to context-based selection (should select "technical" based on context.tone)
    assert result.tone == "technical", (
        f"select_tone() should fall back to context-based selection when user_preference={invalid_tone!r} is invalid. "
        f"Expected tone='technical' based on context.tone='technical', but got tone={result.tone!r}"
    )


# ---------------------------------------------------------------------------
# Property 8: StyleProfiles returns default for unknown users
# ---------------------------------------------------------------------------

def user_id_strategy():
    """Generate random user_id strings for testing StyleProfiles."""
    # Generate a mix of realistic user IDs and random strings
    return st.one_of(
        st.text(min_size=1, max_size=50),  # Random strings
        st.from_regex(r"user_[0-9a-f]{8}", fullmatch=True),  # UUID-like patterns
        st.from_regex(r"[a-z]{3,10}@[a-z]{3,10}\.[a-z]{2,3}", fullmatch=True),  # Email-like patterns
    )


# Feature: luma-personality-layer, Property 8: StyleProfiles returns default for unknown users
@given(user_id=user_id_strategy())
@settings(max_examples=100)
def test_style_profiles_returns_default_for_unknown_users(user_id):
    """
    **Validates: Requirements 4.3**

    For any user_id string with no stored preference, calling StyleProfiles.get_style(user_id)
    SHALL return a StylePreference with style="high_signal_low_noise" and active=True.
    """
    from luma.core.personality.style_profiles import StyleProfiles
    
    # Create a mock storage backend that returns no stored preferences
    class MockStorageBackend:
        def retrieve(self, params):
            # Return empty memories list (no stored preferences)
            return {"memories": []}
    
    # Create a StyleProfiles instance with the mock storage backend
    style_profiles = StyleProfiles(storage_backend=MockStorageBackend())
    
    # Call get_style with the random user_id
    result = style_profiles.get_style(user_id)
    
    # Verify the result is a StylePreference
    assert isinstance(result, StylePreference), (
        f"get_style() should return a StylePreference, got {type(result).__name__}"
    )
    
    # Verify the style is the default "high_signal_low_noise"
    assert result.style == "high_signal_low_noise", (
        f"get_style() should return default style='high_signal_low_noise' for unknown user_id={user_id!r}, "
        f"but returned style={result.style!r}"
    )
    
    # Verify the preference is active
    assert result.active is True, (
        f"get_style() should return active=True for unknown user_id={user_id!r}, "
        f"but returned active={result.active}"
    )
    
    # Verify the description is non-empty
    assert isinstance(result.description, str) and len(result.description) > 0, (
        f"get_style() should return a non-empty description for unknown user_id={user_id!r}, "
        f"but returned description={result.description!r}"
    )


# ---------------------------------------------------------------------------
# Property 9: ResponseGuardrails clean responses pass validation
# ---------------------------------------------------------------------------

def clean_response_strategy():
    """
    Generate response texts with no quality violations.
    
    Clean responses:
    - Short to moderate length (10-400 words)
    - No repetitive sequences
    - No contradiction markers
    - Minimal filler phrases (0-2)
    - Clear structure with concrete examples
    """
    # Pool of diverse concrete sentences
    sentence_pool = [
        "The function returns a validated result.",
        "This approach improves performance by 25%.",
        "The API endpoint accepts JSON payloads.",
        "Users can configure settings in the dashboard.",
        "The system processes 1000 requests per second.",
        "Error handling follows the standard pattern.",
        "The database schema includes three main tables.",
        "Authentication uses JWT tokens with 24-hour expiry.",
        "The cache layer reduces latency to 50ms.",
        "Logging captures all critical events.",
        "The module exports five public functions.",
        "Configuration files use YAML format.",
        "Tests achieve 95% code coverage.",
        "The service runs on port 8080.",
        "Data validation occurs at the boundary layer.",
        "Metrics are collected every 30 seconds.",
        "The queue handles 500 messages per minute.",
        "Backups run daily at midnight UTC.",
        "The algorithm has O(n log n) complexity.",
        "Dependencies are managed via package.json.",
    ]
    
    # Generate unique sentences (no repetition)
    return st.lists(
        st.sampled_from(sentence_pool),
        min_size=2,
        max_size=15,
        unique=True,  # Ensure no repeated sentences
    ).map(lambda sentences: " ".join(sentences))


# Feature: luma-personality-layer, Property 9: ResponseGuardrails clean responses pass validation
@given(response_text=clean_response_strategy())
@settings(max_examples=100)
def test_response_guardrails_clean_responses_pass_validation(response_text):
    """
    **Validates: Requirements 5.3**

    For any response text that contains no quality violations (no rambling, repetition,
    contradiction, or vague filler), calling ResponseGuardrails.validate() SHALL return
    a GuardrailResult with passed=True, an empty violations list, and score=1.0.
    """
    from luma.core.personality.response_guardrails import ResponseGuardrails
    
    # Create a ResponseGuardrails instance
    guardrails = ResponseGuardrails()
    
    # Call validate with no constraints (only check for quality violations)
    result = guardrails.validate(
        response_text=response_text,
        constraints=[],
    )
    
    # Verify the result is a GuardrailResult
    assert isinstance(result, GuardrailResult), (
        f"validate() should return a GuardrailResult, got {type(result).__name__}"
    )
    
    # Verify the validation passed
    assert result.passed is True, (
        f"validate() should return passed=True for clean response, "
        f"but returned passed={result.passed}. "
        f"Violations: {result.violations}, Score: {result.score}, "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the violations list is empty
    assert result.violations == [], (
        f"validate() should return empty violations list for clean response, "
        f"but returned violations={result.violations}. "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the score is 1.0
    assert result.score == 1.0, (
        f"validate() should return score=1.0 for clean response, "
        f"but returned score={result.score}. "
        f"Violations: {result.violations}, "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the notes indicate success
    assert "passed" in result.notes.lower() or "no" in result.notes.lower(), (
        f"validate() notes should indicate success for clean response, "
        f"but returned notes={result.notes!r}"
    )


# ---------------------------------------------------------------------------
# Property 10: ResponseGuardrails violation detection invariants
# ---------------------------------------------------------------------------

def response_with_violations_strategy():
    """
    Generate response texts with one or more quality violations.
    
    Violation types:
    - Rambling: >500 words without structure
    - Repetition: >2 occurrences of identical 5+ word sequences
    - Contradiction: presence of contradictory phrases
    - Vague filler: >3 filler phrases without concrete examples
    """
    # Strategy 1: Rambling (>500 words without structure)
    # Generate a long text by repeating varied sentences
    rambling_strategy = st.builds(
        lambda base_sentences: " ".join(base_sentences * 30),  # Repeat to get >500 words
        base_sentences=st.lists(
            st.sampled_from([
                "This is a sentence about something.",
                "Another sentence with different content.",
                "Yet another sentence here.",
                "More text to add length.",
                "Additional content for the response.",
                "Further elaboration on the topic.",
                "Continuing with more information.",
                "Expanding on the previous points.",
                "Adding more details to the discussion.",
                "Providing additional context.",
                "Explaining further aspects.",
                "Describing more features.",
                "Outlining additional considerations.",
                "Discussing related topics.",
                "Covering more ground.",
                "Addressing other points.",
                "Including more examples.",
                "Presenting further evidence.",
            ]),
            min_size=20,
            max_size=30,
        ),
    )
    
    # Strategy 2: Repetition (repeated 5+ word sequences)
    # Ensure the repeated sequence has exactly 5+ words
    repetition_strategy = st.builds(
        lambda repeated_seq: f"Introduction text here. {repeated_seq} Some middle content. {repeated_seq} More text in between. {repeated_seq} Conclusion text.",
        repeated_seq=st.text(min_size=30, max_size=60, alphabet=st.characters(whitelist_categories=('L', 'Z'))).filter(
            lambda x: len(x.split()) >= 5
        ),
    )
    
    # Strategy 3: Contradiction (contradictory phrases)
    contradiction_strategy = st.builds(
        lambda intro, marker, contradiction: f"{intro} {marker} {contradiction}",
        intro=st.sampled_from([
            "The system is highly reliable and never fails.",
            "This approach is always the best solution.",
            "The feature works perfectly in all cases.",
            "Users love this functionality.",
        ]),
        marker=st.sampled_from([
            "but actually",
            "on the other hand",
            "however, the",
            "in contrast",
            "conversely",
        ]),
        contradiction=st.sampled_from([
            "it frequently encounters errors.",
            "there are better alternatives available.",
            "it has known limitations.",
            "many users find it confusing.",
        ]),
    )
    
    # Strategy 4: Vague filler (>3 filler phrases without concrete examples)
    vague_filler_strategy = st.builds(
        lambda fillers: " ".join(fillers),
        fillers=st.lists(
            st.sampled_from([
                "It depends on the situation and context.",
                "Generally speaking this is how it works.",
                "In most cases you will see this behavior.",
                "Typically this happens in practice.",
                "Usually the result is what you expect.",
                "Often we find that things work out.",
                "Sometimes this occurs in the system.",
                "May or may not work depending on factors.",
                "Could be the case in certain scenarios.",
                "Might be possible under some conditions.",
            ]),
            min_size=4,
            max_size=8,
        ),
    )
    
    # Combine all strategies
    return st.one_of(
        rambling_strategy,
        repetition_strategy,
        contradiction_strategy,
        vague_filler_strategy,
    )


# Feature: luma-personality-layer, Property 10: ResponseGuardrails violation detection invariants
@given(response_text=response_with_violations_strategy())
@settings(max_examples=100)
def test_response_guardrails_violation_detection_invariants(response_text):
    """
    **Validates: Requirements 5.4**

    For any response text that contains one or more quality violations, calling
    ResponseGuardrails.validate() SHALL return a GuardrailResult with a non-empty
    violations list and a score in the range [0.0, 1.0]. The passed field SHALL
    be False if score < 0.75 (2+ violations), or True if score >= 0.75 (0-1 violations).
    """
    from luma.core.personality.response_guardrails import ResponseGuardrails
    
    # Create a ResponseGuardrails instance
    guardrails = ResponseGuardrails()
    
    # Call validate with no constraints (only check for quality violations)
    result = guardrails.validate(
        response_text=response_text,
        constraints=[],
    )
    
    # Verify the result is a GuardrailResult
    assert isinstance(result, GuardrailResult), (
        f"validate() should return a GuardrailResult, got {type(result).__name__}"
    )
    
    # Verify the violations list is non-empty (since we generated text with violations)
    assert len(result.violations) > 0, (
        f"validate() should return non-empty violations list for response with violations, "
        f"but returned violations={result.violations}. "
        f"Score: {result.score}, "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the score is in the range [0.0, 1.0]
    assert 0.0 <= result.score <= 1.0, (
        f"validate() should return score in [0.0, 1.0], "
        f"but returned score={result.score}. "
        f"Violations: {result.violations}, "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the passed field matches the score threshold (>= 0.75 passes)
    expected_passed = result.score >= 0.75
    assert result.passed == expected_passed, (
        f"validate() passed field should be {expected_passed} for score={result.score}, "
        f"but returned passed={result.passed}. "
        f"Violations: {result.violations}, "
        f"Response: {response_text[:100]}..."
    )
    
    # Verify the notes mention violations
    assert "violation" in result.notes.lower(), (
        f"validate() notes should mention violations for response with violations, "
        f"but returned notes={result.notes!r}"
    )


# ---------------------------------------------------------------------------
# Property 11: ResponseGuardrails determinism
# ---------------------------------------------------------------------------

def response_text_strategy():
    """
    Generate random response texts for testing ResponseGuardrails determinism.
    
    Includes a mix of:
    - Clean responses
    - Responses with violations
    - Empty responses
    - Short and long responses
    """
    return st.one_of(
        clean_response_strategy(),
        response_with_violations_strategy(),
        st.just(""),  # Empty response
        st.text(min_size=0, max_size=1000),  # Random text
    )


def constraints_list_strategy():
    """
    Generate random constraints lists for testing ResponseGuardrails.
    
    Includes:
    - Empty list
    - List with "concise"
    - List with multiple constraints
    """
    return st.one_of(
        st.just([]),
        st.just(["concise"]),
        st.lists(
            st.sampled_from(["concise", "detailed", "technical", "friendly"]),
            min_size=0,
            max_size=5,
        ),
    )


# Feature: luma-personality-layer, Property 11: ResponseGuardrails determinism
@given(
    response_text=response_text_strategy(),
    constraints=constraints_list_strategy(),
)
@settings(max_examples=100)
def test_response_guardrails_determinism(response_text, constraints):
    """
    **Validates: Requirements 5.6, 10.3**

    For any response text and constraints list, calling ResponseGuardrails.validate()
    twice with identical inputs SHALL produce identical GuardrailResult objects
    (same passed, violations, score, and notes).
    """
    from luma.core.personality.response_guardrails import ResponseGuardrails
    
    # Create a ResponseGuardrails instance
    guardrails = ResponseGuardrails()
    
    # Call validate twice with identical inputs
    result_first_call = guardrails.validate(
        response_text=response_text,
        constraints=constraints,
    )
    result_second_call = guardrails.validate(
        response_text=response_text,
        constraints=constraints,
    )
    
    # Verify both calls return GuardrailResult objects
    assert isinstance(result_first_call, GuardrailResult), (
        f"First call to validate() should return a GuardrailResult, got {type(result_first_call).__name__}"
    )
    assert isinstance(result_second_call, GuardrailResult), (
        f"Second call to validate() should return a GuardrailResult, got {type(result_second_call).__name__}"
    )
    
    # Verify the passed field is identical
    assert result_first_call.passed == result_second_call.passed, (
        f"validate() should be deterministic for response_text={response_text[:50]!r}..., constraints={constraints}. "
        f"First call returned passed={result_first_call.passed}, "
        f"Second call returned passed={result_second_call.passed}"
    )
    
    # Verify the violations list is identical
    assert result_first_call.violations == result_second_call.violations, (
        f"validate() should be deterministic for response_text={response_text[:50]!r}..., constraints={constraints}. "
        f"First call returned violations={result_first_call.violations}, "
        f"Second call returned violations={result_second_call.violations}"
    )
    
    # Verify the score is identical
    assert result_first_call.score == result_second_call.score, (
        f"validate() should be deterministic for response_text={response_text[:50]!r}..., constraints={constraints}. "
        f"First call returned score={result_first_call.score}, "
        f"Second call returned score={result_second_call.score}"
    )
    
    # Verify the notes are identical
    assert result_first_call.notes == result_second_call.notes, (
        f"validate() should be deterministic for response_text={response_text[:50]!r}..., constraints={constraints}. "
        f"First call returned notes={result_first_call.notes!r}, "
        f"Second call returned notes={result_second_call.notes!r}"
    )


# ---------------------------------------------------------------------------
# Property 12: PersonalityEngine output completeness
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 12: PersonalityEngine output completeness
@given(
    user_id=user_id_strategy(),
    context=adaptation_context_strategy(),
    mode=mode_strategy(),
)
@settings(max_examples=100)
def test_personality_engine_output_completeness(user_id, context, mode):
    """
    **Validates: Requirements 6.3**

    For any valid user_id, AdaptationContext, and mode, calling
    PersonalityEngine.build_instructions() SHALL return a PromptInstructions
    object with non-empty system_identity, tone_guidance, and style_constraints fields.
    """
    from luma.core.personality.personality_engine import PersonalityEngine
    from luma.core.personality.system_prompt import SystemPrompt
    from luma.core.personality.tone_manager import ToneManager
    from luma.core.personality.style_profiles import StyleProfiles
    from luma.core.personality.response_guardrails import ResponseGuardrails
    from luma.core.personality.schemas import PromptInstructions
    
    # Create a mock storage backend for StyleProfiles
    class MockStorageBackend:
        def retrieve(self, params):
            # Return empty memories list (no stored preferences)
            return {"memories": []}
    
    # Create all dependencies
    system_prompt = SystemPrompt()
    tone_manager = ToneManager()
    style_profiles = StyleProfiles(storage_backend=MockStorageBackend())
    response_guardrails = ResponseGuardrails()
    
    # Create PersonalityEngine with all dependencies
    personality_engine = PersonalityEngine(
        system_prompt=system_prompt,
        tone_manager=tone_manager,
        style_profiles=style_profiles,
        response_guardrails=response_guardrails,
    )
    
    # Call build_instructions
    result = personality_engine.build_instructions(
        user_id=user_id,
        context=context,
        mode=mode,
    )
    
    # Verify the result is a PromptInstructions object
    assert isinstance(result, PromptInstructions), (
        f"build_instructions() should return a PromptInstructions, got {type(result).__name__}"
    )
    
    # Verify system_identity is non-empty
    assert isinstance(result.system_identity, str), (
        f"system_identity should be a string, got {type(result.system_identity).__name__}"
    )
    assert len(result.system_identity) > 0, (
        f"system_identity should be non-empty for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got empty string"
    )
    assert result.system_identity.strip(), (
        f"system_identity should not be just whitespace for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got: {result.system_identity!r}"
    )
    
    # Verify tone_guidance is non-empty
    assert isinstance(result.tone_guidance, str), (
        f"tone_guidance should be a string, got {type(result.tone_guidance).__name__}"
    )
    assert len(result.tone_guidance) > 0, (
        f"tone_guidance should be non-empty for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got empty string"
    )
    assert result.tone_guidance.strip(), (
        f"tone_guidance should not be just whitespace for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got: {result.tone_guidance!r}"
    )
    
    # Verify style_constraints is non-empty
    assert isinstance(result.style_constraints, str), (
        f"style_constraints should be a string, got {type(result.style_constraints).__name__}"
    )
    assert len(result.style_constraints) > 0, (
        f"style_constraints should be non-empty for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got empty string"
    )
    assert result.style_constraints.strip(), (
        f"style_constraints should not be just whitespace for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got: {result.style_constraints!r}"
    )
    
    # Verify output_rules is a non-empty list
    assert isinstance(result.output_rules, list), (
        f"output_rules should be a list, got {type(result.output_rules).__name__}"
    )
    assert len(result.output_rules) > 0, (
        f"output_rules should be non-empty for user_id={user_id!r}, context={context}, mode={mode!r}, "
        f"but got empty list"
    )
    
    # Verify metadata is a dict
    assert isinstance(result.metadata, dict), (
        f"metadata should be a dict, got {type(result.metadata).__name__}"
    )


# ---------------------------------------------------------------------------
# Property 13: PersonalityEngine determinism
# ---------------------------------------------------------------------------

# Feature: luma-personality-layer, Property 13: PersonalityEngine determinism
@given(
    user_id=user_id_strategy(),
    context=adaptation_context_strategy(),
    mode=mode_strategy(),
)
@settings(max_examples=100)
def test_personality_engine_determinism(user_id, context, mode):
    """
    **Validates: Requirements 6.7, 10.4**

    For any combination of user_id, AdaptationContext, and mode, calling
    PersonalityEngine.build_instructions() twice with identical inputs SHALL
    produce identical PromptInstructions objects (same system_identity,
    tone_guidance, style_constraints, and output_rules).
    """
    from luma.core.personality.personality_engine import PersonalityEngine
    from luma.core.personality.system_prompt import SystemPrompt
    from luma.core.personality.tone_manager import ToneManager
    from luma.core.personality.style_profiles import StyleProfiles
    from luma.core.personality.response_guardrails import ResponseGuardrails
    from luma.core.personality.schemas import PromptInstructions
    
    # Create a mock storage backend for StyleProfiles
    class MockStorageBackend:
        def retrieve(self, params):
            # Return empty memories list (no stored preferences)
            return {"memories": []}
    
    # Create all dependencies
    system_prompt = SystemPrompt()
    tone_manager = ToneManager()
    style_profiles = StyleProfiles(storage_backend=MockStorageBackend())
    response_guardrails = ResponseGuardrails()
    
    # Create PersonalityEngine with all dependencies
    personality_engine = PersonalityEngine(
        system_prompt=system_prompt,
        tone_manager=tone_manager,
        style_profiles=style_profiles,
        response_guardrails=response_guardrails,
    )
    
    # Call build_instructions twice with identical inputs
    result_first_call = personality_engine.build_instructions(
        user_id=user_id,
        context=context,
        mode=mode,
    )
    result_second_call = personality_engine.build_instructions(
        user_id=user_id,
        context=context,
        mode=mode,
    )
    
    # Verify both calls return PromptInstructions objects
    assert isinstance(result_first_call, PromptInstructions), (
        f"First call to build_instructions() should return a PromptInstructions, "
        f"got {type(result_first_call).__name__}"
    )
    assert isinstance(result_second_call, PromptInstructions), (
        f"Second call to build_instructions() should return a PromptInstructions, "
        f"got {type(result_second_call).__name__}"
    )
    
    # Verify system_identity is identical
    assert result_first_call.system_identity == result_second_call.system_identity, (
        f"build_instructions() should be deterministic for user_id={user_id!r}, "
        f"context={context}, mode={mode!r}. "
        f"First call returned system_identity={result_first_call.system_identity!r}, "
        f"Second call returned system_identity={result_second_call.system_identity!r}"
    )
    
    # Verify tone_guidance is identical
    assert result_first_call.tone_guidance == result_second_call.tone_guidance, (
        f"build_instructions() should be deterministic for user_id={user_id!r}, "
        f"context={context}, mode={mode!r}. "
        f"First call returned tone_guidance={result_first_call.tone_guidance!r}, "
        f"Second call returned tone_guidance={result_second_call.tone_guidance!r}"
    )
    
    # Verify style_constraints is identical
    assert result_first_call.style_constraints == result_second_call.style_constraints, (
        f"build_instructions() should be deterministic for user_id={user_id!r}, "
        f"context={context}, mode={mode!r}. "
        f"First call returned style_constraints={result_first_call.style_constraints!r}, "
        f"Second call returned style_constraints={result_second_call.style_constraints!r}"
    )
    
    # Verify output_rules is identical
    assert result_first_call.output_rules == result_second_call.output_rules, (
        f"build_instructions() should be deterministic for user_id={user_id!r}, "
        f"context={context}, mode={mode!r}. "
        f"First call returned output_rules={result_first_call.output_rules!r}, "
        f"Second call returned output_rules={result_second_call.output_rules!r}"
    )
    
    # Verify metadata is identical
    assert result_first_call.metadata == result_second_call.metadata, (
        f"build_instructions() should be deterministic for user_id={user_id!r}, "
        f"context={context}, mode={mode!r}. "
        f"First call returned metadata={result_first_call.metadata!r}, "
        f"Second call returned metadata={result_second_call.metadata!r}"
    )
