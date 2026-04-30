"""
Unit tests for ResponseGuardrails component.

Tests quality violation detection (rambling, repetition, contradiction, vague filler),
length constraint enforcement, score calculation, and deterministic behavior.
"""

import pytest

from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.personality.schemas import GuardrailResult


@pytest.fixture
def guardrails():
    """Create a ResponseGuardrails instance."""
    return ResponseGuardrails()


class TestResponseGuardrailsEmptyResponse:
    """Test handling of empty responses."""

    def test_empty_string_fails_validation(self, guardrails):
        """Empty response should fail with score 0.0."""
        result = guardrails.validate("", [])
        assert result.passed is False
        assert "empty response" in result.violations
        assert result.score == 0.0
        assert "empty" in result.notes.lower()

    def test_whitespace_only_fails_validation(self, guardrails):
        """Whitespace-only response should fail with score 0.0."""
        result = guardrails.validate("   \n\t  ", [])
        assert result.passed is False
        assert "empty response" in result.violations
        assert result.score == 0.0


class TestResponseGuardrailsCleanResponse:
    """Test clean responses that pass all checks."""

    def test_clean_short_response_passes(self, guardrails):
        """Clean, short response should pass with score 1.0."""
        response = "This is a clear and concise response with no issues."
        result = guardrails.validate(response, [])
        assert result.passed is True
        assert result.violations == []
        assert result.score == 1.0
        assert "passed" in result.notes.lower()

    def test_clean_structured_response_passes(self, guardrails):
        """Clean, structured response should pass with score 1.0."""
        response = """
        Here's a structured response:
        
        1. First point with clear explanation
        2. Second point with specific details
        3. Third point with concrete examples
        
        This response has clear structure and no quality issues.
        """
        result = guardrails.validate(response, [])
        assert result.passed is True
        assert result.violations == []
        assert result.score == 1.0


class TestResponseGuardrailsRambling:
    """Test rambling detection."""

    def test_long_unstructured_response_is_rambling(self, guardrails):
        """Response >500 words without structure should be flagged as rambling."""
        # Create a long, unstructured response (>500 words)
        words = ["word"] * 600
        response = " ".join(words)
        result = guardrails.validate(response, [])
        assert "rambling" in result.violations
        assert result.passed is False

    def test_long_structured_response_not_rambling(self, guardrails):
        """Response >500 words with clear structure should not be flagged."""
        # Create a long but structured response with bullet points
        response = "Here are the key points:\n\n"
        for i in range(100):
            response += f"- Point {i}: " + " ".join(["detail"] * 5) + "\n"
        result = guardrails.validate(response, [])
        assert "rambling" not in result.violations

    def test_short_response_not_rambling(self, guardrails):
        """Response ≤500 words should never be flagged as rambling."""
        words = ["word"] * 400
        response = " ".join(words)
        result = guardrails.validate(response, [])
        assert "rambling" not in result.violations


class TestResponseGuardrailsRepetition:
    """Test repetition detection."""

    def test_repeated_five_word_sequence_detected(self, guardrails):
        """Sequence appearing >2 times should be flagged as repetition."""
        sequence = "this is a repeated five word sequence"
        response = f"{sequence}. Some other text. {sequence}. More text. {sequence}."
        result = guardrails.validate(response, [])
        assert "repetition" in result.violations
        assert result.score == 0.75  # One violation: 1.0 - 0.25 = 0.75
        assert result.passed is True  # 0.75 >= 0.75 threshold

    def test_two_occurrences_not_repetition(self, guardrails):
        """Sequence appearing exactly 2 times should not be flagged."""
        sequence = "this is a repeated five word sequence"
        response = f"{sequence}. Some other text. {sequence}. Different content here."
        result = guardrails.validate(response, [])
        assert "repetition" not in result.violations

    def test_short_response_no_repetition(self, guardrails):
        """Response <5 words should not trigger repetition check."""
        response = "Short response here"
        result = guardrails.validate(response, [])
        assert "repetition" not in result.violations


class TestResponseGuardrailsContradiction:
    """Test contradiction detection."""

    def test_but_actually_detected(self, guardrails):
        """'but actually' should be flagged as contradiction."""
        response = "The answer is yes, but actually it's no."
        result = guardrails.validate(response, [])
        assert "contradiction" in result.violations
        assert result.score == 0.75  # One violation: 1.0 - 0.25 = 0.75
        assert result.passed is True  # 0.75 >= 0.75 threshold

    def test_on_the_other_hand_detected(self, guardrails):
        """'on the other hand' should be flagged as contradiction."""
        response = "This is correct. On the other hand, it's incorrect."
        result = guardrails.validate(response, [])
        assert "contradiction" in result.violations

    def test_however_detected(self, guardrails):
        """'however' followed by 'the/this/that' should be flagged."""
        response = "This is true. However, the opposite is also true."
        result = guardrails.validate(response, [])
        assert "contradiction" in result.violations

    def test_no_contradiction_markers(self, guardrails):
        """Response without contradiction markers should pass."""
        response = "This is a consistent response with no contradictions."
        result = guardrails.validate(response, [])
        assert "contradiction" not in result.violations


class TestResponseGuardrailsVagueFiller:
    """Test vague filler detection."""

    def test_many_filler_phrases_without_examples(self, guardrails):
        """>3 filler phrases without concrete examples should be flagged."""
        response = """
        It depends on the situation. Generally speaking, this is true.
        In most cases, you'll find that typically this happens.
        Usually, it could be different, and sometimes it might be another way.
        """
        result = guardrails.validate(response, [])
        assert "vague filler" in result.violations
        assert result.score == 0.75  # One violation: 1.0 - 0.25 = 0.75
        assert result.passed is True  # 0.75 >= 0.75 threshold

    def test_filler_phrases_with_numbers_not_vague(self, guardrails):
        """Filler phrases with concrete numbers should not be flagged."""
        response = """
        It depends on the situation. Generally speaking, this is true.
        In most cases, you'll find that typically this happens.
        Usually, it could be different. For example, 42 is the answer.
        Sometimes it might be 100 or 200 depending on context.
        """
        result = guardrails.validate(response, [])
        assert "vague filler" not in result.violations

    def test_filler_phrases_with_code_not_vague(self, guardrails):
        """Filler phrases with code snippets should not be flagged."""
        response = """
        It depends on the situation. Generally speaking, use `function()`.
        In most cases, you'll find that `variable = value` works.
        Usually, it could be different with `other_function()`.
        Sometimes it might be `another_approach()` depending on context.
        """
        result = guardrails.validate(response, [])
        assert "vague filler" not in result.violations

    def test_few_filler_phrases_not_vague(self, guardrails):
        """≤3 filler phrases should not be flagged."""
        response = "It depends on the situation. Generally speaking, this is true."
        result = guardrails.validate(response, [])
        assert "vague filler" not in result.violations


class TestResponseGuardrailsLengthConstraint:
    """Test length constraint enforcement."""

    def test_concise_constraint_over_200_words_fails(self, guardrails):
        """Response >200 words with 'concise' constraint should be flagged."""
        words = ["word"] * 250
        response = " ".join(words)
        result = guardrails.validate(response, ["concise"])
        assert "exceeds concise length constraint" in result.violations
        assert result.passed is False

    def test_concise_constraint_under_200_words_passes(self, guardrails):
        """Response ≤200 words with 'concise' constraint should pass."""
        words = ["word"] * 150
        response = " ".join(words)
        result = guardrails.validate(response, ["concise"])
        assert "exceeds concise length constraint" not in result.violations

    def test_no_concise_constraint_ignores_length(self, guardrails):
        """Without 'concise' constraint, length should not be checked."""
        words = ["word"] * 300
        response = " ".join(words)
        result = guardrails.validate(response, [])
        assert "exceeds concise length constraint" not in result.violations


class TestResponseGuardrailsScoreCalculation:
    """Test score calculation and pass/fail logic."""

    def test_zero_violations_score_1_0(self, guardrails):
        """Zero violations should result in score 1.0."""
        response = "Clean response with no issues."
        result = guardrails.validate(response, [])
        assert result.score == 1.0
        assert result.passed is True

    def test_one_violation_score_0_75(self, guardrails):
        """One violation should result in score 0.75 (still passes)."""
        response = "The answer is yes, but actually it's no."
        result = guardrails.validate(response, [])
        assert len(result.violations) == 1
        assert result.score == 0.75
        assert result.passed is True  # 0.75 >= 0.75

    def test_two_violations_score_0_5(self, guardrails):
        """Two violations should result in score 0.5 (fails)."""
        response = """
        The answer is yes, but actually it's no.
        It depends on many things. Generally speaking, this is true.
        In most cases, you'll find that typically this happens.
        Usually, it could be different, and sometimes it might be another way.
        """
        result = guardrails.validate(response, [])
        assert len(result.violations) == 2
        assert result.score == 0.5
        assert result.passed is False  # 0.5 < 0.75

    def test_three_violations_score_0_25(self, guardrails):
        """Three violations should result in score 0.25 (fails)."""
        # Create response with contradiction, vague filler, and repetition
        sequence = "this is a repeated five word sequence"
        response = f"""
        {sequence}. Some text. {sequence}. More text. {sequence}.
        The answer is yes, but actually it's no.
        It depends on many things. Generally speaking, this is true.
        In most cases, you'll find that typically this happens.
        Usually, it could be different, and sometimes it might be another way.
        """
        result = guardrails.validate(response, [])
        assert len(result.violations) == 3
        assert result.score == 0.25
        assert result.passed is False

    def test_four_violations_score_0_0(self, guardrails):
        """Four violations should result in score 0.0 (clamped)."""
        # Create response with all four violations
        sequence = "this is a repeated five word sequence"
        words = ["word"] * 600  # rambling
        response = f"""
        {' '.join(words)}
        {sequence}. Some text. {sequence}. More text. {sequence}.
        The answer is yes, but actually it's no.
        It depends on many things. Generally speaking, this is true.
        In most cases, you'll find that typically this happens.
        Usually, it could be different, and sometimes it might be another way.
        """
        result = guardrails.validate(response, [])
        assert len(result.violations) == 4
        assert result.score == 0.0
        assert result.passed is False


class TestResponseGuardrailsDeterminism:
    """Test deterministic behavior."""

    def test_identical_inputs_produce_identical_outputs(self, guardrails):
        """Calling validate twice with identical inputs should produce identical results."""
        response = "This is a test response with some content."
        constraints = ["concise"]

        result1 = guardrails.validate(response, constraints)
        result2 = guardrails.validate(response, constraints)

        assert result1.passed == result2.passed
        assert result1.violations == result2.violations
        assert result1.score == result2.score
        assert result1.notes == result2.notes

    def test_determinism_with_violations(self, guardrails):
        """Determinism should hold even with violations present."""
        response = "The answer is yes, but actually it's no."

        result1 = guardrails.validate(response, [])
        result2 = guardrails.validate(response, [])

        assert result1.passed == result2.passed
        assert result1.violations == result2.violations
        assert result1.score == result2.score


class TestResponseGuardrailsNotes:
    """Test notes generation."""

    def test_notes_describe_violations(self, guardrails):
        """Notes should describe detected violations."""
        response = "The answer is yes, but actually it's no."
        result = guardrails.validate(response, [])
        assert "violation" in result.notes.lower()
        assert "contradiction" in result.notes.lower()

    def test_notes_confirm_pass(self, guardrails):
        """Notes should confirm when response passes."""
        response = "Clean response with no issues."
        result = guardrails.validate(response, [])
        assert "passed" in result.notes.lower()
