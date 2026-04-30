"""
Unit tests for ResponseParser.

**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.10, 12.5**
"""

import pytest

from luma.core.llm.response_parser import ResponseParser
from luma.core.llm.schemas import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(raw_text: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> LLMResponse:
    return LLMResponse(
        request_id="req-001",
        raw_text=raw_text,
        model="gpt-test",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider="test-provider",
    )


# ---------------------------------------------------------------------------
# Whitespace stripping (Req 4.2)
# ---------------------------------------------------------------------------

def test_strips_leading_whitespace():
    parser = ResponseParser()
    result = parser.parse(make_response("   Hello world."))
    assert result.text == "Hello world."


def test_strips_trailing_whitespace():
    parser = ResponseParser()
    result = parser.parse(make_response("Hello world.   "))
    assert result.text == "Hello world."


def test_strips_leading_and_trailing_whitespace():
    parser = ResponseParser()
    result = parser.parse(make_response("  \n  Hello world.\n  "))
    assert result.text == "Hello world."


# ---------------------------------------------------------------------------
# Empty response (Req 4.5, 12.5)
# ---------------------------------------------------------------------------

def test_empty_raw_text_sets_is_valid_false():
    parser = ResponseParser()
    result = parser.parse(make_response(""))
    assert result.is_valid is False


def test_empty_raw_text_adds_empty_response_note():
    parser = ResponseParser()
    result = parser.parse(make_response(""))
    assert "empty response" in result.validation_notes


def test_whitespace_only_text_treated_as_empty():
    parser = ResponseParser()
    result = parser.parse(make_response("   \t\n  "))
    assert result.is_valid is False
    assert "empty response" in result.validation_notes


def test_empty_response_truncated_is_false():
    parser = ResponseParser()
    result = parser.parse(make_response(""))
    assert result.truncated is False


# ---------------------------------------------------------------------------
# Within-limit response (Req 4.4, 4.6)
# ---------------------------------------------------------------------------

def test_within_limit_is_valid_true():
    parser = ResponseParser(max_response_chars=100)
    result = parser.parse(make_response("Short response."))
    assert result.is_valid is True


def test_within_limit_truncated_false():
    parser = ResponseParser(max_response_chars=100)
    result = parser.parse(make_response("Short response."))
    assert result.truncated is False


def test_within_limit_validation_notes_empty():
    parser = ResponseParser(max_response_chars=100)
    result = parser.parse(make_response("Short response."))
    assert result.validation_notes == []


# ---------------------------------------------------------------------------
# Truncation at sentence boundary (Req 4.3)
# ---------------------------------------------------------------------------

def test_truncation_sets_truncated_true():
    parser = ResponseParser(max_response_chars=20)
    # "Hello world. Extra." is 19 chars, but let's make it exceed 20
    long_text = "Hello world. This is extra text that goes beyond the limit."
    result = parser.parse(make_response(long_text))
    assert result.truncated is True


def test_truncation_sets_is_valid_true():
    parser = ResponseParser(max_response_chars=20)
    long_text = "Hello world. This is extra text that goes beyond the limit."
    result = parser.parse(make_response(long_text))
    assert result.is_valid is True


def test_truncation_at_sentence_boundary():
    parser = ResponseParser(max_response_chars=20)
    # "Hello world." is 12 chars, fits within 20; "Hello world. Extra." is 19 chars
    # "Hello world. Extra text." exceeds 20 — should truncate to "Hello world."
    text = "Hello world. Extra text beyond limit."
    result = parser.parse(make_response(text))
    assert result.text.endswith(".")
    assert len(result.text) <= 20


def test_truncation_sentence_boundary_uses_last_sentence_at_or_before_limit():
    parser = ResponseParser(max_response_chars=30)
    # "First sentence. Second one." = 27 chars (within 30)
    # "First sentence. Second one. Third." = 34 chars (exceeds 30)
    text = "First sentence. Second one. Third sentence here."
    result = parser.parse(make_response(text))
    # Should truncate to "First sentence. Second one." (27 chars)
    assert result.text == "First sentence. Second one."
    assert result.truncated is True


def test_truncation_no_sentence_boundary_hard_truncates():
    parser = ResponseParser(max_response_chars=10)
    # No sentence-ending punctuation in first 10 chars
    text = "abcdefghijklmnopqrstuvwxyz"
    result = parser.parse(make_response(text))
    assert result.text == "abcdefghij"
    assert result.truncated is True


def test_truncation_exclamation_mark_boundary():
    parser = ResponseParser(max_response_chars=20)
    text = "Hello world! Extra text beyond limit."
    result = parser.parse(make_response(text))
    assert result.text == "Hello world!"
    assert result.truncated is True


def test_truncation_question_mark_boundary():
    parser = ResponseParser(max_response_chars=20)
    text = "Are you there? Extra text beyond limit."
    result = parser.parse(make_response(text))
    assert result.text == "Are you there?"
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Token usage (Req 4.7)
# ---------------------------------------------------------------------------

def test_token_usage_populated_from_response():
    parser = ResponseParser()
    result = parser.parse(make_response("Hello.", prompt_tokens=42, completion_tokens=7))
    assert result.token_usage == {"prompt": 42, "completion": 7}


def test_token_usage_populated_for_empty_response():
    parser = ResponseParser()
    result = parser.parse(make_response("", prompt_tokens=5, completion_tokens=0))
    assert result.token_usage == {"prompt": 5, "completion": 0}


def test_token_usage_populated_for_truncated_response():
    parser = ResponseParser(max_response_chars=10)
    result = parser.parse(make_response("abcdefghijklmnop", prompt_tokens=100, completion_tokens=50))
    assert result.token_usage == {"prompt": 100, "completion": 50}


# ---------------------------------------------------------------------------
# Input not mutated (Req 4.10)
# ---------------------------------------------------------------------------

def test_input_llm_response_not_mutated():
    parser = ResponseParser()
    original_text = "  Hello world.  "
    response = make_response(original_text)
    parser.parse(response)
    assert response.raw_text == original_text


# ---------------------------------------------------------------------------
# request_id propagation
# ---------------------------------------------------------------------------

def test_request_id_propagated():
    parser = ResponseParser()
    response = LLMResponse(
        request_id="my-unique-id",
        raw_text="Some text.",
        model="gpt-test",
        prompt_tokens=1,
        completion_tokens=1,
        provider="test",
    )
    result = parser.parse(response)
    assert result.request_id == "my-unique-id"
