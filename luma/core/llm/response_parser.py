"""
ResponseParser — cleans, validates, and structures raw LLM output.

Stateless component: given identical LLMResponse inputs, always produces
identical ParsedResponse outputs. No external calls, no mutation of input.
Only imports from luma/core/llm/schemas.py.
"""

from luma.core.llm.schemas import LLMResponse, ParsedResponse

_SENTENCE_ENDINGS = frozenset(".!?")


class ResponseParser:
    """Cleans and validates raw LLM output into a ParsedResponse."""

    def __init__(self, max_response_chars: int = 4000) -> None:
        self._max_chars = max_response_chars

    def parse(self, response: LLMResponse) -> ParsedResponse:
        # 1. Strip whitespace (do not mutate input)
        cleaned = response.raw_text.strip()

        # 2. Check empty
        if not cleaned:
            return ParsedResponse(
                request_id=response.request_id,
                text="",
                is_valid=False,
                validation_notes=["empty response"],
                token_usage={
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                },
                truncated=False,
            )

        # 3. Check length and truncate if needed
        if len(cleaned) > self._max_chars:
            truncated_text = self._truncate_to_sentence(cleaned, self._max_chars)
            return ParsedResponse(
                request_id=response.request_id,
                text=truncated_text,
                is_valid=True,
                validation_notes=[],
                token_usage={
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                },
                truncated=True,
            )

        # 4. Within limits
        return ParsedResponse(
            request_id=response.request_id,
            text=cleaned,
            is_valid=True,
            validation_notes=[],
            token_usage={
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
            },
            truncated=False,
        )

    @staticmethod
    def _truncate_to_sentence(text: str, limit: int) -> str:
        """Truncate text to the nearest sentence boundary at or below limit."""
        # Search backwards from limit for a sentence-ending character
        candidate = text[:limit]
        for i in range(limit - 1, -1, -1):
            if text[i] in _SENTENCE_ENDINGS:
                return text[: i + 1]
        # No sentence boundary found — hard truncate at limit
        return candidate
