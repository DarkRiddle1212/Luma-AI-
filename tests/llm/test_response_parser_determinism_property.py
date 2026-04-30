"""
Property-based test for ResponseParser determinism.

**Validates: Requirements 4.8, 11.2**

Property: identical LLMResponse inputs always produce identical ParsedResponse outputs.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from luma.core.llm.response_parser import ResponseParser
from luma.core.llm.schemas import LLMResponse


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

llm_response_strategy = st.builds(
    LLMResponse,
    request_id=st.text(min_size=1, max_size=50),
    raw_text=st.text(max_size=8000),
    model=st.text(min_size=1, max_size=50),
    prompt_tokens=st.integers(min_value=0, max_value=10000),
    completion_tokens=st.integers(min_value=0, max_value=10000),
    provider=st.text(min_size=1, max_size=50),
)


# ---------------------------------------------------------------------------
# Property: determinism
# ---------------------------------------------------------------------------

@given(response=llm_response_strategy)
@settings(max_examples=200)
def test_response_parser_is_deterministic(response: LLMResponse) -> None:
    """
    Parsing the same LLMResponse twice must yield identical ParsedResponse objects.

    **Validates: Requirements 4.8, 11.2**
    """
    parser = ResponseParser()
    result1 = parser.parse(response)
    result2 = parser.parse(response)

    assert result1.request_id == result2.request_id
    assert result1.text == result2.text
    assert result1.is_valid == result2.is_valid
    assert result1.validation_notes == result2.validation_notes
    assert result1.token_usage == result2.token_usage
    assert result1.truncated == result2.truncated
