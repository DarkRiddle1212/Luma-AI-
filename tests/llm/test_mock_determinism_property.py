"""
Property-based test for MockProvider determinism.

**Validates: Requirements 11.2, 11.4**

Property 12: Mock Determinism
For any list of response dicts, MockProvider returns responses in order,
one per generate() call, regardless of list length or response content.
"""

import pytest
from hypothesis import given, settings, strategies as st

from luma.core.llm.providers.mock_provider import MockProvider


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single mock response dict with arbitrary string values
response_dict_strategy = st.fixed_dictionaries(
    {
        "text": st.text(min_size=0, max_size=200),
        "model": st.text(min_size=1, max_size=50),
        "prompt_tokens": st.integers(min_value=0, max_value=10000),
        "completion_tokens": st.integers(min_value=0, max_value=10000),
        "provider": st.text(min_size=1, max_size=20),
    }
)

# Lists of 1 to 20 response dicts
response_list_strategy = st.lists(response_dict_strategy, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

class TestMockDeterminism:
    """
    Property 12: Mock Determinism

    **Validates: Requirements 11.2, 11.4**

    Feature: gemini-provider-integration, Property 12: Mock determinism
    """

    @pytest.mark.property_test
    @given(responses=response_list_strategy)
    @settings(max_examples=100, deadline=None)
    def test_mock_returns_responses_in_order(self, responses: list):
        """
        For any list of response dicts of length N, MockProvider returns
        each response at the corresponding index when generate() is called N times.

        **Validates: Requirements 11.2, 11.4**
        """
        provider = MockProvider(config={"responses": responses})

        for i, expected in enumerate(responses):
            result = provider.generate(prompt="test", options={})
            assert result == expected, (
                f"Call {i}: expected {expected!r}, got {result!r}. "
                "MockProvider must return responses in order."
            )
