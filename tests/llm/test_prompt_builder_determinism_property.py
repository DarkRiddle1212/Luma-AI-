"""
Property-based test for PromptBuilder determinism.

**Validates: Requirements 3.5, 11.1**

Property: identical PromptContext inputs always produce identical prompt strings.
"""

import pytest
from hypothesis import given, settings, strategies as st

from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.schemas import PromptContext


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

prompt_context_strategy = st.builds(
    PromptContext,
    system_instructions=non_empty_text,
    user_profile=st.text(),
    relevant_memories=st.lists(st.text(), max_size=10),
    current_input=non_empty_text,
    output_constraints=st.text(),
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

class TestPromptBuilderDeterminism:
    """
    Property: PromptBuilder is stateless and deterministic.

    **Validates: Requirements 3.5, 11.1**
    """

    @pytest.mark.property_test
    @given(context=prompt_context_strategy)
    @settings(max_examples=50, deadline=None)
    def test_identical_inputs_produce_identical_outputs(self, context: PromptContext):
        """
        For any valid PromptContext, calling build() twice returns the same string.

        **Validates: Requirements 3.5, 11.1**
        """
        builder = PromptBuilder()
        result_1 = builder.build(context)
        result_2 = builder.build(context)
        assert result_1 == result_2, (
            "PromptBuilder must be deterministic: same input must always yield same output."
        )

    @pytest.mark.property_test
    @given(context=prompt_context_strategy)
    @settings(max_examples=50, deadline=None)
    def test_two_builder_instances_produce_identical_outputs(self, context: PromptContext):
        """
        Two separate PromptBuilder instances produce the same output for the same input.

        **Validates: Requirements 3.5, 11.1**
        """
        result_1 = PromptBuilder().build(context)
        result_2 = PromptBuilder().build(context)
        assert result_1 == result_2
