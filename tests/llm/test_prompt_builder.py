"""
Unit tests for luma.core.llm.prompt_builder.PromptBuilder.

**Validates: Requirements 3.2, 3.3, 3.4, 3.6, 3.7, 3.9, 12.1, 12.2**
"""

import copy

import pytest

from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.schemas import PromptBuildError, PromptContext


def make_context(**kwargs) -> PromptContext:
    """Helper: build a PromptContext with sensible defaults."""
    defaults = dict(
        system_instructions="You are a helpful assistant.",
        user_profile="User prefers concise answers.",
        relevant_memories=[],
        current_input="Hello, how are you?",
        output_constraints="Be brief.",
    )
    defaults.update(kwargs)
    return PromptContext(**defaults)


class TestSectionOrdering:
    """Requirement 3.2 — section order in output."""

    def test_sections_appear_in_correct_order(self):
        builder = PromptBuilder()
        ctx = make_context(relevant_memories=["Memory one", "Memory two"])
        prompt = builder.build(ctx)

        si_pos = prompt.index("[System Instructions]")
        up_pos = prompt.index("[User Profile]")
        rm_pos = prompt.index("[Relevant Memories]")
        ci_pos = prompt.index("[Current Input]")
        oc_pos = prompt.index("[Output Constraints]")

        assert si_pos < up_pos < rm_pos < ci_pos < oc_pos

    def test_sections_appear_in_correct_order_without_memories(self):
        builder = PromptBuilder()
        ctx = make_context(relevant_memories=[])
        prompt = builder.build(ctx)

        si_pos = prompt.index("[System Instructions]")
        up_pos = prompt.index("[User Profile]")
        ci_pos = prompt.index("[Current Input]")
        oc_pos = prompt.index("[Output Constraints]")

        assert si_pos < up_pos < ci_pos < oc_pos


class TestMemoriesSection:
    """Requirements 3.3, 3.4 — numbered list vs. omitted when empty."""

    def test_memories_numbered_when_present(self):
        builder = PromptBuilder()
        ctx = make_context(relevant_memories=["Memory one", "Memory two"])
        prompt = builder.build(ctx)

        assert "[Relevant Memories]" in prompt
        assert "1. Memory one" in prompt
        assert "2. Memory two" in prompt

    def test_memories_section_omitted_when_empty(self):
        builder = PromptBuilder()
        ctx = make_context(relevant_memories=[])
        prompt = builder.build(ctx)

        assert "[Relevant Memories]" not in prompt

    def test_single_memory_numbered(self):
        builder = PromptBuilder()
        ctx = make_context(relevant_memories=["Only memory"])
        prompt = builder.build(ctx)

        assert "1. Only memory" in prompt


class TestValidation:
    """Requirements 3.6, 3.7, 12.1, 12.2 — PromptBuildError on bad input."""

    def test_raises_on_empty_current_input(self):
        builder = PromptBuilder()
        ctx = make_context(current_input="")
        with pytest.raises(PromptBuildError):
            builder.build(ctx)

    def test_raises_on_whitespace_only_current_input(self):
        builder = PromptBuilder()
        ctx = make_context(current_input="   \t\n  ")
        with pytest.raises(PromptBuildError):
            builder.build(ctx)

    def test_raises_on_empty_system_instructions(self):
        builder = PromptBuilder()
        ctx = make_context(system_instructions="")
        with pytest.raises(PromptBuildError):
            builder.build(ctx)

    def test_raises_on_whitespace_only_system_instructions(self):
        builder = PromptBuilder()
        ctx = make_context(system_instructions="   ")
        with pytest.raises(PromptBuildError):
            builder.build(ctx)

    def test_error_message_mentions_current_input(self):
        builder = PromptBuilder()
        ctx = make_context(current_input="")
        with pytest.raises(PromptBuildError, match="current_input"):
            builder.build(ctx)

    def test_error_message_mentions_system_instructions(self):
        builder = PromptBuilder()
        ctx = make_context(system_instructions="")
        with pytest.raises(PromptBuildError, match="system_instructions"):
            builder.build(ctx)


class TestNoMutation:
    """Requirement 3.9 — input PromptContext must not be mutated."""

    def test_context_not_mutated(self):
        builder = PromptBuilder()
        memories = ["mem one", "mem two"]
        ctx = make_context(relevant_memories=memories)
        original_memories = list(ctx.relevant_memories)
        original_input = ctx.current_input
        original_instructions = ctx.system_instructions

        builder.build(ctx)

        assert list(ctx.relevant_memories) == original_memories
        assert ctx.current_input == original_input
        assert ctx.system_instructions == original_instructions


class TestOutputContent:
    """Verify section headers and content appear correctly."""

    def test_system_instructions_content_present(self):
        builder = PromptBuilder()
        ctx = make_context(system_instructions="Act as a tutor.")
        prompt = builder.build(ctx)
        assert "Act as a tutor." in prompt

    def test_user_profile_content_present(self):
        builder = PromptBuilder()
        ctx = make_context(user_profile="Prefers bullet points.")
        prompt = builder.build(ctx)
        assert "Prefers bullet points." in prompt

    def test_current_input_content_present(self):
        builder = PromptBuilder()
        ctx = make_context(current_input="What is 2+2?")
        prompt = builder.build(ctx)
        assert "What is 2+2?" in prompt

    def test_output_constraints_content_present(self):
        builder = PromptBuilder()
        ctx = make_context(output_constraints="Keep it under 50 words.")
        prompt = builder.build(ctx)
        assert "Keep it under 50 words." in prompt

    def test_sections_separated_by_blank_lines(self):
        builder = PromptBuilder()
        ctx = make_context()
        prompt = builder.build(ctx)
        assert "\n\n" in prompt
