"""
PromptBuilder — assembles a structured prompt string from a PromptContext.

Only imports from luma/core/llm/schemas.py; no other Luma modules.
Stateless: no external calls, no mutation of input.
"""

from luma.core.llm.schemas import PromptBuildError, PromptContext


class PromptBuilder:
    """Builds a structured prompt string from a PromptContext."""

    def build(self, context: PromptContext) -> str:
        """Return a structured prompt string for the given context.

        Section order:
            1. [System Instructions]
            2. [User Profile]
            3. [Relevant Memories]  (omitted when list is empty)
            4. [Current Input]
            5. [Output Constraints]

        Raises:
            PromptBuildError: if system_instructions or current_input is
                              empty or whitespace-only.
        """
        if not context.system_instructions or not context.system_instructions.strip():
            raise PromptBuildError(
                "system_instructions must not be empty or whitespace-only"
            )
        if not context.current_input or not context.current_input.strip():
            raise PromptBuildError(
                "current_input must not be empty or whitespace-only"
            )

        sections: list[str] = []

        sections.append(f"[System Instructions]\n{context.system_instructions}")
        sections.append(f"[User Profile]\n{context.user_profile}")

        if context.relevant_memories:
            numbered = "\n".join(
                f"{i}. {mem}" for i, mem in enumerate(context.relevant_memories, start=1)
            )
            sections.append(f"[Relevant Memories]\n{numbered}")

        sections.append(f"[Current Input]\n{context.current_input}")
        sections.append(f"[Output Constraints]\n{context.output_constraints}")

        return "\n\n".join(sections)
