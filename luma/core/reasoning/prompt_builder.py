"""
Prompt Builder for the Reasoning Engine.

This module provides the Prompt_Builder class that constructs structured prompts
for language models by combining system instructions, context, user queries, and
response instructions.
"""

from typing import Dict, Any, List


class Prompt_Builder:
    """
    Constructs structured prompts for the reasoning engine.
    
    The Prompt_Builder creates prompts with multiple sections:
    - System: Describes Luma as a cognitive memory assistant
    - Context: Injected memories in structured format (if available)
    - User Question: The user's query
    - Instructions: Directs the model to use context and reference memories
    
    The builder handles cases where no context is available by omitting the
    context section entirely.
    """
    
    def build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Build a structured prompt from a query and context.
        
        Constructs a multi-section prompt that includes system instructions,
        optional context with memories, the user's query, and instructions
        for the language model.
        
        Args:
            query (str): The user's query or question to be processed.
            context (Dict[str, Any]): Context dictionary containing memories and metadata.
                Expected structure: {"memories": [...], "metadata": {...}}
        
        Returns:
            str: The constructed prompt string with all relevant sections.
        
        Example:
            >>> builder = Prompt_Builder()
            >>> context = {
            ...     "memories": [
            ...         {"id": "mem_1", "content": "User likes Python", "metadata": {}}
            ...     ],
            ...     "metadata": {}
            ... }
            >>> prompt = builder.build_prompt("What do I like?", context)
        """
        sections = []
        
        # System section
        system_section = self._build_system_section()
        sections.append(system_section)
        
        # Context section (only if memories are present)
        memories = context.get("memories", [])
        if memories:
            context_section = self._build_context_section(memories)
            sections.append(context_section)
        
        # User question section
        user_section = self._build_user_section(query)
        sections.append(user_section)
        
        # Instructions section
        instructions_section = self._build_instructions_section()
        sections.append(instructions_section)
        
        return "\n\n".join(sections)
    
    def _build_system_section(self) -> str:
        """
        Build the system section describing Luma.
        
        Returns:
            str: System section text.
        """
        return (
            "# System\n\n"
            "You are Luma, a cognitive memory assistant. Your purpose is to help users "
            "by leveraging their stored memories to provide personalized, context-aware responses. "
            "You have access to relevant memories that have been retrieved based on the user's query."
        )
    
    def _build_context_section(self, memories: List[Dict[str, Any]]) -> str:
        """
        Build the context section with injected memories.
        
        Args:
            memories (List[Dict[str, Any]]): List of memory objects with id, content, and metadata.
        
        Returns:
            str: Context section text with structured memory information.
        """
        context_lines = ["# Context\n", "The following memories are relevant to this query:\n"]
        
        for i, memory in enumerate(memories, 1):
            memory_id = memory.get("id", "unknown")
            content = memory.get("content", "")
            context_lines.append(f"\n**Memory {i} (ID: {memory_id})**")
            context_lines.append(f"{content}")
        
        return "\n".join(context_lines)
    
    def _build_user_section(self, query: str) -> str:
        """
        Build the user question section.
        
        Args:
            query (str): The user's query.
        
        Returns:
            str: User question section text.
        """
        return f"# User Question\n\n{query}"
    
    def _build_instructions_section(self) -> str:
        """
        Build the instructions section for the language model.
        
        Returns:
            str: Instructions section text.
        """
        return (
            "# Instructions\n\n"
            "Please answer the user's question using the provided context. "
            "Reference specific memories by their IDs when they inform your answer. "
            "If the context doesn't contain relevant information, acknowledge this and "
            "provide the best answer you can based on general knowledge."
        )


__all__ = ['Prompt_Builder']
