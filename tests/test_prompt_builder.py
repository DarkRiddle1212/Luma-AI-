"""
Unit tests for Prompt_Builder.

Tests prompt construction with valid query and context, empty context,
required sections, and memory identifier inclusion.
"""

import pytest
from luma.core.reasoning.prompt_builder import Prompt_Builder


class TestPromptBuilderBasicConstruction:
    """Test suite for basic prompt construction functionality."""
    
    def test_prompt_construction_with_valid_query_and_context(self):
        """Test prompt construction with valid query and context containing memories."""
        builder = Prompt_Builder()
        query = "What programming languages do I like?"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}},
                {"id": "mem_2", "content": "User prefers JavaScript for web development", "metadata": {}}
            ],
            "metadata": {"user_id": "123"}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is a non-empty string
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Verify query is included in the prompt
        assert query in prompt
        
        # Verify memory content is included
        assert "User likes Python" in prompt
        assert "User prefers JavaScript for web development" in prompt
        
        # Verify memory IDs are included
        assert "mem_1" in prompt
        assert "mem_2" in prompt
    
    def test_prompt_construction_with_empty_context(self):
        """Test prompt construction with empty context (no memories)."""
        builder = Prompt_Builder()
        query = "What is Python?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is a non-empty string
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Verify query is included
        assert query in prompt
        
        # Verify context section is NOT included when there are no memories
        # The context section should not appear
        assert "# Context" not in prompt or "The following memories are relevant" not in prompt
    
    def test_prompt_construction_with_missing_memories_key(self):
        """Test prompt construction when context dict doesn't have 'memories' key."""
        builder = Prompt_Builder()
        query = "What is machine learning?"
        context = {"metadata": {"user_id": "456"}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is constructed without errors
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert query in prompt
        
        # Verify no context section is included
        assert "# Context" not in prompt or "The following memories are relevant" not in prompt


class TestPromptBuilderRequiredSections:
    """Test suite for verifying all required sections are present in prompts."""
    
    def test_prompt_contains_system_section(self):
        """Test that prompt contains the system section describing Luma."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify system section is present
        assert "# System" in prompt
        assert "Luma" in prompt
        assert "cognitive memory assistant" in prompt
    
    def test_prompt_contains_user_question_section(self):
        """Test that prompt contains the user question section."""
        builder = Prompt_Builder()
        query = "What is my favorite color?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify user question section is present
        assert "# User Question" in prompt
        assert query in prompt
    
    def test_prompt_contains_instructions_section(self):
        """Test that prompt contains the instructions section."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify instructions section is present
        assert "# Instructions" in prompt
        assert "answer the user's question" in prompt.lower()
        assert "context" in prompt.lower()
    
    def test_prompt_contains_context_section_when_memories_present(self):
        """Test that prompt contains context section when memories are provided."""
        builder = Prompt_Builder()
        query = "What do I like?"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes hiking", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify context section is present
        assert "# Context" in prompt
        assert "relevant to this query" in prompt.lower() or "memories" in prompt.lower()
    
    def test_prompt_sections_order(self):
        """Test that prompt sections appear in the correct order."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"id": "mem_1", "content": "Test memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Find positions of each section
        system_pos = prompt.find("# System")
        context_pos = prompt.find("# Context")
        user_pos = prompt.find("# User Question")
        instructions_pos = prompt.find("# Instructions")
        
        # Verify sections appear in correct order
        assert system_pos < context_pos < user_pos < instructions_pos


class TestPromptBuilderMemoryIdentifiers:
    """Test suite for verifying memory identifiers are included in prompts."""
    
    def test_single_memory_identifier_included(self):
        """Test that a single memory identifier is included in the prompt."""
        builder = Prompt_Builder()
        query = "What do I remember?"
        context = {
            "memories": [
                {"id": "mem_abc123", "content": "Important memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify memory ID is present
        assert "mem_abc123" in prompt
    
    def test_multiple_memory_identifiers_included(self):
        """Test that all memory identifiers are included in the prompt."""
        builder = Prompt_Builder()
        query = "What are my preferences?"
        context = {
            "memories": [
                {"id": "mem_001", "content": "First memory", "metadata": {}},
                {"id": "mem_002", "content": "Second memory", "metadata": {}},
                {"id": "mem_003", "content": "Third memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify all memory IDs are present
        assert "mem_001" in prompt
        assert "mem_002" in prompt
        assert "mem_003" in prompt
    
    def test_memory_identifier_format_in_prompt(self):
        """Test that memory identifiers are formatted correctly in the prompt."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"id": "mem_xyz", "content": "Test content", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify memory ID appears with proper formatting (e.g., "ID: mem_xyz")
        assert "mem_xyz" in prompt
        assert "ID:" in prompt or "id:" in prompt.lower()
    
    def test_memory_content_and_identifier_both_included(self):
        """Test that both memory content and identifier are included."""
        builder = Prompt_Builder()
        query = "What do I know?"
        context = {
            "memories": [
                {"id": "mem_test_123", "content": "User enjoys reading science fiction", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify both ID and content are present
        assert "mem_test_123" in prompt
        assert "User enjoys reading science fiction" in prompt


class TestPromptBuilderEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_empty_query_string(self):
        """Test prompt construction with empty query string."""
        builder = Prompt_Builder()
        query = ""
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is still constructed
        assert isinstance(prompt, str)
        assert "# System" in prompt
        assert "# User Question" in prompt
        assert "# Instructions" in prompt
    
    def test_query_with_special_characters(self):
        """Test prompt construction with special characters in query."""
        builder = Prompt_Builder()
        query = "What's my email? Is it user@example.com?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify query with special characters is included correctly
        assert query in prompt
    
    def test_memory_with_missing_id_field(self):
        """Test handling of memory without 'id' field."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"content": "Memory without ID", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is constructed without errors
        assert isinstance(prompt, str)
        assert "Memory without ID" in prompt
        # Should handle missing ID gracefully (e.g., "unknown")
        assert "unknown" in prompt.lower() or "Memory 1" in prompt
    
    def test_memory_with_empty_content(self):
        """Test handling of memory with empty content."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"id": "mem_empty", "content": "", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is constructed and ID is still included
        assert isinstance(prompt, str)
        assert "mem_empty" in prompt
    
    def test_large_number_of_memories(self):
        """Test prompt construction with many memories."""
        builder = Prompt_Builder()
        query = "What do I know?"
        memories = [
            {"id": f"mem_{i}", "content": f"Memory content {i}", "metadata": {}}
            for i in range(50)
        ]
        context = {"memories": memories, "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify all memories are included
        assert isinstance(prompt, str)
        assert "mem_0" in prompt
        assert "mem_49" in prompt
        assert "Memory content 0" in prompt
        assert "Memory content 49" in prompt
    
    def test_query_with_newlines(self):
        """Test prompt construction with query containing newlines."""
        builder = Prompt_Builder()
        query = "What is Python?\nAnd what is JavaScript?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Verify query with newlines is included
        assert query in prompt
    
    def test_memory_content_with_special_formatting(self):
        """Test memory content with special formatting characters."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User said: \"I love Python!\"", "metadata": {}},
                {"id": "mem_2", "content": "Code snippet: print('hello')", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify special characters are preserved
        assert "mem_1" in prompt
        assert "mem_2" in prompt
        assert "I love Python!" in prompt
        assert "print('hello')" in prompt


class TestPromptBuilderStructure:
    """Test suite for verifying prompt structure and formatting."""
    
    def test_sections_separated_by_newlines(self):
        """Test that sections are properly separated."""
        builder = Prompt_Builder()
        query = "Test query"
        context = {
            "memories": [
                {"id": "mem_1", "content": "Test memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify sections are separated (should have double newlines between sections)
        assert "\n\n" in prompt
    
    def test_prompt_is_readable_string(self):
        """Test that the constructed prompt is a readable string."""
        builder = Prompt_Builder()
        query = "What is my name?"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User's name is Alice", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Verify prompt is a string with reasonable length
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        
        # Verify it contains expected components
        assert "Luma" in prompt
        assert "Alice" in prompt
        assert "mem_1" in prompt
        assert query in prompt
