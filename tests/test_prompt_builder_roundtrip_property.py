"""
Property-Based Tests for Prompt_Builder Round-Trip Consistency

**Validates: Requirements 7.2, 7.3, 7.4, 7.5**

This test verifies that prompt construction is consistent and reversible,
ensuring that query information and memory identifiers are preserved through 
the prompt building process.

Property 1: Query preservation in prompt

For all valid query and context pairs, the Prompt_Builder SHALL produce a prompt
containing the original query text, and the prompt SHALL be parseable to extract
the original query.

Property 2: Memory identifier preservation

For all valid query and context pairs with memories, the Prompt_Builder SHALL 
produce a prompt containing all memory identifiers from the context, and the 
prompt SHALL be parseable to extract all memory identifiers.
"""

import pytest
from hypothesis import given, strategies as st, settings
from luma.core.reasoning.prompt_builder import Prompt_Builder
import re


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def query_strategy(draw):
    """
    Generate random query strings.
    
    Queries can contain:
    - Regular text with spaces
    - Special characters (punctuation, symbols)
    - Newlines and whitespace
    - Unicode characters
    """
    # Generate queries of varying lengths and complexity
    query_type = draw(st.sampled_from([
        "simple",
        "with_punctuation",
        "with_newlines",
        "with_special_chars",
        "unicode"
    ]))
    
    if query_type == "simple":
        # Simple alphanumeric queries with spaces
        return draw(st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
            min_size=1,
            max_size=200
        ))
    elif query_type == "with_punctuation":
        # Queries with punctuation
        return draw(st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd", "Po"),
                whitelist_characters=" ?!.,;:'"
            ),
            min_size=1,
            max_size=200
        ))
    elif query_type == "with_newlines":
        # Queries with newlines
        base_text = draw(st.text(min_size=1, max_size=100))
        lines = draw(st.lists(st.text(min_size=0, max_size=5), min_size=1, max_size=5))
        return "\n".join([base_text] + lines)
    elif query_type == "with_special_chars":
        # Queries with special characters like @, #, $, etc.
        return draw(st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters=" @#$%&*()[]{}+-=_"
            ),
            min_size=1,
            max_size=200
        ))
    else:  # unicode
        # Unicode text
        return draw(st.text(min_size=1, max_size=200))


@st.composite
def memory_strategy(draw):
    """Generate a single memory object with id, content, and metadata."""
    memory_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
        min_size=1,
        max_size=50
    ))
    
    content = draw(st.text(min_size=0, max_size=5))
    
    # Generate optional metadata
    metadata = draw(st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans()
        ),
        max_size=5
    ))
    
    return {
        "id": memory_id,
        "content": content,
        "metadata": metadata
    }


@st.composite
def context_strategy(draw):
    """
    Generate random context dictionaries.
    
    Context can contain:
    - Empty memories list
    - One or more memories
    - Optional metadata
    """
    memories = draw(st.lists(memory_strategy(), min_size=0, max_size=10))
    
    metadata = draw(st.dictionaries(
        st.text(min_size=1, max_size=5),
        st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False)
        ),
        max_size=5
    ))
    
    return {
        "memories": memories,
        "metadata": metadata
    }


# ============================================================================
# Helper Functions for Query Extraction
# ============================================================================

def extract_query_from_prompt(prompt: str) -> str:
    """
    Extract the original query from a constructed prompt.
    
    The prompt structure has a "# User Question" section followed by the query.
    This function parses the prompt to extract the query text.
    
    Args:
        prompt (str): The constructed prompt string.
    
    Returns:
        str: The extracted query text.
    
    Raises:
        ValueError: If the prompt cannot be parsed or query section is missing.
    """
    # Look for the User Question section
    # Pattern: "# User Question\n\n" followed by content until "\n\n#" or end of string
    # Use non-greedy match and explicitly look for the next section marker
    user_question_pattern = r"# User Question\n\n(.*?)(?=\n\n#|\Z)"
    match = re.search(user_question_pattern, prompt, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find User Question section in prompt")
    
    # Don't strip - preserve exact whitespace including trailing newlines
    extracted_query = match.group(1)
    return extracted_query


def extract_memory_identifiers_from_prompt(prompt: str) -> list[str]:
    """
    Extract all memory identifiers from a constructed prompt.
    
    The prompt structure includes memory identifiers in the format:
    "**Memory {i} (ID: {memory_id})**"
    
    This function parses the prompt to extract all memory IDs.
    
    Args:
        prompt (str): The constructed prompt string.
    
    Returns:
        list[str]: List of extracted memory identifiers in order of appearance.
    
    Raises:
        ValueError: If the prompt cannot be parsed.
    """
    # Pattern to match: **Memory {number} (ID: {id})**
    # The ID can contain alphanumeric characters, underscores, and hyphens
    memory_id_pattern = r"\*\*Memory \d+ \(ID: ([^\)]+)\)\*\*"
    matches = re.findall(memory_id_pattern, prompt)
    
    return matches


# ============================================================================
# Property Tests
# ============================================================================

class TestPromptBuilderRoundTripProperty:
    """
    Property 1: Query preservation in prompt
    
    **Validates: Requirements 7.2, 7.4**
    
    These tests verify that queries are preserved through the prompt building
    process and can be extracted back from the constructed prompt.
    """
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_1_query_preservation_in_prompt(self, query, context):
        """
        Property: Query text is preserved in constructed prompt
        
        **Validates: Requirement 7.2**
        
        For all valid query and context pairs, the Prompt_Builder SHALL produce
        a prompt containing the original query text exactly as provided.
        
        This property ensures that no information is lost during prompt construction.
        """
        builder = Prompt_Builder()
        
        # Build the prompt
        prompt = builder.build_prompt(query, context)
        
        # Verify the prompt is a non-empty string
        assert isinstance(prompt, str), "Prompt must be a string"
        assert len(prompt) > 0, "Prompt must not be empty"
        
        # Verify the original query text appears in the prompt
        assert query in prompt, (
            f"Original query must be present in the constructed prompt.\n"
            f"Query: {repr(query)}\n"
            f"Prompt length: {len(prompt)}"
        )
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_1_query_extractable_from_prompt(self, query, context):
        """
        Property: Query is parseable and extractable from prompt
        
        **Validates: Requirement 7.4**
        
        When a prompt is constructed from a query and context, the prompt SHALL
        be parseable to extract the original query text.
        
        This property ensures that prompts maintain a consistent, parseable
        structure that allows for round-trip conversion.
        """
        builder = Prompt_Builder()
        
        # Build the prompt
        prompt = builder.build_prompt(query, context)
        
        # Verify we can extract the query from the prompt
        try:
            extracted_query = extract_query_from_prompt(prompt)
        except ValueError as e:
            pytest.fail(f"Failed to parse prompt and extract query: {e}")
        
        # Verify the extracted query matches the original
        assert extracted_query == query, (
            f"Extracted query must match the original query.\n"
            f"Original: {repr(query)}\n"
            f"Extracted: {repr(extracted_query)}"
        )
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_1_round_trip_consistency(self, query, context):
        """
        Property: Round-trip query preservation (build -> extract -> compare)
        
        **Validates: Requirements 7.2, 7.4**
        
        This test combines both preservation properties: the query must be
        present in the prompt AND extractable in its original form.
        
        This is the complete round-trip test: query -> prompt -> extracted query,
        where extracted query == original query.
        """
        builder = Prompt_Builder()
        
        # Step 1: Build prompt from query and context
        prompt = builder.build_prompt(query, context)
        
        # Step 2: Verify query is present (preservation)
        assert query in prompt, "Query must be preserved in prompt"
        
        # Step 3: Extract query from prompt (parseability)
        extracted_query = extract_query_from_prompt(prompt)
        
        # Step 4: Verify round-trip consistency
        assert extracted_query == query, (
            f"Round-trip must preserve query exactly.\n"
            f"Original: {repr(query)}\n"
            f"After round-trip: {repr(extracted_query)}"
        )


class TestMemoryIdentifierPreservationProperty:
    """
    Property 2: Memory identifier preservation
    
    **Validates: Requirements 7.3, 7.5**
    
    These tests verify that memory identifiers are preserved through the prompt 
    building process and can be extracted back from the constructed prompt.
    """
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_2_memory_identifiers_in_prompt(self, query, context):
        """
        Property: All memory identifiers appear in constructed prompt
        
        **Validates: Requirement 7.3**
        
        For all valid query and context pairs with memories, the Prompt_Builder 
        SHALL produce a prompt containing all memory identifiers from the context.
        
        This property ensures that no memory identifiers are lost during prompt 
        construction.
        """
        builder = Prompt_Builder()
        
        # Build the prompt
        prompt = builder.build_prompt(query, context)
        
        # Verify the prompt is a non-empty string
        assert isinstance(prompt, str), "Prompt must be a string"
        assert len(prompt) > 0, "Prompt must not be empty"
        
        # Extract expected memory IDs from context
        memories = context.get("memories", [])
        expected_ids = [memory.get("id", "unknown") for memory in memories]
        
        # Verify all memory identifiers appear in the prompt
        for memory_id in expected_ids:
            assert memory_id in prompt, (
                f"Memory identifier '{memory_id}' must be present in the constructed prompt.\n"
                f"Expected IDs: {expected_ids}\n"
                f"Prompt length: {len(prompt)}"
            )
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_2_memory_identifiers_extractable_from_prompt(self, query, context):
        """
        Property: Memory identifiers are parseable and extractable from prompt
        
        **Validates: Requirement 7.5**
        
        When a prompt is constructed from a query and context with memories, 
        the prompt SHALL be parseable to extract all memory identifiers.
        
        This property ensures that prompts maintain a consistent, parseable
        structure that allows for memory identifier extraction.
        """
        builder = Prompt_Builder()
        
        # Build the prompt
        prompt = builder.build_prompt(query, context)
        
        # Extract expected memory IDs from context
        memories = context.get("memories", [])
        expected_ids = [memory.get("id", "unknown") for memory in memories]
        
        # If there are no memories, there should be no memory identifiers in the prompt
        if not memories:
            extracted_ids = extract_memory_identifiers_from_prompt(prompt)
            assert extracted_ids == [], (
                f"Prompt with no memories should have no extractable memory identifiers.\n"
                f"Extracted IDs: {extracted_ids}"
            )
            return
        
        # Verify we can extract memory identifiers from the prompt
        try:
            extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        except Exception as e:
            pytest.fail(f"Failed to parse prompt and extract memory identifiers: {e}")
        
        # Verify the extracted IDs match the expected IDs
        assert extracted_ids == expected_ids, (
            f"Extracted memory identifiers must match the original identifiers.\n"
            f"Expected: {expected_ids}\n"
            f"Extracted: {extracted_ids}"
        )
    
    @pytest.mark.property_test
    @given(query=query_strategy(), context=context_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_2_round_trip_memory_identifier_consistency(self, query, context):
        """
        Property: Round-trip memory identifier preservation (build -> extract -> compare)
        
        **Validates: Requirements 7.3, 7.5**
        
        This test combines both memory identifier preservation properties: 
        all memory identifiers must be present in the prompt AND extractable 
        in their original form and order.
        
        This is the complete round-trip test: context -> prompt -> extracted IDs,
        where extracted IDs == original IDs.
        """
        builder = Prompt_Builder()
        
        # Step 1: Build prompt from query and context
        prompt = builder.build_prompt(query, context)
        
        # Step 2: Get expected memory IDs from context
        memories = context.get("memories", [])
        expected_ids = [memory.get("id", "unknown") for memory in memories]
        
        # Step 3: Verify all memory IDs are present (preservation)
        for memory_id in expected_ids:
            assert memory_id in prompt, f"Memory ID '{memory_id}' must be preserved in prompt"
        
        # Step 4: Extract memory IDs from prompt (parseability)
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        
        # Step 5: Verify round-trip consistency
        assert extracted_ids == expected_ids, (
            f"Round-trip must preserve all memory identifiers in order.\n"
            f"Original: {expected_ids}\n"
            f"After round-trip: {extracted_ids}"
        )


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestPromptBuilderRoundTripEdgeCases:
    """
    Edge case tests for query preservation.
    
    These tests verify that the round-trip property holds even for
    challenging edge cases.
    """
    
    @pytest.mark.property_test
    def test_empty_query_preservation(self):
        """Test that empty queries are preserved (edge case)."""
        builder = Prompt_Builder()
        query = ""
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Empty query should still be in the prompt
        assert "# User Question" in prompt
        
        # Extract and verify
        extracted = extract_query_from_prompt(prompt)
        assert extracted == query
    
    @pytest.mark.property_test
    def test_query_with_prompt_section_markers(self):
        """Test queries that contain section markers like '# System'."""
        builder = Prompt_Builder()
        query = "What is # System and # Context in prompts?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Query should be preserved even with section-like markers
        assert query in prompt
        
        # Extract and verify
        extracted = extract_query_from_prompt(prompt)
        assert extracted == query
    
    @pytest.mark.property_test
    def test_query_with_multiple_newlines(self):
        """Test queries with multiple consecutive newlines."""
        builder = Prompt_Builder()
        query = "Line 1\n\n\nLine 2\n\nLine 3"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Query should be preserved with all newlines
        assert query in prompt
        
        # Extract and verify
        extracted = extract_query_from_prompt(prompt)
        assert extracted == query
    
    @pytest.mark.property_test
    def test_very_long_query_preservation(self):
        """Test that very long queries are preserved correctly."""
        builder = Prompt_Builder()
        query = "What is the meaning of life? " * 100  # Very long query
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # Long query should be fully preserved
        assert query in prompt
        
        # Extract and verify
        extracted = extract_query_from_prompt(prompt)
        assert extracted == query


class TestMemoryIdentifierPreservationEdgeCases:
    """
    Edge case tests for memory identifier preservation.
    
    These tests verify that the memory identifier preservation property holds 
    even for challenging edge cases.
    """
    
    @pytest.mark.property_test
    def test_empty_memories_list(self):
        """Test that empty memories list results in no memory identifiers."""
        builder = Prompt_Builder()
        query = "What do I like?"
        context = {"memories": [], "metadata": {}}
        
        prompt = builder.build_prompt(query, context)
        
        # No memories means no Context section
        assert "# Context" not in prompt
        
        # Extract and verify no memory identifiers
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == []
    
    @pytest.mark.property_test
    def test_single_memory_identifier_preservation(self):
        """Test that a single memory identifier is preserved correctly."""
        builder = Prompt_Builder()
        query = "What do I like?"
        context = {
            "memories": [
                {"id": "mem_123", "content": "User likes Python", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Memory ID should be in the prompt
        assert "mem_123" in prompt
        
        # Extract and verify
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == ["mem_123"]
    
    @pytest.mark.property_test
    def test_multiple_memory_identifiers_preservation(self):
        """Test that multiple memory identifiers are preserved in order."""
        builder = Prompt_Builder()
        query = "Tell me about my preferences"
        context = {
            "memories": [
                {"id": "mem_001", "content": "User likes Python", "metadata": {}},
                {"id": "mem_002", "content": "User prefers dark mode", "metadata": {}},
                {"id": "mem_003", "content": "User works remotely", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # All memory IDs should be in the prompt
        assert "mem_001" in prompt
        assert "mem_002" in prompt
        assert "mem_003" in prompt
        
        # Extract and verify order is preserved
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == ["mem_001", "mem_002", "mem_003"]
    
    @pytest.mark.property_test
    def test_memory_identifiers_with_special_characters(self):
        """Test memory identifiers containing special characters like underscores and hyphens."""
        builder = Prompt_Builder()
        query = "What are my tasks?"
        context = {
            "memories": [
                {"id": "mem_user_123", "content": "Task 1", "metadata": {}},
                {"id": "mem-project-456", "content": "Task 2", "metadata": {}},
                {"id": "mem_2024_01_15", "content": "Task 3", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # All memory IDs with special chars should be preserved
        assert "mem_user_123" in prompt
        assert "mem-project-456" in prompt
        assert "mem_2024_01_15" in prompt
        
        # Extract and verify
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == ["mem_user_123", "mem-project-456", "mem_2024_01_15"]
    
    @pytest.mark.property_test
    def test_memory_identifiers_with_numeric_ids(self):
        """Test memory identifiers that are purely numeric."""
        builder = Prompt_Builder()
        query = "Show me my notes"
        context = {
            "memories": [
                {"id": "12345", "content": "Note 1", "metadata": {}},
                {"id": "67890", "content": "Note 2", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Numeric IDs should be preserved
        assert "12345" in prompt
        assert "67890" in prompt
        
        # Extract and verify
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == ["12345", "67890"]
    
    @pytest.mark.property_test
    def test_memory_identifiers_with_long_ids(self):
        """Test memory identifiers that are very long."""
        builder = Prompt_Builder()
        query = "What do you know?"
        long_id = "mem_" + "x" * 100  # Very long ID
        context = {
            "memories": [
                {"id": long_id, "content": "Some content", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Long ID should be fully preserved
        assert long_id in prompt
        
        # Extract and verify
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == [long_id]
    
    @pytest.mark.property_test
    def test_memory_with_missing_id_field(self):
        """Test handling of memories with missing 'id' field (defaults to 'unknown')."""
        builder = Prompt_Builder()
        query = "What do you remember?"
        context = {
            "memories": [
                {"content": "Memory without ID", "metadata": {}}
            ],
            "metadata": {}
        }
        
        prompt = builder.build_prompt(query, context)
        
        # Should default to "unknown"
        assert "unknown" in prompt
        
        # Extract and verify
        extracted_ids = extract_memory_identifiers_from_prompt(prompt)
        assert extracted_ids == ["unknown"]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'property_test'])
