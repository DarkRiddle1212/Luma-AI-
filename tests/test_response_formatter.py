"""
Unit tests for the Response_Formatter class.

Tests verify that the Response_Formatter correctly extracts answer text,
used memory identifiers, and confidence scores from LLM output, and handles
unparseable output gracefully.
"""

import pytest
from luma.core.reasoning.response_formatter import Response_Formatter
from luma.core.reasoning.schemas import Reasoning_Result


class TestResponseFormatter:
    """Test suite for Response_Formatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = Response_Formatter()
    
    def test_format_response_with_all_fields(self):
        """Test formatting response with answer, used memories, and confidence."""
        raw_output = """Answer: Python is a great programming language.
Used Memories: mem_1, mem_2, mem_3
Confidence: 0.95"""
        
        result = self.formatter.format_response(raw_output)
        
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "Python is a great programming language."
        assert result.used_memories == ["mem_1", "mem_2", "mem_3"]
        assert result.confidence == 0.95
    
    def test_format_response_without_confidence(self):
        """Test formatting response without confidence score."""
        raw_output = """Answer: JavaScript is widely used.
Used Memories: mem_4, mem_5"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "JavaScript is widely used."
        assert result.used_memories == ["mem_4", "mem_5"]
        assert result.confidence is None
    
    def test_format_response_without_used_memories(self):
        """Test formatting response without used memories."""
        raw_output = """Answer: This is a general answer.
Confidence: 0.8"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "This is a general answer."
        assert result.used_memories == []
        assert result.confidence == 0.8
    
    def test_format_response_answer_only(self):
        """Test formatting response with only answer text."""
        raw_output = "Answer: Simple answer text."
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "Simple answer text."
        assert result.used_memories == []
        assert result.confidence is None
    
    def test_format_response_multiline_answer(self):
        """Test formatting response with multiline answer."""
        raw_output = """Answer: This is a longer answer
that spans multiple lines
and contains detailed information.
Used Memories: mem_1
Confidence: 0.9"""
        
        result = self.formatter.format_response(raw_output)
        
        assert "This is a longer answer" in result.answer
        assert "multiple lines" in result.answer
        assert result.used_memories == ["mem_1"]
        assert result.confidence == 0.9
    
    def test_format_response_unparseable_output(self):
        """Test graceful handling of unparseable output."""
        raw_output = "This is just random text without any structure."
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "This is just random text without any structure."
        assert result.used_memories == []
        assert result.confidence is None
    
    def test_format_response_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        raw_output = """answer: Case insensitive test.
used memories: mem_1, mem_2
confidence: 0.85"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "Case insensitive test."
        assert result.used_memories == ["mem_1", "mem_2"]
        assert result.confidence == 0.85
    
    def test_format_response_with_response_keyword(self):
        """Test parsing with 'Response' instead of 'Answer'."""
        raw_output = """Response: Using alternative keyword.
Used Memories: mem_10"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.answer == "Using alternative keyword."
        assert result.used_memories == ["mem_10"]
    
    def test_format_response_memories_with_brackets(self):
        """Test parsing memories with bracket notation."""
        raw_output = """Answer: Test with brackets.
Used Memories: [mem_1, mem_2, mem_3]"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.used_memories == ["mem_1", "mem_2", "mem_3"]
    
    def test_format_response_single_memory(self):
        """Test parsing with a single memory."""
        raw_output = """Answer: Single memory test.
Used Memories: mem_1"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.used_memories == ["mem_1"]
    
    def test_format_response_confidence_edge_cases(self):
        """Test confidence score edge cases (0.0 and 1.0)."""
        # Test confidence 0.0
        raw_output1 = """Answer: Low confidence.
Confidence: 0.0"""
        result1 = self.formatter.format_response(raw_output1)
        assert result1.confidence == 0.0
        
        # Test confidence 1.0
        raw_output2 = """Answer: High confidence.
Confidence: 1.0"""
        result2 = self.formatter.format_response(raw_output2)
        assert result2.confidence == 1.0
    
    def test_format_response_invalid_confidence(self):
        """Test that invalid confidence values are ignored."""
        # Confidence > 1.0
        raw_output1 = """Answer: Invalid confidence.
Confidence: 1.5"""
        result1 = self.formatter.format_response(raw_output1)
        assert result1.confidence is None
        
        # Negative confidence
        raw_output2 = """Answer: Negative confidence.
Confidence: -0.5"""
        result2 = self.formatter.format_response(raw_output2)
        assert result2.confidence is None
    
    def test_format_response_memories_with_whitespace(self):
        """Test parsing memories with extra whitespace."""
        raw_output = """Answer: Whitespace test.
Used Memories:  mem_1 ,  mem_2  , mem_3  """
        
        result = self.formatter.format_response(raw_output)
        
        assert result.used_memories == ["mem_1", "mem_2", "mem_3"]
    
    def test_format_response_empty_memories_list(self):
        """Test parsing with empty or 'none' memories."""
        raw_output = """Answer: No memories used.
Used Memories: none"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.used_memories == []
    
    def test_format_response_alternative_memory_keyword(self):
        """Test parsing with 'Memories Used' instead of 'Used Memories'."""
        raw_output = """Answer: Alternative keyword.
Memories Used: mem_1, mem_2"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.used_memories == ["mem_1", "mem_2"]
    
    def test_format_response_confidence_score_keyword(self):
        """Test parsing with 'Confidence Score' instead of 'Confidence'."""
        raw_output = """Answer: Alternative confidence keyword.
Confidence Score: 0.75"""
        
        result = self.formatter.format_response(raw_output)
        
        assert result.confidence == 0.75
    
    def test_format_response_empty_string(self):
        """Test handling of empty string input."""
        result = self.formatter.format_response("")
        
        assert result.answer == ""
        assert result.used_memories == []
        assert result.confidence is None
    
    def test_format_response_only_whitespace(self):
        """Test handling of whitespace-only input."""
        result = self.formatter.format_response("   \n\t  ")
        
        assert result.answer == ""
        assert result.used_memories == []
        assert result.confidence is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
