"""
Tests for ReasoningEngine.process_message method

Tests the message processing orchestration including:
- Input validation
- Context building
- Intent detection
- LLM invocation
- Structured response generation
- Error handling
"""

import pytest
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM


class TestProcessMessage:
    """Test the process_message orchestration method."""
    
    def test_process_message_valid_input(self):
        """Test process_message with valid input returns structured response."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        # Verify structure
        assert isinstance(result, dict)
        assert "response" in result
        assert "intent" in result
        assert "metadata" in result
        
        # Verify types
        assert isinstance(result["response"], str)
        assert isinstance(result["intent"], str)
        assert isinstance(result["metadata"], dict)
        
        # Verify intent is correct
        assert result["intent"] == "education"
    
    def test_process_message_empty_string(self):
        """Test process_message with empty string returns invalid response."""
        engine = ReasoningEngine()
        result = engine.process_message("")
        
        assert result["intent"] == "invalid"
        assert "No message provided" in result["response"]
        assert result["metadata"]["context_keys"] == []
    
    def test_process_message_whitespace_only(self):
        """Test process_message with whitespace-only string returns invalid response."""
        engine = ReasoningEngine()
        result = engine.process_message("   ")
        
        assert result["intent"] == "invalid"
        assert "No message provided" in result["response"]
    
    def test_process_message_metadata_structure(self):
        """Test process_message metadata contains required keys."""
        engine = ReasoningEngine()
        result = engine.process_message("Hello")
        
        metadata = result["metadata"]
        assert "context_keys" in metadata
        assert "timestamp" in metadata
        assert isinstance(metadata["context_keys"], list)
        assert isinstance(metadata["timestamp"], str)
    
    def test_process_message_context_keys(self):
        """Test process_message includes correct context keys in metadata."""
        engine = ReasoningEngine()
        result = engine.process_message("Remember this")
        
        context_keys = result["metadata"]["context_keys"]
        assert "user_message" in context_keys
        assert "timestamp" in context_keys
        assert "intent" in context_keys
        assert "memory_placeholder" in context_keys
        assert "system_state_placeholder" in context_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
