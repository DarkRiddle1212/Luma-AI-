"""
Unit tests for LLM Interface module.

Tests the abstract LLMInterface and StubLLM implementation to ensure
correct behavior and interface compliance.
"""

import pytest
from luma.core.llm_interface import LLMInterface, StubLLM


class TestStubLLMInstantiation:
    """Test StubLLM can be instantiated correctly."""
    
    def test_stub_llm_instantiation(self):
        """Test that StubLLM can be instantiated without errors."""
        llm = StubLLM()
        assert llm is not None
        assert isinstance(llm, StubLLM)
        assert isinstance(llm, LLMInterface)


class TestStubLLMGenerateResponse:
    """Test StubLLM.generate_response method behavior."""
    
    def test_generate_response_returns_string(self):
        """Test that generate_response returns a string."""
        llm = StubLLM()
        prompt = "Test prompt"
        context = {"intent": "general", "user_message": "Hello"}
        
        response = llm.generate_response(prompt, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_response_includes_prompt_text(self):
        """Test that the response includes the prompt text."""
        llm = StubLLM()
        prompt = "Teach me Python loops"
        context = {"intent": "education"}
        
        response = llm.generate_response(prompt, context)
        
        assert prompt in response
        assert "Prompt:" in response
    
    def test_response_includes_context_keys(self):
        """Test that the response includes context keys."""
        llm = StubLLM()
        prompt = "Test"
        context = {
            "intent": "general",
            "user_message": "Hello",
            "timestamp": "2024-01-15T10:30:00"
        }
        
        response = llm.generate_response(prompt, context)
        
        assert "Context Keys:" in response
        assert "intent" in response
        assert "user_message" in response
        assert "timestamp" in response
    
    def test_response_includes_intent_from_context(self):
        """Test that the response includes the intent from context."""
        llm = StubLLM()
        prompt = "Remember this"
        context = {"intent": "store_memory", "user_message": "Remember this"}
        
        response = llm.generate_response(prompt, context)
        
        assert "Intent: store_memory" in response
    
    def test_response_with_unknown_intent(self):
        """Test that response handles missing intent gracefully."""
        llm = StubLLM()
        prompt = "Test"
        context = {"user_message": "Test"}
        
        response = llm.generate_response(prompt, context)
        
        assert "Intent: unknown" in response


class TestLLMInterfaceAbstract:
    """Test that LLMInterface cannot be instantiated directly."""
    
    def test_llm_interface_cannot_be_instantiated(self):
        """Test that abstract LLMInterface cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            LLMInterface()
        
        # Verify the error message indicates it's abstract
        assert "abstract" in str(exc_info.value).lower() or \
               "Can't instantiate" in str(exc_info.value)
