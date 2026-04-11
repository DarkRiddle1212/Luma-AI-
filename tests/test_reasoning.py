"""
Unit Tests for Reasoning Engine

Tests the ReasoningEngine's initialization, context building, intent detection,
and message processing methods to ensure correct behavior.
"""

import pytest
from datetime import datetime
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM, LLMInterface


class TestInitialization:
    """Test suite for ReasoningEngine initialization."""
    
    def test_reasoning_engine_initialization_with_stub_llm(self):
        """Test ReasoningEngine initialization with explicit StubLLM."""
        stub_llm = StubLLM()
        engine = ReasoningEngine(llm=stub_llm)
        
        assert engine is not None
        assert isinstance(engine, ReasoningEngine)
        assert engine.llm is stub_llm
        assert isinstance(engine.llm, StubLLM)
    
    def test_reasoning_engine_initialization_without_llm_defaults_to_stub_llm(self):
        """Test ReasoningEngine initialization without LLM defaults to StubLLM."""
        engine = ReasoningEngine()
        
        assert engine is not None
        assert isinstance(engine, ReasoningEngine)
        assert engine.llm is not None
        assert isinstance(engine.llm, StubLLM)
    
    def test_reasoning_engine_initialization_with_custom_llm_implementation(self):
        """Test ReasoningEngine initialization with custom LLM implementation."""
        # Create a custom LLM implementation
        class CustomLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return f"Custom response to: {prompt}"
        
        custom_llm = CustomLLM()
        engine = ReasoningEngine(llm=custom_llm)
        
        assert engine is not None
        assert isinstance(engine, ReasoningEngine)
        assert engine.llm is custom_llm
        assert isinstance(engine.llm, CustomLLM)
        assert isinstance(engine.llm, LLMInterface)
    
    def test_reasoning_engine_stores_llm_instance_correctly(self):
        """Test ReasoningEngine stores LLM instance correctly."""
        # Test with StubLLM
        stub_llm = StubLLM()
        engine1 = ReasoningEngine(llm=stub_llm)
        assert engine1.llm is stub_llm
        
        # Test with custom LLM
        class AnotherCustomLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return "Another custom response"
        
        custom_llm = AnotherCustomLLM()
        engine2 = ReasoningEngine(llm=custom_llm)
        assert engine2.llm is custom_llm
        
        # Test with default (None)
        engine3 = ReasoningEngine()
        assert engine3.llm is not None
        assert isinstance(engine3.llm, StubLLM)
        
        # Verify each engine has its own LLM instance
        assert engine1.llm is not engine2.llm
        assert engine1.llm is not engine3.llm
        assert engine2.llm is not engine3.llm


class TestContextBuilding:
    """Test suite for ReasoningEngine.build_context method."""
    
    def test_build_context_returns_dictionary(self):
        """Test that build_context returns a dictionary."""
        engine = ReasoningEngine()
        result = engine.build_context("Test message")
        
        assert isinstance(result, dict)
    
    def test_build_context_includes_user_message_key(self):
        """Test that build_context includes user_message key."""
        engine = ReasoningEngine()
        test_message = "Hello, Luma!"
        result = engine.build_context(test_message)
        
        assert "user_message" in result
        assert result["user_message"] == test_message
    
    def test_build_context_includes_timestamp_key(self):
        """Test that build_context includes timestamp key."""
        engine = ReasoningEngine()
        result = engine.build_context("Test message")
        
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)
    
    def test_build_context_includes_memory_placeholder_key(self):
        """Test that build_context includes memories key."""
        engine = ReasoningEngine()
        result = engine.build_context("Test message")
        
        assert "memories" in result
        assert isinstance(result["memories"], list)
        assert result["memories"] == []
    
    def test_build_context_includes_system_state_placeholder_key(self):
        """Test that build_context includes system_state_placeholder key."""
        engine = ReasoningEngine()
        result = engine.build_context("Test message")
        
        assert "system_state_placeholder" in result
        assert isinstance(result["system_state_placeholder"], dict)
        assert result["system_state_placeholder"] == {}
    
    def test_build_context_timestamp_is_valid_iso_format(self):
        """Test that build_context timestamp is valid ISO format."""
        engine = ReasoningEngine()
        result = engine.build_context("Test message")
        
        timestamp = result["timestamp"]
        
        # Verify it's a string
        assert isinstance(timestamp, str)
        
        # Verify it can be parsed as ISO format datetime
        try:
            parsed_datetime = datetime.fromisoformat(timestamp)
            assert isinstance(parsed_datetime, datetime)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp}' is not valid ISO format")


class TestIntentDetection:
    """Test suite for ReasoningEngine.detect_intent method."""
    
    def test_detect_intent_store_memory_remember_this(self):
        """Test that detect_intent returns 'store_memory' for 'remember this'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("remember this")
        
        assert result == "store_memory"
    
    def test_detect_intent_store_memory_store_that(self):
        """Test that detect_intent returns 'store_memory' for 'store that'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("store that")
        
        assert result == "store_memory"
    
    def test_detect_intent_retrieve_memory_what_was(self):
        """Test that detect_intent returns 'retrieve_memory' for 'what was'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("what was")
        
        assert result == "retrieve_memory"
    
    def test_detect_intent_retrieve_memory_recall(self):
        """Test that detect_intent returns 'retrieve_memory' for 'recall'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("recall")
        
        assert result == "retrieve_memory"
    
    def test_detect_intent_education_teach_me(self):
        """Test that detect_intent returns 'education' for 'teach me'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("teach me")
        
        assert result == "education"
    
    def test_detect_intent_education_explain(self):
        """Test that detect_intent returns 'education' for 'explain'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("explain")
        
        assert result == "education"
    
    def test_detect_intent_scheduling_schedule(self):
        """Test that detect_intent returns 'scheduling' for 'schedule'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("schedule")
        
        assert result == "scheduling"
    
    def test_detect_intent_scheduling_remind_me(self):
        """Test that detect_intent returns 'scheduling' for 'remind me'."""
        engine = ReasoningEngine()
        result = engine.detect_intent("remind me")
        
        assert result == "scheduling"
    
    def test_detect_intent_general_for_unmatched_messages(self):
        """Test that detect_intent returns 'general' for unmatched messages."""
        engine = ReasoningEngine()
        
        # Test various unmatched messages
        assert engine.detect_intent("hello") == "general"
        assert engine.detect_intent("how are you") == "general"
        assert engine.detect_intent("what can you do") == "general"
        assert engine.detect_intent("random message") == "general"
    
    def test_detect_intent_is_case_insensitive(self):
        """Test that detect_intent is case-insensitive."""
        engine = ReasoningEngine()
        
        # Test uppercase variations
        assert engine.detect_intent("REMEMBER THIS") == "store_memory"
        assert engine.detect_intent("STORE THAT") == "store_memory"
        assert engine.detect_intent("WHAT WAS") == "retrieve_memory"
        assert engine.detect_intent("RECALL") == "retrieve_memory"
        assert engine.detect_intent("TEACH ME") == "education"
        assert engine.detect_intent("EXPLAIN") == "education"
        assert engine.detect_intent("SCHEDULE") == "scheduling"
        assert engine.detect_intent("REMIND ME") == "scheduling"
        
        # Test mixed case variations
        assert engine.detect_intent("ReMeMbEr ThIs") == "store_memory"
        assert engine.detect_intent("TeAcH mE") == "education"


class TestMessageProcessing:
    """Test suite for ReasoningEngine.process_message method."""
    
    def test_process_message_with_valid_message_returns_dict(self):
        """Test that process_message with valid message returns a dictionary."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert isinstance(result, dict)
    
    def test_process_message_response_has_response_key(self):
        """Test that process_message response has 'response' key."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert "response" in result
        assert isinstance(result["response"], str)
    
    def test_process_message_response_has_intent_key(self):
        """Test that process_message response has 'intent' key."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert "intent" in result
        assert isinstance(result["intent"], str)
    
    def test_process_message_response_has_metadata_key(self):
        """Test that process_message response has 'metadata' key."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
    
    def test_process_message_with_empty_string_returns_invalid_intent(self):
        """Test that process_message with empty string returns invalid intent."""
        engine = ReasoningEngine()
        result = engine.process_message("")
        
        assert result["intent"] == "invalid"
        assert "response" in result
        assert "No message provided" in result["response"]
    
    def test_process_message_with_none_returns_invalid_intent(self):
        """Test that process_message with None returns invalid intent."""
        engine = ReasoningEngine()
        result = engine.process_message(None)
        
        assert result["intent"] == "invalid"
        assert "response" in result
        assert "No message provided" in result["response"]
    
    def test_process_message_with_whitespace_only_returns_invalid_intent(self):
        """Test that process_message with whitespace-only returns invalid intent."""
        engine = ReasoningEngine()
        
        # Test various whitespace-only inputs
        result1 = engine.process_message("   ")
        assert result1["intent"] == "invalid"
        
        result2 = engine.process_message("\t\t")
        assert result2["intent"] == "invalid"
        
        result3 = engine.process_message("\n\n")
        assert result3["intent"] == "invalid"
        
        result4 = engine.process_message("  \t\n  ")
        assert result4["intent"] == "invalid"
    
    def test_process_message_handles_llm_exception_gracefully(self):
        """Test that process_message handles LLM exception gracefully."""
        # Create a mock LLM that raises an exception
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("LLM service unavailable")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test message")
        
        # Should not raise exception, should return error response
        assert isinstance(result, dict)
        assert "response" in result
        assert "intent" in result
        assert "metadata" in result
    
    def test_process_message_returns_error_intent_on_exception(self):
        """Test that process_message returns error intent on exception."""
        # Create a mock LLM that raises an exception
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("LLM service unavailable")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test message")
        
        assert result["intent"] == "error"
        assert "error occurred" in result["response"].lower()
    
    def test_process_message_metadata_includes_context_keys(self):
        """Test that process_message metadata includes context_keys."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert "metadata" in result
        assert "context_keys" in result["metadata"]
        assert isinstance(result["metadata"]["context_keys"], list)
        
        # Verify expected context keys are present
        context_keys = result["metadata"]["context_keys"]
        assert "user_message" in context_keys
        assert "timestamp" in context_keys
        assert "intent" in context_keys
    
    def test_process_message_metadata_includes_timestamp(self):
        """Test that process_message metadata includes timestamp."""
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        assert "metadata" in result
        assert "timestamp" in result["metadata"]
        assert isinstance(result["metadata"]["timestamp"], str)
        
        # Verify timestamp is valid ISO format
        timestamp = result["metadata"]["timestamp"]
        try:
            parsed_datetime = datetime.fromisoformat(timestamp)
            assert isinstance(parsed_datetime, datetime)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp}' is not valid ISO format")
