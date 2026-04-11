"""
Tests for Reasoning Engine Orchestration

Tests the enhanced reasoning engine's ability to:
- Handle messages
- Build context
- Route intent
- Interface with stub LLM
- Return structured responses
"""

import pytest
from luma.core.reasoning import ReasoningEngine, Intent
from luma.core.llm_interface import StubLLM


class TestStubLLM:
    """Test the stub LLM implementation."""
    
    def test_stub_llm_initialization(self):
        """Test that stub LLM initializes correctly."""
        llm = StubLLM()
        assert llm is not None
    
    def test_stub_llm_memory_response(self):
        """Test stub LLM generates appropriate response for memory queries."""
        llm = StubLLM()
        response = llm.generate_response("store this memory", {"key": "value"})
        assert "memory" in response.lower() or "memories" in response.lower()
        assert isinstance(response, str)
    
    def test_stub_llm_schedule_response(self):
        """Test stub LLM generates appropriate response for scheduling queries."""
        llm = StubLLM()
        response = llm.generate_response("schedule a task", {})
        assert "schedule" in response.lower() or "task" in response.lower()
    
    def test_stub_llm_system_response(self):
        """Test stub LLM generates appropriate response for system queries."""
        llm = StubLLM()
        response = llm.generate_response("system status", {})
        assert "system" in response.lower()


class TestReasoningEngineOrchestration:
    """Test the reasoning engine's orchestration capabilities."""
    
    def test_reasoning_engine_initialization(self):
        """Test that reasoning engine initializes with stub LLM."""
        engine = ReasoningEngine()
        assert engine is not None
        assert engine.llm is not None
        assert isinstance(engine.llm, StubLLM)
    
    def test_handle_message_structure(self):
        """Test that handle_message returns properly structured response."""
        engine = ReasoningEngine()
        result = engine.handle_message("Hello, Luma!")
        
        # Verify structure
        assert isinstance(result, dict)
        assert "intent" in result
        assert "response" in result
        assert "context" in result
        assert "metadata" in result
        
        # Verify types
        assert isinstance(result["intent"], str)
        assert isinstance(result["response"], str)
        assert isinstance(result["context"], dict)
        assert isinstance(result["metadata"], dict)
    
    def test_handle_message_with_user_context(self):
        """Test handle_message with additional user context."""
        engine = ReasoningEngine()
        user_context = {"user_id": "123", "session": "abc"}
        result = engine.handle_message("Test message", user_context)
        
        assert result["context"]["user_context"] == user_context
    
    def test_build_context(self):
        """Test context building from message and user context."""
        engine = ReasoningEngine()
        message = "Test message"
        user_context = {"key": "value"}
        
        context = engine.build_context(message, user_context)
        
        # Verify context structure
        assert isinstance(context, dict)
        assert context["message"] == message
        assert context["message_length"] == len(message)
        assert context["user_context"] == user_context
        assert "system_state" in context
        assert "timestamp" in context
        assert "relevant_memories" in context
    
    def test_route_intent_store_memory(self):
        """Test intent routing for memory storage."""
        engine = ReasoningEngine()
        
        test_messages = [
            "remember this",
            "store this memory",
            "save this information"
        ]
        
        for message in test_messages:
            intent = engine.route_intent(message, {})
            assert intent == Intent.STORE_MEMORY
    
    def test_route_intent_retrieve_memory(self):
        """Test intent routing for memory retrieval."""
        engine = ReasoningEngine()
        
        test_messages = [
            "recall that memory",
            "what did I say earlier",
            "retrieve my notes"
        ]
        
        for message in test_messages:
            intent = engine.route_intent(message, {})
            assert intent == Intent.RETRIEVE_MEMORY
    
    def test_route_intent_schedule_task(self):
        """Test intent routing for task scheduling."""
        engine = ReasoningEngine()
        
        test_messages = [
            "schedule a meeting",
            "remind me tomorrow",
            "add this to my todo list"
        ]
        
        for message in test_messages:
            intent = engine.route_intent(message, {})
            assert intent == Intent.SCHEDULE_TASK
    
    def test_route_intent_system_info(self):
        """Test intent routing for system information."""
        engine = ReasoningEngine()
        
        test_messages = [
            "system status",
            "check health",
            "monitor performance"
        ]
        
        for message in test_messages:
            intent = engine.route_intent(message, {})
            assert intent == Intent.SYSTEM_INFO
    
    def test_route_intent_general_query(self):
        """Test intent routing for general queries."""
        engine = ReasoningEngine()
        
        test_messages = [
            "help me",
            "what can you do",
            "show me your capabilities"
        ]
        
        for message in test_messages:
            intent = engine.route_intent(message, {})
            assert intent == Intent.GENERAL_QUERY
    
    def test_route_intent_unknown(self):
        """Test intent routing for unknown intents."""
        engine = ReasoningEngine()
        
        message = "xyzabc random gibberish"
        intent = engine.route_intent(message, {})
        assert intent == Intent.UNKNOWN
    
    def test_handle_message_metadata(self):
        """Test that handle_message includes proper metadata."""
        engine = ReasoningEngine()
        message = "Test message for metadata"
        result = engine.handle_message(message)
        
        metadata = result["metadata"]
        assert "message_length" in metadata
        assert metadata["message_length"] == len(message)
        assert "context_keys" in metadata
        assert "confidence" in metadata
        assert isinstance(metadata["confidence"], (int, float))
    
    def test_process_method_still_works(self):
        """Test that original process method still functions."""
        engine = ReasoningEngine()
        input_data = {"key1": "value1", "key2": "value2"}
        result = engine.process(input_data)
        
        assert result["status"] == "processed"
        assert set(result["input_keys"]) == set(input_data.keys())
    
    def test_analyze_context_method_still_works(self):
        """Test that original analyze_context method still functions."""
        engine = ReasoningEngine()
        context = {"key1": "value1", "key2": "value2"}
        analysis = engine.analyze_context(context)
        
        assert analysis["context_size"] == len(context)
        assert "insights" in analysis
        assert "confidence" in analysis


class TestIntentEnum:
    """Test the Intent enumeration."""
    
    def test_intent_values(self):
        """Test that all expected intents exist."""
        expected_intents = [
            "store_memory",
            "retrieve_memory",
            "schedule_task",
            "system_info",
            "general_query",
            "unknown"
        ]
        
        for intent_value in expected_intents:
            # Verify we can access each intent
            intent = Intent(intent_value)
            assert intent.value == intent_value
