"""
Integration Tests for Reasoning Engine with Different Intent Types

Tests the complete ReasoningEngine flow with all supported intent types:
- store_memory
- retrieve_memory
- education
- scheduling
- general

These integration tests verify that the entire pipeline (context building,
intent detection, LLM invocation, response generation) works correctly
for each intent type.

**Validates: Task 14.2 - Test ReasoningEngine with different intent types**
"""

import pytest
from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM


class TestReasoningEngineIntentIntegration:
    """Integration tests for ReasoningEngine with different intent types."""
    
    def test_store_memory_intent_complete_flow(self):
        """
        Test complete flow for store_memory intent.
        
        Verifies that messages with memory storage keywords are correctly
        classified and processed through the entire pipeline.
        """
        engine = ReasoningEngine()
        
        # Test various store_memory messages
        test_messages = [
            "Remember to buy milk",
            "Store this information for later",
            "Please remember my birthday is June 15th",
            "Store that I prefer Python over JavaScript"
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # Verify response structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result
            
            # Verify intent classification
            assert result["intent"] == "store_memory", f"Failed for message: {message}"
            
            # Verify response contains expected content
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "StubLLM Response" in result["response"]
            assert "store_memory" in result["response"]
            
            # Verify metadata
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
            assert "user_message" in result["metadata"]["context_keys"]
            assert "intent" in result["metadata"]["context_keys"]
    
    def test_retrieve_memory_intent_complete_flow(self):
        """
        Test complete flow for retrieve_memory intent.
        
        Verifies that messages with memory retrieval keywords are correctly
        classified and processed through the entire pipeline.
        """
        engine = ReasoningEngine()
        
        # Test various retrieve_memory messages
        test_messages = [
            "What was my last task?",
            "Recall our conversation from yesterday",
            "Retrieve my notes about the project",
            "What was I working on earlier?"
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # Verify response structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result
            
            # Verify intent classification
            assert result["intent"] == "retrieve_memory", f"Failed for message: {message}"
            
            # Verify response contains expected content
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "StubLLM Response" in result["response"]
            assert "retrieve_memory" in result["response"]
            
            # Verify metadata
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
    
    def test_education_intent_complete_flow(self):
        """
        Test complete flow for education intent.
        
        Verifies that messages with educational keywords are correctly
        classified and processed through the entire pipeline.
        """
        engine = ReasoningEngine()
        
        # Test various education messages
        test_messages = [
            "Teach me Python loops",
            "I want to learn about machine learning",
            "Explain how recursion works",
            "Teach me about data structures",
            "Can you explain quantum computing?"
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # Verify response structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result
            
            # Verify intent classification
            assert result["intent"] == "education", f"Failed for message: {message}"
            
            # Verify response contains expected content
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "StubLLM Response" in result["response"]
            assert "education" in result["response"]
            
            # Verify metadata
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
    
    def test_scheduling_intent_complete_flow(self):
        """
        Test complete flow for scheduling intent.
        
        Verifies that messages with scheduling keywords are correctly
        classified and processed through the entire pipeline.
        """
        engine = ReasoningEngine()
        
        # Test various scheduling messages
        test_messages = [
            "Schedule a meeting for tomorrow",
            "Remind me to call John at 3pm",
            "Schedule my dentist appointment",
            "Remind me about the deadline next week"
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # Verify response structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result
            
            # Verify intent classification
            assert result["intent"] == "scheduling", f"Failed for message: {message}"
            
            # Verify response contains expected content
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "StubLLM Response" in result["response"]
            assert "scheduling" in result["response"]
            
            # Verify metadata
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
    
    def test_general_intent_complete_flow(self):
        """
        Test complete flow for general intent.
        
        Verifies that messages without specific keywords are correctly
        classified as general and processed through the entire pipeline.
        """
        engine = ReasoningEngine()
        
        # Test various general messages
        test_messages = [
            "Hello, how are you?",
            "What can you do?",
            "Tell me a joke",
            "What's the weather like?",
            "Good morning!"
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # Verify response structure
            assert isinstance(result, dict)
            assert "response" in result
            assert "intent" in result
            assert "metadata" in result
            
            # Verify intent classification
            assert result["intent"] == "general", f"Failed for message: {message}"
            
            # Verify response contains expected content
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            assert "StubLLM Response" in result["response"]
            assert "general" in result["response"]
            
            # Verify metadata
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
    
    def test_all_intents_produce_consistent_structure(self):
        """
        Test that all intent types produce consistent response structure.
        
        Verifies that regardless of intent type, the response structure
        remains consistent with all required keys and proper types.
        """
        engine = ReasoningEngine()
        
        # Test messages for each intent type
        intent_messages = {
            "store_memory": "Remember this important fact",
            "retrieve_memory": "What was that thing I mentioned?",
            "education": "Teach me about Python",
            "scheduling": "Schedule a task for tomorrow",
            "general": "Hello there!"
        }
        
        for expected_intent, message in intent_messages.items():
            result = engine.process_message(message)
            
            # Verify consistent structure
            assert isinstance(result, dict)
            assert set(result.keys()) == {"response", "intent", "metadata"}
            
            # Verify types
            assert isinstance(result["response"], str)
            assert isinstance(result["intent"], str)
            assert isinstance(result["metadata"], dict)
            
            # Verify metadata structure
            assert "context_keys" in result["metadata"]
            assert "timestamp" in result["metadata"]
            assert isinstance(result["metadata"]["context_keys"], list)
            assert isinstance(result["metadata"]["timestamp"], str)
            
            # Verify intent matches expected
            assert result["intent"] == expected_intent
    
    def test_intent_detection_with_context_enrichment(self):
        """
        Test that detected intent is properly added to context.
        
        Verifies that the intent detection result is added to the context
        dictionary before being passed to the LLM.
        """
        engine = ReasoningEngine()
        
        test_cases = [
            ("Remember this", "store_memory"),
            ("What was that?", "retrieve_memory"),
            ("Teach me", "education"),
            ("Schedule it", "scheduling"),
            ("Hello", "general")
        ]
        
        for message, expected_intent in test_cases:
            result = engine.process_message(message)
            
            # Verify intent is in context_keys (meaning it was added to context)
            assert "intent" in result["metadata"]["context_keys"]
            
            # Verify the intent value matches expected
            assert result["intent"] == expected_intent
            
            # Verify the response includes the intent (from StubLLM echoing context)
            assert expected_intent in result["response"]
    
    def test_llm_receives_correct_context_for_each_intent(self):
        """
        Test that LLM receives properly formatted context for each intent type.
        
        Verifies that the context passed to the LLM includes all required
        keys and the correct intent classification.
        """
        engine = ReasoningEngine()
        
        intent_messages = {
            "store_memory": "Remember my favorite color is blue",
            "retrieve_memory": "What was my favorite color?",
            "education": "Explain color theory to me",
            "scheduling": "Remind me to paint tomorrow",
            "general": "I like colors"
        }
        
        for expected_intent, message in intent_messages.items():
            result = engine.process_message(message)
            
            # Verify context keys are present in metadata
            context_keys = result["metadata"]["context_keys"]
            
            # All contexts should have these keys
            required_keys = ["user_message", "timestamp", "memory_placeholder", 
                           "system_state_placeholder", "intent"]
            
            for key in required_keys:
                assert key in context_keys, f"Missing key '{key}' for intent '{expected_intent}'"
            
            # Verify the response shows correct intent was passed to LLM
            assert f"Intent: {expected_intent}" in result["response"]
    
    def test_multiple_sequential_messages_with_different_intents(self):
        """
        Test processing multiple sequential messages with different intents.
        
        Verifies that the ReasoningEngine correctly handles multiple messages
        in sequence, properly classifying each one independently.
        """
        engine = ReasoningEngine()
        
        # Sequence of messages with different intents
        message_sequence = [
            ("Remember my name is Alice", "store_memory"),
            ("What was my name?", "retrieve_memory"),
            ("Teach me about names", "education"),
            ("Schedule a name change appointment", "scheduling"),
            ("Names are interesting", "general"),
            ("Store this fact about names", "store_memory")
        ]
        
        for message, expected_intent in message_sequence:
            result = engine.process_message(message)
            
            # Verify each message is classified correctly
            assert result["intent"] == expected_intent, \
                f"Message '{message}' was classified as '{result['intent']}' instead of '{expected_intent}'"
            
            # Verify response structure is consistent
            assert "response" in result
            assert "metadata" in result
            assert isinstance(result["response"], str)
    
    def test_case_insensitive_intent_detection_across_all_intents(self):
        """
        Test that intent detection is case-insensitive for all intent types.
        
        Verifies that keywords in any case (upper, lower, mixed) are correctly
        detected for all intent types.
        """
        engine = ReasoningEngine()
        
        # Test case variations for each intent
        test_cases = [
            ("REMEMBER THIS", "store_memory"),
            ("remember this", "store_memory"),
            ("ReMeMbEr ThIs", "store_memory"),
            ("WHAT WAS THAT", "retrieve_memory"),
            ("what was that", "retrieve_memory"),
            ("WhAt WaS tHaT", "retrieve_memory"),
            ("TEACH ME", "education"),
            ("teach me", "education"),
            ("TeAcH mE", "education"),
            ("SCHEDULE IT", "scheduling"),
            ("schedule it", "scheduling"),
            ("ScHeDuLe It", "scheduling"),
            ("HELLO", "general"),
            ("hello", "general"),
            ("HeLLo", "general")
        ]
        
        for message, expected_intent in test_cases:
            result = engine.process_message(message)
            assert result["intent"] == expected_intent, \
                f"Case-insensitive detection failed for '{message}'"


class TestReasoningEngineIntentEdgeCases:
    """Test edge cases for intent detection in integration scenarios."""
    
    def test_message_with_multiple_intent_keywords(self):
        """
        Test messages that contain keywords for multiple intents.
        
        Verifies that the first matching rule takes precedence when
        multiple intent keywords are present.
        """
        engine = ReasoningEngine()
        
        # Message with both "remember" (store_memory) and "teach" (education)
        # Should match store_memory first based on rule order
        result = engine.process_message("Remember to teach me about Python")
        assert result["intent"] == "store_memory"
        
        # Message with "recall" (retrieve_memory) and "schedule" (scheduling)
        # Should match retrieve_memory first
        result = engine.process_message("Recall when I need to schedule the meeting")
        assert result["intent"] == "retrieve_memory"
    
    def test_intent_keywords_in_longer_sentences(self):
        """
        Test that intent keywords are detected within longer, complex sentences.
        
        Verifies that the intent detection works correctly even when keywords
        are embedded in longer, more natural sentences.
        """
        engine = ReasoningEngine()
        
        test_cases = [
            ("I would really appreciate it if you could remember this important detail for me", "store_memory"),
            ("Can you help me recall what was discussed in our last conversation?", "retrieve_memory"),
            ("I'm trying to learn more about this topic, could you teach me the basics?", "education"),
            ("Would it be possible to schedule a reminder for next Tuesday?", "scheduling"),
            ("I'm just saying hello and wondering how things are going", "general")
        ]
        
        for message, expected_intent in test_cases:
            result = engine.process_message(message)
            assert result["intent"] == expected_intent, \
                f"Failed to detect intent in longer sentence: '{message}'"
    
    def test_intent_with_special_characters_and_punctuation(self):
        """
        Test intent detection with special characters and punctuation.
        
        Verifies that punctuation and special characters don't interfere
        with intent detection.
        """
        engine = ReasoningEngine()
        
        test_cases = [
            ("Remember this!!!", "store_memory"),
            ("What was that???", "retrieve_memory"),
            ("Teach me... please?", "education"),
            ("Schedule: tomorrow @ 3pm", "scheduling"),
            ("Hello! How are you?", "general")
        ]
        
        for message, expected_intent in test_cases:
            result = engine.process_message(message)
            assert result["intent"] == expected_intent


class TestReasoningEngineIntentWithCustomLLM:
    """Test intent handling with custom LLM implementations."""
    
    def test_all_intents_work_with_custom_llm(self):
        """
        Test that all intent types work correctly with a custom LLM.
        
        Verifies that the intent detection and context building work
        independently of the LLM implementation.
        """
        from luma.core.llm_interface import LLMInterface
        
        class CustomLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                intent = context.get("intent", "unknown")
                return f"Custom LLM response for intent: {intent}"
        
        engine = ReasoningEngine(llm=CustomLLM())
        
        intent_messages = {
            "store_memory": "Remember this",
            "retrieve_memory": "What was that?",
            "education": "Teach me",
            "scheduling": "Schedule it",
            "general": "Hello"
        }
        
        for expected_intent, message in intent_messages.items():
            result = engine.process_message(message)
            
            # Verify intent is correctly detected
            assert result["intent"] == expected_intent
            
            # Verify custom LLM received the correct intent
            assert expected_intent in result["response"]
            assert "Custom LLM response" in result["response"]


class TestErrorPropagationThroughPipeline:
    """
    Integration tests for error propagation through the reasoning engine pipeline.
    
    Tests that errors occurring at different stages of the pipeline (context building,
    intent detection, LLM invocation) are properly caught, handled, and propagated
    with appropriate error responses.
    
    **Validates: Task 14.3 - Test error propagation through the pipeline**
    **Validates: Requirement 8 - Error Handling and Robustness**
    """
    
    def test_llm_exception_propagates_as_error_response(self):
        """
        Test that exceptions raised by LLM are caught and returned as error responses.
        
        Verifies that when the LLM raises an exception during generate_response,
        the ReasoningEngine catches it and returns a structured error response
        instead of letting the exception propagate to the caller.
        """
        from luma.core.llm_interface import LLMInterface
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("LLM service unavailable")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test message")
        
        # Verify error response structure
        assert isinstance(result, dict)
        assert "response" in result
        assert "intent" in result
        assert "metadata" in result
        
        # Verify error intent
        assert result["intent"] == "error"
        
        # Verify error message in response
        assert "error occurred" in result["response"].lower()
        assert "LLM service unavailable" in result["response"]
        
        # Verify error in metadata
        assert "error" in result["metadata"]
        assert "LLM service unavailable" in result["metadata"]["error"]
    
    def test_llm_timeout_error_propagates_correctly(self):
        """
        Test that timeout errors from LLM are handled gracefully.
        
        Simulates a timeout scenario where the LLM takes too long to respond.
        """
        from luma.core.llm_interface import LLMInterface
        
        class TimeoutLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise TimeoutError("LLM request timed out after 30 seconds")
        
        engine = ReasoningEngine(llm=TimeoutLLM())
        result = engine.process_message("Teach me Python")
        
        # Verify error is caught and handled
        assert result["intent"] == "error"
        assert "timed out" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_llm_value_error_propagates_correctly(self):
        """
        Test that ValueError from LLM (e.g., invalid input) is handled gracefully.
        
        Simulates scenarios where the LLM rejects invalid input parameters.
        """
        from luma.core.llm_interface import LLMInterface
        
        class ValidatingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                if not prompt:
                    raise ValueError("Prompt cannot be empty")
                return "Valid response"
        
        engine = ReasoningEngine(llm=ValidatingLLM())
        
        # This should work fine since we validate before calling LLM
        result = engine.process_message("Valid message")
        assert result["intent"] != "error"
        
        # But if LLM raises ValueError for other reasons, it should be caught
        class StrictLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise ValueError("Invalid context format")
        
        engine2 = ReasoningEngine(llm=StrictLLM())
        result2 = engine2.process_message("Test")
        
        assert result2["intent"] == "error"
        assert "Invalid context format" in result2["response"]
    
    def test_llm_connection_error_propagates_correctly(self):
        """
        Test that connection errors from LLM API are handled gracefully.
        
        Simulates network/connection failures when calling external LLM services.
        """
        from luma.core.llm_interface import LLMInterface
        
        class DisconnectedLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise ConnectionError("Failed to connect to LLM API endpoint")
        
        engine = ReasoningEngine(llm=DisconnectedLLM())
        result = engine.process_message("Hello")
        
        # Verify connection error is caught
        assert result["intent"] == "error"
        assert "error occurred" in result["response"].lower()
        assert "Failed to connect" in result["response"]
        assert "error" in result["metadata"]
    
    def test_llm_key_error_propagates_correctly(self):
        """
        Test that KeyError from LLM (e.g., missing context keys) is handled.
        
        Simulates scenarios where LLM expects specific context keys that are missing.
        """
        from luma.core.llm_interface import LLMInterface
        
        class StrictContextLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                # Try to access a key that doesn't exist
                required_key = context["nonexistent_key"]
                return f"Response with {required_key}"
        
        engine = ReasoningEngine(llm=StrictContextLLM())
        result = engine.process_message("Test message")
        
        # Verify KeyError is caught
        assert result["intent"] == "error"
        assert "error occurred" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_error_response_maintains_consistent_structure(self):
        """
        Test that error responses maintain the same structure as successful responses.
        
        Verifies that even when errors occur, the response dictionary has all
        required keys with appropriate types.
        """
        from luma.core.llm_interface import LLMInterface
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise Exception("Generic error")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test")
        
        # Verify structure consistency
        assert isinstance(result, dict)
        assert set(result.keys()) == {"response", "intent", "metadata"}
        
        # Verify types
        assert isinstance(result["response"], str)
        assert isinstance(result["intent"], str)
        assert isinstance(result["metadata"], dict)
        
        # Verify metadata structure
        assert "timestamp" in result["metadata"]
        assert "context_keys" in result["metadata"]
        assert "error" in result["metadata"]
        
        # Verify error-specific values
        assert result["intent"] == "error"
        assert len(result["response"]) > 0
    
    def test_multiple_sequential_errors_handled_independently(self):
        """
        Test that multiple sequential errors are handled independently.
        
        Verifies that the engine doesn't maintain error state between calls.
        """
        from luma.core.llm_interface import LLMInterface
        
        class IntermittentLLM(LLMInterface):
            def __init__(self):
                self.call_count = 0
            
            def generate_response(self, prompt: str, context: dict) -> str:
                self.call_count += 1
                if self.call_count % 2 == 1:
                    raise RuntimeError(f"Error on call {self.call_count}")
                return f"Success on call {self.call_count}"
        
        llm = IntermittentLLM()
        engine = ReasoningEngine(llm=llm)
        
        # First call should error
        result1 = engine.process_message("First message")
        assert result1["intent"] == "error"
        assert "Error on call 1" in result1["response"]
        
        # Second call should succeed
        result2 = engine.process_message("Second message")
        assert result2["intent"] != "error"
        assert "Success on call 2" in result2["response"]
        
        # Third call should error again
        result3 = engine.process_message("Third message")
        assert result3["intent"] == "error"
        assert "Error on call 3" in result3["response"]
    
    def test_error_during_context_building_propagates_correctly(self):
        """
        Test that errors during context building are caught and handled.
        
        While the current implementation is unlikely to error during context building,
        this test ensures robustness for future enhancements.
        """
        from luma.core.llm_interface import LLMInterface
        from unittest.mock import patch
        
        class SafeLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return "Safe response"
        
        engine = ReasoningEngine(llm=SafeLLM())
        
        # Mock build_context to raise an exception
        with patch.object(engine, 'build_context', side_effect=RuntimeError("Context building failed")):
            result = engine.process_message("Test message")
            
            # Verify error is caught
            assert result["intent"] == "error"
            assert "error occurred" in result["response"].lower()
            assert "Context building failed" in result["response"]
    
    def test_error_during_intent_detection_propagates_correctly(self):
        """
        Test that errors during intent detection are caught and handled.
        
        While the current implementation is unlikely to error during intent detection,
        this test ensures robustness for future ML-based implementations.
        """
        from luma.core.llm_interface import LLMInterface
        from unittest.mock import patch
        
        class SafeLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return "Safe response"
        
        engine = ReasoningEngine(llm=SafeLLM())
        
        # Mock detect_intent to raise an exception
        with patch.object(engine, 'detect_intent', side_effect=RuntimeError("Intent detection failed")):
            result = engine.process_message("Test message")
            
            # Verify error is caught
            assert result["intent"] == "error"
            assert "error occurred" in result["response"].lower()
            assert "Intent detection failed" in result["response"]
    
    def test_error_with_different_intent_types(self):
        """
        Test that errors are handled consistently across different intent types.
        
        Verifies that the error handling works the same regardless of what
        intent would have been detected.
        """
        from luma.core.llm_interface import LLMInterface
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("LLM failed")
        
        engine = ReasoningEngine(llm=FailingLLM())
        
        # Test messages that would trigger different intents
        test_messages = [
            "Remember this",  # store_memory
            "What was that?",  # retrieve_memory
            "Teach me",  # education
            "Schedule it",  # scheduling
            "Hello"  # general
        ]
        
        for message in test_messages:
            result = engine.process_message(message)
            
            # All should return error intent
            assert result["intent"] == "error", f"Failed for message: {message}"
            assert "error occurred" in result["response"].lower()
            assert "LLM failed" in result["response"]
            assert "error" in result["metadata"]
    
    def test_error_response_includes_timestamp(self):
        """
        Test that error responses include timestamp in metadata.
        
        Verifies that even when errors occur, temporal information is preserved.
        """
        from luma.core.llm_interface import LLMInterface
        from datetime import datetime
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("Error")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test")
        
        # Verify timestamp exists and is valid
        assert "timestamp" in result["metadata"]
        timestamp = result["metadata"]["timestamp"]
        assert isinstance(timestamp, str)
        
        # Verify it's valid ISO format
        try:
            parsed = datetime.fromisoformat(timestamp)
            assert isinstance(parsed, datetime)
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")
    
    def test_error_response_has_empty_context_keys(self):
        """
        Test that error responses have empty context_keys list.
        
        Verifies that when an error occurs, the context_keys list is empty
        since context may not have been fully built.
        """
        from luma.core.llm_interface import LLMInterface
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("Error")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test")
        
        # Verify context_keys is empty on error
        assert "context_keys" in result["metadata"]
        assert isinstance(result["metadata"]["context_keys"], list)
        assert len(result["metadata"]["context_keys"]) == 0
    
    def test_error_message_is_informative(self):
        """
        Test that error messages provide useful information for debugging.
        
        Verifies that error responses include the actual exception message
        to help with troubleshooting.
        """
        from luma.core.llm_interface import LLMInterface
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("Specific error: API key invalid")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test")
        
        # Verify error message is informative
        assert "Specific error: API key invalid" in result["response"]
        assert "Specific error: API key invalid" in result["metadata"]["error"]
    
    def test_exception_with_no_message_handled_gracefully(self):
        """
        Test that exceptions without messages are handled gracefully.
        
        Verifies that even when an exception has no message, the error
        response is still properly formatted.
        """
        from luma.core.llm_interface import LLMInterface
        
        class EmptyErrorLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError()  # No message
        
        engine = ReasoningEngine(llm=EmptyErrorLLM())
        result = engine.process_message("Test")
        
        # Verify error is handled even without message
        assert result["intent"] == "error"
        assert "error occurred" in result["response"].lower()
        assert "error" in result["metadata"]
    
    def test_nested_exception_propagates_correctly(self):
        """
        Test that nested exceptions (exception chains) are handled correctly.
        
        Verifies that when an exception is raised from another exception,
        the error handling captures the relevant information.
        """
        from luma.core.llm_interface import LLMInterface
        
        class NestedErrorLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                try:
                    raise ValueError("Inner error")
                except ValueError as e:
                    raise RuntimeError("Outer error") from e
        
        engine = ReasoningEngine(llm=NestedErrorLLM())
        result = engine.process_message("Test")
        
        # Verify outer error is caught
        assert result["intent"] == "error"
        assert "Outer error" in result["response"]
        assert "error" in result["metadata"]


# ============================================================================
# 14.4 Test logging output at different levels
# ============================================================================

class TestLoggingOutputAtDifferentLevels:
    """
    Integration tests for logging output at different levels throughout the
    reasoning engine pipeline.
    
    Tests that appropriate log messages are generated at DEBUG, INFO, WARNING,
    and ERROR levels during different stages of message processing.
    
    **Validates: Task 14.4 - Test logging output at different levels**
    **Validates: Requirement 8.5 - Log errors and warnings appropriately**
    """
    
    def test_info_level_logging_on_successful_processing(self, caplog):
        """
        Test that INFO level logs are generated during successful message processing.
        
        Verifies that when a message is successfully processed, an INFO log
        is generated indicating the detected intent.
        """
        import logging
        
        # Set log level to INFO to capture INFO and above
        caplog.set_level(logging.INFO)
        
        engine = ReasoningEngine()
        result = engine.process_message("Teach me Python")
        
        # Verify successful processing
        assert result["intent"] == "education"
        
        # Verify INFO log was generated
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert len(info_logs) > 0
        
        # Verify log contains intent information
        log_messages = [record.message for record in info_logs]
        assert any("Processing message with intent: education" in msg for msg in log_messages)
    
    def test_info_level_logging_on_initialization(self, caplog):
        """
        Test that INFO level logs are generated during ReasoningEngine initialization.
        
        Verifies that when the ReasoningEngine is initialized, an INFO log
        is generated indicating which LLM implementation is being used.
        """
        import logging
        
        caplog.set_level(logging.INFO)
        
        # Initialize with StubLLM
        engine = ReasoningEngine(llm=StubLLM())
        
        # Verify INFO log was generated
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert len(info_logs) > 0
        
        # Verify log contains LLM type information
        log_messages = [record.message for record in info_logs]
        assert any("ReasoningEngine initialized with StubLLM" in msg for msg in log_messages)
    
    def test_debug_level_logging_on_context_building(self, caplog):
        """
        Test that DEBUG level logs are generated during context building.
        
        Verifies that when context is built, a DEBUG log is generated
        showing the context keys.
        """
        import logging
        
        # Set log level to DEBUG to capture all logs
        caplog.set_level(logging.DEBUG)
        
        engine = ReasoningEngine()
        result = engine.process_message("Test message")
        
        # Verify DEBUG logs were generated
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) > 0
        
        # Verify log contains context keys information
        log_messages = [record.message for record in debug_logs]
        assert any("Built context with keys:" in msg for msg in log_messages)
    
    def test_debug_level_logging_on_intent_detection(self, caplog):
        """
        Test that DEBUG level logs are generated during intent detection.
        
        Verifies that when intent is detected, a DEBUG log is generated
        showing the detected intent.
        """
        import logging
        
        caplog.set_level(logging.DEBUG)
        
        engine = ReasoningEngine()
        result = engine.process_message("Remember this important fact")
        
        # Verify successful processing
        assert result["intent"] == "store_memory"
        
        # Verify DEBUG logs were generated
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) > 0
        
        # Verify log contains intent detection information
        log_messages = [record.message for record in debug_logs]
        assert any("Detected intent: store_memory" in msg for msg in log_messages)
    
    def test_warning_level_logging_on_empty_message(self, caplog):
        """
        Test that WARNING level logs are generated for empty messages.
        
        Verifies that when an empty message is received, a WARNING log
        is generated indicating the invalid input.
        """
        import logging
        
        caplog.set_level(logging.WARNING)
        
        engine = ReasoningEngine()
        result = engine.process_message("")
        
        # Verify invalid intent returned
        assert result["intent"] == "invalid"
        
        # Verify WARNING log was generated
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_logs) > 0
        
        # Verify log contains empty message warning
        log_messages = [record.message for record in warning_logs]
        assert any("Empty message received" in msg for msg in log_messages)
    
    def test_warning_level_logging_on_whitespace_only_message(self, caplog):
        """
        Test that WARNING level logs are generated for whitespace-only messages.
        
        Verifies that whitespace-only messages trigger the same WARNING
        as empty messages.
        """
        import logging
        
        caplog.set_level(logging.WARNING)
        
        engine = ReasoningEngine()
        result = engine.process_message("   \t\n   ")
        
        # Verify invalid intent returned
        assert result["intent"] == "invalid"
        
        # Verify WARNING log was generated
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_logs) > 0
        
        # Verify log message
        log_messages = [record.message for record in warning_logs]
        assert any("Empty message received" in msg for msg in log_messages)
    
    def test_error_level_logging_on_llm_exception(self, caplog):
        """
        Test that ERROR level logs are generated when LLM raises an exception.
        
        Verifies that when the LLM raises an exception, an ERROR log is
        generated with the exception details and stack trace.
        """
        import logging
        from luma.core.llm_interface import LLMInterface
        
        caplog.set_level(logging.ERROR)
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise RuntimeError("LLM service unavailable")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test message")
        
        # Verify error intent returned
        assert result["intent"] == "error"
        
        # Verify ERROR log was generated
        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_logs) > 0
        
        # Verify log contains error information
        log_messages = [record.message for record in error_logs]
        assert any("Error processing message" in msg for msg in log_messages)
        assert any("LLM service unavailable" in msg for msg in log_messages)
    
    def test_error_level_logging_includes_exception_info(self, caplog):
        """
        Test that ERROR level logs include exception information (exc_info).
        
        Verifies that when an error is logged, the full exception traceback
        is included for debugging purposes.
        """
        import logging
        from luma.core.llm_interface import LLMInterface
        
        caplog.set_level(logging.ERROR)
        
        class FailingLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                raise ValueError("Invalid context format")
        
        engine = ReasoningEngine(llm=FailingLLM())
        result = engine.process_message("Test")
        
        # Verify ERROR log was generated
        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_logs) > 0
        
        # Verify exception info is included
        error_record = error_logs[0]
        assert error_record.exc_info is not None
        assert error_record.exc_info[0] == ValueError
    
    def test_logging_across_multiple_intents(self, caplog):
        """
        Test that logging works consistently across different intent types.
        
        Verifies that INFO and DEBUG logs are generated correctly for
        all intent types.
        """
        import logging
        
        caplog.set_level(logging.DEBUG)
        
        engine = ReasoningEngine()
        
        intent_messages = {
            "store_memory": "Remember this",
            "retrieve_memory": "What was that?",
            "education": "Teach me",
            "scheduling": "Schedule it",
            "general": "Hello"
        }
        
        for expected_intent, message in intent_messages.items():
            caplog.clear()
            result = engine.process_message(message)
            
            # Verify intent
            assert result["intent"] == expected_intent
            
            # Verify INFO log for processing
            info_logs = [record for record in caplog.records if record.levelname == "INFO"]
            assert len(info_logs) > 0
            log_messages = [record.message for record in info_logs]
            assert any(f"Processing message with intent: {expected_intent}" in msg for msg in log_messages)
            
            # Verify DEBUG log for intent detection
            debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
            assert len(debug_logs) > 0
            log_messages = [record.message for record in debug_logs]
            assert any(f"Detected intent: {expected_intent}" in msg for msg in log_messages)
    
    def test_logging_with_custom_llm(self, caplog):
        """
        Test that logging works correctly with custom LLM implementations.
        
        Verifies that the logging behavior is consistent regardless of
        which LLM implementation is used.
        """
        import logging
        from luma.core.llm_interface import LLMInterface
        
        caplog.set_level(logging.INFO)
        
        class CustomLLM(LLMInterface):
            def generate_response(self, prompt: str, context: dict) -> str:
                return "Custom response"
        
        engine = ReasoningEngine(llm=CustomLLM())
        
        # Verify initialization log
        init_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert len(init_logs) > 0
        log_messages = [record.message for record in init_logs]
        assert any("ReasoningEngine initialized with CustomLLM" in msg for msg in log_messages)
        
        caplog.clear()
        
        # Process message and verify logging
        result = engine.process_message("Test message")
        
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert len(info_logs) > 0
    
    def test_no_debug_logs_when_level_is_info(self, caplog):
        """
        Test that DEBUG logs are not captured when log level is set to INFO.
        
        Verifies that log level filtering works correctly and DEBUG logs
        are only generated when the level is set to DEBUG or lower.
        """
        import logging
        
        # Set log level to INFO (should not capture DEBUG)
        caplog.set_level(logging.INFO)
        
        engine = ReasoningEngine()
        result = engine.process_message("Test message")
        
        # Verify no DEBUG logs were captured
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) == 0
        
        # But INFO logs should be captured
        info_logs = [record for record in caplog.records if record.levelname == "INFO"]
        assert len(info_logs) > 0
    
    def test_logging_module_name_is_correct(self, caplog):
        """
        Test that log records have the correct module name.
        
        Verifies that logs are generated from the correct module
        (luma.core.reasoning) for proper log filtering and routing.
        """
        import logging
        
        caplog.set_level(logging.DEBUG)
        
        engine = ReasoningEngine()
        result = engine.process_message("Test message")
        
        # Verify all logs are from the correct module
        for record in caplog.records:
            assert record.name == "luma.core.reasoning"
    
    def test_logging_sequence_through_pipeline(self, caplog):
        """
        Test the complete logging sequence through the entire pipeline.
        
        Verifies that logs are generated in the correct order:
        1. INFO: Initialization
        2. DEBUG: Context building
        3. DEBUG: Intent detection
        4. INFO: Message processing
        """
        import logging
        
        caplog.set_level(logging.DEBUG)
        
        # Initialize engine (should log INFO)
        engine = ReasoningEngine()
        
        # Clear logs after initialization
        init_log_count = len(caplog.records)
        caplog.clear()
        
        # Process message
        result = engine.process_message("Teach me Python")
        
        # Verify logs were generated in sequence
        assert len(caplog.records) > 0
        
        # Check for expected log sequence
        log_sequence = [(record.levelname, record.message) for record in caplog.records]
        
        # Should have DEBUG logs for context and intent, then INFO for processing
        debug_logs = [msg for level, msg in log_sequence if level == "DEBUG"]
        info_logs = [msg for level, msg in log_sequence if level == "INFO"]
        
        assert len(debug_logs) >= 2  # Context building + intent detection
        assert len(info_logs) >= 1   # Message processing
        
        # Verify order: DEBUG logs should come before INFO processing log
        context_log_idx = next(i for i, (level, msg) in enumerate(log_sequence) 
                              if level == "DEBUG" and "Built context" in msg)
        intent_log_idx = next(i for i, (level, msg) in enumerate(log_sequence) 
                             if level == "DEBUG" and "Detected intent" in msg)
        processing_log_idx = next(i for i, (level, msg) in enumerate(log_sequence) 
                                 if level == "INFO" and "Processing message" in msg)
        
        assert context_log_idx < processing_log_idx
        assert intent_log_idx < processing_log_idx
    
    def test_error_logging_with_different_exception_types(self, caplog):
        """
        Test that ERROR logs are generated for different exception types.
        
        Verifies that all exception types are properly logged at ERROR level.
        """
        import logging
        from luma.core.llm_interface import LLMInterface
        
        exception_types = [
            (RuntimeError, "Runtime error occurred"),
            (ValueError, "Value error occurred"),
            (TypeError, "Type error occurred"),
            (ConnectionError, "Connection error occurred"),
            (TimeoutError, "Timeout error occurred")
        ]
        
        for exception_class, error_message in exception_types:
            caplog.clear()
            caplog.set_level(logging.ERROR)
            
            class FailingLLM(LLMInterface):
                def generate_response(self, prompt: str, context: dict) -> str:
                    raise exception_class(error_message)
            
            engine = ReasoningEngine(llm=FailingLLM())
            result = engine.process_message("Test")
            
            # Verify ERROR log was generated
            error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
            assert len(error_logs) > 0, f"No ERROR log for {exception_class.__name__}"
            
            # Verify error message is in log
            log_messages = [record.message for record in error_logs]
            assert any(error_message in msg for msg in log_messages), \
                f"Error message not found in logs for {exception_class.__name__}"
    
    def test_logging_performance_minimal_overhead(self, caplog):
        """
        Test that logging doesn't significantly impact performance.
        
        Verifies that the logging infrastructure doesn't add significant
        overhead to message processing.
        """
        import logging
        import time
        
        caplog.set_level(logging.DEBUG)
        
        engine = ReasoningEngine()
        
        # Process multiple messages and measure time
        start_time = time.time()
        for i in range(100):
            result = engine.process_message(f"Test message {i}")
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Should complete 100 messages in reasonable time (< 1 second with StubLLM)
        assert elapsed_time < 1.0, f"Logging overhead too high: {elapsed_time}s for 100 messages"
        
        # Verify logs were generated
        assert len(caplog.records) > 0
