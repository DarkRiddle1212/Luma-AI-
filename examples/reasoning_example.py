"""
Reasoning Engine Example

This example demonstrates the complete functionality of Luma's Reasoning Engine Brain,
including basic usage, intent classification, error handling, and custom LLM implementations.

The Reasoning Engine is Luma's cognitive orchestration layer that coordinates:
- Message processing
- Context building
- Intent detection
- Response generation

Run this file to see all examples in action:
    python examples/reasoning_example.py
"""

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import LLMInterface, StubLLM
from typing import Dict


def print_section(title: str):
    """Helper function to print section headers."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(result: Dict):
    """Helper function to print structured results."""
    print(f"Intent: {result['intent']}")
    print(f"Response:\n{result['response']}")
    print(f"Metadata: {result['metadata']}")
    print()


# =============================================================================
# Example 1: Basic ReasoningEngine Usage
# =============================================================================
# This example shows the simplest way to use the ReasoningEngine with the
# default StubLLM implementation. The engine automatically uses StubLLM when
# no LLM is explicitly provided.

def example_basic_usage():
    """Demonstrate basic ReasoningEngine usage with default StubLLM."""
    print_section("Example 1: Basic ReasoningEngine Usage")
    
    # Initialize ReasoningEngine with default StubLLM
    # The engine uses dependency injection, defaulting to StubLLM if no LLM is provided
    engine = ReasoningEngine()
    
    # Process a simple message
    # The engine will:
    # 1. Build context (user_message, timestamp, placeholders)
    # 2. Detect intent (rule-based classification)
    # 3. Generate response via StubLLM
    # 4. Return structured response dictionary
    result = engine.process_message("Hello, Luma! How are you today?")
    
    print("Processing: 'Hello, Luma! How are you today?'")
    print_result(result)
    
    # The result is a structured dictionary with:
    # - response: The generated text from the LLM
    # - intent: The detected intent classification
    # - metadata: Processing information (context_keys, timestamp)


# =============================================================================
# Example 2: Different Intent Classifications
# =============================================================================
# This example demonstrates how the ReasoningEngine detects different user
# intents using rule-based classification. The intent detection is deterministic
# and case-insensitive.

def example_intent_classifications():
    """Demonstrate different intent classifications."""
    print_section("Example 2: Different Intent Classifications")
    
    engine = ReasoningEngine()
    
    # Test messages for each intent type
    test_messages = [
        # store_memory intent - triggered by "remember" or "store"
        ("Remember to buy milk tomorrow", "store_memory"),
        
        # retrieve_memory intent - triggered by "what was", "recall", "retrieve"
        ("What was my last task?", "retrieve_memory"),
        
        # education intent - triggered by "teach", "learn", "explain"
        ("Teach me about Python loops", "education"),
        
        # scheduling intent - triggered by "schedule", "remind"
        ("Schedule a meeting for 3pm", "scheduling"),
        
        # general intent - default when no keywords match
        ("Hello there!", "general"),
    ]
    
    for message, expected_intent in test_messages:
        result = engine.process_message(message)
        print(f"Message: '{message}'")
        print(f"Expected Intent: {expected_intent}")
        print(f"Detected Intent: {result['intent']}")
        print(f"Match: {'✓' if result['intent'] == expected_intent else '✗'}")
        print()


# =============================================================================
# Example 3: Error Handling
# =============================================================================
# This example demonstrates how the ReasoningEngine handles various error
# conditions gracefully without crashing. The engine validates inputs and
# catches exceptions, always returning a valid response dictionary.

def example_error_handling():
    """Demonstrate error handling for invalid inputs."""
    print_section("Example 3: Error Handling")
    
    engine = ReasoningEngine()
    
    # Test 1: Empty string
    # The engine detects empty input and returns an "invalid" intent
    print("Test 1: Empty string")
    result = engine.process_message("")
    print_result(result)
    
    # Test 2: Whitespace-only string
    # The engine strips whitespace and detects it as empty
    print("Test 2: Whitespace-only string")
    result = engine.process_message("   ")
    print_result(result)
    
    # Test 3: None value
    # The engine handles None gracefully without crashing
    print("Test 3: None value")
    result = engine.process_message(None)
    print_result(result)
    
    # Note: All error cases return a valid response dictionary with:
    # - response: User-friendly error message
    # - intent: "invalid" or "error"
    # - metadata: Empty context_keys and timestamp


# =============================================================================
# Example 4: Custom LLM Implementation
# =============================================================================
# This example shows how to create and use a custom LLM implementation.
# The ReasoningEngine uses dependency injection, allowing you to swap
# LLM implementations without changing orchestration code.

class CustomLLM(LLMInterface):
    """
    Custom LLM implementation that generates personalized responses.
    
    This example demonstrates how to create your own LLM implementation
    by inheriting from LLMInterface and implementing generate_response.
    
    In production, this could be:
    - OpenAI GPT integration
    - Local model (Ollama, LLaMA)
    - Cloud API (Anthropic, Cohere)
    - Hybrid fallback system
    """
    
    def __init__(self, name: str = "CustomLLM"):
        """Initialize with a custom name."""
        self.name = name
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        """
        Generate a custom response based on intent.
        
        This implementation demonstrates intent-aware response generation.
        Real implementations would use actual AI models here.
        """
        intent = context.get("intent", "unknown")
        user_message = context.get("user_message", "")
        
        # Generate different responses based on detected intent
        if intent == "store_memory":
            return f"[{self.name}] I'll remember that: '{user_message}'"
        
        elif intent == "retrieve_memory":
            return f"[{self.name}] Let me search my memory for: '{user_message}'"
        
        elif intent == "education":
            return f"[{self.name}] I'd be happy to teach you about that! You asked: '{user_message}'"
        
        elif intent == "scheduling":
            return f"[{self.name}] I'll help you schedule that: '{user_message}'"
        
        else:  # general intent
            return f"[{self.name}] Thanks for your message: '{user_message}'"


def example_custom_llm():
    """Demonstrate custom LLM implementation."""
    print_section("Example 4: Custom LLM Implementation")
    
    # Create a custom LLM instance
    custom_llm = CustomLLM(name="MyPersonalizedLLM")
    
    # Initialize ReasoningEngine with custom LLM
    # This demonstrates dependency injection - the engine works with any
    # LLMInterface implementation without modification
    engine = ReasoningEngine(llm=custom_llm)
    
    # Test the custom LLM with different intents
    test_messages = [
        "Remember to call mom",
        "What was my last meeting about?",
        "Teach me about recursion",
        "Schedule a dentist appointment",
        "Hello!",
    ]
    
    for message in test_messages:
        result = engine.process_message(message)
        print(f"Message: '{message}'")
        print_result(result)


# =============================================================================
# Example 5: Comparing StubLLM vs Custom LLM
# =============================================================================
# This example compares the output of StubLLM (for testing) with a custom
# LLM implementation (for production) using the same message.

def example_llm_comparison():
    """Compare StubLLM and CustomLLM responses."""
    print_section("Example 5: Comparing StubLLM vs Custom LLM")
    
    message = "Teach me about Python decorators"
    
    # Process with StubLLM
    print("Using StubLLM (for testing/development):")
    stub_engine = ReasoningEngine(llm=StubLLM())
    stub_result = stub_engine.process_message(message)
    print_result(stub_result)
    
    # Process with CustomLLM
    print("Using CustomLLM (simulating production):")
    custom_engine = ReasoningEngine(llm=CustomLLM())
    custom_result = custom_engine.process_message(message)
    print_result(custom_result)
    
    # Both engines return the same structured format, but with different responses
    print("Both return the same structure:")
    print(f"  - Keys: {list(stub_result.keys())}")
    print(f"  - Intent: {stub_result['intent']} (same for both)")
    print(f"  - Response: Different based on LLM implementation")


# =============================================================================
# Example 6: Context Building and Metadata
# =============================================================================
# This example shows how the ReasoningEngine builds context and includes
# metadata in responses. The context includes placeholders for future
# integration with memory and system monitoring modules.

def example_context_and_metadata():
    """Demonstrate context building and metadata."""
    print_section("Example 6: Context Building and Metadata")
    
    engine = ReasoningEngine()
    
    # Process a message and examine the metadata
    result = engine.process_message("Explain machine learning")
    
    print("Message: 'Explain machine learning'")
    print(f"Intent: {result['intent']}")
    print(f"\nMetadata:")
    print(f"  - Context Keys: {result['metadata']['context_keys']}")
    print(f"  - Timestamp: {result['metadata']['timestamp']}")
    print(f"\nContext keys explained:")
    print("  - user_message: The original user input")
    print("  - timestamp: ISO 8601 format timestamp")
    print("  - intent: Detected intent classification")
    print("  - memory_placeholder: Reserved for future memory integration")
    print("  - system_state_placeholder: Reserved for future system monitoring")
    print()


# =============================================================================
# Example 7: Intent Detection Edge Cases
# =============================================================================
# This example demonstrates edge cases in intent detection, including
# case-insensitivity, multiple keywords, and priority ordering.

def example_intent_edge_cases():
    """Demonstrate intent detection edge cases."""
    print_section("Example 7: Intent Detection Edge Cases")
    
    engine = ReasoningEngine()
    
    # Test case-insensitivity
    print("Test 1: Case-insensitivity")
    messages = ["REMEMBER this", "Remember this", "remember this"]
    for msg in messages:
        result = engine.process_message(msg)
        print(f"  '{msg}' -> {result['intent']}")
    print()
    
    # Test multiple keywords (first match wins)
    print("Test 2: Multiple keywords (first match wins)")
    # This message contains both "remember" (store_memory) and "teach" (education)
    # The first matching rule (store_memory) wins
    result = engine.process_message("Remember to teach me Python")
    print(f"  'Remember to teach me Python' -> {result['intent']}")
    print(f"  (Contains both 'remember' and 'teach', but 'store_memory' rule comes first)")
    print()
    
    # Test partial word matches
    print("Test 3: Partial word matches")
    # Keywords must be substrings, not whole words
    result1 = engine.process_message("I remembered something")  # Contains "remember"
    result2 = engine.process_message("The teacher is here")  # Contains "teach"
    print(f"  'I remembered something' -> {result1['intent']}")
    print(f"  'The teacher is here' -> {result2['intent']}")
    print()


# =============================================================================
# Example 8: Simulating LLM Errors
# =============================================================================
# This example demonstrates how the ReasoningEngine handles LLM errors
# gracefully by catching exceptions and returning error responses.

class ErrorLLM(LLMInterface):
    """LLM implementation that simulates errors for testing."""
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        """Simulate an LLM error."""
        raise RuntimeError("Simulated LLM API error: Connection timeout")


def example_llm_error_handling():
    """Demonstrate LLM error handling."""
    print_section("Example 8: Simulating LLM Errors")
    
    # Create an engine with an LLM that always fails
    error_engine = ReasoningEngine(llm=ErrorLLM())
    
    # Process a message - the engine will catch the exception
    result = error_engine.process_message("Hello")
    
    print("Processing: 'Hello' with ErrorLLM")
    print_result(result)
    
    print("Note: The engine caught the exception and returned an error response")
    print("      with intent='error' and error details in metadata.")
    print()


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  REASONING ENGINE EXAMPLES")
    print("  Demonstrating Luma's Cognitive Orchestration Layer")
    print("=" * 70)
    
    # Run all examples in sequence
    example_basic_usage()
    example_intent_classifications()
    example_error_handling()
    example_custom_llm()
    example_llm_comparison()
    example_context_and_metadata()
    example_intent_edge_cases()
    example_llm_error_handling()
    
    # Summary
    print_section("Summary")
    print("These examples demonstrated:")
    print("  ✓ Basic ReasoningEngine usage with StubLLM")
    print("  ✓ Intent classification for different message types")
    print("  ✓ Error handling for invalid inputs")
    print("  ✓ Custom LLM implementation via dependency injection")
    print("  ✓ Context building and metadata structure")
    print("  ✓ Intent detection edge cases")
    print("  ✓ LLM error handling and recovery")
    print()
    print("Next steps:")
    print("  - Replace StubLLM with a real LLM (OpenAI, Ollama, etc.)")
    print("  - Integrate with memory module for context enrichment")
    print("  - Add ML-based intent classification")
    print("  - Connect to FastAPI endpoints for API access")
    print()


if __name__ == "__main__":
    main()
