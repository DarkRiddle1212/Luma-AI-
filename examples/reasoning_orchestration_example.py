"""
Example: Reasoning Engine & Orchestration

This example demonstrates Luma's brain skeleton:
- Handling messages
- Building context
- Routing intent
- Interfacing with stub LLM
- Returning structured responses
"""

import sys
sys.path.insert(0, '.')

from luma.core.reasoning import ReasoningEngine


def main():
    """Demonstrate the reasoning engine's orchestration capabilities."""
    
    print("=" * 60)
    print("Luma Reasoning Engine & Orchestration Demo")
    print("=" * 60)
    print()
    
    # Initialize the reasoning engine
    engine = ReasoningEngine()
    print("✓ Reasoning engine initialized with stub LLM")
    print()
    
    # Test different types of messages
    test_messages = [
        {
            "message": "Remember that I have a meeting tomorrow at 3pm",
            "context": {"user_id": "user123"}
        },
        {
            "message": "What did I say about the meeting?",
            "context": {"user_id": "user123"}
        },
        {
            "message": "Schedule a task to review the project",
            "context": {"priority": "high"}
        },
        {
            "message": "What's the system status?",
            "context": {}
        },
        {
            "message": "Help me understand what you can do",
            "context": {}
        },
        {
            "message": "Random message without clear intent",
            "context": {}
        }
    ]
    
    # Process each message
    for i, test in enumerate(test_messages, 1):
        print(f"Test {i}: {test['message']}")
        print("-" * 60)
        
        # Handle the message
        result = engine.handle_message(test["message"], test["context"])
        
        # Display results
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response']}")
        print(f"Context Keys: {', '.join(result['context'].keys())}")
        print(f"Confidence: {result['metadata']['confidence']}")
        print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("✓ Message handling with structured responses")
    print("✓ Context building from user input and system state")
    print("✓ Intent routing using keyword matching")
    print("✓ Stub LLM integration for response generation")
    print("✓ Metadata tracking for confidence and context")
    print()
    print("Future Enhancements:")
    print("• Replace stub LLM with real LLM (OpenAI, Anthropic, etc.)")
    print("• Integrate with memory module for context retrieval")
    print("• Add ML-based intent classification")
    print("• Implement conversation history tracking")
    print("• Add multi-turn dialogue support")


if __name__ == "__main__":
    main()
