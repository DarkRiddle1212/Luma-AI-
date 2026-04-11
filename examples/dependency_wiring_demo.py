"""
Dependency Wiring Demonstration

This script demonstrates how to use the dependency wiring functions
to initialize the Luma application with all components properly configured.
"""

from luma.container import initialize_application, verify_dependencies, cleanup_application


def main():
    """Demonstrate dependency wiring and application usage."""
    print("=" * 60)
    print("Luma Dependency Wiring Demonstration")
    print("=" * 60)
    print()
    
    # Step 1: Initialize application with all dependencies
    print("Step 1: Initializing application...")
    engine, storage = initialize_application(
        db_path="./data/demo_memory.db",
        return_storage=True
    )
    print("✓ Application initialized successfully")
    print()
    
    try:
        # Step 2: Verify all dependencies are configured
        print("Step 2: Verifying dependencies...")
        verify_dependencies(engine)
        print("✓ All dependencies verified")
        print()
        
        # Step 3: Test general message processing
        print("Step 3: Testing general message processing...")
        result = engine.process_message("Hello, Luma!")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response'][:100]}...")
        print()
        
        # Step 4: Test memory storage
        print("Step 4: Testing memory storage...")
        result = engine.process_message("Remember to buy milk and eggs")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response']}")
        if 'memory_id' in result['metadata']:
            print(f"Memory ID: {result['metadata']['memory_id']}")
        print()
        
        # Step 5: Test memory retrieval
        print("Step 5: Testing memory retrieval...")
        result = engine.process_message("What was I supposed to buy?")
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response'][:100]}...")
        print()
        
        print("=" * 60)
        print("Demonstration completed successfully!")
        print("=" * 60)
        
    finally:
        # Step 6: Cleanup resources
        print()
        print("Cleaning up resources...")
        cleanup_application(storage)
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
