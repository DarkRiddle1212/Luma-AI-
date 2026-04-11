"""
Basic usage example for Luma Memory Module.
"""

from luma_memory import MemoryManager, MemoryType, SQLiteStorage
from datetime import datetime, timedelta


def main():
    """Demonstrate basic memory operations."""
    
    # Initialize memory manager with SQLite storage
    print("Initializing Luma Memory Manager...")
    manager = MemoryManager(storage=SQLiteStorage("example_memory.db"))
    
    # Store some memories
    print("\n1. Storing memories...")
    
    # Store a user action
    action_id = manager.store_memory(
        content="User opened Chrome browser",
        memory_type=MemoryType.ACTION,
        source="laptop",
        metadata={"app": "chrome", "window_title": "New Tab"},
        tags=["browser", "productivity"]
    )
    print(f"   Stored action: {action_id}")
    
    # Store context information
    context_id = manager.store_memory(
        content="User is at home office",
        memory_type=MemoryType.CONTEXT,
        source="laptop",
        metadata={"location": "home", "wifi": "HomeNetwork"},
        tags=["location", "context"]
    )
    print(f"   Stored context: {context_id}")
    
    # Store a conversation
    conv_id = manager.store_memory(
        content="User asked: 'What's the weather today?'",
        memory_type=MemoryType.CONVERSATION,
        source="phone",
        metadata={"intent": "weather_query"},
        tags=["conversation", "weather"]
    )
    print(f"   Stored conversation: {conv_id}")
    
    # Retrieve all memories
    print("\n2. Retrieving all memories...")
    all_memories = manager.retrieve_memories(limit=10)
    print(f"   Found {len(all_memories)} memories")
    for memory in all_memories:
        print(f"   - [{memory.memory_type.value}] {memory.content[:50]}...")
    
    # Retrieve memories by type
    print("\n3. Retrieving ACTION memories only...")
    actions = manager.retrieve_memories(memory_type=MemoryType.ACTION)
    print(f"   Found {len(actions)} action memories")
    
    # Retrieve memories by source
    print("\n4. Retrieving memories from laptop...")
    laptop_memories = manager.retrieve_memories(source="laptop")
    print(f"   Found {len(laptop_memories)} laptop memories")
    
    # Retrieve memories by time range
    print("\n5. Retrieving recent memories (last hour)...")
    recent = manager.retrieve_memories(
        start_time=datetime.utcnow() - timedelta(hours=1)
    )
    print(f"   Found {len(recent)} recent memories")
    
    # Retrieve memories by tags
    print("\n6. Retrieving memories with 'browser' tag...")
    tagged = manager.retrieve_memories(tags=["browser"])
    print(f"   Found {len(tagged)} tagged memories")
    
    # Get a specific memory
    print("\n7. Retrieving specific memory by ID...")
    specific = manager.get_memory(action_id)
    if specific:
        print(f"   Content: {specific.content}")
        print(f"   Metadata: {specific.metadata}")
        print(f"   Tags: {specific.tags}")
    
    # Get context summary
    print("\n8. Getting context summary...")
    summary = manager.summarize_context()
    print(f"   Total entries: {summary['total_entries']}")
    print(f"   By type: {summary['by_type']}")
    print(f"   By source: {summary['by_source']}")
    
    # Delete a memory
    print("\n9. Deleting a memory...")
    deleted = manager.delete_memory(conv_id)
    print(f"   Deletion successful: {deleted}")
    
    # Verify deletion
    remaining = manager.retrieve_memories()
    print(f"   Remaining memories: {len(remaining)}")
    
    # Cleanup
    print("\n10. Closing memory manager...")
    manager.close()
    print("Done!")


if __name__ == "__main__":
    main()
