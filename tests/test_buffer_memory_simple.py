"""Simple test to verify buffer_memory and get_session_memories work."""

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self):
        self.stored_memories = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Store a memory and return a mock ID."""
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict) -> dict:
        """Retrieve memories (not used in this test)."""
        return {"memories": []}


def test_buffer_memory_basic():
    """Test that buffer_memory adds memories to session buffer."""
    config = SessionConfig(
        timeout_seconds=1800,
        cleanup_interval_seconds=300,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer some memories
        session_manager.buffer_memory(
            session_id=session_id,
            content="Test memory 1",
            metadata={"category": "test"}
        )
        session_manager.buffer_memory(
            session_id=session_id,
            content="Test memory 2",
            metadata={"category": "test"}
        )
        
        # Get buffered memories
        buffered = session_manager.get_session_memories(session_id)
        
        # Verify
        assert len(buffered) == 2
        assert buffered[0]["content"] == "Test memory 1"
        assert buffered[1]["content"] == "Test memory 2"
        assert "buffered_at" in buffered[0]
        
        # Verify memories are NOT persisted yet
        assert len(memory.stored_memories) == 0
        
        print("✓ buffer_memory and get_session_memories work correctly")
        
    finally:
        session_manager.shutdown()


def test_buffer_overflow():
    """Test that buffer overflow triggers flushing of oldest memories."""
    config = SessionConfig(
        timeout_seconds=1800,
        cleanup_interval_seconds=300,
        max_buffer_size=5,  # Small buffer to trigger overflow
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Buffer more memories than max_buffer_size
        for i in range(7):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"Test memory {i}",
                metadata={"category": "test"}
            )
        
        # Get buffered memories
        buffered = session_manager.get_session_memories(session_id)
        
        # Verify buffer size is controlled (should have flushed oldest half)
        # After adding 7 memories to a buffer of max 5:
        # - 6th memory triggers overflow, flushes 3 oldest (half of 6), leaves 3
        # - 7th memory added, buffer now has 4
        assert len(buffered) <= 5, f"Buffer should be controlled, got {len(buffered)}"
        
        # Verify some memories were persisted
        assert len(memory.stored_memories) > 0, "Oldest memories should be flushed to storage"
        
        print(f"✓ Buffer overflow handled correctly: {len(memory.stored_memories)} flushed, {len(buffered)} buffered")
        
    finally:
        session_manager.shutdown()


def test_get_session_memories_nonexistent():
    """Test that get_session_memories returns empty list for nonexistent session."""
    config = SessionConfig(
        timeout_seconds=1800,
        cleanup_interval_seconds=300,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Try to get memories for nonexistent session
        buffered = session_manager.get_session_memories("nonexistent-session-id")
        
        # Verify empty list returned
        assert buffered == []
        
        print("✓ get_session_memories returns empty list for nonexistent session")
        
    finally:
        session_manager.shutdown()


if __name__ == "__main__":
    test_buffer_memory_basic()
    test_buffer_overflow()
    test_get_session_memories_nonexistent()
    print("\n✅ All buffer memory tests passed!")
