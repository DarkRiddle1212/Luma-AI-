"""Test to verify _persist_buffered_memories works correctly."""

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface, MemoryStorageError


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing."""
    
    def __init__(self, fail_on_index=None):
        self.stored_memories = []
        self.fail_on_index = fail_on_index
        self.store_count = 0
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Store a memory and return a mock ID."""
        if self.fail_on_index is not None and self.store_count == self.fail_on_index:
            self.store_count += 1
            raise MemoryStorageError(f"Simulated storage failure at index {self.fail_on_index}")
        
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        self.store_count += 1
        return memory_id
    
    def retrieve(self, params: dict) -> dict:
        """Retrieve memories (not used in this test)."""
        return {"memories": []}


def test_persist_buffered_memories_on_end_session():
    """Test that buffered memories are persisted when session ends."""
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
        session_manager.buffer_memory(
            session_id=session_id,
            content="Test memory 3",
            metadata={"category": "test"}
        )
        
        # Verify memories are buffered, not persisted
        assert len(memory.stored_memories) == 0
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == 3
        
        # End session with persist=True
        persisted_count = session_manager.end_session(session_id, persist=True)
        
        # Verify all memories were persisted
        assert persisted_count == 3
        assert len(memory.stored_memories) == 3
        assert memory.stored_memories[0]["content"] == "Test memory 1"
        assert memory.stored_memories[1]["content"] == "Test memory 2"
        assert memory.stored_memories[2]["content"] == "Test memory 3"
        
        # Verify session is removed
        session = session_manager.get_session(session_id)
        assert session is None
        
        print("✓ Buffered memories persisted correctly on end_session")
        
    finally:
        session_manager.shutdown()


def test_persist_buffered_memories_graceful_error_handling():
    """Test that persistence continues even if some memories fail to store."""
    # Create mock that fails on the second memory
    memory = MockMemoryInterface(fail_on_index=1)
    
    config = SessionConfig(
        timeout_seconds=1800,
        cleanup_interval_seconds=300,
        max_buffer_size=100,
        enable_buffering=True
    )
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
            content="Test memory 2 (will fail)",
            metadata={"category": "test"}
        )
        session_manager.buffer_memory(
            session_id=session_id,
            content="Test memory 3",
            metadata={"category": "test"}
        )
        
        # End session with persist=True
        persisted_count = session_manager.end_session(session_id, persist=True)
        
        # Verify that 2 out of 3 memories were persisted (one failed)
        assert persisted_count == 2
        assert len(memory.stored_memories) == 2
        assert memory.stored_memories[0]["content"] == "Test memory 1"
        assert memory.stored_memories[1]["content"] == "Test memory 3"
        
        print("✓ Graceful error handling works: 2/3 memories persisted despite failure")
        
    finally:
        session_manager.shutdown()


def test_end_session_without_persist():
    """Test that buffered memories are discarded when persist=False."""
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
        
        # End session with persist=False
        persisted_count = session_manager.end_session(session_id, persist=False)
        
        # Verify no memories were persisted
        assert persisted_count == 0
        assert len(memory.stored_memories) == 0
        
        print("✓ Buffered memories discarded correctly when persist=False")
        
    finally:
        session_manager.shutdown()


def test_persist_clears_buffer():
    """Test that buffer is cleared after successful persistence."""
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
        
        # Get the session directly to check buffer
        session = session_manager.get_session(session_id)
        assert len(session.buffered_memories) == 1
        
        # Manually call _persist_buffered_memories (simulating what end_session does)
        with session_manager.lock:
            persisted_count = session_manager._persist_buffered_memories(session_id)
        
        # Verify buffer is cleared
        assert persisted_count == 1
        session = session_manager.get_session(session_id)
        assert len(session.buffered_memories) == 0
        
        print("✓ Buffer cleared after successful persistence")
        
    finally:
        session_manager.shutdown()


if __name__ == "__main__":
    test_persist_buffered_memories_on_end_session()
    test_persist_buffered_memories_graceful_error_handling()
    test_end_session_without_persist()
    test_persist_clears_buffer()
    print("\n✅ All _persist_buffered_memories tests passed!")
