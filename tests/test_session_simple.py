"""Simple test to verify Session_Manager works."""

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface."""
    
    def __init__(self):
        self.stored_memories = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({"id": memory_id, "content": content, "metadata": metadata or {}})
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        return True
    
    def delete(self, memory_id: str) -> bool:
        return True


def test_simple_session_creation():
    """Test that we can create a session and shut down cleanly."""
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create a session
        session_id = session_manager.create_session()
        
        # Verify it's a string
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Verify we can retrieve it
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        
        print(f"Test passed! Created session: {session_id}")
    finally:
        # Always shutdown
        session_manager.shutdown()
