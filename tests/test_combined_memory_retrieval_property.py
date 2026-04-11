"""
Property-Based Test for Combined Memory Retrieval

This module implements property-based tests using Hypothesis to verify
that the ReasoningEngine correctly combines retrieved long-term memories
and buffered session memories when building context during an active session.

Feature: memory-write-strategy-session-management
Property 8: Combined memory retrieval
Validates: Requirements 3.3, 7.3, 11.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any
from datetime import datetime, UTC

from luma.core.reasoning import ReasoningEngine
from luma.core.llm_interface import StubLLM
from luma.core.memory_interface import MemoryInterface
from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface that returns predefined long-term memories."""
    
    def __init__(self, long_term_memories: List[Dict[str, Any]] = None):
        self.long_term_memories = long_term_memories or []
        self.stored_memories = []
    
    def store(self, content: str, metadata: dict = None) -> str:
        """Mock store method."""
        memory_id = f"mem_{len(self.stored_memories)}"
        self.stored_memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {}
        })
        return memory_id
    
    def retrieve(self, params: dict = None) -> dict:
        """Mock retrieve method that returns predefined long-term memories."""
        return {"memories": self.long_term_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Helper Strategies
# ============================================================================

@st.composite
def memory_entry_strategy(draw, prefix: str = ""):
    """Generate valid memory entry dictionaries."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789"))
    content = draw(st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz "))
    category = draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"))
    tags = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=0,
        max_size=5
    ))
    
    return {
        "id": f"{prefix}{memory_id}",
        "content": f"{prefix}{content}",
        "metadata": {
            "category": category,
            "tags": tags
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "category": category,
        "tags": tags
    }


# ============================================================================
# Property 8: Combined Memory Retrieval
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    num_long_term=st.integers(min_value=0, max_value=20),
    num_session=st.integers(min_value=0, max_value=20),
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_combined_memory_retrieval(num_long_term, num_session, user_message):
    """
    Property: For any active session with both buffered session memories and
    persisted long-term memories, retrieval should return both types combined.
    
    **Validates: Requirements 3.3, 7.3, 11.5**
    
    This test verifies that:
    1. build_context() includes both long-term and session memories (Requirement 3.3)
    2. ReasoningEngine includes both newly stored and retrieved memories (Requirement 7.3)
    3. Session_Manager returns both buffered and persisted memories (Requirement 11.5)
    4. The combined list maintains the correct order (long-term first, then session)
    5. All memory entries are preserved with their metadata
    """
    # Generate long-term memories
    long_term_memories = []
    for i in range(num_long_term):
        long_term_memories.append({
            "id": f"lt_mem_{i}",
            "content": f"Long term memory {i}",
            "metadata": {"category": "long_term", "index": i},
            "timestamp": datetime.now(UTC).isoformat(),
            "category": "long_term",
            "tags": ["long_term"]
        })
    
    # Create mock memory with predefined long-term memories
    memory = MockMemoryInterface(long_term_memories=long_term_memories)
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create ReasoningEngine with session manager
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
        
        # Start a session
        session_id = engine.start_session()
        assert session_id is not None, "Session should be created"
        
        # Buffer session memories
        for i in range(num_session):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"Session memory {i}",
                metadata={"category": "session", "index": i}
            )
        
        # Build context with retrieved long-term memories
        context = engine.build_context(
            user_message=user_message,
            retrieved_memories=long_term_memories
        )
        
        # Property 1: Context should contain "memories" key
        assert "memories" in context, "Context must contain 'memories' key"
        
        # Property 2: Combined memories should include both long-term and session
        combined_memories = context["memories"]
        expected_total = num_long_term + num_session
        assert len(combined_memories) == expected_total, \
            f"Expected {expected_total} total memories ({num_long_term} long-term + {num_session} session), " \
            f"got {len(combined_memories)}"
        
        # Property 3: Long-term memories should come first
        for i in range(num_long_term):
            assert combined_memories[i]["id"] == f"lt_mem_{i}", \
                f"Memory at index {i} should be long-term memory {i}, got {combined_memories[i]['id']}"
            assert combined_memories[i]["content"] == f"Long term memory {i}", \
                f"Long-term memory {i} content mismatch"
            assert combined_memories[i]["metadata"]["category"] == "long_term", \
                f"Long-term memory {i} should have category 'long_term'"
        
        # Property 4: Session memories should come after long-term memories
        for i in range(num_session):
            session_memory_index = num_long_term + i
            assert combined_memories[session_memory_index]["content"] == f"Session memory {i}", \
                f"Session memory {i} content mismatch at index {session_memory_index}"
            assert combined_memories[session_memory_index]["metadata"]["category"] == "session", \
                f"Session memory {i} should have category 'session'"
        
        # Property 5: Context should include session_id
        assert context["session_id"] == session_id, \
            f"Context should include session_id {session_id}"
        
        # Property 6: Context should include other required keys
        assert "user_message" in context, "Context must contain 'user_message'"
        assert "timestamp" in context, "Context must contain 'timestamp'"
        assert "system_state_placeholder" in context, "Context must contain 'system_state_placeholder'"
        
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    num_long_term=st.integers(min_value=1, max_value=10),
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_no_session_only_long_term(num_long_term, user_message):
    """
    Property: When no session is active, build_context should only include
    long-term memories without session memories.
    
    **Validates: Requirements 3.3, 7.3**
    
    This test verifies that:
    1. Without an active session, only long-term memories are included
    2. The system handles the absence of session gracefully
    3. Context structure is still valid
    """
    # Generate long-term memories
    long_term_memories = []
    for i in range(num_long_term):
        long_term_memories.append({
            "id": f"lt_mem_{i}",
            "content": f"Long term memory {i}",
            "metadata": {"category": "long_term", "index": i},
            "timestamp": datetime.now(UTC).isoformat(),
            "category": "long_term",
            "tags": ["long_term"]
        })
    
    # Create mock memory
    memory = MockMemoryInterface(long_term_memories=long_term_memories)
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create ReasoningEngine WITHOUT starting a session
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
        
        # Build context without active session
        context = engine.build_context(
            user_message=user_message,
            retrieved_memories=long_term_memories
        )
        
        # Property 1: Context should contain only long-term memories
        combined_memories = context["memories"]
        assert len(combined_memories) == num_long_term, \
            f"Expected {num_long_term} long-term memories, got {len(combined_memories)}"
        
        # Property 2: All memories should be long-term memories
        for i in range(num_long_term):
            assert combined_memories[i]["id"] == f"lt_mem_{i}", \
                f"Memory at index {i} should be long-term memory {i}"
            assert combined_memories[i]["content"] == f"Long term memory {i}", \
                f"Long-term memory {i} content mismatch"
        
        # Property 3: session_id should be None
        assert context["session_id"] is None, \
            "Context session_id should be None when no session is active"
        
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    num_session=st.integers(min_value=1, max_value=10),
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_session_only_no_long_term(num_session, user_message):
    """
    Property: When an active session has buffered memories but no long-term
    memories are retrieved, build_context should only include session memories.
    
    **Validates: Requirements 3.3, 11.5**
    
    This test verifies that:
    1. Session memories are included even when no long-term memories exist
    2. The system handles empty long-term memory list gracefully
    3. Context structure is still valid
    """
    # Create mock memory with no long-term memories
    memory = MockMemoryInterface(long_term_memories=[])
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create ReasoningEngine with session manager
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
        
        # Start a session
        session_id = engine.start_session()
        
        # Buffer session memories
        for i in range(num_session):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"Session memory {i}",
                metadata={"category": "session", "index": i}
            )
        
        # Build context with no long-term memories
        context = engine.build_context(
            user_message=user_message,
            retrieved_memories=[]
        )
        
        # Property 1: Context should contain only session memories
        combined_memories = context["memories"]
        assert len(combined_memories) == num_session, \
            f"Expected {num_session} session memories, got {len(combined_memories)}"
        
        # Property 2: All memories should be session memories
        for i in range(num_session):
            assert combined_memories[i]["content"] == f"Session memory {i}", \
                f"Session memory {i} content mismatch"
            assert combined_memories[i]["metadata"]["category"] == "session", \
                f"Session memory {i} should have category 'session'"
        
        # Property 3: session_id should be set
        assert context["session_id"] == session_id, \
            f"Context should include session_id {session_id}"
        
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_no_memories_at_all(user_message):
    """
    Property: When no long-term memories are retrieved and no session is active,
    build_context should include an empty memories list.
    
    **Validates: Requirements 3.3, 7.3**
    
    This test verifies that:
    1. Empty memory list is handled gracefully
    2. Context structure is still valid with empty memories
    3. No errors occur when both memory sources are empty
    """
    # Create mock memory with no long-term memories
    memory = MockMemoryInterface(long_term_memories=[])
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create ReasoningEngine WITHOUT starting a session
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
        
        # Build context with no memories and no session
        context = engine.build_context(
            user_message=user_message,
            retrieved_memories=[]
        )
        
        # Property 1: Context should contain empty memories list
        combined_memories = context["memories"]
        assert isinstance(combined_memories, list), \
            "Memories should be a list"
        assert len(combined_memories) == 0, \
            f"Expected empty memories list, got {len(combined_memories)} memories"
        
        # Property 2: session_id should be None
        assert context["session_id"] is None, \
            "Context session_id should be None when no session is active"
        
        # Property 3: Other context keys should still be present
        assert "user_message" in context, "Context must contain 'user_message'"
        assert "timestamp" in context, "Context must contain 'timestamp'"
        assert "system_state_placeholder" in context, "Context must contain 'system_state_placeholder'"
        
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    num_long_term=st.integers(min_value=1, max_value=10),
    num_session=st.integers(min_value=1, max_value=10),
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_metadata_preservation(num_long_term, num_session, user_message):
    """
    Property: For any combined memory retrieval, all metadata fields from both
    long-term and session memories should be preserved exactly.
    
    **Validates: Requirements 3.3, 7.3, 11.5**
    
    This test verifies that:
    1. Metadata from long-term memories is preserved
    2. Metadata from session memories is preserved
    3. No metadata is lost or modified during combination
    4. Complex metadata structures are maintained
    """
    # Generate long-term memories with complex metadata
    long_term_memories = []
    for i in range(num_long_term):
        long_term_memories.append({
            "id": f"lt_mem_{i}",
            "content": f"Long term memory {i}",
            "metadata": {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 4)],
                "custom_field": f"value_{i}",
                "number": i * 10,
                "nested": {"key": f"nested_{i}"}
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "category": f"category_{i % 3}",
            "tags": [f"tag_{j}" for j in range(i % 4)]
        })
    
    # Create mock memory
    memory = MockMemoryInterface(long_term_memories=long_term_memories)
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create ReasoningEngine
        llm = StubLLM()
        engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
        
        # Start a session
        session_id = engine.start_session()
        
        # Buffer session memories with complex metadata
        for i in range(num_session):
            session_manager.buffer_memory(
                session_id=session_id,
                content=f"Session memory {i}",
                metadata={
                    "category": f"session_cat_{i % 2}",
                    "tags": [f"session_tag_{j}" for j in range(i % 3)],
                    "session_field": f"session_value_{i}",
                    "session_number": i * 5,
                    "session_nested": {"session_key": f"session_nested_{i}"}
                }
            )
        
        # Build context
        context = engine.build_context(
            user_message=user_message,
            retrieved_memories=long_term_memories
        )
        
        combined_memories = context["memories"]
        
        # Property 1: Verify long-term memory metadata is preserved
        for i in range(num_long_term):
            memory_entry = combined_memories[i]
            expected_metadata = {
                "category": f"category_{i % 3}",
                "tags": [f"tag_{j}" for j in range(i % 4)],
                "custom_field": f"value_{i}",
                "number": i * 10,
                "nested": {"key": f"nested_{i}"}
            }
            assert memory_entry["metadata"] == expected_metadata, \
                f"Long-term memory {i} metadata not preserved"
        
        # Property 2: Verify session memory metadata is preserved
        for i in range(num_session):
            memory_entry = combined_memories[num_long_term + i]
            expected_metadata = {
                "category": f"session_cat_{i % 2}",
                "tags": [f"session_tag_{j}" for j in range(i % 3)],
                "session_field": f"session_value_{i}",
                "session_number": i * 5,
                "session_nested": {"session_key": f"session_nested_{i}"}
            }
            assert memory_entry["metadata"] == expected_metadata, \
                f"Session memory {i} metadata not preserved"
        
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 8: Combined memory retrieval
@given(
    num_sessions=st.integers(min_value=2, max_value=5),
    memories_per_session=st.integers(min_value=1, max_value=10),
    user_message=st.text(min_size=5, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz ")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_8_session_isolation_in_retrieval(num_sessions, memories_per_session, user_message):
    """
    Property: For any set of concurrent sessions, each session's build_context
    should only include its own session memories, not memories from other sessions.
    
    **Validates: Requirements 3.3, 11.5**
    
    This test verifies that:
    1. Session memory retrieval is isolated per session
    2. One session's memories don't leak into another session's context
    3. Each session maintains independent memory state
    """
    # Create mock memory with no long-term memories
    memory = MockMemoryInterface(long_term_memories=[])
    
    # Create session manager
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create multiple ReasoningEngine instances (simulating different sessions)
        engines = []
        session_ids = []
        
        for i in range(num_sessions):
            llm = StubLLM()
            engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
            session_id = engine.start_session()
            engines.append(engine)
            session_ids.append(session_id)
            
            # Buffer unique memories for each session
            for j in range(memories_per_session):
                session_manager.buffer_memory(
                    session_id=session_id,
                    content=f"Session_{i}_Memory_{j}",
                    metadata={"session_index": i, "memory_index": j}
                )
        
        # Property 1: Each engine should only see its own session memories
        for i, engine in enumerate(engines):
            context = engine.build_context(
                user_message=user_message,
                retrieved_memories=[]
            )
            
            combined_memories = context["memories"]
            
            # Should have exactly memories_per_session memories
            assert len(combined_memories) == memories_per_session, \
                f"Session {i} should have {memories_per_session} memories, got {len(combined_memories)}"
            
            # All memories should belong to this session
            for j in range(memories_per_session):
                assert combined_memories[j]["content"] == f"Session_{i}_Memory_{j}", \
                    f"Session {i} memory {j} content mismatch"
                assert combined_memories[j]["metadata"]["session_index"] == i, \
                    f"Session {i} should only see its own memories"
        
    finally:
        session_manager.shutdown()
