"""
Property-Based Test for Immediate Persistence Bypass

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly bypasses session buffering when
immediate persistence is required.

Feature: memory-write-strategy-session-management
Property 9: Immediate persistence bypass
Validates: Requirements 3.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any
import uuid

from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig
from luma.core.session_manager import Session_Manager, SessionConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing write strategy."""
    
    def __init__(self):
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
        """Mock retrieve method."""
        return {"memories": self.stored_memories}
    
    def update(self, memory_id: str, content: str = None, metadata: dict = None) -> bool:
        """Mock update method."""
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Mock delete method."""
        return True


# ============================================================================
# Property 9: Immediate Persistence Bypass
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    content=st.text(min_size=10, max_size=200, alphabet=st.characters(blacklist_categories=('Cs', 'Cc'))),
    category=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_immediate_flag_bypasses_buffering(content, category):
    """
    Property: For any message marked for immediate persistence (immediate=True),
    it should be written directly to long-term storage bypassing session buffering.
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. When immediate=True, memory is persisted directly to storage
    2. Memory is NOT added to session buffer
    3. A real memory_id is returned (not a buffer_id)
    4. This behavior occurs even when an active session exists
    """
    # Setup
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=[],
        similarity_threshold=0.9,
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create an active session
        session_id = session_manager.create_session()
        session_manager.current_session_id = session_id
        
        # Store memory with immediate=True
        metadata = {"category": category}
        memory_id = write_strategy.store_memory(
            content=content,
            metadata=metadata,
            immediate=True
        )
        
        # Property 1: Memory should be persisted immediately
        assert len(memory.stored_memories) == 1, \
            f"Expected 1 persisted memory with immediate=True, got {len(memory.stored_memories)}"
        
        # Property 2: Memory should NOT be in session buffer
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == 0, \
            f"Expected 0 buffered memories with immediate=True, got {len(buffered)}"
        
        # Property 3: Returned ID should be a real memory_id (not buffer_id)
        assert not memory_id.startswith("buffered:"), \
            f"Expected real memory_id with immediate=True, got buffer_id: {memory_id}"
        assert memory_id.startswith("mem_"), \
            f"Expected memory_id format 'mem_*', got: {memory_id}"
        
        # Property 4: Persisted memory should have correct content
        assert memory.stored_memories[0]["content"] == content, \
            f"Persisted memory content mismatch"
        
        # Property 5: Persisted memory should have normalized category
        stored_category = memory.stored_memories[0]["metadata"].get("category", "")
        assert stored_category == category.strip().lower(), \
            f"Expected normalized category '{category.strip().lower()}', got '{stored_category}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    num_memories=st.integers(min_value=1, max_value=20),
    content_prefix=st.text(min_size=5, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_immediate_pattern_bypasses_buffering(num_memories, content_prefix):
    """
    Property: For any message matching immediate_persist_patterns,
    it should be written directly to long-term storage bypassing session buffering.
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. Messages matching immediate_persist_patterns are persisted immediately
    2. Messages NOT matching patterns are buffered normally
    3. Pattern matching is case-insensitive
    """
    # Setup with immediate persist patterns
    immediate_patterns = ["urgent", "critical", "important"]
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=immediate_patterns,
        similarity_threshold=1.0,  # Disable near-duplicate detection for test isolation
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create an active session
        session_id = session_manager.create_session()
        session_manager.current_session_id = session_id
        
        # Store memories - some with immediate patterns, some without
        immediate_count = 0
        buffered_count = 0
        
        for i in range(num_memories):
            # Alternate between immediate and buffered
            if i % 2 == 0:
                # Include immediate pattern
                content = f"{content_prefix} urgent message {i} {uuid.uuid4()}"
                immediate_count += 1
            else:
                # No immediate pattern
                content = f"{content_prefix} normal message {i} {uuid.uuid4()}"
                buffered_count += 1
            
            memory_id = write_strategy.store_memory(
                content=content,
                metadata={"category": "test", "index": i}
            )
        
        # Property 1: Immediate pattern messages should be persisted
        assert len(memory.stored_memories) == immediate_count, \
            f"Expected {immediate_count} persisted memories, got {len(memory.stored_memories)}"
        
        # Property 2: Non-immediate messages should be buffered
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == buffered_count, \
            f"Expected {buffered_count} buffered memories, got {len(buffered)}"
        
        # Property 3: Persisted memories should contain immediate pattern
        for stored_memory in memory.stored_memories:
            content_lower = stored_memory["content"].lower()
            has_pattern = any(pattern in content_lower for pattern in immediate_patterns)
            assert has_pattern, \
                f"Persisted memory should contain immediate pattern: {stored_memory['content']}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    content=st.text(min_size=10, max_size=100, alphabet=st.characters(blacklist_categories=('Cs', 'Cc'))),
    category=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_no_session_forces_immediate_persistence(content, category):
    """
    Property: For any message stored when no active session exists,
    it should be written directly to long-term storage (immediate persistence).
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. Without an active session, all memories are persisted immediately
    2. No buffering occurs when no session is active
    3. Real memory_id is returned
    """
    # Setup
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=[],
        similarity_threshold=0.9,
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # DO NOT create a session - test without active session
        
        # Store memory without active session
        metadata = {"category": category}
        memory_id = write_strategy.store_memory(
            content=content,
            metadata=metadata,
            immediate=False  # Even with immediate=False, should persist due to no session
        )
        
        # Property 1: Memory should be persisted immediately (no session)
        assert len(memory.stored_memories) == 1, \
            f"Expected 1 persisted memory without active session, got {len(memory.stored_memories)}"
        
        # Property 2: Returned ID should be a real memory_id
        assert not memory_id.startswith("buffered:"), \
            f"Expected real memory_id without session, got buffer_id: {memory_id}"
        assert memory_id.startswith("mem_"), \
            f"Expected memory_id format 'mem_*', got: {memory_id}"
        
        # Property 3: Persisted memory should have correct content
        assert memory.stored_memories[0]["content"] == content, \
            f"Persisted memory content mismatch"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    num_immediate=st.integers(min_value=1, max_value=10),
    num_buffered=st.integers(min_value=1, max_value=10),
    content=st.text(min_size=10, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_mixed_immediate_and_buffered(num_immediate, num_buffered, content):
    """
    Property: For any mix of immediate and buffered storage operations,
    immediate memories should be persisted and buffered memories should be buffered.
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. Immediate and buffered operations can coexist
    2. Each type is handled correctly
    3. Counts match expectations
    """
    # Setup
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=[],
        similarity_threshold=1.0,  # Disable near-duplicate detection for test isolation
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create an active session
        session_id = session_manager.create_session()
        session_manager.current_session_id = session_id
        
        # Store immediate memories
        immediate_ids: List[str] = []
        for i in range(num_immediate):
            memory_id = write_strategy.store_memory(
                content=f"{content}_immediate_{i}_{uuid.uuid4()}",
                metadata={"category": "immediate", "index": i},
                immediate=True
            )
            immediate_ids.append(memory_id)
        
        # Store buffered memories
        buffered_ids: List[str] = []
        for i in range(num_buffered):
            memory_id = write_strategy.store_memory(
                content=f"{content}_buffered_{i}_{uuid.uuid4()}",
                metadata={"category": "buffered", "index": i},
                immediate=False
            )
            buffered_ids.append(memory_id)
        
        # Property 1: Immediate memories should be persisted
        assert len(memory.stored_memories) == num_immediate, \
            f"Expected {num_immediate} persisted memories, got {len(memory.stored_memories)}"
        
        # Property 2: Buffered memories should be in session buffer
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == num_buffered, \
            f"Expected {num_buffered} buffered memories, got {len(buffered)}"
        
        # Property 3: Immediate IDs should be real memory_ids
        for memory_id in immediate_ids:
            assert not memory_id.startswith("buffered:"), \
                f"Immediate memory should have real ID, got: {memory_id}"
        
        # Property 4: Buffered IDs should be buffer_ids
        for memory_id in buffered_ids:
            assert memory_id.startswith("buffered:"), \
                f"Buffered memory should have buffer ID, got: {memory_id}"
        
        # Property 5: Persisted memories should have "immediate" category
        for stored_memory in memory.stored_memories:
            assert stored_memory["metadata"]["category"] == "immediate", \
                f"Persisted memory should have 'immediate' category"
        
        # Property 6: Buffered memories should have "buffered" category
        for buffered_memory in buffered:
            assert buffered_memory["metadata"]["category"] == "buffered", \
                f"Buffered memory should have 'buffered' category"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    pattern=st.sampled_from(["URGENT", "Critical", "ImPoRtAnT"]),
    content=st.text(min_size=10, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_pattern_matching_case_insensitive(pattern, content):
    """
    Property: For any immediate_persist_pattern, matching should be case-insensitive.
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. Pattern matching is case-insensitive
    2. "URGENT", "urgent", "Urgent" all trigger immediate persistence
    """
    # Setup with lowercase patterns
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=["urgent", "critical", "important"],
        similarity_threshold=0.9,
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create an active session
        session_id = session_manager.create_session()
        session_manager.current_session_id = session_id
        
        # Store memory with mixed-case pattern
        message_content = f"{content} {pattern} message"
        memory_id = write_strategy.store_memory(
            content=message_content,
            metadata={"category": "test"}
        )
        
        # Property 1: Memory should be persisted immediately (case-insensitive match)
        assert len(memory.stored_memories) == 1, \
            f"Expected 1 persisted memory with pattern '{pattern}', got {len(memory.stored_memories)}"
        
        # Property 2: Memory should NOT be in session buffer
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == 0, \
            f"Expected 0 buffered memories with pattern '{pattern}', got {len(buffered)}"
        
        # Property 3: Returned ID should be a real memory_id
        assert not memory_id.startswith("buffered:"), \
            f"Expected real memory_id with pattern '{pattern}', got buffer_id: {memory_id}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 9: Immediate persistence bypass
@given(
    num_operations=st.integers(min_value=1, max_value=20),
    content=st.text(min_size=10, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz")
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_9_immediate_persistence_idempotent(num_operations, content):
    """
    Property: For any number of immediate persistence operations,
    each should be persisted independently without affecting others.
    
    **Validates: Requirements 3.4**
    
    This test verifies that:
    1. Multiple immediate persistence operations work correctly
    2. Each operation is independent
    3. All memories are persisted
    """
    # Setup
    write_config = WriteStrategyConfig(
        trivial_patterns=["hello", "hi"],
        min_content_length=3,
        repetition_window=5,
        immediate_persist_patterns=[],
        similarity_threshold=1.0,  # Disable near-duplicate detection for test isolation
        enable_conflict_detection=False
    )
    session_config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(session_config, memory)
    write_strategy = Memory_Write_Strategy(write_config, session_manager, memory)
    
    try:
        # Create an active session
        session_id = session_manager.create_session()
        session_manager.current_session_id = session_id
        
        # Store multiple memories with immediate=True
        memory_ids: List[str] = []
        for i in range(num_operations):
            memory_id = write_strategy.store_memory(
                content=f"{content}_{i}_{uuid.uuid4()}",
                metadata={"category": "test", "index": i},
                immediate=True
            )
            memory_ids.append(memory_id)
        
        # Property 1: All memories should be persisted
        assert len(memory.stored_memories) == num_operations, \
            f"Expected {num_operations} persisted memories, got {len(memory.stored_memories)}"
        
        # Property 2: No memories should be buffered
        buffered = session_manager.get_session_memories(session_id)
        assert len(buffered) == 0, \
            f"Expected 0 buffered memories with immediate=True, got {len(buffered)}"
        
        # Property 3: All IDs should be unique
        assert len(set(memory_ids)) == num_operations, \
            f"Expected {num_operations} unique memory IDs, got {len(set(memory_ids))}"
        
        # Property 4: All IDs should be real memory_ids
        for memory_id in memory_ids:
            assert not memory_id.startswith("buffered:"), \
                f"Expected real memory_id, got buffer_id: {memory_id}"
    
    finally:
        session_manager.shutdown()
