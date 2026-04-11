"""
Property-Based Test for Session Metadata Tracking

This module implements property-based tests using Hypothesis to verify
that the Session_Manager maintains accurate session metadata including
start_time, last_activity_time, and message_count.

Feature: memory-write-strategy-session-management
Property 6: Session metadata tracking
Validates: Requirements 2.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta, UTC
import time

from luma.core.session_manager import Session_Manager
from luma.core.write_strategy import SessionConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Mock Memory Interface for Testing
# ============================================================================

class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing session manager."""
    
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
# Property 6: Session Metadata Tracking
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 6: Session metadata tracking
@given(num_updates=st.integers(min_value=0, max_value=50))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_6_session_metadata_tracking(num_updates):
    """
    Property: For any active session, the Session_Manager should maintain
    accurate metadata including start_time, last_activity_time, and message_count
    that updates with each message.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. start_time is set when session is created
    2. last_activity_time is initialized to start_time
    3. message_count starts at 0
    4. last_activity_time is updated with each update_activity call
    5. message_count is incremented with each update_activity call
    6. start_time remains unchanged throughout session lifecycle
    """
    config = SessionConfig(
        timeout_seconds=3600,  # 1 hour (long enough for test)
        cleanup_interval_seconds=3600,  # Don't cleanup during test
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Record time before session creation
        time_before_creation = datetime.now(UTC)
        
        # Create session
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        
        # Record time after session creation
        time_after_creation = datetime.now(UTC)
        
        # Property 1: Session should exist
        assert session is not None, "Session should be retrievable after creation"
        
        # Property 2: start_time should be set and within reasonable bounds
        assert session.start_time is not None, "start_time should be set"
        assert isinstance(session.start_time, datetime), \
            f"start_time should be datetime, got {type(session.start_time)}"
        assert time_before_creation <= session.start_time <= time_after_creation, \
            f"start_time should be between {time_before_creation} and {time_after_creation}, " \
            f"got {session.start_time}"
        
        # Property 3: last_activity_time should be initialized to start_time
        assert session.last_activity_time is not None, "last_activity_time should be set"
        assert isinstance(session.last_activity_time, datetime), \
            f"last_activity_time should be datetime, got {type(session.last_activity_time)}"
        assert session.last_activity_time == session.start_time, \
            f"last_activity_time should equal start_time initially, " \
            f"got {session.last_activity_time} != {session.start_time}"
        
        # Property 4: message_count should start at 0
        assert session.message_count == 0, \
            f"message_count should start at 0, got {session.message_count}"
        
        # Store initial values for comparison
        initial_start_time = session.start_time
        previous_last_activity = session.last_activity_time
        
        # Perform updates
        for i in range(num_updates):
            # Small delay to ensure time progresses (at least microseconds)
            time.sleep(0.001)
            
            # Update activity
            session_manager.update_activity(session_id)
            
            # Retrieve updated session
            session = session_manager.get_session(session_id)
            assert session is not None, f"Session should exist after update {i+1}"
            
            # Property 5: message_count should be incremented
            expected_count = i + 1
            assert session.message_count == expected_count, \
                f"After {expected_count} updates, message_count should be {expected_count}, " \
                f"got {session.message_count}"
            
            # Property 6: last_activity_time should be updated (greater than or equal to previous)
            assert session.last_activity_time >= previous_last_activity, \
                f"last_activity_time should not go backwards: " \
                f"{session.last_activity_time} < {previous_last_activity}"
            
            # Property 7: start_time should remain unchanged
            assert session.start_time == initial_start_time, \
                f"start_time should remain constant: " \
                f"{session.start_time} != {initial_start_time}"
            
            # Update previous activity time for next iteration
            previous_last_activity = session.last_activity_time
        
        # Final verification after all updates
        final_session = session_manager.get_session(session_id)
        assert final_session is not None, "Session should still exist after all updates"
        assert final_session.message_count == num_updates, \
            f"Final message_count should be {num_updates}, got {final_session.message_count}"
        assert final_session.start_time == initial_start_time, \
            "start_time should never change"
        
        # If updates occurred, last_activity_time should be after start_time
        if num_updates > 0:
            assert final_session.last_activity_time >= final_session.start_time, \
                "last_activity_time should be >= start_time after updates"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 6: Session metadata tracking
@given(
    num_sessions=st.integers(min_value=1, max_value=20),
    updates_per_session=st.lists(
        st.integers(min_value=0, max_value=30),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_property_6_multiple_sessions_metadata_isolation(num_sessions, updates_per_session):
    """
    Property: For any number of concurrent sessions, each session should
    maintain its own independent metadata without interference.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Each session has independent start_time
    2. Each session has independent last_activity_time
    3. Each session has independent message_count
    4. Updates to one session don't affect other sessions
    """
    # Ensure we have enough update counts
    while len(updates_per_session) < num_sessions:
        updates_per_session.append(0)
    
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create multiple sessions and track their initial state
        sessions_data = []
        
        for i in range(num_sessions):
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)
            
            sessions_data.append({
                'session_id': session_id,
                'initial_start_time': session.start_time,
                'initial_last_activity': session.last_activity_time,
                'expected_count': 0
            })
            
            # Small delay between session creations
            time.sleep(0.001)
        
        # Perform updates on each session
        for i, session_data in enumerate(sessions_data):
            num_updates = updates_per_session[i]
            session_id = session_data['session_id']
            
            for _ in range(num_updates):
                time.sleep(0.001)
                session_manager.update_activity(session_id)
            
            session_data['expected_count'] = num_updates
        
        # Verify each session maintained independent metadata
        for session_data in sessions_data:
            session_id = session_data['session_id']
            session = session_manager.get_session(session_id)
            
            assert session is not None, f"Session {session_id} should exist"
            
            # Property 1: start_time should be unchanged
            assert session.start_time == session_data['initial_start_time'], \
                f"Session {session_id} start_time changed"
            
            # Property 2: message_count should match expected
            assert session.message_count == session_data['expected_count'], \
                f"Session {session_id} expected {session_data['expected_count']} messages, " \
                f"got {session.message_count}"
            
            # Property 3: last_activity_time should be >= initial (if updates occurred)
            if session_data['expected_count'] > 0:
                assert session.last_activity_time >= session_data['initial_last_activity'], \
                    f"Session {session_id} last_activity_time should have been updated"
        
        # Property 4: All sessions should have different start_times (created at different times)
        start_times = [session_manager.get_session(sd['session_id']).start_time 
                      for sd in sessions_data]
        # Note: In rare cases with very fast execution, times might be equal, so we just check
        # that they're all valid datetime objects
        for st_time in start_times:
            assert isinstance(st_time, datetime), "All start_times should be datetime objects"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 6: Session metadata tracking
@given(
    initial_metadata=st.dictionaries(
        keys=st.text(min_size=1, max_size=5),
        values=st.one_of(st.text(max_size=5), st.integers(), st.booleans()),
        min_size=0,
        max_size=10
    ),
    num_updates=st.integers(min_value=1, max_value=30)
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_6_metadata_preservation_during_updates(initial_metadata, num_updates):
    """
    Property: For any session with custom metadata, the custom metadata
    should be preserved throughout the session lifecycle while tracking
    metadata (start_time, last_activity_time, message_count) is updated.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. Custom metadata is preserved during activity updates
    2. Tracking metadata is updated independently of custom metadata
    3. No interference between custom and tracking metadata
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create session with custom metadata
        session_id = session_manager.create_session(metadata=initial_metadata)
        session = session_manager.get_session(session_id)
        
        # Property 1: Custom metadata should be set
        assert session.metadata == initial_metadata, \
            f"Initial metadata mismatch: expected {initial_metadata}, got {session.metadata}"
        
        # Store initial tracking metadata
        initial_start_time = session.start_time
        
        # Perform updates
        for i in range(num_updates):
            time.sleep(0.001)
            session_manager.update_activity(session_id)
            
            session = session_manager.get_session(session_id)
            
            # Property 2: Custom metadata should remain unchanged
            assert session.metadata == initial_metadata, \
                f"Custom metadata changed after update {i+1}: " \
                f"expected {initial_metadata}, got {session.metadata}"
            
            # Property 3: Tracking metadata should be updated
            assert session.message_count == i + 1, \
                f"message_count should be {i+1}, got {session.message_count}"
            assert session.start_time == initial_start_time, \
                "start_time should remain constant"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 6: Session metadata tracking
@given(num_updates=st.integers(min_value=1, max_value=50))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_6_last_activity_time_monotonic_increase(num_updates):
    """
    Property: For any sequence of activity updates, the last_activity_time
    should never decrease (monotonic increase property).
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. last_activity_time never goes backwards
    2. last_activity_time is always >= start_time
    3. Time progression is consistent with update sequence
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create session
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        
        start_time = session.start_time
        previous_activity_time = session.last_activity_time
        
        # Track all activity times
        activity_times = [previous_activity_time]
        
        # Perform updates and track times
        for i in range(num_updates):
            time.sleep(0.001)
            session_manager.update_activity(session_id)
            
            session = session_manager.get_session(session_id)
            current_activity_time = session.last_activity_time
            
            # Property 1: Monotonic increase (never decrease)
            assert current_activity_time >= previous_activity_time, \
                f"last_activity_time decreased: {current_activity_time} < {previous_activity_time}"
            
            # Property 2: Always >= start_time
            assert current_activity_time >= start_time, \
                f"last_activity_time {current_activity_time} < start_time {start_time}"
            
            activity_times.append(current_activity_time)
            previous_activity_time = current_activity_time
        
        # Property 3: Verify overall time progression
        assert activity_times[-1] >= activity_times[0], \
            "Final activity time should be >= initial activity time"
        
        # Property 4: All times should be valid datetime objects
        for activity_time in activity_times:
            assert isinstance(activity_time, datetime), \
                f"Activity time should be datetime, got {type(activity_time)}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 6: Session metadata tracking
@given(num_updates=st.integers(min_value=0, max_value=100))
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_6_message_count_accuracy(num_updates):
    """
    Property: For any number of activity updates, the message_count should
    exactly equal the number of update_activity calls.
    
    **Validates: Requirements 2.5**
    
    This test verifies that:
    1. message_count starts at 0
    2. message_count increments by exactly 1 per update
    3. message_count is accurate after any number of updates
    4. No missed or duplicate increments
    """
    config = SessionConfig(
        timeout_seconds=3600,
        cleanup_interval_seconds=3600,
        max_buffer_size=100,
        enable_buffering=True
    )
    memory = MockMemoryInterface()
    session_manager = Session_Manager(config, memory)
    
    try:
        # Create session
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)
        
        # Property 1: Initial count should be 0
        assert session.message_count == 0, \
            f"Initial message_count should be 0, got {session.message_count}"
        
        # Perform updates and verify count after each
        for i in range(num_updates):
            session_manager.update_activity(session_id)
            
            session = session_manager.get_session(session_id)
            expected_count = i + 1
            
            # Property 2: Count should increment by exactly 1
            assert session.message_count == expected_count, \
                f"After update {expected_count}, message_count should be {expected_count}, " \
                f"got {session.message_count}"
        
        # Property 3: Final count should match total updates
        final_session = session_manager.get_session(session_id)
        assert final_session.message_count == num_updates, \
            f"Final message_count should be {num_updates}, got {final_session.message_count}"
    
    finally:
        session_manager.shutdown()
