"""
Property-Based Tests for Timestamp Attachment

This module implements property-based tests using Hypothesis to verify
that the Memory_Write_Strategy correctly attaches timestamps in ISO 8601
format to all stored memories.

Feature: memory-write-strategy-session-management
Property 14: Timestamp attachment
Validates: Requirements 6.1
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, UTC
import re

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
        self.default_category = None
        self.default_tags = []
    
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
# Helper Functions
# ============================================================================

def is_valid_iso8601_timestamp(timestamp_str: str) -> bool:
    """
    Validate that a string is a valid ISO 8601 timestamp.
    
    ISO 8601 format examples:
    - 2024-01-15T10:30:45.123456
    - 2024-01-15T10:30:45
    - 2024-01-15T10:30:45Z
    - 2024-01-15T10:30:45+00:00
    """
    # ISO 8601 regex pattern
    iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    
    if not re.match(iso8601_pattern, timestamp_str):
        return False
    
    # Try to parse it to ensure it's a valid datetime
    try:
        # Handle different ISO 8601 formats
        if timestamp_str.endswith('Z'):
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            datetime.fromisoformat(timestamp_str)
        return True
    except (ValueError, AttributeError):
        return False


# ============================================================================
# Property 14: Timestamp Attachment
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 14: Timestamp attachment
@given(
    metadata=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P')),
            min_size=1,
            max_size=20
        ).filter(lambda x: x != "timestamp"),  # Exclude "timestamp" key
        values=st.one_of(
            st.text(max_size=5),
            st.integers(),
            st.booleans(),
            st.lists(st.text(max_size=5), max_size=5)
        ),
        min_size=0,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_14_timestamp_attached_when_missing(metadata):
    """
    Property: For any metadata dictionary without a timestamp field,
    the normalized metadata should have a timestamp in ISO 8601 format attached.
    
    **Validates: Requirements 6.1**
    
    This test verifies that:
    1. A timestamp is automatically added when not present
    2. The timestamp is in valid ISO 8601 format
    3. The timestamp represents a valid datetime
    4. Other metadata fields are preserved
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Ensure metadata doesn't have timestamp
        test_metadata = dict(metadata)
        if "timestamp" in test_metadata:
            del test_metadata["timestamp"]
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(test_metadata)
        
        # Verify timestamp is present
        assert "timestamp" in normalized, \
            "Normalized metadata should contain a 'timestamp' field"
        
        # Verify timestamp is a string
        assert isinstance(normalized["timestamp"], str), \
            f"Timestamp should be a string, got {type(normalized['timestamp'])}"
        
        # Verify timestamp is in ISO 8601 format
        assert is_valid_iso8601_timestamp(normalized["timestamp"]), \
            f"Timestamp should be in ISO 8601 format, got: '{normalized['timestamp']}'"
        
        # Verify other metadata fields are preserved
        for key, value in test_metadata.items():
            if key != "timestamp":
                assert key in normalized, \
                    f"Original metadata key '{key}' should be preserved"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 14: Timestamp attachment
@given(
    existing_timestamp=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_14_existing_timestamp_preserved(existing_timestamp):
    """
    Property: For any metadata dictionary with an existing timestamp,
    the normalized metadata should preserve the original timestamp.
    
    **Validates: Requirements 6.1**
    
    This test verifies that:
    1. Existing timestamps are not overwritten
    2. User-provided timestamps take precedence
    3. The timestamp format is preserved
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata with existing timestamp
        timestamp_str = existing_timestamp.isoformat()
        metadata = {
            "timestamp": timestamp_str,
            "category": "test",
            "tags": ["test"]
        }
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify timestamp is preserved
        assert "timestamp" in normalized, \
            "Normalized metadata should contain a 'timestamp' field"
        
        assert normalized["timestamp"] == timestamp_str, \
            f"Existing timestamp should be preserved. Expected: '{timestamp_str}', Got: '{normalized['timestamp']}'"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 14: Timestamp attachment
@given(
    metadata_list=st.lists(
        st.dictionaries(
            keys=st.text(
                alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
                min_size=1,
                max_size=20
            ).filter(lambda x: x != "timestamp"),
            values=st.text(max_size=5),
            min_size=0,
            max_size=5
        ),
        min_size=2,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_14_unique_timestamps_for_sequential_calls(metadata_list):
    """
    Property: For any sequence of metadata normalization calls,
    each should receive a unique timestamp (or very close timestamps).
    
    **Validates: Requirements 6.1**
    
    This test verifies that:
    1. Each normalization call gets a timestamp
    2. Timestamps are in chronological order (or equal if very fast)
    3. All timestamps are valid ISO 8601 format
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        normalized_list = []
        
        # Normalize each metadata
        for metadata in metadata_list:
            test_metadata = dict(metadata)
            if "timestamp" in test_metadata:
                del test_metadata["timestamp"]
            
            normalized = strategy.normalize_metadata(test_metadata)
            normalized_list.append(normalized)
        
        # Verify all have timestamps
        for i, normalized in enumerate(normalized_list):
            assert "timestamp" in normalized, \
                f"Metadata at index {i} should have a timestamp"
            
            assert is_valid_iso8601_timestamp(normalized["timestamp"]), \
                f"Timestamp at index {i} should be in ISO 8601 format: '{normalized['timestamp']}'"
        
        # Verify timestamps are in chronological order (or equal)
        for i in range(len(normalized_list) - 1):
            ts1 = datetime.fromisoformat(normalized_list[i]["timestamp"])
            ts2 = datetime.fromisoformat(normalized_list[i + 1]["timestamp"])
            
            assert ts1 <= ts2, \
                f"Timestamps should be in chronological order. " \
                f"Index {i}: {ts1.isoformat()}, Index {i+1}: {ts2.isoformat()}"
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 14: Timestamp attachment
@given(
    content=st.text(min_size=10, max_size=100),
    category=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_14_timestamp_format_consistency(content, category):
    """
    Property: For any content and category, the timestamp format
    should be consistent across all normalizations.
    
    **Validates: Requirements 6.1**
    
    This test verifies that:
    1. Timestamp format is always ISO 8601
    2. Format is consistent regardless of input
    3. Timestamp can be parsed back to datetime
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Create metadata
        metadata = {
            "category": category,
            "content": content
        }
        
        # Normalize metadata
        normalized = strategy.normalize_metadata(metadata)
        
        # Verify timestamp exists and is valid
        assert "timestamp" in normalized, \
            "Normalized metadata should contain a 'timestamp' field"
        
        timestamp_str = normalized["timestamp"]
        
        # Verify it's a string
        assert isinstance(timestamp_str, str), \
            f"Timestamp should be a string, got {type(timestamp_str)}"
        
        # Verify ISO 8601 format
        assert is_valid_iso8601_timestamp(timestamp_str), \
            f"Timestamp should be in ISO 8601 format: '{timestamp_str}'"
        
        # Verify it can be parsed back to datetime
        try:
            parsed_dt = datetime.fromisoformat(timestamp_str)
            assert isinstance(parsed_dt, datetime), \
                "Timestamp should be parseable to datetime object"
        except (ValueError, AttributeError) as e:
            pytest.fail(f"Timestamp should be parseable: {e}")
    
    finally:
        session_manager.shutdown()


# Feature: memory-write-strategy-session-management, Property 14: Timestamp attachment
@given(
    metadata=st.dictionaries(
        keys=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ).filter(lambda x: x != "timestamp"),
        values=st.one_of(
            st.text(max_size=5),
            st.integers(),
            st.lists(st.text(max_size=5), max_size=5)
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_property_14_timestamp_represents_current_time(metadata):
    """
    Property: For any metadata without a timestamp, the attached timestamp
    should represent the current time (within a reasonable tolerance).
    
    **Validates: Requirements 6.1**
    
    This test verifies that:
    1. Timestamp represents approximately current time
    2. Timestamp is not from the past (more than 1 second ago)
    3. Timestamp is not from the future (more than 1 second ahead)
    """
    # Create write strategy
    config = WriteStrategyConfig()
    session_manager = Session_Manager(
        SessionConfig(),
        MockMemoryInterface()
    )
    memory = MockMemoryInterface()
    strategy = Memory_Write_Strategy(config, session_manager, memory)
    
    try:
        # Record time before normalization
        before_time = datetime.now(UTC)
        
        # Normalize metadata
        test_metadata = dict(metadata)
        if "timestamp" in test_metadata:
            del test_metadata["timestamp"]
        
        normalized = strategy.normalize_metadata(test_metadata)
        
        # Record time after normalization
        after_time = datetime.now(UTC)
        
        # Verify timestamp exists
        assert "timestamp" in normalized, \
            "Normalized metadata should contain a 'timestamp' field"
        
        # Parse the timestamp
        timestamp_str = normalized["timestamp"]
        timestamp_dt = datetime.fromisoformat(timestamp_str)
        
        # Verify timestamp is between before and after times (with 1 second tolerance)
        time_diff_before = (timestamp_dt - before_time).total_seconds()
        time_diff_after = (after_time - timestamp_dt).total_seconds()
        
        assert time_diff_before >= -1, \
            f"Timestamp should not be more than 1 second before normalization call. Diff: {time_diff_before}s"
        
        assert time_diff_after >= -1, \
            f"Timestamp should not be more than 1 second after normalization call. Diff: {time_diff_after}s"
    
    finally:
        session_manager.shutdown()
