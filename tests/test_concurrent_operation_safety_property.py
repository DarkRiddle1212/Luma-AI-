"""
Property-Based Test for Concurrent Operation Safety

This module implements Property 7: Concurrent Operation Safety using Hypothesis to verify
that the system handles concurrent memory operations safely without data corruption,
deadlocks, or race conditions.

Feature: intent-based-memory-retrieval-enhancements
Task: 7.1 Write property test for concurrent operation safety
Property: 7 - Concurrent Operation Safety
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from typing import List, Tuple, Any
import threading
import time

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import QueryParameters


# ============================================================================
# Helper Strategies for Generating Test Data
# ============================================================================

@st.composite
def store_operation(draw):
    """Generate a store operation with random content and metadata."""
    content = draw(st.text(min_size=1, max_size=100))
    category = draw(st.sampled_from(["general", "education", "work", "personal", "system"]))
    tags = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        min_size=0,
        max_size=3,
        unique=True
    ))
    
    metadata = {
        "category": category,
        "tags": tags
    }
    
    return ("store", content, metadata)


@st.composite
def retrieve_operation(draw):
    """Generate a retrieve operation with random query parameters."""
    # Generate optional query string
    query = None
    if draw(st.booleans()):
        query = draw(st.text(min_size=1, max_size=5))
    
    # Generate optional category filter
    category = None
    if draw(st.booleans()):
        category = draw(st.sampled_from(["general", "education", "work", "personal", "system"]))
    
    # Generate optional tags filter
    tags = None
    if draw(st.booleans()):
        tags = draw(st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            min_size=1,
            max_size=2,
            unique=True
        ))
    
    # Build params
    params: QueryParameters = {"limit": 10}
    if query:
        params["query"] = query
    if category:
        params["category"] = category
    if tags:
        params["tags"] = tags
    
    return ("retrieve", params)


@st.composite
def operation_sequence(draw):
    """Generate a sequence of mixed store and retrieve operations."""
    num_operations = draw(st.integers(min_value=10, max_value=30))
    operations = []
    
    for _ in range(num_operations):
        # 60% store, 40% retrieve to ensure we have data to retrieve
        if draw(st.booleans()) or draw(st.booleans()):
            operations.append(draw(store_operation()))
        else:
            operations.append(draw(retrieve_operation()))
    
    return operations


# ============================================================================
# Helper Functions
# ============================================================================

def create_thread_safe_mock_memory_manager():
    """
    Create a mock MemoryManager that simulates thread-safe behavior.
    
    This mock uses locks to ensure thread-safe access to its internal state,
    simulating what a real MemoryManager would do.
    """
    mock_manager = MagicMock()
    
    # Thread-safe storage for created memories
    memories = []
    memory_lock = threading.Lock()
    memory_counter = [0]  # Use list to allow modification in nested function
    
    def create_memory_impl(action, context, device_id, tags):
        """Thread-safe create_memory implementation."""
        with memory_lock:
            memory_counter[0] += 1
            memory_id = f"memory-{memory_counter[0]}"
            
            # Create mock entry
            mock_entry = Mock()
            mock_entry.id = memory_id
            mock_entry.action = action
            mock_entry.context = context
            mock_entry.tags = tags
            mock_entry.device_id = device_id
            mock_entry.created_at = datetime.now()
            mock_entry.timestamp = datetime.now()
            
            memories.append(mock_entry)
            
            # Simulate some processing time
            time.sleep(0.001)
            
            return memory_id
    
    def query_memories_impl(action_type=None, start_time=None, end_time=None, tags=None, limit=10):
        """Thread-safe query_memories implementation."""
        with memory_lock:
            # Filter memories based on criteria
            results = []
            
            for entry in memories:
                # Check action_type filter
                if action_type and action_type not in entry.action:
                    continue
                
                # Check timestamp filters
                if start_time and entry.created_at < start_time:
                    continue
                if end_time and entry.created_at > end_time:
                    continue
                
                # Check tags filter (must contain all specified tags)
                if tags:
                    entry_tags = set(entry.tags or [])
                    required_tags = set(tags)
                    if not required_tags.issubset(entry_tags):
                        continue
                
                results.append(entry)
                
                if len(results) >= limit:
                    break
            
            # Simulate some processing time
            time.sleep(0.001)
            
            return results
    
    mock_manager.create_memory.side_effect = create_memory_impl
    mock_manager.query_memories.side_effect = query_memories_impl
    
    # Store references for verification
    mock_manager._test_memories = memories
    mock_manager._test_lock = memory_lock
    
    return mock_manager


def execute_operation(adapter: SQLiteMemoryAdapter, operation: Tuple[str, ...]) -> Tuple[bool, Any, str]:
    """
    Execute a single operation (store or retrieve) and return result.
    
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        op_type = operation[0]
        
        if op_type == "store":
            _, content, metadata = operation
            memory_id = adapter.store(content, metadata)
            return (True, memory_id, "")
        
        elif op_type == "retrieve":
            _, params = operation
            result = adapter.retrieve(params=params)
            return (True, result, "")
        
        else:
            return (False, None, f"Unknown operation type: {op_type}")
    
    except Exception as e:
        return (False, None, str(e))


# ============================================================================
# Property 7: Concurrent Operation Safety
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 7: Concurrent Operation Safety
@given(operations=operation_sequence())
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_concurrent_operation_safety_property(operations):
    """
    Property 7: Concurrent Operation Safety
    
    For any set of concurrent memory operations (stores and retrieves), all operations
    must complete successfully without data corruption, deadlocks, or race conditions,
    and each retrieve operation must return correct results.
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    
    This property verifies that:
    1. Multiple threads can store memories simultaneously without corruption (Req 7.1)
    2. Multiple threads can retrieve memories simultaneously with correct results (Req 7.2)
    3. Mixed store/retrieve operations maintain data consistency (Req 7.3)
    4. No deadlocks occur during concurrent operations (Req 7.4)
    5. No race conditions occur during concurrent operations (Req 7.5)
    
    Test Strategy:
    - Generate random sequences of store and retrieve operations
    - Execute operations concurrently using ThreadPoolExecutor
    - Use timeouts to detect deadlocks
    - Verify all operations complete successfully
    - Verify data integrity after concurrent operations
    """
    # Create thread-safe mock MemoryManager
    mock_memory_manager = create_thread_safe_mock_memory_manager()
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(
        mock_memory_manager,
        device_id="test-device",
        default_category="general"
    )
    
    # Track results
    results = []
    errors = []
    
    # Execute operations concurrently
    max_workers = min(10, len(operations))  # Limit concurrent threads
    timeout_seconds = 30  # Timeout to detect deadlocks (Requirement 7.4)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all operations
            futures = {
                executor.submit(execute_operation, adapter, op): op
                for op in operations
            }
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=timeout_seconds):
                operation = futures[future]
                try:
                    success, result, error_msg = future.result(timeout=5)
                    
                    if success:
                        results.append((operation, result))
                    else:
                        errors.append((operation, error_msg))
                
                except TimeoutError:
                    errors.append((operation, "Operation timed out - possible deadlock"))
                except Exception as e:
                    errors.append((operation, f"Unexpected error: {str(e)}"))
    
    except TimeoutError:
        pytest.fail("Concurrent operations timed out - possible deadlock detected (Requirement 7.4)")
    
    # PROPERTY VERIFICATION
    
    # Requirement 7.4: No deadlocks - all operations should complete
    assert len(errors) == 0, \
        f"Some operations failed or timed out:\n" + \
        "\n".join([f"  {op}: {err}" for op, err in errors[:5]])  # Show first 5 errors
    
    # Requirement 7.1, 7.2, 7.3: All operations completed successfully
    assert len(results) == len(operations), \
        f"Expected {len(operations)} successful operations, got {len(results)}"
    
    # Requirement 7.5: No race conditions - verify data integrity
    # Count store operations
    store_count = sum(1 for op in operations if op[0] == "store")
    
    # Verify that all stored memories are in the mock manager
    with mock_memory_manager._test_lock:
        actual_memory_count = len(mock_memory_manager._test_memories)
    
    assert actual_memory_count == store_count, \
        f"Expected {store_count} memories stored, but found {actual_memory_count}. " \
        f"This indicates data corruption or race condition (Requirement 7.5)"
    
    # Verify retrieve operations returned valid results
    retrieve_results = [r for op, r in results if op[0] == "retrieve"]
    for result in retrieve_results:
        # Each retrieve result should be a valid RetrievalResult
        assert "memories" in result, "Retrieve result missing 'memories' field"
        assert "total_count" in result, "Retrieve result missing 'total_count' field"
        assert "query_metadata" in result, "Retrieve result missing 'query_metadata' field"
        assert isinstance(result["memories"], list), "Memories should be a list"
        assert isinstance(result["total_count"], int), "Total count should be an integer"
        assert result["total_count"] >= 0, "Total count should be non-negative"


# ============================================================================
# Additional Concurrency Property Tests
# ============================================================================

# Feature: intent-based-memory-retrieval-enhancements, Property 7: Concurrent Operation Safety
@given(
    num_stores=st.integers(min_value=10, max_value=30),
    content=st.text(min_size=1, max_size=5)
)
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_concurrent_stores_property(num_stores, content):
    """
    Property: Multiple threads storing memories simultaneously complete without corruption.
    
    **Validates: Requirement 7.1**
    """
    mock_memory_manager = create_thread_safe_mock_memory_manager()
    adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
    
    stored_ids = []
    errors = []
    
    def store_memory(index):
        try:
            memory_id = adapter.store(f"{content}-{index}", {"category": "test"})
            stored_ids.append(memory_id)
        except Exception as e:
            errors.append(str(e))
    
    # Execute concurrent stores
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(store_memory, i) for i in range(num_stores)]
        for future in as_completed(futures, timeout=30):
            future.result()  # Wait for completion
    
    # Verify no errors
    assert len(errors) == 0, f"Errors during concurrent stores: {errors}"
    
    # Verify all stores completed
    assert len(stored_ids) == num_stores, \
        f"Expected {num_stores} stores, got {len(stored_ids)}"
    
    # Verify data integrity
    with mock_memory_manager._test_lock:
        assert len(mock_memory_manager._test_memories) == num_stores


# Feature: intent-based-memory-retrieval-enhancements, Property 7: Concurrent Operation Safety
@given(num_retrievals=st.integers(min_value=10, max_value=30))
@settings(max_examples=10, deadline=None)
@pytest.mark.property_test
def test_concurrent_retrievals_property(num_retrievals):
    """
    Property: Multiple threads retrieving memories simultaneously return correct results.
    
    **Validates: Requirement 7.2**
    """
    mock_memory_manager = create_thread_safe_mock_memory_manager()
    adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
    
    # Pre-populate with some memories
    for i in range(10):
        adapter.store(f"test content {i}", {"category": "test", "tags": ["test"]})
    
    results = []
    errors = []
    
    def retrieve_memories():
        try:
            result = adapter.retrieve(params={"limit": 10})
            results.append(result)
        except Exception as e:
            errors.append(str(e))
    
    # Execute concurrent retrievals
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(retrieve_memories) for _ in range(num_retrievals)]
        for future in as_completed(futures, timeout=30):
            future.result()  # Wait for completion
    
    # Verify no errors
    assert len(errors) == 0, f"Errors during concurrent retrievals: {errors}"
    
    # Verify all retrievals completed
    assert len(results) == num_retrievals, \
        f"Expected {num_retrievals} retrievals, got {len(results)}"
    
    # Verify all results are valid and consistent
    for result in results:
        assert "memories" in result
        assert "total_count" in result
        # All retrievals should return the same count (data consistency)
        assert result["total_count"] == 10, \
            f"Expected 10 memories, got {result['total_count']}"
