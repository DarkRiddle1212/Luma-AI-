"""
Unit Tests for Concurrent Memory Operations

This module implements unit tests for specific concurrency scenarios to verify
that the SQLiteMemoryAdapter handles concurrent operations safely.

Feature: intent-based-memory-retrieval-enhancements
Task: 7.2 Write unit tests for specific concurrency scenarios
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, MagicMock
from datetime import datetime
import threading
import time

from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import QueryParameters


# ============================================================================
# Helper Functions
# ============================================================================

def create_thread_safe_mock_memory_manager():
    """
    Create a mock MemoryManager that simulates thread-safe behavior.
    
    This mock uses locks to ensure thread-safe access to its internal state.
    """
    mock_manager = MagicMock()
    
    # Thread-safe storage
    memories = []
    memory_lock = threading.Lock()
    memory_counter = [0]
    
    def create_memory_impl(action, context, device_id, tags):
        with memory_lock:
            memory_counter[0] += 1
            memory_id = f"memory-{memory_counter[0]}"
            
            mock_entry = Mock()
            mock_entry.id = memory_id
            mock_entry.action = action
            mock_entry.context = context
            mock_entry.tags = tags
            mock_entry.device_id = device_id
            mock_entry.created_at = datetime.now()
            mock_entry.timestamp = datetime.now()
            
            memories.append(mock_entry)
            time.sleep(0.001)  # Simulate processing
            
            return memory_id
    
    def query_memories_impl(action_type=None, start_time=None, end_time=None, tags=None, limit=10):
        with memory_lock:
            results = []
            
            for entry in memories:
                if action_type and action_type not in entry.action:
                    continue
                if start_time and entry.created_at < start_time:
                    continue
                if end_time and entry.created_at > end_time:
                    continue
                if tags:
                    entry_tags = set(entry.tags or [])
                    required_tags = set(tags)
                    if not required_tags.issubset(entry_tags):
                        continue
                
                results.append(entry)
                if len(results) >= limit:
                    break
            
            time.sleep(0.001)  # Simulate processing
            return results
    
    mock_manager.create_memory.side_effect = create_memory_impl
    mock_manager.query_memories.side_effect = query_memories_impl
    mock_manager._test_memories = memories
    mock_manager._test_lock = memory_lock
    
    return mock_manager


# ============================================================================
# Unit Tests for Concurrent Operations
# ============================================================================

class TestConcurrentStoreOperations:
    """Tests for concurrent store operations."""
    
    def test_multiple_threads_storing_simultaneously(self):
        """
        Test that multiple threads can store memories simultaneously without errors.
        
        **Validates: Requirement 7.1**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(
            mock_memory_manager,
            device_id="test-device",
            default_category="test"
        )
        
        num_threads = 20
        stores_per_thread = 5
        stored_ids = []
        errors = []
        
        def store_memories(thread_id):
            try:
                for i in range(stores_per_thread):
                    content = f"Thread {thread_id} - Memory {i}"
                    metadata = {"category": "test", "tags": [f"thread-{thread_id}"]}
                    memory_id = adapter.store(content, metadata)
                    stored_ids.append(memory_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Execute concurrent stores
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(store_memories, i) for i in range(num_threads)]
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all stores completed
        expected_count = num_threads * stores_per_thread
        assert len(stored_ids) == expected_count, \
            f"Expected {expected_count} stores, got {len(stored_ids)}"
        
        # Verify data integrity
        with mock_memory_manager._test_lock:
            actual_count = len(mock_memory_manager._test_memories)
        assert actual_count == expected_count, \
            f"Expected {expected_count} memories in storage, got {actual_count}"
    
    def test_concurrent_stores_with_different_metadata(self):
        """
        Test concurrent stores with varying metadata don't interfere with each other.
        
        **Validates: Requirement 7.1**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        categories = ["work", "personal", "education", "system"]
        stored_data = []
        errors = []
        
        def store_with_category(category, index):
            try:
                content = f"Content for {category} - {index}"
                metadata = {"category": category, "tags": [category, f"item-{index}"]}
                memory_id = adapter.store(content, metadata)
                stored_data.append((memory_id, category, content))
            except Exception as e:
                errors.append((category, index, str(e)))
        
        # Store concurrently with different categories
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(25):
                category = categories[i % len(categories)]
                futures.append(executor.submit(store_with_category, category, i))
            
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all stores completed
        assert len(stored_data) == 25
        
        # Verify data integrity - each category should have correct count
        category_counts = {}
        for _, category, _ in stored_data:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Each category should appear roughly equal times (25 / 4 ≈ 6-7)
        for category in categories:
            assert category in category_counts
            assert category_counts[category] >= 5  # Allow some variance


class TestConcurrentRetrieveOperations:
    """Tests for concurrent retrieve operations."""
    
    def test_multiple_threads_retrieving_simultaneously(self):
        """
        Test that multiple threads can retrieve memories simultaneously with correct results.
        
        **Validates: Requirement 7.2**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        # Pre-populate with memories
        num_memories = 15
        for i in range(num_memories):
            adapter.store(f"Memory {i}", {"category": "test", "tags": ["test"]})
        
        num_threads = 20
        results = []
        errors = []
        
        def retrieve_memories(thread_id):
            try:
                result = adapter.retrieve(params={"limit": 20})
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Execute concurrent retrievals
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(retrieve_memories, i) for i in range(num_threads)]
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all retrievals completed
        assert len(results) == num_threads
        
        # Verify all results are consistent (data consistency)
        for thread_id, result in results:
            assert "memories" in result
            assert "total_count" in result
            assert result["total_count"] == num_memories, \
                f"Thread {thread_id} got {result['total_count']} memories, expected {num_memories}"
    
    def test_concurrent_retrievals_with_different_filters(self):
        """
        Test concurrent retrievals with different filters return correct results.
        
        **Validates: Requirement 7.2**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        # Pre-populate with categorized memories
        categories = ["work", "personal", "education"]
        for category in categories:
            for i in range(5):
                adapter.store(
                    f"{category} memory {i}",
                    {"category": category, "tags": [category]}
                )
        
        results = []
        errors = []
        
        def retrieve_by_category(category):
            try:
                result = adapter.retrieve(params={"category": category, "limit": 20})
                results.append((category, result))
            except Exception as e:
                errors.append((category, str(e)))
        
        # Retrieve concurrently with different filters
        with ThreadPoolExecutor(max_workers=len(categories)) as executor:
            futures = [executor.submit(retrieve_by_category, cat) for cat in categories]
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify each category returned correct count
        for category, result in results:
            # Category filtering happens in post-processing, so we should get the right count
            assert result["total_count"] == 5, \
                f"Category {category} should have 5 memories, got {result['total_count']}"
            
            # Verify all returned memories match the category
            for memory in result["memories"]:
                assert memory["category"] == category, \
                    f"Memory has category {memory['category']}, expected {category}"


class TestMixedConcurrentOperations:
    """Tests for mixed store and retrieve operations."""
    
    def test_concurrent_stores_and_retrievals(self):
        """
        Test that concurrent store and retrieve operations maintain data consistency.
        
        **Validates: Requirement 7.3**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        num_stores = 20
        num_retrievals = 20
        store_results = []
        retrieve_results = []
        errors = []
        
        def store_memory(index):
            try:
                memory_id = adapter.store(f"Memory {index}", {"category": "test"})
                store_results.append(memory_id)
            except Exception as e:
                errors.append(("store", index, str(e)))
        
        def retrieve_memories(index):
            try:
                result = adapter.retrieve(params={"limit": 50})
                retrieve_results.append((index, result["total_count"]))
            except Exception as e:
                errors.append(("retrieve", index, str(e)))
        
        # Execute mixed operations concurrently
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            
            # Submit stores and retrievals in mixed order
            for i in range(max(num_stores, num_retrievals)):
                if i < num_stores:
                    futures.append(executor.submit(store_memory, i))
                if i < num_retrievals:
                    futures.append(executor.submit(retrieve_memories, i))
            
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all stores completed
        assert len(store_results) == num_stores
        
        # Verify all retrievals completed
        assert len(retrieve_results) == num_retrievals
        
        # Verify data consistency - final count should match stores
        with mock_memory_manager._test_lock:
            final_count = len(mock_memory_manager._test_memories)
        assert final_count == num_stores, \
            f"Expected {num_stores} memories, got {final_count}"
    
    def test_high_contention_scenario(self):
        """
        Test system behavior under high contention with many concurrent operations.
        
        **Validates: Requirements 7.3, 7.4, 7.5**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        num_operations = 50
        operation_results = []
        errors = []
        
        def perform_operation(index):
            try:
                # Alternate between store and retrieve
                if index % 2 == 0:
                    memory_id = adapter.store(
                        f"High contention memory {index}",
                        {"category": "stress-test", "tags": ["stress"]}
                    )
                    operation_results.append(("store", memory_id))
                else:
                    result = adapter.retrieve(params={"limit": 10})
                    operation_results.append(("retrieve", result["total_count"]))
            except Exception as e:
                errors.append((index, str(e)))
        
        # Execute with high concurrency
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(perform_operation, i) for i in range(num_operations)]
            
            # Use timeout to detect deadlocks
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify no errors (no deadlocks, no race conditions)
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all operations completed
        assert len(operation_results) == num_operations
        
        # Count operations
        store_count = sum(1 for op_type, _ in operation_results if op_type == "store")
        retrieve_count = sum(1 for op_type, _ in operation_results if op_type == "retrieve")
        
        assert store_count == num_operations // 2
        assert retrieve_count == num_operations // 2


class TestDeadlockPrevention:
    """Tests to verify no deadlocks occur."""
    
    def test_no_deadlock_with_rapid_operations(self):
        """
        Test that rapid concurrent operations don't cause deadlocks.
        
        **Validates: Requirement 7.4**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        num_operations = 100
        completed = []
        
        def rapid_operation(index):
            # Perform operation without any delays
            if index % 3 == 0:
                adapter.store(f"Rapid {index}", {"category": "rapid"})
            else:
                adapter.retrieve(params={"limit": 5})
            completed.append(index)
        
        # Execute rapidly with timeout to detect deadlocks
        try:
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(rapid_operation, i) for i in range(num_operations)]
                
                # If this times out, we have a deadlock
                for future in as_completed(futures, timeout=30):
                    future.result()
        except Exception as e:
            pytest.fail(f"Deadlock or timeout detected: {e}")
        
        # Verify all operations completed
        assert len(completed) == num_operations, \
            f"Only {len(completed)} of {num_operations} operations completed - possible deadlock"
    
    def test_no_deadlock_with_nested_operations(self):
        """
        Test that operations don't deadlock even with complex patterns.
        
        **Validates: Requirement 7.4**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        completed = []
        
        def complex_operation(index):
            # Store, then immediately retrieve
            adapter.store(f"Complex {index}", {"category": "complex"})
            adapter.retrieve(params={"category": "complex", "limit": 10})
            completed.append(index)
        
        # Execute with timeout
        try:
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = [executor.submit(complex_operation, i) for i in range(30)]
                
                for future in as_completed(futures, timeout=30):
                    future.result()
        except Exception as e:
            pytest.fail(f"Deadlock detected in nested operations: {e}")
        
        # Verify all completed
        assert len(completed) == 30


class TestRaceConditionPrevention:
    """Tests to verify no race conditions occur."""
    
    def test_no_race_condition_in_counter(self):
        """
        Test that concurrent stores don't have race conditions in ID generation.
        
        **Validates: Requirement 7.5**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        num_stores = 50
        stored_ids = []
        
        def store_and_collect_id(index):
            memory_id = adapter.store(f"Race test {index}", {"category": "race"})
            stored_ids.append(memory_id)
        
        # Execute concurrent stores
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(store_and_collect_id, i) for i in range(num_stores)]
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify all IDs are unique (no race condition in ID generation)
        assert len(stored_ids) == num_stores
        assert len(set(stored_ids)) == num_stores, \
            "Duplicate IDs detected - race condition in ID generation"
    
    def test_no_race_condition_in_data_integrity(self):
        """
        Test that concurrent operations maintain data integrity.
        
        **Validates: Requirement 7.5**
        """
        mock_memory_manager = create_thread_safe_mock_memory_manager()
        adapter = SQLiteMemoryAdapter(mock_memory_manager, device_id="test-device")
        
        # Store memories with specific tags
        num_stores = 30
        tag_counts = {"tag-a": 0, "tag-b": 0, "tag-c": 0}
        
        def store_with_tag(index):
            tag = f"tag-{chr(97 + (index % 3))}"  # tag-a, tag-b, or tag-c
            adapter.store(f"Tagged {index}", {"category": "tagged", "tags": [tag]})
            tag_counts[tag] += 1
        
        # Store concurrently
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(store_with_tag, i) for i in range(num_stores)]
            for future in as_completed(futures, timeout=30):
                future.result()
        
        # Verify total count
        with mock_memory_manager._test_lock:
            total_stored = len(mock_memory_manager._test_memories)
        
        assert total_stored == num_stores, \
            f"Expected {num_stores} memories, got {total_stored} - data integrity issue"
        
        # Verify each tag appears correct number of times
        expected_per_tag = num_stores // 3
        for tag, count in tag_counts.items():
            assert count == expected_per_tag, \
                f"Tag {tag} should appear {expected_per_tag} times, got {count}"
