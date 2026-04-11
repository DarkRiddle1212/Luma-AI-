"""
Performance tests for Luma Memory Module.

This test suite validates performance requirements including:
- Store operation latency (< 100ms)
- Retrieve operation latency (< 200ms)
- Memory usage (< 100MB)
- Concurrent request handling
- Large dataset queries

**Validates: Requirements 6.1, 6.2, 6.3**
"""

import pytest
import time
import tempfile
import os
import psutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.processing.validation import ValidationManager
from luma_memory.config import MemoryModuleConfig
from luma_memory.models import SensitivityLevel


class TestPerformance:
    """Performance tests to verify latency and resource requirements."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Provide a temporary database path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)
    
    @pytest.fixture
    def sqlite_storage(self, temp_db_path):
        """Create a SQLite storage backend."""
        storage = SQLiteStorage(temp_db_path)
        yield storage
        if hasattr(storage, 'connection_pool'):
            storage.connection_pool.close_all()
    
    @pytest.fixture
    def memory_storage(self):
        """Create an in-memory storage backend."""
        return MemoryStorage()
    
    @pytest.fixture
    def config(self):
        """Create a configuration with metrics enabled."""
        return MemoryModuleConfig(enable_metrics=True)
    
    @pytest.fixture
    def memory_manager(self, sqlite_storage, config):
        """Create a memory manager with SQLite storage."""
        return MemoryManager(
            storage=sqlite_storage,
            validation=ValidationManager(),
            config=config
        )
    
    @pytest.fixture
    def memory_manager_in_memory(self, memory_storage, config):
        """Create a memory manager with in-memory storage."""
        return MemoryManager(
            storage=memory_storage,
            validation=ValidationManager(),
            config=config
        )
    
    def test_store_operation_latency(self, memory_manager):
        """
        Test that store operations complete within 100ms.
        
        **Validates: Requirement 6.1**
        """
        # Perform multiple store operations and measure latency
        latencies = []
        num_operations = 10
        
        for i in range(num_operations):
            start_time = time.time()
            
            entry_id = memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"key": f"value_{i}", "index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["performance", "test"]
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)
            
            assert entry_id is not None, "Entry ID should be returned"
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nStore operation latencies:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        
        # Assert that average latency is under 100ms
        assert avg_latency < 100, f"Average store latency {avg_latency:.2f}ms exceeds 100ms target"
        
        # Assert that max latency is reasonable (allow some variance)
        assert max_latency < 150, f"Max store latency {max_latency:.2f}ms is too high"
    
    def test_retrieve_operation_latency(self, memory_manager):
        """
        Test that retrieve operations complete within 200ms.
        
        **Validates: Requirement 6.2**
        """
        # Create test entries
        entry_ids = []
        for i in range(10):
            entry_id = memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"key": f"value_{i}", "index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["performance", "test"]
            )
            entry_ids.append(entry_id)
        
        # Measure retrieve latencies
        latencies = []
        
        for entry_id in entry_ids:
            start_time = time.time()
            
            entry = memory_manager.get_memory(entry_id)
            
            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)
            
            assert entry is not None, f"Entry {entry_id} should be retrieved"
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nRetrieve operation latencies:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        
        # Assert that average latency is under 200ms
        assert avg_latency < 200, f"Average retrieve latency {avg_latency:.2f}ms exceeds 200ms target"
        
        # Assert that max latency is reasonable
        assert max_latency < 250, f"Max retrieve latency {max_latency:.2f}ms is too high"
    
    def test_query_operation_latency(self, memory_manager):
        """
        Test that query operations returning up to 100 entries complete within 200ms.
        
        **Validates: Requirement 6.2**
        """
        # Create 100 test entries
        for i in range(100):
            memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"key": f"value_{i}", "index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["performance", "test"]
            )
        
        # Measure query latency
        start_time = time.time()
        
        entries = memory_manager.query_memories(
            tags=["performance"],
            limit=100
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"\nQuery operation latency (100 entries): {elapsed_ms:.2f}ms")
        
        assert len(entries) == 100, "Should retrieve 100 entries"
        assert elapsed_ms < 200, f"Query latency {elapsed_ms:.2f}ms exceeds 200ms target"
    
    def test_memory_usage(self, memory_manager_in_memory):
        """
        Test that memory usage stays under 100MB during normal operation.
        
        **Validates: Requirement 6.3**
        """
        # Get current process
        process = psutil.Process()
        
        # Get baseline memory usage
        baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Create a significant number of entries
        num_entries = 1000
        for i in range(num_entries):
            memory_manager_in_memory.create_memory(
                action=f"test_action_{i}",
                context={
                    "key": f"value_{i}",
                    "index": i,
                    "data": "x" * 100  # Add some data to make entries larger
                },
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["performance", "test"]
            )
        
        # Query entries multiple times
        for _ in range(10):
            entries = memory_manager_in_memory.query_memories(limit=100)
            assert len(entries) > 0
        
        # Get current memory usage
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_increase_mb = current_memory_mb - baseline_memory_mb
        
        print(f"\nMemory usage:")
        print(f"  Baseline: {baseline_memory_mb:.2f}MB")
        print(f"  Current: {current_memory_mb:.2f}MB")
        print(f"  Increase: {memory_increase_mb:.2f}MB")
        
        # Assert that memory increase is under 100MB
        assert memory_increase_mb < 100, f"Memory increase {memory_increase_mb:.2f}MB exceeds 100MB limit"
    
    def test_concurrent_create_operations(self, memory_manager):
        """
        Test concurrent create operations to verify thread safety and performance.
        
        **Validates: Requirement 6.1**
        """
        num_threads = 10
        operations_per_thread = 5
        
        def create_entries(thread_id):
            """Create entries in a thread."""
            latencies = []
            for i in range(operations_per_thread):
                start_time = time.time()
                
                entry_id = memory_manager.create_memory(
                    action=f"concurrent_action_t{thread_id}_i{i}",
                    context={"thread": thread_id, "index": i},
                    device_id=f"device-{thread_id}",
                    sensitivity=SensitivityLevel.PUBLIC,
                    tags=["concurrent", "test"]
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                latencies.append(elapsed_ms)
                
                assert entry_id is not None
            
            return latencies
        
        # Execute concurrent operations
        all_latencies = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(create_entries, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                latencies = future.result()
                all_latencies.extend(latencies)
        
        # Calculate statistics
        avg_latency = sum(all_latencies) / len(all_latencies)
        max_latency = max(all_latencies)
        
        print(f"\nConcurrent create operations:")
        print(f"  Total operations: {len(all_latencies)}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        
        # Assert performance under concurrent load
        # Note: Concurrent operations are expected to be slower due to database locking
        assert avg_latency < 200, f"Average concurrent latency {avg_latency:.2f}ms is too high"
        assert max_latency < 350, f"Max concurrent latency {max_latency:.2f}ms is too high"
    
    def test_concurrent_query_operations(self, memory_manager):
        """
        Test concurrent query operations to verify thread safety and performance.
        
        **Validates: Requirement 6.2**
        """
        # Create test data
        for i in range(50):
            memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["concurrent", "test"]
            )
        
        num_threads = 10
        queries_per_thread = 5
        
        def query_entries(thread_id):
            """Query entries in a thread."""
            latencies = []
            for i in range(queries_per_thread):
                start_time = time.time()
                
                entries = memory_manager.query_memories(
                    tags=["concurrent"],
                    limit=20
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                latencies.append(elapsed_ms)
                
                assert len(entries) > 0
            
            return latencies
        
        # Execute concurrent queries
        all_latencies = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(query_entries, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                latencies = future.result()
                all_latencies.extend(latencies)
        
        # Calculate statistics
        avg_latency = sum(all_latencies) / len(all_latencies)
        max_latency = max(all_latencies)
        
        print(f"\nConcurrent query operations:")
        print(f"  Total operations: {len(all_latencies)}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Max latency: {max_latency:.2f}ms")
        
        # Assert performance under concurrent load
        assert avg_latency < 250, f"Average concurrent query latency {avg_latency:.2f}ms is too high"
        assert max_latency < 400, f"Max concurrent query latency {max_latency:.2f}ms is too high"
    
    def test_large_dataset_query_performance(self, memory_manager):
        """
        Test query performance with a large dataset.
        
        **Validates: Requirement 6.2**
        """
        # Create a large dataset
        num_entries = 500
        print(f"\nCreating {num_entries} entries...")
        
        for i in range(num_entries):
            memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"index": i, "data": f"data_{i}"},
                device_id=f"device-{i % 10}",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=[f"tag_{i % 5}", "large_dataset"]
            )
        
        # Test various query patterns
        test_cases = [
            {"name": "Query by tag", "params": {"tags": ["large_dataset"], "limit": 100}},
            {"name": "Query by time range", "params": {
                "start_time": datetime.now() - timedelta(hours=1),
                "limit": 100
            }},
            {"name": "Query with pagination", "params": {"limit": 50, "offset": 100}},
        ]
        
        for test_case in test_cases:
            start_time = time.time()
            
            entries = memory_manager.query_memories(**test_case["params"])
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            print(f"  {test_case['name']}: {elapsed_ms:.2f}ms ({len(entries)} entries)")
            
            assert elapsed_ms < 200, f"{test_case['name']} latency {elapsed_ms:.2f}ms exceeds 200ms target"
    
    def test_concurrent_mixed_operations(self, memory_manager):
        """
        Test concurrent mixed operations (create, read, query) to verify thread safety.
        
        **Validates: Requirement 6.1, 6.2**
        """
        # Create initial test data
        initial_entry_ids = []
        for i in range(20):
            entry_id = memory_manager.create_memory(
                action=f"initial_action_{i}",
                context={"index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["mixed", "test"]
            )
            initial_entry_ids.append(entry_id)
        
        num_threads = 8
        operations_per_thread = 3
        
        def mixed_operations(thread_id):
            """Perform mixed operations in a thread."""
            results = []
            
            for i in range(operations_per_thread):
                # Create operation
                entry_id = memory_manager.create_memory(
                    action=f"mixed_action_t{thread_id}_i{i}",
                    context={"thread": thread_id, "index": i},
                    device_id=f"device-{thread_id}",
                    sensitivity=SensitivityLevel.PUBLIC,
                    tags=["mixed", "concurrent"]
                )
                results.append(("create", entry_id is not None))
                
                # Read operation
                if initial_entry_ids:
                    entry = memory_manager.get_memory(initial_entry_ids[i % len(initial_entry_ids)])
                    results.append(("read", entry is not None))
                
                # Query operation
                entries = memory_manager.query_memories(
                    tags=["mixed"],
                    limit=10
                )
                results.append(("query", len(entries) > 0))
            
            return results
        
        # Execute concurrent mixed operations
        all_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # Verify all operations succeeded
        create_count = sum(1 for op, success in all_results if op == "create" and success)
        read_count = sum(1 for op, success in all_results if op == "read" and success)
        query_count = sum(1 for op, success in all_results if op == "query" and success)
        
        print(f"\nConcurrent mixed operations:")
        print(f"  Create operations: {create_count}/{num_threads * operations_per_thread}")
        print(f"  Read operations: {read_count}/{num_threads * operations_per_thread}")
        print(f"  Query operations: {query_count}/{num_threads * operations_per_thread}")
        
        # Assert all operations succeeded
        assert create_count == num_threads * operations_per_thread, "Some create operations failed"
        assert read_count == num_threads * operations_per_thread, "Some read operations failed"
        assert query_count == num_threads * operations_per_thread, "Some query operations failed"
    
    def test_performance_metrics_collection(self, memory_manager):
        """
        Test that performance metrics are collected correctly.
        
        **Validates: Requirement 6.1, 6.2**
        """
        # Reset metrics
        memory_manager.reset_performance_metrics()
        
        # Perform various operations
        entry_ids = []
        for i in range(5):
            entry_id = memory_manager.create_memory(
                action=f"test_action_{i}",
                context={"index": i},
                device_id="test-device",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=["metrics", "test"]
            )
            entry_ids.append(entry_id)
        
        # Retrieve entries
        for entry_id in entry_ids:
            memory_manager.get_memory(entry_id)
        
        # Query entries
        memory_manager.query_memories(tags=["metrics"], limit=10)
        
        # Get performance metrics
        metrics = memory_manager.get_performance_metrics()
        
        print(f"\nPerformance metrics:")
        for operation, stats in metrics.items():
            if stats['count'] > 0:
                print(f"  {operation}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg: {stats['avg_time_ms']:.2f}ms")
                print(f"    Min: {stats['min_time_ms']:.2f}ms")
                print(f"    Max: {stats['max_time_ms']:.2f}ms")
        
        # Verify metrics were collected
        assert metrics['create_memory']['count'] == 5
        assert metrics['get_memory']['count'] == 5
        assert metrics['query_memories']['count'] == 1
        
        # Verify latencies are reasonable
        assert metrics['create_memory']['avg_time_ms'] < 100
        assert metrics['get_memory']['avg_time_ms'] < 200
        assert metrics['query_memories']['avg_time_ms'] < 200
