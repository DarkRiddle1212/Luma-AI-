"""
Property-based tests for MetricsCollector thread safety.

**Validates: Requirements 1.2**
"""

import pytest
import threading
from hypothesis import given, settings, strategies as st
from luma.core.metrics_collector import MetricsCollector


@settings(max_examples=10, deadline=None)
@given(
    num_threads=st.integers(min_value=2, max_value=20),
    increments_per_thread=st.integers(min_value=1, max_value=100),
    increment_value=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
)
def test_property_concurrent_increments_preserve_total_count(num_threads, increments_per_thread, increment_value):
    """
    Property: Concurrent increments preserve total count.
    
    **Validates: Requirements 1.2**
    
    For any number of threads, any number of increments per thread, and any increment value,
    when multiple threads call increment() concurrently, the final counter value should equal
    the expected total (num_threads * increments_per_thread * increment_value).
    
    This property verifies that the MetricsCollector's locking mechanism correctly prevents
    race conditions and ensures all increments are accurately recorded.
    """
    collector = MetricsCollector()
    counter_name = 'test_counter'
    
    def increment_counter():
        for _ in range(increments_per_thread):
            collector.increment(counter_name, increment_value)
    
    # Create and start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=increment_counter)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify the total count is accurate
    snapshot = collector.get_snapshot()
    expected_total = num_threads * increments_per_thread * increment_value
    actual_total = snapshot['counters'][counter_name]
    
    # Use approximate equality for floating point comparisons
    assert abs(actual_total - expected_total) < 0.01, (
        f"Expected total {expected_total}, but got {actual_total}. "
        f"Threads: {num_threads}, Increments/thread: {increments_per_thread}, "
        f"Increment value: {increment_value}"
    )


@settings(max_examples=10, deadline=None)
@given(
    num_threads=st.integers(min_value=2, max_value=20),
    recordings_per_thread=st.integers(min_value=1, max_value=50),
    duration_values=st.lists(
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10
    )
)
def test_property_concurrent_duration_recordings_preserve_count(num_threads, recordings_per_thread, duration_values):
    """
    Property: Concurrent duration recordings preserve total count.
    
    **Validates: Requirements 1.3**
    
    For any number of threads and any number of duration recordings per thread,
    when multiple threads call record_duration() concurrently, the final timer count
    should equal the expected total (num_threads * recordings_per_thread).
    
    This property verifies that the MetricsCollector correctly records all duration
    measurements without losing data due to race conditions.
    """
    collector = MetricsCollector()
    timer_name = 'test_timer'
    
    def record_durations():
        for i in range(recordings_per_thread):
            # Cycle through the provided duration values
            duration = duration_values[i % len(duration_values)]
            collector.record_duration(timer_name, duration)
    
    # Create and start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=record_durations)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify the total count is accurate
    snapshot = collector.get_snapshot()
    expected_count = num_threads * recordings_per_thread
    actual_count = snapshot['timers'][timer_name]['count']
    
    assert actual_count == expected_count, (
        f"Expected count {expected_count}, but got {actual_count}. "
        f"Threads: {num_threads}, Recordings/thread: {recordings_per_thread}"
    )


@settings(max_examples=10, deadline=None)
@given(
    num_increment_threads=st.integers(min_value=1, max_value=10),
    num_snapshot_threads=st.integers(min_value=1, max_value=5),
    operations_per_thread=st.integers(min_value=10, max_value=50)
)
def test_property_snapshot_consistency_during_concurrent_updates(
    num_increment_threads, num_snapshot_threads, operations_per_thread
):
    """
    Property: Snapshots are consistent during concurrent updates.
    
    **Validates: Requirements 1.4**
    
    For any number of threads performing updates and snapshots concurrently,
    when get_snapshot() is called during concurrent metric updates, it should
    return a consistent snapshot with valid structure and no corrupted data.
    
    This property verifies that the MetricsCollector's locking mechanism ensures
    snapshot operations see a consistent view of the data even during concurrent
    modifications.
    """
    collector = MetricsCollector()
    snapshots = []
    snapshot_lock = threading.Lock()
    
    def increment_operations():
        for i in range(operations_per_thread):
            collector.increment('counter1', 1)
            collector.increment('counter2', 2)
            collector.record_duration('timer1', float(i))
    
    def snapshot_operations():
        for _ in range(operations_per_thread):
            snapshot = collector.get_snapshot()
            with snapshot_lock:
                snapshots.append(snapshot)
    
    # Create and start threads
    threads = []
    
    # Start increment threads
    for _ in range(num_increment_threads):
        thread = threading.Thread(target=increment_operations)
        threads.append(thread)
        thread.start()
    
    # Start snapshot threads
    for _ in range(num_snapshot_threads):
        thread = threading.Thread(target=snapshot_operations)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify all snapshots have valid structure
    assert len(snapshots) > 0, "No snapshots were captured"
    
    for i, snapshot in enumerate(snapshots):
        # Check structure
        assert 'counters' in snapshot, f"Snapshot {i} missing 'counters' key"
        assert 'timers' in snapshot, f"Snapshot {i} missing 'timers' key"
        assert isinstance(snapshot['counters'], dict), f"Snapshot {i} counters not a dict"
        assert isinstance(snapshot['timers'], dict), f"Snapshot {i} timers not a dict"
        
        # Check that counter values are non-negative (monotonically increasing)
        for counter_name, value in snapshot['counters'].items():
            assert value >= 0, f"Snapshot {i} has negative counter value: {counter_name}={value}"
        
        # Check that timer statistics are valid
        for timer_name, stats in snapshot['timers'].items():
            if stats['count'] > 0:
                assert stats['sum'] >= 0, f"Snapshot {i} timer {timer_name} has negative sum"
                assert stats['min'] >= 0, f"Snapshot {i} timer {timer_name} has negative min"
                assert stats['max'] >= stats['min'], f"Snapshot {i} timer {timer_name} max < min"
                assert abs(stats['mean'] - stats['sum'] / stats['count']) < 0.01, (
                    f"Snapshot {i} timer {timer_name} mean calculation incorrect"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
