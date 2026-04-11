"""
Thread safety verification tests for MetricsCollector.
"""

import pytest
import threading
from luma.core.metrics_collector import MetricsCollector


def test_concurrent_counter_increments():
    """Test that concurrent increments from multiple threads are accurate."""
    collector = MetricsCollector()
    num_threads = 10
    increments_per_thread = 100
    
    def increment_counter():
        for _ in range(increments_per_thread):
            collector.increment('test_counter')
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=increment_counter)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    snapshot = collector.get_snapshot()
    expected_total = num_threads * increments_per_thread
    assert snapshot['counters']['test_counter'] == expected_total


def test_concurrent_timer_recordings():
    """Test that concurrent duration recordings from multiple threads are accurate."""
    collector = MetricsCollector()
    num_threads = 10
    recordings_per_thread = 50
    
    def record_durations():
        for i in range(recordings_per_thread):
            collector.record_duration('test_timer', float(i))
    
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=record_durations)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    snapshot = collector.get_snapshot()
    expected_count = num_threads * recordings_per_thread
    assert snapshot['timers']['test_timer']['count'] == expected_count


def test_concurrent_snapshot_during_updates():
    """Test that get_snapshot returns consistent data during concurrent updates."""
    collector = MetricsCollector()
    stop_flag = threading.Event()
    
    def continuous_increment():
        while not stop_flag.is_set():
            collector.increment('counter1')
            collector.increment('counter2')
    
    def continuous_record():
        while not stop_flag.is_set():
            collector.record_duration('timer1', 100)
    
    # Start background threads
    threads = [
        threading.Thread(target=continuous_increment),
        threading.Thread(target=continuous_record)
    ]
    for thread in threads:
        thread.start()
    
    # Take multiple snapshots while updates are happening
    snapshots = []
    for _ in range(10):
        snapshot = collector.get_snapshot()
        snapshots.append(snapshot)
        # Verify snapshot structure is consistent
        assert 'counters' in snapshot
        assert 'timers' in snapshot
    
    # Stop background threads
    stop_flag.set()
    for thread in threads:
        thread.join()
    
    # All snapshots should have valid structure
    for snapshot in snapshots:
        assert isinstance(snapshot['counters'], dict)
        assert isinstance(snapshot['timers'], dict)


def test_concurrent_mixed_operations():
    """Test concurrent increments, recordings, and snapshots."""
    collector = MetricsCollector()
    num_operations = 100
    
    def mixed_operations():
        for i in range(num_operations):
            collector.increment('counter1')
            collector.record_duration('timer1', float(i))
            if i % 10 == 0:
                collector.get_snapshot()
    
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=mixed_operations)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    snapshot = collector.get_snapshot()
    assert snapshot['counters']['counter1'] == 5 * num_operations
    assert snapshot['timers']['timer1']['count'] == 5 * num_operations


def test_concurrent_reset():
    """Test that reset is thread-safe."""
    collector = MetricsCollector()
    
    # Pre-populate with data
    for i in range(100):
        collector.increment('counter1')
        collector.record_duration('timer1', float(i))
    
    def reset_operation():
        collector.reset()
    
    def increment_operation():
        for _ in range(50):
            collector.increment('counter2')
    
    # Start concurrent reset and increment operations
    threads = [
        threading.Thread(target=reset_operation),
        threading.Thread(target=increment_operation)
    ]
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # After reset and concurrent increments, we should have a valid state
    snapshot = collector.get_snapshot()
    assert isinstance(snapshot['counters'], dict)
    assert isinstance(snapshot['timers'], dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
