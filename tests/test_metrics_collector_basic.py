"""
Basic verification tests for MetricsCollector implementation.
"""

import pytest
from luma.core.metrics_collector import MetricsCollector


def test_counter_increment_default():
    """Test counter increment with default value of 1."""
    collector = MetricsCollector()
    collector.increment('test_counter')
    snapshot = collector.get_snapshot()
    assert snapshot['counters']['test_counter'] == 1


def test_counter_increment_custom_value():
    """Test counter increment with custom value."""
    collector = MetricsCollector()
    collector.increment('test_counter', 5)
    snapshot = collector.get_snapshot()
    assert snapshot['counters']['test_counter'] == 5


def test_counter_multiple_increments():
    """Test multiple increments accumulate correctly."""
    collector = MetricsCollector()
    collector.increment('test_counter', 3)
    collector.increment('test_counter', 2)
    collector.increment('test_counter')
    snapshot = collector.get_snapshot()
    assert snapshot['counters']['test_counter'] == 6


def test_timer_record_duration():
    """Test recording a single duration."""
    collector = MetricsCollector()
    collector.record_duration('test_timer', 100.5)
    snapshot = collector.get_snapshot()
    
    assert 'test_timer' in snapshot['timers']
    timer_stats = snapshot['timers']['test_timer']
    assert timer_stats['count'] == 1
    assert timer_stats['sum'] == 100.5
    assert timer_stats['min'] == 100.5
    assert timer_stats['max'] == 100.5
    assert timer_stats['mean'] == 100.5


def test_timer_multiple_durations():
    """Test recording multiple durations and statistics calculation."""
    collector = MetricsCollector()
    collector.record_duration('test_timer', 100)
    collector.record_duration('test_timer', 200)
    collector.record_duration('test_timer', 150)
    
    snapshot = collector.get_snapshot()
    timer_stats = snapshot['timers']['test_timer']
    
    assert timer_stats['count'] == 3
    assert timer_stats['sum'] == 450
    assert timer_stats['min'] == 100
    assert timer_stats['max'] == 200
    assert timer_stats['mean'] == 150


def test_snapshot_structure():
    """Test that snapshot has correct structure."""
    collector = MetricsCollector()
    collector.increment('counter1')
    collector.record_duration('timer1', 50)
    
    snapshot = collector.get_snapshot()
    
    assert 'counters' in snapshot
    assert 'timers' in snapshot
    assert isinstance(snapshot['counters'], dict)
    assert isinstance(snapshot['timers'], dict)


def test_reset_clears_counters():
    """Test that reset clears all counters."""
    collector = MetricsCollector()
    collector.increment('counter1', 10)
    collector.increment('counter2', 20)
    
    collector.reset()
    
    snapshot = collector.get_snapshot()
    assert len(snapshot['counters']) == 0


def test_reset_clears_timers():
    """Test that reset clears all timers."""
    collector = MetricsCollector()
    collector.record_duration('timer1', 100)
    collector.record_duration('timer2', 200)
    
    collector.reset()
    
    snapshot = collector.get_snapshot()
    assert len(snapshot['timers']) == 0


def test_multiple_counters():
    """Test tracking multiple independent counters."""
    collector = MetricsCollector()
    collector.increment('counter1', 5)
    collector.increment('counter2', 10)
    collector.increment('counter3', 15)
    
    snapshot = collector.get_snapshot()
    assert snapshot['counters']['counter1'] == 5
    assert snapshot['counters']['counter2'] == 10
    assert snapshot['counters']['counter3'] == 15


def test_multiple_timers():
    """Test tracking multiple independent timers."""
    collector = MetricsCollector()
    collector.record_duration('timer1', 100)
    collector.record_duration('timer2', 200)
    
    snapshot = collector.get_snapshot()
    assert 'timer1' in snapshot['timers']
    assert 'timer2' in snapshot['timers']
    assert snapshot['timers']['timer1']['mean'] == 100
    assert snapshot['timers']['timer2']['mean'] == 200


def test_empty_snapshot():
    """Test snapshot of empty collector."""
    collector = MetricsCollector()
    snapshot = collector.get_snapshot()
    
    assert snapshot['counters'] == {}
    assert snapshot['timers'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_increment_performance_o1():
    """Test that increment operations complete in O(1) time."""
    import time
    
    collector = MetricsCollector()
    
    # Measure time for first increment
    start = time.perf_counter()
    collector.increment('test_counter')
    first_duration = time.perf_counter() - start
    
    # Add many increments to build up state
    for i in range(10000):
        collector.increment(f'counter_{i}')
    
    # Measure time for increment after many metrics exist
    start = time.perf_counter()
    collector.increment('test_counter')
    second_duration = time.perf_counter() - start
    
    # The second increment should not be significantly slower
    # Allow 10x tolerance for timing variance
    assert second_duration < first_duration * 10, (
        f"Increment operation appears to scale with number of metrics. "
        f"First: {first_duration:.6f}s, Second: {second_duration:.6f}s"
    )


def test_record_duration_performance_o1():
    """Test that record_duration operations complete in O(1) time."""
    import time
    
    collector = MetricsCollector()
    
    # Measure time for first recording
    start = time.perf_counter()
    collector.record_duration('test_timer', 100.0)
    first_duration = time.perf_counter() - start
    
    # Add many recordings to build up state
    for i in range(10000):
        collector.record_duration(f'timer_{i}', float(i))
    
    # Measure time for recording after many metrics exist
    start = time.perf_counter()
    collector.record_duration('test_timer', 100.0)
    second_duration = time.perf_counter() - start
    
    # The second recording should not be significantly slower
    # Allow 10x tolerance for timing variance
    assert second_duration < first_duration * 10, (
        f"Record duration operation appears to scale with number of metrics. "
        f"First: {first_duration:.6f}s, Second: {second_duration:.6f}s"
    )


def test_get_snapshot_performance():
    """Test that get_snapshot completes in reasonable time relative to number of metrics."""
    import time
    
    collector = MetricsCollector()
    
    # Add a moderate number of metrics
    num_metrics = 100
    for i in range(num_metrics):
        collector.increment(f'counter_{i}', i)
        collector.record_duration(f'timer_{i}', float(i))
    
    # Measure snapshot time
    start = time.perf_counter()
    snapshot = collector.get_snapshot()
    duration = time.perf_counter() - start
    
    # Snapshot should complete quickly (under 10ms for 100 metrics)
    assert duration < 0.01, (
        f"Snapshot operation took {duration:.6f}s for {num_metrics} metrics, "
        f"which is too slow"
    )
    
    # Verify snapshot contains all metrics
    assert len(snapshot['counters']) == num_metrics
    assert len(snapshot['timers']) == num_metrics
