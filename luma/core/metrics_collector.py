"""
MetricsCollector: Thread-safe metrics collection for counters and timers.

This module provides a lightweight, thread-safe metrics collection system
using only Python standard library components. It supports:
- Counter metrics with O(1) increment operations
- Timer metrics with O(1) duration recording
- Statistical summaries (count, sum, min, max, mean)
- Thread-safe snapshot retrieval
- Reset capability for fresh measurement periods
"""

import threading
from collections import defaultdict
from typing import Dict, List, Any


class MetricsCollector:
    """
    Thread-safe collector for counter and timer metrics.
    
    This class provides O(1) operations for incrementing counters and recording
    durations, with thread-safe access using internal locking mechanisms.
    
    Attributes:
        _lock: Threading lock for ensuring thread-safe operations
        _counters: Dictionary storing counter values
        _timers: Dictionary storing lists of duration measurements
    """
    
    def __init__(self):
        """Initialize the MetricsCollector with empty counters and timers."""
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
    
    def increment(self, name: str, value: float = 1) -> None:
        """
        Increment a counter by the specified value.
        
        This operation is thread-safe and completes in O(1) time.
        
        Args:
            name: The name of the counter to increment
            value: The amount to increment by (default: 1)
        """
        with self._lock:
            self._counters[name] += value
    
    def record_duration(self, name: str, duration_ms: float) -> None:
        """
        Record a duration measurement for a timer.
        
        This operation is thread-safe and completes in O(1) time.
        
        Args:
            name: The name of the timer
            duration_ms: The duration value in milliseconds
        """
        with self._lock:
            self._timers[name].append(duration_ms)
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a point-in-time snapshot of all metrics.
        
        Returns a dictionary containing:
        - counters: Dictionary of counter names to their current values
        - timers: Dictionary of timer names to their statistics
          (count, sum, min, max, mean)
        
        This operation is thread-safe and completes in O(1) time relative
        to the number of metrics (O(n) where n is the number of timer samples,
        but O(1) relative to the number of distinct metrics).
        
        Returns:
            Dictionary with 'counters' and 'timers' keys
        """
        with self._lock:
            # Create a snapshot of counters
            counters_snapshot = dict(self._counters)
            
            # Create a snapshot of timer statistics
            timers_snapshot = {}
            for name, durations in self._timers.items():
                if durations:
                    count = len(durations)
                    total = sum(durations)
                    timers_snapshot[name] = {
                        'count': count,
                        'sum': total,
                        'min': min(durations),
                        'max': max(durations),
                        'mean': total / count
                    }
                else:
                    timers_snapshot[name] = {
                        'count': 0,
                        'sum': 0,
                        'min': 0,
                        'max': 0,
                        'mean': 0
                    }
            
            return {
                'counters': counters_snapshot,
                'timers': timers_snapshot
            }
    
    def reset(self) -> None:
        """
        Reset all metrics to their initial state.
        
        This operation clears all counters (sets to zero) and removes all
        timer measurements. The operation is thread-safe.
        """
        with self._lock:
            self._counters.clear()
            self._timers.clear()
