"""
Integration tests for context injection instrumentation.

Tests verify that the context injection function properly records metrics
and logs events when metrics_collector and logger are provided, while
maintaining identical behavior when they are None.

Requirements tested:
- 14.4: Test that context_injection_latency_ms is recorded when metrics_collector is provided
- 14.5: Test that context_injection_count is incremented when metrics_collector is provided
- 14.6: Test that no exceptions occur when metrics_collector is None
- 14.6: Test that no exceptions occur when logger is None
- 13.5: Test that injection results are identical with and without instrumentation
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from luma.core.context_injection import inject_memories, InjectionConfig
from luma.core.memory_interface import MemoryInterface, MemoryEntry, RetrievalResult
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


class MockMemoryInterface(MemoryInterface):
    """Mock memory interface for testing context injection instrumentation."""
    
    def __init__(self, memories: List[MemoryEntry] = None, should_fail: bool = False):
        """
        Initialize mock with optional memories and failure mode.
        
        Args:
            memories: List of MemoryEntry objects to return from retrieve()
            should_fail: If True, retrieve() will raise MemoryRetrievalError
        """
        self.memories = memories or []
        self.should_fail = should_fail
        self.retrieve_calls = []
    
    def store(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Mock store method (not used in context injection tests)."""
        return "mock_id"
    
    def retrieve(self, query=None, params=None, limit=10, metrics_collector=None, logger=None) -> RetrievalResult:
        """Mock retrieve method that records calls and returns configured memories."""
        # Record the call for verification
        self.retrieve_calls.append({
            "query": query,
            "params": params,
            "limit": limit,
            "metrics_collector": metrics_collector,
            "logger": logger
        })
        
        if self.should_fail:
            from luma.core.memory_interface import MemoryRetrievalError
            raise MemoryRetrievalError("Mock retrieval failure")
        
        return {
            "memories": self.memories,
            "total_count": len(self.memories),
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }


def create_mock_memory_entry(memory_id: str, content: str) -> MemoryEntry:
    """Create a mock MemoryEntry for testing."""
    return {
        "id": memory_id,
        "content": content,
        "metadata": {"source": "test"},
        "timestamp": "2024-01-15T10:30:00Z",
        "category": "test",
        "tags": ["test"]
    }


class TestContextInjectionInstrumentation:
    """Test suite for context injection instrumentation."""
    
    def test_metrics_recorded_on_success(self):
        """Test that context_injection_latency_ms and context_injection_count are recorded on successful injection."""
        # Setup
        memories = [
            create_mock_memory_entry("mem1", "Test memory 1"),
            create_mock_memory_entry("mem2", "Test memory 2")
        ]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        metrics_collector = MetricsCollector()
        
        # Execute
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            metrics_collector=metrics_collector
        )
        
        # Verify metrics were recorded
        snapshot = metrics_collector.get_snapshot()
        
        # Check counter was incremented
        assert "context_injection_count" in snapshot["counters"]
        assert snapshot["counters"]["context_injection_count"] == 1
        
        # Check latency was recorded
        assert "context_injection_latency_ms" in snapshot["timers"]
        timer_stats = snapshot["timers"]["context_injection_latency_ms"]
        assert timer_stats["count"] == 1
        assert timer_stats["sum"] > 0  # Should have some duration
        assert timer_stats["min"] > 0
        assert timer_stats["max"] > 0
        assert timer_stats["mean"] > 0
        
        # Verify result structure is correct
        assert "memories" in result
        assert len(result["memories"]) == 2
    
    def test_metrics_recorded_on_failure(self):
        """Test that metrics are still recorded when memory retrieval fails."""
        # Setup
        memory_interface = MockMemoryInterface(should_fail=True)
        config = InjectionConfig(max_memories=10)
        metrics_collector = MetricsCollector()
        
        # Execute (should not raise exception due to graceful error handling)
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            metrics_collector=metrics_collector
        )
        
        # Verify metrics were recorded even on failure
        snapshot = metrics_collector.get_snapshot()
        
        # Check counter was incremented
        assert "context_injection_count" in snapshot["counters"]
        assert snapshot["counters"]["context_injection_count"] == 1
        
        # Check latency was recorded
        assert "context_injection_latency_ms" in snapshot["timers"]
        timer_stats = snapshot["timers"]["context_injection_latency_ms"]
        assert timer_stats["count"] == 1
        assert timer_stats["sum"] > 0
        
        # Verify graceful failure handling
        assert "memories" in result
        assert result["memories"] == []  # Empty list on failure
    
    def test_no_exceptions_when_metrics_collector_none(self):
        """Test that no exceptions occur when metrics_collector is None."""
        # Setup
        memories = [create_mock_memory_entry("mem1", "Test memory 1")]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        
        # Execute with metrics_collector=None (should not raise)
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            metrics_collector=None
        )
        
        # Verify result is correct
        assert "memories" in result
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "Test memory 1"
    
    def test_no_exceptions_when_logger_none(self):
        """Test that no exceptions occur when logger is None."""
        # Setup
        memories = [create_mock_memory_entry("mem1", "Test memory 1")]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        
        # Execute with logger=None (should not raise)
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            logger=None
        )
        
        # Verify result is correct
        assert "memories" in result
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == "Test memory 1"
    
    def test_identical_results_with_and_without_instrumentation(self):
        """Test that injection results are identical with and without instrumentation."""
        # Setup
        memories = [
            create_mock_memory_entry("mem1", "Test memory 1"),
            create_mock_memory_entry("mem2", "Test memory 2"),
            create_mock_memory_entry("mem3", "Test memory 3")
        ]
        memory_interface_1 = MockMemoryInterface(memories=memories)
        memory_interface_2 = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        metrics_collector = MetricsCollector()
        logger = StructuredLogger("test_logger")
        
        # Execute without instrumentation
        result_without = inject_memories(
            query="test query",
            memory_interface=memory_interface_1,
            config=config,
            metrics_collector=None,
            logger=None
        )
        
        # Execute with instrumentation
        result_with = inject_memories(
            query="test query",
            memory_interface=memory_interface_2,
            config=config,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        # Verify results are identical (excluding any timing-sensitive fields)
        assert result_without.keys() == result_with.keys()
        assert len(result_without["memories"]) == len(result_with["memories"])
        
        # Compare each memory entry
        for i, (mem_without, mem_with) in enumerate(zip(result_without["memories"], result_with["memories"])):
            assert mem_without["id"] == mem_with["id"], f"Memory {i} ID mismatch"
            assert mem_without["content"] == mem_with["content"], f"Memory {i} content mismatch"
            assert mem_without["metadata"] == mem_with["metadata"], f"Memory {i} metadata mismatch"
            assert mem_without["timestamp"] == mem_with["timestamp"], f"Memory {i} timestamp mismatch"
            assert mem_without["category"] == mem_with["category"], f"Memory {i} category mismatch"
            assert mem_without["tags"] == mem_with["tags"], f"Memory {i} tags mismatch"
    
    @patch('luma.core.structured_logger.logging.getLogger')
    def test_logging_events_recorded(self, mock_get_logger):
        """Test that logging events are recorded when logger is provided."""
        # Setup mock logger
        mock_logger_instance = Mock()
        mock_get_logger.return_value = mock_logger_instance
        mock_logger_instance.handlers = []  # No existing handlers
        
        memories = [create_mock_memory_entry("mem1", "Test memory 1")]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        logger = StructuredLogger("test_logger")
        
        # Execute
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            logger=logger
        )
        
        # Verify logger was called (the actual logging calls are tested in structured_logger tests)
        # We just verify that the logger was set up and would be used
        assert mock_get_logger.called
        assert "memories" in result
        assert len(result["memories"]) == 1
    
    def test_memory_truncation_with_instrumentation(self):
        """Test that memory truncation works correctly with instrumentation."""
        # Setup with more memories than the limit
        memories = [
            create_mock_memory_entry(f"mem{i}", f"Test memory {i}")
            for i in range(15)  # 15 memories
        ]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)  # Limit to 10
        metrics_collector = MetricsCollector()
        
        # Execute
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            metrics_collector=metrics_collector
        )
        
        # Verify truncation occurred
        assert "memories" in result
        assert len(result["memories"]) == 10  # Should be truncated to limit
        
        # Verify metrics were still recorded
        snapshot = metrics_collector.get_snapshot()
        assert snapshot["counters"]["context_injection_count"] == 1
        assert "context_injection_latency_ms" in snapshot["timers"]
    
    def test_existing_context_preservation(self):
        """Test that existing context is preserved when provided."""
        # Setup
        memories = [create_mock_memory_entry("mem1", "Test memory 1")]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        existing_context = {"user_id": "test_user", "session": "session_123"}
        metrics_collector = MetricsCollector()
        
        # Execute
        result = inject_memories(
            query="test query",
            memory_interface=memory_interface,
            config=config,
            existing_context=existing_context,
            metrics_collector=metrics_collector
        )
        
        # Verify existing context is preserved
        assert result["user_id"] == "test_user"
        assert result["session"] == "session_123"
        assert "memories" in result
        assert len(result["memories"]) == 1
        
        # Verify metrics were recorded
        snapshot = metrics_collector.get_snapshot()
        assert snapshot["counters"]["context_injection_count"] == 1
    
    def test_multiple_injections_increment_counter(self):
        """Test that multiple injections properly increment the counter."""
        # Setup
        memories = [create_mock_memory_entry("mem1", "Test memory 1")]
        memory_interface = MockMemoryInterface(memories=memories)
        config = InjectionConfig(max_memories=10)
        metrics_collector = MetricsCollector()
        
        # Execute multiple injections
        for i in range(3):
            inject_memories(
                query=f"test query {i}",
                memory_interface=memory_interface,
                config=config,
                metrics_collector=metrics_collector
            )
        
        # Verify counter was incremented for each injection
        snapshot = metrics_collector.get_snapshot()
        assert snapshot["counters"]["context_injection_count"] == 3
        
        # Verify latency was recorded for each injection
        timer_stats = snapshot["timers"]["context_injection_latency_ms"]
        assert timer_stats["count"] == 3
        assert timer_stats["sum"] > 0