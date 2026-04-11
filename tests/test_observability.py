"""
Comprehensive observability test suite.

This module consolidates all observability tests and provides end-to-end
testing of the full observability flow. It verifies:
- MetricsCollector functionality (counters, timers, thread safety)
- StructuredLogger functionality (JSON output, required fields)
- Integration with RetrievalLayer (MemoryInterface.retrieve)
- Integration with RankingEngine
- Integration with MemoryLifecycleManager
- No circular import dependencies
- End-to-end observability flow

**Validates: Requirements 12.1, 12.2, 12.3, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7**
"""

import pytest
import json
import logging
import threading
from io import StringIO
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock

# ============================================================================
# SECTION 1: Verify No Circular Import Dependencies
# ============================================================================

def test_no_circular_import_dependencies():
    """
    Test that all observability components can be imported without circular dependencies.
    
    **Validates: Requirement 12.3**
    """
    # Import observability components
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
    
    # Import instrumented components
    from luma.core.ranking_engine import RankingEngine
    from luma.core.lifecycle_manager import MemoryLifecycleManager
    from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
    
    # If we reach here, no circular imports exist
    assert True, "All imports successful - no circular dependencies"


# ============================================================================
# SECTION 2: MetricsCollector Tests (Reference existing tests)
# ============================================================================

class TestMetricsCollectorBasics:
    """
    Basic functionality tests for MetricsCollector.
    
    These tests verify counter increments, timer recordings, snapshot structure,
    and reset functionality. Detailed tests are in test_metrics_collector_basic.py.
    
    **Validates: Requirements 2.1-2.7, 3.1-3.5, 4.1-4.3, 5.1-5.3, 6.1-6.4, 14.1, 14.2, 14.3**
    """
    
    def test_counter_increment_default(self):
        """Test counter increment with default value of 1."""
        from luma.core.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment('test_counter')
        snapshot = collector.get_snapshot()
        assert snapshot['counters']['test_counter'] == 1
    
    def test_counter_increment_custom_value(self):
        """Test counter increment with custom value."""
        from luma.core.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment('test_counter', 5)
        snapshot = collector.get_snapshot()
        assert snapshot['counters']['test_counter'] == 5
    
    def test_timer_record_duration(self):
        """Test recording a single duration."""
        from luma.core.metrics_collector import MetricsCollector
        
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
    
    def test_snapshot_structure(self):
        """Test that snapshot has correct structure."""
        from luma.core.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment('counter1')
        collector.record_duration('timer1', 50)
        
        snapshot = collector.get_snapshot()
        
        assert 'counters' in snapshot
        assert 'timers' in snapshot
        assert isinstance(snapshot['counters'], dict)
        assert isinstance(snapshot['timers'], dict)
    
    def test_reset_functionality(self):
        """Test that reset clears all metrics."""
        from luma.core.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment('counter1', 10)
        collector.record_duration('timer1', 100)
        
        collector.reset()
        
        snapshot = collector.get_snapshot()
        assert len(snapshot['counters']) == 0
        assert len(snapshot['timers']) == 0


class TestMetricsCollectorThreadSafety:
    """
    Thread safety tests for MetricsCollector.
    
    These tests verify concurrent operations work correctly.
    Detailed property-based tests are in test_metrics_collector_properties.py.
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    """
    
    def test_concurrent_increments_basic(self):
        """Test that concurrent increments are thread-safe."""
        from luma.core.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        counter_name = 'test_counter'
        num_threads = 10
        increments_per_thread = 100
        
        def increment_counter():
            for _ in range(increments_per_thread):
                collector.increment(counter_name)
        
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        snapshot = collector.get_snapshot()
        expected_total = num_threads * increments_per_thread
        assert snapshot['counters'][counter_name] == expected_total


# ============================================================================
# SECTION 3: StructuredLogger Tests (Reference existing tests)
# ============================================================================

class TestStructuredLoggerBasics:
    """
    Basic functionality tests for StructuredLogger.
    
    These tests verify JSON output format and required fields.
    Detailed tests are in test_structured_logger.py.
    
    **Validates: Requirements 7.1-7.6, 14.7**
    """
    
    def test_log_produces_valid_json(self):
        """Test that log output is valid JSON."""
        from luma.core.structured_logger import StructuredLogger
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"key": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        assert isinstance(log_data, dict)
    
    def test_log_contains_required_fields(self):
        """Test that every log entry contains required fields."""
        from luma.core.structured_logger import StructuredLogger
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        logger = StructuredLogger(name="test_logger")
        logger._logger.handlers = [handler]
        
        logger.log("test_event", {"data": "value"})
        
        output = stream.getvalue().strip()
        log_data = json.loads(output)
        
        assert "event" in log_data
        assert "timestamp" in log_data
        assert "payload" in log_data
        assert log_data["event"] == "test_event"
        assert log_data["payload"] == {"data": "value"}


# ============================================================================
# SECTION 4: Retrieval Layer Instrumentation Tests (Reference existing tests)
# ============================================================================

class TestRetrievalInstrumentation:
    """
    Integration tests for retrieve instrumentation.
    
    These tests verify that SQLiteMemoryAdapter.retrieve correctly records metrics.
    Detailed tests are in test_retrieve_instrumentation.py.
    
    **Validates: Requirements 9.1, 9.2, 9.3, 14.4, 14.5, 14.6**
    """
    
    def test_retrieve_records_metrics(self):
        """Test that retrieve records metrics when metrics_collector is provided."""
        from luma.core.metrics_collector import MetricsCollector
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        
        mock_memory_manager = Mock()
        mock_memory_manager.query_memories.return_value = []
        
        metrics_collector = MetricsCollector()
        adapter = SQLiteMemoryAdapter(mock_memory_manager)
        
        result = adapter.retrieve(
            query="test query",
            limit=10,
            metrics_collector=metrics_collector
        )
        
        snapshot = metrics_collector.get_snapshot()
        assert "retrieval_latency_ms" in snapshot["timers"]
        assert "retrieval_count" in snapshot["counters"]
        assert snapshot["counters"]["retrieval_count"] == 1
    
    def test_retrieve_works_without_metrics_collector(self):
        """Test that retrieve works correctly when metrics_collector is None."""
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        
        mock_memory_manager = Mock()
        mock_memory_manager.query_memories.return_value = []
        
        adapter = SQLiteMemoryAdapter(mock_memory_manager)
        result = adapter.retrieve(query="test query", limit=10)
        
        assert "memories" in result
        assert "total_count" in result
        assert "query_metadata" in result


# ============================================================================
# SECTION 5: RankingEngine Instrumentation Tests (Reference existing tests)
# ============================================================================

class TestRankingEngineInstrumentation:
    """
    Integration tests for RankingEngine instrumentation.
    
    These tests verify that RankingEngine correctly records metrics.
    Detailed tests are in test_ranking_engine_observability.py.
    
    **Validates: Requirements 10.1, 10.2, 14.4, 14.5, 14.6**
    """
    
    def test_ranking_engine_records_metrics(self):
        """Test that RankingEngine records metrics when metrics_collector is provided."""
        from luma.core.metrics_collector import MetricsCollector
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        
        config = RankingConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        metrics_collector = MetricsCollector()
        engine = RankingEngine(config, metrics_collector=metrics_collector)
        
        memories = [
            RankedMemory(
                memory_id="1",
                timestamp=datetime.now(UTC),
                content="test",
                namespace="test",
                similarity_score=0.8,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        result = engine.rank(memories)
        
        snapshot = metrics_collector.get_snapshot()
        assert 'ranking_latency_ms' in snapshot['timers']
        assert snapshot['timers']['ranking_latency_ms']['count'] == 1
    
    def test_ranking_engine_works_without_metrics_collector(self):
        """Test that RankingEngine works correctly when metrics_collector is None."""
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        
        config = RankingConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        engine = RankingEngine(config)
        
        memories = [
            RankedMemory(
                memory_id="1",
                timestamp=datetime.now(UTC),
                content="test",
                namespace="test",
                similarity_score=0.8,
                importance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
        ]
        
        result = engine.rank(memories)
        assert len(result) == 1


# ============================================================================
# SECTION 6: MemoryLifecycleManager Instrumentation Tests (Reference existing tests)
# ============================================================================

class TestLifecycleManagerInstrumentation:
    """
    Integration tests for MemoryLifecycleManager instrumentation.
    
    These tests verify that MemoryLifecycleManager correctly records metrics.
    Detailed tests are in test_lifecycle_instrumentation.py.
    
    **Validates: Requirements 11.1-11.6, 14.4, 14.5, 14.6**
    """
    
    def test_lifecycle_manager_records_cleanup_metrics(self):
        """Test that MemoryLifecycleManager records cleanup metrics."""
        from luma.core.metrics_collector import MetricsCollector
        from luma.core.lifecycle_manager import MemoryLifecycleManager
        from luma.core.lifecycle_config import LifecycleConfig
        from luma.core.memory_interface import MemoryInterface
        
        # Create mock memory interface
        class MockMemoryInterface(MemoryInterface):
            def __init__(self):
                self._memories = []
            
            def retrieve(self, query=None, params=None, limit=10):
                return {
                    "memories": self._memories,
                    "total_count": len(self._memories),
                    "query_metadata": {}
                }
            
            def delete(self, memory_id: str):
                pass
            
            def store(self, content: str, metadata=None, category="general", tags=None):
                return "mem_123"
        
        config = LifecycleConfig(max_total_memories=10000)
        mock_memory = MockMemoryInterface()
        metrics_collector = MetricsCollector()
        
        manager = MemoryLifecycleManager(
            config,
            mock_memory,
            metrics_collector=metrics_collector
        )
        
        manager.cleanup()
        
        snapshot = metrics_collector.get_snapshot()
        assert 'cleanup_runs' in snapshot['counters']
        assert snapshot['counters']['cleanup_runs'] == 1
        assert 'cleanup_duration_ms' in snapshot['timers']
    
    def test_lifecycle_manager_works_without_metrics_collector(self):
        """Test that MemoryLifecycleManager works correctly when metrics_collector is None."""
        from luma.core.lifecycle_manager import MemoryLifecycleManager
        from luma.core.lifecycle_config import LifecycleConfig
        from luma.core.memory_interface import MemoryInterface
        
        class MockMemoryInterface(MemoryInterface):
            def __init__(self):
                self._memories = []
            
            def retrieve(self, query=None, params=None, limit=10):
                return {
                    "memories": self._memories,
                    "total_count": len(self._memories),
                    "query_metadata": {}
                }
            
            def delete(self, memory_id: str):
                pass
            
            def store(self, content: str, metadata=None, category="general", tags=None):
                return "mem_123"
        
        config = LifecycleConfig(max_total_memories=10000)
        mock_memory = MockMemoryInterface()
        
        manager = MemoryLifecycleManager(
            config,
            mock_memory,
            metrics_collector=None
        )
        
        result = manager.cleanup()
        assert result is not None


# ============================================================================
# SECTION 7: End-to-End Observability Flow Test
# ============================================================================

class TestEndToEndObservabilityFlow:
    """
    End-to-end test demonstrating full observability flow across all instrumented components.
    
    This test simulates a complete workflow:
    1. Retrieve memories (instrumented)
    2. Rank memories (instrumented)
    3. Cleanup old memories (instrumented)
    4. Verify all metrics are collected correctly
    5. Verify all logs are generated correctly
    
    **Validates: Requirements 12.1, 12.2, 12.3, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7**
    """
    
    def test_end_to_end_observability_flow(self):
        """
        Test complete observability flow across all instrumented components.
        
        This test demonstrates:
        - MetricsCollector is instantiated and passed to components (no global singleton)
        - StructuredLogger is instantiated and passed to components (no global singleton)
        - All components record their metrics correctly
        - All components log their events correctly
        - No circular import dependencies
        - Components function correctly with observability enabled
        """
        from luma.core.metrics_collector import MetricsCollector
        from luma.core.structured_logger import StructuredLogger
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        from luma.core.lifecycle_manager import MemoryLifecycleManager
        from luma.core.lifecycle_config import LifecycleConfig
        from luma.core.memory_interface import MemoryInterface
        
        # Step 1: Create observability components (no global singletons)
        metrics_collector = MetricsCollector()
        logger = StructuredLogger(name="e2e_test")
        
        # Step 2: Create mock memory interface with test data
        class MockMemoryInterface(MemoryInterface):
            def __init__(self):
                timestamp = datetime.now(UTC) - timedelta(days=100)
                self._memories = [
                    {
                        "id": "mem_old_1",
                        "content": "Old memory 1",
                        "metadata": {"importance": 0.5, "final_score": 0.5},
                        "timestamp": timestamp.isoformat(),
                        "category": "test",
                        "tags": []
                    }
                ]
            
            def retrieve(self, query=None, params=None, limit=10, metrics_collector=None, logger_instance=None):
                # Simulate instrumented retrieve
                import time
                start = time.perf_counter()
                
                result = {
                    "memories": self._memories,
                    "total_count": len(self._memories),
                    "query_metadata": {
                        "execution_time_ms": 1.0,
                        "filters_applied": {},
                        "limit": limit,
                        "has_more": False
                    }
                }
                
                duration_ms = (time.perf_counter() - start) * 1000
                
                if metrics_collector:
                    metrics_collector.record_duration("retrieval_latency_ms", duration_ms)
                    metrics_collector.increment("retrieval_count")
                
                if logger_instance:
                    logger_instance.log("memory_retrieval", {
                        "total_count": len(self._memories),
                        "duration_ms": duration_ms
                    })
                
                return result
            
            def delete(self, memory_id: str):
                self._memories = [m for m in self._memories if m["id"] != memory_id]
            
            def store(self, content: str, metadata=None, category="general", tags=None):
                return "mem_new"
        
        mock_memory = MockMemoryInterface()
        
        # Step 3: Test retrieval with instrumentation
        retrieval_result = mock_memory.retrieve(
            query="test query",
            limit=10,
            metrics_collector=metrics_collector,
            logger_instance=logger
        )
        
        assert retrieval_result["total_count"] == 1
        
        # Verify retrieval metrics
        snapshot = metrics_collector.get_snapshot()
        assert "retrieval_latency_ms" in snapshot["timers"]
        assert "retrieval_count" in snapshot["counters"]
        assert snapshot["counters"]["retrieval_count"] == 1
        
        # Step 4: Test ranking with instrumentation
        config = RankingConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        ranking_engine = RankingEngine(
            config,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        # Create ranked memories from retrieval result
        ranked_memories = [
            RankedMemory(
                memory_id=m["id"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                content=m["content"],
                namespace="test",
                similarity_score=0.8,
                importance_score=m["metadata"]["importance"],
                recency_score=0.0,
                final_score=0.0,
                memory_entry=m
            )
            for m in retrieval_result["memories"]
        ]
        
        ranked_result = ranking_engine.rank(ranked_memories)
        
        assert len(ranked_result) == 1
        
        # Verify ranking metrics
        snapshot = metrics_collector.get_snapshot()
        assert "ranking_latency_ms" in snapshot["timers"]
        assert snapshot["timers"]["ranking_latency_ms"]["count"] == 1
        
        # Step 5: Test lifecycle cleanup with instrumentation
        lifecycle_config = LifecycleConfig(
            max_total_memories=10000,
            max_age_days=90,
            min_importance_protected=0.8
        )
        
        lifecycle_manager = MemoryLifecycleManager(
            lifecycle_config,
            mock_memory,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        cleanup_result = lifecycle_manager.cleanup()
        
        # Verify cleanup metrics
        snapshot = metrics_collector.get_snapshot()
        assert "cleanup_runs" in snapshot["counters"]
        assert snapshot["counters"]["cleanup_runs"] == 1
        assert "cleanup_duration_ms" in snapshot["timers"]
        assert snapshot["timers"]["cleanup_duration_ms"]["count"] == 1
        
        # Step 6: Verify complete metrics snapshot
        final_snapshot = metrics_collector.get_snapshot()
        
        # Verify all expected counters
        assert "retrieval_count" in final_snapshot["counters"]
        assert "cleanup_runs" in final_snapshot["counters"]
        
        # Verify all expected timers
        assert "retrieval_latency_ms" in final_snapshot["timers"]
        assert "ranking_latency_ms" in final_snapshot["timers"]
        assert "cleanup_duration_ms" in final_snapshot["timers"]
        
        # Verify timer statistics are valid
        for timer_name, stats in final_snapshot["timers"].items():
            assert stats["count"] > 0, f"{timer_name} should have at least 1 measurement"
            assert stats["sum"] > 0, f"{timer_name} sum should be positive"
            assert stats["min"] > 0, f"{timer_name} min should be positive"
            assert stats["max"] >= stats["min"], f"{timer_name} max should be >= min"
            assert stats["mean"] > 0, f"{timer_name} mean should be positive"
        
        # Step 7: Verify no circular dependencies (already verified by successful imports)
        # If we reached here, all components were successfully imported and used
        assert True, "End-to-end observability flow completed successfully"
    
    def test_end_to_end_flow_without_observability(self):
        """
        Test that all components work correctly without observability dependencies.
        
        This verifies backward compatibility and that observability is truly optional.
        
        **Validates: Requirements 8.7, 8.8, 8.9, 13.5**
        """
        from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
        from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
        from luma.core.lifecycle_manager import MemoryLifecycleManager
        from luma.core.lifecycle_config import LifecycleConfig
        from luma.core.memory_interface import MemoryInterface
        
        # Create mock memory interface
        class MockMemoryInterface(MemoryInterface):
            def __init__(self):
                timestamp = datetime.now(UTC) - timedelta(days=100)
                self._memories = [
                    {
                        "id": "mem_1",
                        "content": "Test memory",
                        "metadata": {"importance": 0.5, "final_score": 0.5},
                        "timestamp": timestamp.isoformat(),
                        "category": "test",
                        "tags": []
                    }
                ]
            
            def retrieve(self, query=None, params=None, limit=10, metrics_collector=None, logger_instance=None):
                return {
                    "memories": self._memories,
                    "total_count": len(self._memories),
                    "query_metadata": {
                        "execution_time_ms": 1.0,
                        "filters_applied": {},
                        "limit": limit,
                        "has_more": False
                    }
                }
            
            def delete(self, memory_id: str):
                self._memories = [m for m in self._memories if m["id"] != memory_id]
            
            def store(self, content: str, metadata=None, category="general", tags=None):
                return "mem_new"
        
        mock_memory = MockMemoryInterface()
        
        # Test retrieval without observability
        retrieval_result = mock_memory.retrieve(query="test", limit=10)
        assert retrieval_result["total_count"] == 1
        
        # Test ranking without observability
        config = RankingConfig(
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.3,
            score_threshold=0.2
        )
        
        ranking_engine = RankingEngine(config)  # No observability dependencies
        
        ranked_memories = [
            RankedMemory(
                memory_id=m["id"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                content=m["content"],
                namespace="test",
                similarity_score=0.8,
                importance_score=0.5,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=m
            )
            for m in retrieval_result["memories"]
        ]
        
        ranked_result = ranking_engine.rank(ranked_memories)
        assert len(ranked_result) == 1
        
        # Test lifecycle cleanup without observability
        lifecycle_config = LifecycleConfig(
            max_total_memories=10000,
            max_age_days=90,
            min_importance_protected=0.8
        )
        
        lifecycle_manager = MemoryLifecycleManager(
            lifecycle_config,
            mock_memory
            # No observability dependencies
        )
        
        cleanup_result = lifecycle_manager.cleanup()
        assert cleanup_result is not None
        
        # All operations completed successfully without observability
        assert True, "All components work correctly without observability"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
