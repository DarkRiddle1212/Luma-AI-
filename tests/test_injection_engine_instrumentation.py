"""
Integration tests for InjectionEngine instrumentation.

Tests verify that InjectionEngine correctly records metrics and handles
observability components (metrics_collector, logger) according to requirements.

**Validates: Requirements 14.4, 14.5, 14.6**
"""

import pytest
from datetime import datetime, timezone
from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig,
    InjectionResult,
    InjectedMemory
)
from luma.core.ranking_engine import RankedMemory
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


def create_test_memory(
    memory_id: str,
    content: str,
    category: str = "test",
    similarity_score: float = 0.8,
    final_score: float = 0.8,
    token_count: int = 10,
    embedding: list = None
) -> RankedMemory:
    """Helper to create test RankedMemory objects."""
    if embedding is None:
        # Create a unique embedding based on memory_id to avoid redundancy filtering
        try:
            num = int(memory_id.split('_')[-1])
            # Create embeddings with different "directions" to ensure low similarity
            if num == 1:
                embedding = [1.0, 0.0, 0.0]
            elif num == 2:
                embedding = [0.0, 1.0, 0.0]
            elif num == 3:
                embedding = [0.0, 0.0, 1.0]
            else:
                # For other numbers, use a mix
                embedding = [0.5, 0.5, float(num) * 0.1]
        except (ValueError, IndexError):
            # Fallback for non-standard memory_ids
            embedding = [0.1, 0.2, 0.3]
    
    return RankedMemory(
        memory_id=memory_id,
        content=content,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        namespace="test",
        category=category,
        similarity_score=similarity_score,
        final_score=final_score,
        recency_score=0.9,
        importance_score=0.7,
        metadata={
            "token_count": token_count,
            "embedding": embedding,
            "source": "test"
        },
        memory_entry=None
    )


class TestInjectionEngineInstrumentation:
    """
    Integration tests for InjectionEngine instrumentation.
    
    These tests verify that InjectionEngine correctly records metrics and logs
    when observability components are provided, and works correctly when they
    are not provided.
    
    **Validates: Requirements 14.4, 14.5, 14.6**
    """
    
    def test_injection_engine_latency_ms_recorded_with_metrics_collector(self):
        """Test that injection_engine_latency_ms is recorded when metrics_collector is provided."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create metrics collector
        metrics_collector = MetricsCollector()
        
        # Create injection engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
            create_test_memory("mem_2", "Second memory", token_count=60),
        ]
        
        # Perform injection
        result = engine.inject(memories)
        
        # Verify metrics were recorded
        snapshot = metrics_collector.get_snapshot()
        
        # Check that injection_engine_latency_ms timer exists and has one measurement
        assert 'injection_engine_latency_ms' in snapshot['timers']
        timer_stats = snapshot['timers']['injection_engine_latency_ms']
        assert timer_stats['count'] == 1
        assert timer_stats['sum'] > 0  # Should have some positive duration
        assert timer_stats['min'] > 0
        assert timer_stats['max'] > 0
        assert timer_stats['mean'] > 0
        
        # Verify injection still worked correctly
        assert isinstance(result, InjectionResult)
        assert len(result.memories) == 2  # Both memories should fit in budget
    
    def test_injection_engine_count_incremented_with_metrics_collector(self):
        """Test that injection_engine_count is incremented when metrics_collector is provided."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create metrics collector
        metrics_collector = MetricsCollector()
        
        # Create injection engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
        ]
        
        # Perform multiple injections
        result1 = engine.inject(memories)
        result2 = engine.inject(memories)
        
        # Verify counter was incremented
        snapshot = metrics_collector.get_snapshot()
        
        # Check that injection_engine_count counter exists and was incremented twice
        assert 'injection_engine_count' in snapshot['counters']
        assert snapshot['counters']['injection_engine_count'] == 2
        
        # Verify injections still worked correctly
        assert isinstance(result1, InjectionResult)
        assert isinstance(result2, InjectionResult)
        assert len(result1.memories) == 1
        assert len(result2.memories) == 1
    
    def test_injection_engine_metrics_recorded_for_empty_input(self):
        """Test that metrics are recorded even for empty input."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create metrics collector
        metrics_collector = MetricsCollector()
        
        # Create injection engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Perform injection with empty input
        result = engine.inject([])
        
        # Verify metrics were recorded even for empty input
        snapshot = metrics_collector.get_snapshot()
        
        # Check that metrics were recorded
        assert 'injection_engine_latency_ms' in snapshot['timers']
        assert snapshot['timers']['injection_engine_latency_ms']['count'] == 1
        assert 'injection_engine_count' in snapshot['counters']
        assert snapshot['counters']['injection_engine_count'] == 1
        
        # Verify empty result
        assert isinstance(result, InjectionResult)
        assert len(result.memories) == 0
        assert result.input_count == 0
    
    def test_no_exceptions_when_metrics_collector_is_none(self):
        """Test that no exceptions occur when metrics_collector is None."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create injection engine without metrics collector
        engine = InjectionEngine(config, metrics_collector=None)
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
            create_test_memory("mem_2", "Second memory", token_count=60),
        ]
        
        # Perform injection - should not raise any exceptions
        result = engine.inject(memories)
        
        # Verify injection worked correctly
        assert isinstance(result, InjectionResult)
        assert len(result.memories) == 2  # Both memories should fit in budget
        assert result.input_count == 2
        assert result.total_tokens > 0
    
    def test_no_exceptions_when_logger_is_none(self):
        """Test that no exceptions occur when logger is None."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create injection engine without logger
        engine = InjectionEngine(config, logger=None)
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
            create_test_memory("mem_2", "Second memory", token_count=60),
        ]
        
        # Perform injection - should not raise any exceptions
        result = engine.inject(memories)
        
        # Verify injection worked correctly
        assert isinstance(result, InjectionResult)
        assert len(result.memories) == 2  # Both memories should fit in budget
        assert result.input_count == 2
        assert result.total_tokens > 0
    
    def test_injection_results_identical_with_and_without_instrumentation(self):
        """Test that injection results are identical with and without instrumentation."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
            create_test_memory("mem_2", "Second memory", token_count=60),
            create_test_memory("mem_3", "Third memory", token_count=70),
        ]
        
        # Create injection engine without instrumentation
        engine_without = InjectionEngine(config)
        result_without = engine_without.inject(memories)
        
        # Create injection engine with instrumentation
        metrics_collector = MetricsCollector()
        logger = StructuredLogger("test_injection")
        engine_with = InjectionEngine(config, metrics_collector=metrics_collector, logger=logger)
        result_with = engine_with.inject(memories)
        
        # Verify results are identical
        assert len(result_without.memories) == len(result_with.memories)
        assert result_without.total_tokens == result_with.total_tokens
        assert result_without.input_count == result_with.input_count
        assert result_without.filtered_by_category == result_with.filtered_by_category
        assert result_without.filtered_by_redundancy == result_with.filtered_by_redundancy
        assert result_without.filtered_by_budget == result_with.filtered_by_budget
        
        # Verify memory contents are identical
        for mem_without, mem_with in zip(result_without.memories, result_with.memories):
            assert mem_without.memory_id == mem_with.memory_id
            assert mem_without.content == mem_with.content
            assert mem_without.metadata == mem_with.metadata
            assert mem_without.similarity_score == mem_with.similarity_score
            assert mem_without.timestamp == mem_with.timestamp
            assert mem_without.category == mem_with.category
    
    def test_injection_engine_metrics_recorded_on_exception(self):
        """Test that metrics are recorded even when injection fails with exception."""
        # Create test configuration that will cause an exception
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create metrics collector
        metrics_collector = MetricsCollector()
        
        # Create injection engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Create invalid input that should cause an exception
        # (using None instead of proper RankedMemory objects)
        invalid_memories = [None]
        
        # Perform injection - should raise an exception but still record metrics
        with pytest.raises(AttributeError):  # Will fail when trying to access attributes of None
            engine.inject(invalid_memories)
        
        # Verify metrics were still recorded despite the exception
        snapshot = metrics_collector.get_snapshot()
        
        # Check that metrics were recorded
        assert 'injection_engine_latency_ms' in snapshot['timers']
        assert snapshot['timers']['injection_engine_latency_ms']['count'] == 1
        assert 'injection_engine_count' in snapshot['counters']
        assert snapshot['counters']['injection_engine_count'] == 1
    
    def test_injection_engine_with_both_metrics_and_logger(self):
        """Test that injection engine works correctly with both metrics_collector and logger."""
        # Create test configuration
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False,
            token_estimation_factor=1.3
        )
        
        # Create both metrics collector and logger
        metrics_collector = MetricsCollector()
        logger = StructuredLogger("test_injection")
        
        # Create injection engine with both observability components
        engine = InjectionEngine(config, metrics_collector=metrics_collector, logger=logger)
        
        # Create test memories
        memories = [
            create_test_memory("mem_1", "First memory", token_count=50),
            create_test_memory("mem_2", "Second memory", token_count=60),
        ]
        
        # Perform injection
        result = engine.inject(memories)
        
        # Verify metrics were recorded
        snapshot = metrics_collector.get_snapshot()
        assert 'injection_engine_latency_ms' in snapshot['timers']
        assert snapshot['timers']['injection_engine_latency_ms']['count'] == 1
        assert 'injection_engine_count' in snapshot['counters']
        assert snapshot['counters']['injection_engine_count'] == 1
        
        # Verify injection worked correctly
        assert isinstance(result, InjectionResult)
        assert len(result.memories) == 2
        assert result.input_count == 2
        assert result.total_tokens > 0