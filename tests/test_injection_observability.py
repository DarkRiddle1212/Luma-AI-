"""
Unit tests for InjectionEngine observability instrumentation.

This module tests that the InjectionEngine correctly records metrics and logs
events when metrics_collector and logger are provided.

Requirements tested:
- 1.1: Accept list of RankedMemory objects and record observability metrics
"""

from unittest.mock import Mock
from datetime import datetime, timezone
from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig
)
from luma.core.ranking_engine import RankedMemory


class TestInjectionObservability:
    """Test suite for InjectionEngine observability instrumentation."""
    
    def test_inject_records_latency_metric(self):
        """Test that inject() records injection_engine_latency_ms metric."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock metrics collector
        metrics_collector = Mock()
        
        # Create engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Create test memories
        memories = [
            RankedMemory(
                memory_id="mem_1",
                content="Test memory 1",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                namespace="test",
                category="test",
                similarity_score=0.9,
                final_score=0.9,
                recency_score=0.8,
                importance_score=0.7,
                metadata={"token_count": 10},
                memory_entry=None
            )
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        # Verify record_duration was called with correct metric name
        metrics_collector.record_duration.assert_called_once()
        call_args = metrics_collector.record_duration.call_args
        assert call_args[0][0] == 'injection_engine_latency_ms'
        
        # Verify duration is a positive number
        duration_ms = call_args[0][1]
        assert isinstance(duration_ms, (int, float))
        assert duration_ms >= 0
    
    def test_inject_logs_completion_event(self):
        """Test that inject() logs injection_completed event with diagnostic info."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock logger
        logger = Mock()
        
        # Create engine with logger
        engine = InjectionEngine(config, logger=logger)
        
        # Create test memories
        memories = [
            RankedMemory(
                memory_id="mem_1",
                content="Test memory 1",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                namespace="test",
                category="test",
                similarity_score=0.9,
                final_score=0.9,
                recency_score=0.8,
                importance_score=0.7,
                metadata={"token_count": 10},
                memory_entry=None
            ),
            RankedMemory(
                memory_id="mem_2",
                content="Test memory 2",
                timestamp=datetime(2024, 1, 16, tzinfo=timezone.utc),
                namespace="test",
                category="test",
                similarity_score=0.8,
                final_score=0.8,
                recency_score=0.7,
                importance_score=0.6,
                metadata={"token_count": 15},
                memory_entry=None
            )
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        # Verify log was called with correct event name and payload
        logger.log.assert_called_once()
        call_args = logger.log.call_args
        
        # Check event name
        assert call_args[0][0] == 'injection_completed'
        
        # Check payload structure
        payload = call_args[0][1]
        assert 'input_count' in payload
        assert 'output_count' in payload
        assert 'duration_ms' in payload
        
        # Check payload values
        assert payload['input_count'] == 2
        assert payload['output_count'] == 2
        assert isinstance(payload['duration_ms'], (int, float))
        assert payload['duration_ms'] >= 0
    
    def test_inject_records_metrics_and_logs_together(self):
        """Test that inject() records both metrics and logs when both are provided."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock metrics collector and logger
        metrics_collector = Mock()
        logger = Mock()
        
        # Create engine with both observability components
        engine = InjectionEngine(config, metrics_collector=metrics_collector, logger=logger)
        
        # Create test memories
        memories = [
            RankedMemory(
                memory_id="mem_1",
                content="Test memory 1",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                namespace="test",
                category="test",
                similarity_score=0.9,
                final_score=0.9,
                recency_score=0.8,
                importance_score=0.7,
                metadata={"token_count": 10},
                memory_entry=None
            )
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        # Verify both metrics and logging were called
        metrics_collector.record_duration.assert_called_once()
        logger.log.assert_called_once()
        
        # Verify they recorded similar durations (within 10ms tolerance due to separate measurements)
        metrics_duration = metrics_collector.record_duration.call_args[0][1]
        log_duration = logger.log.call_args[0][1]['duration_ms']
        assert abs(metrics_duration - log_duration) < 10.0  # Allow 10ms difference
    
    def test_inject_works_without_observability_components(self):
        """Test that inject() works correctly when no observability components are provided."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create engine without observability components
        engine = InjectionEngine(config)
        
        # Create test memories
        memories = [
            RankedMemory(
                memory_id="mem_1",
                content="Test memory 1",
                timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
                namespace="test",
                category="test",
                similarity_score=0.9,
                final_score=0.9,
                recency_score=0.8,
                importance_score=0.7,
                metadata={"token_count": 10},
                memory_entry=None
            )
        ]
        
        # Act - should not raise any errors
        result = engine.inject(memories)
        
        # Assert - verify result is correct
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_1"
        assert result.total_tokens == 10
    
    def test_inject_logs_empty_input_correctly(self):
        """Test that inject() logs correct counts for empty input."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock logger
        logger = Mock()
        
        # Create engine with logger
        engine = InjectionEngine(config, logger=logger)
        
        # Act
        result = engine.inject([])
        
        # Assert
        logger.log.assert_called_once()
        payload = logger.log.call_args[0][1]
        
        # Check that counts are zero for empty input
        assert payload['input_count'] == 0
        assert payload['output_count'] == 0
        assert payload['duration_ms'] >= 0
    
    def test_inject_records_metrics_for_empty_input(self):
        """Test that inject() records metrics even for empty input."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock metrics collector
        metrics_collector = Mock()
        
        # Create engine with metrics collector
        engine = InjectionEngine(config, metrics_collector=metrics_collector)
        
        # Act
        result = engine.inject([])
        
        # Assert
        # Verify metrics were recorded even for empty input
        metrics_collector.record_duration.assert_called_once()
        call_args = metrics_collector.record_duration.call_args
        assert call_args[0][0] == 'injection_engine_latency_ms'
        assert call_args[0][1] >= 0
    
    def test_inject_logs_on_exception(self):
        """Test that inject() logs even when an exception occurs during processing."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Create mock logger
        logger = Mock()
        
        # Create engine with logger
        engine = InjectionEngine(config, logger=logger)
        
        # Create a memory with invalid data that will cause an error during processing
        # We'll use a mock that raises an exception when accessed
        invalid_memory = Mock()
        invalid_memory.memory_id = "invalid"
        # Make the filter method raise an exception
        engine.category_filter.filter = Mock(side_effect=RuntimeError("Test error"))
        
        # Act & Assert
        try:
            result = engine.inject([invalid_memory])
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert str(e) == "Test error"
        
        # Verify that logging still occurred in the exception handler
        logger.log.assert_called_once()
        event_name = logger.log.call_args[0][0]
        payload = logger.log.call_args[0][1]
        
        # Check that the logger handled the exception gracefully
        assert event_name == 'injection_failed'
        assert 'duration_ms' in payload
        assert payload['duration_ms'] >= 0
        # input_count should be 1 since it was set before the exception
        assert payload['input_count'] == 1
        # injection_failed event includes error info instead of output_count
        assert 'error_type' in payload
        assert payload['error_type'] == 'RuntimeError'
        assert 'error_message' in payload
        assert payload['error_message'] == 'Test error'
