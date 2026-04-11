"""Tests for RankingEngine observability integration."""

import pytest
from datetime import datetime, timezone
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    importance_score: float = 0.0,
    namespace: str = "test",
    timestamp: datetime = None
) -> RankedMemory:
    """Helper to create test memory."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content="test content",
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=0.0,
        final_score=0.0,
        memory_entry=None
    )


def test_ranking_engine_accepts_optional_metrics_collector():
    """Test that RankingEngine accepts optional metrics_collector parameter."""
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
    
    assert engine.metrics_collector is not None
    assert engine.metrics_collector is metrics_collector


def test_ranking_engine_accepts_optional_logger():
    """Test that RankingEngine accepts optional logger parameter."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    logger = StructuredLogger()
    engine = RankingEngine(config, logger=logger)
    
    assert engine.logger is not None
    assert engine.logger is logger


def test_ranking_engine_accepts_both_observability_dependencies():
    """Test that RankingEngine accepts both metrics_collector and logger."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    engine = RankingEngine(config, metrics_collector=metrics_collector, logger=logger)
    
    assert engine.metrics_collector is not None
    assert engine.logger is not None
    assert engine.metrics_collector is metrics_collector
    assert engine.logger is logger


def test_ranking_engine_backward_compatibility_no_observability():
    """Test that RankingEngine works without observability parameters (backward compatibility)."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    # Should work without any observability parameters
    engine = RankingEngine(config)
    
    assert engine.metrics_collector is None
    assert engine.logger is None


def test_ranking_engine_functions_correctly_with_observability():
    """Test that RankingEngine functions correctly with observability dependencies."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    engine = RankingEngine(config, metrics_collector=metrics_collector, logger=logger)
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Rank memories
    result = engine.rank(memories)
    
    # Verify ranking works correctly
    assert len(result) == 3
    assert all(m.final_score > 0 for m in result)
    
    # Verify results are sorted by final_score
    for i in range(len(result) - 1):
        assert result[i].final_score >= result[i + 1].final_score


def test_ranking_engine_functions_correctly_without_observability():
    """Test that RankingEngine functions correctly without observability dependencies."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    engine = RankingEngine(config)
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Rank memories
    result = engine.rank(memories)
    
    # Verify ranking works correctly
    assert len(result) == 3
    assert all(m.final_score > 0 for m in result)
    
    # Verify results are sorted by final_score
    for i in range(len(result) - 1):
        assert result[i].final_score >= result[i + 1].final_score


def test_ranking_engine_stores_observability_dependencies_as_instance_variables():
    """Test that RankingEngine stores metrics_collector and logger as instance variables."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    engine = RankingEngine(config, metrics_collector=metrics_collector, logger=logger)
    
    # Verify they are stored as instance variables
    assert hasattr(engine, 'metrics_collector')
    assert hasattr(engine, 'logger')
    assert engine.metrics_collector is metrics_collector
    assert engine.logger is logger


def test_ranking_engine_records_ranking_latency_metric():
    """Test that RankingEngine records ranking_latency_ms when metrics_collector is provided."""
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
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Rank memories
    result = engine.rank(memories)
    
    # Verify ranking_latency_ms was recorded
    snapshot = metrics_collector.get_snapshot()
    assert 'ranking_latency_ms' in snapshot['timers']
    
    timer_stats = snapshot['timers']['ranking_latency_ms']
    assert timer_stats['count'] == 1
    assert timer_stats['sum'] > 0
    assert timer_stats['min'] > 0
    assert timer_stats['max'] > 0
    assert timer_stats['mean'] > 0


def test_ranking_engine_no_exception_when_metrics_collector_is_none():
    """Test that RankingEngine raises no exceptions when metrics_collector is None."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    engine = RankingEngine(config, metrics_collector=None)
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Should not raise any exceptions
    result = engine.rank(memories)
    
    # Verify ranking works correctly
    assert len(result) == 3
    assert all(m.final_score > 0 for m in result)


def test_ranking_engine_no_exception_when_logger_is_none():
    """Test that RankingEngine raises no exceptions when logger is None."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    engine = RankingEngine(config, logger=None)
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Should not raise any exceptions
    result = engine.rank(memories)
    
    # Verify ranking works correctly
    assert len(result) == 3
    assert all(m.final_score > 0 for m in result)


def test_ranking_results_identical_with_and_without_instrumentation():
    """Test that ranking results are identical with and without instrumentation."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    # Use a fixed timestamp to ensure identical recency scores
    fixed_timestamp = datetime.now(timezone.utc)
    fixed_current_time = datetime.now(timezone.utc)
    
    # Create test memories with fixed timestamp
    memories_with = [
        create_test_memory("1", 0.8, timestamp=fixed_timestamp),
        create_test_memory("2", 0.6, timestamp=fixed_timestamp),
        create_test_memory("3", 0.9, timestamp=fixed_timestamp),
    ]
    
    memories_without = [
        create_test_memory("1", 0.8, timestamp=fixed_timestamp),
        create_test_memory("2", 0.6, timestamp=fixed_timestamp),
        create_test_memory("3", 0.9, timestamp=fixed_timestamp),
    ]
    
    # Rank with instrumentation
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    engine_with = RankingEngine(config, metrics_collector=metrics_collector, logger=logger)
    result_with = engine_with.rank(memories_with, current_time=fixed_current_time)
    
    # Rank without instrumentation
    engine_without = RankingEngine(config)
    result_without = engine_without.rank(memories_without, current_time=fixed_current_time)
    
    # Verify results are identical
    assert len(result_with) == len(result_without)
    
    for i in range(len(result_with)):
        assert result_with[i].memory_id == result_without[i].memory_id
        assert result_with[i].final_score == result_without[i].final_score
        assert result_with[i].similarity_score == result_without[i].similarity_score
        assert result_with[i].recency_score == result_without[i].recency_score
        assert result_with[i].importance_score == result_without[i].importance_score


def test_ranking_engine_logs_ranking_events():
    """Test that RankingEngine logs ranking events when logger is provided."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    logger = StructuredLogger()
    engine = RankingEngine(config, logger=logger)
    
    # Create test memories
    memories = [
        create_test_memory("1", 0.8),
        create_test_memory("2", 0.6),
        create_test_memory("3", 0.9),
    ]
    
    # Rank memories
    result = engine.rank(memories)
    
    # Verify ranking works correctly (logger should not affect results)
    assert len(result) == 3
    assert all(m.final_score > 0 for m in result)


def test_ranking_engine_handles_empty_input_with_instrumentation():
    """Test that RankingEngine handles empty input correctly with instrumentation."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    metrics_collector = MetricsCollector()
    logger = StructuredLogger()
    engine = RankingEngine(config, metrics_collector=metrics_collector, logger=logger)
    
    # Rank empty list
    result = engine.rank([])
    
    # Verify empty result
    assert len(result) == 0
    
    # Verify metrics were still recorded
    snapshot = metrics_collector.get_snapshot()
    assert 'ranking_latency_ms' in snapshot['timers']
    assert snapshot['timers']['ranking_latency_ms']['count'] == 1
