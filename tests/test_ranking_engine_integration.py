"""Integration tests for RankingEngine."""

import pytest
from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory


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


def test_complete_ranking_pipeline():
    """Test complete ranking pipeline with various configurations."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2,
        namespace=None
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, 0.7, timestamp=base_time - timedelta(hours=1)),
        create_test_memory("2", 0.6, 0.5, timestamp=base_time),
        create_test_memory("3", 0.9, 0.8, timestamp=base_time - timedelta(hours=2)),
        create_test_memory("4", 0.2, 0.1, timestamp=base_time),  # Below threshold
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Memory 4 should be filtered out (similarity below threshold)
    assert len(result) == 3
    
    # Check that scores were computed
    for memory in result:
        assert memory.recency_score > 0
        assert memory.final_score > 0
    
    # Check that results are sorted by final_score
    for i in range(len(result) - 1):
        assert result[i].final_score >= result[i + 1].final_score


def test_empty_input_handling():
    """Test that empty input returns empty output."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2
    )
    
    engine = RankingEngine(config)
    result = engine.rank([])
    
    assert len(result) == 0


def test_all_memories_filtered_scenario():
    """Test scenario where all memories are filtered by thresholds."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.9,  # Very high threshold
        score_threshold=0.9
    )
    
    memories = [
        create_test_memory("1", 0.5, 0.0),
        create_test_memory("2", 0.6, 0.0),
        create_test_memory("3", 0.7, 0.0),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories)
    
    assert len(result) == 0


def test_namespace_filtering_integration():
    """Test namespace filtering integration."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0,
        namespace="conversation"
    )
    
    memories = [
        create_test_memory("1", 0.8, namespace="conversation"),
        create_test_memory("2", 0.9, namespace="system"),
        create_test_memory("3", 0.7, namespace="conversation"),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories)
    
    # Only conversation namespace should be included
    assert len(result) == 2
    assert all(m.namespace == "conversation" for m in result)


def test_threshold_filtering_integration():
    """Test threshold filtering integration."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, timestamp=base_time),
        create_test_memory("2", 0.3, timestamp=base_time),  # Below similarity threshold
        create_test_memory("3", 0.6, timestamp=base_time),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Memory 2 should be filtered out
    assert len(result) == 2
    assert all(m.similarity_score >= 0.5 for m in result)
    assert all(m.final_score >= 0.3 for m in result)


def test_recency_scoring_integration():
    """Test that recency scoring works correctly in the pipeline."""
    config = RankingConfig(
        alpha=0.0,  # No similarity weight
        beta=1.0,   # Only recency
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("old", 0.5, timestamp=base_time - timedelta(hours=10)),
        create_test_memory("new", 0.5, timestamp=base_time),
        create_test_memory("middle", 0.5, timestamp=base_time - timedelta(hours=5)),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Should be ordered by recency (newest first)
    assert len(result) == 3
    assert result[0].memory_id == "new"
    assert result[1].memory_id == "middle"
    assert result[2].memory_id == "old"


def test_importance_scoring_integration():
    """Test that importance scoring works correctly in the pipeline."""
    config = RankingConfig(
        alpha=0.0,
        beta=0.0,
        gamma=1.0,  # Only importance
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("low", 0.5, importance_score=0.3, timestamp=base_time),
        create_test_memory("high", 0.5, importance_score=0.9, timestamp=base_time),
        create_test_memory("medium", 0.5, importance_score=0.6, timestamp=base_time),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Should be ordered by importance (highest first)
    assert len(result) == 3
    assert result[0].memory_id == "high"
    assert result[1].memory_id == "medium"
    assert result[2].memory_id == "low"


def test_default_current_time():
    """Test that current_time defaults to now when not provided."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    memories = [
        create_test_memory("1", 0.8),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories)  # No current_time provided
    
    assert len(result) == 1
    assert result[0].recency_score > 0


def test_invalid_config_raises_error():
    """Test that invalid configuration raises error."""
    with pytest.raises(ValueError):
        config = RankingConfig(
            alpha=0.5,
            beta=0.3,  # Sum != 1.0
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.0,
            score_threshold=0.0
        )
        engine = RankingEngine(config)


def test_balanced_scoring():
    """Test balanced scoring with all three components."""
    config = RankingConfig(
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        # High similarity, low recency, low importance
        create_test_memory("1", 0.9, 0.1, timestamp=base_time - timedelta(hours=10)),
        # Medium similarity, high recency, medium importance
        create_test_memory("2", 0.6, 0.5, timestamp=base_time),
        # Low similarity, medium recency, high importance
        create_test_memory("3", 0.3, 0.9, timestamp=base_time - timedelta(hours=5)),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # All should be included
    assert len(result) == 3
    
    # Check that final scores are computed correctly
    for memory in result:
        assert 0 <= memory.final_score <= 1
        assert memory.recency_score > 0
