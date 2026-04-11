"""Unit tests for ThresholdFilter component."""

import pytest
from datetime import datetime, timezone
from luma.core.ranking_engine import ThresholdFilter, RankedMemory, RankingConfig


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    final_score: float
) -> RankedMemory:
    """Helper to create test memory."""
    return RankedMemory(
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc),
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=0.0,
        recency_score=0.0,
        final_score=final_score,
        memory_entry=None
    )


def test_filter_by_similarity_threshold():
    """Test filtering by similarity threshold."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.0
    )
    
    memories = [
        create_test_memory("1", similarity_score=0.8, final_score=0.7),
        create_test_memory("2", similarity_score=0.3, final_score=0.7),
        create_test_memory("3", similarity_score=0.6, final_score=0.7),
    ]
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(memories)
    
    assert len(result) == 2
    assert result[0].memory_id == "1"
    assert result[1].memory_id == "3"


def test_filter_by_score_threshold():
    """Test filtering by score threshold."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.5
    )
    
    memories = [
        create_test_memory("1", similarity_score=0.8, final_score=0.7),
        create_test_memory("2", similarity_score=0.8, final_score=0.3),
        create_test_memory("3", similarity_score=0.8, final_score=0.6),
    ]
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(memories)
    
    assert len(result) == 2
    assert result[0].memory_id == "1"
    assert result[1].memory_id == "3"


def test_filter_by_both_thresholds():
    """Test filtering by both thresholds."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.5
    )
    
    memories = [
        create_test_memory("1", similarity_score=0.8, final_score=0.7),
        create_test_memory("2", similarity_score=0.3, final_score=0.7),
        create_test_memory("3", similarity_score=0.8, final_score=0.3),
        create_test_memory("4", similarity_score=0.6, final_score=0.6),
    ]
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(memories)
    
    assert len(result) == 2
    assert result[0].memory_id == "1"
    assert result[1].memory_id == "4"


def test_filter_empty_result():
    """Test empty result when all memories filtered."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.9,
        score_threshold=0.9
    )
    
    memories = [
        create_test_memory("1", similarity_score=0.5, final_score=0.5),
        create_test_memory("2", similarity_score=0.6, final_score=0.6),
    ]
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(memories)
    
    assert len(result) == 0


def test_filter_empty_input():
    """Test filtering with empty input list."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.5
    )
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter([])
    
    assert len(result) == 0


def test_filter_boundary_values():
    """Test filtering with boundary values (exact threshold match)."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.5
    )
    
    memories = [
        create_test_memory("1", similarity_score=0.5, final_score=0.5),  # Exactly at threshold
        create_test_memory("2", similarity_score=0.49999, final_score=0.5),  # Just below
        create_test_memory("3", similarity_score=0.5, final_score=0.49999),  # Just below
    ]
    
    filter_obj = ThresholdFilter(config)
    result = filter_obj.filter(memories)
    
    assert len(result) == 1
    assert result[0].memory_id == "1"
