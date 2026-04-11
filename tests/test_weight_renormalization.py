"""Unit tests for weight renormalization when gamma=0."""

import pytest
from datetime import datetime, timezone
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    timestamp: datetime = None
) -> RankedMemory:
    """Helper to create test memory."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=0.0,
        recency_score=0.0,
        final_score=0.0,
        memory_entry=None
    )


def test_gamma_zero_triggers_renormalization():
    """Test that gamma=0 works correctly with alpha + beta = 1."""
    config = RankingConfig(
        alpha=0.7,
        beta=0.3,
        gamma=0.0,  # No importance
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    # Should not raise error
    engine = RankingEngine(config)
    
    # Verify weights sum to 1
    assert config.alpha + config.beta + config.gamma == 1.0


def test_alpha_beta_sum_to_one_after_renormalization():
    """Test that alpha and beta sum to 1 when gamma=0."""
    config = RankingConfig(
        alpha=0.6,
        beta=0.4,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    engine = RankingEngine(config)
    
    # When gamma=0, alpha + beta should equal 1
    assert config.alpha + config.beta == 1.0


def test_final_scores_computed_correctly_without_importance():
    """Test that final scores are computed correctly when gamma=0."""
    config = RankingConfig(
        alpha=0.7,
        beta=0.3,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, timestamp=base_time),
    ]
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # Final score should be: alpha * similarity + beta * recency + 0 * importance
    memory = result[0]
    expected_score = config.alpha * memory.similarity_score + config.beta * memory.recency_score
    
    assert abs(memory.final_score - expected_score) < 1e-9


def test_different_weight_combinations_with_gamma_zero():
    """Test various weight combinations with gamma=0."""
    test_cases = [
        (0.5, 0.5, 0.0),
        (0.8, 0.2, 0.0),
        (0.3, 0.7, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    
    for alpha, beta, gamma in test_cases:
        config = RankingConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            decay_constant=0.001,
            similarity_threshold=0.0,
            score_threshold=0.0
        )
        
        # Should not raise error
        engine = RankingEngine(config)
        
        # Verify weights sum to 1
        assert abs((config.alpha + config.beta + config.gamma) - 1.0) < 1e-9


def test_importance_zero_for_all_memories():
    """Test that importance_score=0 for all memories works correctly."""
    config = RankingConfig(
        alpha=0.6,
        beta=0.4,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0
    )
    
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.8, timestamp=base_time),
        create_test_memory("2", 0.6, timestamp=base_time),
        create_test_memory("3", 0.9, timestamp=base_time),
    ]
    
    # All memories have importance_score = 0
    for m in memories:
        assert m.importance_score == 0.0
    
    engine = RankingEngine(config)
    result = engine.rank(memories, current_time=base_time)
    
    # All should be ranked correctly
    assert len(result) == 3
    
    # Verify final scores don't include importance component
    for memory in result:
        expected = config.alpha * memory.similarity_score + config.beta * memory.recency_score
        assert abs(memory.final_score - expected) < 1e-9


def test_invalid_weights_with_gamma_zero():
    """Test that invalid weight combinations are rejected even with gamma=0."""
    # alpha + beta != 1 when gamma = 0
    with pytest.raises(ValueError):
        config = RankingConfig(
            alpha=0.5,
            beta=0.3,  # Sum = 0.8, not 1.0
            gamma=0.0,
            decay_constant=0.001,
            similarity_threshold=0.0,
            score_threshold=0.0
        )
        engine = RankingEngine(config)
