"""
Unit tests for ScoreComputer.

**Validates: Requirements 2.1, 2.2, 2.5, 7.3, 7.4**
"""

import pytest
import math
from datetime import datetime, timedelta, timezone
from luma.core.ranking_engine import RankingConfig, ScoreComputer, RankedMemory


def test_recency_score_current_time():
    """Test recency score for current time (age=0)."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    score = computer.compute_recency_score(current_time)
    assert score == 1.0


def test_recency_score_one_hour_ago():
    """Test recency score for memory from 1 hour ago."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    one_hour_ago = current_time - timedelta(hours=1)
    score = computer.compute_recency_score(one_hour_ago)
    
    # Expected: e^(-0.001 × 3600) = e^(-3.6) ≈ 0.0273
    expected = math.exp(-0.001 * 3600)
    assert abs(score - expected) < 1e-9


def test_recency_score_future_timestamp():
    """Test that future timestamps produce score=1.0."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    future_time = current_time + timedelta(hours=1)
    score = computer.compute_recency_score(future_time)
    
    assert score == 1.0


def test_recency_score_very_old_timestamp():
    """Test numerical stability for very old timestamps."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # 10 years ago
    very_old = current_time - timedelta(days=3650)
    score = computer.compute_recency_score(very_old)
    
    # Should be 0.0 due to numerical stability handling
    assert score == 0.0


def test_final_score_computation_balanced():
    """Test final score with balanced weights."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    similarity = 0.8
    recency = 0.6
    importance = 0.4
    
    final_score = computer.compute_final_score(similarity, recency, importance)
    
    # Expected: 0.5 × 0.8 + 0.3 × 0.6 + 0.2 × 0.4 = 0.4 + 0.18 + 0.08 = 0.66
    expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4
    assert abs(final_score - expected) < 1e-9


def test_final_score_computation_no_importance():
    """Test final score with gamma=0 (no importance)."""
    config = RankingConfig(
        alpha=0.7,
        beta=0.3,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    similarity = 0.8
    recency = 0.6
    importance = 0.5  # Should be ignored
    
    final_score = computer.compute_final_score(similarity, recency, importance)
    
    # Expected: 0.7 × 0.8 + 0.3 × 0.6 + 0.0 × 0.5 = 0.56 + 0.18 = 0.74
    expected = 0.7 * 0.8 + 0.3 * 0.6
    assert abs(final_score - expected) < 1e-9


def test_final_score_clamping():
    """Test that final score computation clamps inputs to [0, 1]."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # Test with values outside [0, 1]
    final_score = computer.compute_final_score(1.5, -0.5, 2.0)
    
    # Should clamp to [0, 1]: 0.5 × 1.0 + 0.3 × 0.0 + 0.2 × 1.0 = 0.7
    expected = 0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 1.0
    assert abs(final_score - expected) < 1e-9


def test_compute_scores_integration():
    """Test compute_scores method that computes both recency and final scores."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # Create a memory
    memory = RankedMemory(
        memory_id="test-1",
        timestamp=current_time - timedelta(hours=1),
        content="Test memory",
        namespace="test",
        similarity_score=0.8,
        importance_score=0.6,
        recency_score=0.0,  # Will be computed
        final_score=0.0,    # Will be computed
        memory_entry=None,
    )
    
    # Compute scores
    result = computer.compute_scores(memory)
    
    # Verify recency score was computed
    expected_recency = math.exp(-0.001 * 3600)
    assert abs(result.recency_score - expected_recency) < 1e-9
    
    # Verify final score was computed
    expected_final = 0.5 * 0.8 + 0.3 * expected_recency + 0.2 * 0.6
    assert abs(result.final_score - expected_final) < 1e-9


def test_different_decay_constants():
    """Test recency scores with different decay constants."""
    current_time = datetime.now(timezone.utc)
    one_hour_ago = current_time - timedelta(hours=1)
    
    # Fast decay
    fast_config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.01,  # Fast decay
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    fast_computer = ScoreComputer(fast_config, current_time)
    fast_score = fast_computer.compute_recency_score(one_hour_ago)
    
    # Slow decay
    slow_config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.0001,  # Slow decay
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    slow_computer = ScoreComputer(slow_config, current_time)
    slow_score = slow_computer.compute_recency_score(one_hour_ago)
    
    # Slow decay should produce higher score for same age
    assert slow_score > fast_score


def test_edge_case_all_zeros():
    """Test final score computation with all zero inputs."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    final_score = computer.compute_final_score(0.0, 0.0, 0.0)
    assert final_score == 0.0


def test_edge_case_all_ones():
    """Test final score computation with all maximum inputs."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    final_score = computer.compute_final_score(1.0, 1.0, 1.0)
    # Expected: 0.5 × 1.0 + 0.3 × 1.0 + 0.2 × 1.0 = 1.0
    assert abs(final_score - 1.0) < 1e-9
