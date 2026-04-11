"""
Property-based tests for recency score computation.

**Property 2: Recency score monotonicity**
**Validates: Requirements 2.1, 2.2, 2.5**
"""

import pytest
from datetime import datetime, timedelta, timezone
from hypothesis import given, strategies as st, assume
from luma.core.ranking_engine import RankingConfig, ScoreComputer


@given(
    decay_constant=st.floats(min_value=0.0001, max_value=0.1),
    age_seconds=st.integers(min_value=0, max_value=100000),
)
def test_recency_score_monotonicity(decay_constant, age_seconds):
    """
    Property: Newer memories always have higher or equal recency scores.
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=decay_constant,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # Create two timestamps: one older, one newer
    older_timestamp = current_time - timedelta(seconds=age_seconds + 1)
    newer_timestamp = current_time - timedelta(seconds=age_seconds)
    
    older_score = computer.compute_recency_score(older_timestamp)
    newer_score = computer.compute_recency_score(newer_timestamp)
    
    # Newer memories should have higher or equal recency scores
    assert newer_score >= older_score, \
        f"Newer memory should have higher recency score: {newer_score} >= {older_score}"


@given(
    decay_constant=st.floats(min_value=0.0001, max_value=0.1),
    age_seconds=st.integers(min_value=0, max_value=1000000),
)
def test_recency_score_in_range(decay_constant, age_seconds):
    """
    Property: Recency score is always in [0, 1].
    
    **Validates: Requirements 2.1**
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=decay_constant,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    timestamp = current_time - timedelta(seconds=age_seconds)
    score = computer.compute_recency_score(timestamp)
    
    assert 0.0 <= score <= 1.0, f"Recency score must be in [0, 1], got {score}"


def test_recency_score_zero_age():
    """
    Test that age=0 produces score=1.0.
    
    **Validates: Requirements 2.5**
    """
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
    
    # Same timestamp as current time
    score = computer.compute_recency_score(current_time)
    
    assert score == 1.0, f"Recency score for age=0 should be 1.0, got {score}"


def test_recency_score_future_timestamp():
    """
    Test that future timestamps produce score=1.0.
    
    **Validates: Requirements 2.2**
    """
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
    
    # Future timestamp
    future_timestamp = current_time + timedelta(hours=1)
    score = computer.compute_recency_score(future_timestamp)
    
    assert score == 1.0, f"Recency score for future timestamp should be 1.0, got {score}"


@given(
    decay_constant=st.floats(min_value=0.0001, max_value=0.01),
)
def test_recency_score_very_old_timestamp(decay_constant):
    """
    Test that very old timestamps don't cause numerical overflow.
    
    **Validates: Requirements 2.1**
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=decay_constant,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # Very old timestamp (10 years ago)
    old_timestamp = current_time - timedelta(days=3650)
    score = computer.compute_recency_score(old_timestamp)
    
    # Should be a valid number (not NaN or inf)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # For very old memories, score should be very close to 0
    assert score < 0.01


@given(
    alpha=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(min_value=0.0, max_value=1.0),
)
def test_exponential_decay_formula(alpha, beta):
    """
    Test that recency score follows exponential decay formula: e^(-λ × age).
    
    **Validates: Requirements 2.1**
    """
    gamma = 1.0 - alpha - beta
    assume(gamma >= 0)
    
    config = RankingConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    current_time = datetime.now(timezone.utc)
    computer = ScoreComputer(config, current_time)
    
    # Test at specific age
    age_seconds = 1000
    timestamp = current_time - timedelta(seconds=age_seconds)
    score = computer.compute_recency_score(timestamp)
    
    # Expected score using exponential decay formula
    import math
    expected_score = math.exp(-config.decay_constant * age_seconds)
    
    # Should match within floating point tolerance
    assert abs(score - expected_score) < 1e-9, \
        f"Score {score} should match expected {expected_score}"
