"""
Property-based tests for RankingConfig validation.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**

Tests that invalid configurations are rejected with descriptive errors.
"""

import pytest
from hypothesis import given, strategies as st
from luma.core.ranking_engine import RankingConfig


# Property 1: Configuration validation completeness
@given(
    alpha=st.floats(min_value=-10.0, max_value=10.0),
    beta=st.floats(min_value=-10.0, max_value=10.0),
    gamma=st.floats(min_value=-10.0, max_value=10.0),
    decay_constant=st.floats(min_value=-10.0, max_value=10.0),
    similarity_threshold=st.floats(min_value=-1.0, max_value=2.0),
    score_threshold=st.floats(min_value=-1.0, max_value=2.0),
)
def test_invalid_configs_rejected(
    alpha, beta, gamma, decay_constant, similarity_threshold, score_threshold
):
    """
    Property: Invalid configurations are rejected with descriptive errors.
    
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
    """
    config = RankingConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        decay_constant=decay_constant,
        similarity_threshold=similarity_threshold,
        score_threshold=score_threshold,
    )
    
    # Check if configuration should be valid
    is_valid = (
        alpha >= 0 and
        beta >= 0 and
        gamma >= 0 and
        abs((alpha + beta + gamma) - 1.0) < 1e-9 and
        decay_constant > 0 and
        0 <= similarity_threshold <= 1 and
        0 <= score_threshold <= 1
    )
    
    if is_valid:
        # Should not raise
        config.validate()
    else:
        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message is descriptive
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0, "Error message should be descriptive"


@given(
    alpha=st.floats(min_value=0.0, max_value=1.0),
    beta=st.floats(min_value=0.0, max_value=1.0),
)
def test_weight_sum_validation(alpha, beta):
    """
    Property: Weight sum must equal 1.0.
    
    **Validates: Requirements 8.1**
    """
    gamma = 1.0 - alpha - beta
    
    config = RankingConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    
    # Should not raise if gamma is non-negative
    if gamma >= 0:
        config.validate()
    else:
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "non-negative" in str(exc_info.value).lower()


def test_negative_weights_rejected():
    """
    Test that negative weights are rejected.
    
    **Validates: Requirements 8.2**
    """
    # Negative alpha
    config = RankingConfig(
        alpha=-0.1,
        beta=0.6,
        gamma=0.5,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "alpha" in str(exc_info.value).lower()
    assert "non-negative" in str(exc_info.value).lower()
    
    # Negative beta
    config = RankingConfig(
        alpha=0.5,
        beta=-0.1,
        gamma=0.6,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "beta" in str(exc_info.value).lower()
    assert "non-negative" in str(exc_info.value).lower()
    
    # Negative gamma
    config = RankingConfig(
        alpha=0.6,
        beta=0.5,
        gamma=-0.1,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "gamma" in str(exc_info.value).lower()
    assert "non-negative" in str(exc_info.value).lower()


def test_invalid_decay_constant_rejected():
    """
    Test that decay constant <= 0 is rejected.
    
    **Validates: Requirements 8.3**
    """
    # Zero decay constant
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.0,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "decay_constant" in str(exc_info.value).lower()
    assert "greater than 0" in str(exc_info.value).lower()
    
    # Negative decay constant
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=-0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "decay_constant" in str(exc_info.value).lower()


def test_invalid_threshold_ranges_rejected():
    """
    Test that thresholds outside [0, 1] are rejected.
    
    **Validates: Requirements 8.4, 8.5**
    """
    # similarity_threshold < 0
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=-0.1,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "similarity_threshold" in str(exc_info.value).lower()
    
    # similarity_threshold > 1
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=1.1,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "similarity_threshold" in str(exc_info.value).lower()
    
    # score_threshold < 0
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=-0.1,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "score_threshold" in str(exc_info.value).lower()
    
    # score_threshold > 1
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=1.1,
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    assert "score_threshold" in str(exc_info.value).lower()


def test_valid_config_accepted():
    """
    Test that valid configurations are accepted.
    
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
    """
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    # Should not raise
    config.validate()
