"""
Unit tests for ConfigValidator.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
"""

import pytest
from luma.core.ranking_engine import RankingConfig, ConfigValidator


def test_valid_config_passes():
    """Test that valid configurations pass validation."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    # Should not raise
    ConfigValidator.validate(config)


def test_invalid_weight_sum_raises_error():
    """Test that invalid weight sum raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.3,  # Sum = 1.1, not 1.0
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "weight sum" in error_msg.lower() or "alpha + beta + gamma" in error_msg.lower()
    assert "1.0" in error_msg


def test_negative_alpha_raises_error():
    """Test that negative alpha raises descriptive error."""
    config = RankingConfig(
        alpha=-0.1,
        beta=0.6,
        gamma=0.5,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "alpha" in error_msg.lower()
    assert "non-negative" in error_msg.lower()


def test_negative_beta_raises_error():
    """Test that negative beta raises descriptive error."""
    config = RankingConfig(
        alpha=0.6,
        beta=-0.1,
        gamma=0.5,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "beta" in error_msg.lower()
    assert "non-negative" in error_msg.lower()


def test_negative_gamma_raises_error():
    """Test that negative gamma raises descriptive error."""
    config = RankingConfig(
        alpha=0.6,
        beta=0.5,
        gamma=-0.1,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "gamma" in error_msg.lower()
    assert "non-negative" in error_msg.lower()


def test_zero_decay_constant_raises_error():
    """Test that decay constant = 0 raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.0,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "decay_constant" in error_msg.lower()
    assert "greater than 0" in error_msg.lower()


def test_negative_decay_constant_raises_error():
    """Test that negative decay constant raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=-0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "decay_constant" in error_msg.lower()


def test_similarity_threshold_below_zero_raises_error():
    """Test that similarity_threshold < 0 raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=-0.1,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "similarity_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg


def test_similarity_threshold_above_one_raises_error():
    """Test that similarity_threshold > 1 raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=1.1,
        score_threshold=0.3,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "similarity_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg


def test_score_threshold_below_zero_raises_error():
    """Test that score_threshold < 0 raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=-0.1,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "score_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg


def test_score_threshold_above_one_raises_error():
    """Test that score_threshold > 1 raises descriptive error."""
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=1.1,
    )
    with pytest.raises(ValueError) as exc_info:
        ConfigValidator.validate(config)
    
    error_msg = str(exc_info.value)
    assert "score_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg


def test_boundary_values_accepted():
    """Test that boundary values (0 and 1) are accepted for thresholds."""
    # Test with thresholds at 0
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0,
    )
    ConfigValidator.validate(config)
    
    # Test with thresholds at 1
    config = RankingConfig(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        decay_constant=0.001,
        similarity_threshold=1.0,
        score_threshold=1.0,
    )
    ConfigValidator.validate(config)


def test_floating_point_tolerance_for_weight_sum():
    """Test that floating point tolerance is applied to weight sum validation."""
    # Weights that sum to 1.0 within floating point tolerance
    config = RankingConfig(
        alpha=0.333333333333,
        beta=0.333333333333,
        gamma=0.333333333334,  # Sum is very close to 1.0
        decay_constant=0.001,
        similarity_threshold=0.5,
        score_threshold=0.3,
    )
    # Should not raise
    ConfigValidator.validate(config)
