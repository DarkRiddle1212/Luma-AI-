"""
Unit tests for InjectionConfig validation.

**Validates: Requirements 1.2**
"""

import pytest
from luma.core.injection_engine import InjectionConfig


def test_valid_config_passes():
    """Test that valid configurations are accepted."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    # Should not raise
    config.validate()


def test_valid_config_with_category_isolation_passes():
    """Test that valid configuration with category isolation is accepted."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming", "education"]
    )
    # Should not raise
    config.validate()


def test_negative_max_token_budget_raises_error():
    """Test that negative max_token_budget raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=-100,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "max_token_budget" in error_msg.lower()
    assert "positive" in error_msg.lower()
    assert "-100" in error_msg


def test_zero_max_token_budget_raises_error():
    """Test that max_token_budget = 0 raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=0,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "max_token_budget" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_negative_max_memory_count_raises_error():
    """Test that negative max_memory_count raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=-10,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "max_memory_count" in error_msg.lower()
    assert "positive" in error_msg.lower()
    assert "-10" in error_msg


def test_zero_max_memory_count_raises_error():
    """Test that max_memory_count = 0 raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=0,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "max_memory_count" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_redundancy_threshold_below_zero_raises_error():
    """Test that redundancy_similarity_threshold < 0 raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=-0.1,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "redundancy_similarity_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg
    assert "-0.1" in error_msg


def test_redundancy_threshold_above_one_raises_error():
    """Test that redundancy_similarity_threshold > 1 raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=1.5,
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "redundancy_similarity_threshold" in error_msg.lower()
    assert "[0, 1]" in error_msg
    assert "1.5" in error_msg


def test_boundary_threshold_zero_accepted():
    """Test that redundancy_similarity_threshold = 0.0 is accepted."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.0,
        enable_category_isolation=False
    )
    # Should not raise
    config.validate()


def test_boundary_threshold_one_accepted():
    """Test that redundancy_similarity_threshold = 1.0 is accepted."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=1.0,
        enable_category_isolation=False
    )
    # Should not raise
    config.validate()


def test_negative_token_estimation_factor_raises_error():
    """Test that negative token_estimation_factor raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False,
        token_estimation_factor=-1.3
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "token_estimation_factor" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_zero_token_estimation_factor_raises_error():
    """Test that token_estimation_factor = 0 raises ValueError with descriptive message."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False,
        token_estimation_factor=0.0
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "token_estimation_factor" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_category_isolation_without_allowed_categories_raises_error():
    """Test that enable_category_isolation=True without allowed_categories raises ValueError."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=None
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "allowed_categories" in error_msg.lower()
    assert "enable_category_isolation" in error_msg.lower()


def test_category_isolation_with_empty_allowed_categories_raises_error():
    """Test that enable_category_isolation=True with empty allowed_categories raises ValueError."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=[]
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    error_msg = str(exc_info.value)
    assert "allowed_categories" in error_msg.lower()
    assert "non-empty" in error_msg.lower()


def test_category_isolation_disabled_with_allowed_categories_passes():
    """Test that enable_category_isolation=False with allowed_categories is accepted."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False,
        allowed_categories=["programming"]
    )
    # Should not raise - allowed_categories can be specified even when isolation is disabled
    config.validate()


def test_multiple_validation_errors_first_error_raised():
    """Test that when multiple validation errors exist, the first one is raised."""
    config = InjectionConfig(
        max_token_budget=-100,  # Invalid
        max_memory_count=-10,   # Invalid
        redundancy_similarity_threshold=2.0,  # Invalid
        enable_category_isolation=False
    )
    with pytest.raises(ValueError) as exc_info:
        config.validate()
    
    # Should raise error for max_token_budget (first validation check)
    error_msg = str(exc_info.value)
    assert "max_token_budget" in error_msg.lower()
