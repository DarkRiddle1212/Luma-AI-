"""
Unit tests for context injection configuration validation.

Tests configuration validation logic for InjectionConfig, ensuring that
max_memories parameter is properly validated within the range [5, 20].

Requirements tested:
- 3.2: Configuration validation for max_memories range
- 7.1: Unit test coverage for configuration validation
"""

import pytest
from luma.core.context_injection import InjectionConfig


class TestConfigurationValidation:
    """Test suite for InjectionConfig validation."""
    
    def test_valid_config_min_boundary(self):
        """Test valid configuration at minimum boundary (5 memories)."""
        config = InjectionConfig(max_memories=5)
        config.validate()  # Should not raise
    
    def test_valid_config_mid_range(self):
        """Test valid configuration in middle of range (10 memories)."""
        config = InjectionConfig(max_memories=10)
        config.validate()  # Should not raise
    
    def test_valid_config_upper_mid_range(self):
        """Test valid configuration in upper middle range (15 memories)."""
        config = InjectionConfig(max_memories=15)
        config.validate()  # Should not raise
    
    def test_valid_config_max_boundary(self):
        """Test valid configuration at maximum boundary (20 memories)."""
        config = InjectionConfig(max_memories=20)
        config.validate()  # Should not raise
    
    def test_invalid_config_zero(self):
        """Test invalid configuration with zero memories."""
        config = InjectionConfig(max_memories=0)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "max_memories must be in [5, 20]" in str(exc_info.value)
        assert "got 0" in str(exc_info.value)
    
    def test_invalid_config_below_min(self):
        """Test invalid configuration below minimum (4 memories)."""
        config = InjectionConfig(max_memories=4)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "max_memories must be in [5, 20]" in str(exc_info.value)
        assert "got 4" in str(exc_info.value)
    
    def test_invalid_config_above_max(self):
        """Test invalid configuration above maximum (21 memories)."""
        config = InjectionConfig(max_memories=21)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "max_memories must be in [5, 20]" in str(exc_info.value)
        assert "got 21" in str(exc_info.value)
    
    def test_invalid_config_far_above_max(self):
        """Test invalid configuration far above maximum (100 memories)."""
        config = InjectionConfig(max_memories=100)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "max_memories must be in [5, 20]" in str(exc_info.value)
        assert "got 100" in str(exc_info.value)
    
    def test_invalid_config_negative(self):
        """Test invalid configuration with negative value (-1 memories)."""
        config = InjectionConfig(max_memories=-1)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "max_memories must be in [5, 20]" in str(exc_info.value)
        assert "got -1" in str(exc_info.value)
