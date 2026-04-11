"""
Unit tests for InjectionEngine initialization.

This module tests the initialization behavior of the InjectionEngine class,
including configuration validation, component initialization, and error handling.

Requirements tested:
- 1.2: Accept InjectionConfig object with all configuration parameters
- 9.1: Do not compute or modify final_score values
- 9.2: Do not implement ranking algorithms or scoring logic
- 9.3: Accept pre-computed similarity scores from ranking phase
"""

import pytest
from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig,
    CategoryFilter,
    TokenEstimator,
    RedundancyGuard,
    TokenBudgetEnforcer
)
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


class TestInjectionEngineInitialization:
    """Test suite for InjectionEngine initialization."""
    
    def test_initialization_with_valid_config(self):
        """Test that InjectionEngine initializes successfully with valid config."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Act
        engine = InjectionEngine(config)
        
        # Assert
        assert engine.config == config
        assert engine.metrics_collector is None
        assert engine.logger is None
        assert isinstance(engine.category_filter, CategoryFilter)
        assert isinstance(engine.token_estimator, TokenEstimator)
        assert isinstance(engine.redundancy_guard, RedundancyGuard)
        assert isinstance(engine.budget_enforcer, TokenBudgetEnforcer)
    
    def test_initialization_with_observability_components(self):
        """Test that InjectionEngine accepts optional metrics and logger."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        metrics = MetricsCollector()
        logger = StructuredLogger(name="test_injection_engine")
        
        # Act
        engine = InjectionEngine(config, metrics, logger)
        
        # Assert
        assert engine.config == config
        assert engine.metrics_collector is metrics
        assert engine.logger is logger
    
    def test_initialization_validates_config(self):
        """Test that InjectionEngine validates config on initialization."""
        # Arrange - invalid config with negative token budget
        config = InjectionConfig(
            max_token_budget=-100,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="max_token_budget must be positive"):
            InjectionEngine(config)
    
    def test_initialization_validates_memory_count(self):
        """Test that InjectionEngine validates max_memory_count."""
        # Arrange - invalid config with zero memory count
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=0,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="max_memory_count must be positive"):
            InjectionEngine(config)
    
    def test_initialization_validates_threshold(self):
        """Test that InjectionEngine validates redundancy threshold."""
        # Arrange - invalid config with threshold > 1
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=1.5,
            enable_category_isolation=False
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="redundancy_similarity_threshold must be in"):
            InjectionEngine(config)
    
    def test_initialization_validates_category_isolation(self):
        """Test that InjectionEngine validates category isolation config."""
        # Arrange - invalid config with isolation enabled but no categories
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=None
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="allowed_categories must be specified"):
            InjectionEngine(config)
    
    def test_initialization_validates_empty_categories(self):
        """Test that InjectionEngine validates non-empty categories list."""
        # Arrange - invalid config with empty categories list
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=[]
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="allowed_categories must be non-empty"):
            InjectionEngine(config)
    
    def test_component_initialization_with_correct_parameters(self):
        """Test that all components are initialized with correct parameters."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming", "documentation"],
            token_estimation_factor=1.5
        )
        
        # Act
        engine = InjectionEngine(config)
        
        # Assert - verify component parameters
        assert engine.category_filter.enabled is True
        assert engine.category_filter.allowed_categories == ["programming", "documentation"]
        assert engine.token_estimator.estimation_factor == 1.5
        assert engine.redundancy_guard.threshold == 0.85
        assert engine.budget_enforcer.max_token_budget == 2048
        assert engine.budget_enforcer.max_memory_count == 50
        assert engine.budget_enforcer.token_estimator is engine.token_estimator
    
    def test_initialization_with_category_isolation_disabled(self):
        """Test initialization with category isolation disabled."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=2048,
            max_memory_count=50,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        # Act
        engine = InjectionEngine(config)
        
        # Assert
        assert engine.category_filter.enabled is False
        assert engine.category_filter.allowed_categories == []
    
    def test_initialization_with_boundary_values(self):
        """Test initialization with boundary values for parameters."""
        # Arrange - minimum valid values
        config = InjectionConfig(
            max_token_budget=1,
            max_memory_count=1,
            redundancy_similarity_threshold=0.0,
            enable_category_isolation=False
        )
        
        # Act
        engine = InjectionEngine(config)
        
        # Assert
        assert engine.config.max_token_budget == 1
        assert engine.config.max_memory_count == 1
        assert engine.config.redundancy_similarity_threshold == 0.0
        
        # Arrange - maximum valid threshold
        config2 = InjectionConfig(
            max_token_budget=1,
            max_memory_count=1,
            redundancy_similarity_threshold=1.0,
            enable_category_isolation=False
        )
        
        # Act
        engine2 = InjectionEngine(config2)
        
        # Assert
        assert engine2.config.redundancy_similarity_threshold == 1.0
