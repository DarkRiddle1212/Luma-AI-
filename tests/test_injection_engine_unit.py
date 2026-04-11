"""
Unit tests for InjectionEngine orchestration.

This module provides focused unit tests for the InjectionEngine class,
testing its orchestration logic and component interactions.

Tests cover:
- Empty input handling (Requirement 1.4)
- Single memory injection
- Component interaction (category filter, redundancy guard, budget enforcer)
- Diagnostic counts accuracy
- Metadata preservation
- Order preservation

**Validates: Requirements 1.4**
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig,
    InjectionResult,
    InjectedMemory,
    CategoryFilter,
    RedundancyGuard,
    TokenBudgetEnforcer,
    TokenEstimator
)
from luma.core.ranking_engine import RankedMemory


def create_test_memory(
    memory_id: str,
    content: str,
    category: str = "test",
    similarity_score: float = 0.8,
    final_score: float = 0.8,
    token_count: int = 10,
    embedding: list = None
) -> RankedMemory:
    """Helper to create test RankedMemory objects."""
    if embedding is None:
        # Create distinct embeddings to avoid unintended redundancy filtering
        try:
            num = int(memory_id.split('_')[-1])
            if num == 1:
                embedding = [1.0, 0.0, 0.0]
            elif num == 2:
                embedding = [0.0, 1.0, 0.0]
            elif num == 3:
                embedding = [0.0, 0.0, 1.0]
            else:
                embedding = [0.5, 0.5, float(num) * 0.1]
        except (ValueError, IndexError):
            embedding = [0.1, 0.2, 0.3]
    
    return RankedMemory(
        memory_id=memory_id,
        content=content,
        timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        namespace="test",
        category=category,
        similarity_score=similarity_score,
        final_score=final_score,
        recency_score=0.9,
        importance_score=0.7,
        metadata={
            "token_count": token_count,
            "embedding": embedding,
            "source": "test"
        },
        memory_entry=None
    )


class TestInjectionEngineEmptyInput:
    """Test InjectionEngine with empty input (Requirement 1.4)."""
    
    def test_empty_input_returns_empty_result(self):
        """
        WHEN the input memory list is empty,
        THE Injection_Engine SHALL return an empty Injection_Result
        with total_tokens set to zero.
        
        Validates: Requirement 1.4
        """
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        # Act
        result = engine.inject([])
        
        # Assert
        assert isinstance(result, InjectionResult)
        assert result.memories == []
        assert result.total_tokens == 0
        assert result.input_count == 0
        assert result.filtered_by_category == 0
        assert result.filtered_by_redundancy == 0
        assert result.filtered_by_budget == 0
    
    def test_empty_input_with_observability(self):
        """Empty input should work with observability components."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        
        metrics_collector = Mock()
        logger = Mock()
        engine = InjectionEngine(config, metrics_collector, logger)
        
        # Act
        result = engine.inject([])
        
        # Assert
        assert result.memories == []
        assert result.total_tokens == 0
        
        # Verify observability was called
        metrics_collector.record_duration.assert_called_once()
        logger.log.assert_called_once()


class TestInjectionEngineSingleMemory:
    """Test InjectionEngine with single memory."""
    
    def test_single_memory_within_budget(self):
        """Single memory within budget should be selected."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Test content", token_count=50)
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_1"
        assert result.memories[0].content == "Test content"
        assert result.total_tokens == 50
        assert result.input_count == 1
        assert result.filtered_by_category == 0
        assert result.filtered_by_redundancy == 0
        assert result.filtered_by_budget == 0
    
    def test_single_memory_exceeds_budget(self):
        """Single memory exceeding budget should be filtered."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=10,  # Very small budget
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Test content", token_count=50)
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        assert len(result.memories) == 0
        assert result.total_tokens == 0
        assert result.input_count == 1
        assert result.filtered_by_budget == 1


class TestInjectionEngineComponentInteraction:
    """Test interaction between InjectionEngine components."""
    
    def test_category_filter_then_redundancy_guard(self):
        """Category filter should run before redundancy guard."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming"]
        )
        engine = InjectionEngine(config)
        
        # Create memories with same embedding for redundancy testing
        same_embedding = [0.5, 0.5, 0.5]
        
        memories = [
            create_test_memory("mem_1", "Python", category="programming", 
                             token_count=30, embedding=same_embedding),
            create_test_memory("mem_2", "Math", category="mathematics", 
                             token_count=30, embedding=same_embedding),  # Filtered by category
            create_test_memory("mem_3", "Java", category="programming", 
                             token_count=30, embedding=same_embedding),  # Filtered by redundancy
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert - mem_2 filtered by category, mem_3 filtered by redundancy
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_1"
        assert result.filtered_by_category == 1
        assert result.filtered_by_redundancy == 1
        assert result.filtered_by_budget == 0
    
    def test_redundancy_guard_then_budget_enforcer(self):
        """Redundancy guard should run before budget enforcer."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=100,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        # Create memories with same embedding for redundancy testing
        same_embedding = [0.5, 0.5, 0.5]
        
        memories = [
            create_test_memory("mem_1", "First", token_count=40, embedding=same_embedding),
            create_test_memory("mem_2", "Second", token_count=40, embedding=same_embedding),  # Filtered by redundancy
            create_test_memory("mem_3", "Third", token_count=40),  # Would be filtered by budget if mem_2 wasn't redundant
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert - mem_2 filtered by redundancy, mem_3 selected (budget allows it)
        assert len(result.memories) == 2
        assert result.memories[0].memory_id == "mem_1"
        assert result.memories[1].memory_id == "mem_3"
        assert result.filtered_by_redundancy == 1
        assert result.total_tokens == 80
    
    def test_all_components_working_together(self):
        """Test all components (category, redundancy, budget) working together."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=100,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming"]
        )
        engine = InjectionEngine(config)
        
        # Create memories with same embedding for redundancy testing
        same_embedding = [0.5, 0.5, 0.5]
        
        memories = [
            create_test_memory("mem_1", "Python", category="programming", 
                             token_count=30, embedding=same_embedding),
            create_test_memory("mem_2", "Math", category="mathematics", 
                             token_count=30),  # Filtered by category
            create_test_memory("mem_3", "Java", category="programming", 
                             token_count=30, embedding=same_embedding),  # Filtered by redundancy
            create_test_memory("mem_4", "C++", category="programming", 
                             token_count=50, embedding=same_embedding),  # Filtered by redundancy
            create_test_memory("mem_5", "Rust", category="programming", 
                             token_count=30, embedding=same_embedding),  # Filtered by redundancy
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert - only mem_1 selected, others filtered
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_1"
        assert result.input_count == 5
        assert result.filtered_by_category == 1  # mem_2
        assert result.filtered_by_redundancy == 3  # mem_3, mem_4, mem_5
        assert result.filtered_by_budget == 0
        assert result.total_tokens == 30


class TestInjectionEngineDiagnosticCounts:
    """Test diagnostic counts accuracy."""
    
    def test_diagnostic_counts_with_category_filtering(self):
        """Diagnostic counts should accurately track category filtering."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming"]
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Python", category="programming", token_count=30),
            create_test_memory("mem_2", "Math", category="mathematics", token_count=30),
            create_test_memory("mem_3", "History", category="history", token_count=30),
            create_test_memory("mem_4", "Java", category="programming", token_count=30),
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        assert result.input_count == 4
        assert result.filtered_by_category == 2  # mem_2, mem_3
        assert len(result.memories) == 2
    
    def test_diagnostic_counts_with_redundancy_filtering(self):
        """Diagnostic counts should accurately track redundancy filtering."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        # Create memories with same embedding for redundancy testing
        same_embedding = [0.5, 0.5, 0.5]
        
        memories = [
            create_test_memory("mem_1", "First", token_count=30, embedding=same_embedding),
            create_test_memory("mem_2", "Second", token_count=30, embedding=same_embedding),
            create_test_memory("mem_3", "Third", token_count=30, embedding=same_embedding),
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        assert result.input_count == 3
        assert result.filtered_by_redundancy == 2  # mem_2, mem_3
        assert len(result.memories) == 1
    
    def test_diagnostic_counts_with_budget_filtering(self):
        """Diagnostic counts should accurately track budget filtering."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=100,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "First", token_count=40),
            create_test_memory("mem_2", "Second", token_count=40),
            create_test_memory("mem_3", "Third", token_count=40),  # Would exceed budget
            create_test_memory("mem_4", "Fourth", token_count=40),  # Would exceed budget
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        assert result.input_count == 4
        assert result.filtered_by_budget == 2  # mem_3, mem_4
        assert len(result.memories) == 2
        assert result.total_tokens == 80
    
    def test_diagnostic_counts_sum_correctly(self):
        """Sum of selected + filtered should equal input count."""
        # Arrange
        config = InjectionConfig(
            max_token_budget=100,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming"]
        )
        engine = InjectionEngine(config)
        
        # Create memories with same embedding for redundancy testing
        same_embedding = [0.5, 0.5, 0.5]
        
        memories = [
            create_test_memory("mem_1", "Python", category="programming", 
                             token_count=30, embedding=same_embedding),
            create_test_memory("mem_2", "Math", category="mathematics", 
                             token_count=30),  # Filtered by category
            create_test_memory("mem_3", "Java", category="programming", 
                             token_count=30, embedding=same_embedding),  # Filtered by redundancy
            create_test_memory("mem_4", "C++", category="programming", 
                             token_count=50),  # Filtered by budget
            create_test_memory("mem_5", "Rust", category="programming", 
                             token_count=30),  # Selected
        ]
        
        # Act
        result = engine.inject(memories)
        
        # Assert
        selected_count = len(result.memories)
        filtered_count = (result.filtered_by_category + 
                         result.filtered_by_redundancy + 
                         result.filtered_by_budget)
        
        assert selected_count + filtered_count == result.input_count
        assert result.input_count == 5
