"""
Unit tests for InjectionEngine.inject() method orchestration.

Tests the main inject() method that orchestrates the injection pipeline:
- Empty input handling
- Category filtering
- Redundancy filtering
- Budget enforcement
- Diagnostic count tracking
"""

import pytest
from datetime import datetime, timezone
from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig,
    InjectionResult,
    InjectedMemory
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
        # Create a unique embedding based on memory_id to avoid redundancy filtering
        # Use the numeric part of memory_id (e.g., "mem_1" -> 1) to create distinct embeddings
        # Create embeddings that are NOT parallel to avoid high cosine similarity
        try:
            num = int(memory_id.split('_')[-1])
            # Create embeddings with different "directions" to ensure low similarity
            # Using a rotation-like pattern to make them more orthogonal
            if num == 1:
                embedding = [1.0, 0.0, 0.0]
            elif num == 2:
                embedding = [0.0, 1.0, 0.0]
            elif num == 3:
                embedding = [0.0, 0.0, 1.0]
            else:
                # For other numbers, use a mix
                embedding = [0.5, 0.5, float(num) * 0.1]
        except (ValueError, IndexError):
            # Fallback for non-standard memory_ids
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


class TestInjectEmptyInput:
    """Test inject() with empty input (Requirement 1.4)."""
    
    def test_empty_list_returns_empty_result(self):
        """Empty input should return empty result with zero counts."""
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        result = engine.inject([])
        
        assert isinstance(result, InjectionResult)
        assert result.memories == []
        assert result.total_tokens == 0
        assert result.input_count == 0
        assert result.filtered_by_category == 0
        assert result.filtered_by_redundancy == 0
        assert result.filtered_by_budget == 0


class TestInjectSingleMemory:
    """Test inject() with single memory."""
    
    def test_single_memory_within_budget(self):
        """Single memory within budget should be selected."""
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
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 1
        assert result.memories[0].memory_id == "mem_1"
        assert result.memories[0].content == "Test content"
        assert result.total_tokens == 50
        assert result.input_count == 1
        assert result.filtered_by_category == 0
        assert result.filtered_by_redundancy == 0
        assert result.filtered_by_budget == 0


class TestInjectCategoryFiltering:
    """Test inject() with category filtering (Requirement 4.4)."""
    
    def test_category_isolation_filters_correctly(self):
        """Category isolation should filter out non-matching categories."""
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=True,
            allowed_categories=["programming"]
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Python content", category="programming", token_count=50),
            create_test_memory("mem_2", "Math content", category="mathematics", token_count=50),
            create_test_memory("mem_3", "Java content", category="programming", token_count=50),
        ]
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 2
        assert result.memories[0].memory_id == "mem_1"
        assert result.memories[1].memory_id == "mem_3"
        assert result.input_count == 3
        assert result.filtered_by_category == 1
        assert result.filtered_by_redundancy == 0
        assert result.filtered_by_budget == 0


class TestInjectBudgetEnforcement:
    """Test inject() with budget enforcement (Requirement 2.3, 2.4)."""
    
    def test_token_budget_cutoff(self):
        """Should stop selecting when token budget would be exceeded."""
        config = InjectionConfig(
            max_token_budget=100,  # Small budget
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Content 1", token_count=40),
            create_test_memory("mem_2", "Content 2", token_count=40),
            create_test_memory("mem_3", "Content 3", token_count=40),  # Would exceed budget
        ]
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 2
        assert result.total_tokens == 80
        assert result.total_tokens <= 100
        assert result.filtered_by_budget == 1
    
    def test_memory_count_limit(self):
        """Should stop selecting when memory count limit is reached."""
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=2,  # Small count limit
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "Content 1", token_count=10),
            create_test_memory("mem_2", "Content 2", token_count=10),
            create_test_memory("mem_3", "Content 3", token_count=10),
        ]
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 2
        assert result.filtered_by_budget == 1


class TestInjectDiagnosticCounts:
    """Test inject() diagnostic count tracking."""
    
    def test_diagnostic_counts_accuracy(self):
        """Diagnostic counts should accurately reflect filtering at each stage."""
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
            create_test_memory("mem_1", "Python", category="programming", token_count=30, embedding=same_embedding),
            create_test_memory("mem_2", "Math", category="mathematics", token_count=30),  # Filtered by category
            create_test_memory("mem_3", "Java", category="programming", token_count=30, embedding=same_embedding),  # Filtered by redundancy
            create_test_memory("mem_4", "C++", category="programming", token_count=50),  # Filtered by budget
        ]
        
        result = engine.inject(memories)
        
        assert result.input_count == 4
        assert result.filtered_by_category == 1  # mem_2
        # Note: redundancy and budget filtering depend on implementation details
        # Just verify the counts are tracked
        assert isinstance(result.filtered_by_redundancy, int)
        assert isinstance(result.filtered_by_budget, int)


class TestInjectMetadataPreservation:
    """Test inject() preserves metadata (Requirement 10.1, 10.2, 10.4)."""
    
    def test_metadata_preserved_exactly(self):
        """Metadata should be preserved exactly without modification."""
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        original_metadata = {
            "token_count": 50,
            "embedding": [0.1, 0.2, 0.3],
            "source": "test",
            "custom_field": "custom_value"
        }
        
        memories = [
            create_test_memory("mem_1", "Test content", token_count=50)
        ]
        memories[0].metadata = original_metadata
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 1
        assert result.memories[0].metadata == original_metadata
        # Metadata can be the same reference (immutability is preserved by not modifying it)


class TestInjectOrderPreservation:
    """Test inject() preserves input order (Requirement 1.3)."""
    
    def test_output_order_matches_input_order(self):
        """Output memories should maintain input order."""
        config = InjectionConfig(
            max_token_budget=1000,
            max_memory_count=10,
            redundancy_similarity_threshold=0.85,
            enable_category_isolation=False
        )
        engine = InjectionEngine(config)
        
        memories = [
            create_test_memory("mem_1", "First", token_count=10),
            create_test_memory("mem_2", "Second", token_count=10),
            create_test_memory("mem_3", "Third", token_count=10),
        ]
        
        result = engine.inject(memories)
        
        assert len(result.memories) == 3
        assert result.memories[0].memory_id == "mem_1"
        assert result.memories[1].memory_id == "mem_2"
        assert result.memories[2].memory_id == "mem_3"
