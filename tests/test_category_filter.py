"""
Unit tests for CategoryFilter.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

import pytest
from datetime import datetime, timezone
from luma.core.injection_engine import CategoryFilter, InjectionConfig


class MockMemory:
    """Mock memory object for testing CategoryFilter."""
    
    def __init__(self, memory_id: str, category: str = None):
        self.memory_id = memory_id
        self.category = category


def test_filter_with_isolation_enabled_and_matching_categories():
    """Test that CategoryFilter only passes memories with matching categories when isolation is enabled."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming", "documentation"]
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="programming"),
        MockMemory("mem_2", category="cooking"),
        MockMemory("mem_3", category="documentation"),
        MockMemory("mem_4", category="sports"),
        MockMemory("mem_5", category="programming")
    ]
    
    filtered = filter.filter(memories)
    
    # Should only include programming and documentation categories
    assert len(filtered) == 3
    assert filtered[0].memory_id == "mem_1"
    assert filtered[0].category == "programming"
    assert filtered[1].memory_id == "mem_3"
    assert filtered[1].category == "documentation"
    assert filtered[2].memory_id == "mem_5"
    assert filtered[2].category == "programming"


def test_filter_with_isolation_disabled_passes_all_categories():
    """Test that CategoryFilter passes all memories when isolation is disabled."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="programming"),
        MockMemory("mem_2", category="cooking"),
        MockMemory("mem_3", category="documentation"),
        MockMemory("mem_4", category="sports")
    ]
    
    filtered = filter.filter(memories)
    
    # Should include all memories
    assert len(filtered) == 4
    assert filtered[0].memory_id == "mem_1"
    assert filtered[1].memory_id == "mem_2"
    assert filtered[2].memory_id == "mem_3"
    assert filtered[3].memory_id == "mem_4"


def test_filter_with_isolation_enabled_and_no_matching_categories():
    """Test that CategoryFilter returns empty list when no memories match the allowed categories."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming", "documentation"]
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="cooking"),
        MockMemory("mem_2", category="sports"),
        MockMemory("mem_3", category="music")
    ]
    
    filtered = filter.filter(memories)
    
    # Should return empty list
    assert len(filtered) == 0


def test_filter_preserves_order():
    """Test that CategoryFilter preserves the order of memories in the input list."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming", "documentation", "education"]
    )
    
    filter = CategoryFilter(config)
    
    # Create memories in specific order
    memories = [
        MockMemory("mem_5", category="programming"),
        MockMemory("mem_1", category="cooking"),
        MockMemory("mem_3", category="documentation"),
        MockMemory("mem_2", category="sports"),
        MockMemory("mem_7", category="education"),
        MockMemory("mem_4", category="programming")
    ]
    
    filtered = filter.filter(memories)
    
    # Should preserve order: mem_5, mem_3, mem_7, mem_4
    assert len(filtered) == 4
    assert filtered[0].memory_id == "mem_5"
    assert filtered[1].memory_id == "mem_3"
    assert filtered[2].memory_id == "mem_7"
    assert filtered[3].memory_id == "mem_4"


def test_filter_with_empty_input():
    """Test that CategoryFilter handles empty input list correctly."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming"]
    )
    
    filter = CategoryFilter(config)
    
    filtered = filter.filter([])
    
    # Should return empty list
    assert len(filtered) == 0


def test_filter_with_none_categories():
    """Test that CategoryFilter handles memories with None category correctly."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming"]
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="programming"),
        MockMemory("mem_2", category=None),
        MockMemory("mem_3", category="programming")
    ]
    
    filtered = filter.filter(memories)
    
    # Should only include memories with matching categories (None doesn't match)
    assert len(filtered) == 2
    assert filtered[0].memory_id == "mem_1"
    assert filtered[1].memory_id == "mem_3"


def test_filter_with_single_allowed_category():
    """Test that CategoryFilter works correctly with a single allowed category."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=True,
        allowed_categories=["programming"]
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="programming"),
        MockMemory("mem_2", category="documentation"),
        MockMemory("mem_3", category="programming")
    ]
    
    filtered = filter.filter(memories)
    
    # Should only include programming category
    assert len(filtered) == 2
    assert filtered[0].memory_id == "mem_1"
    assert filtered[1].memory_id == "mem_3"


def test_filter_disabled_with_none_categories():
    """Test that CategoryFilter passes memories with None category when isolation is disabled."""
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    
    filter = CategoryFilter(config)
    
    memories = [
        MockMemory("mem_1", category="programming"),
        MockMemory("mem_2", category=None),
        MockMemory("mem_3", category="documentation")
    ]
    
    filtered = filter.filter(memories)
    
    # Should include all memories including None category
    assert len(filtered) == 3
    assert filtered[0].memory_id == "mem_1"
    assert filtered[1].memory_id == "mem_2"
    assert filtered[2].memory_id == "mem_3"
