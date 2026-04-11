"""
Property-based test for behavior preservation in instrumented components.

This module verifies that instrumentation is strictly additive and does not
change business logic. It tests that all instrumented operations return
identical results whether observability is enabled or disabled.

**Validates: Requirements 13.5**
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, UTC, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock

from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.ranking_engine import RankingEngine, RankingConfig, RankedMemory
from luma.core.lifecycle_manager import MemoryLifecycleManager
from luma.core.lifecycle_config import LifecycleConfig
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def memory_entry_strategy(draw):
    """Generate a valid memory entry."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    content = draw(st.text(min_size=1, max_size=100))
    importance = draw(st.floats(min_value=0.0, max_value=1.0))
    final_score = draw(st.floats(min_value=0.0, max_value=1.0))
    
    # Generate timestamp within last 365 days
    days_ago = draw(st.integers(min_value=0, max_value=365))
    timestamp = datetime.now(UTC) - timedelta(days=days_ago)
    
    return {
        "id": memory_id,
        "content": content,
        "metadata": {
            "importance": importance,
            "final_score": final_score
        },
        "timestamp": timestamp.isoformat(),
        "category": "test",
        "tags": []
    }


@st.composite
def memory_list_strategy(draw):
    """Generate a list of memory entries."""
    return draw(st.lists(memory_entry_strategy(), min_size=0, max_size=5))


@st.composite
def ranking_config_strategy(draw):
    """Generate a valid ranking configuration."""
    # Generate weights that sum to 1.0
    alpha = draw(st.floats(min_value=0.0, max_value=1.0))
    beta = draw(st.floats(min_value=0.0, max_value=1.0 - alpha))
    gamma = 1.0 - alpha - beta
    
    return RankingConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        decay_constant=draw(st.floats(min_value=0.001, max_value=0.1)),
        similarity_threshold=draw(st.floats(min_value=0.0, max_value=1.0)),
        score_threshold=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@st.composite
def ranked_memory_strategy(draw):
    """Generate a valid RankedMemory instance."""
    memory_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    content = draw(st.text(min_size=1, max_size=100))
    
    # Generate timestamp within last 365 days
    days_ago = draw(st.integers(min_value=0, max_value=365))
    timestamp = datetime.now(UTC) - timedelta(days=days_ago)
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content=content,
        namespace="test",
        similarity_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        importance_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        recency_score=0.0,
        final_score=0.0,
        memory_entry=None
    )


# ============================================================================
# Property Test: Behavior Preservation
# ============================================================================

@given(memory_list_strategy())
@settings(max_examples=10, deadline=None)
def test_retrieve_behavior_preservation(memories):
    """
    Property: SQLiteMemoryAdapter.retrieve produces identical results with and without metrics_collector.
    
    **Validates: Requirements 13.5**
    
    This test verifies that instrumentation is strictly additive and does not
    change the business logic of the retrieve operation. The test compares
    results from retrieve() with metrics_collector=None vs with a MetricsCollector
    instance and ensures they are identical.
    """
    # Create mock memory manager
    mock_memory_manager = Mock()
    
    # Convert memory entries to mock MemoryManager format
    mock_entries = []
    for memory in memories:
        mock_entry = Mock()
        mock_entry.id = memory["id"]
        mock_entry.action = memory["content"]
        mock_entry.context = memory["metadata"]
        mock_entry.created_at = datetime.fromisoformat(memory["timestamp"].replace('Z', '+00:00'))
        mock_entry.timestamp = mock_entry.created_at
        mock_entry.tags = memory["tags"]
        mock_entries.append(mock_entry)
    
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Create adapter
    adapter = SQLiteMemoryAdapter(mock_memory_manager)
    
    # Test 1: Retrieve without metrics_collector
    result_without_metrics = adapter.retrieve(
        query="test query",
        limit=10,
        metrics_collector=None,
        logger_instance=None
    )
    
    # Reset mock to ensure clean state
    mock_memory_manager.query_memories.reset_mock()
    mock_memory_manager.query_memories.return_value = mock_entries
    
    # Test 2: Retrieve with metrics_collector
    metrics_collector = MetricsCollector()
    result_with_metrics = adapter.retrieve(
        query="test query",
        limit=10,
        metrics_collector=metrics_collector,
        logger_instance=None
    )
    
    # Verify results are identical (excluding timing metadata)
    assert result_without_metrics["total_count"] == result_with_metrics["total_count"], \
        "total_count should be identical with and without metrics_collector"
    
    assert len(result_without_metrics["memories"]) == len(result_with_metrics["memories"]), \
        "Number of memories should be identical with and without metrics_collector"
    
    # Compare each memory entry
    for mem_without, mem_with in zip(result_without_metrics["memories"], result_with_metrics["memories"]):
        assert mem_without["id"] == mem_with["id"], \
            f"Memory IDs should match: {mem_without['id']} vs {mem_with['id']}"
        assert mem_without["content"] == mem_with["content"], \
            f"Memory content should match for {mem_without['id']}"
        assert mem_without["metadata"] == mem_with["metadata"], \
            f"Memory metadata should match for {mem_without['id']}"
        assert mem_without["timestamp"] == mem_with["timestamp"], \
            f"Memory timestamp should match for {mem_without['id']}"
        assert mem_without["category"] == mem_with["category"], \
            f"Memory category should match for {mem_without['id']}"
        assert mem_without["tags"] == mem_with["tags"], \
            f"Memory tags should match for {mem_without['id']}"


@given(ranking_config_strategy(), st.lists(ranked_memory_strategy(), min_size=0, max_size=5))
@settings(max_examples=10, deadline=None)
def test_ranking_behavior_preservation(config, memories):
    """
    Property: RankingEngine.rank produces identical results with and without metrics_collector.
    
    **Validates: Requirements 13.5**
    
    This test verifies that instrumentation is strictly additive and does not
    change the business logic of the ranking operation. The test compares
    results from rank() with metrics_collector=None vs with a MetricsCollector
    instance and ensures they are identical.
    """
    # Test 1: Rank without metrics_collector
    engine_without_metrics = RankingEngine(config)
    result_without_metrics = engine_without_metrics.rank(memories)
    
    # Test 2: Rank with metrics_collector
    metrics_collector = MetricsCollector()
    engine_with_metrics = RankingEngine(config, metrics_collector=metrics_collector)
    result_with_metrics = engine_with_metrics.rank(memories)
    
    # Verify results are identical
    assert len(result_without_metrics) == len(result_with_metrics), \
        "Number of ranked memories should be identical with and without metrics_collector"
    
    # Compare each ranked memory
    for rank_without, rank_with in zip(result_without_metrics, result_with_metrics):
        assert rank_without.memory_id == rank_with.memory_id, \
            f"Memory IDs should match at same rank position"
        assert rank_without.final_score == rank_with.final_score, \
            f"Final scores should match for {rank_without.memory_id}"
        assert rank_without.recency_score == rank_with.recency_score, \
            f"Recency scores should match for {rank_without.memory_id}"
        assert rank_without.similarity_score == rank_with.similarity_score, \
            f"Similarity scores should match for {rank_without.memory_id}"
        assert rank_without.importance_score == rank_with.importance_score, \
            f"Importance scores should match for {rank_without.memory_id}"


@given(memory_list_strategy())
@settings(max_examples=10, deadline=None)
def test_lifecycle_cleanup_behavior_preservation(memories):
    """
    Property: MemoryLifecycleManager.cleanup produces identical results with and without metrics_collector.
    
    **Validates: Requirements 13.5**
    
    This test verifies that instrumentation is strictly additive and does not
    change the business logic of the cleanup operation. The test compares
    results from cleanup() with metrics_collector=None vs with a MetricsCollector
    instance and ensures they produce identical deletion behavior.
    """
    # Create mock memory interface
    class MockMemoryInterface(MemoryInterface):
        def __init__(self, initial_memories):
            self._memories = list(initial_memories)
            self._deleted_ids = []
        
        def retrieve(self, query=None, params=None, limit=10, metrics_collector=None, logger_instance=None):
            return {
                "memories": self._memories,
                "total_count": len(self._memories),
                "query_metadata": {}
            }
        
        def delete(self, memory_id: str):
            self._deleted_ids.append(memory_id)
            self._memories = [m for m in self._memories if m["id"] != memory_id]
        
        def store(self, content: str, metadata=None, category="general", tags=None):
            return "mem_new"
    
    # Create lifecycle config
    config = LifecycleConfig(
        max_total_memories=10,
        max_age_days=30,
        min_importance_protected=0.9
    )
    
    # Test 1: Cleanup without metrics_collector
    mock_memory_without = MockMemoryInterface(memories)
    manager_without = MemoryLifecycleManager(
        config,
        mock_memory_without,
        metrics_collector=None,
        logger=None
    )
    result_without = manager_without.cleanup()
    deleted_without = set(mock_memory_without._deleted_ids)
    
    # Test 2: Cleanup with metrics_collector
    mock_memory_with = MockMemoryInterface(memories)
    metrics_collector = MetricsCollector()
    manager_with = MemoryLifecycleManager(
        config,
        mock_memory_with,
        metrics_collector=metrics_collector,
        logger=None
    )
    result_with = manager_with.cleanup()
    deleted_with = set(mock_memory_with._deleted_ids)
    
    # Verify results are identical
    assert result_without.total_deleted == result_with.total_deleted, \
        "Total deleted count should be identical with and without metrics_collector"
    
    assert result_without.age_pruned == result_with.age_pruned, \
        "Age pruned count should be identical with and without metrics_collector"
    
    assert result_without.score_pruned == result_with.score_pruned, \
        "Score pruned count should be identical with and without metrics_collector"
    
    assert result_without.cap_pruned == result_with.cap_pruned, \
        "Cap pruned count should be identical with and without metrics_collector"
    
    assert result_without.final_count == result_with.final_count, \
        "Final count should be identical with and without metrics_collector"
    
    assert deleted_without == deleted_with, \
        "Exact same memory IDs should be deleted with and without metrics_collector"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
