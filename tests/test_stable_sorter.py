"""Unit tests for StableSorter component."""

import pytest
from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import StableSorter, RankedMemory


def create_test_memory(
    memory_id: str,
    similarity_score: float,
    final_score: float,
    timestamp: datetime
) -> RankedMemory:
    """Helper to create test memory."""
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content="test content",
        namespace="test",
        similarity_score=similarity_score,
        importance_score=0.0,
        recency_score=0.0,
        final_score=final_score,
        memory_entry=None
    )


def test_sort_by_final_score():
    """Test sorting with distinct final scores."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.5, 0.3, base_time),
        create_test_memory("2", 0.5, 0.8, base_time),
        create_test_memory("3", 0.5, 0.5, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 3
    assert result[0].memory_id == "2"  # 0.8
    assert result[1].memory_id == "3"  # 0.5
    assert result[2].memory_id == "1"  # 0.3


def test_tie_breaking_by_similarity():
    """Test tie-breaking at similarity level."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.3, 0.5, base_time),
        create_test_memory("2", 0.8, 0.5, base_time),
        create_test_memory("3", 0.5, 0.5, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 3
    assert result[0].memory_id == "2"  # sim 0.8
    assert result[1].memory_id == "3"  # sim 0.5
    assert result[2].memory_id == "1"  # sim 0.3


def test_tie_breaking_by_timestamp():
    """Test tie-breaking at timestamp level."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.5, 0.5, base_time - timedelta(hours=2)),
        create_test_memory("2", 0.5, 0.5, base_time),
        create_test_memory("3", 0.5, 0.5, base_time - timedelta(hours=1)),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 3
    assert result[0].memory_id == "2"  # newest
    assert result[1].memory_id == "3"  # middle
    assert result[2].memory_id == "1"  # oldest


def test_tie_breaking_by_memory_id():
    """Test tie-breaking at memory_id level."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("mem_c", 0.5, 0.5, base_time),
        create_test_memory("mem_a", 0.5, 0.5, base_time),
        create_test_memory("mem_b", 0.5, 0.5, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 3
    assert result[0].memory_id == "mem_a"  # lexicographical
    assert result[1].memory_id == "mem_b"
    assert result[2].memory_id == "mem_c"


def test_complete_tie_breaking_sequence():
    """Test complete 4-level tie-breaking sequence."""
    base_time = datetime.now(timezone.utc)
    memories = [
        # Different final scores
        create_test_memory("1", 0.5, 0.8, base_time),
        create_test_memory("2", 0.5, 0.6, base_time),
        # Same final score, different similarity
        create_test_memory("3", 0.7, 0.5, base_time),
        create_test_memory("4", 0.3, 0.5, base_time),
        # Same final and similarity, different timestamp
        create_test_memory("5", 0.5, 0.3, base_time),
        create_test_memory("6", 0.5, 0.3, base_time - timedelta(hours=1)),
        # Everything same, different id
        create_test_memory("7b", 0.2, 0.2, base_time),
        create_test_memory("7a", 0.2, 0.2, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 8
    # Highest final score
    assert result[0].memory_id == "1"
    assert result[1].memory_id == "2"
    # Same final score (0.5), ordered by similarity
    assert result[2].memory_id == "3"  # sim 0.7
    assert result[3].memory_id == "4"  # sim 0.3
    # Same final score (0.3), same similarity (0.5), ordered by timestamp
    assert result[4].memory_id == "5"  # newer
    assert result[5].memory_id == "6"  # older
    # Everything same, ordered by id
    assert result[6].memory_id == "7a"
    assert result[7].memory_id == "7b"


def test_sorting_preserves_input():
    """Test that sorting creates new list and doesn't modify input."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("1", 0.5, 0.3, base_time),
        create_test_memory("2", 0.5, 0.8, base_time),
    ]
    
    original_ids = [m.memory_id for m in memories]
    result = StableSorter.sort(memories)
    
    # Input unchanged
    assert [m.memory_id for m in memories] == original_ids
    # Result is different list
    assert result is not memories
    # Result is sorted
    assert result[0].memory_id == "2"
    assert result[1].memory_id == "1"


def test_empty_list():
    """Test sorting empty list."""
    result = StableSorter.sort([])
    assert len(result) == 0


def test_single_memory():
    """Test sorting single memory."""
    base_time = datetime.now(timezone.utc)
    memory = create_test_memory("1", 0.5, 0.5, base_time)
    
    result = StableSorter.sort([memory])
    
    assert len(result) == 1
    assert result[0].memory_id == "1"


def test_identical_memories_different_ids():
    """Test sorting memories with identical scores but different IDs."""
    base_time = datetime.now(timezone.utc)
    memories = [
        create_test_memory("id_3", 0.5, 0.5, base_time),
        create_test_memory("id_1", 0.5, 0.5, base_time),
        create_test_memory("id_2", 0.5, 0.5, base_time),
    ]
    
    result = StableSorter.sort(memories)
    
    assert len(result) == 3
    assert result[0].memory_id == "id_1"
    assert result[1].memory_id == "id_2"
    assert result[2].memory_id == "id_3"
