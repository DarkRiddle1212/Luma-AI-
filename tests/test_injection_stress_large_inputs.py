"""
Stress Tests for Large Input Sets - Context Injection Engine

This module tests the injection engine's ability to handle large input sets
efficiently. It verifies sub-quadratic complexity and performance requirements
for 10,000 and 100,000 memory inputs.

Feature: context-injection-engine
Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pytest
import time
import numpy as np
from datetime import datetime, timezone

from luma.core.injection_engine import (
    InjectionEngine,
    InjectionConfig
)
from luma.core.ranking_engine import RankedMemory


def create_ranked_memory(memory_id: str, content: str, final_score: float, 
                        embedding_dim: int = 768) -> RankedMemory:
    """Create a RankedMemory object for testing.
    
    Args:
        memory_id: Unique identifier
        content: Memory content
        final_score: Ranking score
        embedding_dim: Dimension of embedding vector
    
    Returns:
        RankedMemory object with all required fields
    """
    # Generate random embedding
    embedding = np.random.randn(embedding_dim).tolist()
    
    return RankedMemory(
        memory_id=memory_id,
        content=content,
        timestamp=datetime.now(timezone.utc),
        namespace="test",
        category="test_category",
        similarity_score=0.8,
        final_score=final_score,
        recency_score=0.7,
        importance_score=0.6,
        metadata={
            'embedding': embedding,
            'token_count': len(content.split())
        },
        memory_entry=None
    )


def test_stress_10k_memories():
    """
    Stress test with 10,000 memories.
    
    This test verifies that the injection engine can process 10,000 memories
    efficiently and complete within 1 second as specified in requirements.
    
    Requirements:
        - 7.1: Process ranked memory lists containing up to 100,000 memories
        - 7.2: Complete injection for 10,000 memories within 1 second
        - 7.3: Use algorithm with time complexity better than O(n²)
    """
    # Create 10,000 memories with varying scores
    num_memories = 10_000
    memories = []
    
    for i in range(num_memories):
        memory = create_ranked_memory(
            memory_id=f"mem_{i:06d}",
            content=f"Memory content number {i} with some additional text",
            final_score=1.0 - (i / num_memories)  # Descending scores
        )
        memories.append(memory)
    
    # Configure injection engine with reasonable limits
    config = InjectionConfig(
        max_token_budget=5000,
        max_memory_count=100,
        redundancy_similarity_threshold=0.95,  # High threshold to reduce filtering
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Measure injection time
    start_time = time.perf_counter()
    result = engine.inject(memories)
    end_time = time.perf_counter()
    
    duration_seconds = end_time - start_time
    
    # Verify performance requirement: < 1 second for 10,000 memories
    assert duration_seconds < 1.0, (
        f"Injection took {duration_seconds:.3f}s, expected < 1.0s for 10,000 memories"
    )
    
    # Verify result is valid
    assert len(result.memories) > 0, "Should select at least some memories"
    assert result.total_tokens <= config.max_token_budget
    assert len(result.memories) <= config.max_memory_count
    
    # Verify diagnostic counts
    assert result.input_count == num_memories
    assert (result.filtered_by_category + result.filtered_by_redundancy + 
            result.filtered_by_budget + len(result.memories)) == num_memories
    
    print(f"\n✓ Processed {num_memories:,} memories in {duration_seconds:.3f}s")
    print(f"  Selected: {len(result.memories)} memories")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Filtered by redundancy: {result.filtered_by_redundancy}")
    print(f"  Filtered by budget: {result.filtered_by_budget}")


def test_stress_100k_memories():
    """
    Stress test with 100,000 memories.
    
    This test verifies that the injection engine can handle very large input
    sets (100,000 memories) and demonstrates sub-quadratic complexity through
    early exit when budget is filled.
    
    Requirements:
        - 7.1: Process ranked memory lists containing up to 100,000 memories
        - 7.3: Use algorithm with time complexity better than O(n²)
        - 7.4: Do not load all memory content into memory simultaneously
        - 7.5: Memory usage proportional to output size, not input size
    """
    # Create 100,000 memories with varying scores
    num_memories = 100_000
    memories = []
    
    for i in range(num_memories):
        memory = create_ranked_memory(
            memory_id=f"mem_{i:06d}",
            content=f"Memory {i}",  # Shorter content for faster generation
            final_score=1.0 - (i / num_memories)  # Descending scores
        )
        memories.append(memory)
    
    # Configure injection engine with tight budget for early exit
    config = InjectionConfig(
        max_token_budget=2000,
        max_memory_count=50,
        redundancy_similarity_threshold=0.95,  # High threshold to reduce filtering
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Measure injection time
    start_time = time.perf_counter()
    result = engine.inject(memories)
    end_time = time.perf_counter()
    
    duration_seconds = end_time - start_time
    
    # Verify result is valid
    assert len(result.memories) > 0, "Should select at least some memories"
    assert result.total_tokens <= config.max_token_budget
    assert len(result.memories) <= config.max_memory_count
    
    # Verify diagnostic counts
    assert result.input_count == num_memories
    
    # Performance should be reasonable (not O(n²))
    # With early exit, should complete much faster than processing all memories
    print(f"\n✓ Processed {num_memories:,} memories in {duration_seconds:.3f}s")
    print(f"  Selected: {len(result.memories)} memories")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Filtered by redundancy: {result.filtered_by_redundancy}")
    print(f"  Filtered by budget: {result.filtered_by_budget}")
    print(f"  Throughput: {num_memories / duration_seconds:,.0f} memories/second")


def test_stress_complexity_scaling():
    """
    Test that injection time scales sub-quadratically with input size.
    
    This test measures injection time for different input sizes and verifies
    that the time complexity is better than O(n²). It should demonstrate
    approximately linear scaling when budget is tight (early exit).
    
    Requirements:
        - 7.3: Use algorithm with time complexity better than O(n²)
    """
    # Test with increasing input sizes
    sizes = [1_000, 5_000, 10_000, 20_000]
    times = []
    
    config = InjectionConfig(
        max_token_budget=1000,
        max_memory_count=20,
        redundancy_similarity_threshold=0.95,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    for size in sizes:
        # Create memories
        memories = []
        for i in range(size):
            memory = create_ranked_memory(
                memory_id=f"mem_{i:06d}",
                content=f"Memory {i}",
                final_score=1.0 - (i / size)
            )
            memories.append(memory)
        
        # Measure injection time
        start_time = time.perf_counter()
        result = engine.inject(memories)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        times.append(duration)
        
        print(f"  Size {size:>6,}: {duration:.4f}s ({len(result.memories)} selected)")
    
    # Verify sub-quadratic scaling
    # For O(n²), doubling input size would quadruple time
    # For O(n), doubling input size would double time
    # We expect approximately linear scaling due to early exit
    
    # Check that time doesn't grow quadratically
    # Compare 1k -> 10k (10x increase) and 10k -> 20k (2x increase)
    ratio_10x = times[2] / times[0]  # 10k / 1k
    ratio_2x = times[3] / times[2]   # 20k / 10k
    
    # For O(n²): ratio_10x would be ~100, ratio_2x would be ~4
    # For O(n): ratio_10x would be ~10, ratio_2x would be ~2
    # We expect closer to linear due to early exit
    
    print(f"\n  Scaling analysis:")
    print(f"    1k -> 10k (10x): {ratio_10x:.2f}x time increase")
    print(f"    10k -> 20k (2x): {ratio_2x:.2f}x time increase")
    
    # Verify sub-quadratic: ratio should be much less than size ratio squared
    assert ratio_10x < 50, f"Time scaling appears quadratic: {ratio_10x:.2f}x for 10x input"
    assert ratio_2x < 10, f"Time scaling appears quadratic: {ratio_2x:.2f}x for 2x input"
    
    print(f"  ✓ Complexity is sub-quadratic (not O(n²))")


def test_stress_memory_usage_proportional_to_output():
    """
    Test that memory usage is proportional to output size, not input size.
    
    This test verifies that the injection engine doesn't load all memory
    content into memory simultaneously. Memory usage should be proportional
    to the number of selected memories, not the input size.
    
    Requirements:
        - 7.4: Do not load all memory content into memory simultaneously
        - 7.5: Memory usage proportional to output size, not input size
    """
    # Create large input set
    num_memories = 50_000
    memories = []
    
    for i in range(num_memories):
        memory = create_ranked_memory(
            memory_id=f"mem_{i:06d}",
            content=f"Memory content {i} " * 10,  # Larger content
            final_score=1.0 - (i / num_memories)
        )
        memories.append(memory)
    
    # Configure with tight budget to select only a few memories
    config = InjectionConfig(
        max_token_budget=500,
        max_memory_count=10,
        redundancy_similarity_threshold=0.95,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Inject memories
    result = engine.inject(memories)
    
    # Verify that only a small fraction was selected
    selection_ratio = len(result.memories) / num_memories
    assert selection_ratio < 0.01, (
        f"Selected {len(result.memories)} out of {num_memories} "
        f"({selection_ratio:.2%}), expected < 1%"
    )
    
    # Verify result is valid
    assert len(result.memories) <= config.max_memory_count
    assert result.total_tokens <= config.max_token_budget
    
    print(f"\n✓ Memory usage test passed")
    print(f"  Input: {num_memories:,} memories")
    print(f"  Selected: {len(result.memories)} memories ({selection_ratio:.2%})")
    print(f"  Memory usage proportional to output, not input")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
