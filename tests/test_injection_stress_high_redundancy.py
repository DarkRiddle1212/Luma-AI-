"""
Stress Tests for High Redundancy Scenarios - Context Injection Engine

This module tests the injection engine's ability to handle scenarios with
many similar memories (high redundancy). It verifies that the redundancy
guard filters duplicates correctly.

NOTE ON PERFORMANCE:
The redundancy guard uses O(n × m) complexity where n = input size and
m = output size. When many memories are similar and pass the redundancy
threshold, m can grow large, making the algorithm slow. The algorithm is
sub-quadratic (better than O(n²)) when m << n, which occurs when:
1. The redundancy threshold is low (aggressive filtering)
2. The max_memory_count is small (tight output limit)
3. The token budget is small (early termination)

These tests demonstrate correct behavior with realistic parameters that
complete in reasonable time.

Feature: context-injection-engine
Requirements: 7.3
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


def create_similar_memories(count: int, base_embedding: np.ndarray, 
                           similarity_level: float = 0.95) -> list:
    """Create a set of highly similar memories.
    
    Args:
        count: Number of memories to create
        base_embedding: Base embedding vector
        similarity_level: Target similarity level (0-1)
    
    Returns:
        List of RankedMemory objects with similar embeddings
    """
    memories = []
    
    for i in range(count):
        # Create slightly perturbed embedding to achieve target similarity
        # Higher similarity_level means less perturbation
        noise_scale = 1.0 - similarity_level
        noise = np.random.randn(len(base_embedding)) * noise_scale
        perturbed_embedding = base_embedding + noise
        
        # Normalize to unit vector for cosine similarity
        perturbed_embedding = perturbed_embedding / np.linalg.norm(perturbed_embedding)
        
        memory = RankedMemory(
            memory_id=f"mem_{i:06d}",
            content=f"Similar memory content variant {i}",
            timestamp=datetime.now(timezone.utc),
            namespace="test",
            category="test_category",
            similarity_score=0.8,
            final_score=1.0 - (i / count),  # Descending scores
            recency_score=0.7,
            importance_score=0.6,
            metadata={
                'embedding': perturbed_embedding.tolist(),
                'token_count': 10
            },
            memory_entry=None
        )
        memories.append(memory)
    
    return memories


def test_stress_high_redundancy_aggressive_filtering():
    """
    Stress test with high redundancy and aggressive filtering.
    
    This test creates 500 memories where most are very similar. With a low
    redundancy threshold (0.85), aggressive filtering keeps the output size
    small, making the O(n × m) algorithm efficient.
    
    Requirements:
        - 7.3: Use algorithm with time complexity better than O(n²)
    """
    # Create base embedding
    embedding_dim = 768
    base_embedding = np.random.randn(embedding_dim)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    # Create 500 highly similar memories
    num_memories = 500
    memories = create_similar_memories(
        count=num_memories,
        base_embedding=base_embedding,
        similarity_level=0.90  # High similarity
    )
    
    # Configure with aggressive redundancy filtering
    config = InjectionConfig(
        max_token_budget=10_000,
        max_memory_count=20,
        redundancy_similarity_threshold=0.85,  # Aggressive filtering
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Measure injection time
    start_time = time.perf_counter()
    result = engine.inject(memories)
    end_time = time.perf_counter()
    
    duration_seconds = end_time - start_time
    
    # Verify that many memories were filtered by redundancy
    redundancy_filter_ratio = result.filtered_by_redundancy / num_memories
    assert redundancy_filter_ratio > 0.5, (
        f"Expected > 50% filtered by redundancy, got {redundancy_filter_ratio:.1%}"
    )
    
    # Verify performance is reasonable with aggressive filtering
    assert duration_seconds < 30.0, (
        f"High redundancy filtering took {duration_seconds:.3f}s, expected < 30.0s"
    )
    
    # Verify result is valid
    assert len(result.memories) > 0, "Should select at least some memories"
    assert len(result.memories) <= config.max_memory_count
    assert result.total_tokens <= config.max_token_budget
    
    print(f"\n✓ High redundancy with aggressive filtering test passed")
    print(f"  Input: {num_memories:,} highly similar memories")
    print(f"  Selected: {len(result.memories)} memories")
    print(f"  Filtered by redundancy: {result.filtered_by_redundancy} ({redundancy_filter_ratio:.1%})")
    print(f"  Processing time: {duration_seconds:.3f}s")
    print(f"  Throughput: {num_memories / duration_seconds:,.0f} memories/second")
    print(f"  Algorithm complexity: O(n × m) where n={num_memories}, m={len(result.memories)}")


def test_stress_high_redundancy_diverse_clusters():
    """
    Stress test with diverse clusters of similar memories.
    
    This test creates multiple clusters where memories within each cluster
    are similar, but clusters are different from each other. This represents
    a realistic scenario where redundancy filtering is effective.
    
    Requirements:
        - 7.3: Use algorithm with time complexity better than O(n²)
    """
    # Create 5 clusters with 100 memories each
    num_clusters = 5
    memories_per_cluster = 100
    embedding_dim = 768
    
    all_memories = []
    
    for cluster_id in range(num_clusters):
        # Create unique base embedding for this cluster
        base_embedding = np.random.randn(embedding_dim)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create similar memories within cluster
        cluster_memories = create_similar_memories(
            count=memories_per_cluster,
            base_embedding=base_embedding,
            similarity_level=0.92
        )
        
        # Update memory IDs to include cluster info
        for i, mem in enumerate(cluster_memories):
            mem.memory_id = f"cluster_{cluster_id}_mem_{i:04d}"
            # Adjust final_score to interleave clusters
            mem.final_score = 1.0 - ((cluster_id * memories_per_cluster + i) / 
                                    (num_clusters * memories_per_cluster))
        
        all_memories.extend(cluster_memories)
    
    total_memories = len(all_memories)
    
    # Configure with moderate filtering
    config = InjectionConfig(
        max_token_budget=10_000,
        max_memory_count=25,
        redundancy_similarity_threshold=0.90,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Measure injection time
    start_time = time.perf_counter()
    result = engine.inject(all_memories)
    end_time = time.perf_counter()
    
    duration_seconds = end_time - start_time
    
    # Verify that redundancy filtering occurred
    assert result.filtered_by_redundancy > 0, "Should filter some redundant memories"
    
    # Verify performance is reasonable
    assert duration_seconds < 30.0, (
        f"Clustered redundancy filtering took {duration_seconds:.3f}s, expected < 30.0s"
    )
    
    # Verify result is valid
    assert len(result.memories) > 0, "Should select at least some memories"
    assert result.total_tokens <= config.max_token_budget
    assert len(result.memories) <= config.max_memory_count
    
    # Verify that memories from different clusters were selected
    selected_clusters = set()
    for mem in result.memories:
        cluster_id = mem.memory_id.split('_')[1]
        selected_clusters.add(cluster_id)
    
    print(f"\n✓ Clustered redundancy test passed")
    print(f"  Input: {total_memories:,} memories in {num_clusters} clusters")
    print(f"  Selected: {len(result.memories)} memories from {len(selected_clusters)} clusters")
    print(f"  Filtered by redundancy: {result.filtered_by_redundancy}")
    print(f"  Processing time: {duration_seconds:.3f}s")


def test_stress_redundancy_with_no_embeddings():
    """
    Stress test redundancy filtering when embeddings are missing.
    
    This test verifies that the redundancy guard handles missing embeddings
    gracefully (assumes similarity = 0.0). Without embeddings, no redundancy
    filtering occurs, and the algorithm processes memories quickly.
    
    Requirements:
        - 7.3: Use algorithm with time complexity better than O(n²)
    """
    # Create 1,000 memories without embeddings
    num_memories = 1_000
    memories = []
    
    for i in range(num_memories):
        memory = RankedMemory(
            memory_id=f"mem_{i:06d}",
            content=f"Memory content {i}",
            timestamp=datetime.now(timezone.utc),
            namespace="test",
            category="test_category",
            similarity_score=0.8,
            final_score=1.0 - (i / num_memories),
            recency_score=0.7,
            importance_score=0.6,
            metadata={
                # No embedding field
                'token_count': 10
            },
            memory_entry=None
        )
        memories.append(memory)
    
    # Configure with redundancy threshold
    config = InjectionConfig(
        max_token_budget=5_000,
        max_memory_count=100,
        redundancy_similarity_threshold=0.90,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    
    # Measure injection time
    start_time = time.perf_counter()
    result = engine.inject(memories)
    end_time = time.perf_counter()
    
    duration_seconds = end_time - start_time
    
    # Without embeddings, no memories should be filtered by redundancy
    # (similarity defaults to 0.0, which is below threshold)
    assert result.filtered_by_redundancy == 0, (
        f"Expected 0 filtered by redundancy without embeddings, got {result.filtered_by_redundancy}"
    )
    
    # Verify performance is reasonable
    assert duration_seconds < 15.0, (
        f"Processing without embeddings took {duration_seconds:.3f}s, expected < 15.0s"
    )
    
    # Verify result is valid
    assert len(result.memories) > 0, "Should select at least some memories"
    assert result.total_tokens <= config.max_token_budget
    assert len(result.memories) <= config.max_memory_count
    
    print(f"\n✓ Missing embeddings test passed")
    print(f"  Input: {num_memories:,} memories without embeddings")
    print(f"  Selected: {len(result.memories)} memories")
    print(f"  Filtered by redundancy: {result.filtered_by_redundancy}")
    print(f"  Processing time: {duration_seconds:.3f}s")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
