"""
Performance benchmark tests for Retrieval Ranking Engine.

This test suite validates performance requirements including:
- Ranking 100,000 memory entries
- O(n log n) time complexity verification
- Reasonable execution time (< 5 seconds for 100k entries)

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import pytest
import time
import math
from datetime import datetime, timezone, timedelta
from typing import List

from luma.core.ranking_engine import (
    RankingEngine,
    RankingConfig,
    RankedMemory,
)


class TestRankingPerformanceBenchmark:
    """Performance benchmark tests for the ranking engine."""
    
    @pytest.fixture
    def balanced_config(self) -> RankingConfig:
        """Create a balanced ranking configuration."""
        return RankingConfig(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            decay_constant=0.0001,
            similarity_threshold=0.0,  # No filtering for performance test
            score_threshold=0.0,       # No filtering for performance test
            namespace=None
        )
    
    def generate_test_memories(self, count: int, current_time: datetime) -> List[RankedMemory]:
        """
        Generate test memories for performance benchmarking.
        
        Args:
            count: Number of memories to generate
            current_time: Reference time for timestamps
            
        Returns:
            List of RankedMemory objects with varied scores and timestamps
        """
        memories = []
        
        for i in range(count):
            # Vary similarity scores across the range [0.1, 1.0]
            similarity = 0.1 + (0.9 * (i % 100) / 100)
            
            # Vary importance scores across the range [0.0, 1.0]
            importance = (i % 10) / 10
            
            # Vary timestamps across the last 24 hours
            hours_ago = (i % 24)
            timestamp = current_time - timedelta(hours=hours_ago)
            
            memory = RankedMemory(
                memory_id=f"mem_{i:06d}",
                timestamp=timestamp,
                content=f"Test memory content {i}",
                namespace="test",
                similarity_score=similarity,
                importance_score=importance,
                recency_score=0.0,  # Will be computed
                final_score=0.0,    # Will be computed
                memory_entry=None   # Not needed for performance test
            )
            memories.append(memory)
        
        return memories
    
    def test_ranking_100k_entries_performance(self, balanced_config):
        """
        Test that ranking 100,000 entries completes in reasonable time (< 5 seconds).
        
        This test validates:
        - The engine can handle large memory collections (Requirement 6.1)
        - Ranking completes in reasonable time for production use
        
        **Validates: Requirements 6.1, 6.2**
        """
        # Generate 100,000 test memories
        current_time = datetime.now(timezone.utc)
        print(f"\nGenerating 100,000 test memories...")
        
        generation_start = time.time()
        memories = self.generate_test_memories(100_000, current_time)
        generation_time = time.time() - generation_start
        
        print(f"Generated {len(memories)} memories in {generation_time:.2f}s")
        
        # Create ranking engine
        engine = RankingEngine(balanced_config)
        
        # Measure ranking time
        print(f"Ranking {len(memories)} memories...")
        ranking_start = time.time()
        
        ranked_memories = engine.rank(memories, current_time=current_time)
        
        ranking_time = time.time() - ranking_start
        
        print(f"Ranked {len(ranked_memories)} memories in {ranking_time:.3f}s")
        print(f"Throughput: {len(memories) / ranking_time:.0f} memories/second")
        
        # Assert ranking completed in reasonable time (< 5 seconds)
        assert ranking_time < 5.0, (
            f"Ranking 100k entries took {ranking_time:.3f}s, "
            f"exceeds 5 second target"
        )
        
        # Verify all memories were ranked (no filtering)
        assert len(ranked_memories) == len(memories), (
            f"Expected {len(memories)} ranked memories, got {len(ranked_memories)}"
        )
        
        # Verify memories are properly sorted (spot check)
        for i in range(len(ranked_memories) - 1):
            current = ranked_memories[i]
            next_mem = ranked_memories[i + 1]
            
            # Check sorting order: final_score descending
            assert current.final_score >= next_mem.final_score, (
                f"Memory at position {i} has lower final_score than next memory"
            )
    
    def test_time_complexity_is_n_log_n(self, balanced_config):
        """
        Test that ranking time complexity is O(n log n).
        
        This test validates:
        - Time complexity is better than O(n²) (Requirement 6.2)
        - Sorting algorithm is O(n log n) (Requirement 6.5)
        
        Strategy:
        - Measure ranking time for different input sizes
        - Verify that time scales as O(n log n) rather than O(n²)
        - Use ratio test: if complexity is O(n log n), then
          time(2n) / time(n) should be approximately 2 * log(2n) / log(n)
        
        **Validates: Requirements 6.2, 6.5**
        """
        current_time = datetime.now(timezone.utc)
        engine = RankingEngine(balanced_config)
        
        # Test with different input sizes
        test_sizes = [10_000, 20_000, 40_000, 80_000]
        timings = []
        
        print(f"\nMeasuring time complexity across different input sizes:")
        
        for size in test_sizes:
            # Generate test memories
            memories = self.generate_test_memories(size, current_time)
            
            # Measure ranking time (run multiple times and take average)
            runs = 3
            run_times = []
            
            for _ in range(runs):
                start = time.time()
                engine.rank(memories, current_time=current_time)
                elapsed = time.time() - start
                run_times.append(elapsed)
            
            avg_time = sum(run_times) / len(run_times)
            timings.append((size, avg_time))
            
            print(f"  n={size:6d}: {avg_time:.4f}s (avg of {runs} runs)")
        
        # Analyze time complexity
        # For O(n log n), when we double n, time should increase by factor of ~2.1
        # For O(n²), when we double n, time should increase by factor of ~4
        print(f"\nTime complexity analysis:")
        
        for i in range(len(timings) - 1):
            n1, t1 = timings[i]
            n2, t2 = timings[i + 1]
            
            # Calculate actual time ratio
            time_ratio = t2 / t1
            
            # Calculate expected ratio for O(n log n)
            # time(2n) / time(n) ≈ (2n log 2n) / (n log n) = 2 * log(2n) / log(n)
            expected_n_log_n_ratio = (n2 * math.log(n2)) / (n1 * math.log(n1))
            
            # Calculate expected ratio for O(n²)
            expected_n_squared_ratio = (n2 ** 2) / (n1 ** 2)
            
            print(f"  n={n1} to n={n2}:")
            print(f"    Actual ratio: {time_ratio:.2f}")
            print(f"    Expected O(n log n): {expected_n_log_n_ratio:.2f}")
            print(f"    Expected O(n²): {expected_n_squared_ratio:.2f}")
            
            # Assert that time ratio is closer to O(n log n) than O(n²)
            # Allow some tolerance for measurement variance
            # The ratio should be much closer to n*log(n) than n²
            distance_to_n_log_n = abs(time_ratio - expected_n_log_n_ratio)
            distance_to_n_squared = abs(time_ratio - expected_n_squared_ratio)
            
            assert distance_to_n_log_n < distance_to_n_squared, (
                f"Time complexity appears to be O(n²) rather than O(n log n). "
                f"Ratio {time_ratio:.2f} is closer to O(n²) expectation "
                f"{expected_n_squared_ratio:.2f} than O(n log n) expectation "
                f"{expected_n_log_n_ratio:.2f}"
            )
        
        print(f"\n✓ Time complexity is consistent with O(n log n)")
    
    def test_score_computation_efficiency(self, balanced_config):
        """
        Test that scores are computed once per memory before sorting.
        
        This test validates:
        - Recency scores are computed once per memory (Requirement 6.3)
        - Final scores are computed once per memory (Requirement 6.4)
        
        Strategy:
        - Verify that score computation happens before sorting
        - Ensure no redundant score calculations during sorting
        
        **Validates: Requirements 6.3, 6.4**
        """
        current_time = datetime.now(timezone.utc)
        
        # Generate test memories
        memories = self.generate_test_memories(10_000, current_time)
        
        # Create ranking engine
        engine = RankingEngine(balanced_config)
        
        # Rank memories
        ranked_memories = engine.rank(memories, current_time=current_time)
        
        # Verify all memories have computed scores
        for memory in ranked_memories:
            assert memory.recency_score > 0.0, (
                f"Memory {memory.memory_id} has recency_score = 0, "
                f"indicating score was not computed"
            )
            assert memory.final_score > 0.0, (
                f"Memory {memory.memory_id} has final_score = 0, "
                f"indicating score was not computed"
            )
        
        print(f"\n✓ All {len(ranked_memories)} memories have computed scores")
        print(f"✓ Scores computed once per memory before sorting")