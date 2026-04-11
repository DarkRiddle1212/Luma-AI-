"""
Property-based tests for performance scalability characteristics.

**Property 10: Performance scalability**
**Validates: Requirements 6.1, 6.2, 6.5**

This test suite validates that the ranking engine's performance scales
sub-quadratically (better than O(n²)) with input size, confirming the
O(n log n) time complexity of the sorting algorithm.
"""

import pytest
import time
import math
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

from luma.core.ranking_engine import (
    RankingEngine,
    RankingConfig,
    RankedMemory,
)


class TestPerformanceScalability:
    """Property-based tests for performance scalability."""
    
    @pytest.fixture
    def performance_config(self) -> RankingConfig:
        """Create a configuration for performance testing."""
        return RankingConfig(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            decay_constant=0.0001,
            similarity_threshold=0.0,  # No filtering for performance test
            score_threshold=0.0,       # No filtering for performance test
            namespace=None
        )
    
    def generate_memories(self, count: int, current_time: datetime) -> List[RankedMemory]:
        """
        Generate test memories for performance testing.
        
        Args:
            count: Number of memories to generate
            current_time: Reference time for timestamps
            
        Returns:
            List of RankedMemory objects with varied scores
        """
        memories = []
        
        for i in range(count):
            # Vary similarity scores
            similarity = 0.1 + (0.9 * (i % 100) / 100)
            
            # Vary importance scores
            importance = (i % 10) / 10
            
            # Vary timestamps
            hours_ago = (i % 24)
            timestamp = current_time - timedelta(hours=hours_ago)
            
            memory = RankedMemory(
                memory_id=f"mem_{i:06d}",
                timestamp=timestamp,
                content=f"Test memory {i}",
                namespace="test",
                similarity_score=similarity,
                importance_score=importance,
                recency_score=0.0,
                final_score=0.0,
                memory_entry=None
            )
            memories.append(memory)
        
        return memories
    
    def measure_ranking_time(
        self,
        engine: RankingEngine,
        memories: List[RankedMemory],
        current_time: datetime,
        runs: int = 3
    ) -> float:
        """
        Measure average ranking time over multiple runs.
        
        Args:
            engine: Ranking engine instance
            memories: Memories to rank
            current_time: Reference time
            runs: Number of runs to average
            
        Returns:
            Average ranking time in seconds
        """
        times = []
        
        for _ in range(runs):
            start = time.time()
            engine.rank(memories, current_time=current_time)
            elapsed = time.time() - start
            times.append(elapsed)
        
        return sum(times) / len(times)
    
    def test_sub_quadratic_scaling(self, performance_config):
        """
        Property: Ranking time scales sub-quadratically with input size.
        
        This test verifies that the ranking engine's time complexity is better
        than O(n²) by measuring actual performance across different input sizes
        and comparing the growth rate to quadratic growth.
        
        For O(n log n) complexity:
        - When input size doubles, time should increase by ~2x (not 4x as in O(n²))
        - The ratio time(2n) / time(n) should be close to 2 * log(2n) / log(n)
        
        **Validates: Requirements 6.1, 6.2, 6.5**
        """
        current_time = datetime.now(timezone.utc)
        engine = RankingEngine(performance_config)
        
        # Test with progressively larger input sizes
        test_sizes = [1_000, 10_000, 100_000]
        timings: List[Tuple[int, float]] = []
        
        print(f"\nMeasuring performance scalability:")
        
        for size in test_sizes:
            # Generate test memories
            memories = self.generate_memories(size, current_time)
            
            # Measure ranking time
            avg_time = self.measure_ranking_time(
                engine, memories, current_time, runs=3
            )
            
            timings.append((size, avg_time))
            print(f"  n={size:7d}: {avg_time:.4f}s")
        
        # Verify sub-quadratic scaling
        # Compare each consecutive pair of measurements
        for i in range(len(timings) - 1):
            n1, t1 = timings[i]
            n2, t2 = timings[i + 1]
            
            # Calculate actual time ratio
            time_ratio = t2 / t1
            
            # Calculate size ratio
            size_ratio = n2 / n1
            
            # For O(n²), time_ratio would equal size_ratio²
            quadratic_ratio = size_ratio ** 2
            
            # For O(n log n), time_ratio should be approximately:
            # (n2 log n2) / (n1 log n1) = size_ratio * (log n2 / log n1)
            expected_nlogn_ratio = size_ratio * (math.log(n2) / math.log(n1))
            
            print(f"\n  Scaling from n={n1} to n={n2}:")
            print(f"    Size ratio: {size_ratio:.2f}x")
            print(f"    Time ratio: {time_ratio:.2f}x")
            print(f"    Expected for O(n log n): {expected_nlogn_ratio:.2f}x")
            print(f"    Expected for O(n²): {quadratic_ratio:.2f}x")
            
            # Assert that time scaling is much closer to O(n log n) than O(n²)
            # Time ratio should be significantly less than quadratic ratio
            # Allow some variance due to system noise and constant factors
            
            # The time ratio should be less than 80% of the quadratic ratio
            # This gives us confidence that complexity is sub-quadratic
            assert time_ratio < 0.8 * quadratic_ratio, (
                f"Time scaling appears quadratic or worse: "
                f"time_ratio={time_ratio:.2f}x is not significantly less than "
                f"quadratic_ratio={quadratic_ratio:.2f}x for n={n1} to n={n2}"
            )
            
            # Additionally, verify time ratio is reasonably close to O(n log n)
            # Allow up to 3x variance due to constant factors and system noise
            assert time_ratio < 3 * expected_nlogn_ratio, (
                f"Time scaling is worse than expected for O(n log n): "
                f"time_ratio={time_ratio:.2f}x exceeds "
                f"3 * expected_nlogn_ratio={3 * expected_nlogn_ratio:.2f}x"
            )
    
    def test_handles_100k_memories(self, performance_config):
        """
        Property: Ranking engine can process 100,000 memories.
        
        This test validates that the ranking engine can handle large
        collections as specified in the requirements.
        
        **Validates: Requirements 6.1**
        """
        current_time = datetime.now(timezone.utc)
        engine = RankingEngine(performance_config)
        
        # Generate 100,000 memories
        print(f"\nGenerating 100,000 memories...")
        memories = self.generate_memories(100_000, current_time)
        
        assert len(memories) == 100_000, "Should generate 100k memories"
        
        # Rank the memories
        print(f"Ranking 100,000 memories...")
        start = time.time()
        ranked = engine.rank(memories, current_time=current_time)
        elapsed = time.time() - start
        
        print(f"Ranked {len(ranked)} memories in {elapsed:.3f}s")
        print(f"Throughput: {len(memories) / elapsed:.0f} memories/second")
        
        # Verify all memories were processed
        assert len(ranked) == 100_000, (
            f"Expected 100,000 ranked memories, got {len(ranked)}"
        )
        
        # Verify memories are properly sorted
        # Note: Current implementation sorts by final_score first, then similarity_score
        for i in range(len(ranked) - 1):
            current = ranked[i]
            next_mem = ranked[i + 1]
            
            # Verify sorting order (final_score primary, similarity_score secondary)
            if current.final_score == next_mem.final_score:
                assert current.similarity_score >= next_mem.similarity_score, (
                    f"Memory at position {i} has lower similarity_score than next"
                )
            else:
                assert current.final_score >= next_mem.final_score, (
                    f"Memory at position {i} has lower final_score than next"
                )
    
    def test_linear_score_computation(self, performance_config):
        """
        Property: Score computation time scales linearly with input size.
        
        This test validates that recency and final scores are computed once
        per memory (O(n) complexity) before sorting.
        
        **Validates: Requirements 6.3, 6.4**
        """
        current_time = datetime.now(timezone.utc)
        engine = RankingEngine(performance_config)
        
        # Test with different sizes
        test_sizes = [1_000, 10_000, 50_000]
        timings = []
        
        print(f"\nMeasuring score computation scalability:")
        
        for size in test_sizes:
            memories = self.generate_memories(size, current_time)
            
            # Measure time
            avg_time = self.measure_ranking_time(
                engine, memories, current_time, runs=3
            )
            
            timings.append((size, avg_time))
            print(f"  n={size:6d}: {avg_time:.4f}s")
        
        # Verify that time per memory is relatively constant
        # (indicating O(n) score computation + O(n log n) sorting)
        for i in range(len(timings) - 1):
            n1, t1 = timings[i]
            n2, t2 = timings[i + 1]
            
            # Time per memory
            time_per_memory_1 = t1 / n1
            time_per_memory_2 = t2 / n2
            
            # For O(n log n), time per memory should grow slowly (by log factor)
            # time_per_memory_2 / time_per_memory_1 ≈ log(n2) / log(n1)
            expected_ratio = math.log(n2) / math.log(n1)
            actual_ratio = time_per_memory_2 / time_per_memory_1
            
            print(f"\n  Time per memory from n={n1} to n={n2}:")
            print(f"    {time_per_memory_1*1e6:.2f}µs -> {time_per_memory_2*1e6:.2f}µs")
            print(f"    Ratio: {actual_ratio:.2f}x (expected ~{expected_ratio:.2f}x for O(n log n))")
            
            # Verify the ratio is reasonable for O(n log n)
            # Allow up to 2x variance
            assert actual_ratio < 2 * expected_ratio, (
                f"Time per memory growing too fast: "
                f"actual_ratio={actual_ratio:.2f}x exceeds "
                f"2 * expected_ratio={2 * expected_ratio:.2f}x"
            )
    
    def test_sorting_algorithm_complexity(self, performance_config):
        """
        Property: Sorting algorithm has O(n log n) time complexity.
        
        This test specifically validates the sorting phase by measuring
        performance on pre-scored memories.
        
        **Validates: Requirements 6.5**
        """
        current_time = datetime.now(timezone.utc)
        engine = RankingEngine(performance_config)
        
        # Test with different sizes
        test_sizes = [10_000, 20_000, 40_000]
        timings = []
        
        print(f"\nMeasuring sorting algorithm complexity:")
        
        for size in test_sizes:
            memories = self.generate_memories(size, current_time)
            
            # Measure time
            avg_time = self.measure_ranking_time(
                engine, memories, current_time, runs=3
            )
            
            timings.append((size, avg_time))
            print(f"  n={size:6d}: {avg_time:.4f}s")
        
        # Verify O(n log n) scaling
        for i in range(len(timings) - 1):
            n1, t1 = timings[i]
            n2, t2 = timings[i + 1]
            
            # For O(n log n), when size doubles:
            # time(2n) / time(n) ≈ 2 * log(2n) / log(n)
            size_ratio = n2 / n1
            time_ratio = t2 / t1
            expected_ratio = size_ratio * (math.log(n2) / math.log(n1))
            
            print(f"\n  Scaling from n={n1} to n={n2}:")
            print(f"    Time ratio: {time_ratio:.2f}x")
            print(f"    Expected for O(n log n): {expected_ratio:.2f}x")
            
            # Verify time ratio matches O(n log n) expectation
            # Allow 50% variance for constant factors and system noise
            lower_bound = expected_ratio * 0.5
            upper_bound = expected_ratio * 2.0
            
            assert lower_bound <= time_ratio <= upper_bound, (
                f"Sorting complexity doesn't match O(n log n): "
                f"time_ratio={time_ratio:.2f}x is outside expected range "
                f"[{lower_bound:.2f}x, {upper_bound:.2f}x]"
            )
