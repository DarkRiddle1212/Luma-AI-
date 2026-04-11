"""
Retrieval Ranking Engine Usage Examples

This script demonstrates various usage patterns for the Retrieval Ranking Engine,
including different configuration strategies and use cases.

Run from project root: python examples/ranking_engine_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone, timedelta
from luma.core.ranking_engine import (
    RankingEngine, 
    RankingConfig, 
    RankedMemory,
    memory_entry_to_ranked_memory
)
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


def create_sample_memory(
    memory_id: str,
    content: str,
    similarity_score: float,
    importance_score: float = 0.0,
    namespace: str = "conversation",
    hours_ago: int = 0
) -> RankedMemory:
    """Helper to create sample memory entries."""
    timestamp = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    
    return RankedMemory(
        memory_id=memory_id,
        timestamp=timestamp,
        content=content,
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance_score,
        recency_score=0.0,  # Will be computed
        final_score=0.0,    # Will be computed
        memory_entry=None   # Would be actual Memory object in production
    )


def example_1_balanced_configuration():
    """
    Example 1: Balanced Configuration
    
    Use case: General-purpose retrieval that balances similarity, recency, and importance.
    Good for: Conversational AI, knowledge retrieval, general Q&A systems.
    """
    print("=" * 80)
    print("Example 1: Balanced Configuration")
    print("=" * 80)
    
    # Configure with balanced weights
    config = RankingConfig(
        alpha=0.5,  # 50% similarity weight
        beta=0.3,   # 30% recency weight
        gamma=0.2,  # 20% importance weight
        decay_constant=0.0001,  # Slow decay (~2.7 hours half-life)
        similarity_threshold=0.3,
        score_threshold=0.2,
        namespace="conversation"
    )
    
    # Create sample memories
    memories = [
        create_sample_memory("mem1", "User asked about Python", 0.85, 0.7, hours_ago=1),
        create_sample_memory("mem2", "User asked about JavaScript", 0.60, 0.5, hours_ago=0),
        create_sample_memory("mem3", "User asked about databases", 0.90, 0.8, hours_ago=5),
        create_sample_memory("mem4", "User asked about APIs", 0.40, 0.3, hours_ago=2),
    ]
    
    # Rank memories
    engine = RankingEngine(config)
    ranked = engine.rank(memories)
    
    # Display results
    print(f"\nConfiguration:")
    print(f"  Weights: α={config.alpha}, β={config.beta}, γ={config.gamma}")
    print(f"  Decay constant: {config.decay_constant}")
    print(f"  Thresholds: similarity={config.similarity_threshold}, score={config.score_threshold}")
    
    print(f"\nRanked Results ({len(ranked)} memories):")
    for i, mem in enumerate(ranked, 1):
        print(f"  {i}. {mem.memory_id}: {mem.content}")
        print(f"     Similarity: {mem.similarity_score:.3f}, Recency: {mem.recency_score:.3f}, "
              f"Importance: {mem.importance_score:.3f}")
        print(f"     Final Score: {mem.final_score:.3f}")
    print()


def example_2_similarity_focused():
    """
    Example 2: Similarity-Focused Configuration
    
    Use case: Semantic search where relevance is paramount.
    Good for: Document retrieval, FAQ systems, knowledge bases.
    """
    print("=" * 80)
    print("Example 2: Similarity-Focused Configuration (No Importance)")
    print("=" * 80)
    
    # Configure with high similarity weight, no importance
    config = RankingConfig(
        alpha=0.7,  # 70% similarity weight
        beta=0.3,   # 30% recency weight
        gamma=0.0,  # No importance (renormalized)
        decay_constant=0.001,  # Faster decay (~17 min half-life)
        similarity_threshold=0.5,
        score_threshold=0.3,
        namespace=None  # No namespace filtering
    )
    
    # Create sample memories
    memories = [
        create_sample_memory("mem1", "Python tutorial", 0.95, namespace="docs", hours_ago=10),
        create_sample_memory("mem2", "Python basics", 0.80, namespace="docs", hours_ago=1),
        create_sample_memory("mem3", "Java tutorial", 0.40, namespace="docs", hours_ago=0),
        create_sample_memory("mem4", "Python advanced", 0.85, namespace="docs", hours_ago=5),
    ]
    
    # Rank memories
    engine = RankingEngine(config)
    ranked = engine.rank(memories)
    
    # Display results
    print(f"\nConfiguration:")
    print(f"  Weights: α={config.alpha}, β={config.beta}, γ={config.gamma}")
    print(f"  Note: gamma=0 means alpha + beta = 1.0 (renormalized)")
    print(f"  Decay constant: {config.decay_constant}")
    
    print(f"\nRanked Results ({len(ranked)} memories):")
    for i, mem in enumerate(ranked, 1):
        print(f"  {i}. {mem.memory_id}: {mem.content}")
        print(f"     Similarity: {mem.similarity_score:.3f}, Recency: {mem.recency_score:.3f}")
        print(f"     Final Score: {mem.final_score:.3f}")
    print()


def example_3_recency_focused():
    """
    Example 3: Recency-Focused Configuration
    
    Use case: Recent conversation context, real-time systems.
    Good for: Chatbots, live support, time-sensitive applications.
    """
    print("=" * 80)
    print("Example 3: Recency-Focused Configuration")
    print("=" * 80)
    
    # Configure with high recency weight
    config = RankingConfig(
        alpha=0.3,  # 30% similarity weight
        beta=0.6,   # 60% recency weight
        gamma=0.1,  # 10% importance weight
        decay_constant=0.01,   # Very fast decay (~1.7 min half-life)
        similarity_threshold=0.2,
        score_threshold=0.1,
        namespace="conversation"
    )
    
    # Create sample memories with varying recency
    memories = [
        create_sample_memory("mem1", "Recent question", 0.60, 0.5, hours_ago=0),
        create_sample_memory("mem2", "Old but relevant", 0.90, 0.8, hours_ago=24),
        create_sample_memory("mem3", "Medium age", 0.70, 0.6, hours_ago=2),
        create_sample_memory("mem4", "Very recent", 0.50, 0.4, hours_ago=0),
    ]
    
    # Rank memories
    engine = RankingEngine(config)
    ranked = engine.rank(memories)
    
    # Display results
    print(f"\nConfiguration:")
    print(f"  Weights: α={config.alpha}, β={config.beta}, γ={config.gamma}")
    print(f"  Decay constant: {config.decay_constant} (fast decay)")
    
    print(f"\nRanked Results ({len(ranked)} memories):")
    for i, mem in enumerate(ranked, 1):
        age_hours = (datetime.now(timezone.utc) - mem.timestamp).total_seconds() / 3600
        print(f"  {i}. {mem.memory_id}: {mem.content}")
        print(f"     Age: {age_hours:.1f} hours, Similarity: {mem.similarity_score:.3f}, "
              f"Recency: {mem.recency_score:.3f}")
        print(f"     Final Score: {mem.final_score:.3f}")
    print()


def example_4_namespace_filtering():
    """
    Example 4: Namespace Filtering
    
    Use case: Isolating memories by category or context.
    Good for: Multi-tenant systems, context-specific retrieval.
    """
    print("=" * 80)
    print("Example 4: Namespace Filtering")
    print("=" * 80)
    
    # Configure with namespace filter
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.0,
        score_threshold=0.0,
        namespace="work"  # Only retrieve "work" namespace
    )
    
    # Create memories in different namespaces
    memories = [
        create_sample_memory("mem1", "Work project update", 0.80, namespace="work", hours_ago=1),
        create_sample_memory("mem2", "Personal note", 0.90, namespace="personal", hours_ago=0),
        create_sample_memory("mem3", "Work meeting notes", 0.70, namespace="work", hours_ago=2),
        create_sample_memory("mem4", "System log", 0.85, namespace="system", hours_ago=0),
    ]
    
    # Rank memories
    engine = RankingEngine(config)
    ranked = engine.rank(memories)
    
    # Display results
    print(f"\nConfiguration:")
    print(f"  Namespace filter: '{config.namespace}'")
    print(f"  Total input memories: {len(memories)}")
    
    print(f"\nRanked Results ({len(ranked)} memories):")
    for i, mem in enumerate(ranked, 1):
        print(f"  {i}. {mem.memory_id}: {mem.content} [namespace: {mem.namespace}]")
        print(f"     Final Score: {mem.final_score:.3f}")
    print()


def example_5_threshold_filtering():
    """
    Example 5: Threshold Filtering
    
    Use case: Quality control, filtering low-relevance results.
    Good for: High-precision retrieval, reducing noise.
    """
    print("=" * 80)
    print("Example 5: Threshold Filtering")
    print("=" * 80)
    
    # Configure with strict thresholds
    config = RankingConfig(
        alpha=0.5,
        beta=0.5,
        gamma=0.0,
        decay_constant=0.001,
        similarity_threshold=0.6,  # High similarity threshold
        score_threshold=0.5,       # High final score threshold
        namespace=None
    )
    
    # Create memories with varying quality
    memories = [
        create_sample_memory("mem1", "Highly relevant", 0.90, hours_ago=0),
        create_sample_memory("mem2", "Somewhat relevant", 0.50, hours_ago=0),
        create_sample_memory("mem3", "Very relevant", 0.85, hours_ago=1),
        create_sample_memory("mem4", "Not relevant", 0.30, hours_ago=0),
        create_sample_memory("mem5", "Moderately relevant", 0.65, hours_ago=2),
    ]
    
    # Rank memories
    engine = RankingEngine(config)
    ranked = engine.rank(memories)
    
    # Display results
    print(f"\nConfiguration:")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Score threshold: {config.score_threshold}")
    print(f"  Total input memories: {len(memories)}")
    
    print(f"\nRanked Results ({len(ranked)} memories passed thresholds):")
    for i, mem in enumerate(ranked, 1):
        print(f"  {i}. {mem.memory_id}: {mem.content}")
        print(f"     Similarity: {mem.similarity_score:.3f}, Final Score: {mem.final_score:.3f}")
    
    print(f"\nFiltered out: {len(memories) - len(ranked)} memories")
    print()


def example_6_memory_entry_adapter():
    """
    Example 6: Using MemoryEntry Adapter
    
    Use case: Converting MemoryEntry objects from the memory store to RankedMemory.
    Good for: Integration with existing memory storage systems.
    """
    print("=" * 80)
    print("Example 6: MemoryEntry to RankedMemory Adapter")
    print("=" * 80)
    
    # Create sample MemoryEntry objects (as would come from memory store)
    memory_entries = [
        MemoryEntry(
            id="mem_001",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            action="User asked about Python programming",
            context={"importance": 0.8, "source": "conversation"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=["python", "programming"]
        ),
        MemoryEntry(
            id="mem_002",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=3),
            action="User asked about machine learning",
            context={"importance": 0.9, "source": "conversation"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=["ml", "ai"]
        ),
        MemoryEntry(
            id="mem_003",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            action="User asked about web development",
            context={"source": "conversation"},  # No importance specified
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device_1",
            sync_status=SyncStatus.SYNCED,
            tags=["web", "development"]
        ),
    ]
    
    # Simulate vector search results with similarity scores
    similarity_scores = [0.85, 0.92, 0.70]
    
    # Convert MemoryEntry objects to RankedMemory using adapter
    print("\nConverting MemoryEntry objects to RankedMemory:")
    ranked_memories = []
    for entry, sim_score in zip(memory_entries, similarity_scores):
        ranked = memory_entry_to_ranked_memory(
            entry,
            similarity_score=sim_score,
            namespace="conversation"
        )
        ranked_memories.append(ranked)
        print(f"  - {entry.id}: similarity={sim_score:.2f}, importance={ranked.importance_score:.2f}")
    
    # Configure ranking engine
    config = RankingConfig(
        alpha=0.5,  # 50% similarity
        beta=0.3,   # 30% recency
        gamma=0.2,  # 20% importance
        decay_constant=0.001,
        similarity_threshold=0.3,
        score_threshold=0.2,
        namespace="conversation"
    )
    
    # Rank the converted memories
    engine = RankingEngine(config)
    ranked = engine.rank(ranked_memories)
    
    # Display results
    print(f"\nRanked Results ({len(ranked)} memories):")
    for i, mem in enumerate(ranked, 1):
        print(f"  {i}. {mem.memory_id}: {mem.content}")
        print(f"     Similarity: {mem.similarity_score:.3f}, Recency: {mem.recency_score:.3f}, "
              f"Importance: {mem.importance_score:.3f}")
        print(f"     Final Score: {mem.final_score:.3f}")
        print(f"     Tags: {mem.memory_entry.tags}")
    print()
    
    print("Note: The adapter automatically:")
    print("  - Extracts importance from context metadata (defaults to 0.0)")
    print("  - Clamps importance values to [0, 1] range")
    print("  - Preserves the original MemoryEntry for later use")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("RETRIEVAL RANKING ENGINE - USAGE EXAMPLES")
    print("*" * 80)
    print("\n")
    
    example_1_balanced_configuration()
    example_2_similarity_focused()
    example_3_recency_focused()
    example_4_namespace_filtering()
    example_5_threshold_filtering()
    example_6_memory_entry_adapter()
    
    print("*" * 80)
    print("Examples completed!")
    print("*" * 80)
    print()


if __name__ == "__main__":
    main()
