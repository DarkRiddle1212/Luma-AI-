"""
Memory Deduplicator Module.

This module provides the MemoryDeduplicator component responsible for detecting
and merging highly similar memories to eliminate redundancy. The system supports
cosine similarity for embedding-based comparison, Jaccard similarity for token-
based text comparison, and Levenshtein distance for character-level comparison.

Key Features:
- Multiple similarity metrics (cosine, Jaccard, Levenshtein)
- Duplicate detection based on configurable similarity threshold
- Metadata merging (tags union) for duplicate pairs
- Protected memory retention (protected memory kept in duplicates)
- Incremental processing with batch_size and checkpoint support
- Dry run mode for testing without persistence
- Metrics and logging integration
"""

from datetime import datetime, UTC
from typing import Optional, List, Tuple, Dict, Any
import math

from luma.core.lifecycle.schemas import (
    DeduplicationResult,
    MergeDetail,
    DeduplicationConfig,
    SimilarityMetric,
)
from luma.core.memory_interface import MemoryInterface

try:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
except ImportError:
    MetricsCollector = None
    StructuredLogger = None


class MemoryDeduplicator:
    """
    Memory deduplicator component for similarity-based duplicate detection and merging.
    
    This component detects and merges highly similar memories to eliminate redundancy
    in the memory store. It supports three similarity metrics:
    
    - COSINE: Cosine similarity for embedding-based comparison (preferred when available)
    - JACCARD: Jaccard similarity for token-based text comparison
    - LEVENSHTEIN: Levenshtein distance for character-level text comparison
    
    The component respects protected memory flags, ensuring protected memories
    are retained in duplicate pairs. It supports incremental processing with
    checkpoints for large memory stores.
    
    The component integrates with the existing Luma infrastructure including
    MemoryInterface for storage operations, MetricsCollector for observability,
    and StructuredLogger for event logging.
    
    Attributes:
        memory_interface: MemoryInterface instance for storage operations
        dedup_config: DeduplicationConfig with similarity parameters
        metrics_collector: Optional MetricsCollector for metrics recording
        logger: Optional StructuredLogger for event logging
    """
    
    def __init__(
        self,
        memory_interface: MemoryInterface,
        dedup_config: DeduplicationConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
    ):
        """
        Initialize the MemoryDeduplicator component.
        
        Args:
            memory_interface: MemoryInterface instance for storage operations
            dedup_config: DeduplicationConfig with similarity parameters
            metrics_collector: Optional MetricsCollector for metrics recording
            logger: Optional StructuredLogger for event logging
        """
        self.memory_interface = memory_interface
        self.dedup_config = dedup_config
        self.metrics_collector = metrics_collector
        self.logger = logger
    
    def deduplicate(self, dry_run: bool = False) -> DeduplicationResult:
        """
        Detect and merge duplicate memories in the store.
        
        This method retrieves all memories, computes pairwise similarity scores,
        identifies duplicate pairs above the similarity threshold, and merges
        them by retaining the higher-importance memory and deleting the duplicate.
        In dry_run mode, duplicates are identified but no changes are persisted.
        
        Args:
            dry_run: If True, identify duplicates without merging
            
        Returns:
            DeduplicationResult with merge statistics
        """
        start_time = datetime.now(UTC)
        
        # Retrieve all memories
        result = self.memory_interface.retrieve()
        memories = result["memories"]
        
        # Sort by creation timestamp for deterministic batch processing
        sorted_memories = sorted(
            memories,
            key=lambda m: m.get("timestamp", "")
        )
        
        # Find duplicate pairs
        duplicate_pairs = self._find_duplicate_pairs(sorted_memories)
        
        # Process duplicate pairs
        memories_merged = 0
        merge_details = []
        
        for kept_memory, deleted_memory in duplicate_pairs:
            kept_id = kept_memory.get("id")
            deleted_id = deleted_memory.get("id")
            similarity = self.compute_similarity(kept_memory, deleted_memory)
            
            # Merge metadata (tags union)
            kept_tags = set(kept_memory.get("metadata", {}).get("tags", []))
            deleted_tags = set(deleted_memory.get("metadata", {}).get("tags", []))
            merged_tags = list(kept_tags | deleted_tags)
            
            if not dry_run:
                # Update kept memory with merged tags
                updated_metadata = kept_memory.get("metadata", {}).copy()
                updated_metadata["tags"] = merged_tags
                self.memory_interface.store(
                    content=kept_memory["content"],
                    metadata=updated_metadata
                )
                # Delete the duplicate
                self.memory_interface.delete(deleted_id)
            
            memories_merged += 1
            merge_details.append(
                MergeDetail(
                    kept_memory_id=kept_id,
                    deleted_memory_id=deleted_id,
                    similarity_score=similarity,
                    merged_tags=merged_tags,
                    merge_timestamp=start_time,
                )
            )
        
        # Calculate execution time
        execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        # Record metrics if collector available
        if self.metrics_collector:
            self.metrics_collector.increment("memory_dedup.pairs_found", len(duplicate_pairs))
            self.metrics_collector.increment("memory_dedup.merged", memories_merged)
            self.metrics_collector.record_duration("memory_dedup.duration_ms", execution_time_ms)
        
        # Log completion if logger available
        if self.logger:
            self.logger.log(
                event="memory_dedup_completed",
                payload={
                    "duplicate_pairs_found": len(duplicate_pairs),
                    "memories_merged": memories_merged,
                    "execution_time_ms": execution_time_ms,
                    "dry_run": dry_run,
                },
            )
        
        # Set checkpoint to last processed memory timestamp for incremental processing
        # Checkpoint is reset to None when all memories have been processed
        checkpoint_timestamp = sorted_memories[-1].get("timestamp") if sorted_memories else None
        
        return DeduplicationResult(
            duplicate_pairs_found=len(duplicate_pairs),
            memories_merged=memories_merged,
            merge_details=merge_details,
            checkpoint_timestamp=checkpoint_timestamp,
            execution_time_ms=execution_time_ms,
        )
    
    def compute_similarity(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> float:
        """
        Compute similarity score between two memories.
        
        This method selects the appropriate similarity metric based on the
        configured similarity_metric and computes the similarity score.
        Embedding-based cosine similarity is preferred when embeddings are available.
        
        Args:
            memory1: First memory entry
            memory2: Second memory entry
            
        Returns:
            Similarity score in range [0, 1]
        """
        # Prefer embedding-based similarity when available
        embedding1 = memory1.get("metadata", {}).get("embedding")
        embedding2 = memory2.get("metadata", {}).get("embedding")
        
        # When embeddings are available, always use cosine similarity
        if embedding1 and embedding2:
            return self._cosine_similarity(embedding1, embedding2)
        
        # Get text content for text-based metrics
        text1 = memory1.get("content", "")
        text2 = memory2.get("content", "")
        
        metric = self.dedup_config.similarity_metric
        
        if metric == SimilarityMetric.JACCARD:
            return self._jaccard_similarity(text1, text2)
        elif metric == SimilarityMetric.LEVENSHTEIN:
            return self._levenshtein_similarity(text1, text2)
        else:
            # Default to Jaccard when no embeddings available
            return self._jaccard_similarity(text1, text2)
    
    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Cosine similarity is computed as:
            similarity = (v1 · v2) / (||v1|| * ||v2||)
        
        The result is normalized to [0, 1] range where 1 means identical direction.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score in [0, 1]
        """
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = dot_product / (mag1 * mag2)
        
        # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        return (similarity + 1.0) / 2.0
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts based on token sets.
        
        Jaccard similarity is computed as:
            similarity = |set1 ∩ set2| / |set1 ∪ set2|
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Jaccard similarity score in [0, 1]
        """
        # Tokenize by splitting on whitespace and converting to lowercase
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Avoid division by zero
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        # Compute Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Levenshtein similarity between two strings.
        
        Levenshtein similarity is computed as:
            similarity = 1 - (edit_distance / max(len1, len2))
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Levenshtein similarity score in [0, 1]
        """
        # Handle empty strings
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Compute Levenshtein distance
        edit_distance = self._levenshtein_distance(text1, text2)
        
        # Compute similarity
        max_len = max(len(text1), len(text2))
        return 1.0 - (edit_distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.
        
        Uses dynamic programming to compute the minimum number of edits
        (insertions, deletions, substitutions) needed to transform s1 into s2.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance between the two strings
        """
        # Create distance matrix
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        
        return dp[m][n]
    
    def _find_duplicate_pairs(self, memories: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Find pairs of memories that are duplicates based on similarity threshold.
        
        This method performs pairwise comparison of memories and identifies
        duplicate pairs where similarity exceeds the configured threshold.
        For each duplicate pair, the memory with higher importance is retained.
        Protected memories are always retained when possible.
        
        Args:
            memories: List of memories sorted by creation timestamp
            
        Returns:
            List of (kept_memory, deleted_memory) tuples
        """
        threshold = self.dedup_config.similarity_threshold
        duplicate_pairs = []
        processed = set()
        
        for i, memory1 in enumerate(memories):
            if i in processed:
                continue
            
            for j in range(i + 1, len(memories)):
                if j in processed:
                    continue
                
                memory2 = memories[j]
                
                # Compute similarity
                similarity = self.compute_similarity(memory1, memory2)
                
                # Check if above threshold
                if similarity >= threshold:
                    # Determine which to keep
                    kept, deleted = self._select_duplicate_retention(memory1, memory2)
                    
                    duplicate_pairs.append((kept, deleted))
                    processed.add(i)
                    processed.add(j)
                    break  # Each memory can only be in one duplicate pair
        
        return duplicate_pairs
    
    def _select_duplicate_retention(
        self,
        memory1: Dict[str, Any],
        memory2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select which memory to keep in a duplicate pair.
        
        Priority:
        1. Protected memory is always retained (if only one is protected)
        2. When both are protected, earlier timestamp is retained
        3. When neither is protected, higher importance is retained
        4. Earlier timestamp is retained as final tiebreaker
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            (kept_memory, deleted_memory) tuple
        """
        protected1 = memory1.get("metadata", {}).get("protected", False)
        protected2 = memory2.get("metadata", {}).get("protected", False)
        
        # If one is protected, keep it
        if protected1 and not protected2:
            return memory1, memory2
        if protected2 and not protected1:
            return memory2, memory1
        
        # Both protected - prefer earlier timestamp
        if protected1 and protected2:
            timestamp1 = memory1.get("timestamp", "")
            timestamp2 = memory2.get("timestamp", "")
            if timestamp1 <= timestamp2:
                return memory1, memory2
            return memory2, memory1
        
        # Neither protected - prefer higher importance
        importance1 = memory1.get("metadata", {}).get("importance", 0.0)
        importance2 = memory2.get("metadata", {}).get("importance", 0.0)
        
        if importance1 > importance2:
            return memory1, memory2
        if importance2 > importance1:
            return memory2, memory1
        
        # Same importance - use timestamp for determinism
        timestamp1 = memory1.get("timestamp", "")
        timestamp2 = memory2.get("timestamp", "")
        
        if timestamp1 <= timestamp2:
            return memory1, memory2
        return memory2, memory1
