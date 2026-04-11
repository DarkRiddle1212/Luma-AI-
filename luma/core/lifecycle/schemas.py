"""
Lifecycle Schemas Module.

This module provides structured data models for memory lifecycle operations
including decay, pruning, and deduplication. All schemas use Python dataclasses
with validation to ensure type safety and data integrity.

The schemas support the comprehensive memory lifecycle management system with
proper validation of numeric fields, score ranges, and timestamp formats.

Example:
    >>> from luma.core.lifecycle.schemas import MemoryDecayResult, LifecycleReport
    >>> 
    >>> decay_result = MemoryDecayResult(
    ...     memories_processed=1000,
    ...     memories_updated=800,
    ...     average_decay_applied=0.15,
    ...     execution_time_ms=250.5
    ... )
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from enum import Enum


class DecayFunctionType(Enum):
    """
    Types of decay functions for memory importance reduction.
    
    Attributes:
        EXPONENTIAL: Exponential decay using e^(-decay_rate * age_days)
        LINEAR: Linear decay using max(0, importance - decay_rate * age_days)
        STEP: Step decay using discrete intervals
    """
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


class PruningStrategy(Enum):
    """
    Strategies for memory pruning operations.
    
    Attributes:
        THRESHOLD: Remove memories below fixed importance threshold
        PERCENTILE: Remove bottom N% of memories by importance
        CAPACITY: Remove lowest-importance memories when count exceeds limit
    """
    THRESHOLD = "threshold"
    PERCENTILE = "percentile"
    CAPACITY = "capacity"


class SimilarityMetric(Enum):
    """
    Similarity metrics for memory deduplication.
    
    Attributes:
        COSINE: Cosine similarity for embedding-based comparison
        JACCARD: Jaccard similarity for token-based text comparison
        LEVENSHTEIN: Levenshtein distance for character-level text comparison
    """
    COSINE = "cosine"
    JACCARD = "jaccard"
    LEVENSHTEIN = "levenshtein"


@dataclass
class MemoryDecayResult:
    """
    Result of memory decay operation.
    
    Contains statistics about the decay operation including counts of memories
    processed and updated, average decay applied, and execution time.
    
    Attributes:
        memories_processed: Number of memories processed during decay operation.
                          Must be non-negative.
        memories_updated: Number of memories that had their importance updated.
                        Must be non-negative and <= memories_processed.
        average_decay_applied: Average decay factor applied across all memories.
                             Must be in range [0, 1].
        execution_time_ms: Time taken to execute decay operation in milliseconds.
                          Must be non-negative.
    
    Validation Rules:
        - memories_processed >= 0
        - memories_updated >= 0
        - memories_updated <= memories_processed
        - 0.0 <= average_decay_applied <= 1.0
        - execution_time_ms >= 0.0
    
    Example:
        >>> result = MemoryDecayResult(
        ...     memories_processed=1000,
        ...     memories_updated=800,
        ...     average_decay_applied=0.15,
        ...     execution_time_ms=250.5
        ... )
    """
    memories_processed: int
    memories_updated: int
    average_decay_applied: float
    execution_time_ms: float
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if self.memories_processed < 0:
            raise ValueError(f"memories_processed must be non-negative, got {self.memories_processed}")
        
        if self.memories_updated < 0:
            raise ValueError(f"memories_updated must be non-negative, got {self.memories_updated}")
        
        if self.memories_updated > self.memories_processed:
            raise ValueError(f"memories_updated ({self.memories_updated}) cannot exceed memories_processed ({self.memories_processed})")
        
        if not 0.0 <= self.average_decay_applied <= 1.0:
            raise ValueError(f"average_decay_applied must be in [0, 1], got {self.average_decay_applied}")
        
        if self.execution_time_ms < 0.0:
            raise ValueError(f"execution_time_ms must be non-negative, got {self.execution_time_ms}")


@dataclass
class PrunedMemory:
    """
    Record of a pruned memory.
    
    Contains details about a memory that was deleted during pruning operations
    including its ID, scores, and deletion timestamp.
    
    Attributes:
        memory_id: Unique identifier of the deleted memory.
        importance_score: Original importance score of the memory.
                        Must be in range [0, 1].
        final_score: Final computed score of the memory.
                   Must be in range [0, 1].
        deletion_timestamp: ISO 8601 timestamp when memory was deleted.
        reason: Reason for deletion (e.g., "age", "score", "capacity").
    
    Validation Rules:
        - 0.0 <= importance_score <= 1.0
        - 0.0 <= final_score <= 1.0
        - deletion_timestamp must be valid ISO 8601 format
        - memory_id must be non-empty string
        - reason must be non-empty string
    
    Example:
        >>> pruned = PrunedMemory(
        ...     memory_id="mem_123",
        ...     importance_score=0.2,
        ...     final_score=0.15,
        ...     deletion_timestamp=datetime.now(),
        ...     reason="score"
        ... )
    """
    memory_id: str
    importance_score: float
    final_score: float
    deletion_timestamp: datetime
    reason: str
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if not self.memory_id or not isinstance(self.memory_id, str):
            raise ValueError("memory_id must be a non-empty string")
        
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError(f"importance_score must be in [0, 1], got {self.importance_score}")
        
        if not 0.0 <= self.final_score <= 1.0:
            raise ValueError(f"final_score must be in [0, 1], got {self.final_score}")
        
        if not isinstance(self.deletion_timestamp, datetime):
            raise ValueError("deletion_timestamp must be a datetime object")
        
        if not self.reason or not isinstance(self.reason, str):
            raise ValueError("reason must be a non-empty string")


@dataclass
class PruningResult:
    """
    Result of memory pruning operation.
    
    Contains statistics about the pruning operation including counts of memories
    deleted, failures, and detailed records of pruned memories.
    
    Attributes:
        memories_deleted: Number of memories successfully deleted.
                        Must be non-negative.
        deletion_failures: Number of deletion operations that failed.
                         Must be non-negative.
        pruned_memories: List of detailed records for each pruned memory.
        execution_time_ms: Time taken to execute pruning operation in milliseconds.
                          Must be non-negative.
    
    Validation Rules:
        - memories_deleted >= 0
        - deletion_failures >= 0
        - len(pruned_memories) == memories_deleted
        - execution_time_ms >= 0.0
    
    Example:
        >>> result = PruningResult(
        ...     memories_deleted=50,
        ...     deletion_failures=2,
        ...     pruned_memories=[...],
        ...     execution_time_ms=150.0
        ... )
    """
    memories_deleted: int
    deletion_failures: int
    pruned_memories: List[PrunedMemory]
    execution_time_ms: float
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if self.memories_deleted < 0:
            raise ValueError(f"memories_deleted must be non-negative, got {self.memories_deleted}")
        
        if self.deletion_failures < 0:
            raise ValueError(f"deletion_failures must be non-negative, got {self.deletion_failures}")
        
        if len(self.pruned_memories) != self.memories_deleted:
            raise ValueError(f"pruned_memories length ({len(self.pruned_memories)}) must equal memories_deleted ({self.memories_deleted})")
        
        if self.execution_time_ms < 0.0:
            raise ValueError(f"execution_time_ms must be non-negative, got {self.execution_time_ms}")


@dataclass
class MergeDetail:
    """
    Details of a memory merge operation.
    
    Contains information about merging duplicate memories including which
    memory was kept, which was deleted, and merge metadata.
    
    Attributes:
        kept_memory_id: ID of the memory that was retained.
        deleted_memory_id: ID of the memory that was deleted.
        similarity_score: Similarity score between the memories.
                        Must be in range [0, 1].
        merged_tags: List of tags from both memories combined.
        merge_timestamp: ISO 8601 timestamp when merge occurred.
    
    Validation Rules:
        - kept_memory_id must be non-empty string
        - deleted_memory_id must be non-empty string
        - kept_memory_id != deleted_memory_id
        - 0.0 <= similarity_score <= 1.0
        - merge_timestamp must be valid datetime
    
    Example:
        >>> detail = MergeDetail(
        ...     kept_memory_id="mem_123",
        ...     deleted_memory_id="mem_456",
        ...     similarity_score=0.95,
        ...     merged_tags=["tag1", "tag2"],
        ...     merge_timestamp=datetime.now()
        ... )
    """
    kept_memory_id: str
    deleted_memory_id: str
    similarity_score: float
    merged_tags: List[str]
    merge_timestamp: datetime
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if not self.kept_memory_id or not isinstance(self.kept_memory_id, str):
            raise ValueError("kept_memory_id must be a non-empty string")
        
        if not self.deleted_memory_id or not isinstance(self.deleted_memory_id, str):
            raise ValueError("deleted_memory_id must be a non-empty string")
        
        if self.kept_memory_id == self.deleted_memory_id:
            raise ValueError("kept_memory_id and deleted_memory_id must be different")
        
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(f"similarity_score must be in [0, 1], got {self.similarity_score}")
        
        if not isinstance(self.merge_timestamp, datetime):
            raise ValueError("merge_timestamp must be a datetime object")
        
        if not isinstance(self.merged_tags, list):
            raise ValueError("merged_tags must be a list")


@dataclass
class DeduplicationResult:
    """
    Result of memory deduplication operation.
    
    Contains statistics about the deduplication operation including counts of
    duplicate pairs found, memories merged, and detailed merge information.
    
    Attributes:
        duplicate_pairs_found: Number of duplicate pairs detected.
                             Must be non-negative.
        memories_merged: Number of memories successfully merged/deleted.
                       Must be non-negative and <= duplicate_pairs_found.
        merge_details: List of detailed records for each merge operation.
        checkpoint_timestamp: Optional timestamp for incremental processing.
        execution_time_ms: Time taken to execute deduplication in milliseconds.
                          Must be non-negative.
    
    Validation Rules:
        - duplicate_pairs_found >= 0
        - memories_merged >= 0
        - memories_merged <= duplicate_pairs_found
        - len(merge_details) == memories_merged
        - execution_time_ms >= 0.0
    
    Example:
        >>> result = DeduplicationResult(
        ...     duplicate_pairs_found=25,
        ...     memories_merged=20,
        ...     merge_details=[...],
        ...     checkpoint_timestamp=datetime.now(),
        ...     execution_time_ms=500.0
        ... )
    """
    duplicate_pairs_found: int
    memories_merged: int
    merge_details: List[MergeDetail]
    checkpoint_timestamp: Optional[datetime]
    execution_time_ms: float
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if self.duplicate_pairs_found < 0:
            raise ValueError(f"duplicate_pairs_found must be non-negative, got {self.duplicate_pairs_found}")
        
        if self.memories_merged < 0:
            raise ValueError(f"memories_merged must be non-negative, got {self.memories_merged}")
        
        if self.memories_merged > self.duplicate_pairs_found:
            raise ValueError(f"memories_merged ({self.memories_merged}) cannot exceed duplicate_pairs_found ({self.duplicate_pairs_found})")
        
        if len(self.merge_details) != self.memories_merged:
            raise ValueError(f"merge_details length ({len(self.merge_details)}) must equal memories_merged ({self.memories_merged})")
        
        if self.execution_time_ms < 0.0:
            raise ValueError(f"execution_time_ms must be non-negative, got {self.execution_time_ms}")


@dataclass
class LifecycleReport:
    """
    Comprehensive report of memory lifecycle maintenance cycle.
    
    Contains results from all lifecycle operations (decay, pruning, deduplication)
    along with overall execution statistics and metadata.
    
    Attributes:
        decay_result: Result of memory decay operation.
        pruning_result: Result of memory pruning operation.
        deduplication_result: Result of memory deduplication operation.
        total_execution_time_ms: Total time for entire maintenance cycle in milliseconds.
                               Must be non-negative.
        maintenance_timestamp: ISO 8601 timestamp when maintenance started.
        dry_run: Whether this was a dry run (no actual changes made).
    
    Validation Rules:
        - total_execution_time_ms >= 0.0
        - maintenance_timestamp must be valid datetime
        - dry_run must be boolean
    
    Example:
        >>> report = LifecycleReport(
        ...     decay_result=decay_result,
        ...     pruning_result=pruning_result,
        ...     deduplication_result=dedup_result,
        ...     total_execution_time_ms=1250.5,
        ...     maintenance_timestamp=datetime.now(),
        ...     dry_run=False
        ... )
    """
    decay_result: MemoryDecayResult
    pruning_result: PruningResult
    deduplication_result: DeduplicationResult
    total_execution_time_ms: float
    maintenance_timestamp: datetime
    dry_run: bool
    
    def __post_init__(self):
        """Validate all fields after initialization."""
        if not isinstance(self.decay_result, MemoryDecayResult):
            raise ValueError("decay_result must be a MemoryDecayResult instance")
        
        if not isinstance(self.pruning_result, PruningResult):
            raise ValueError("pruning_result must be a PruningResult instance")
        
        if not isinstance(self.deduplication_result, DeduplicationResult):
            raise ValueError("deduplication_result must be a DeduplicationResult instance")
        
        if self.total_execution_time_ms < 0.0:
            raise ValueError(f"total_execution_time_ms must be non-negative, got {self.total_execution_time_ms}")
        
        if not isinstance(self.maintenance_timestamp, datetime):
            raise ValueError("maintenance_timestamp must be a datetime object")
        
        if not isinstance(self.dry_run, bool):
            raise ValueError("dry_run must be a boolean")


@dataclass
class DecayConfig:
    """
    Configuration for memory decay operations.
    
    Defines how memory importance scores should decay over time using
    configurable decay functions and parameters.
    
    Attributes:
        decay_function_type: Type of decay function to use.
        decay_rate: Rate parameter for decay function. Must be positive.
        step_interval_days: Days between step decay applications (for STEP function).
                          Must be positive if decay_function_type is STEP.
        step_percentage: Percentage reduction per step (for STEP function).
                       Must be in range (0, 1) if decay_function_type is STEP.
    
    Validation Rules:
        - decay_rate > 0.0
        - If decay_function_type is STEP:
          - step_interval_days must be provided and > 0
          - step_percentage must be provided and in (0, 1)
        - If decay_function_type is not STEP:
          - step_interval_days and step_percentage should be None
    
    Example:
        >>> # Exponential decay
        >>> config = DecayConfig(
        ...     decay_function_type=DecayFunctionType.EXPONENTIAL,
        ...     decay_rate=0.1
        ... )
        
        >>> # Step decay
        >>> config = DecayConfig(
        ...     decay_function_type=DecayFunctionType.STEP,
        ...     decay_rate=0.05,
        ...     step_interval_days=7,
        ...     step_percentage=0.1
        ... )
    """
    decay_function_type: DecayFunctionType
    decay_rate: float
    step_interval_days: Optional[int] = None
    step_percentage: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.decay_rate <= 0.0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")
        
        if self.decay_function_type == DecayFunctionType.STEP:
            if self.step_interval_days is None or self.step_interval_days <= 0:
                raise ValueError("step_interval_days must be provided and positive for STEP decay function")
            
            if self.step_percentage is None or not 0.0 < self.step_percentage < 1.0:
                raise ValueError("step_percentage must be provided and in range (0, 1) for STEP decay function")
        
        else:
            if self.step_interval_days is not None:
                raise ValueError("step_interval_days should only be provided for STEP decay function")
            
            if self.step_percentage is not None:
                raise ValueError("step_percentage should only be provided for STEP decay function")


@dataclass
class PruningConfig:
    """
    Configuration for memory pruning operations.
    
    Defines strategy and parameters for removing memories based on various
    criteria such as importance thresholds, percentiles, or capacity limits.
    
    Attributes:
        strategy: Pruning strategy to use.
        threshold: Importance threshold for THRESHOLD strategy.
                 Must be in range [0, 1] if strategy is THRESHOLD.
        percentile: Percentile for PERCENTILE strategy.
                  Must be in range (0, 100) if strategy is PERCENTILE.
        capacity_limit: Maximum memory count for CAPACITY strategy.
                      Must be positive if strategy is CAPACITY.
        min_importance_protected: Importance threshold for protection.
                                Must be in range [0, 1]. Memories with
                                importance >= this value are never deleted.
    
    Validation Rules:
        - If strategy is THRESHOLD:
          - threshold must be provided and in [0, 1]
          - percentile and capacity_limit should be None
        - If strategy is PERCENTILE:
          - percentile must be provided and in (0, 100)
          - threshold and capacity_limit should be None
        - If strategy is CAPACITY:
          - capacity_limit must be provided and > 0
          - threshold and percentile should be None
        - 0.0 <= min_importance_protected <= 1.0
    
    Example:
        >>> # Threshold-based pruning
        >>> config = PruningConfig(
        ...     strategy=PruningStrategy.THRESHOLD,
        ...     threshold=0.3,
        ...     min_importance_protected=0.8
        ... )
        
        >>> # Percentile-based pruning
        >>> config = PruningConfig(
        ...     strategy=PruningStrategy.PERCENTILE,
        ...     percentile=10.0,
        ...     min_importance_protected=0.8
        ... )
    """
    strategy: PruningStrategy
    threshold: Optional[float] = None
    percentile: Optional[float] = None
    capacity_limit: Optional[int] = None
    min_importance_protected: float = 0.8
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.min_importance_protected <= 1.0:
            raise ValueError(f"min_importance_protected must be in [0, 1], got {self.min_importance_protected}")
        
        if self.strategy == PruningStrategy.THRESHOLD:
            if self.threshold is None or not 0.0 <= self.threshold <= 1.0:
                raise ValueError("threshold must be provided and in [0, 1] for THRESHOLD strategy")
            
            if self.percentile is not None:
                raise ValueError("percentile should not be provided for THRESHOLD strategy")
            
            if self.capacity_limit is not None:
                raise ValueError("capacity_limit should not be provided for THRESHOLD strategy")
        
        elif self.strategy == PruningStrategy.PERCENTILE:
            if self.percentile is None or not 0.0 < self.percentile < 100.0:
                raise ValueError("percentile must be provided and in (0, 100) for PERCENTILE strategy")
            
            if self.threshold is not None:
                raise ValueError("threshold should not be provided for PERCENTILE strategy")
            
            if self.capacity_limit is not None:
                raise ValueError("capacity_limit should not be provided for PERCENTILE strategy")
        
        elif self.strategy == PruningStrategy.CAPACITY:
            if self.capacity_limit is None or self.capacity_limit <= 0:
                raise ValueError("capacity_limit must be provided and positive for CAPACITY strategy")
            
            if self.threshold is not None:
                raise ValueError("threshold should not be provided for CAPACITY strategy")
            
            if self.percentile is not None:
                raise ValueError("percentile should not be provided for CAPACITY strategy")


@dataclass
class DeduplicationConfig:
    """
    Configuration for memory deduplication operations.
    
    Defines similarity metrics, thresholds, and processing parameters for
    detecting and merging duplicate memories.
    
    Attributes:
        similarity_metric: Similarity metric to use for comparison.
        similarity_threshold: Minimum similarity score to consider duplicates.
                            Must be in range [0, 1].
        batch_size: Number of memories to process per deduplication cycle.
                  Must be positive.
        checkpoint_enabled: Whether to enable incremental processing with checkpoints.
    
    Validation Rules:
        - 0.0 <= similarity_threshold <= 1.0
        - batch_size > 0
        - checkpoint_enabled must be boolean
    
    Example:
        >>> config = DeduplicationConfig(
        ...     similarity_metric=SimilarityMetric.COSINE,
        ...     similarity_threshold=0.9,
        ...     batch_size=1000,
        ...     checkpoint_enabled=True
        ... )
    """
    similarity_metric: SimilarityMetric
    similarity_threshold: float
    batch_size: int
    checkpoint_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if not isinstance(self.checkpoint_enabled, bool):
            raise ValueError("checkpoint_enabled must be a boolean")