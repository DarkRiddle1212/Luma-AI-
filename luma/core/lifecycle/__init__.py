"""
Memory Lifecycle Management Module.

This module provides comprehensive memory lifecycle management capabilities
including memory decay, pruning, and deduplication operations. The system
maintains memory store health through time-based importance reduction,
threshold-based removal, and similarity-based merging.

Key Components:
- Schemas: Data models and configuration classes for lifecycle operations
- MemoryDecay: Time-based importance score reduction
- MemoryPruner: Threshold-based memory removal
- MemoryDeduplicator: Similarity-based duplicate detection and merging
- LifecycleManager: Orchestrator for all lifecycle operations

Example:
    >>> from luma.core.lifecycle import (
    ...     LifecycleManager, DecayConfig, PruningConfig, DeduplicationConfig,
    ...     DecayFunctionType, PruningStrategy, SimilarityMetric
    ... )
    >>> 
    >>> # Configure lifecycle operations
    >>> decay_config = DecayConfig(
    ...     decay_function_type=DecayFunctionType.EXPONENTIAL,
    ...     decay_rate=0.1
    ... )
    >>> 
    >>> pruning_config = PruningConfig(
    ...     strategy=PruningStrategy.THRESHOLD,
    ...     threshold=0.3,
    ...     min_importance_protected=0.8
    ... )
    >>> 
    >>> dedup_config = DeduplicationConfig(
    ...     similarity_metric=SimilarityMetric.COSINE,
    ...     similarity_threshold=0.9,
    ...     batch_size=1000
    ... )
"""

# Import all schemas and data models
from .schemas import (
    # Result schemas
    MemoryDecayResult,
    PrunedMemory,
    PruningResult,
    MergeDetail,
    DeduplicationResult,
    LifecycleReport,
    
    # Configuration schemas
    DecayConfig,
    PruningConfig,
    DeduplicationConfig,
    
    # Enums
    DecayFunctionType,
    PruningStrategy,
    SimilarityMetric,
)

# Import lifecycle components
from .memory_decay import MemoryDecay
from .memory_pruner import MemoryPruner
from .memory_deduplicator import MemoryDeduplicator
from .lifecycle_manager import LifecycleManager

# Export all public symbols
__all__ = [
    # Result schemas
    "MemoryDecayResult",
    "PrunedMemory", 
    "PruningResult",
    "MergeDetail",
    "DeduplicationResult",
    "LifecycleReport",
    
    # Configuration schemas
    "DecayConfig",
    "PruningConfig", 
    "DeduplicationConfig",
    
    # Enums
    "DecayFunctionType",
    "PruningStrategy",
    "SimilarityMetric",
    
    # Lifecycle components
    "MemoryDecay",
    "MemoryPruner",
    "MemoryDeduplicator",
    "LifecycleManager",
]
