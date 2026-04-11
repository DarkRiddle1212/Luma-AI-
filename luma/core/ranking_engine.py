"""
Retrieval Ranking Engine: Deterministic, scalable ranking of memory entries.

This module provides a production-grade ranking engine that combines similarity,
recency, and importance scores using a configurable weighted formula. The engine
ensures deterministic ordering through comprehensive tie-breaking and supports
optional observability through metrics collection and structured logging.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


@dataclass
class RankingConfig:
    """Configuration for the ranking engine."""
    
    # Weight parameters (must sum to 1.0)
    alpha: float  # Similarity weight
    beta: float   # Recency weight
    gamma: float  # Importance weight
    
    # Decay parameter for recency scoring
    decay_constant: float  # λ > 0
    
    # Filtering thresholds
    similarity_threshold: float  # [0, 1]
    score_threshold: float       # [0, 1]
    
    # Optional namespace filtering
    namespace: Optional[str] = None
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid with descriptive message
        """
        ConfigValidator.validate(self)


@dataclass
class RankedMemory:
    """Memory entry with computed ranking scores."""
    
    # Original memory data
    memory_id: str
    timestamp: datetime
    content: str
    namespace: Optional[str]
    
    # Input scores
    similarity_score: float  # From vector search, [0, 1]
    importance_score: float  # Optional, [0, 1], defaults to 0
    
    # Computed scores
    recency_score: float     # Computed via exponential decay
    final_score: float       # Weighted combination
    
    # Original memory entry for retrieval
    memory_entry: Any  # MemoryEntry type
    
    # Additional fields for injection engine compatibility
    category: Optional[str] = None  # Category for filtering (can be same as namespace)
    metadata: Dict[str, Any] = None  # Additional metadata (embeddings, token_count, etc.)
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.metadata is None:
            self.metadata = {}
        # If category is not set, use namespace as category
        if self.category is None and self.namespace is not None:
            self.category = self.namespace


class ConfigValidator:
    """Validates ranking configuration parameters."""
    
    @staticmethod
    def validate(config: RankingConfig) -> None:
        """
        Validate all configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: With descriptive message identifying invalid parameter
        """
        # Validate weight sum (with floating point tolerance)
        weight_sum = config.alpha + config.beta + config.gamma
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Weights must sum to 1.0 (weight sum = alpha + beta + gamma), got {weight_sum} "
                f"(alpha={config.alpha}, beta={config.beta}, gamma={config.gamma})"
            )
        
        # Validate weight non-negativity
        if config.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {config.alpha}")
        if config.beta < 0:
            raise ValueError(f"beta must be non-negative, got {config.beta}")
        if config.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {config.gamma}")
        
        # Validate decay constant > 0
        if config.decay_constant <= 0:
            raise ValueError(
                f"decay_constant must be greater than 0, got {config.decay_constant}"
            )
        
        # Validate threshold ranges [0, 1]
        if not (0 <= config.similarity_threshold <= 1):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {config.similarity_threshold}"
            )
        if not (0 <= config.score_threshold <= 1):
            raise ValueError(
                f"score_threshold must be in [0, 1], got {config.score_threshold}"
            )


class NamespaceFilter:
    """Filters memories by namespace."""
    
    @staticmethod
    def filter(
        memories: List[RankedMemory],
        namespace: Optional[str]
    ) -> List[RankedMemory]:
        """
        Filter memories by namespace.
        
        Args:
            memories: Input memory collection
            namespace: Target namespace (None = no filtering)
            
        Returns:
            Filtered memory list
        """
        if namespace is None:
            return memories
        return [m for m in memories if m.namespace == namespace]


class ScoreComputer:
    """Computes recency and final scores for memories."""
    
    def __init__(self, config: RankingConfig, current_time: datetime):
        """
        Initialize score computer.
        
        Args:
            config: Ranking configuration
            current_time: Reference time for recency calculation
        """
        self.config = config
        self.current_time = current_time
    
    def compute_recency_score(self, timestamp: datetime) -> float:
        """
        Compute recency score using exponential decay.
        
        Formula: e^(-λ × age_in_seconds)
        
        Args:
            timestamp: Memory timestamp
            
        Returns:
            Recency score in [0, 1]
        """
        # Handle future timestamps (age = 0)
        age_seconds = max(0, (self.current_time - timestamp).total_seconds())
        
        # Apply exponential decay with numerical stability
        exponent = -self.config.decay_constant * age_seconds
        if exponent < -100:  # e^-100 ≈ 3.7e-44, effectively 0
            return 0.0
        
        return math.exp(exponent)
    
    def compute_final_score(
        self,
        similarity: float,
        recency: float,
        importance: float
    ) -> float:
        """
        Compute weighted final score.
        
        Formula: (α × similarity) + (β × recency) + (γ × importance)
        
        Args:
            similarity: Similarity score [0, 1]
            recency: Recency score [0, 1]
            importance: Importance score [0, 1]
            
        Returns:
            Final score [0, 1]
        """
        return (
            self.config.alpha * similarity +
            self.config.beta * recency +
            self.config.gamma * importance
        )
    
    def compute_scores(self, memory: RankedMemory) -> RankedMemory:
        """
        Compute all scores for a memory.
        
        Args:
            memory: Memory with similarity and importance scores
            
        Returns:
            Memory with recency_score and final_score populated
        """
        memory.recency_score = self.compute_recency_score(memory.timestamp)
        memory.final_score = self.compute_final_score(
            memory.similarity_score,
            memory.recency_score,
            memory.importance_score
        )
        return memory


class ThresholdFilter:
    """Filters memories by score thresholds."""
    
    def __init__(self, config: RankingConfig):
        """
        Initialize threshold filter.
        
        Args:
            config: Ranking configuration with thresholds
        """
        self.similarity_threshold = config.similarity_threshold
        self.score_threshold = config.score_threshold
    
    def filter(self, memories: List[RankedMemory]) -> List[RankedMemory]:
        """
        Filter memories by thresholds.
        
        Args:
            memories: Memories with computed scores
            
        Returns:
            Filtered memory list
        """
        return [
            m for m in memories
            if m.similarity_score >= self.similarity_threshold
            and m.final_score >= self.score_threshold
        ]


class StableSorter:
    """Implements stable deterministic sorting with tie-breaking."""
    
    @staticmethod
    def sort(memories: List[RankedMemory]) -> List[RankedMemory]:
        """
        Sort memories using deterministic tie-breaking.
        
        Sort order (descending unless noted):
        1. final_score (primary, descending)
        2. similarity_score (secondary, descending)
        3. timestamp (tertiary, descending - newer first)
        4. memory_id (quaternary, ascending - lexicographical)
        
        Args:
            memories: Memories with computed scores
            
        Returns:
            Sorted memory list (new list, input unchanged)
        """
        def comparison_key(memory: RankedMemory) -> tuple:
            return (
                -memory.final_score,
                -memory.similarity_score,
                -memory.timestamp.timestamp(),
                memory.memory_id
            )
        
        return sorted(memories, key=comparison_key)


class RankingEngine:
    """
    Main ranking engine that orchestrates the ranking pipeline.
    
    Pipeline stages:
    1. Configuration validation
    2. Namespace filtering
    3. Score computation
    4. Threshold filtering
    5. Stable sorting
    """
    
    def __init__(
        self,
        config: RankingConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize ranking engine with optional observability dependencies.
        
        Args:
            config: Ranking configuration
            metrics_collector: Optional metrics collector for instrumentation
            logger: Optional structured logger for observability
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        config.validate()
        
        # Store configuration
        self.config = config
        
        # Store observability dependencies (optional)
        self.metrics_collector = metrics_collector
        self.logger = logger
        
        # Initialize components
        self.namespace_filter = NamespaceFilter()
        self.threshold_filter = ThresholdFilter(config)
        self.stable_sorter = StableSorter()
    
    def rank(
        self,
        memories: List[RankedMemory],
        current_time: Optional[datetime] = None
    ) -> List[RankedMemory]:
        """
        Rank memories using the configured algorithm.
        
        Args:
            memories: Input memories with similarity scores
            current_time: Reference time (defaults to now)
            
        Returns:
            Ranked and filtered memory list
        """
        # Start timing measurement
        import time
        start_time = time.perf_counter()
        
        try:
            # Handle empty input
            if not memories:
                return []
            
            # Default current_time to now
            if current_time is None:
                current_time = datetime.now(timezone.utc)
            
            # 1. Namespace filtering
            filtered = self.namespace_filter.filter(memories, self.config.namespace)
            
            # 2. Score computation
            score_computer = ScoreComputer(self.config, current_time)
            scored = [score_computer.compute_scores(m) for m in filtered]
            
            # 3. Threshold filtering
            thresholded = self.threshold_filter.filter(scored)
            
            # 4. Stable sorting
            ranked = self.stable_sorter.sort(thresholded)
            
            return ranked
        finally:
            # Record timing measurement
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Record metrics if collector is available
            if self.metrics_collector is not None:
                self.metrics_collector.record_duration('ranking_latency_ms', duration_ms)
            
            # Log ranking event if logger is available
            if self.logger is not None:
                self.logger.log('ranking_completed', {
                    'input_count': len(memories),
                    'output_count': len(ranked) if 'ranked' in locals() else 0,
                    'duration_ms': duration_ms
                })


def memory_entry_to_ranked_memory(
    entry,
    similarity_score: float,
    namespace: Optional[str] = None
) -> RankedMemory:
    """
    Convert a MemoryEntry to a RankedMemory instance.

    This adapter function bridges the memory storage layer and the ranking engine
    by converting MemoryEntry objects into RankedMemory objects suitable for ranking.

    Args:
        entry: MemoryEntry instance from the storage layer
        similarity_score: Similarity score for this memory (0.0 to 1.0)
        namespace: Optional namespace for filtering

    Returns:
        RankedMemory instance ready for ranking
    """
    # Extract importance from context, with validation and clamping
    importance = 0.0
    if hasattr(entry, 'context') and isinstance(entry.context, dict):
        raw_importance = entry.context.get('importance', 0.0)
        try:
            importance = float(raw_importance)
            # Clamp to valid range [0.0, 1.0]
            importance = max(0.0, min(1.0, importance))
        except (TypeError, ValueError):
            importance = 0.0

    return RankedMemory(
        memory_id=entry.id,
        timestamp=entry.timestamp,
        content=entry.action,
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance,
        recency_score=0.0,  # Will be computed by RankingEngine
        final_score=0.0,  # Will be computed by RankingEngine
        memory_entry=entry
    )



def memory_entry_to_ranked_memory(
    entry,
    similarity_score: float,
    namespace: Optional[str] = None
) -> RankedMemory:
    """
    Convert a MemoryEntry to a RankedMemory instance.
    
    This adapter function bridges the memory storage layer and the ranking engine
    by converting MemoryEntry objects into RankedMemory objects suitable for ranking.
    
    Args:
        entry: MemoryEntry instance from the storage layer
        similarity_score: Similarity score for this memory (0.0 to 1.0)
        namespace: Optional namespace for filtering
        
    Returns:
        RankedMemory instance ready for ranking
    """
    # Extract importance from context, with validation and clamping
    importance = 0.0
    if hasattr(entry, 'context') and isinstance(entry.context, dict):
        raw_importance = entry.context.get('importance', 0.0)
        try:
            importance = float(raw_importance)
            # Clamp to valid range [0.0, 1.0]
            importance = max(0.0, min(1.0, importance))
        except (TypeError, ValueError):
            importance = 0.0
    
    return RankedMemory(
        memory_id=entry.id,
        timestamp=entry.timestamp,
        content=entry.action,
        namespace=namespace,
        similarity_score=similarity_score,
        importance_score=importance,
        recency_score=0.0,  # Will be computed by RankingEngine
        final_score=0.0,  # Will be computed by RankingEngine
        memory_entry=entry
    )
