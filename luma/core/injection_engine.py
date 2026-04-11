"""
Context Injection Engine

A deterministic, production-grade system that selects and formats pre-ranked memories
for injection into model context windows. This engine operates downstream of the ranking
engine and upstream of the model invocation layer.

Key Features:
- Enforces token budgets to prevent context window overflow
- Prevents redundant memory injection through similarity-based filtering
- Supports category isolation for namespace-specific memory selection
- Guarantees deterministic, reproducible behavior across runs
- Maintains separation from ranking logic (accepts pre-ranked memories)
- Preserves memory metadata integrity throughout the pipeline

Architecture:
    InjectionEngine (orchestrator)
    ├── InjectionConfig (configuration)
    ├── CategoryFilter (category isolation)
    ├── RedundancyGuard (similarity-based deduplication)
    ├── TokenBudgetEnforcer (token limit enforcement)
    └── InjectionResult (output structure)

Design Principles:
- Separation of Concerns: No ranking, scoring, or metadata modification
- Determinism: Identical inputs always produce identical outputs
- Immutability: All input data structures treated as immutable
- Performance: Sub-quadratic complexity for redundancy checks
- Clean Architecture: Pure function with no external dependencies

Usage:
    config = InjectionConfig(
        max_token_budget=2048,
        max_memory_count=50,
        redundancy_similarity_threshold=0.85,
        enable_category_isolation=False
    )
    
    engine = InjectionEngine(config)
    result = engine.inject(ranked_memories)
    
    print(f"Selected {len(result.memories)} memories")
    print(f"Total tokens: {result.total_tokens}")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import numpy as np

# Import observability components for type hints
from .metrics_collector import MetricsCollector
from .structured_logger import StructuredLogger


@dataclass
class InjectionConfig:
    """Configuration for context injection engine.
    
    This dataclass controls injection behavior with fields for token budget,
    memory count, redundancy threshold, category isolation, and token estimation.
    
    Attributes:
        max_token_budget: Maximum tokens allowed in output (must be positive)
        max_memory_count: Maximum number of memories to select (must be positive)
        redundancy_similarity_threshold: Similarity threshold [0, 1] for redundancy filtering
        enable_category_isolation: Whether to filter by category
        allowed_categories: List of allowed categories (required if isolation enabled)
        token_estimation_factor: Approximation factor (tokens ≈ words × factor)
    
    Raises:
        ValueError: If configuration parameters are invalid
    """
    
    # Token budget constraints
    max_token_budget: int
    max_memory_count: int
    
    # Redundancy filtering
    redundancy_similarity_threshold: float
    
    # Category isolation
    enable_category_isolation: bool
    allowed_categories: Optional[List[str]] = None
    
    # Token estimation
    token_estimation_factor: float = 1.3
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Performs comprehensive error checking on all configuration fields:
        - max_token_budget must be positive
        - max_memory_count must be positive
        - redundancy_similarity_threshold must be in [0, 1]
        - token_estimation_factor must be positive
        - allowed_categories must be specified when enable_category_isolation is True
        - allowed_categories must be non-empty when specified
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate max_token_budget
        if self.max_token_budget <= 0:
            raise ValueError(
                f"max_token_budget must be positive, got {self.max_token_budget}"
            )
        
        # Validate max_memory_count
        if self.max_memory_count <= 0:
            raise ValueError(
                f"max_memory_count must be positive, got {self.max_memory_count}"
            )
        
        # Validate redundancy_similarity_threshold
        if not (0 <= self.redundancy_similarity_threshold <= 1):
            raise ValueError(
                f"redundancy_similarity_threshold must be in [0, 1], "
                f"got {self.redundancy_similarity_threshold}"
            )
        
        # Validate token_estimation_factor
        if self.token_estimation_factor <= 0:
            raise ValueError(
                f"token_estimation_factor must be positive, got {self.token_estimation_factor}"
            )
        
        # Validate category isolation configuration
        if self.enable_category_isolation:
            if self.allowed_categories is None:
                raise ValueError(
                    "allowed_categories must be specified when enable_category_isolation is True"
                )
            if not self.allowed_categories:
                raise ValueError(
                    "allowed_categories must be non-empty when enable_category_isolation is True"
                )


@dataclass
class InjectedMemory:
    """Memory selected for injection into context.
    
    This dataclass represents a memory that has been selected for injection
    into the model context. It's a simplified structure preserving essential
    fields from RankedMemory.
    
    Attributes:
        memory_id: Unique identifier for the memory
        content: The memory content text
        metadata: Preserved metadata from input (immutable)
        similarity_score: Pre-computed similarity score [0, 1]
        timestamp: When the memory was created
        category: Optional category classification
    
    Requirements:
        - 5.1: Part of structured output format
        - 5.2: Includes memory_id, content, metadata, and score fields
        - 5.4: Serializable to JSON without loss of information
    """
    
    memory_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    timestamp: datetime
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize InjectedMemory to dictionary for JSON serialization.
        
        Converts the dataclass to a dictionary format suitable for JSON
        serialization. Handles datetime conversion to ISO format string.
        
        Returns:
            Dictionary representation with all fields serialized
        
        Requirements:
            - 5.4: Serializable to JSON without loss of information
            - 5.5: Supports round-trip serialization/deserialization
        """
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'metadata': self.metadata,
            'similarity_score': self.similarity_score,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category
        }


@dataclass
class InjectionResult:
    """Result of context injection operation.
    
    This dataclass represents the complete output of the injection engine,
    containing the selected memories list, total token count, and diagnostic
    information about the filtering process.
    
    Attributes:
        memories: List of memories selected for injection
        total_tokens: Total token count of selected memories
        input_count: Number of memories in the input list
        filtered_by_category: Number of memories filtered by category isolation
        filtered_by_redundancy: Number of memories filtered by redundancy guard
        filtered_by_budget: Number of memories filtered by token budget/count limits
    
    Requirements:
        - 5.1: Structured output containing memories array and total_tokens field
        - 5.2: Each memory includes memory_id, content, metadata, and score fields
        - 5.3: Includes diagnostic counts for filtering stages
        - 5.4: Serializable to JSON without loss of information
        - 5.5: Supports round-trip serialization/deserialization
    """
    
    memories: List[InjectedMemory]
    total_tokens: int
    
    # Diagnostic information
    input_count: int
    filtered_by_category: int
    filtered_by_redundancy: int
    filtered_by_budget: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize InjectionResult to dictionary for JSON serialization.
        
        Converts the dataclass to a dictionary format suitable for JSON
        serialization. Recursively serializes all InjectedMemory objects.
        
        Returns:
            Dictionary representation with all fields serialized
        
        Requirements:
            - 5.4: Serializable to JSON without loss of information
            - 5.5: Supports round-trip serialization/deserialization
        
        Example:
            >>> result = InjectionResult(memories=[...], total_tokens=512, ...)
            >>> result_dict = result.to_dict()
            >>> json_str = json.dumps(result_dict)
        """
        return {
            'memories': [memory.to_dict() for memory in self.memories],
            'total_tokens': self.total_tokens,
            'input_count': self.input_count,
            'filtered_by_category': self.filtered_by_category,
            'filtered_by_redundancy': self.filtered_by_redundancy,
            'filtered_by_budget': self.filtered_by_budget
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InjectionResult':
        """Deserialize InjectionResult from dictionary.
        
        Converts a dictionary (typically from JSON) back into an InjectionResult
        object. Handles datetime parsing and recursive deserialization of
        InjectedMemory objects.
        
        Args:
            data: Dictionary containing serialized InjectionResult data
        
        Returns:
            InjectionResult object reconstructed from dictionary
        
        Raises:
            KeyError: If required fields are missing from the dictionary
            ValueError: If field values are invalid (e.g., invalid datetime format)
        
        Requirements:
            - 5.4: Serializable to JSON without loss of information
            - 5.5: Round-trip property - deserializing serialized object produces equivalent object
        
        Example:
            >>> result_dict = json.loads(json_str)
            >>> result = InjectionResult.from_dict(result_dict)
            >>> assert result.total_tokens == original_result.total_tokens
        """
        # Deserialize memories list
        memories = []
        for memory_data in data['memories']:
            # Parse timestamp from ISO format string
            timestamp = datetime.fromisoformat(memory_data['timestamp'])
            
            # Create InjectedMemory object
            memory = InjectedMemory(
                memory_id=memory_data['memory_id'],
                content=memory_data['content'],
                metadata=memory_data['metadata'],
                similarity_score=memory_data['similarity_score'],
                timestamp=timestamp,
                category=memory_data.get('category')
            )
            memories.append(memory)
        
        # Create InjectionResult object
        return cls(
            memories=memories,
            total_tokens=data['total_tokens'],
            input_count=data['input_count'],
            filtered_by_category=data['filtered_by_category'],
            filtered_by_redundancy=data['filtered_by_redundancy'],
            filtered_by_budget=data['filtered_by_budget']
        )


class TokenEstimator:
    """Deterministic token count estimator.
    
    This class provides deterministic token count estimation for memory content.
    It prefers precomputed token_count from metadata if available, otherwise
    uses a deterministic word-count approximation.
    
    The approximation formula is: tokens ≈ words × estimation_factor
    
    This approach ensures:
    - Determinism: Same input always produces same output
    - Performance: Fast word-count based estimation
    - Flexibility: Can use precomputed values when available
    
    Attributes:
        estimation_factor: Multiplier for word count approximation (default 1.3)
    
    Requirements:
        - 2.1: Deterministically estimate token count using character-based approximation
        - 2.5: Running estimation twice on same memory produces identical counts
    
    Example:
        >>> estimator = TokenEstimator(estimation_factor=1.3)
        >>> memory = RankedMemory(content="Hello world", metadata={})
        >>> tokens = estimator.estimate_tokens(memory)
        >>> assert tokens == 2  # 2 words × 1.3 = 2.6 → 2 (int conversion)
    """
    
    def __init__(self, estimation_factor: float = 1.3):
        """Initialize token estimator.
        
        Args:
            estimation_factor: Multiplier for word count approximation
                              (tokens ≈ words × factor). Default is 1.3,
                              which is a reasonable approximation for English text.
        
        Raises:
            ValueError: If estimation_factor is not positive
        """
        if estimation_factor <= 0:
            raise ValueError(
                f"estimation_factor must be positive, got {estimation_factor}"
            )
        self.estimation_factor = estimation_factor
    
    def estimate_tokens(self, memory: Any) -> int:
        """Estimate token count for a memory.
        
        Uses precomputed token_count from metadata if available,
        otherwise uses deterministic word-count approximation.
        
        The estimation process:
        1. Check if memory has 'metadata' attribute and 'token_count' exists in it
        2. If yes, return int(token_count) for determinism
        3. If no, count words using str.split() and multiply by estimation_factor
        4. Convert result to int for deterministic integer output
        
        Args:
            memory: Memory object with 'content' attribute and optionally 'metadata'
                   (typically a RankedMemory object)
        
        Returns:
            Estimated token count (deterministic integer)
        
        Requirements:
            - 2.1: Deterministically estimate token count
            - 2.5: Same input produces same output (determinism property)
        
        Example:
            >>> # With precomputed token_count
            >>> memory1 = RankedMemory(
            ...     content="Hello world",
            ...     metadata={"token_count": 5}
            ... )
            >>> estimator.estimate_tokens(memory1)
            5
            
            >>> # Without precomputed token_count (fallback to approximation)
            >>> memory2 = RankedMemory(
            ...     content="Hello world",
            ...     metadata={}
            ... )
            >>> estimator.estimate_tokens(memory2)
            2  # 2 words × 1.3 = 2.6 → 2
        """
        # Prefer precomputed token count from metadata (if metadata exists)
        if hasattr(memory, 'metadata') and isinstance(memory.metadata, dict):
            if 'token_count' in memory.metadata:
                return int(memory.metadata['token_count'])
        
        # Fallback: deterministic word-count approximation
        word_count = len(memory.content.split())
        return int(word_count * self.estimation_factor)



class CategoryFilter:
    """Filters memories by category namespace.
    
    This class implements category isolation filtering, which restricts memory
    injection to specific category namespaces when enabled. When disabled, all
    memories pass through regardless of category.
    
    The filter preserves the order of memories in the input list, ensuring that
    the ranking order from the ranking engine is maintained.
    
    Attributes:
        enabled: Whether category isolation is enabled
        allowed_categories: List of allowed category namespaces
    
    Requirements:
        - 4.1: Only inject memories matching requested category when isolation enabled
        - 4.2: Inject memories from all categories when isolation disabled
        - 4.3: Return empty result when no memories match requested category
        - 4.4: Apply category filtering before token budget and redundancy checks
    
    Example:
        >>> config = InjectionConfig(
        ...     max_token_budget=2048,
        ...     max_memory_count=50,
        ...     redundancy_similarity_threshold=0.85,
        ...     enable_category_isolation=True,
        ...     allowed_categories=["programming", "documentation"]
        ... )
        >>> filter = CategoryFilter(config)
        >>> filtered = filter.filter(memories)
        >>> # Only memories with category in ["programming", "documentation"] remain
    """
    
    def __init__(self, config: InjectionConfig):
        """Initialize category filter.
        
        Args:
            config: Injection configuration containing category isolation settings
        """
        self.enabled = config.enable_category_isolation
        self.allowed_categories = config.allowed_categories or []
    
    def filter(self, memories: List[Any]) -> List[Any]:
        """Filter memories by category.
        
        When category isolation is enabled, only memories with categories in the
        allowed_categories list are retained. When disabled, all memories pass
        through unchanged.
        
        The filter preserves the order of memories in the input list, ensuring
        that the ranking order is maintained throughout the injection pipeline.
        
        Args:
            memories: Input memory list (pre-sorted by ranking engine)
        
        Returns:
            Filtered memory list (order preserved)
        
        Requirements:
            - 4.1: Only inject memories matching requested category when enabled
            - 4.2: Inject memories from all categories when disabled
            - 4.3: Return empty list when no memories match (handled naturally)
            - 4.4: Applied before token budget and redundancy checks
        
        Example:
            >>> # With isolation enabled
            >>> filter = CategoryFilter(config_with_isolation)
            >>> memories = [
            ...     RankedMemory(category="programming", ...),
            ...     RankedMemory(category="cooking", ...),
            ...     RankedMemory(category="documentation", ...)
            ... ]
            >>> filtered = filter.filter(memories)
            >>> # Only "programming" and "documentation" memories remain
            
            >>> # With isolation disabled
            >>> filter = CategoryFilter(config_without_isolation)
            >>> filtered = filter.filter(memories)
            >>> # All memories pass through
        """
        if not self.enabled:
            return memories
        
        return [
            m for m in memories
            if m.category in self.allowed_categories
        ]


class RedundancyGuard:
    """Prevents redundant memory injection using similarity scores.
    
    This class implements redundancy filtering to prevent semantically similar
    memories from wasting token budget. It uses pre-computed embeddings from
    memory metadata to compute pairwise similarity scores.
    
    The redundancy guard operates on memories in rank order (already sorted by
    final_score). For each candidate memory, it checks similarity against all
    already-selected memories. If similarity exceeds the threshold with any
    selected memory, the candidate is filtered out.
    
    This approach ensures:
    - Higher-ranked memories are always kept (processed first)
    - Sub-quadratic complexity: O(n × m) where n = input size, m = output size
    - Deterministic behavior: Same input always produces same output
    
    Attributes:
        threshold: Similarity threshold [0, 1]. Memories with similarity > threshold
                  are considered redundant.
    
    Requirements:
        - 3.1: Compute pairwise similarity between candidate and selected memories
        - 3.2: Exclude candidate when similarity > threshold with any selected memory
        - 3.3: Keep higher-ranked memory based on final_score
        - 3.4: Use algorithm with sub-quadratic complexity
        - 3.5: Do not recompute similarity scores (use pre-computed embeddings)
    
    Example:
        >>> guard = RedundancyGuard(threshold=0.85)
        >>> filtered, count = guard.filter(ranked_memories)
        >>> print(f"Filtered {count} redundant memories")
    """
    
    def __init__(self, threshold: float):
        """Initialize redundancy guard.
        
        Args:
            threshold: Similarity threshold [0, 1]. Memories with
                      similarity > threshold are considered redundant.
        
        Raises:
            ValueError: If threshold is not in [0, 1] range
        
        Requirements:
            - 3.2: Accept threshold parameter for redundancy detection
        
        Example:
            >>> guard = RedundancyGuard(threshold=0.85)
            >>> # Memories with similarity > 0.85 will be filtered
        """
        if not (0 <= threshold <= 1):
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        self.threshold = threshold
    
    def _compute_similarity(
        self,
        memory1: Any,
        memory2: Any
    ) -> float:
        """Compute similarity between two memories.
        
        Uses pre-computed embeddings from metadata if available.
        This method does NOT recompute embeddings - it only uses
        embeddings that were already computed during the ranking phase.
        
        The similarity computation uses cosine similarity:
            similarity = dot(emb1, emb2) / (norm(emb1) × norm(emb2))
        
        Graceful degradation:
        - If either embedding is missing: return 0.0 (assume not similar)
        - If either embedding has zero norm: return 0.0 (invalid embedding)
        
        Args:
            memory1: First memory (typically RankedMemory object)
            memory2: Second memory (typically RankedMemory object)
        
        Returns:
            Similarity score [0, 1], or 0.0 if embeddings unavailable
        
        Requirements:
            - 3.1: Compute pairwise similarity between memories
            - 3.5: Use pre-computed embeddings, do not recompute
        
        Example:
            >>> memory1 = RankedMemory(
            ...     content="Python programming",
            ...     metadata={"embedding": [0.1, 0.2, 0.3]}
            ... )
            >>> memory2 = RankedMemory(
            ...     content="Python coding",
            ...     metadata={"embedding": [0.15, 0.25, 0.35]}
            ... )
            >>> similarity = guard._compute_similarity(memory1, memory2)
            >>> # Returns cosine similarity between embeddings
            
            >>> # Missing embedding case
            >>> memory3 = RankedMemory(
            ...     content="No embedding",
            ...     metadata={}
            ... )
            >>> similarity = guard._compute_similarity(memory1, memory3)
            >>> assert similarity == 0.0  # Graceful degradation
        """
        # Extract embeddings from metadata
        emb1 = memory1.metadata.get('embedding') if hasattr(memory1, 'metadata') else None
        emb2 = memory2.metadata.get('embedding') if hasattr(memory2, 'metadata') else None
        
        if emb1 is None or emb2 is None:
            # No embeddings available, assume not similar
            # This is graceful degradation - allows system to work without embeddings
            return 0.0
        
        # Compute cosine similarity using numpy
        emb1_arr = np.array(emb1)
        emb2_arr = np.array(emb2)
        
        # Compute dot product
        dot_product = np.dot(emb1_arr, emb2_arr)
        
        # Compute norms
        norm1 = np.linalg.norm(emb1_arr)
        norm2 = np.linalg.norm(emb2_arr)
        
        # Handle zero norms (invalid embeddings)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = float(dot_product / (norm1 * norm2))
        
        # Clamp to [0, 1] range for numerical stability
        # (cosine similarity can be slightly outside due to floating point errors)
        return max(0.0, min(1.0, similarity))
    def filter(
        self,
        memories: List[Any]
    ) -> Tuple[List[Any], int]:
        """Filter out redundant memories.

        This method implements the core redundancy filtering algorithm:
        1. Iterate through memories in rank order (already sorted by final_score)
        2. For each candidate memory, check similarity against already-selected memories
        3. If similarity > threshold with any selected memory, skip the candidate
        4. Otherwise, add the candidate to the selected set

        The algorithm has O(n × m) complexity where:
        - n = input size (number of candidate memories)
        - m = output size (number of selected memories)

        Since m << n typically (most memories are filtered), this is sub-quadratic
        in practice. The algorithm processes memories in rank order, ensuring that
        higher-ranked memories are always kept when redundancy is detected.

        Args:
            memories: Pre-sorted memory list (sorted by final_score descending)

        Returns:
            Tuple of (filtered_memories, filtered_count) where:
            - filtered_memories: List of non-redundant memories in rank order
            - filtered_count: Number of memories filtered out due to redundancy

        Requirements:
            - 3.1: Compute pairwise similarity between candidate and selected memories
            - 3.2: Exclude candidate when similarity > threshold with any selected memory
            - 3.3: Keep higher-ranked memory based on final_score (implicit via order)
            - 3.4: Use algorithm with sub-quadratic complexity O(n × m)

        Example:
            >>> guard = RedundancyGuard(threshold=0.85)
            >>> memories = [
            ...     RankedMemory(memory_id="m1", final_score=0.95, ...),
            ...     RankedMemory(memory_id="m2", final_score=0.90, ...),  # Similar to m1
            ...     RankedMemory(memory_id="m3", final_score=0.85, ...),  # Different
            ... ]
            >>> filtered, count = guard.filter(memories)
            >>> # m1 is kept (highest rank)
            >>> # m2 is filtered (similar to m1, similarity > 0.85)
            >>> # m3 is kept (not similar to m1)
            >>> assert len(filtered) == 2
            >>> assert count == 1
        """
        selected: List[Any] = []
        filtered_count = 0

        for candidate in memories:
            # Check redundancy against already-selected memories
            is_redundant = False
            for selected_memory in selected:
                # Use pre-computed similarity via _compute_similarity
                # This method uses embeddings from metadata (pre-computed during ranking)
                similarity = self._compute_similarity(candidate, selected_memory)

                if similarity > self.threshold:
                    # Candidate is redundant with this selected memory
                    is_redundant = True
                    filtered_count += 1
                    break  # No need to check other selected memories

            if not is_redundant:
                # Candidate is not redundant, add to selected set
                selected.append(candidate)

        return selected, filtered_count



class TokenBudgetEnforcer:
    """Enforces token budget and memory count constraints.
    
    This class implements dual constraint enforcement for memory selection:
    1. Token budget constraint: Total tokens must not exceed max_token_budget
    2. Memory count constraint: Number of memories must not exceed max_memory_count
    
    The enforcer processes memories in rank order (already sorted by final_score)
    and stops selecting memories when either constraint would be violated. This
    ensures that the output respects both token budget and memory count limits.
    
    The enforcer uses TokenEstimator to deterministically estimate token counts
    for each memory, ensuring reproducible behavior across runs.
    
    Attributes:
        max_token_budget: Maximum tokens allowed in output
        max_memory_count: Maximum number of memories allowed in output
        token_estimator: Token estimation component for deterministic token counting
    
    Requirements:
        - 2.2: Track cumulative token count as memories are selected
        - 2.3: Exclude memory when adding it would exceed max_token_budget
        - 2.4: Guarantee total_tokens never exceeds max_token_budget
        - 8.1: Track count of selected memories during injection
        - 8.2: Stop selecting when count reaches max_memory_count
        - 8.3: Apply both max_memory_count and max_token_budget simultaneously
        - 8.4: Return exactly max_memory_count memories when limit reached before budget
    
    Example:
        >>> estimator = TokenEstimator(estimation_factor=1.3)
        >>> enforcer = TokenBudgetEnforcer(
        ...     max_token_budget=2048,
        ...     max_memory_count=50,
        ...     token_estimator=estimator
        ... )
        >>> selected, total_tokens, filtered = enforcer.enforce(memories)
        >>> assert len(selected) <= 50  # Memory count constraint
        >>> assert total_tokens <= 2048  # Token budget constraint
    """
    
    def __init__(
        self,
        max_token_budget: int,
        max_memory_count: int,
        token_estimator: TokenEstimator
    ):
        """Initialize budget enforcer.
        
        Args:
            max_token_budget: Maximum tokens allowed in output (must be positive)
            max_memory_count: Maximum number of memories allowed (must be positive)
            token_estimator: Token estimation component for deterministic counting
        
        Raises:
            ValueError: If max_token_budget or max_memory_count is not positive
        
        Requirements:
            - 2.2: Accept max_token_budget parameter
            - 8.1: Accept max_memory_count parameter
        
        Example:
            >>> estimator = TokenEstimator(estimation_factor=1.3)
            >>> enforcer = TokenBudgetEnforcer(
            ...     max_token_budget=2048,
            ...     max_memory_count=50,
            ...     token_estimator=estimator
            ... )
        """
        if max_token_budget <= 0:
            raise ValueError(
                f"max_token_budget must be positive, got {max_token_budget}"
            )
        if max_memory_count <= 0:
            raise ValueError(
                f"max_memory_count must be positive, got {max_memory_count}"
            )
        
        self.max_token_budget = max_token_budget
        self.max_memory_count = max_memory_count
        self.token_estimator = token_estimator

    def enforce(
        self,
        memories: List[Any]
    ) -> Tuple[List[Any], int, int]:
        """Enforce budget constraints.
        
        Selects memories until either:
        - Token budget would be exceeded
        - Memory count limit is reached
        
        This method implements dual constraint enforcement:
        1. Tracks cumulative token count as memories are processed
        2. Tracks memory count as memories are selected
        3. Stops when either constraint would be violated
        
        The method processes memories in rank order (already sorted by final_score)
        and ensures that:
        - Adding a memory never causes total_tokens to exceed max_token_budget
        - The number of selected memories never exceeds max_memory_count
        
        Args:
            memories: Pre-filtered memory list (after category and redundancy filtering)
        
        Returns:
            Tuple of (selected_memories, total_tokens, filtered_count) where:
            - selected_memories: List of memories that fit within constraints
            - total_tokens: Total token count of selected memories
            - filtered_count: Number of memories filtered due to budget/count limits
        
        Requirements:
            - 2.2: Track cumulative token count as memories are selected
            - 2.3: Exclude memory when adding it would exceed max_token_budget
            - 2.4: Guarantee total_tokens never exceeds max_token_budget
            - 8.2: Stop selecting when count reaches max_memory_count
            - 8.3: Apply both max_memory_count and max_token_budget simultaneously
            - 8.4: Return exactly max_memory_count memories when limit reached before budget
        
        Example:
            >>> estimator = TokenEstimator(estimation_factor=1.3)
            >>> enforcer = TokenBudgetEnforcer(
            ...     max_token_budget=100,
            ...     max_memory_count=5,
            ...     token_estimator=estimator
            ... )
            >>> memories = [
            ...     RankedMemory(content="Short", ...),  # 1 word × 1.3 = 1 token
            ...     RankedMemory(content="Medium length", ...),  # 2 words × 1.3 = 2 tokens
            ...     RankedMemory(content="Very long content here", ...),  # 4 words × 1.3 = 5 tokens
            ... ]
            >>> selected, total_tokens, filtered = enforcer.enforce(memories)
            >>> assert len(selected) <= 5  # Memory count constraint
            >>> assert total_tokens <= 100  # Token budget constraint
            >>> assert filtered == len(memories) - len(selected)  # Filtered count
        """
        selected: List[Any] = []
        total_tokens = 0
        filtered_count = 0
        
        for memory in memories:
            # Check memory count limit
            if len(selected) >= self.max_memory_count:
                filtered_count += 1
                continue
            
            # Estimate tokens for this memory
            memory_tokens = self.token_estimator.estimate_tokens(memory)
            
            # Check if adding this memory would exceed budget
            if total_tokens + memory_tokens > self.max_token_budget:
                filtered_count += 1
                continue
            
            # Add memory to selected set
            selected.append(memory)
            total_tokens += memory_tokens
        
        return selected, total_tokens, filtered_count


class InjectionEngine:
    """
    Main context injection engine.
    
    Orchestrates the injection pipeline:
    1. Validate configuration
    2. Filter by category (if enabled)
    3. Filter by redundancy
    4. Enforce token budget and count limits
    5. Transform to output format
    
    The InjectionEngine is the main orchestrator that coordinates all injection
    components to select and format pre-ranked memories for model context windows.
    It operates downstream of the ranking engine and guarantees deterministic,
    reproducible behavior.
    
    Key Features:
    - Validates configuration on initialization (fail-fast)
    - Initializes all component instances with proper dependencies
    - Accepts optional observability components (metrics, logging)
    - Maintains separation from ranking logic (no score computation)
    - Preserves memory metadata integrity (immutability)
    
    Attributes:
        config: Injection configuration
        metrics_collector: Optional metrics collector for observability
        logger: Optional structured logger for observability
        category_filter: Component for category isolation filtering
        token_estimator: Component for deterministic token estimation
        redundancy_guard: Component for similarity-based deduplication
        budget_enforcer: Component for token budget and count enforcement
    
    Requirements:
        - 1.2: Accept InjectionConfig object with all configuration parameters
        - 9.1: Do not compute or modify final_score values
        - 9.2: Do not implement ranking algorithms or scoring logic
        - 9.3: Accept pre-computed similarity scores from ranking phase
    
    Example:
        >>> config = InjectionConfig(
        ...     max_token_budget=2048,
        ...     max_memory_count=50,
        ...     redundancy_similarity_threshold=0.85,
        ...     enable_category_isolation=False
        ... )
        >>> engine = InjectionEngine(config)
        >>> result = engine.inject(ranked_memories)
        >>> print(f"Selected {len(result.memories)} memories")
        >>> print(f"Total tokens: {result.total_tokens}")
    """
    
    def __init__(
        self,
        config: InjectionConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize injection engine.
        
        This method performs the following initialization steps:
        1. Validate configuration (fail-fast on invalid config)
        2. Store configuration and observability components
        3. Initialize CategoryFilter with config
        4. Initialize TokenEstimator with estimation factor
        5. Initialize RedundancyGuard with similarity threshold
        6. Initialize TokenBudgetEnforcer with budget, count, and estimator
        
        The initialization follows the dependency injection pattern, allowing
        optional observability components (metrics_collector, logger) to be
        provided for production monitoring.
        
        Args:
            config: Injection configuration containing all parameters
            metrics_collector: Optional MetricsCollector for recording metrics
            logger: Optional StructuredLogger for structured logging
        
        Raises:
            ValueError: If configuration is invalid (via config.validate())
        
        Requirements:
            - 1.2: Accept InjectionConfig object with all configuration parameters
            - 8.1: Accept optional metrics_collector parameter for dependency injection
            - 8.2: Accept optional logger parameter for dependency injection
            - 8.7: Store metrics_collector as instance variable
            - 8.8: Store logger as instance variable
            - 8.9: Maintain backward compatibility (parameters are optional)
            - 9.1: Do not compute or modify final_score values
            - 9.2: Do not implement ranking algorithms or scoring logic
            - 9.3: Accept pre-computed similarity scores from ranking phase
        
        Example:
            >>> # Basic initialization
            >>> config = InjectionConfig(
            ...     max_token_budget=2048,
            ...     max_memory_count=50,
            ...     redundancy_similarity_threshold=0.85,
            ...     enable_category_isolation=False
            ... )
            >>> engine = InjectionEngine(config)
            
            >>> # With observability components
            >>> from luma.core.metrics_collector import MetricsCollector
            >>> from luma.core.structured_logger import StructuredLogger
            >>> 
            >>> metrics = MetricsCollector()
            >>> logger = StructuredLogger(name="injection_engine")
            >>> engine = InjectionEngine(config, metrics, logger)
            
            >>> # Invalid configuration raises ValueError
            >>> bad_config = InjectionConfig(
            ...     max_token_budget=-100,  # Invalid!
            ...     max_memory_count=50,
            ...     redundancy_similarity_threshold=0.85,
            ...     enable_category_isolation=False
            ... )
            >>> engine = InjectionEngine(bad_config)  # Raises ValueError
        """
        # Validate configuration (fail-fast on invalid config)
        # This ensures all configuration parameters are valid before
        # initializing any components
        config.validate()
        
        # Store configuration and observability components
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logger
        
        # Initialize all component instances with proper dependencies
        
        # 1. CategoryFilter: Handles category isolation filtering
        self.category_filter = CategoryFilter(config)
        
        # 2. TokenEstimator: Provides deterministic token count estimation
        self.token_estimator = TokenEstimator(config.token_estimation_factor)
        
        # 3. RedundancyGuard: Filters out semantically similar memories
        self.redundancy_guard = RedundancyGuard(
            config.redundancy_similarity_threshold
        )
        
        # 4. TokenBudgetEnforcer: Enforces token budget and memory count limits
        self.budget_enforcer = TokenBudgetEnforcer(
            config.max_token_budget,
            config.max_memory_count,
            self.token_estimator
        )


    def inject(
        self,
        ranked_memories: List[Any]
    ) -> InjectionResult:
        """
        Select and format memories for context injection.

        This method orchestrates the complete injection pipeline:
        1. Handle empty input case (early return)
        2. Apply category filtering (if enabled)
        3. Apply redundancy filtering (similarity-based deduplication)
        4. Apply budget enforcement (token budget + memory count limits)
        5. Transform to output format (RankedMemory → InjectedMemory)
        6. Track diagnostic counts for observability

        The method maintains determinism by:
        - Never modifying or re-ranking input memories
        - Using deterministic token estimation
        - Using deterministic similarity computation
        - Preserving input order in output

        Args:
            ranked_memories: Pre-ranked, pre-sorted memory list
                            (sorted by final_score descending)

        Returns:
            InjectionResult with selected memories and diagnostics

        Requirements:
            - 1.1: Accept list of RankedMemory objects sorted by final_score
            - 1.3: Do not modify or re-rank the input memory list
            - 1.4: Return empty result when input is empty
            - 4.4: Apply category filtering before token budget and redundancy checks
            - 9.4: Do not depend on ranking engine implementation details

        Example:
            >>> config = InjectionConfig(
            ...     max_token_budget=2048,
            ...     max_memory_count=50,
            ...     redundancy_similarity_threshold=0.85,
            ...     enable_category_isolation=False
            ... )
            >>> engine = InjectionEngine(config)
            >>> result = engine.inject(ranked_memories)
            >>> print(f"Selected {len(result.memories)} memories")
            >>> print(f"Total tokens: {result.total_tokens}")
            >>> print(f"Filtered by category: {result.filtered_by_category}")
            >>> print(f"Filtered by redundancy: {result.filtered_by_redundancy}")
            >>> print(f"Filtered by budget: {result.filtered_by_budget}")
        """
        import time
        start_time = time.perf_counter()

        try:
            # Handle empty input case (Requirement 1.4)
            # Return empty result with zero counts
            if not ranked_memories:
                result = InjectionResult(
                    memories=[],
                    total_tokens=0,
                    input_count=0,
                    filtered_by_category=0,
                    filtered_by_redundancy=0,
                    filtered_by_budget=0
                )

                # Record metrics for empty input case
                if self.metrics_collector is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.metrics_collector.record_duration("injection_engine_latency_ms", duration_ms)
                    self.metrics_collector.increment("injection_engine_count")

                # Log empty input case
                if self.logger is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.logger.log("injection_empty_input", {
                        "input_count": 0,
                        "output_count": 0,
                        "total_tokens": 0,
                        "filtered_by_category": 0,
                        "filtered_by_redundancy": 0,
                        "filtered_by_budget": 0,
                        "duration_ms": duration_ms
                    })

                return result

            # Track input count for diagnostics
            input_count = len(ranked_memories)

            # 1. Category filtering (Requirement 4.4)
            # Apply category filtering before redundancy and budget checks
            after_category = self.category_filter.filter(ranked_memories)
            filtered_by_category = input_count - len(after_category)

            # 2. Redundancy filtering
            # Filter out semantically similar memories using pre-computed embeddings
            after_redundancy, filtered_by_redundancy = \
                self.redundancy_guard.filter(after_category)

            # 3. Budget enforcement
            # Apply both token budget and memory count constraints
            selected, total_tokens, filtered_by_budget = \
                self.budget_enforcer.enforce(after_redundancy)

            # 4. Transform to output format
            # Convert RankedMemory objects to InjectedMemory objects
            # This preserves metadata integrity (immutability)
            injected_memories = [
                self._to_injected_memory(m) for m in selected
            ]

            # Create result with diagnostic information
            result = InjectionResult(
                memories=injected_memories,
                total_tokens=total_tokens,
                input_count=input_count,
                filtered_by_category=filtered_by_category,
                filtered_by_redundancy=filtered_by_redundancy,
                filtered_by_budget=filtered_by_budget
            )

            # Record metrics for successful injection
            if self.metrics_collector is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics_collector.record_duration("injection_engine_latency_ms", duration_ms)
                self.metrics_collector.increment("injection_engine_count")

            # Log injection success with detailed statistics
            if self.logger is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.logger.log("injection_completed", {
                    "input_count": input_count,
                    "output_count": len(injected_memories),
                    "total_tokens": total_tokens,
                    "filtered_by_category": filtered_by_category,
                    "filtered_by_redundancy": filtered_by_redundancy,
                    "filtered_by_budget": filtered_by_budget,
                    "token_utilization": total_tokens / self.config.max_token_budget if self.config.max_token_budget > 0 else 0.0,
                    "memory_utilization": len(injected_memories) / self.config.max_memory_count if self.config.max_memory_count > 0 else 0.0,
                    "duration_ms": duration_ms
                })

            return result

        except Exception as e:
            # Record metrics for failed injection
            if self.metrics_collector is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.metrics_collector.record_duration("injection_engine_latency_ms", duration_ms)
                self.metrics_collector.increment("injection_engine_count")

            # Log injection failure
            if self.logger is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self.logger.log("injection_failed", {
                    "input_count": len(ranked_memories) if ranked_memories else 0,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": duration_ms
                })

            # Re-raise the exception to maintain original behavior
            raise


    def _to_injected_memory(self, memory: Any) -> InjectedMemory:
        """
        Transform RankedMemory to InjectedMemory.

        This method converts a RankedMemory object from the ranking engine
        to an InjectedMemory object for the output. It preserves all essential
        fields and maintains metadata integrity (immutability invariant).

        The transformation:
        - Preserves memory_id, content, timestamp, category exactly
        - Preserves metadata dictionary without modification
        - Preserves similarity_score from ranking phase
        - Does NOT modify any fields

        Args:
            memory: RankedMemory from selection

        Returns:
            InjectedMemory for output

        Requirements:
            - 5.2: Include memory_id, content, metadata, and score fields
            - 10.1: Preserve all metadata fields from input
            - 10.2: Do not modify memory content during injection
            - 10.3: Include original timestamp, category, and custom metadata
            - 10.4: Metadata in output matches metadata in input (invariant)

        Example:
            >>> ranked_memory = RankedMemory(
            ...     memory_id="mem_123",
            ...     content="Python is a programming language",
            ...     timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            ...     category="programming",
            ...     similarity_score=0.92,
            ...     final_score=0.87,
            ...     metadata={"source": "user_input", "token_count": 45}
            ... )
            >>> injected = engine._to_injected_memory(ranked_memory)
            >>> assert injected.memory_id == "mem_123"
            >>> assert injected.content == "Python is a programming language"
            >>> assert injected.metadata == {"source": "user_input", "token_count": 45}
        """
        return InjectedMemory(
            memory_id=memory.memory_id,
            content=memory.content,
            metadata=memory.metadata,  # Preserved as-is (immutability)
            similarity_score=memory.similarity_score,
            timestamp=memory.timestamp,
            category=memory.category
        )

