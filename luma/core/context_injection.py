"""
Context Injection Module

This module provides a stateless mechanism for injecting ranked memory retrieval
results into reasoning context. It bridges the gap between the memory retrieval
system (MemoryInterface + RankingEngine) and the ReasoningEngine's context
assembly process.

Key Features:
- Stateless operation with no retained state between invocations
- Graceful degradation on retrieval failures (returns empty list)
- Configurable memory limits (5-20) to prevent context overflow
- Preserves ranking order and metadata integrity
- Deterministic behavior for identical inputs
- Pure data structures (no storage-specific objects)

Example:
    >>> from luma.core.context_injection import inject_memories, InjectionConfig
    >>> from luma.core.memory_interface import MemoryInterface
    >>> 
    >>> config = InjectionConfig(max_memories=10)
    >>> context = inject_memories(
    ...     query="Python programming",
    ...     memory_interface=memory_interface,
    ...     config=config
    ... )
    >>> print(f"Found {len(context['memories'])} memories")
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult,
    MemoryRetrievalError
)
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class InjectionConfig:
    """
    Configuration for context injection.
    
    Controls how memories are retrieved and injected into the reasoning context.
    Supports configurable limits and optional filtering by category, tags, and
    time range.
    
    Attributes:
        max_memories: Maximum number of memories to inject. Must be in range
                     [5, 20] inclusive. This limit prevents context overflow
                     while balancing information richness.
        category_filter: Optional category filter for retrieval. Only memories
                        with this exact category will be retrieved.
        tag_filters: Optional list of tags for filtering. Only memories
                    containing ALL specified tags will be retrieved (AND logic).
        time_range: Optional time range filter as (start_time, end_time) tuple.
                   Only memories within this range will be retrieved.
    
    Example:
        >>> config = InjectionConfig(
        ...     max_memories=10,
        ...     category_filter="education",
        ...     tag_filters=["python", "programming"]
        ... )
        >>> config.validate()  # Raises ValueError if invalid
    """
    max_memories: int
    category_filter: Optional[str] = None
    tag_filters: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Ensures max_memories is within the valid range [5, 20]. Configuration
        errors indicate programming errors that should be caught during
        initialization, not runtime.
        
        Raises:
            ValueError: If max_memories is outside the range [5, 20]
        
        Example:
            >>> config = InjectionConfig(max_memories=25)
            >>> config.validate()
            Traceback (most recent call last):
            ValueError: max_memories must be in [5, 20], got 25
        """
        if not (5 <= self.max_memories <= 20):
            raise ValueError(
                f"max_memories must be in [5, 20], got {self.max_memories}"
            )


# ============================================================================
# Transformation Functions
# ============================================================================


def transform_memory_entry(entry: MemoryEntry) -> Dict[str, Any]:
    """
    Transform MemoryEntry to pure dictionary with primitive types only.
    
    Converts a MemoryEntry TypedDict to a plain dictionary containing only
    primitive types (str, int, float, bool, None, list, dict). This ensures
    no storage-specific objects leak into the reasoning context.
    
    All metadata is preserved exactly as-is without modification (round-trip
    property). The transformation maintains all required fields: id, content,
    category, timestamp, metadata, and tags.
    
    Args:
        entry: MemoryEntry to transform
    
    Returns:
        Dictionary with primitive types only, containing all required fields
    
    Example:
        >>> entry: MemoryEntry = {
        ...     "id": "mem_123",
        ...     "content": "Python is a programming language",
        ...     "category": "education",
        ...     "timestamp": "2024-01-15T10:30:00",
        ...     "metadata": {"source": "user_input"},
        ...     "tags": ["programming", "python"]
        ... }
        >>> transformed = transform_memory_entry(entry)
        >>> assert transformed["metadata"] == entry["metadata"]  # Preserved
    """
    return {
        "id": entry["id"],
        "content": entry["content"],
        "category": entry["category"],
        "timestamp": entry["timestamp"],  # Already ISO 8601 string
        "metadata": entry["metadata"],    # Preserve as-is
        "tags": entry["tags"]             # Preserve as-is
    }


# ============================================================================
# Main Injection Function
# ============================================================================


def inject_memories(
    query: str,
    memory_interface: MemoryInterface,
    config: InjectionConfig,
    existing_context: Optional[Dict[str, Any]] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    logger: Optional[StructuredLogger] = None
) -> Dict[str, Any]:
    """
    Inject retrieved memories into reasoning context.
    
    This is the main entry point for context injection. It retrieves memories
    from the MemoryInterface, transforms them to pure data structures, applies
    size limits, and injects them under the "memories" key in the context.
    
    The function is stateless and has no side effects beyond logging. It handles
    all failures gracefully by logging errors and returning an empty memories
    list, ensuring the reasoning process can continue even when retrieval fails.
    
    Behavior:
    - Validates configuration (raises ValueError if invalid)
    - Constructs QueryParameters from query and config
    - Retrieves memories via MemoryInterface
    - Transforms MemoryEntry objects to pure dictionaries
    - Applies size limit (truncates to max_memories)
    - Injects under "memories" key in context
    - Handles failures gracefully (logs error, returns empty list)
    - Never propagates exceptions to caller
    
    Args:
        query: User query for memory retrieval. Used to search within memory
              content using the underlying search mechanism.
        memory_interface: Memory retrieval interface. Must implement the
                         MemoryInterface abstract class.
        config: Injection configuration controlling limits and filters.
               Must pass validation (max_memories in [5, 20]).
        existing_context: Optional existing context dictionary to enrich.
                         If provided, memories are added to this context.
                         If None, a new context dictionary is created.
        metrics_collector: Optional metrics collector for recording performance
                          metrics. If provided, records context_injection_latency_ms
                          and increments context_injection_count.
        logger: Optional structured logger for recording injection events.
               If provided, logs injection success/failure with memory counts.
    
    Returns:
        Context dictionary with "memories" key containing list of memory dicts.
        The "memories" key is guaranteed to exist and be a list type, even on
        failures (empty list). Each memory dict contains: id, content, category,
        timestamp, metadata, tags.
    
    Raises:
        ValueError: If config.validate() fails (configuration error)
    
    Note:
        This function never propagates MemoryRetrievalError or other exceptions
        from the memory interface. All failures are handled gracefully with
        logging and empty list fallback.
    
    Example:
        >>> config = InjectionConfig(max_memories=10)
        >>> context = inject_memories(
        ...     query="Python programming",
        ...     memory_interface=memory_interface,
        ...     config=config
        ... )
        >>> assert "memories" in context
        >>> assert isinstance(context["memories"], list)
        >>> for memory in context["memories"]:
        ...     print(f"- {memory['content'][:50]}...")
    """
    # Validate configuration (fail fast on config errors)
    config.validate()
    
    # Initialize context
    context = existing_context.copy() if existing_context else {}
    
    # Start timing measurement
    start_time = time.perf_counter()
    
    try:
        # Construct query parameters
        params: QueryParameters = {
            "query": query,
            "limit": config.max_memories
        }
        
        # Add optional filters only if specified
        if config.category_filter:
            params["category"] = config.category_filter
        if config.tag_filters:
            params["tags"] = config.tag_filters
        if config.time_range:
            params["start_time"] = config.time_range[0]
            params["end_time"] = config.time_range[1]
        
        # Retrieve memories from interface
        result: RetrievalResult = memory_interface.retrieve(params=params)
        
        # Extract memories and apply size limit (truncate if needed)
        retrieved_memories = result["memories"]
        truncated = len(retrieved_memories) > config.max_memories
        
        # Transform to pure dictionaries (only up to max_memories)
        memories = [
            transform_memory_entry(entry)
            for entry in retrieved_memories[:config.max_memories]
        ]
        
        # Inject into context
        context["memories"] = memories
        
        # Record metrics for successful injection
        if metrics_collector is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics_collector.record_duration("context_injection_latency_ms", duration_ms)
            metrics_collector.increment("context_injection_count")
        
        # Log success
        if logger is not None:
            logger.log(
                "context_injection_success",
                {
                    "query": query,
                    "memory_count": len(memories),
                    "truncated": truncated,
                    "original_count": len(retrieved_memories)
                }
            )
            
            # Log truncation if it occurred
            if truncated:
                logger.log(
                    "context_injection_truncated",
                    {
                        "original_count": len(retrieved_memories),
                        "final_count": len(memories),
                        "limit": config.max_memories
                    }
                )
        
    except MemoryRetrievalError as e:
        # Record metrics for failed injection
        if metrics_collector is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics_collector.record_duration("context_injection_latency_ms", duration_ms)
            metrics_collector.increment("context_injection_count")
        
        # Handle retrieval failure gracefully
        if logger is not None:
            logger.log(
                "context_injection_failure",
                {
                    "query": query,
                    "config": {
                        "max_memories": config.max_memories,
                        "category_filter": config.category_filter,
                        "tag_filters": config.tag_filters,
                        "time_range": config.time_range
                    },
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
        context["memories"] = []
        
    except Exception as e:
        # Record metrics for failed injection
        if metrics_collector is not None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics_collector.record_duration("context_injection_latency_ms", duration_ms)
            metrics_collector.increment("context_injection_count")
        
        # Handle unexpected errors gracefully
        if logger is not None:
            logger.log(
                "context_injection_error",
                {
                    "query": query,
                    "config": {
                        "max_memories": config.max_memories,
                        "category_filter": config.category_filter,
                        "tag_filters": config.tag_filters,
                        "time_range": config.time_range
                    },
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
        context["memories"] = []
    
    return context
