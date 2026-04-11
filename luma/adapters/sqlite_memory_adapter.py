"""
SQLite Memory Adapter Module.

This module provides the SQLiteMemoryAdapter class that wraps the existing
MemoryManager to implement the MemoryInterface abstraction. This adapter
enables clean architecture by decoupling the reasoning engine from concrete
memory implementations.
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError
)
from luma_memory.memory_manager import MemoryManager

if TYPE_CHECKING:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger


logger = logging.getLogger(__name__)


class SQLiteMemoryAdapter(MemoryInterface):
    """
    Adapter that wraps MemoryManager to implement MemoryInterface.
    
    This adapter translates between the MemoryInterface contract and the
    existing MemoryManager implementation, enabling clean architecture
    without modifying existing memory code.
    
    The adapter follows the Adapter pattern to bridge between two incompatible
    interfaces, allowing the ReasoningEngine to work with any memory
    implementation through the MemoryInterface abstraction.
    
    Attributes:
        memory_manager: The wrapped MemoryManager instance
    
    Example:
        >>> from luma_memory.storage.sqlite_storage import SQLiteStorage
        >>> from luma_memory.memory_manager import MemoryManager
        >>> 
        >>> storage = SQLiteStorage("./data/memory.db")
        >>> memory_manager = MemoryManager(storage=storage)
        >>> adapter = SQLiteMemoryAdapter(memory_manager)
        >>> 
        >>> # Store memory
        >>> memory_id = adapter.store(
        ...     "Python is a programming language",
        ...     metadata={"tags": ["programming"], "category": "education"}
        ... )
        >>> 
        >>> # Retrieve memories
        >>> results = adapter.retrieve("Python", limit=5)
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        device_id: Optional[str] = None,
        default_category: Optional[str] = None,
        default_tags: Optional[List[str]] = None
    ):
        """
        Initialize adapter with existing MemoryManager and configuration options.

        Args:
            memory_manager: Configured MemoryManager instance to wrap
            device_id: Optional device identifier (default: "reasoning-engine")
            default_category: Optional default category (default: "general")
            default_tags: Optional default tags to merge with all memories

        Raises:
            ValueError: If memory_manager is None
        """
        if memory_manager is None:
            raise ValueError("memory_manager cannot be None")

        self.memory_manager = memory_manager
        self.device_id = device_id or "reasoning-engine"
        self.default_category = default_category or "general"
        self.default_tags = default_tags or []
        self._closed = False

        logger.info(
            f"SQLiteMemoryAdapter initialized with device_id={self.device_id}, "
            f"default_category={self.default_category}, "
            f"default_tags={self.default_tags}"
        )


    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content with configuration defaults applied.
        
        Applies default_category if no category in metadata.
        Merges default_tags with metadata tags.
        Passes device_id to MemoryManager.create_memory().
        
        Args:
            content: The text content to store
            metadata: Optional dictionary of metadata (tags, category, etc.)
        
        Returns:
            str: Unique identifier for the stored memory entry
        
        Raises:
            MemoryStorageError: If storage operation fails
        
        Example:
            >>> memory_id = adapter.store(
            ...     "User completed Python tutorial",
            ...     metadata={"tags": ["learning", "python"], "category": "education"}
            ... )
        """
        try:
            # Extract and merge tags
            metadata_tags = metadata.get("tags", []) if metadata else []
            merged_tags = list(set(self.default_tags + metadata_tags))
            
            # Extract category with default fallback
            category = metadata.get("category", self.default_category) if metadata else self.default_category
            
            # Build context
            context = {"content": content, "category": category}
            if metadata:
                for key, value in metadata.items():
                    if key not in ["tags", "category"]:
                        context[key] = value
            
            # Create memory with device_id
            entry_id = self.memory_manager.create_memory(
                action=content,
                context=context,
                device_id=self.device_id,
                tags=merged_tags
            )
            
            logger.debug(f"Stored memory {entry_id} with category={category}, tags={merged_tags}")
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise MemoryStorageError(f"Storage failed: {e}") from e

    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional['QueryParameters'] = None,
        limit: int = 10,
        metrics_collector: Optional['MetricsCollector'] = None,
        logger_instance: Optional['StructuredLogger'] = None
    ) -> 'RetrievalResult':
        """
        Retrieve memories with enhanced query parameters.
        
        Maps QueryParameters to MemoryManager.query_memories() parameters.
        Validates parameters before execution.
        Returns structured RetrievalResult with metadata.
        
        Args:
            query: Optional text query for backward compatibility
            params: Optional structured query parameters
            limit: Maximum number of results (default: 10)
            metrics_collector: Optional MetricsCollector for recording metrics
            logger_instance: Optional StructuredLogger for logging events
        
        Returns:
            RetrievalResult: Structured result with memories and metadata
        
        Raises:
            MemoryRetrievalError: If retrieval operation fails
            ValueError: If query parameters are invalid
        """
        import time
        
        # Start timing for instrumentation
        start_perf = time.perf_counter()
        
        try:
            # Validate and normalize parameters
            validated_params = self._validate_and_normalize_params(query, params)
            
            # Use limit from params if provided, otherwise use the limit parameter
            if params and "limit" in params:
                validated_params["limit"] = params["limit"]
            else:
                validated_params["limit"] = limit
            
            # Track query execution time
            start_time = time.time()
            
            # Query MemoryManager
            entries = self.memory_manager.query_memories(
                action_type=validated_params.get("query"),
                start_time=validated_params.get("start_time"),
                end_time=validated_params.get("end_time"),
                tags=validated_params.get("tags"),
                limit=validated_params.get("limit", 10)
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Transform to MemoryEntry format
            memories = self._transform_entries(entries)
            
            # Apply category filter in post-processing (MemoryManager doesn't support it)
            category_filter = validated_params.get("category")
            if category_filter is not None:
                memories = [
                    m for m in memories
                    if m["category"] == category_filter
                ]
            
            # Build result with metadata
            result = {
                "memories": memories,
                "total_count": len(memories),
                "query_metadata": {
                    "execution_time_ms": execution_time_ms,
                    "filters_applied": {
                        k: v for k, v in validated_params.items()
                        if v is not None and k != "limit"
                    },
                    "limit": validated_params.get("limit", 10),
                    "has_more": False  # Future: implement pagination
                }
            }
            
            # Record metrics if metrics_collector is provided
            end_perf = time.perf_counter()
            retrieval_duration_ms = (end_perf - start_perf) * 1000
            
            if metrics_collector is not None:
                metrics_collector.record_duration("retrieval_latency_ms", retrieval_duration_ms)
                metrics_collector.increment("retrieval_count")
            
            # Log retrieval event if logger is provided
            if logger_instance is not None:
                logger_instance.log("memory_retrieval", {
                    "total_count": len(memories),
                    "duration_ms": retrieval_duration_ms,
                    "filters": result['query_metadata']['filters_applied']
                })
            
            logger.debug(
                f"Retrieved {len(memories)} memories in {execution_time_ms:.2f}ms "
                f"with filters: {result['query_metadata']['filters_applied']}"
            )
            
            return result
            
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Invalid query parameters: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise MemoryRetrievalError(f"Retrieval failed: {e}") from e

    def _validate_and_normalize_params(
        self,
        query: Optional[str],
        params: Optional['QueryParameters']
    ) -> Dict[str, Any]:
        """
        Validate and normalize query parameters.
        
        Handles backward compatibility with legacy query string.
        Validates timestamp ranges, limit values, and query strings.
        
        Args:
            query: Optional text query for backward compatibility
            params: Optional structured query parameters
        
        Returns:
            Dict with normalized parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        from datetime import datetime
        
        # Start with params if provided, otherwise use query
        if params:
            normalized = dict(params)
        elif query is not None:
            normalized = {"query": query}
        else:
            normalized = {}
        
        # Validate query string
        query_str = normalized.get("query")
        if query_str is not None:
            if not isinstance(query_str, str):
                raise ValueError(f"query must be a string, got {type(query_str).__name__}")
            # Treat empty/whitespace as None
            if not query_str.strip():
                normalized["query"] = None
        
        # Validate limit
        limit = normalized.get("limit", 10)
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        normalized["limit"] = limit
        
        # Validate timestamp range
        start_time = normalized.get("start_time")
        end_time = normalized.get("end_time")
        if start_time and end_time:
            if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
                raise ValueError("start_time and end_time must be datetime objects")
            if start_time > end_time:
                raise ValueError(f"start_time ({start_time}) must be <= end_time ({end_time})")
        
        # Validate tags
        tags = normalized.get("tags")
        if tags is not None:
            if not isinstance(tags, list):
                raise ValueError(f"tags must be a list, got {type(tags).__name__}")
            if not all(isinstance(tag, str) for tag in tags):
                raise ValueError("all tags must be strings")
        
        # Validate category
        category = normalized.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError(f"category must be a string, got {type(category).__name__}")
        
        # Note: embedding field is ignored for now (future vector search)
        if "embedding" in normalized:
            logger.debug("embedding parameter provided but not yet implemented, ignoring")
            normalized.pop("embedding")
        
        return normalized

    def _transform_entries(self, entries: List[Any]) -> List['MemoryEntry']:
        """
        Transform MemoryManager entries to MemoryEntry format.
        
        Extracts fields from MemoryManager's MemoryEntry objects and
        formats them according to the MemoryEntry TypedDict contract.
        
        Args:
            entries: List of MemoryManager MemoryEntry objects
        
        Returns:
            List of MemoryEntry dictionaries
        """
        results = []
        for entry in entries:
            memory_entry = {
                "id": entry.id,
                "content": entry.action,
                "metadata": entry.context or {},
                "timestamp": (
                    entry.created_at.isoformat()
                    if entry.created_at
                    else entry.timestamp.isoformat()
                ),
                "category": entry.context.get("category", self.default_category) if entry.context else self.default_category,
                "tags": entry.tags or []
            }
            results.append(memory_entry)
        return results

    def close(self) -> None:
        """
        Close the adapter and release underlying resources.
        
        This method provides an explicit lifecycle hook for deterministic shutdown
        in services and tests. It delegates to the underlying storage's close()
        method if available. Safe to call multiple times (idempotent).
        
        After calling close(), the adapter should not be used for further operations.
        This method is optional but recommended for production deployments to ensure
        predictable resource cleanup without relying on garbage collection.
        
        Example:
            >>> adapter = SQLiteMemoryAdapter(memory_manager)
            >>> try:
            ...     adapter.store("content", metadata={})
            ... finally:
            ...     adapter.close()  # Ensure cleanup
            
            >>> # Or use as context manager (if implemented)
            >>> with SQLiteMemoryAdapter(memory_manager) as adapter:
            ...     adapter.store("content", metadata={})
        
        Note:
            This method is idempotent - calling it multiple times is safe and
            will not raise errors. Subsequent calls after the first are no-ops.
        """
        if self._closed:
            logger.debug("Adapter already closed, skipping")
            return
        
        try:
            # Attempt to close the underlying storage if it has a close method
            if hasattr(self.memory_manager, 'storage') and self.memory_manager.storage:
                storage = self.memory_manager.storage
                if hasattr(storage, 'close') and callable(storage.close):
                    storage.close()
                    logger.info("Closed underlying storage via MemoryManager")
            
            self._closed = True
            logger.info("SQLiteMemoryAdapter closed successfully")
            
        except Exception as e:
            logger.warning(f"Error during adapter close (non-fatal): {e}")
            # Mark as closed anyway to prevent repeated attempts
            self._closed = True
