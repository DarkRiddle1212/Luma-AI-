"""
Memory Interface Module

This module defines the abstract interface for memory storage and retrieval
implementations and custom exception classes for memory operations.

The MemoryInterface allows swapping between different memory backends (SQLite,
vector databases, cloud storage) without changing reasoning engine code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypedDict, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger


# ============================================================================
# Typed Contracts
# ============================================================================


class QueryParameters(TypedDict, total=False):
    """
    Parameters for memory retrieval queries.
    
    All fields are optional to support flexible querying.
    When multiple filters are provided, they are combined with AND logic.
    
    Fields:
        query (str): Text query for content matching. Used to search within
                    memory content using the underlying search mechanism.
        category (str): Filter by category (exact match). Only memories with
                       this exact category will be returned.
        start_time (datetime): Filter by timestamp >= start_time. Only memories
                              created at or after this time will be returned.
        end_time (datetime): Filter by timestamp <= end_time. Only memories
                            created at or before this time will be returned.
        tags (List[str]): Filter by tags (must contain all specified tags).
                         Only memories containing ALL specified tags will be
                         returned (AND logic).
        limit (int): Maximum number of results to return. Defaults to 10 if
                    not specified. Must be a positive integer.
        embedding (Optional[List[float]]): Reserved for future vector search.
                                          Vector representation for semantic
                                          similarity search. Not yet implemented.
    
    Example:
        >>> params: QueryParameters = {
        ...     "query": "Python programming",
        ...     "category": "education",
        ...     "tags": ["programming", "python"],
        ...     "limit": 5
        ... }
    """
    query: str
    category: str
    start_time: datetime
    end_time: datetime
    tags: List[str]
    limit: int
    embedding: Optional[List[float]]


class MemoryEntry(TypedDict):
    """
    Represents a single memory entry returned from retrieval.
    
    This is the standard format for all memory entries across the system.
    All fields are required to ensure consistent memory representation.
    
    Fields:
        id (str): Unique identifier for the memory entry. Used to reference
                 specific memories in future operations.
        content (str): The stored content text. This is the primary data
                      that was persisted.
        metadata (Dict[str, Any]): Associated metadata dictionary. May include
                                  custom fields specific to the memory type.
        timestamp (str): ISO 8601 formatted creation timestamp. Represents
                        when the memory was created.
        category (str): Memory category for classification. Used to organize
                       memories into logical groups.
        tags (List[str]): Associated tags for filtering and organization.
                         Used to enable tag-based retrieval.
    
    Example:
        >>> entry: MemoryEntry = {
        ...     "id": "mem_123",
        ...     "content": "Python is a programming language",
        ...     "metadata": {"source": "user_input"},
        ...     "timestamp": "2024-01-15T10:30:00",
        ...     "category": "education",
        ...     "tags": ["programming", "python"]
        ... }
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    category: str
    tags: List[str]


class RetrievalResult(TypedDict):
    """
    Result of a memory retrieval operation including metadata.
    
    Provides both the retrieved memories and metadata about the query execution.
    This structure enables monitoring, debugging, and pagination support.
    
    Fields:
        memories (List[MemoryEntry]): Retrieved memory entries matching the
                                     query criteria. Empty list if no matches.
        total_count (int): Total number of matching memories returned. This
                          may be less than the limit if fewer matches exist.
        query_metadata (Dict[str, Any]): Query execution metadata containing:
            - execution_time_ms (float): Query execution time in milliseconds
            - filters_applied (Dict[str, Any]): Dictionary of filters that were
                                               applied to the query
            - limit (int): Result limit that was used
            - has_more (bool): Whether more results exist beyond the limit
                              (for pagination support)
    
    Example:
        >>> result: RetrievalResult = {
        ...     "memories": [
        ...         {
        ...             "id": "mem_123",
        ...             "content": "Python is a programming language",
        ...             "metadata": {},
        ...             "timestamp": "2024-01-15T10:30:00",
        ...             "category": "education",
        ...             "tags": ["programming"]
        ...         }
        ...     ],
        ...     "total_count": 1,
        ...     "query_metadata": {
        ...         "execution_time_ms": 15.3,
        ...         "filters_applied": {"category": "education"},
        ...         "limit": 10,
        ...         "has_more": False
        ...     }
        ... }
    """
    memories: List[MemoryEntry]
    total_count: int
    query_metadata: Dict[str, Any]


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_query_string(query: Optional[str]) -> Optional[str]:
    """
    Validate and normalize a query string.
    
    Treats empty strings and whitespace-only strings as None.
    Ensures query is a string type if provided.
    
    Args:
        query: Query string to validate, or None
    
    Returns:
        Normalized query string, or None if empty/whitespace
    
    Raises:
        ValueError: If query is not a string or None
    
    Example:
        >>> validate_query_string("Python")
        'Python'
        >>> validate_query_string("  ")
        None
        >>> validate_query_string("")
        None
    """
    if query is None:
        return None
    
    if not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")
    
    # Treat empty/whitespace as None
    if not query.strip():
        return None
    
    return query


def validate_limit(limit: Optional[int]) -> int:
    """
    Validate the limit parameter.
    
    Ensures limit is a positive integer. Defaults to 10 if None.
    
    Args:
        limit: Maximum number of results, or None for default
    
    Returns:
        Validated limit value (defaults to 10)
    
    Raises:
        ValueError: If limit is not a positive integer
    
    Example:
        >>> validate_limit(5)
        5
        >>> validate_limit(None)
        10
        >>> validate_limit(0)
        Traceback (most recent call last):
        ValueError: limit must be a positive integer, got 0
    """
    if limit is None:
        return 10
    
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    
    return limit


def validate_timestamp_range(
    start_time: Optional[datetime],
    end_time: Optional[datetime]
) -> None:
    """
    Validate timestamp range parameters.
    
    Ensures start_time and end_time are datetime objects if provided,
    and that start_time <= end_time.
    
    Args:
        start_time: Start of time range, or None
        end_time: End of time range, or None
    
    Raises:
        ValueError: If timestamps are not datetime objects or if start > end
    
    Example:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 31)
        >>> validate_timestamp_range(start, end)  # No error
        >>> validate_timestamp_range(end, start)
        Traceback (most recent call last):
        ValueError: start_time must be <= end_time
    """
    if start_time is not None and not isinstance(start_time, datetime):
        raise ValueError(
            f"start_time must be a datetime object, got {type(start_time).__name__}"
        )
    
    if end_time is not None and not isinstance(end_time, datetime):
        raise ValueError(
            f"end_time must be a datetime object, got {type(end_time).__name__}"
        )
    
    if start_time and end_time and start_time > end_time:
        raise ValueError(
            f"start_time must be <= end_time, got start_time={start_time}, end_time={end_time}"
        )


def validate_tags(tags: Optional[List[str]]) -> Optional[List[str]]:
    """
    Validate tags parameter.
    
    Ensures tags is a list of strings if provided.
    
    Args:
        tags: List of tag strings, or None
    
    Returns:
        Validated tags list, or None
    
    Raises:
        ValueError: If tags is not a list or contains non-string elements
    
    Example:
        >>> validate_tags(["python", "programming"])
        ['python', 'programming']
        >>> validate_tags(None)
        None
        >>> validate_tags(["python", 123])
        Traceback (most recent call last):
        ValueError: all tags must be strings
    """
    if tags is None:
        return None
    
    if not isinstance(tags, list):
        raise ValueError(f"tags must be a list, got {type(tags).__name__}")
    
    if not all(isinstance(tag, str) for tag in tags):
        raise ValueError("all tags must be strings")
    
    return tags


def validate_category(category: Optional[str]) -> Optional[str]:
    """
    Validate category parameter.
    
    Ensures category is a string if provided.
    
    Args:
        category: Category string, or None
    
    Returns:
        Validated category, or None
    
    Raises:
        ValueError: If category is not a string or None
    
    Example:
        >>> validate_category("education")
        'education'
        >>> validate_category(None)
        None
        >>> validate_category(123)
        Traceback (most recent call last):
        ValueError: category must be a string, got int
    """
    if category is None:
        return None
    
    if not isinstance(category, str):
        raise ValueError(f"category must be a string, got {type(category).__name__}")
    
    return category


def validate_query_parameters(params: Optional[QueryParameters]) -> Dict[str, Any]:
    """
    Validate and normalize all query parameters.
    
    Performs comprehensive validation of all QueryParameters fields.
    Returns a normalized dictionary with validated values.
    
    Args:
        params: QueryParameters dictionary to validate, or None
    
    Returns:
        Dictionary with normalized and validated parameters
    
    Raises:
        ValueError: If any parameter is invalid
    
    Example:
        >>> params: QueryParameters = {
        ...     "query": "Python",
        ...     "limit": 5,
        ...     "tags": ["programming"]
        ... }
        >>> validated = validate_query_parameters(params)
        >>> validated["limit"]
        5
    """
    if params is None:
        return {"limit": 10}
    
    normalized: Dict[str, Any] = {}
    
    # Validate query string
    if "query" in params:
        normalized["query"] = validate_query_string(params.get("query"))
    
    # Validate limit
    normalized["limit"] = validate_limit(params.get("limit"))
    
    # Validate timestamp range
    start_time = params.get("start_time")
    end_time = params.get("end_time")
    validate_timestamp_range(start_time, end_time)
    if start_time is not None:
        normalized["start_time"] = start_time
    if end_time is not None:
        normalized["end_time"] = end_time
    
    # Validate tags
    tags = validate_tags(params.get("tags"))
    if tags is not None:
        normalized["tags"] = tags
    
    # Validate category
    category = validate_category(params.get("category"))
    if category is not None:
        normalized["category"] = category
    
    # Note: embedding field is for future use, ignore if present
    if "embedding" in params:
        # Future: validate embedding vector
        pass
    
    return normalized


# ============================================================================
# Exception Classes
# ============================================================================


class MemoryStorageError(Exception):
    """
    Exception raised when memory storage operation fails.
    
    This exception wraps underlying storage errors to provide a consistent
    error handling interface for memory operations. It should be raised
    when store() operations fail due to database errors, validation issues,
    or other storage-related problems.
    
    Example:
        >>> try:
        ...     memory.store("content", metadata={"tags": ["test"]})
        ... except MemoryStorageError as e:
        ...     print(f"Storage failed: {e}")
    """
    pass


class MemoryRetrievalError(Exception):
    """
    Exception raised when memory retrieval operation fails.
    
    This exception wraps underlying retrieval errors to provide a consistent
    error handling interface for memory operations. It should be raised
    when retrieve() operations fail due to database errors, query issues,
    or other retrieval-related problems.
    
    Example:
        >>> try:
        ...     results = memory.retrieve("query", limit=10)
        ... except MemoryRetrievalError as e:
        ...     print(f"Retrieval failed: {e}")
    """
    pass


class MemoryInterface(ABC):
    """
    Abstract interface for memory storage and retrieval implementations.
    
    This interface defines the contract for memory operations that the
    ReasoningEngine depends on. Concrete implementations must provide
    these operations while handling their own storage mechanisms.
    
    The interface enables clean architecture by decoupling the reasoning
    engine from specific memory implementations, allowing future extensibility
    to vector databases, cloud memory, or multi-device synchronization.
    
    All memory implementations must inherit from this class and implement
    the store and retrieve methods.
    
    Example:
        >>> class CustomMemory(MemoryInterface):
        ...     def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        ...         # Store content and return unique ID
        ...         return "memory_id_123"
        ...     
        ...     def retrieve(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        ...         # Retrieve and return matching memories
        ...         return [{"id": "123", "content": "stored content", "metadata": {}, "timestamp": "2024-01-15T10:30:00"}]
        >>> 
        >>> memory = CustomMemory()
        >>> memory_id = memory.store("Important information", metadata={"tags": ["important"]})
        >>> results = memory.retrieve("information", limit=5)
    """
    
    @abstractmethod
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content to memory with optional metadata.
        
        This method persists the provided content along with any associated
        metadata to the underlying storage mechanism. The implementation
        should generate and return a unique identifier for the stored entry.
        
        Args:
            content: The text content to store. This is the primary data
                    that will be persisted and later retrieved.
            metadata: Optional dictionary of metadata to associate with the
                     content. May include tags (List[str]), category (str),
                     source (str), or other implementation-specific fields.
                     Defaults to None if not provided.
        
        Returns:
            str: Unique identifier for the stored memory entry. This ID
                can be used to reference the specific memory in future
                operations.
        
        Raises:
            MemoryStorageError: If the storage operation fails due to
                              database errors, validation issues, or
                              other storage-related problems.
        
        Example:
            >>> memory = SQLiteMemoryAdapter(memory_manager)
            >>> memory_id = memory.store(
            ...     "Python is a programming language",
            ...     metadata={"tags": ["programming", "python"], "category": "education"}
            ... )
            >>> print(f"Stored with ID: {memory_id}")
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10,
        metrics_collector: Optional["MetricsCollector"] = None,
        logger: Optional["StructuredLogger"] = None
    ) -> RetrievalResult:
        """
        Retrieve memories matching the query and parameters.
        
        This method supports two calling patterns for backward compatibility:
        
        1. **Legacy API** (backward compatible):
           retrieve(query="text", limit=10)
           Returns memories matching the query string with specified limit.
        
        2. **Enhanced API** (recommended):
           retrieve(params={"query": "text", "category": "education", "tags": ["python"]})
           Returns memories matching structured query parameters with rich filtering.
        
        When both query and params are provided, params takes precedence.
        The enhanced API returns a RetrievalResult with execution metadata,
        while maintaining compatibility with existing code expecting a list.
        
        Args:
            query: Optional text query string for backward compatibility.
                  The implementation determines how this query is interpreted
                  (exact match, fuzzy search, semantic similarity, etc.).
                  Defaults to None.
            params: Optional structured query parameters supporting rich filtering:
                   - query (str): Text query for content matching
                   - category (str): Filter by exact category match
                   - start_time (datetime): Filter by timestamp >= start_time
                   - end_time (datetime): Filter by timestamp <= end_time
                   - tags (List[str]): Filter by tags (must contain all)
                   - limit (int): Maximum results (overrides limit parameter)
                   - embedding (List[float]): Reserved for future vector search
                   Defaults to None.
            limit: Maximum number of results to return. Defaults to 10.
                  This parameter is used when params is None or params doesn't
                  specify a limit. Ignored if params["limit"] is provided.
            metrics_collector: Optional MetricsCollector instance for recording
                             retrieval metrics (latency, counts). When provided,
                             implementations should record retrieval_latency_ms
                             and increment retrieval counters. Defaults to None.
            logger: Optional StructuredLogger instance for logging retrieval
                   events. When provided, implementations should log retrieval
                   operations with relevant context. Defaults to None.
        
        Returns:
            RetrievalResult: Structured result containing:
                - memories (List[MemoryEntry]): Retrieved memory entries with
                  fields: id, content, metadata, timestamp, category, tags
                - total_count (int): Number of memories returned
                - query_metadata (Dict[str, Any]): Execution metadata including:
                  * execution_time_ms (float): Query execution time
                  * filters_applied (Dict): Applied filter parameters
                  * limit (int): Result limit used
                  * has_more (bool): Whether more results exist
                
                Returns empty memories list if no matches found.
        
        Raises:
            MemoryRetrievalError: If the retrieval operation fails due to
                                database errors, query issues, or other
                                retrieval-related problems.
            ValueError: If query parameters are invalid (wrong types, invalid
                       ranges, negative limits, etc.).
        
        Note:
            The embedding field in params is reserved for future vector search
            capabilities. Current implementations should accept but ignore this
            parameter to maintain forward compatibility.
        
        Examples:
            Legacy API (backward compatible):
            >>> memory = SQLiteMemoryAdapter(memory_manager)
            >>> result = memory.retrieve("Python programming", limit=5)
            >>> for entry in result["memories"]:
            ...     print(f"Content: {entry['content']}")
            
            Enhanced API with filters:
            >>> params: QueryParameters = {
            ...     "query": "Python",
            ...     "category": "education",
            ...     "tags": ["programming", "python"],
            ...     "limit": 5
            ... }
            >>> result = memory.retrieve(params=params)
            >>> print(f"Found {result['total_count']} memories")
            >>> print(f"Execution time: {result['query_metadata']['execution_time_ms']}ms")
            >>> for entry in result["memories"]:
            ...     print(f"Content: {entry['content']}")
            ...     print(f"Category: {entry['category']}")
            ...     print(f"Tags: {entry['tags']}")
            
            Time range filtering:
            >>> from datetime import datetime, timedelta
            >>> end = datetime.now()
            >>> start = end - timedelta(days=7)
            >>> params: QueryParameters = {
            ...     "start_time": start,
            ...     "end_time": end,
            ...     "limit": 10
            ... }
            >>> result = memory.retrieve(params=params)
            >>> print(f"Memories from last 7 days: {result['total_count']}")
        """
        pass
