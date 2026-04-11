# Memory Retrieval Enhancement Architecture

## Table of Contents

1. [Overview](#overview)
2. [Retrieval Flow Diagram](#retrieval-flow-diagram)
3. [Query Parameter Mapping](#query-parameter-mapping)
4. [Fallback Behavior](#fallback-behavior)
5. [Error Handling Strategy](#error-handling-strategy)
6. [Component Interactions](#component-interactions)
7. [Type Safety and Contracts](#type-safety-and-contracts)
8. [Performance Considerations](#performance-considerations)

---

## Overview

The Memory Retrieval Enhancement extends Luma's memory system with production-ready features including typed contracts, rich query parameters, comprehensive error handling, and concurrency safety. This document describes the architecture of these enhancements and how they integrate with the existing memory system.

### Key Enhancements

- **Typed Contracts**: TypedDict definitions for QueryParameters, MemoryEntry, and RetrievalResult
- **Rich Query Parameters**: Support for category, timestamp range, and tag filters
- **Configuration Flexibility**: Adapter configuration with device_id, default_category, and default_tags
- **Comprehensive Error Handling**: Graceful fallback behavior and detailed error reporting
- **Backward Compatibility**: Legacy API support maintained
- **Future-Proof Design**: Preparation for vector search capabilities

---

## Retrieval Flow Diagram

### Enhanced Retrieval Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ReasoningEngine                               │
│  - Detects memory-related intents                                    │
│  - Constructs QueryParameters from user message                      │
│  - Handles errors gracefully                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ retrieve(params=QueryParameters)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MemoryInterface                                 │
│  - Abstract contract for memory operations                           │
│  - Defines typed contracts (QueryParameters, RetrievalResult)        │
│  - Documents expected behavior                                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ retrieve(query, params, limit)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SQLiteMemoryAdapter                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. Validate and Normalize Parameters                          │  │
│  │    - Validate query string (handle None, empty, whitespace)   │  │
│  │    - Validate limit (must be positive integer)                │  │
│  │    - Validate timestamp range (start <= end)                  │  │
│  │    - Validate tags (must be list of strings)                  │  │
│  │    - Validate category (must be string)                       │  │
│  │    - Handle embedding parameter (log and ignore)              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                         │                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 2. Map Parameters to MemoryManager                            │  │
│  │    QueryParameters → MemoryManager.query_memories()           │  │
│  │    - query → action_type                                      │  │
│  │    - start_time → start_time                                  │  │
│  │    - end_time → end_time                                      │  │
│  │    - tags → tags                                              │  │
│  │    - limit → limit                                            │  │
│  │    - category → (post-processing filter)                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                         │                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 3. Execute Query                                              │  │
│  │    - Track execution time                                     │  │
│  │    - Call MemoryManager.query_memories()                      │  │
│  │    - Transform results to MemoryEntry format                  │  │
│  │    - Apply category filter (post-processing)                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                         │                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 4. Build RetrievalResult                                      │  │
│  │    - memories: List[MemoryEntry]                              │  │
│  │    - total_count: int                                         │  │
│  │    - query_metadata:                                          │  │
│  │      * execution_time_ms                                      │  │
│  │      * filters_applied                                        │  │
│  │      * limit                                                  │  │
│  │      * has_more                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ RetrievalResult
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MemoryManager                                   │
│  - Executes SQL query with filters                                  │
│  - Returns MemoryEntry objects                                      │
│  - Handles encryption/decryption                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Retrieval Flow Steps

1. **Intent Detection**: ReasoningEngine detects memory retrieval intent from user message
2. **Parameter Construction**: Build QueryParameters from user message (extract category, tags, time range)
3. **Validation**: SQLiteMemoryAdapter validates all parameters before execution
4. **Parameter Mapping**: Map QueryParameters to MemoryManager.query_memories() parameters
5. **Query Execution**: MemoryManager executes SQL query with filters
6. **Result Transformation**: Transform MemoryManager entries to MemoryEntry format
7. **Post-Processing**: Apply category filter (MemoryManager doesn't support it directly)
8. **Metadata Collection**: Build query_metadata with execution time and filters
9. **Result Return**: Return RetrievalResult with memories and metadata
10. **Context Injection**: ReasoningEngine injects memories into LLM context

---

## Query Parameter Mapping

### Parameter Mapping Table

The SQLiteMemoryAdapter maps QueryParameters to MemoryManager.query_memories() parameters:

| QueryParameters Field | MemoryManager Parameter | Notes |
|----------------------|------------------------|-------|
| `query` (str) | `action_type` (str) | Text query for content matching |
| `start_time` (datetime) | `start_time` (datetime) | Inclusive start of time range |
| `end_time` (datetime) | `end_time` (datetime) | Inclusive end of time range |
| `tags` (List[str]) | `tags` (List[str]) | Must contain all specified tags (AND logic) |
| `limit` (int) | `limit` (int) | Maximum number of results |
| `category` (str) | N/A | Applied in post-processing (MemoryManager doesn't support) |
| `embedding` (List[float]) | N/A | Reserved for future vector search, currently ignored |

### Mapping Implementation

```python
def retrieve(self, query, params, limit):
    # Validate and normalize parameters
    validated_params = self._validate_and_normalize_params(query, params)
    
    # Map to MemoryManager parameters
    entries = self.memory_manager.query_memories(
        action_type=validated_params.get("query"),      # query → action_type
        start_time=validated_params.get("start_time"),  # start_time → start_time
        end_time=validated_params.get("end_time"),      # end_time → end_time
        tags=validated_params.get("tags"),              # tags → tags
        limit=validated_params.get("limit", 10)         # limit → limit
    )
    
    # Transform results
    memories = self._transform_entries(entries)
    
    # Apply category filter (post-processing)
    category_filter = validated_params.get("category")
    if category_filter is not None:
        memories = [m for m in memories if m["category"] == category_filter]
    
    return RetrievalResult(...)
```

### Why Post-Processing for Category?

The MemoryManager.query_memories() method doesn't support category filtering directly. To maintain backward compatibility and avoid modifying the existing MemoryManager, category filtering is applied in post-processing after retrieving results.

**Trade-offs**:
- **Pro**: No changes to existing MemoryManager code
- **Pro**: Maintains backward compatibility
- **Con**: May retrieve more entries than needed (filtered after retrieval)
- **Future**: Consider adding category support to MemoryManager for efficiency

---

## Fallback Behavior

### Retrieval Failure Fallback

When memory retrieval fails, the system gracefully falls back to LLM-only processing:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Normal Retrieval Flow                             │
│                                                                       │
│  User Message → Intent Detection → Retrieve Memories                 │
│                                   ↓                                   │
│                            Inject into Context                        │
│                                   ↓                                   │
│                            Generate Response                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  Fallback Flow (Retrieval Fails)                     │
│                                                                       │
│  User Message → Intent Detection → Retrieve Memories                 │
│                                   ↓                                   │
│                            MemoryRetrievalError                       │
│                                   ↓                                   │
│                            Catch Exception                            │
│                                   ↓                                   │
│                            Log Error Details                          │
│                                   ↓                                   │
│                     Inject Empty Memories List                        │
│                                   ↓                                   │
│                     Generate Response (LLM-only)                      │
│                                   ↓                                   │
│                     Add Fallback Metadata                             │
│                     {"fallback": true, "error": "..."}                │
└─────────────────────────────────────────────────────────────────────┘
```

### Fallback Implementation

```python
def _handle_retrieve_memory(self, user_message, context):
    try:
        # Build query parameters
        params = self._build_query_parameters(user_message)
        
        # Retrieve memories
        result = self.memory.retrieve(params=params)
        
        # Inject into context
        context["memories"] = result["memories"]
        
        # Log success
        logger.info(f"Retrieved {result['total_count']} memories")
        
        return self._generate_response(context)
        
    except MemoryRetrievalError as e:
        # Log error with full details
        logger.error(f"Memory retrieval failed: {e}", exc_info=True)
        
        # Fall back to LLM-only processing
        context["memories"] = []
        
        # Generate response without memories
        response = self._generate_response(context)
        
        # Add fallback metadata
        response["metadata"] = {
            "fallback": True,
            "error": str(e),
            "message": "Proceeding without memory context"
        }
        
        return response
```

### Fallback Guarantees

1. **No Crashes**: System never crashes due to retrieval failures
2. **Continued Operation**: User receives a response even when retrieval fails
3. **Transparency**: Response metadata indicates fallback occurred
4. **Detailed Logging**: Full error details logged for debugging
5. **Graceful Degradation**: LLM can still provide useful responses without memories

---

## Error Handling Strategy

### Error Categories

The system handles three categories of errors with different strategies:

#### 1. Validation Errors (ValueError)

**When**: Invalid query parameters (wrong types, invalid ranges, negative limits)

**Strategy**: Fail fast with descriptive error message

**Flow**:
```
Invalid Parameters → Validation → ValueError → Return Error to Caller
```

**Example**:
```python
try:
    result = adapter.retrieve(params={"limit": -5})
except ValueError as e:
    # Error: "limit must be a positive integer, got -5"
    # Caller can fix parameters and retry
```

**Rationale**: Validation errors indicate programmer error or malformed input. Failing fast helps catch bugs early.

#### 2. Retrieval Errors (MemoryRetrievalError)

**When**: Database errors, query failures, or other retrieval-related problems

**Strategy**: Catch, log, and fall back to LLM-only processing

**Flow**:
```
Retrieval Fails → MemoryRetrievalError → Catch → Log → Fallback → Continue
```

**Example**:
```python
try:
    result = adapter.retrieve(params={"query": "test"})
except MemoryRetrievalError as e:
    logger.error(f"Retrieval failed: {e}")
    # Fall back to empty results
    result = {"memories": [], "total_count": 0, "query_metadata": {}}
```

**Rationale**: Retrieval errors are often transient (database locked, disk full). System should continue operating.

#### 3. Storage Errors (MemoryStorageError)

**When**: Database errors, disk full, or other storage-related problems

**Strategy**: Catch, log, and return user-friendly error message

**Flow**:
```
Storage Fails → MemoryStorageError → Catch → Log → Return Error Message
```

**Example**:
```python
try:
    memory_id = adapter.store("content", metadata={})
except MemoryStorageError as e:
    logger.error(f"Storage failed: {e}")
    return {
        "success": False,
        "error": "Failed to store memory. Please try again.",
        "metadata": {"error_details": str(e)}
    }
```

**Rationale**: Storage errors require user awareness (data not saved). Clear feedback enables retry.

### Error Handling Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Error Handling Flow                           │
│                                                                       │
│  Operation Attempted                                                 │
│         │                                                             │
│         ├─→ Validation Error (ValueError)                            │
│         │      ↓                                                      │
│         │   Fail Fast → Return Error to Caller                       │
│         │                                                             │
│         ├─→ Retrieval Error (MemoryRetrievalError)                   │
│         │      ↓                                                      │
│         │   Catch → Log → Fallback → Continue                        │
│         │                                                             │
│         └─→ Storage Error (MemoryStorageError)                       │
│                ↓                                                      │
│             Catch → Log → Return User-Friendly Error                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Error Logging

All errors are logged with comprehensive details:

```python
# Validation Error
logger.error(f"Invalid query parameters: {e}")

# Retrieval Error
logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
logger.error(f"Query parameters: {params}")

# Storage Error
logger.error(f"Failed to store memory: {e}", exc_info=True)
logger.error(f"Content length: {len(content)}, Metadata: {metadata}")
```

**Logging Levels**:
- **ERROR**: Retrieval and storage failures
- **WARNING**: Validation errors (expected in some cases)
- **DEBUG**: Parameter validation details, query execution details
- **INFO**: Successful operations with metadata

---

## Component Interactions

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ReasoningEngine                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ _handle_retrieve_memory()                                     │  │
│  │  - Build QueryParameters from user message                    │  │
│  │  - Call memory.retrieve(params=params)                        │  │
│  │  - Inject memories into context                               │  │
│  │  - Handle MemoryRetrievalError with fallback                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ _handle_store_memory()                                        │  │
│  │  - Extract content and metadata from message                  │  │
│  │  - Call memory.store(content, metadata)                       │  │
│  │  - Handle MemoryStorageError with user feedback               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ MemoryInterface
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SQLiteMemoryAdapter                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Configuration                                                 │  │
│  │  - device_id: str                                             │  │
│  │  - default_category: str                                      │  │
│  │  - default_tags: List[str]                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ store(content, metadata)                                      │  │
│  │  - Apply default_category if not in metadata                  │  │
│  │  - Merge default_tags with metadata tags                      │  │
│  │  - Pass device_id to MemoryManager                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ retrieve(query, params, limit)                                │  │
│  │  - Validate and normalize parameters                          │  │
│  │  - Map to MemoryManager parameters                            │  │
│  │  - Execute query and track time                               │  │
│  │  - Transform results to MemoryEntry format                    │  │
│  │  - Build RetrievalResult with metadata                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         │ MemoryManager API
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MemoryManager                                   │
│  - create_memory(action, context, device_id, tags)                  │
│  - query_memories(action_type, start_time, end_time, tags, limit)   │
│  - Handles encryption/decryption                                    │
│  - Manages SQLite storage                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Interaction Sequence

#### Retrieval Sequence

```
ReasoningEngine                SQLiteMemoryAdapter              MemoryManager
      │                               │                              │
      │ retrieve(params)              │                              │
      ├──────────────────────────────>│                              │
      │                               │                              │
      │                               │ _validate_and_normalize()    │
      │                               ├──────────────┐               │
      │                               │              │               │
      │                               │<─────────────┘               │
      │                               │                              │
      │                               │ query_memories()             │
      │                               ├─────────────────────────────>│
      │                               │                              │
      │                               │                              │ Execute SQL
      │                               │                              ├──────────┐
      │                               │                              │          │
      │                               │                              │<─────────┘
      │                               │                              │
      │                               │ List[MemoryEntry]            │
      │                               │<─────────────────────────────┤
      │                               │                              │
      │                               │ _transform_entries()         │
      │                               ├──────────────┐               │
      │                               │              │               │
      │                               │<─────────────┘               │
      │                               │                              │
      │ RetrievalResult               │                              │
      │<──────────────────────────────┤                              │
      │                               │                              │
```

#### Storage Sequence

```
ReasoningEngine                SQLiteMemoryAdapter              MemoryManager
      │                               │                              │
      │ store(content, metadata)      │                              │
      ├──────────────────────────────>│                              │
      │                               │                              │
      │                               │ Apply defaults               │
      │                               ├──────────────┐               │
      │                               │              │               │
      │                               │<─────────────┘               │
      │                               │                              │
      │                               │ create_memory()              │
      │                               ├─────────────────────────────>│
      │                               │                              │
      │                               │                              │ Store to DB
      │                               │                              ├──────────┐
      │                               │                              │          │
      │                               │                              │<─────────┘
      │                               │                              │
      │                               │ entry_id                     │
      │                               │<─────────────────────────────┤
      │                               │                              │
      │ entry_id                      │                              │
      │<──────────────────────────────┤                              │
      │                               │                              │
```

---

## Type Safety and Contracts

### Typed Contracts Overview

The system uses TypedDict for type definitions to provide runtime type checking and IDE support:

```python
from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime

class QueryParameters(TypedDict, total=False):
    """All fields optional for flexible querying"""
    query: str
    category: str
    start_time: datetime
    end_time: datetime
    tags: List[str]
    limit: int
    embedding: Optional[List[float]]

class MemoryEntry(TypedDict):
    """All fields required for consistent representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    category: str
    tags: List[str]

class RetrievalResult(TypedDict):
    """Structured result with execution metadata"""
    memories: List[MemoryEntry]
    total_count: int
    query_metadata: Dict[str, Any]
```

### Type Safety Benefits

1. **IDE Support**: Autocomplete and type hints in IDEs
2. **Runtime Validation**: Type checking at runtime (with type checkers)
3. **Documentation**: Types serve as documentation
4. **Refactoring Safety**: Type errors caught during refactoring
5. **API Clarity**: Clear contract for all memory operations

### Contract Guarantees

**QueryParameters Contract**:
- All fields are optional (total=False)
- Multiple filters combined with AND logic
- Invalid parameters raise ValueError before execution

**MemoryEntry Contract**:
- All fields are required
- Consistent format across all memory operations
- Timestamp in ISO 8601 format

**RetrievalResult Contract**:
- Always includes memories list (empty if no results)
- Always includes total_count (0 if no results)
- Always includes query_metadata with execution details

---

## Performance Considerations

### Performance Targets

- **Validation**: < 1ms for typical parameter sets
- **Query Execution**: < 200ms for queries returning up to 100 entries
- **Parameter Mapping**: < 1ms overhead
- **Result Transformation**: < 10ms for 100 entries

### Optimization Strategies

#### 1. Parameter Validation Caching

Validation logic is optimized to minimize overhead:
- Type checks use isinstance() (fast)
- Range checks are simple comparisons
- No complex regex or parsing

#### 2. Efficient Parameter Mapping

Direct dictionary access with minimal transformation:
```python
# Fast: Direct mapping
entries = self.memory_manager.query_memories(
    action_type=validated_params.get("query"),
    start_time=validated_params.get("start_time"),
    # ...
)
```

#### 3. Lazy Result Transformation

Results are transformed only when needed:
- Transform happens after database query
- Category filtering applied after transformation
- No unnecessary object creation

#### 4. Metadata Collection

Execution time tracking has minimal overhead:
```python
import time
start_time = time.time()
# ... execute query ...
execution_time_ms = (time.time() - start_time) * 1000
```

### Performance Monitoring

Track performance metrics for all operations:

```python
# Execution time
result["query_metadata"]["execution_time_ms"] = execution_time_ms

# Filters applied
result["query_metadata"]["filters_applied"] = {
    k: v for k, v in validated_params.items()
    if v is not None and k != "limit"
}

# Result count
result["total_count"] = len(memories)
```

**Monitoring Benefits**:
- Identify slow queries
- Track filter effectiveness
- Detect performance regressions
- Optimize based on real usage

---

## Conclusion

The Memory Retrieval Enhancement architecture provides:

1. **Type Safety**: Typed contracts for all data structures
2. **Rich Querying**: Support for category, tags, and time range filters
3. **Error Resilience**: Comprehensive error handling with graceful fallback
4. **Backward Compatibility**: Legacy API support maintained
5. **Performance**: Optimized for sub-200ms retrieval operations
6. **Extensibility**: Future-proof design for vector search

### Key Architectural Decisions

1. **TypedDict over Pydantic**: Simpler, no additional dependencies
2. **Post-Processing for Category**: Maintains backward compatibility
3. **Graceful Fallback**: System continues operating when retrieval fails
4. **Explicit Validation**: Clear error messages for invalid parameters
5. **Metadata-Rich Results**: Enables monitoring and debugging

### Future Enhancements

1. **Vector Search**: Use embedding field for semantic similarity
2. **Pagination**: Implement has_more flag and offset support
3. **Category Indexing**: Add category support to MemoryManager for efficiency
4. **Caching**: Cache query results for frequently accessed data
5. **Async Support**: Add async retrieve() method for concurrent operations

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Related Specs**: intent-based-memory-retrieval-enhancements  
**Authors**: Luma Memory Enhancement Team
