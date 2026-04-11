"""
REST API routes for Luma Memory Module.

This module provides FastAPI endpoints for memory operations including
creating, retrieving, querying, and managing memory entries.
"""

from fastapi import FastAPI, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from contextlib import asynccontextmanager
import logging
import time

from luma_memory.models import SensitivityLevel, SyncStatus
from luma_memory.memory_manager import MemoryManager
from luma_memory.processing.validation import ValidationError, ValidationManager
from luma_memory.storage.backend import StorageError

# Set up logger for API routes
logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class CreateMemoryRequest(BaseModel):
    """Request model for creating a new memory entry."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "User opened document",
                "context": {"file": "report.pdf", "page": 1},
                "device_id": "laptop-001",
                "sensitivity": "private",
                "tags": ["document", "work"]
            }
        }
    )
    
    action: str = Field(..., description="Description of the user action", min_length=1, max_length=1000)
    context: Dict[str, Any] = Field(..., description="Contextual information dictionary")
    device_id: str = Field(..., description="Identifier of the device creating the entry", min_length=1, max_length=255)
    sensitivity: str = Field(default="public", description="Privacy level: public, private, or sensitive")
    tags: List[str] = Field(default_factory=list, description="List of tags for categorization", max_length=50)
    
    
    @field_validator('sensitivity')
    @classmethod
    def validate_sensitivity(cls, v):
        """Validate that sensitivity is one of the allowed values."""
        allowed = ['public', 'private', 'sensitive']
        if v.lower() not in allowed:
            raise ValueError(f"Invalid sensitivity level: {v}. Must be one of: {', '.join(allowed)}")
        return v.lower()
    
    @field_validator('tags')
    @classmethod
    def validate_tag_lengths(cls, v):
        """Validate that each tag is within the maximum length."""
        if v:
            for tag in v:
                if len(tag) > 100:
                    raise ValueError(f"Tag exceeds maximum length of 100 characters: {tag[:20]}...")
        return v
    
    

class CreateMemoryResponse(BaseModel):
    """Response model for memory creation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entry_id": "abc-123-def-456",
                "message": "Memory entry created successfully"
            }
        }
    )
    
    entry_id: str = Field(..., description="Unique identifier of the created entry")
    message: str = Field(default="Memory entry created successfully")


class MemoryEntryResponse(BaseModel):
    """Response model for a single memory entry."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc-123-def-456",
                "timestamp": "2024-01-15T10:30:00Z",
                "action": "User opened document",
                "context": {"file": "report.pdf"},
                "sensitivity": "private",
                "device_id": "laptop-001",
                "sync_status": "pending",
                "tags": ["document", "work"],
                "summary": None,
                "parent_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    id: str
    timestamp: str
    action: str
    context: Dict[str, Any]
    sensitivity: str
    device_id: str
    sync_status: str
    tags: List[str]
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class QueryMemoryRequest(BaseModel):
    """Request model for querying memory entries."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-31T23:59:59Z",
                "tags": ["work"],
                "action_type": "document",
                "limit": 50,
                "offset": 0
            }
        }
    )
    
    start_time: Optional[str] = Field(None, description="Start of time range (ISO format)")
    end_time: Optional[str] = Field(None, description="End of time range (ISO format)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags", max_length=50)
    action_type: Optional[str] = Field(None, description="Filter by action type (partial match)", max_length=1000)
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of entries to return")
    offset: int = Field(default=0, ge=0, description="Number of entries to skip for pagination")
    
    @field_validator('tags')
    @classmethod
    def validate_tag_lengths(cls, v):
        """Validate that each tag is within the maximum length."""
        if v:
            for tag in v:
                if len(tag) > 100:
                    raise ValueError(f"Tag exceeds maximum length of 100 characters: {tag[:20]}...")
        return v


class QueryMemoryResponse(BaseModel):
    """Response model for memory query results."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entries": [],
                "total": 0,
                "limit": 100,
                "offset": 0
            }
        }
    )
    
    entries: List[MemoryEntryResponse]
    total: int = Field(..., description="Total number of entries returned")
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current server timestamp")


class StatsResponse(BaseModel):
    """Response model for statistics."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_entries": 1000,
                "storage_size_bytes": 1048576,
                "encryption_enabled": True,
                "summarizer_enabled": True,
                "config": {
                    "cache_size": 1000,
                    "max_storage_size_mb": 1000
                },
                "performance": {
                    "create_memory": {
                        "count": 100,
                        "avg_time_ms": 45.2
                    }
                }
            }
        }
    )
    
    total_entries: int
    storage_size_bytes: int
    encryption_enabled: bool
    summarizer_enabled: bool
    config: Dict[str, Any]
    performance: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation failed",
                "detail": "Action cannot be empty"
            }
        }
    )
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and service status endpoints",
    },
    {
        "name": "Statistics",
        "description": "Storage and performance statistics endpoints",
    },
    {
        "name": "Memory Operations",
        "description": "Core memory entry operations including create, retrieve, and query",
    },
]

# Create FastAPI application
# Note: lifespan is set in server.py via create_app()
app = FastAPI(
    title="Luma Memory Module API",
    description="""
## Luma Memory Module API

The Luma Memory Module provides persistent storage and retrieval of user actions and context summaries 
for the Luma personal AI system. It serves as the central memory layer for lightweight agents running 
on laptop and phone devices.

### Features

* **Store Memory Entries**: Persist user actions with context, metadata, and optional encryption
* **Query Memories**: Retrieve entries with flexible filtering by time, tags, and action type
* **Automatic Summarization**: Reduce storage overhead by consolidating similar entries
* **Local-First Storage**: All data stored locally using SQLite with optional encryption
* **Performance Optimized**: Sub-100ms storage, sub-200ms retrieval with LRU caching

### Authentication

Currently, the API does not require authentication. Future versions will support API key authentication.

### Rate Limiting

No rate limiting is currently enforced. Clients should implement their own throttling if needed.

### Error Handling

All endpoints return standard HTTP status codes:
- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters or validation error
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Service not ready

Error responses include a JSON body with `error` and optional `detail` fields.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    contact={
        "name": "Luma Memory Module",
        "url": "https://github.com/luma/memory-module",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Logging middleware for request/response logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware to log all HTTP requests and responses.
    
    Logs:
    - Request method, path, and query parameters
    - Request headers (excluding sensitive data)
    - Response status code
    - Request processing time
    - Client IP address
    - User agent
    
    This middleware provides comprehensive request/response logging for
    debugging, monitoring, and auditing purposes.
    """
    # Generate unique request ID for tracing
    request_id = f"{int(time.time() * 1000)}-{id(request)}"
    
    # Record start time
    start_time = time.time()
    
    # Extract request information
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    query_params = dict(request.query_params) if request.query_params else {}
    
    # Log incoming request
    logger.info(
        f"Incoming request: {method} {path}",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "query_params": query_params,
            "client_host": client_host,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
    )
    
    # Process request and capture response
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Log response
        logger.info(
            f"Request completed: {method} {path} - Status: {response.status_code}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time, 2),
                "client_host": client_host,
            }
        )
        
        # Add custom headers for debugging
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response
        
    except Exception as e:
        # Calculate processing time even for errors
        process_time = (time.time() - start_time) * 1000
        
        # Log error
        logger.error(
            f"Request failed: {method} {path} - Error: {str(e)}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time_ms": round(process_time, 2),
                "client_host": client_host,
            },
            exc_info=True
        )
        
        # Re-raise the exception to be handled by FastAPI
        raise

# Global memory manager instance (will be initialized on startup)
memory_manager: Optional[MemoryManager] = None

# Global validation manager for request validation
validation_manager = ValidationManager(strict_mode=True)


def get_memory_manager() -> MemoryManager:
    """
    Get the global memory manager instance.
    
    Returns:
        MemoryManager instance
    
    Raises:
        HTTPException: If memory manager is not initialized
    """
    if memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not initialized"
        )
    return memory_manager


def set_memory_manager(manager: MemoryManager) -> None:
    """
    Set the global memory manager instance.
    
    Args:
        manager: MemoryManager instance to use
    """
    global memory_manager
    memory_manager = manager


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    responses={
        200: {
            "description": "Service is healthy and operational",
            "model": HealthResponse
        }
    }
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the API. Use this endpoint to verify
    that the service is running and accepting requests.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    )


@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["Statistics"],
    summary="Get Storage and Performance Statistics",
    responses={
        200: {
            "description": "Statistics retrieved successfully",
            "model": StatsResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    }
)
async def get_stats():
    """
    Get storage and performance statistics.
    
    Returns comprehensive statistics including:
    - Total number of memory entries stored
    - Storage size in bytes
    - Component status (encryption, summarizer)
    - Configuration settings
    - Performance metrics (if monitoring is enabled)
    
    This endpoint is useful for monitoring system health and resource usage.
    """
    try:
        manager = get_memory_manager()
        stats = manager.get_stats()
        return StatsResponse(**stats)
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
@app.get(
    "/api/v1/metrics",
    tags=["Statistics"],
    summary="Get Performance Metrics",
    responses={
        200: {
            "description": "Performance metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "create_memory": {
                            "count": 100,
                            "avg_time_ms": 45.2,
                            "min_time_ms": 12.5,
                            "max_time_ms": 98.7,
                            "errors": 2,
                            "error_rate": 2.0
                        },
                        "get_memory": {
                            "count": 500,
                            "avg_time_ms": 15.3,
                            "min_time_ms": 5.1,
                            "max_time_ms": 185.2,
                            "errors": 0,
                            "error_rate": 0.0
                        },
                        "system_resources": {
                            "memory_usage_mb": 45.2,
                            "memory_usage_percent": 2.5,
                            "cpu_percent": 1.2,
                            "num_threads": 8
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - memory manager not initialized",
            "model": ErrorResponse
        }
    }
)
async def get_metrics():
    """
    Get performance metrics for all operations.

    Returns detailed performance metrics including:
    - Operation counts (number of times each operation was called)
    - Latency statistics (average, min, max response times in milliseconds)
    - Error counts and error rates
    - System resource usage (memory, CPU, threads)

    **Metrics by Operation:**
    - `create_memory`: Memory entry creation performance
    - `get_memory`: Memory entry retrieval performance
    - `query_memories`: Memory query performance
    - `update_memory`: Memory entry update performance
    - `delete_memory`: Memory entry deletion performance

    **System Resources:**
    - `memory_usage_mb`: Current memory usage in megabytes
    - `memory_usage_percent`: Memory usage as percentage of system memory
    - `cpu_percent`: CPU usage percentage
    - `num_threads`: Number of active threads

    **Use Cases:**
    - Monitor API performance and identify bottlenecks
    - Track error rates and reliability
    - Verify performance requirements are met (< 100ms store, < 200ms retrieve)
    - Monitor system resource consumption
    - Set up alerts for performance degradation

    **Note:** Metrics are collected only if `enable_metrics` is set to `true` in configuration.
    If metrics are disabled, this endpoint returns an empty metrics object.

    Returns:
        JSON object with performance metrics for all operations

    Raises:
        HTTPException: 503 if memory manager not initialized, 500 for other errors
    """
    try:
        manager = get_memory_manager()

        # Check if metrics are enabled
        if not manager.config.enable_metrics:
            return {
                "message": "Metrics collection is disabled. Set enable_metrics=true in configuration to enable.",
                "metrics": {}
            }

        # Get performance metrics
        metrics = manager.get_performance_metrics()

        return metrics

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )



@app.post(
    "/api/v1/memory",
    response_model=CreateMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Memory Operations"],
    summary="Create New Memory Entry",
    responses={
        201: {
            "description": "Memory entry created successfully",
            "model": CreateMemoryResponse
        },
        400: {
            "description": "Validation error - invalid request data",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - memory manager not initialized",
            "model": ErrorResponse
        }
    }
)
async def create_memory(request: CreateMemoryRequest):
    """
    Create a new memory entry.
    
    Stores a new memory entry with the provided action, context, and metadata.
    The entry will be:
    1. Validated for required fields and data types
    2. Optionally encrypted based on sensitivity level
    3. Persisted to SQLite storage
    4. Cached for fast retrieval
    
    **Sensitivity Levels:**
    - `public`: No encryption, suitable for non-sensitive data
    - `private`: Encrypted, suitable for personal information
    - `sensitive`: Encrypted with additional security measures
    
    **Performance:**
    - Typical response time: < 100ms
    - Includes validation, encryption, and storage operations
    
    Args:
        request: CreateMemoryRequest containing entry data
    
    Returns:
        CreateMemoryResponse with the created entry ID
    
    Raises:
        HTTPException: 400 for validation errors, 500 for storage errors
    """
    try:
        manager = get_memory_manager()
        
        # Sanitize input data to prevent injection attacks
        sanitized_data = validation_manager.sanitize_input({
            "action": request.action,
            "context": request.context,
            "device_id": request.device_id,
            "sensitivity": request.sensitivity,
            "tags": request.tags
        })
        
        # Validate action is not empty after sanitization
        if not sanitized_data["action"] or not sanitized_data["action"].strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Action cannot be empty or whitespace only"
            )
        
        # Validate device_id is not empty after sanitization
        if not sanitized_data["device_id"] or not sanitized_data["device_id"].strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device ID cannot be empty or whitespace only"
            )
        
        # Validate tags are not empty after sanitization
        for i, tag in enumerate(sanitized_data["tags"]):
            if not tag or not tag.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tag at index {i} cannot be empty or whitespace only"
                )
        
        # Parse sensitivity level (already validated by Pydantic)
        sensitivity = SensitivityLevel(sanitized_data["sensitivity"])
        
        # Create memory entry with sanitized data
        entry_id = manager.create_memory(
            action=sanitized_data["action"],
            context=sanitized_data["context"],
            device_id=sanitized_data["device_id"],
            sensitivity=sensitivity,
            tags=sanitized_data["tags"]
        )
        
        return CreateMemoryResponse(
            entry_id=entry_id,
            message="Memory entry created successfully"
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get(
    "/api/v1/memory/{entry_id}",
    response_model=MemoryEntryResponse,
    tags=["Memory Operations"],
    summary="Retrieve Memory Entry by ID",
    responses={
        200: {
            "description": "Memory entry retrieved successfully",
            "model": MemoryEntryResponse
        },
        404: {
            "description": "Entry not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - memory manager not initialized",
            "model": ErrorResponse
        }
    }
)
async def get_memory(entry_id: str):
    """
    Retrieve a specific memory entry by ID.
    
    Fetches a memory entry from storage and decrypts it if necessary.
    The entry is first checked in the LRU cache for fast retrieval,
    then fetched from SQLite storage if not cached.
    
    **Performance:**
    - Cached entries: < 10ms
    - Uncached entries: < 200ms
    
    Args:
        entry_id: Unique identifier of the entry to retrieve
    
    Returns:
        MemoryEntryResponse with the complete entry data including:
        - Action description
        - Context dictionary
        - Metadata (device_id, tags, timestamps)
        - Sensitivity level
        - Sync status
        - Optional summary and parent_id for summarized entries
    
    Raises:
        HTTPException: 404 if entry not found, 500 for storage errors
    """
    try:
        manager = get_memory_manager()
        
        # Sanitize entry_id to prevent injection attacks
        sanitized_data = validation_manager.sanitize_input({"entry_id": entry_id})
        sanitized_entry_id = sanitized_data["entry_id"]
        
        # Validate entry_id is not empty after sanitization
        if not sanitized_entry_id or not sanitized_entry_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entry ID cannot be empty or whitespace only"
            )
        
        # No strict format validation; attempt retrieval and return 404 if not found
        
        entry = manager.get_memory(sanitized_entry_id)
        
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory entry not found: {sanitized_entry_id}"
            )
        
        # Convert entry to response model
        return MemoryEntryResponse(
            id=entry.id,
            timestamp=entry.timestamp.isoformat().replace('+00:00', 'Z'),
            action=entry.action,
            context=entry.context,
            sensitivity=entry.sensitivity.value,
            device_id=entry.device_id,
            sync_status=entry.sync_status.value,
            tags=entry.tags,
            summary=entry.summary,
            parent_id=entry.parent_id,
            created_at=entry.created_at.isoformat().replace('+00:00', 'Z') if entry.created_at else None,
            updated_at=entry.updated_at.isoformat().replace('+00:00', 'Z') if entry.updated_at else None
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get(
    "/api/v1/metrics",
    tags=["Statistics"],
    summary="Get Performance Metrics",
    responses={
        200: {
            "description": "Performance metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "create_memory": {
                            "count": 100,
                            "avg_time_ms": 45.2,
                            "min_time_ms": 12.5,
                            "max_time_ms": 98.7,
                            "errors": 2,
                            "error_rate": 2.0
                        },
                        "get_memory": {
                            "count": 500,
                            "avg_time_ms": 15.3,
                            "min_time_ms": 5.1,
                            "max_time_ms": 185.2,
                            "errors": 0,
                            "error_rate": 0.0
                        },
                        "system_resources": {
                            "memory_usage_mb": 45.2,
                            "memory_usage_percent": 2.5,
                            "cpu_percent": 1.2,
                            "num_threads": 8
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - memory manager not initialized",
            "model": ErrorResponse
        }
    }
)
async def get_metrics():
    """
    Get performance metrics for all operations.
    
    Returns detailed performance metrics including:
    - Operation counts (number of times each operation was called)
    - Latency statistics (average, min, max response times in milliseconds)
    - Error counts and error rates
    - System resource usage (memory, CPU, threads)
    
    **Metrics by Operation:**
    - `create_memory`: Memory entry creation performance
    - `get_memory`: Memory entry retrieval performance
    - `query_memories`: Memory query performance
    - `update_memory`: Memory entry update performance
    - `delete_memory`: Memory entry deletion performance
    
    **System Resources:**
    - `memory_usage_mb`: Current memory usage in megabytes
    - `memory_usage_percent`: Memory usage as percentage of system memory
    - `cpu_percent`: CPU usage percentage
    - `num_threads`: Number of active threads
    
    **Use Cases:**
    - Monitor API performance and identify bottlenecks
    - Track error rates and reliability
    - Verify performance requirements are met (< 100ms store, < 200ms retrieve)
    - Monitor system resource consumption
    - Set up alerts for performance degradation
    
    **Note:** Metrics are collected only if `enable_metrics` is set to `true` in configuration.
    If metrics are disabled, this endpoint returns an empty metrics object.
    
    Returns:
        JSON object with performance metrics for all operations
    
    Raises:
        HTTPException: 503 if memory manager not initialized, 500 for other errors
    """
    try:
        manager = get_memory_manager()
        
        # Check if metrics are enabled
        if not manager.config.enable_metrics:
            return {
                "message": "Metrics collection is disabled. Set enable_metrics=true in configuration to enable.",
                "metrics": {}
            }
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        
        return metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@app.post(
    "/api/v1/memory/query",
    response_model=QueryMemoryResponse,
    tags=["Memory Operations"],
    summary="Query Memory Entries with Filters",
    responses={
        200: {
            "description": "Query executed successfully",
            "model": QueryMemoryResponse
        },
        400: {
            "description": "Invalid query parameters",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - memory manager not initialized",
            "model": ErrorResponse
        }
    }
)
async def query_memories(request: QueryMemoryRequest):
    """
    Query memory entries with filters.
    
    Retrieves memory entries matching the specified filters including
    time range, tags, and action type. Results are paginated and returned
    in reverse chronological order (newest first).
    
    **Filtering Options:**
    - `start_time` / `end_time`: Filter by timestamp range (ISO 8601 format)
    - `tags`: Filter by one or more tags (entries must have all specified tags)
    - `action_type`: Filter by action description (partial match, case-insensitive)
    - `limit`: Maximum number of results (1-1000, default: 100)
    - `offset`: Number of results to skip for pagination (default: 0)
    
    **Pagination:**
    Use `limit` and `offset` to paginate through large result sets:
    - Page 1: offset=0, limit=100
    - Page 2: offset=100, limit=100
    - Page 3: offset=200, limit=100
    
    **Performance:**
    - Typical response time: < 200ms for 100 entries
    - Indexed queries (by timestamp, tags) are optimized
    
    **Example Query:**
    ```json
    {
      "start_time": "2024-01-01T00:00:00Z",
      "end_time": "2024-01-31T23:59:59Z",
      "tags": ["work", "document"],
      "action_type": "opened",
      "limit": 50,
      "offset": 0
    }
    ```
    
    Args:
        request: QueryMemoryRequest with filter parameters
    
    Returns:
        QueryMemoryResponse with matching entries and pagination info
    
    Raises:
        HTTPException: 400 for invalid parameters, 500 for storage errors
    """
    try:
        manager = get_memory_manager()
        
        # Sanitize string inputs
        sanitized_action_type = None
        if request.action_type:
            sanitized_data = validation_manager.sanitize_input({"action_type": request.action_type})
            sanitized_action_type = sanitized_data["action_type"]
            
            # Validate action_type is not empty after sanitization
            if not sanitized_action_type or not sanitized_action_type.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Action type cannot be empty or whitespace only"
                )
        
        # Sanitize tags
        sanitized_tags = None
        if request.tags:
            sanitized_data = validation_manager.sanitize_input({"tags": request.tags})
            sanitized_tags = sanitized_data["tags"]
            
            # Validate tags are not empty after sanitization
            for i, tag in enumerate(sanitized_tags):
                if not tag or not tag.strip():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Tag at index {i} cannot be empty or whitespace only"
                    )
        
        # Parse datetime strings if provided
        start_time = None
        end_time = None
        
        if request.start_time:
            try:
                # Handle various ISO format variations
                time_str = request.start_time
                # If it has both +00:00 and Z, remove the Z
                if '+00:00Z' in time_str or '-00:00Z' in time_str:
                    time_str = time_str.rstrip('Z')
                # If it only has Z, replace with +00:00
                elif time_str.endswith('Z'):
                    time_str = time_str[:-1] + '+00:00'
                start_time = datetime.fromisoformat(time_str)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid start_time format: {request.start_time}. Use ISO format."
                )
        
        if request.end_time:
            try:
                # Handle various ISO format variations
                time_str = request.end_time
                # If it has both +00:00 and Z, remove the Z
                if '+00:00Z' in time_str or '-00:00Z' in time_str:
                    time_str = time_str.rstrip('Z')
                # If it only has Z, replace with +00:00
                elif time_str.endswith('Z'):
                    time_str = time_str[:-1] + '+00:00'
                end_time = datetime.fromisoformat(time_str)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid end_time format: {request.end_time}. Use ISO format."
                )
        
        # Query entries with sanitized inputs
        entries = manager.query_memories(
            start_time=start_time,
            end_time=end_time,
            tags=sanitized_tags,
            action_type=sanitized_action_type,
            limit=request.limit,
            offset=request.offset
        )
        
        # Convert entries to response models
        entry_responses = [
            MemoryEntryResponse(
                id=entry.id,
                timestamp=entry.timestamp.isoformat().replace('+00:00', 'Z'),
                action=entry.action,
                context=entry.context,
                sensitivity=entry.sensitivity.value,
                device_id=entry.device_id,
                sync_status=entry.sync_status.value,
                tags=entry.tags,
                summary=entry.summary,
                parent_id=entry.parent_id,
                created_at=entry.created_at.isoformat().replace('+00:00', 'Z') if entry.created_at else None,
                updated_at=entry.updated_at.isoformat().replace('+00:00', 'Z') if entry.updated_at else None
            )
            for entry in entries
        ]
        
        return QueryMemoryResponse(
            entries=entry_responses,
            total=len(entry_responses),
            limit=request.limit,
            offset=request.offset
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except StorageError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
