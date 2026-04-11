"""
API server initialization for Luma Memory Module.

This module provides server initialization, configuration, and lifecycle management
for the FastAPI application.
"""

import logging
import sys
import signal
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from luma_memory.config import MemoryModuleConfig
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.api.routes import set_memory_manager


# Import structured logging configuration
from luma_memory.utils.logging_config import setup_structured_logging


# Configure logging (wrapper for backward compatibility)
def setup_logging(
    log_level: str = "INFO",
    log_format: str = "human",
    log_file: Optional[str] = None,
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_count: int = 5
) -> None:
    """
    Configure logging for the application with structured formatting.
    
    Sets up logging with:
    - Structured JSON format or human-readable format
    - Console output to stdout
    - Optional file output with rotation
    - Configurable log level
    - Proper formatting for exceptions and stack traces
    - Context propagation for request tracing
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" for structured JSON, "human" for readable)
        log_file: Optional path to log file for persistent logging
        log_max_bytes: Maximum size of log file before rotation (default: 10MB)
        log_backup_count: Number of backup log files to keep (default: 5)
    """
    setup_structured_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        include_context=True,
        max_bytes=log_max_bytes,
        backup_count=log_backup_count
    )


logger = logging.getLogger(__name__)

# Global reference to memory manager for cleanup
_memory_manager: Optional[MemoryManager] = None
_shutdown_requested = False


def handle_shutdown_signal(signum, frame):
    """
    Handle shutdown signals (SIGTERM, SIGINT) for graceful shutdown.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global _shutdown_requested
    
    if _shutdown_requested:
        logger.warning("Shutdown already in progress, ignoring signal")
        return
    
    _shutdown_requested = True
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
    
    # The actual cleanup will be handled by the lifespan context manager
    # This just logs the signal reception


def initialize_memory_manager(config: MemoryModuleConfig) -> MemoryManager:
    """
    Initialize the memory manager with all required components.
    
    Args:
        config: Configuration settings
    
    Returns:
        Initialized MemoryManager instance
    
    Raises:
        Exception: If initialization fails
    """
    logger.info("Initializing memory manager components...")
    
    # Ensure data directory exists
    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database path: {db_path}")
    
    # Ensure encryption key directory exists
    key_path = Path(config.encryption_key_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Encryption key path: {key_path}")
    
    # Initialize storage backend
    logger.info("Initializing SQLite storage backend...")
    storage = SQLiteStorage(
        db_path=str(db_path),
        cache_size=config.cache_size
    )
    
    # Initialize encryption service
    logger.info("Initializing encryption service...")
    encryption = EncryptionService(key_path=str(key_path))
    
    # Initialize validation manager
    logger.info("Initializing validation manager...")
    validation = ValidationManager()
    
    # Initialize context summarizer
    logger.info("Initializing context summarizer...")
    summarizer = ContextSummarizer(
        similarity_threshold=config.similarity_threshold
    )
    
    # Initialize memory manager
    logger.info("Initializing memory manager...")
    manager = MemoryManager(
        storage=storage,
        encryption=encryption,
        validation=validation,
        summarizer=summarizer,
        config=config
    )
    
    logger.info("Memory manager initialized successfully")
    return manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    global _memory_manager
    
    # Startup
    logger.info("Starting Luma Memory Module API server...")
    
    try:
        # Load configuration
        config = MemoryModuleConfig.load_config()
        logger.info(f"Configuration loaded: {config.model_dump()}")
        
        # Initialize memory manager
        manager = initialize_memory_manager(config)
        _memory_manager = manager
        
        # Set global memory manager in routes
        set_memory_manager(manager)
        
        # Perform health check
        stats = manager.get_stats()
        logger.info(f"Server health check passed. Total entries: {stats.get('total_entries', 0)}")
        
        logger.info(f"Server started successfully on {config.api_host}:{config.api_port}")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Luma Memory Module API server...")
    
    try:
        # Cleanup memory manager resources
        if _memory_manager is not None:
            logger.info("Closing memory manager resources...")
            
            # Close storage backend connections
            if hasattr(_memory_manager.storage, 'close'):
                logger.info("Closing storage backend connections...")
                _memory_manager.storage.close()
                logger.info("Storage backend closed successfully")
            
            # Clear cache if applicable
            if hasattr(_memory_manager.storage, 'cache'):
                logger.info("Clearing storage cache...")
                _memory_manager.storage.cache.clear()
                logger.info("Cache cleared")
            
            _memory_manager = None
            logger.info("Memory manager cleanup completed")
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        # Don't re-raise - we want shutdown to complete even if cleanup fails
    
    logger.info("Server shutdown complete")


def create_app(config: Optional[MemoryModuleConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application with lifespan events.
    
    Args:
        config: Optional configuration. If not provided, loads from environment.
    
    Returns:
        Configured FastAPI application with startup/shutdown events
    """
    if config is None:
        config = MemoryModuleConfig.load_config()
    
    # Setup structured logging with configuration
    setup_logging(
        log_level=config.log_level,
        log_format=config.log_format,
        log_file=config.log_file,
        log_max_bytes=config.log_max_bytes,
        log_backup_count=config.log_backup_count
    )
    
    # OpenAPI metadata
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
    
    # Create new app with lifespan for startup/shutdown events
    configured_app = FastAPI(
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
        lifespan=lifespan
    )
    
    # Import and include routes
    from luma_memory.api.routes import (
        health_check, get_stats, get_metrics, create_memory, get_memory, query_memories
    )
    
    # Register all endpoints
    configured_app.get("/api/v1/health")(health_check)
    configured_app.get("/api/v1/stats")(get_stats)
    configured_app.get("/api/v1/metrics")(get_metrics)
    configured_app.post("/api/v1/memory")(create_memory)
    configured_app.get("/api/v1/memory/{entry_id}")(get_memory)
    configured_app.post("/api/v1/memory/query")(query_memories)
    
    return configured_app


# Create module-level app instance with lifespan for uvicorn
# This is used when running with: uvicorn luma_memory.api.server:app
app = create_app()


def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    log_level: Optional[str] = None,
    reload: bool = False
) -> None:
    """
    Run the API server with uvicorn.
    
    Args:
        host: Host to bind to (overrides config)
        port: Port to bind to (overrides config)
        workers: Number of worker processes (overrides config)
        log_level: Logging level (overrides config)
        reload: Enable auto-reload for development (default: False)
    
    Note:
        - For production, use multiple workers for better performance
        - Workers > 1 enables process-based parallelism
        - Reload mode is incompatible with workers > 1
        - For production deployment, consider using uvicorn directly with
          a process manager like systemd or supervisord
        - Graceful shutdown is handled via signal handlers (SIGTERM, SIGINT)
    """
    import uvicorn
    
    # Load configuration
    config = MemoryModuleConfig.load_config()
    
    # Override with provided values
    host = host or config.api_host
    port = port or config.api_port
    workers = workers or config.api_workers
    log_level = (log_level or config.log_level).lower()
    
    # Setup logging with configuration
    setup_logging(
        log_level=config.log_level,
        log_format=config.log_format,
        log_file=config.log_file,
        log_max_bytes=config.log_max_bytes,
        log_backup_count=config.log_backup_count
    )
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.info("Registered signal handlers for graceful shutdown (SIGTERM, SIGINT)")
    
    # Validate worker configuration
    if reload and workers > 1:
        logger.warning(
            "Reload mode is incompatible with multiple workers. "
            "Setting workers to 1 for development mode."
        )
        workers = 1
    
    logger.info(f"Starting server with {workers} worker(s) on {host}:{port}")
    if workers > 1:
        logger.info(
            f"Multi-worker mode enabled: {workers} processes will handle requests"
        )
    
    # Run server with uvicorn
    # Using module path string allows workers to spawn correctly
    uvicorn.run(
        "luma_memory.api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        reload=reload,
        # Additional production-ready settings
        timeout_keep_alive=5,
        limit_concurrency=1000,
        limit_max_requests=10000
    )


if __name__ == "__main__":
    run_server()
