"""
Example demonstrating structured logging in the Luma Memory Module.

This example shows:
1. Setting up structured logging with JSON format
2. Setting up structured logging with human-readable format
3. Using LogContext to add contextual information
4. Logging with extra fields
5. Exception logging with stack traces
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from luma_memory.utils.logging_config import (
    setup_structured_logging,
    LogContext,
    get_logger
)


def example_json_logging():
    """Example of structured JSON logging."""
    print("\n" + "="*80)
    print("Example 1: Structured JSON Logging")
    print("="*80 + "\n")
    
    # Setup structured logging with JSON format
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        include_context=True
    )
    
    logger = logging.getLogger("example.json")
    
    # Basic log message
    logger.info("Application started")
    
    # Log with extra context
    logger.info(
        "Memory entry created",
        extra={
            "entry_id": "abc-123",
            "device_id": "laptop-001",
            "elapsed_ms": 45.2
        }
    )
    
    # Log with context manager
    with LogContext(operation="create_memory", user_id="user-456"):
        logger.info("Starting operation")
        logger.debug("Processing entry")
        logger.info("Operation completed")
    
    # Log an error with exception
    try:
        raise ValueError("Invalid entry ID")
    except ValueError:
        logger.error("Operation failed", exc_info=True)


def example_human_readable_logging():
    """Example of human-readable logging."""
    print("\n" + "="*80)
    print("Example 2: Human-Readable Logging")
    print("="*80 + "\n")
    
    # Setup structured logging with human-readable format
    setup_structured_logging(
        log_level="INFO",
        log_format="human",
        include_context=True
    )
    
    logger = logging.getLogger("example.human")
    
    # Basic log messages
    logger.debug("Debug message (won't show at INFO level)")
    logger.info("Application started")
    logger.warning("Storage size approaching limit")
    logger.error("Failed to connect to database")
    
    # Log with extra context (context won't show in human format)
    logger.info(
        "Memory entry created",
        extra={
            "entry_id": "abc-123",
            "device_id": "laptop-001"
        }
    )


def example_logger_with_default_context():
    """Example of using get_logger with default context."""
    print("\n" + "="*80)
    print("Example 3: Logger with Default Context")
    print("="*80 + "\n")
    
    # Setup structured logging
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        include_context=True
    )
    
    # Create logger with default context
    logger = get_logger(
        "example.context",
        service="memory_module",
        version="1.0.0",
        environment="production"
    )
    
    # All logs will include service, version, and environment
    logger.info("Service started")
    logger.info("Processing request")
    logger.info("Service stopped")


def example_performance_logging():
    """Example of performance monitoring with logging."""
    print("\n" + "="*80)
    print("Example 4: Performance Monitoring")
    print("="*80 + "\n")
    
    import time
    
    # Setup structured logging
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        include_context=True
    )
    
    logger = logging.getLogger("example.performance")
    
    # Simulate an operation with timing
    operation_id = "op-123"
    
    with LogContext(operation="create_memory", operation_id=operation_id):
        start_time = time.time()
        
        logger.info("Operation started")
        
        # Simulate work
        time.sleep(0.05)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "Operation completed",
            extra={"elapsed_ms": round(elapsed_ms, 2)}
        )
        
        # Log warning if operation took too long
        if elapsed_ms > 100:
            logger.warning(
                "Operation exceeded threshold",
                extra={
                    "elapsed_ms": round(elapsed_ms, 2),
                    "threshold_ms": 100
                }
            )


def example_file_logging():
    """Example of logging to a file."""
    print("\n" + "="*80)
    print("Example 5: File Logging")
    print("="*80 + "\n")
    
    import tempfile
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_file = f.name
    
    print(f"Logging to file: {log_file}")
    
    # Setup structured logging with file output
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file=log_file,
        include_context=True
    )
    
    logger = logging.getLogger("example.file")
    
    # Log some messages
    logger.info("Application started")
    logger.info("Processing data", extra={"records": 100})
    logger.warning("Low memory warning")
    logger.info("Application stopped")
    
    # Close file handlers to flush logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    
    # Read and display log file
    print("\nLog file contents:")
    print("-" * 80)
    with open(log_file, 'r') as f:
        print(f.read())
    
    # Clean up
    import os
    try:
        os.unlink(log_file)
    except:
        pass


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Structured Logging Examples for Luma Memory Module")
    print("="*80)
    
    # Run examples
    example_json_logging()
    example_human_readable_logging()
    example_logger_with_default_context()
    example_performance_logging()
    example_file_logging()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
