# Structured Logging Guide

## Overview

The Luma Memory Module uses structured logging to provide consistent, searchable, and machine-readable log output. Structured logging formats log messages as JSON objects with contextual information, making it easier to analyze logs in production environments.

## Features

- **JSON Format**: Logs are formatted as JSON objects for easy parsing and analysis
- **Human-Readable Format**: Alternative format for development and console output
- **Contextual Information**: Automatically includes operation context, request IDs, and custom fields
- **Exception Tracking**: Properly formats exceptions with stack traces
- **Performance Monitoring**: Includes timing information for operations
- **Configurable Levels**: Support for DEBUG, INFO, WARNING, ERROR, and CRITICAL levels
- **File Output**: Optional persistent logging to files

## Configuration

### Environment Variables

Configure logging through environment variables or `.env` file:

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Logging format (json for structured JSON, human for readable)
LOG_FORMAT=json

# Optional path to log file for persistent logging
LOG_FILE=./logs/luma_memory.log
```

### Programmatic Configuration

```python
from luma_memory.utils.logging_config import setup_structured_logging

# Setup with JSON format
setup_structured_logging(
    log_level="INFO",
    log_format="json",
    log_file="./logs/app.log",
    include_context=True
)
```

## Log Formats

### JSON Format (Structured)

Recommended for production environments. Each log entry is a JSON object:

```json
{
  "timestamp": "2024-02-14T10:30:45.123456",
  "level": "INFO",
  "logger": "luma_memory.memory_manager",
  "message": "Created MemoryEntry with id=abc-123",
  "context": {
    "operation": "create_memory",
    "entry_id": "abc-123",
    "elapsed_ms": 45.2,
    "device_id": "laptop-001"
  },
  "source": {
    "file": "/path/to/memory_manager.py",
    "line": 142,
    "function": "create_memory"
  }
}
```

### Human-Readable Format

Recommended for development and console output:

```
2024-02-14 10:30:45 - luma_memory.memory_manager - INFO - Created MemoryEntry with id=abc-123
```

## Adding Context to Logs

### Using LogContext

Add contextual information to all logs within a scope:

```python
from luma_memory.utils.logging_config import LogContext
import logging

logger = logging.getLogger(__name__)

with LogContext(operation="create_memory", entry_id="abc-123", user_id="user-456"):
    logger.info("Starting operation")
    # All logs within this context will include operation, entry_id, and user_id
    logger.debug("Processing entry")
    logger.info("Operation completed")
```

### Using Extra Fields

Add context to individual log messages:

```python
logger.info(
    "Memory entry created",
    extra={
        "entry_id": "abc-123",
        "device_id": "laptop-001",
        "elapsed_ms": 45.2
    }
)
```

### Using get_logger with Default Context

Create a logger with default context for all messages:

```python
from luma_memory.utils.logging_config import get_logger

logger = get_logger(
    __name__,
    service="memory_module",
    version="1.0.0",
    environment="production"
)

# All logs from this logger will include service, version, and environment
logger.info("Service started")
```

## Exception Logging

Structured logging properly formats exceptions with stack traces:

```python
try:
    # Some operation
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
```

JSON output includes exception details:

```json
{
  "timestamp": "2024-02-14T10:30:45.123456",
  "level": "ERROR",
  "logger": "luma_memory.memory_manager",
  "message": "Operation failed",
  "exception": {
    "type": "ValueError",
    "message": "Invalid entry ID",
    "traceback": "Traceback (most recent call last):\n  File ..."
  }
}
```

## Performance Monitoring

The Memory Manager automatically logs performance metrics:

```python
# Automatically logged by MemoryManager
logger.info(
    "create_memory completed in 45.20ms for entry abc-123",
    extra={
        "operation": "create_memory",
        "entry_id": "abc-123",
        "elapsed_ms": 45.2
    }
)

# Warning if operation exceeds threshold
logger.warning(
    "create_memory exceeded 100ms target: 125.50ms for entry abc-123",
    extra={
        "operation": "create_memory",
        "entry_id": "abc-123",
        "elapsed_ms": 125.5,
        "threshold_ms": 100
    }
)
```

## Log Levels

### DEBUG
Detailed information for diagnosing problems. Use for development only.

```python
logger.debug("Cache hit for entry abc-123")
logger.debug("Query parameters: start_time=2024-01-01, limit=100")
```

### INFO
Confirmation that things are working as expected.

```python
logger.info("Memory entry created successfully")
logger.info("Server started on port 8000")
```

### WARNING
Indication that something unexpected happened, but the application is still working.

```python
logger.warning("Storage size approaching limit: 950MB / 1000MB")
logger.warning("Query took longer than expected: 250ms")
```

### ERROR
A serious problem that prevented a function from completing.

```python
logger.error("Failed to create memory entry", exc_info=True)
logger.error("Database connection failed")
```

### CRITICAL
A very serious error that may cause the application to stop.

```python
logger.critical("Database file corrupted, cannot continue")
logger.critical("Out of disk space, cannot write logs")
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# Good
logger.debug("Processing entry with 5 tags")
logger.info("Entry created successfully")
logger.warning("Cache size approaching limit")
logger.error("Failed to encrypt data", exc_info=True)

# Bad
logger.info("Processing entry with 5 tags")  # Too verbose for INFO
logger.error("Entry created successfully")   # Wrong level
```

### 2. Include Relevant Context

```python
# Good
logger.info(
    "Memory entry created",
    extra={
        "entry_id": entry.id,
        "device_id": entry.device_id,
        "elapsed_ms": elapsed_ms
    }
)

# Bad
logger.info("Entry created")  # Missing context
```

### 3. Use Structured Fields

```python
# Good
logger.info(
    "Query completed",
    extra={
        "result_count": len(results),
        "elapsed_ms": elapsed_ms,
        "filters": {"tags": ["work"], "limit": 100}
    }
)

# Bad
logger.info(f"Query returned {len(results)} results in {elapsed_ms}ms")
```

### 4. Log Exceptions Properly

```python
# Good
try:
    result = operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True, extra={"operation": "create"})
    raise

# Bad
try:
    result = operation()
except Exception as e:
    logger.error(f"Error: {e}")  # Missing stack trace
```

### 5. Avoid Logging Sensitive Data

```python
# Good
logger.info(
    "User authenticated",
    extra={"user_id": user.id, "device_id": device.id}
)

# Bad
logger.info(
    "User authenticated",
    extra={"password": user.password, "api_key": user.api_key}
)
```

## Analyzing Logs

### Using jq for JSON Logs

```bash
# Filter by log level
cat logs/app.log | jq 'select(.level == "ERROR")'

# Extract specific fields
cat logs/app.log | jq '{timestamp, message, context}'

# Filter by operation
cat logs/app.log | jq 'select(.context.operation == "create_memory")'

# Calculate average elapsed time
cat logs/app.log | jq -s 'map(select(.context.elapsed_ms)) | map(.context.elapsed_ms) | add / length'

# Find slow operations
cat logs/app.log | jq 'select(.context.elapsed_ms > 100)'
```

### Using grep for Human-Readable Logs

```bash
# Filter by log level
grep "ERROR" logs/app.log

# Filter by logger
grep "luma_memory.memory_manager" logs/app.log

# Filter by message
grep "create_memory" logs/app.log
```

## Integration with Log Aggregation Tools

### Elasticsearch

JSON logs can be directly ingested into Elasticsearch:

```bash
# Using Filebeat
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/luma_memory/*.log
  json.keys_under_root: true
  json.add_error_key: true
```

### Splunk

```bash
# Using Splunk Universal Forwarder
[monitor:///var/log/luma_memory/*.log]
sourcetype = _json
index = luma_memory
```

### CloudWatch Logs

```python
# Using watchtower
import watchtower
import logging

logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler())
```

## Troubleshooting

### Logs Not Appearing

1. Check log level configuration
2. Verify logger name matches module
3. Ensure handlers are properly configured

### Duplicate Logs

1. Check for multiple handler registrations
2. Verify logger propagation settings
3. Clear existing handlers before setup

### Performance Impact

1. Use appropriate log levels (avoid DEBUG in production)
2. Consider async logging for high-throughput scenarios
3. Rotate log files regularly to prevent disk space issues

## Examples

See `examples/` directory for complete examples:

- `examples/basic_usage.py` - Basic logging setup
- `examples/performance_monitoring_example.py` - Performance monitoring with logs
- `examples/multi_agent_coordination.py` - Logging in distributed systems
