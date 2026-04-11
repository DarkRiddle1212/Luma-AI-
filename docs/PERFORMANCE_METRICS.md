# Performance Metrics Collection

This document describes the performance metrics collection system implemented in the Luma Memory Module.

## Overview

The Luma Memory Module includes comprehensive performance metrics collection at multiple layers:

1. **Memory Manager Layer**: Operation-level metrics (latency, throughput, errors)
2. **Storage Layer**: Cache performance and database operation metrics
3. **System Layer**: Resource usage metrics (memory, CPU)

## Memory Manager Metrics

The `MemoryManager` class tracks performance metrics for all operations when `enable_metrics=True` in the configuration.

### Tracked Operations

- `create_memory`: Memory entry creation
- `get_memory`: Memory entry retrieval
- `query_memories`: Memory entry queries
- `update_memory`: Memory entry updates
- `delete_memory`: Memory entry deletion

### Metrics Per Operation

For each operation, the following metrics are collected:

- `count`: Total number of operations
- `avg_time_ms`: Average operation time in milliseconds
- `min_time_ms`: Minimum operation time in milliseconds
- `max_time_ms`: Maximum operation time in milliseconds
- `errors`: Total number of errors
- `error_rate`: Percentage of operations that resulted in errors

### Example Usage

```python
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.config import MemoryModuleConfig

# Initialize with metrics enabled
config = MemoryModuleConfig(enable_metrics=True)
storage = SQLiteStorage("./data/memory.db")
manager = MemoryManager(storage=storage, config=config)

# Perform operations
manager.create_memory(
    action="User action",
    context={"key": "value"},
    device_id="device-001"
)

# Get performance metrics
metrics = manager.get_performance_metrics()
print(f"Average create time: {metrics['create_memory']['avg_time_ms']:.2f}ms")
print(f"Total operations: {metrics['create_memory']['count']}")
print(f"Error rate: {metrics['create_memory']['error_rate']:.2f}%")
```

## Storage Layer Metrics

The `SQLiteStorage` class tracks cache performance and database operation metrics.

### Cache Metrics

- `cache_hits`: Number of successful cache lookups
- `cache_misses`: Number of cache misses requiring database queries
- `cache_hit_rate`: Percentage of cache hits (calculated)

### Database Operation Metrics

- `total_queries`: Total number of database queries
- `total_inserts`: Total number of insert operations
- `total_updates`: Total number of update operations
- `total_deletes`: Total number of delete operations

### Example Usage

```python
from luma_memory.storage.sqlite_storage import SQLiteStorage

storage = SQLiteStorage("./data/memory.db", cache_size=1000)

# Perform operations
entry_id = storage.create_entry(entry)
retrieved = storage.get_entry(entry_id)  # Cache hit

# Get storage statistics including metrics
stats = storage.get_storage_stats()
print(f"Cache hit rate: {stats['storage_metrics']['cache_hit_rate']:.2f}%")
print(f"Total queries: {stats['storage_metrics']['total_queries']}")
print(f"Total inserts: {stats['storage_metrics']['total_inserts']}")
```

## System Resource Metrics

When `psutil` is available, the system automatically collects resource usage metrics:

- `memory_usage_mb`: Memory usage in megabytes
- `memory_usage_percent`: Memory usage as percentage of total
- `cpu_percent`: CPU usage percentage
- `num_threads`: Number of active threads

### Example Usage

```python
metrics = manager.get_performance_metrics()

if 'system_resources' in metrics:
    sys_metrics = metrics['system_resources']
    print(f"Memory usage: {sys_metrics['memory_usage_mb']:.2f} MB")
    print(f"CPU usage: {sys_metrics['cpu_percent']:.2f}%")
    print(f"Threads: {sys_metrics['num_threads']}")
```

## Configuration

Metrics collection is controlled by the `enable_metrics` configuration setting:

```python
# Enable metrics (default)
config = MemoryModuleConfig(enable_metrics=True)

# Disable metrics
config = MemoryModuleConfig(enable_metrics=False)
```

You can also set this via environment variable:

```bash
export ENABLE_METRICS=true
```

## API Integration

Performance metrics are included in the `/api/v1/stats` endpoint response when metrics are enabled:

```json
{
  "total_entries": 1000,
  "storage_size_bytes": 1048576,
  "performance": {
    "create_memory": {
      "count": 1000,
      "avg_time_ms": 45.23,
      "min_time_ms": 12.45,
      "max_time_ms": 98.76,
      "errors": 0,
      "error_rate": 0.0
    },
    "get_memory": {
      "count": 5000,
      "avg_time_ms": 15.67,
      "min_time_ms": 5.23,
      "max_time_ms": 45.89,
      "errors": 0,
      "error_rate": 0.0
    },
    "system_resources": {
      "memory_usage_mb": 85.34,
      "memory_usage_percent": 2.15,
      "cpu_percent": 5.67,
      "num_threads": 12
    }
  },
  "storage_metrics": {
    "cache_hits": 4500,
    "cache_misses": 500,
    "cache_hit_rate": 90.0,
    "total_queries": 500,
    "total_inserts": 1000,
    "total_updates": 100,
    "total_deletes": 50
  }
}
```

## Performance Monitoring Best Practices

1. **Enable metrics in production**: Metrics have minimal overhead and provide valuable insights
2. **Monitor cache hit rate**: Low cache hit rates may indicate need for larger cache size
3. **Track operation latencies**: Ensure operations meet SLA requirements (< 100ms create, < 200ms retrieve)
4. **Monitor error rates**: Non-zero error rates indicate issues requiring investigation
5. **Watch system resources**: High memory or CPU usage may indicate need for optimization

## Resetting Metrics

You can reset performance metrics to start a new monitoring period:

```python
manager.reset_performance_metrics()
```

This is useful for:
- Testing specific scenarios
- Starting fresh monitoring periods
- Isolating performance measurements

## Testing

Comprehensive tests are provided for metrics collection:

- `tests/test_performance_monitoring.py`: Memory manager metrics tests
- `tests/test_storage_metrics.py`: Storage layer metrics tests
- `tests/test_system_metrics.py`: System resource metrics tests

Run tests with:

```bash
pytest tests/test_performance_monitoring.py -v
pytest tests/test_storage_metrics.py -v
pytest tests/test_system_metrics.py -v
```

## Dependencies

- `psutil>=5.9.6`: Required for system resource metrics (optional, gracefully degrades if not available)

## Future Enhancements

Potential future improvements to metrics collection:

1. **Metrics export**: Export metrics to monitoring systems (Prometheus, Datadog, etc.)
2. **Histogram metrics**: Track latency distributions
3. **Custom metrics**: Allow applications to define custom metrics
4. **Alerting**: Automatic alerts when metrics exceed thresholds
5. **Metrics persistence**: Store historical metrics for trend analysis
