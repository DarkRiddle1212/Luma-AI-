# Luma Memory Module - Agent Usage Examples

This directory contains comprehensive examples for agent developers integrating with the Luma Memory Module API.

## Overview

The Luma Memory Module provides a REST API for storing and retrieving user actions and context summaries. These examples demonstrate how to build agents (laptop, phone, or other devices) that interact with the Memory Module.

## Quick Start

### Prerequisites

1. **Start the Memory API server:**
   ```bash
   python -m luma_memory.api.server
   ```

2. **Verify the server is running:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Run the Quick Start Example

For a minimal introduction, run:

```bash
python examples/agent_quickstart.py
```

This demonstrates the three core operations:
- Creating a memory entry
- Retrieving a specific entry
- Querying entries with filters

## Comprehensive Examples

### Agent Usage Guide

For production-ready examples with best practices, run:

```bash
python examples/agent_usage_guide.py
```

This interactive guide includes 7 comprehensive examples:

1. **Laptop Agent - Productivity Tracking**
   - Tracking user actions (opening apps, editing files, running tests)
   - Storing rich contextual information
   - Querying recent activities

2. **Phone Agent - Context Awareness**
   - Location tracking
   - Notification handling
   - App usage monitoring

3. **Sensitive Data Handling**
   - Automatic encryption for sensitive data
   - Secure storage of authentication events
   - Transparent decryption on retrieval

4. **Pagination for Large Datasets**
   - Efficiently retrieving large result sets
   - Implementing page-by-page navigation
   - Managing memory usage

5. **Error Handling Best Practices**
   - Validation error handling
   - HTTP error handling (404, 500, 503)
   - Retry logic with exponential backoff

6. **Time-Based Queries**
   - Querying entries from specific time ranges
   - Recent activity queries (last hour, last day)
   - Time-window filtering

7. **Performance Monitoring**
   - Tracking API statistics
   - Measuring operation latency
   - Monitoring storage usage

## Example Files

### For Agent Developers

| File | Description | Use Case |
|------|-------------|----------|
| `agent_quickstart.py` | Minimal quick start example | Getting started quickly |
| `agent_usage_guide.py` | Comprehensive production examples | Building production agents |
| `agent_client.py` | Legacy client implementation | Reference (uses old API) |

### For Module Developers

| File | Description | Use Case |
|------|-------------|----------|
| `basic_usage.py` | Direct module usage (no API) | Testing storage layer |
| `api_server.py` | Server startup example | Running the API server |
| `run_server_example.py` | Server configuration example | Custom server setup |
| `config_summarizer_example.py` | Summarization features | Context summarization |
| `performance_monitoring_example.py` | Performance tracking | Monitoring and optimization |

## API Endpoints

The examples use these core endpoints:

### Health Check
```
GET /api/v1/health
```

### Get Statistics
```
GET /api/v1/stats
```

### Create Memory Entry
```
POST /api/v1/memory
Body: {
  "action": "string",
  "context": {},
  "device_id": "string",
  "sensitivity": "public|private|sensitive",
  "tags": []
}
```

### Get Memory Entry
```
GET /api/v1/memory/{entry_id}
```

### Query Memory Entries
```
POST /api/v1/memory/query
Body: {
  "start_time": "ISO 8601 datetime",
  "end_time": "ISO 8601 datetime",
  "tags": [],
  "action_type": "string",
  "limit": 100,
  "offset": 0
}
```

## Best Practices for Agents

### 1. Use Unique Device IDs

Each agent should have a unique device identifier:

```python
agent = LumaMemoryAgent(device_id="laptop-001")
```

### 2. Choose Appropriate Sensitivity Levels

- **public**: Non-sensitive information (e.g., "User opened weather app")
- **private**: Personal information (e.g., "User opened document: work_report.pdf")
- **sensitive**: Highly sensitive (e.g., "User logged into banking app") - automatically encrypted

### 3. Add Meaningful Tags

Tags improve query performance and organization:

```python
tags=["productivity", "coding", "python"]
```

### 4. Include Rich Context

Store relevant contextual information:

```python
context={
    "file": "report.pdf",
    "page": 1,
    "application": "Adobe Reader",
    "duration_seconds": 120
}
```

### 5. Implement Retry Logic

Handle transient errors with exponential backoff:

```python
for attempt in range(max_retries):
    try:
        return create_memory(...)
    except HTTPError as e:
        if e.response.status_code in [500, 503]:
            time.sleep(2 ** attempt)
            continue
        raise
```

### 6. Validate Input Before Sending

Catch errors early with client-side validation:

```python
if not action or not action.strip():
    raise ValueError("Action cannot be empty")
```

### 7. Use Pagination for Large Queries

Avoid memory issues with pagination:

```python
results = query_memories(limit=50, offset=0)  # First page
results = query_memories(limit=50, offset=50)  # Second page
```

### 8. Handle Errors Gracefully

Different error types require different handling:

- **400 Bad Request**: Fix the request data (don't retry)
- **404 Not Found**: Entry doesn't exist (don't retry)
- **500 Internal Server Error**: Retry with backoff
- **503 Service Unavailable**: Wait and retry

## Common Use Cases

### Laptop Agent: Productivity Tracking

```python
# Track application usage
create_memory(
    action="User opened VS Code",
    context={"workspace": "/projects/luma", "window_count": 1},
    tags=["productivity", "coding"]
)

# Track file editing
create_memory(
    action="User edited file",
    context={"file": "api.py", "lines_changed": 45},
    tags=["coding", "python"]
)

# Query recent coding activities
results = query_memories(
    start_time=datetime.utcnow() - timedelta(hours=1),
    tags=["coding"]
)
```

### Phone Agent: Context Awareness

```python
# Track location changes
create_memory(
    action="User arrived at location",
    context={"location_type": "home", "wifi": "HomeNetwork"},
    sensitivity="private",
    tags=["location", "context"]
)

# Track notifications
create_memory(
    action="User received notification",
    context={"app": "Slack", "channel": "engineering"},
    tags=["notification", "communication"]
)

# Query recent notifications
results = query_memories(tags=["notification"], limit=10)
```

### Smart Home Agent: Device Interactions

```python
# Track device control
create_memory(
    action="User adjusted thermostat",
    context={"device": "nest", "temperature": 72, "mode": "heat"},
    tags=["smart_home", "climate"]
)

# Track automation triggers
create_memory(
    action="Automation triggered",
    context={"rule": "evening_routine", "devices": ["lights", "thermostat"]},
    tags=["automation", "smart_home"]
)
```

## Troubleshooting

### Service Not Available

If you get connection errors:

1. Verify the server is running:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. Check the server logs for errors

3. Ensure the correct port (default: 8000)

### Validation Errors

If you get 400 Bad Request errors:

1. Verify all required fields are present (`action`, `device_id`)
2. Check that `context` is a dictionary
3. Ensure `sensitivity` is one of: `public`, `private`, `sensitive`
4. Verify datetime format is ISO 8601 with Z suffix

### Performance Issues

If operations are slow:

1. Check storage statistics: `GET /api/v1/stats`
2. Use pagination for large queries
3. Add appropriate indexes (handled automatically)
4. Monitor cache hit rates

## Additional Resources

- **API Documentation**: See `API_DOCUMENTATION.md` for complete API reference
- **Design Document**: See `.kiro/specs/luma-memory-module/design.md` for architecture details
- **Requirements**: See `.kiro/specs/luma-memory-module/requirements.md` for specifications

## Support

For issues or questions:

1. Check the API documentation
2. Review the comprehensive examples in `agent_usage_guide.py`
3. Enable debug logging: `LUMA_LOG_LEVEL=DEBUG`
4. Check server logs for detailed error information

## License

See the main project LICENSE file for licensing information.
