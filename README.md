# Luma Memory Module

Central memory system for storing user actions, context, and providing retrieval APIs for the Luma personal AI system.

## Overview

The Luma Memory Module is a Python-based core component that provides persistent memory capabilities for lightweight agents running on laptop and phone devices. It implements a three-layer architecture with local-first storage, context summarization, and encryption support.

**Key Features:**
- Local SQLite-based persistent storage
- REST API for agent communication
- AES-256 encryption for sensitive data
- Automatic context summarization to reduce storage overhead
- LRU caching for fast retrieval
- Configurable retention policies
- Performance monitoring and metrics

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/luma/luma-memory.git
cd luma-memory
```

2. Install the package:
```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

3. Create configuration file:
```bash
cp .env.example .env
```

4. Start the API server:
```bash
luma-memory-server
```

The API server will start on `http://localhost:8000` by default.

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Storage Settings
LUMA_DB_PATH=./data/luma_memory.db
LUMA_CACHE_SIZE=1000
LUMA_MAX_STORAGE_SIZE_MB=1000

# Summarization Settings
LUMA_SUMMARIZATION_THRESHOLD=1000
LUMA_SIMILARITY_THRESHOLD=0.8
LUMA_RETENTION_DAYS_RAW=30
LUMA_RETENTION_DAYS_SUMMARY=365

# Encryption Settings
LUMA_ENCRYPTION_KEY_PATH=./keys/encryption.key

# API Settings
LUMA_API_HOST=0.0.0.0
LUMA_API_PORT=8000
LUMA_API_WORKERS=4

# Performance Settings
LUMA_QUERY_TIMEOUT_MS=200
LUMA_CONNECTION_POOL_SIZE=10

# Monitoring Settings
LUMA_ENABLE_METRICS=true
LUMA_LOG_LEVEL=INFO
```

### Configuration File

Alternatively, use a YAML configuration file (see `config.default.yaml` for all options):

```yaml
storage:
  db_path: "./data/luma_memory.db"
  cache_size: 1000

api:
  api_host: "0.0.0.0"
  api_port: 8000
  api_workers: 4
```

## Usage

### Starting the Server

Start the API server with default settings:
```bash
luma-memory-server
```

Start with custom configuration:
```bash
LUMA_API_PORT=9000 luma-memory-server
```

### API Endpoints

The module exposes a REST API for memory operations:

- `POST /api/v1/memory` - Create a new memory entry
- `GET /api/v1/memory/{entry_id}` - Retrieve a specific memory entry
- `POST /api/v1/memory/query` - Query memory entries with filters
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/stats` - Get storage and performance statistics

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed API documentation.

### Python Client Example

```python
import requests

# Create a memory entry
response = requests.post(
    "http://localhost:8000/api/v1/memory",
    json={
        "action": "User opened document",
        "context": {
            "file": "report.pdf",
            "page": 1
        },
        "device_id": "laptop-001",
        "sensitivity": "private",
        "tags": ["document", "work"]
    }
)
entry_id = response.json()["entry_id"]

# Retrieve the entry
response = requests.get(f"http://localhost:8000/api/v1/memory/{entry_id}")
entry = response.json()

# Query entries
response = requests.post(
    "http://localhost:8000/api/v1/memory/query",
    json={
        "tags": ["work"],
        "limit": 50
    }
)
entries = response.json()["entries"]
```

### Using as a Library

```python
from luma_memory import MemoryManager, SQLiteStorage, MemoryModuleConfig
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager

# Initialize components
config = MemoryModuleConfig()
storage = SQLiteStorage(config.db_path, config.cache_size)
encryption = EncryptionService(config.encryption_key_path)
validation = ValidationManager()

# Create memory manager
manager = MemoryManager(
    storage=storage,
    encryption_service=encryption,
    validation_manager=validation,
    config=config
)

# Create a memory entry
entry_id = manager.create_memory(
    action="User opened document",
    context={"file": "report.pdf"},
    device_id="laptop-001",
    sensitivity="private",
    tags=["document", "work"]
)

# Query memories
entries = manager.query_memories(
    tags=["work"],
    limit=50
)
```

## Plugin System

The Luma Memory Module supports a flexible plugin system for extending functionality with custom entry types. Plugins can add validation, processing, and metadata handling for specific action types.

### Loading Plugins

```python
from luma_memory.plugins.plugin_loader import load_plugins_from_directory

# Load all plugins from a directory
plugins = load_plugins_from_directory("path/to/plugins")

# Plugins are automatically registered and ready to use
```

### Creating a Plugin

```python
from luma_memory.plugins.plugin_interface import MemoryEntryPlugin
from typing import List, Dict, Any, Optional

class MyPlugin(MemoryEntryPlugin):
    @property
    def name(self) -> str:
        return "my_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["custom_action"]
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        # Add custom validation logic
        if "required_field" not in context:
            return False, "Missing required_field"
        return True, None
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        # Enrich entry before storage
        entry.context["processed"] = True
        return entry
```

For detailed plugin development instructions, see the [Plugin Development Guide](docs/PLUGIN_DEVELOPMENT.md).

## Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/luma/luma-memory.git
cd luma-memory
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest
```

4. Run tests with coverage:
```bash
pytest --cov=luma_memory --cov-report=html
```

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_memory_manager.py
```

Run with verbose output:
```bash
pytest -v
```

Run with coverage report:
```bash
pytest --cov=luma_memory --cov-report=term-missing
```

### Project Structure

```
luma-memory/
├── luma_memory/           # Main package
│   ├── api/              # REST API layer
│   │   ├── routes.py     # API endpoints
│   │   └── server.py     # Server initialization
│   ├── storage/          # Storage backends
│   │   ├── backend.py    # Abstract interface
│   │   ├── sqlite_storage.py
│   │   └── memory_storage.py
│   ├── processing/       # Processing layer
│   │   ├── encryption.py
│   │   ├── summarizer.py
│   │   └── validation.py
│   ├── models.py         # Data models
│   ├── config.py         # Configuration
│   └── memory_manager.py # Core manager
├── tests/                # Test suite
├── examples/             # Usage examples
├── docs/                 # Documentation
├── data/                 # Database files (created at runtime)
├── keys/                 # Encryption keys (created at runtime)
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lightweight Agents                        │
│                  (Laptop Agent, Phone Agent)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     REST API Layer                           │
│                  (FastAPI Endpoints)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Context    │  │  Encryption  │  │  Validation  │      │
│  │ Summarizer   │  │   Service    │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Memory     │  │    SQLite    │  │  LRU Cache   │      │
│  │   Manager    │  │   Database   │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

- **REST API Layer**: Exposes HTTP endpoints for memory operations
- **Context Summarizer**: Identifies redundant context and consolidates entries
- **Encryption Service**: Handles AES-256 encryption/decryption for sensitive data
- **Validation Manager**: Validates memory entry fields and enforces data integrity
- **Memory Manager**: Coordinates storage operations and manages transactions
- **SQLite Database**: Persistent storage backend with ACID guarantees
- **LRU Cache**: In-memory cache for frequently accessed entries

## Performance

The Luma Memory Module is designed for lightweight operation:

- **Store operations**: < 100ms for typical entry sizes
- **Retrieve operations**: < 200ms for queries returning up to 100 entries
- **Memory usage**: < 100MB during normal operation
- **Concurrent requests**: Supports multiple workers for parallel processing

## Security

### Data Privacy

- **Local-first storage**: All data stored locally by default
- **No cloud dependencies**: Operates without external services
- **Encryption at rest**: AES-256 encryption for sensitive data
- **Input validation**: All input sanitized to prevent injection attacks

### Encryption

Sensitive data (marked with `sensitivity: "sensitive"`) is automatically encrypted using AES-256:

```python
# Sensitive data is encrypted automatically
entry_id = manager.create_memory(
    action="User entered password",
    context={"service": "banking"},
    device_id="laptop-001",
    sensitivity="sensitive"  # Triggers encryption
)
```

### Key Management

Encryption keys are stored in the `keys/` directory and generated automatically on first use. Keep these keys secure and backed up.

**Important**: If you lose the encryption key, encrypted data cannot be recovered.

## Troubleshooting

### Common Issues

#### Database Issues

**Issue: "Database is locked"**
- **Cause**: Concurrent write operations or stale database connections
- **Solution**: 
  - Retry the operation after a short delay (100-500ms)
  - Check if another process is accessing the database
  - Increase `LUMA_CONNECTION_POOL_SIZE` in configuration
  - Ensure proper connection cleanup in your code

**Issue: "Permission denied" when accessing database**
- **Cause**: Insufficient file permissions on database file or directory
- **Solution**: 
  - Ensure the `data/` directory exists and is writable: `chmod 755 data/`
  - Check database file permissions: `chmod 644 data/luma_memory.db`
  - Verify the user running the server has write access
  - On Windows, check folder security settings

**Issue: "Database disk image is malformed"**
- **Cause**: Database corruption due to improper shutdown or disk issues
- **Solution**:
  - Stop the server gracefully (CTRL+C or SIGTERM)
  - Restore from backup if available
  - Use SQLite recovery tools: `sqlite3 luma_memory.db ".recover" | sqlite3 recovered.db`
  - Check disk health and available space

#### Encryption Issues

**Issue: "Encryption key not found"**
- **Cause**: Missing encryption key file
- **Solution**: 
  - Ensure `keys/encryption.key` exists or configure `LUMA_ENCRYPTION_KEY_PATH`
  - Create the keys directory: `mkdir -p keys`
  - The key will be auto-generated on first server start
  - Check file path is absolute or relative to working directory

**Issue: "Invalid encryption key"**
- **Cause**: Corrupted or incorrectly formatted key file
- **Solution**:
  - Backup existing key file
  - Delete corrupted key and restart server to generate new one
  - **Warning**: Existing encrypted data cannot be decrypted with new key
  - Restore from backup if you have encrypted data

**Issue: "Decryption failed"**
- **Cause**: Data encrypted with different key or data corruption
- **Solution**:
  - Verify correct encryption key is being used
  - Check if key rotation occurred without re-encrypting data
  - Restore from backup if data is corrupted

#### API Server Issues

**Issue: "Service starting up, please retry"**
- **Cause**: Server is initializing database and components
- **Solution**: 
  - Wait 2-5 seconds and retry the request
  - Check server logs for initialization errors
  - Verify database is accessible and not corrupted

**Issue: "Address already in use"**
- **Cause**: Port is already bound by another process
- **Solution**:
  - Change port: `LUMA_API_PORT=9000 luma-memory-server`
  - Find and stop conflicting process: `lsof -i :8000` (Linux/Mac) or `netstat -ano | findstr :8000` (Windows)
  - Kill the process or use a different port

**Issue: "Connection refused"**
- **Cause**: Server not running or firewall blocking connection
- **Solution**:
  - Verify server is running: `curl http://localhost:8000/api/v1/health`
  - Check firewall settings
  - Ensure correct host/port configuration
  - Try binding to `0.0.0.0` instead of `localhost`

**Issue: "Request timeout"**
- **Cause**: Query taking too long or server overloaded
- **Solution**:
  - Reduce query result limit
  - Add more specific filters to queries
  - Increase `LUMA_QUERY_TIMEOUT_MS` setting
  - Check database indexes are created
  - Monitor server resource usage

#### Performance Issues

**Issue: Slow query performance**
- **Cause**: Missing indexes, large dataset, or inefficient queries
- **Solution**:
  - Verify database indexes exist: Check `idx_timestamp`, `idx_device_id`, `idx_sync_status`, `idx_tags`
  - Use more specific query filters (time range, tags, action type)
  - Reduce query limit to smaller batches
  - Enable caching: Increase `LUMA_CACHE_SIZE`
  - Run `VACUUM` on database to optimize: `sqlite3 luma_memory.db "VACUUM;"`

**Issue: High memory usage**
- **Cause**: Large cache size or memory leaks
- **Solution**:
  - Reduce `LUMA_CACHE_SIZE` setting
  - Monitor memory with: `ps aux | grep luma-memory-server`
  - Restart server periodically if memory grows unbounded
  - Check for large context objects in memory entries

**Issue: Slow storage operations**
- **Cause**: Disk I/O bottleneck or large entries
- **Solution**:
  - Use SSD instead of HDD for database storage
  - Reduce entry context size
  - Enable automatic summarization to consolidate entries
  - Check disk space and health

#### Configuration Issues

**Issue: "Configuration validation error"**
- **Cause**: Invalid configuration values
- **Solution**:
  - Check `.env` file syntax (no spaces around `=`)
  - Verify all paths are valid and accessible
  - Ensure numeric values are within valid ranges
  - Review `config.default.yaml` for valid options
  - Check for typos in environment variable names (must start with `LUMA_`)

**Issue: Environment variables not loaded**
- **Cause**: `.env` file not found or not in correct location
- **Solution**:
  - Ensure `.env` file is in the working directory where server starts
  - Use absolute path for configuration file
  - Export variables manually: `export LUMA_API_PORT=9000`
  - Verify file is named `.env` not `env` or `.env.txt`

#### Installation Issues

**Issue: "Module not found" errors**
- **Cause**: Package not installed or incorrect Python environment
- **Solution**:
  - Reinstall package: `pip install -e .`
  - Verify correct Python environment is activated
  - Check Python version: `python --version` (requires 3.8+)
  - Install dependencies: `pip install -r requirements.txt`

**Issue: "Permission denied" during installation**
- **Cause**: Installing to system Python without sudo
- **Solution**:
  - Use virtual environment: `python -m venv venv && source venv/bin/activate`
  - Install with user flag: `pip install --user -e .`
  - Use sudo only if necessary: `sudo pip install -e .`

### Debug Logging

Enable debug logging for detailed troubleshooting:

```bash
# Set log level to DEBUG
LUMA_LOG_LEVEL=DEBUG luma-memory-server
```

Log levels available:
- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages for potentially harmful situations
- `ERROR`: Error messages for serious problems

View logs in real-time:
```bash
# Linux/Mac
tail -f /var/log/luma-memory.log

# Or redirect to file
luma-memory-server > luma-memory.log 2>&1
```

### Health Check

Check if the server is running and healthy:

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

If health check fails:
1. Verify server is running: `ps aux | grep luma-memory-server`
2. Check server logs for errors
3. Verify port is correct
4. Test with verbose curl: `curl -v http://localhost:8000/api/v1/health`

### Testing Database Connection

Test database connectivity directly:

```bash
# Open database with SQLite CLI
sqlite3 data/luma_memory.db

# Run test query
sqlite> SELECT COUNT(*) FROM memory_entries;
sqlite> .exit
```

### Verifying Installation

Run the test suite to verify installation:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test category
pytest tests/test_memory_manager.py -v
```

### Getting Help

If you're still experiencing issues:

1. **Check logs**: Review server logs for error messages and stack traces
2. **Search issues**: Check [GitHub Issues](https://github.com/luma/luma-memory/issues) for similar problems
3. **Minimal reproduction**: Create a minimal example that reproduces the issue
4. **System information**: Gather OS, Python version, package version
5. **Open an issue**: Provide logs, configuration, and reproduction steps

**When reporting issues, include:**
- Operating system and version
- Python version (`python --version`)
- Package version
- Configuration (sanitize sensitive data)
- Full error message and stack trace
- Steps to reproduce the issue

### Performance Monitoring

Monitor server performance:

```bash
# Check server stats endpoint
curl http://localhost:8000/api/v1/stats

# Monitor system resources
top -p $(pgrep -f luma-memory-server)

# Check database size
du -h data/luma_memory.db
```

### Backup and Recovery

**Backup database:**
```bash
# Stop server first for consistent backup
sqlite3 data/luma_memory.db ".backup data/luma_memory_backup.db"

# Or copy file (server must be stopped)
cp data/luma_memory.db data/luma_memory_backup.db
```

**Restore from backup:**
```bash
# Stop server
# Replace database file
cp data/luma_memory_backup.db data/luma_memory.db
# Restart server
```

**Backup encryption keys:**
```bash
# Critical: Backup encryption keys separately
cp keys/encryption.key keys/encryption.key.backup
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py` - Basic memory operations
- `agent_quickstart.py` - Quick start guide for agents
- `agent_usage_guide.py` - Comprehensive agent integration guide
- `multi_agent_coordination.py` - Multi-agent coordination patterns
- `performance_monitoring_example.py` - Performance monitoring
- `config_summarizer_example.py` - Configuration and summarization

## Documentation

- [API Documentation](API_DOCUMENTATION.md) - Complete API reference
- [Configuration Guide](CONFIG_GUIDE.md) - Detailed configuration options
- [Plugin Development Guide](docs/PLUGIN_DEVELOPMENT.md) - Guide for creating custom plugins
- [Implementation Checklist](IMPLEMENTATION_CHECKLIST.md) - Development checklist

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Test edge cases and error conditions
- Use descriptive test names

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:

- GitHub Issues: https://github.com/luma/luma-memory/issues
- Email: team@luma.ai
- Documentation: https://docs.luma.ai/memory-module

## Changelog

### Version 0.1.0 (Current)

- Initial release
- SQLite storage backend
- REST API with FastAPI
- AES-256 encryption for sensitive data
- Context summarization
- LRU caching
- Performance monitoring
- Comprehensive test suite

## Roadmap

Future features planned:

- Cloud synchronization support
- Multi-device conflict resolution
- Plugin system for custom entry types
- Advanced query capabilities (semantic search)
- Real-time event streaming
- Backup and restore utilities
- Web-based management interface

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - SQL toolkit
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Cryptography](https://cryptography.io/) - Encryption library
