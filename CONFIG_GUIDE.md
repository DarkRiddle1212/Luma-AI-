# Configuration Guide

## Overview

The Luma Memory Module uses a flexible configuration system that supports multiple configuration sources with clear precedence rules.

## Configuration Sources (Priority Order)

1. **Environment Variables** (Highest Priority)
2. **.env File** (Medium Priority)
3. **Default Values** (Lowest Priority)

## Default Configuration Files

The project includes two default configuration files for reference:

- `config.default.json` - JSON format with schema and descriptions
- `config.default.yaml` - YAML format with inline comments (recommended for readability)

**Note:** These files are for reference only. The application does not load them directly. They document the available settings and their default values.

## Quick Start

### Option 1: Use Defaults (Simplest)

No configuration needed! The application will use sensible defaults:

```python
from luma_memory.config import MemoryModuleConfig

config = MemoryModuleConfig()
# Uses all default values
```

### Option 2: Environment Variables

Set environment variables to override specific settings:

```bash
# Windows (CMD)
set DB_PATH=C:\data\luma.db
set API_PORT=9000
set LOG_LEVEL=DEBUG

# Windows (PowerShell)
$env:DB_PATH="C:\data\luma.db"
$env:API_PORT="9000"
$env:LOG_LEVEL="DEBUG"

# Linux/Mac
export DB_PATH=/data/luma.db
export API_PORT=9000
export LOG_LEVEL=DEBUG
```

Then run your application:

```python
from luma_memory.config import MemoryModuleConfig

config = MemoryModuleConfig()
# Loads from environment variables
```

### Option 3: .env File (Recommended)

1. Copy the example file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   DB_PATH=./data/luma_memory.db
   API_PORT=8000
   LOG_LEVEL=INFO
   ```

3. Load configuration:
   ```python
   from luma_memory.config import MemoryModuleConfig
   
   config = MemoryModuleConfig()
   # Automatically loads from .env file
   ```

### Option 4: Custom .env File

Use a custom configuration file for different environments:

```python
from luma_memory.config import MemoryModuleConfig

# Load production config
config = MemoryModuleConfig.load_config(env_file=".env.production")

# Load development config
config = MemoryModuleConfig.load_config(env_file=".env.development")
```

## Configuration Settings

### Storage Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DB_PATH` | `./data/luma_memory.db` | Path to SQLite database file |
| `CACHE_SIZE` | `1000` | Number of entries to cache in memory |
| `MAX_STORAGE_SIZE_MB` | `1000` | Maximum storage size before warning |

### Summarization Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `SUMMARIZATION_THRESHOLD` | `1000` | Entries before triggering summarization |
| `SIMILARITY_THRESHOLD` | `0.8` | Similarity threshold (0.0-1.0) |
| `RETENTION_DAYS_RAW` | `30` | Days to retain raw entries |
| `RETENTION_DAYS_SUMMARY` | `365` | Days to retain summaries |

### Encryption Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `ENCRYPTION_KEY_PATH` | `./keys/encryption.key` | Path to encryption key file |

### API Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server host address |
| `API_PORT` | `8000` | API server port |
| `API_WORKERS` | `4` | Number of worker processes |

### Performance Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `QUERY_TIMEOUT_MS` | `200` | Query timeout in milliseconds |
| `CONNECTION_POOL_SIZE` | `10` | Database connection pool size |

### Monitoring Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_METRICS` | `true` | Enable metrics collection |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | `human` | Logging format (json, human) |
| `LOG_FILE` | `None` | Optional path to log file |
| `LOG_MAX_BYTES` | `10485760` | Max log file size before rotation (10MB) |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files to keep |

## Examples

### Development Configuration

```bash
# .env.development
DB_PATH=./dev_data/luma.db
LOG_LEVEL=DEBUG
ENABLE_METRICS=true
API_HOST=127.0.0.1
API_PORT=8000
```

### Production Configuration

```bash
# .env.production
DB_PATH=/var/lib/luma/memory.db
LOG_LEVEL=WARNING
ENABLE_METRICS=true
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
CONNECTION_POOL_SIZE=20
```

### Testing Configuration

```bash
# .env.test
DB_PATH=:memory:
LOG_LEVEL=ERROR
ENABLE_METRICS=false
```

## Validation

The configuration system automatically validates settings. Invalid values will raise clear errors:

```python
from luma_memory.config import MemoryModuleConfig

try:
    config = MemoryModuleConfig()
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Accessing Configuration Values

```python
from luma_memory.config import MemoryModuleConfig

config = MemoryModuleConfig()

# Access settings
print(f"Database path: {config.db_path}")
print(f"API port: {config.api_port}")
print(f"Log level: {config.log_level}")

# Check if .env file was loaded
if config.is_env_loaded():
    print("Configuration loaded from .env file")
else:
    print("Using default configuration")
```

## Best Practices

1. **Never commit .env files** - Add `.env` to `.gitignore`
2. **Use .env.example as template** - Document all available settings
3. **Use environment variables in production** - More secure than files
4. **Keep encryption keys secure** - Back up and protect key files
5. **Use absolute paths in production** - Avoid relative path issues
6. **Monitor storage size** - Adjust retention policies as needed
7. **Tune performance settings** - Based on available resources

## Troubleshooting

### Configuration not loading

1. Check that `.env` file exists in the working directory
2. Verify environment variable names are uppercase
3. Check for syntax errors in `.env` file

### Invalid values

1. Check data types (numbers should not have quotes in .env)
2. Verify boolean values are `true` or `false` (lowercase)
3. Check file paths are valid for your operating system

### Performance issues

1. Increase `CACHE_SIZE` for better read performance
2. Increase `CONNECTION_POOL_SIZE` for more concurrent operations
3. Adjust `QUERY_TIMEOUT_MS` if queries are slow

## Reference Files

- `config.default.json` - JSON format with schema
- `config.default.yaml` - YAML format with comments
- `.env.example` - Environment variable template
- `luma_memory/config.py` - Configuration implementation
