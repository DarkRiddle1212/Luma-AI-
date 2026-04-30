# Luma Storage Module

Persistence layer for Luma — provides SQLAlchemy-backed repositories for memories, insights, user profiles, and learning progress. Supports SQLite (development) and PostgreSQL (production).

## Configuration

All settings use the `LUMA_STORAGE_` prefix and can be set via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `LUMA_STORAGE_DATABASE_URL` | `sqlite:///luma.db` | SQLAlchemy database URL |
| `LUMA_STORAGE_ENVIRONMENT` | `development` | `development` or `production` |
| `LUMA_STORAGE_POOL_SIZE` | `5` | Connection pool size (ignored for SQLite) |
| `LUMA_STORAGE_MAX_OVERFLOW` | `10` | Max overflow connections |
| `LUMA_STORAGE_POOL_TIMEOUT` | `30` | Pool checkout timeout (seconds) |
| `LUMA_STORAGE_ECHO_SQL` | `false` | Log all SQL statements |

> Note: Setting `LUMA_STORAGE_ENVIRONMENT=production` with a SQLite URL raises a `StorageConfigurationError` at startup.

## Migrating from SQLite to PostgreSQL

### Step 1: Update the database URL

```bash
export LUMA_STORAGE_DATABASE_URL="postgresql://user:password@host:5432/luma"
```

### Step 2: Set the environment to production

```bash
export LUMA_STORAGE_ENVIRONMENT=production
```

### Step 3: Run migrations

```python
from luma.storage import DatabaseManager, MigrationRunner, StorageConfig

config = StorageConfig()
db = DatabaseManager(config)
runner = MigrationRunner(db)
runner.run_pending()
print(f"Schema version: {runner.get_current_version()}")
```

### Step 4: Verify the schema

After migrations complete, the following tables should exist:

- `schema_version` — tracks applied migration versions
- `memories` — user memory records
- `insights` — generated insight records
- `user_profiles` — personalization profiles
- `learning_progress` — per-user, per-topic progress

You can verify with `psql`:

```sql
\dt
SELECT version FROM schema_version;
```

## Module Structure

```
luma/storage/
├── __init__.py              # Public API — all exports live here
├── config.py                # StorageConfig (Pydantic BaseSettings)
├── database.py              # DatabaseManager + SQLAlchemy Base
├── models.py                # ORM models (never returned outside this module)
├── migrations/
│   ├── __init__.py          # MigrationRunner
│   └── v001_initial_schema.py
└── repositories/
    ├── memory_repository.py
    ├── insight_repository.py
    ├── personalization_repository.py
    └── teacher_repository.py
```

All repositories accept a `Session` and never call `session.commit()` — transaction control is handled by `DatabaseManager.get_session()`.
