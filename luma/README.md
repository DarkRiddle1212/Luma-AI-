# Luma AI System

A local-first modular monolith personal AI system built with Python, FastAPI, SQLite, and SQLAlchemy.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn luma.main:app --reload

# Access the API
# Root: http://localhost:8000
# API docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

## Project Structure

```
luma/
├── main.py                 # Application entry point
├── config.py              # Configuration management
├── core/                  # Core reasoning and scheduling
│   ├── reasoning.py       # Reasoning engine
│   └── scheduler.py       # Task scheduler
├── memory/                # Memory module
│   ├── models.py          # SQLAlchemy models
│   ├── repository.py      # Data access layer
│   └── service.py         # Business logic
├── api/                   # API layer
│   └── routes.py          # FastAPI routes
├── agents/                # Agent system
│   └── laptop_agent.py    # Laptop-specific agent
└── utils/                 # Utilities
    └── logger.py          # Logging configuration
```

## Architecture

### Layered Architecture

The system follows clean architecture principles with clear separation of concerns:

1. **API Layer** (`api/routes.py`)
   - Handles HTTP requests/responses
   - Request validation and serialization
   - Delegates to service layer
   - NO business logic or database access

2. **Service Layer** (`memory/service.py`)
   - Implements business logic
   - Validates data
   - Orchestrates operations
   - Delegates to repository layer

3. **Repository Layer** (`memory/repository.py`)
   - Handles all database operations
   - Does NOT call commit()
   - Session lifecycle controlled by dependency injection

4. **Core Layer** (`core/`)
   - Reasoning engine for decision-making
   - Task scheduler for timed operations
   - Designed for future extensibility

5. **Agent Layer** (`agents/`)
   - Specialized agents for specific tasks
   - Laptop agent for system operations
   - Extensible interface for new agents

### Database Session Management

Critical production safety pattern:

```python
def get_db() -> Generator[Session, None, None]:
    """
    Provides database session with proper lifecycle:
    - Creates session at request start
    - Yields session for use
    - Commits ONLY on successful completion
    - Rolls back on ANY exception
    - Always closes session in finally block
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit only if no exception
    except Exception:
        db.rollback()  # Rollback on error
        raise
    finally:
        db.close()  # Always close
```

Repository methods do NOT commit - all transaction control happens in `get_db()`.

### Dependency Injection

FastAPI's dependency injection wires everything together:

```python
# In routes.py
def get_memory_service(db: Session = Depends(get_db)) -> MemoryService:
    repository = MemoryRepository(db)
    return MemoryService(repository)

@router.post("/memories")
async def create_memory(
    memory: MemoryCreate,
    service: MemoryService = Depends(get_memory_service)
):
    return service.store_memory(memory.content, memory.metadata)
```

## Production Safety Features

### 1. No Mutable Defaults

```python
# WRONG
metadata = Column(JSON, default={})

# CORRECT
metadata = Column(JSON, default=dict)
```

### 2. Proper Indexing

```python
class Memory(Base):
    id = Column(Integer, primary_key=True, index=True)  # Auto-indexed
    created_at = Column(DateTime, default=datetime.utcnow, index=True)  # Indexed for queries
```

### 3. Service Layer Validation

All validation happens in the service layer:

```python
def _validate_content(self, content: str) -> None:
    if not content or not content.strip():
        raise ValidationError("Content cannot be empty")
    if len(content.strip()) < 1:
        raise ValidationError("Content must have at least 1 character")
```

### 4. Clean Error Handling

```python
try:
    memory = service.store_memory(content, metadata)
    return memory
except ValidationError as e:
    raise HTTPException(status_code=400, detail=str(e))
except NotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
```

## API Endpoints

### Memory Operations

- `POST /api/v1/memories` - Create a memory
- `GET /api/v1/memories/{id}` - Get a memory
- `GET /api/v1/memories` - List memories (with pagination)
- `PUT /api/v1/memories/{id}` - Update a memory
- `DELETE /api/v1/memories/{id}` - Delete a memory

### System Endpoints

- `GET /` - Root endpoint (returns "Luma is alive")
- `GET /health` - Health check

## Configuration

Configuration is managed via environment variables or `.env` file:

```bash
# Copy example config
cp .env.example .env

# Edit configuration
DATABASE_URL=sqlite:///./luma.db
API_PREFIX=/api/v1
LOG_LEVEL=INFO
```

## Development

```bash
# Run with auto-reload
uvicorn luma.main:app --reload

# Run on custom port
uvicorn luma.main:app --reload --port 8080

# Run with debug logging
LOG_LEVEL=DEBUG uvicorn luma.main:app --reload
```

## Future Extensibility

The architecture is designed to support:

- **Multiple Database Backends**: Repository pattern allows swapping SQLite for PostgreSQL/MySQL
- **Additional Agents**: Agent interface supports new specialized agents
- **Advanced Reasoning**: Reasoning engine can integrate ML models, rule engines
- **Distributed Scheduling**: Scheduler can be replaced with Celery
- **API Versioning**: API prefix supports multiple versions
- **Authentication**: Middleware can be added for security
- **Caching**: Service layer can integrate Redis

## Architecture Decisions

### Why Modular Monolith?

- **Simplicity**: Single deployment, no distributed system complexity
- **Local-first**: Works offline, no cloud dependencies
- **Performance**: No network overhead between modules
- **Easy to reason about**: All code in one place
- **Future-proof**: Can extract modules to microservices later if needed

### Why This Layering?

- **Testability**: Each layer can be tested independently
- **Maintainability**: Clear responsibilities, easy to find code
- **Flexibility**: Can swap implementations without affecting other layers
- **Scalability**: Clean boundaries make it easy to optimize or extract modules

### Why No Docker Yet?

- **YAGNI**: You Aren't Gonna Need It (yet)
- **Local development**: Simpler to run directly
- **Faster iteration**: No container build times
- **Add later**: Easy to containerize when needed

## Production Deployment

For production:

1. Use environment variables for configuration
2. Enable HTTPS/TLS
3. Configure log rotation
4. Use Gunicorn with Uvicorn workers
5. Implement rate limiting
6. Add health check monitoring
7. Configure database connection pooling
8. Implement backup strategy

```bash
# Production server
gunicorn luma.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## License

MIT
