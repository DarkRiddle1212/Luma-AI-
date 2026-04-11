# Dependency Wiring Documentation

## Overview

The dependency wiring module (`luma/container.py`) provides functions for initializing the Luma application with all required components following clean architecture principles. It handles the complete dependency graph from storage layer up to the reasoning engine.

## Functions

### `initialize_application()`

Initializes the application with dependency injection.

**Parameters:**
- `db_path` (str): Path to SQLite database file. Defaults to `"./data/luma_memory.db"`. Directory will be created if it doesn't exist.
- `llm` (Optional[LLMInterface]): Optional LLM implementation. If None, uses StubLLM for testing.
- `return_storage` (bool): If True, returns tuple of (engine, storage) for cleanup. Defaults to False.

**Returns:**
- `ReasoningEngine`: Fully configured reasoning engine with all dependencies
- Or `tuple[ReasoningEngine, SQLiteStorage]` if `return_storage=True`

**Initialization Steps:**
1. Initialize SQLiteStorage with database path
2. Initialize MemoryManager with storage
3. Create SQLiteMemoryAdapter wrapping MemoryManager
4. Get LLM implementation (use provided or default to StubLLM)
5. Create ReasoningEngine with LLM and memory adapter
6. Log each initialization step

**Example:**
```python
from luma.container import initialize_application

# Initialize with default settings (StubLLM)
engine = initialize_application()

# Initialize with custom database path
engine = initialize_application(db_path="./custom/memory.db")

# Initialize with custom LLM
from luma.core.llm_interface import LLMInterface

class MyLLM(LLMInterface):
    def generate_response(self, prompt: str, context: dict) -> str:
        return "Custom response"

engine = initialize_application(llm=MyLLM())

# Initialize with cleanup support
engine, storage = initialize_application(return_storage=True)
# ... use engine ...
cleanup_application(storage)
```

### `verify_dependencies()`

Verifies all required dependencies are properly configured.

**Parameters:**
- `reasoning_engine` (ReasoningEngine): The reasoning engine instance to verify

**Raises:**
- `RuntimeError`: If LLM dependency is not configured (required)

**Verification Steps:**
1. Check if reasoning_engine.llm is not None
2. Raise RuntimeError if LLM missing (required dependency)
3. Check if reasoning_engine.memory is not None
4. Log warning if memory missing (optional, not error)
5. Log success if all dependencies present

**Example:**
```python
from luma.container import initialize_application, verify_dependencies

engine = initialize_application()
verify_dependencies(engine)  # Logs success

# Example with missing LLM (will raise error)
from luma.core.reasoning import ReasoningEngine
engine = ReasoningEngine(llm=None, memory=None)
verify_dependencies(engine)  # Raises RuntimeError
```

### `cleanup_application()`

Cleans up application resources, particularly database connections.

**Parameters:**
- `storage` (SQLiteStorage): The SQLiteStorage instance to cleanup

**Purpose:**
This function should be called when shutting down the application to ensure all database connections are properly closed. This is especially important on Windows where open connections can prevent file deletion.

**Example:**
```python
from luma.container import initialize_application, cleanup_application

# Initialize with cleanup support
engine, storage = initialize_application(return_storage=True)

try:
    # Use the engine
    result = engine.process_message("Hello!")
finally:
    # Always cleanup
    cleanup_application(storage)
```

## Architecture

The dependency wiring follows clean architecture principles:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│                  (luma/container.py)                         │
│                  - initialize_application()                  │
│                  - verify_dependencies()                     │
│                  - cleanup_application()                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core/Domain Layer                       │
│                                                              │
│  ┌──────────────────┐              ┌──────────────────┐    │
│  │ ReasoningEngine  │─────uses────▶│ MemoryInterface  │    │
│  │                  │              │      (ABC)       │    │
│  └──────────────────┘              └──────────────────┘    │
│         