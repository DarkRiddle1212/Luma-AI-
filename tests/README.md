# Luma Memory Module Tests

This directory contains the test suite for the Luma Memory Module.

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Shared pytest fixtures and configuration
├── test_models.py              # Unit tests for data models (Task 13)
├── test_sqlite_storage.py      # Unit tests for SQLite storage backend (Task 14)
├── test_memory_storage.py      # Unit tests for in-memory storage backend (Task 14)
├── test_encryption.py          # Unit tests for encryption service (Task 15)
├── test_validation.py          # Unit tests for validation manager (Task 16)
├── test_summarizer.py          # Unit tests for context summarizer (Task 17)
├── test_memory_manager.py      # Integration tests for memory manager (Task 18)
├── test_api.py                 # API integration tests (Task 19)
└── test_performance.py         # Performance tests (Task 20)
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_models.py
```

### Run with coverage report
```bash
pytest --cov=luma_memory --cov-report=html
```

### Run tests by marker
```bash
pytest -m unit          # Run only unit tests
pytest -m integration   # Run only integration tests
pytest -m performance   # Run only performance tests
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_db_path`: Provides a temporary database file path for testing
- `temp_key_path`: Provides a temporary encryption key file path
- `sample_memory_entry_data`: Sample data for creating memory entries
- `test_data_dir`: Temporary directory for test data

## Test Categories

Tests are organized by markers:

- **unit**: Tests for individual components in isolation
- **integration**: Tests for component interactions
- **performance**: Tests that verify latency and resource requirements
- **slow**: Tests that take significant time to run

## Writing Tests

When writing tests:

1. Use descriptive test names that explain what is being tested
2. Follow the Arrange-Act-Assert pattern
3. Use appropriate fixtures from `conftest.py`
4. Add markers to categorize tests
5. Ensure tests are isolated and don't depend on each other
6. Clean up resources (files, connections) after tests

## Test Implementation Status

- [ ] Task 13: Unit tests for data models
- [ ] Task 14: Unit tests for storage layer
- [ ] Task 15: Unit tests for encryption
- [ ] Task 16: Unit tests for validation
- [ ] Task 17: Unit tests for context summarizer
- [ ] Task 18: Integration tests for memory manager
- [ ] Task 19: API integration tests
- [ ] Task 20: Performance tests
