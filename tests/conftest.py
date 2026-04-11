"""Pytest configuration and shared fixtures for Luma Memory Module tests."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def temp_key_path():
    """Provide a temporary encryption key path for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.key', delete=False) as f:
        key_path = f.name
    
    yield key_path
    
    # Cleanup
    if os.path.exists(key_path):
        os.remove(key_path)


@pytest.fixture
def sample_memory_entry_data():
    """Provide sample data for creating memory entries."""
    return {
        "action": "user_search",
        "context": {
            "query": "weather forecast",
            "location": "San Francisco",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        "sensitivity": "public",
        "device_id": "laptop-001",
        "tags": ["search", "weather"]
    }


@pytest.fixture
def test_data_dir():
    """Provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
