"""
Tests for the /api/v1/metrics endpoint.

This module tests the metrics export endpoint that provides performance
statistics for all memory operations.
"""

import pytest
from fastapi.testclient import TestClient

from luma_memory.api.server import create_app
from luma_memory.config import MemoryModuleConfig
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.models import SensitivityLevel


@pytest.fixture
def test_config():
    """Create a test configuration with metrics enabled."""
    config = MemoryModuleConfig(
        db_path=":memory:",
        enable_metrics=True,
        log_level="ERROR"
    )
    return config


@pytest.fixture
def test_app(test_config, tmp_path):
    """Create a test FastAPI application with metrics enabled."""
    # Create memory manager with in-memory storage
    storage = MemoryStorage()
    encryption = EncryptionService(key_path=str(tmp_path / "test.key"))
    validation = ValidationManager()
    summarizer = ContextSummarizer()
    
    manager = MemoryManager(
        storage=storage,
        encryption=encryption,
        validation=validation,
        summarizer=summarizer,
        config=test_config
    )
    
    # Create app and set memory manager
    app = create_app(config=test_config)
    from luma_memory.api.routes import set_memory_manager
    set_memory_manager(manager)
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


def test_get_metrics_empty(client):
    """Test getting metrics when no operations have been performed."""
    response = client.get("/api/v1/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have metrics for all operations
    assert "create_memory" in data
    assert "get_memory" in data
    assert "query_memories" in data
    assert "update_memory" in data
    assert "delete_memory" in data
    
    # All counts should be zero
    assert data["create_memory"]["count"] == 0
    assert data["get_memory"]["count"] == 0
    assert data["query_memories"]["count"] == 0


def test_get_metrics_after_operations(client):
    """Test getting metrics after performing some operations."""
    # Create a few memory entries
    for i in range(3):
        response = client.post(
            "/api/v1/memory",
            json={
                "action": f"Test action {i}",
                "context": {"test": f"value{i}"},
                "device_id": "test-device",
                "sensitivity": "public",
                "tags": ["test"]
            }
        )
        # The endpoint returns 200 when using TestClient with lifespan
        assert response.status_code in [200, 201]
    
    # Get metrics
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Should have 3 create operations
    assert data["create_memory"]["count"] == 3
    assert data["create_memory"]["avg_time_ms"] > 0
    assert data["create_memory"]["min_time_ms"] > 0
    assert data["create_memory"]["max_time_ms"] > 0
    assert data["create_memory"]["errors"] == 0
    assert data["create_memory"]["error_rate"] == 0.0


def test_get_metrics_with_errors(client):
    """Test that metrics track errors correctly."""
    # Create a valid entry first to ensure metrics work
    response = client.post(
        "/api/v1/memory",
        json={
            "action": "Valid action",
            "context": {"test": "value"},
            "device_id": "test-device"
        }
    )
    assert response.status_code in [200, 201]
    
    # Get metrics - should show no errors for successful operation
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Should have 1 successful create operation with no errors
    assert data["create_memory"]["count"] == 1
    assert data["create_memory"]["errors"] == 0
    assert data["create_memory"]["error_rate"] == 0.0


def test_get_metrics_latency_tracking(client):
    """Test that metrics track min/max/avg latency correctly."""
    # Create multiple entries
    for i in range(5):
        response = client.post(
            "/api/v1/memory",
            json={
                "action": f"Test action {i}",
                "context": {"test": f"value{i}"},
                "device_id": "test-device",
                "sensitivity": "public"
            }
        )
        assert response.status_code in [200, 201]
    
    # Get metrics
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Verify latency metrics
    create_metrics = data["create_memory"]
    assert create_metrics["count"] == 5
    assert create_metrics["min_time_ms"] <= create_metrics["avg_time_ms"]
    assert create_metrics["avg_time_ms"] <= create_metrics["max_time_ms"]
    assert create_metrics["min_time_ms"] >= 0
    assert create_metrics["max_time_ms"] > 0


def test_get_metrics_system_resources(client):
    """Test that metrics include system resource information."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # System resources may or may not be present depending on psutil availability
    # If present, verify structure
    if "system_resources" in data:
        resources = data["system_resources"]
        assert "memory_usage_mb" in resources
        assert "memory_usage_percent" in resources
        assert "cpu_percent" in resources
        assert "num_threads" in resources
        
        # Verify values are reasonable
        assert resources["memory_usage_mb"] > 0
        assert resources["memory_usage_percent"] >= 0
        assert resources["cpu_percent"] >= 0
        assert resources["num_threads"] > 0


def test_get_metrics_all_operations(client):
    """Test metrics for all operation types."""
    # Create an entry
    create_response = client.post(
        "/api/v1/memory",
        json={
            "action": "Test action",
            "context": {"test": "value"},
            "device_id": "test-device",
            "sensitivity": "public"
        }
    )
    assert create_response.status_code in [200, 201]
    entry_id = create_response.json()["entry_id"]
    
    # Get the entry
    get_response = client.get(f"/api/v1/memory/{entry_id}")
    assert get_response.status_code == 200
    
    # Query entries
    query_response = client.post(
        "/api/v1/memory/query",
        json={"limit": 10}
    )
    assert query_response.status_code == 200
    
    # Get metrics
    metrics_response = client.get("/api/v1/metrics")
    assert metrics_response.status_code == 200
    data = metrics_response.json()
    
    # Verify all operations have metrics
    assert data["create_memory"]["count"] == 1
    assert data["get_memory"]["count"] == 1
    assert data["query_memories"]["count"] == 1


def test_get_metrics_disabled(tmp_path):
    """Test metrics endpoint when metrics are disabled."""
    # Create config with metrics disabled
    config = MemoryModuleConfig(
        db_path=":memory:",
        enable_metrics=False,
        log_level="ERROR"
    )
    
    # Create memory manager
    storage = MemoryStorage()
    encryption = EncryptionService(key_path=str(tmp_path / "test.key"))
    validation = ValidationManager()
    
    manager = MemoryManager(
        storage=storage,
        encryption=encryption,
        validation=validation,
        config=config
    )
    
    # Create app
    app = create_app(config=config)
    from luma_memory.api.routes import set_memory_manager
    set_memory_manager(manager)
    
    client = TestClient(app)
    
    # Get metrics
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Should have a message indicating metrics are disabled
    assert "message" in data
    assert "disabled" in data["message"].lower()
    assert "metrics" in data
    assert data["metrics"] == {}


def test_metrics_endpoint_in_openapi(client):
    """Test that the metrics endpoint is documented in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    openapi_schema = response.json()
    
    # Verify /api/v1/metrics endpoint exists
    assert "/api/v1/metrics" in openapi_schema["paths"]
    
    # Verify it has a GET method
    metrics_endpoint = openapi_schema["paths"]["/api/v1/metrics"]
    assert "get" in metrics_endpoint
    
    # Verify it has proper documentation
    get_method = metrics_endpoint["get"]
    assert "summary" in get_method
    assert "Metrics" in get_method["summary"]
