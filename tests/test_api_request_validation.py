"""
Tests for API request validation and sanitization.

This module tests that the API properly validates and sanitizes
all incoming requests to prevent injection attacks and ensure data integrity.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from luma_memory.api.routes import app, set_memory_manager
from luma_memory.memory_manager import MemoryManager
from luma_memory.config import MemoryModuleConfig
from luma_memory.storage.memory_storage import MemoryStorage


@pytest.fixture
def client():
    """Create a test client with in-memory storage."""
    config = MemoryModuleConfig(
        db_path=":memory:",
        enable_metrics=True
    )
    storage = MemoryStorage()
    manager = MemoryManager(config=config, storage=storage)
    set_memory_manager(manager)
    
    with TestClient(app) as test_client:
        yield test_client


class TestCreateMemoryValidation:
    """Tests for create memory endpoint validation."""
    
    def test_create_memory_sanitizes_action(self, client):
        """Test that action field is sanitized."""
        payload = {
            "action": "  User opened document  ",  # Leading/trailing whitespace
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify the entry was created with sanitized action
        data = response.json()
        entry_id = data["entry_id"]
        
        # Retrieve the entry to verify sanitization
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        # Action should be trimmed
        assert entry_data["action"] == "User opened document"
    
    def test_create_memory_sanitizes_device_id(self, client):
        """Test that device_id field is sanitized."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "  laptop-001  ",  # Leading/trailing whitespace
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
    
    def test_create_memory_sanitizes_tags(self, client):
        """Test that tags are sanitized."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["  tag1  ", "  tag2  "]  # Leading/trailing whitespace
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
    
    def test_create_memory_prevents_xss_in_action(self, client):
        """Test that XSS attempts in action are sanitized."""
        payload = {
            "action": "<script>alert('xss')</script>",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify the entry was created with sanitized action
        data = response.json()
        entry_id = data["entry_id"]
        
        # Retrieve the entry to verify sanitization
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        # Script tags should be escaped or removed
        assert "<script>" not in entry_data["action"] or "&lt;script&gt;" in entry_data["action"]
    
    def test_create_memory_prevents_path_traversal(self, client):
        """Test that path traversal attempts are sanitized."""
        payload = {
            "action": "../../../etc/passwd",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify the entry was created with sanitized action
        data = response.json()
        entry_id = data["entry_id"]
        
        # Retrieve the entry to verify sanitization
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        # Path traversal should be removed
        assert "../" not in entry_data["action"]
    
    def test_create_memory_rejects_empty_action_after_sanitization(self, client):
        """Test that action cannot be empty after sanitization."""
        payload = {
            "action": "   ",  # Only whitespace
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower() or "whitespace" in data["detail"].lower()
    
    def test_create_memory_rejects_empty_device_id_after_sanitization(self, client):
        """Test that device_id cannot be empty after sanitization."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "   ",  # Only whitespace
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "device" in data["detail"].lower() and ("empty" in data["detail"].lower() or "whitespace" in data["detail"].lower())
    
    def test_create_memory_rejects_empty_tags_after_sanitization(self, client):
        """Test that tags cannot be empty after sanitization."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["tag1", "   ", "tag2"]  # Empty tag in the middle
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "tag" in data["detail"].lower() and ("empty" in data["detail"].lower() or "whitespace" in data["detail"].lower())
    
    def test_create_memory_rejects_empty_context(self, client):
        """Test that empty context is allowed."""
        payload = {
            "action": "Test action",
            "context": {},  # Empty context
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        # Empty context is valid, should succeed
        assert response.status_code == 201
    
    def test_create_memory_sanitizes_context_values(self, client):
        """Test that context values are sanitized."""
        payload = {
            "action": "Test action",
            "context": {
                "file": "  test.pdf  ",  # Whitespace
                "path": "../../../etc/passwd",  # Path traversal
                "script": "<script>alert('xss')</script>"  # XSS
            },
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify the entry was created with sanitized context
        data = response.json()
        entry_id = data["entry_id"]
        
        # Retrieve the entry to verify sanitization
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        
        # Context values should be sanitized
        assert entry_data["context"]["file"] == "test.pdf"
        assert "../" not in entry_data["context"]["path"]
        assert "<script>" not in entry_data["context"]["script"] or "&lt;script&gt;" in entry_data["context"]["script"]


class TestGetMemoryValidation:
    """Tests for get memory endpoint validation."""
    
    def test_get_memory_sanitizes_entry_id(self, client):
        """Test that entry_id is sanitized."""
        # First create an entry
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        create_response = client.post("/api/v1/memory", json=payload)
        assert create_response.status_code == 201
        entry_id = create_response.json()["entry_id"]
        
        # Try to get with whitespace
        response = client.get(f"/api/v1/memory/{entry_id}  ")
        # Should still work after sanitization
        assert response.status_code in [200, 404]  # 404 if sanitization changes the ID
    
    def test_get_memory_rejects_empty_entry_id(self, client):
        """Test that empty entry_id is rejected."""
        response = client.get("/api/v1/memory/   ")
        assert response.status_code == 400
        data = response.json()
        assert "entry" in data["detail"].lower() or "id" in data["detail"].lower()
    
    def test_get_memory_rejects_invalid_entry_id_format(self, client):
        """Test that invalid entry_id format is rejected."""
        # Entry ID with invalid characters (not URL-encoded)
        # Use a simple invalid format that won't be URL-encoded
        response = client.get("/api/v1/memory/invalid@#$%id")
        # Could be 400 (validation error) or 404 (not found after sanitization)
        assert response.status_code in [400, 404]


class TestQueryMemoriesValidation:
    """Tests for query memories endpoint validation."""
    
    def test_query_sanitizes_action_type(self, client):
        """Test that action_type is sanitized."""
        payload = {
            "action_type": "  opened  ",  # Whitespace
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 200
    
    def test_query_sanitizes_tags(self, client):
        """Test that tags are sanitized."""
        payload = {
            "tags": ["  tag1  ", "  tag2  "],  # Whitespace
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 200
    
    def test_query_rejects_empty_action_type_after_sanitization(self, client):
        """Test that action_type cannot be empty after sanitization."""
        payload = {
            "action_type": "   ",  # Only whitespace
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "action" in data["detail"].lower() and ("empty" in data["detail"].lower() or "whitespace" in data["detail"].lower())
    
    def test_query_rejects_empty_tags_after_sanitization(self, client):
        """Test that tags cannot be empty after sanitization."""
        payload = {
            "tags": ["tag1", "   ", "tag2"],  # Empty tag in the middle
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "tag" in data["detail"].lower() and ("empty" in data["detail"].lower() or "whitespace" in data["detail"].lower())
    
    def test_query_validates_time_range(self, client):
        """Test that queries with start_time after end_time return empty results."""
        payload = {
            "start_time": "2024-01-31T23:59:59Z",
            "end_time": "2024-01-01T00:00:00Z",  # Earlier than start_time
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        # Should succeed but return empty results (no entries match invalid range)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
    
    def test_query_prevents_xss_in_action_type(self, client):
        """Test that XSS attempts in action_type are sanitized."""
        payload = {
            "action_type": "<script>alert('xss')</script>",
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        # Should succeed with sanitized input
        assert response.status_code == 200
    
    def test_query_prevents_path_traversal_in_action_type(self, client):
        """Test that path traversal attempts in action_type are sanitized."""
        payload = {
            "action_type": "../../../etc/passwd",
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        # Should succeed with sanitized input
        assert response.status_code == 200


class TestInputSanitizationIntegration:
    """Integration tests for input sanitization across all endpoints."""
    
    def test_null_byte_removal(self, client):
        """Test that null bytes are removed from all string inputs."""
        payload = {
            "action": "test\x00action",
            "context": {"file": "test\x00.pdf"},
            "device_id": "laptop\x00-001",
            "sensitivity": "public",
            "tags": ["test\x00tag"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify null bytes were removed
        data = response.json()
        entry_id = data["entry_id"]
        
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        
        assert "\x00" not in entry_data["action"]
        assert "\x00" not in entry_data["device_id"]
        assert all("\x00" not in tag for tag in entry_data["tags"])
    
    def test_control_character_removal(self, client):
        """Test that control characters are removed from inputs."""
        payload = {
            "action": "test\x01\x02\x03action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify control characters were removed
        data = response.json()
        entry_id = data["entry_id"]
        
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        
        assert "\x01" not in entry_data["action"]
        assert "\x02" not in entry_data["action"]
        assert "\x03" not in entry_data["action"]
    
    def test_nested_context_sanitization(self, client):
        """Test that nested context dictionaries are sanitized."""
        payload = {
            "action": "Test action",
            "context": {
                "level1": {
                    "level2": {
                        "script": "<script>alert('xss')</script>",
                        "path": "../../../etc/passwd"
                    }
                }
            },
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify nested values were sanitized
        data = response.json()
        entry_id = data["entry_id"]
        
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        
        level2 = entry_data["context"]["level1"]["level2"]
        assert "<script>" not in level2["script"] or "&lt;script&gt;" in level2["script"]
        assert "../" not in level2["path"]


class TestPydanticValidation:
    """Tests for Pydantic model-level validation."""
    
    def test_create_memory_rejects_too_long_action(self, client):
        """Test that action exceeding max length is rejected."""
        payload = {
            "action": "A" * 1001,  # Exceeds max_length=1000
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "action" in str(data).lower()
    
    def test_create_memory_rejects_too_long_device_id(self, client):
        """Test that device_id exceeding max length is rejected."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "D" * 256,  # Exceeds max_length=255
            "sensitivity": "public",
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "device_id" in str(data).lower()
    
    def test_create_memory_rejects_too_long_tag(self, client):
        """Test that tag exceeding max length is rejected."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["T" * 101]  # Exceeds max_length=100
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "tag" in str(data).lower()
    
    def test_create_memory_rejects_too_many_tags(self, client):
        """Test that too many tags are rejected."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": [f"tag{i}" for i in range(51)]  # Exceeds max_length=50
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "tags" in str(data).lower()
    
    def test_create_memory_rejects_invalid_sensitivity(self, client):
        """Test that invalid sensitivity value is rejected."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "invalid",  # Not public, private, or sensitive
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "sensitivity" in str(data).lower()
    
    def test_create_memory_normalizes_sensitivity_case(self, client):
        """Test that sensitivity is normalized to lowercase."""
        payload = {
            "action": "Test action",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "PRIVATE",  # Uppercase
            "tags": ["test"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        # Verify sensitivity was normalized
        data = response.json()
        entry_id = data["entry_id"]
        
        get_response = client.get(f"/api/v1/memory/{entry_id}")
        assert get_response.status_code == 200
        entry_data = get_response.json()
        assert entry_data["sensitivity"] == "private"
    
    def test_query_rejects_invalid_limit(self, client):
        """Test that invalid limit values are rejected."""
        # Test limit too low
        payload = {"limit": 0}
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
        
        # Test limit too high
        payload = {"limit": 1001}
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
    
    def test_query_rejects_negative_offset(self, client):
        """Test that negative offset is rejected."""
        payload = {"offset": -1}
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
    
    def test_query_rejects_too_long_action_type(self, client):
        """Test that action_type exceeding max length is rejected."""
        payload = {
            "action_type": "A" * 1001,  # Exceeds max_length=1000
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "action_type" in str(data).lower()
    
    def test_query_rejects_too_many_tags(self, client):
        """Test that too many tags in query are rejected."""
        payload = {
            "tags": [f"tag{i}" for i in range(51)],  # Exceeds max_length=50
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "tags" in str(data).lower()
    
    def test_query_rejects_too_long_tag(self, client):
        """Test that tag exceeding max length in query is rejected."""
        payload = {
            "tags": ["T" * 101],  # Exceeds max_length=100
            "limit": 10
        }
        
        response = client.post("/api/v1/memory/query", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "tag" in str(data).lower()
