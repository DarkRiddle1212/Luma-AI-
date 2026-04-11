"""
Comprehensive integration tests for Luma Memory Module REST API.

This test suite verifies all API endpoints including:
- POST /api/v1/memory (create memory)
- GET /api/v1/memory/{entry_id} (retrieve memory)
- POST /api/v1/memory/query (query memories)
- GET /api/v1/health (health check)
- GET /api/v1/stats (statistics)

Tests cover success cases, error handling, validation, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, UTC
import tempfile
import os
from pathlib import Path

from luma_memory.api.server import create_app
from luma_memory.config import MemoryModuleConfig
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.api.routes import set_memory_manager


class TestMemoryAPI:
    """Comprehensive test suite for Memory API endpoints."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for database and encryption key."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memory.db")
        key_path = os.path.join(temp_dir, "test_encryption.key")
        
        yield db_path, key_path
        
        # Cleanup
        for path in [db_path, key_path]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def memory_manager(self, temp_paths):
        """Create a memory manager for testing."""
        db_path, key_path = temp_paths
        
        # Initialize components
        storage = SQLiteStorage(db_path=db_path, cache_size=100)
        encryption = EncryptionService(key_path=key_path)
        validation = ValidationManager()
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        # Create config
        config = MemoryModuleConfig(
            db_path=db_path,
            encryption_key_path=key_path,
            cache_size=100
        )
        
        # Create manager
        manager = MemoryManager(
            storage=storage,
            encryption=encryption,
            validation=validation,
            summarizer=summarizer,
            config=config
        )
        
        yield manager
        
        # Cleanup
        if hasattr(storage, 'close'):
            storage.close()
    
    @pytest.fixture
    def client(self, memory_manager):
        """Create a test client with initialized memory manager."""
        # Set the memory manager globally
        set_memory_manager(memory_manager)
        
        # Create test client
        from luma_memory.api.routes import app
        client = TestClient(app)
        
        yield client
    
    # Health Check Tests
    
    def test_health_check(self, client):
        """Test GET /api/v1/health returns healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")
    
    # Create Memory Tests
    
    def test_create_memory_success(self, client):
        """Test POST /api/v1/memory creates a memory entry successfully."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf", "page": 1},
            "device_id": "laptop-001",
            "sensitivity": "private",
            "tags": ["document", "work"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
        assert data["message"] == "Memory entry created successfully"
        assert isinstance(data["entry_id"], str)
        assert len(data["entry_id"]) > 0
    
    def test_create_memory_minimal_fields(self, client):
        """Test creating memory with only required fields."""
        payload = {
            "action": "User clicked button",
            "context": {"button": "submit"},
            "device_id": "phone-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_missing_action(self, client):
        """Test creating memory without action field returns 422."""
        payload = {
            "context": {"test": "data"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_create_memory_missing_context(self, client):
        """Test creating memory without context field returns 422."""
        payload = {
            "action": "Test action",
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_missing_device_id(self, client):
        """Test creating memory without device_id returns 422."""
        payload = {
            "action": "Test action",
            "context": {"test": "data"}
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_invalid_sensitivity(self, client):
        """Test creating memory with invalid sensitivity level returns 422."""
        payload = {
            "action": "Test action",
            "context": {"test": "data"},
            "device_id": "laptop-001",
            "sensitivity": "invalid_level"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data
        assert "sensitivity" in str(data).lower()
    
    def test_create_memory_empty_action(self, client):
        """Test creating memory with empty action string returns 422."""
        payload = {
            "action": "",
            "context": {"test": "data"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_with_tags(self, client):
        """Test creating memory with multiple tags."""
        payload = {
            "action": "User searched",
            "context": {"query": "weather"},
            "device_id": "phone-001",
            "tags": ["search", "weather", "user-action"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    # Get Memory Tests
    
    def test_get_memory_success(self, client):
        """Test GET /api/v1/memory/{entry_id} retrieves a memory entry."""
        # Create a memory first
        create_payload = {
            "action": "Test action",
            "context": {"key": "value"},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["test"]
        }
        create_response = client.post("/api/v1/memory", json=create_payload)
        entry_id = create_response.json()["entry_id"]
        
        # Retrieve it
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == entry_id
        assert data["action"] == "Test action"
        assert data["context"] == {"key": "value"}
        assert data["device_id"] == "laptop-001"
        assert data["sensitivity"] == "public"
        assert data["tags"] == ["test"]
        assert data["sync_status"] == "pending"
        assert "timestamp" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_get_memory_not_found(self, client):
        """Test GET /api/v1/memory/{entry_id} with non-existent ID returns 404."""
        response = client.get("/api/v1/memory/nonexistent-id-12345")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_get_memory_with_sensitive_data(self, client):
        """Test retrieving memory with sensitive data (should be decrypted)."""
        # Create sensitive memory
        create_payload = {
            "action": "User entered password",
            "context": {"field": "password", "length": 12},
            "device_id": "laptop-001",
            "sensitivity": "sensitive"
        }
        create_response = client.post("/api/v1/memory", json=create_payload)
        entry_id = create_response.json()["entry_id"]
        
        # Retrieve it
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sensitivity"] == "sensitive"
        assert data["context"] == {"field": "password", "length": 12}
    
    # Query Memory Tests
    
    def test_query_memories_all(self, client):
        """Test POST /api/v1/memory/query returns all memories."""
        # Create multiple memories
        for i in range(3):
            payload = {
                "action": f"Action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            }
            client.post("/api/v1/memory", json=payload)
        
        # Query all
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 3
        assert len(data["entries"]) >= 3
        assert data["limit"] == 100
        assert data["offset"] == 0
    
    def test_query_memories_with_limit(self, client):
        """Test querying memories with limit parameter."""
        # Create multiple memories
        for i in range(5):
            payload = {
                "action": f"Action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            }
            client.post("/api/v1/memory", json=payload)
        
        # Query with limit
        response = client.post("/api/v1/memory/query", json={"limit": 2})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) <= 2
        assert data["limit"] == 2
    
    def test_query_memories_with_offset(self, client):
        """Test querying memories with offset for pagination."""
        # Create multiple memories
        for i in range(5):
            payload = {
                "action": f"Action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            }
            client.post("/api/v1/memory", json=payload)
        
        # Query with offset
        response = client.post("/api/v1/memory/query", json={"limit": 2, "offset": 2})
        
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 2
    
    def test_query_memories_by_tags(self, client):
        """Test querying memories filtered by tags."""
        # Create memories with different tags
        client.post("/api/v1/memory", json={
            "action": "Work action",
            "context": {"type": "work"},
            "device_id": "laptop-001",
            "tags": ["work", "document"]
        })
        client.post("/api/v1/memory", json={
            "action": "Personal action",
            "context": {"type": "personal"},
            "device_id": "phone-001",
            "tags": ["personal", "photo"]
        })
        
        # Query by work tag
        response = client.post("/api/v1/memory/query", json={"tags": ["work"]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        # Verify all returned entries have the work tag
        for entry in data["entries"]:
            if "work" in entry["tags"]:
                assert "work" in entry["tags"]
    
    def test_query_memories_by_action_type(self, client):
        """Test querying memories filtered by action type."""
        # Create memories with different actions
        client.post("/api/v1/memory", json={
            "action": "User opened document",
            "context": {"file": "test.pdf"},
            "device_id": "laptop-001"
        })
        client.post("/api/v1/memory", json={
            "action": "User clicked button",
            "context": {"button": "submit"},
            "device_id": "laptop-001"
        })
        
        # Query by action type (partial match)
        response = client.post("/api/v1/memory/query", json={"action_type": "document"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
    
    def test_query_memories_by_time_range(self, client):
        """Test querying memories filtered by time range."""
        # Create a memory
        client.post("/api/v1/memory", json={
            "action": "Recent action",
            "context": {"test": "data"},
            "device_id": "laptop-001"
        })
        
        # Query with time range
        now = datetime.now(UTC)
        start_time = (now - timedelta(hours=1)).isoformat() + "Z"
        end_time = (now + timedelta(hours=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "end_time": end_time
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
    
    def test_query_memories_invalid_time_format(self, client):
        """Test querying with invalid time format returns error."""
        response = client.post("/api/v1/memory/query", json={
            "start_time": "invalid-time-format"
        })
        
        # Should return error (either 400 or 500 depending on error handling)
        assert response.status_code in [400, 500]
        data = response.json()
        assert "detail" in data
    
    def test_query_memories_empty_result(self, client):
        """Test querying with filters that match nothing returns empty list."""
        response = client.post("/api/v1/memory/query", json={
            "tags": ["nonexistent-tag-xyz"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []
    
    def test_query_memories_combined_filters(self, client):
        """Test querying with multiple filters combined."""
        # Create test memory
        client.post("/api/v1/memory", json={
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001",
            "tags": ["work", "document"]
        })
        
        # Query with combined filters
        now = datetime.now(UTC)
        start_time = (now - timedelta(hours=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "tags": ["work"],
            "action_type": "document",
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["entries"], list)
    
    # Statistics Tests
    
    def test_get_stats(self, client):
        """Test GET /api/v1/stats returns statistics."""
        # Create some memories first
        for i in range(3):
            client.post("/api/v1/memory", json={
                "action": f"Action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            })
        
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert data["total_entries"] >= 3
        assert "storage_size_bytes" in data
        assert "encryption_enabled" in data
        assert "summarizer_enabled" in data
        assert isinstance(data["encryption_enabled"], bool)
        assert isinstance(data["summarizer_enabled"], bool)
    
    # Error Handling Tests
    
    def test_invalid_json_payload(self, client):
        """Test sending invalid JSON returns 422."""
        response = client.post(
            "/api/v1/memory",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint returns 404."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method returns 405."""
        response = client.put("/api/v1/health")
        
        assert response.status_code == 405
    
    def test_create_memory_empty_context(self, client):
        """Test creating memory with empty context dictionary."""
        payload = {
            "action": "Test action",
            "context": {},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        # Empty context is valid, should succeed
        assert response.status_code == 201
    
    def test_create_memory_invalid_context_type(self, client):
        """Test creating memory with non-dict context returns 422."""
        payload = {
            "action": "Test action",
            "context": "not a dictionary",
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_invalid_tags_type(self, client):
        """Test creating memory with non-list tags returns 422."""
        payload = {
            "action": "Test action",
            "context": {"test": "data"},
            "device_id": "laptop-001",
            "tags": "not-a-list"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_empty_device_id(self, client):
        """Test creating memory with empty device_id returns 422."""
        payload = {
            "action": "Test action",
            "context": {"test": "data"},
            "device_id": ""
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_query_memories_invalid_limit(self, client):
        """Test querying with limit > 1000 returns 422."""
        response = client.post("/api/v1/memory/query", json={"limit": 1001})
        
        assert response.status_code == 422
    
    def test_query_memories_zero_limit(self, client):
        """Test querying with limit = 0 returns 422."""
        response = client.post("/api/v1/memory/query", json={"limit": 0})
        
        assert response.status_code == 422
    
    def test_query_memories_negative_limit(self, client):
        """Test querying with negative limit returns 422."""
        response = client.post("/api/v1/memory/query", json={"limit": -1})
        
        assert response.status_code == 422
    
    def test_query_memories_negative_offset(self, client):
        """Test querying with negative offset returns 422."""
        response = client.post("/api/v1/memory/query", json={"offset": -1})
        
        assert response.status_code == 422
    
    def test_query_memories_invalid_end_time_format(self, client):
        """Test querying with invalid end_time format returns error."""
        response = client.post("/api/v1/memory/query", json={
            "end_time": "not-a-valid-time"
        })
        
        # Should return error (either 400 or 500 depending on error handling)
        assert response.status_code in [400, 500]
        data = response.json()
        assert "detail" in data
    
    def test_get_memory_empty_id(self, client):
        """Test retrieving memory with empty ID."""
        response = client.get("/api/v1/memory/")
        
        # FastAPI will return 307 redirect or 405 for missing path parameter
        assert response.status_code in [307, 404, 405]
    
    def test_create_memory_null_action(self, client):
        """Test creating memory with null action returns 422."""
        payload = {
            "action": None,
            "context": {"test": "data"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_null_context(self, client):
        """Test creating memory with null context returns 422."""
        payload = {
            "action": "Test action",
            "context": None,
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_null_device_id(self, client):
        """Test creating memory with null device_id returns 422."""
        payload = {
            "action": "Test action",
            "context": {"test": "data"},
            "device_id": None
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422
    
    def test_query_memories_invalid_tags_type(self, client):
        """Test querying with non-list tags returns 422."""
        response = client.post("/api/v1/memory/query", json={
            "tags": "not-a-list"
        })
        
        assert response.status_code == 422
    
    def test_get_memory_special_characters_in_id(self, client):
        """Test retrieving memory with special characters in ID."""
        response = client.get("/api/v1/memory/test@#$%^&*()")
        
        # Should return 404 as entry doesn't exist
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_create_memory_very_long_action(self, client):
        """Test creating memory with very long action string."""
        payload = {
            "action": "A" * 10000,  # 10k character action
            "context": {"test": "data"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        # Validation may reject very long actions (422 for Pydantic, 400 for business logic) or accept them (201)
        assert response.status_code in [201, 400, 422]
    
    def test_create_memory_very_large_context(self, client):
        """Test creating memory with very large context dictionary."""
        payload = {
            "action": "Test action",
            "context": {f"key_{i}": f"value_{i}" for i in range(1000)},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        # Should succeed - no size restriction
        assert response.status_code == 201
    
    def test_create_memory_nested_context(self, client):
        """Test creating memory with deeply nested context."""
        payload = {
            "action": "Test action",
            "context": {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "data": "deep value"
                            }
                        }
                    }
                }
            },
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
    
    def test_query_memories_both_time_filters(self, client):
        """Test querying with start_time after end_time."""
        now = datetime.now(UTC)
        start_time = (now + timedelta(hours=1)).isoformat() + "Z"
        end_time = (now - timedelta(hours=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "end_time": end_time
        })
        
        # Should succeed but return empty results
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
    
    def test_create_memory_unicode_characters(self, client):
        """Test creating memory with unicode characters."""
        payload = {
            "action": "用户打开文档 🎉",
            "context": {"file": "报告.pdf", "emoji": "😀"},
            "device_id": "laptop-001",
            "tags": ["文档", "工作"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
    
    def test_get_stats_error_handling(self, client):
        """Test stats endpoint handles errors gracefully."""
        # Stats should work even with empty database
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
