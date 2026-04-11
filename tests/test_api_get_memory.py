"""
Test for GET /api/v1/memory/{entry_id} endpoint (Task 11.3).

This test verifies that the REST API endpoint for retrieving memory entries
works correctly with proper error handling and response formatting.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from luma_memory.api.routes import app, set_memory_manager
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.processing.validation import ValidationManager
from luma_memory.config import MemoryModuleConfig
from luma_memory.models import SensitivityLevel


class TestGetMemoryEndpoint:
    """Test suite for GET /api/v1/memory/{entry_id} endpoint."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a MemoryManager with in-memory storage for testing."""
        storage = MemoryStorage()
        validation = ValidationManager()
        config = MemoryModuleConfig()
        manager = MemoryManager(
            storage=storage,
            validation=validation,
            config=config
        )
        return manager
    
    @pytest.fixture
    def client(self, memory_manager):
        """Create a test client with initialized memory manager."""
        set_memory_manager(memory_manager)
        return TestClient(app)
    
    def test_get_memory_success(self, client, memory_manager):
        """Test retrieving a memory entry successfully."""
        # Create a memory entry first
        entry_id = memory_manager.create_memory(
            action="User opened document",
            context={"file": "report.pdf", "page": 1},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["document", "work"]
        )
        
        # Retrieve the entry via API
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure and content
        assert data["id"] == entry_id
        assert data["action"] == "User opened document"
        assert data["context"]["file"] == "report.pdf"
        assert data["context"]["page"] == 1
        assert data["device_id"] == "laptop-001"
        assert data["sensitivity"] == "public"
        assert "document" in data["tags"]
        assert "work" in data["tags"]
        assert data["sync_status"] == "pending"
    
    def test_get_memory_not_found(self, client):
        """Test retrieving a non-existent memory entry."""
        response = client.get("/api/v1/memory/nonexistent-id-12345")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_get_memory_with_private_sensitivity(self, client, memory_manager):
        """Test retrieving a memory entry with private sensitivity."""
        entry_id = memory_manager.create_memory(
            action="User entered password",
            context={"field": "login_form"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PRIVATE,
            tags=["auth"]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sensitivity"] == "private"
    
    def test_get_memory_with_sensitive_data(self, client, memory_manager):
        """Test retrieving a memory entry with sensitive data."""
        entry_id = memory_manager.create_memory(
            action="User viewed medical record",
            context={"record_id": "12345"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.SENSITIVE,
            tags=["health", "private"]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sensitivity"] == "sensitive"

    def test_get_memory_response_format(self, client, memory_manager):
        """Test that the response format matches the specification."""
        entry_id = memory_manager.create_memory(
            action="User clicked button",
            context={"button": "submit"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["ui"]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        required_fields = [
            "id", "timestamp", "action", "context", "sensitivity",
            "device_id", "sync_status", "tags"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify optional fields
        assert "summary" in data
        assert "parent_id" in data
        assert "created_at" in data
        assert "updated_at" in data
        
        # Verify field types
        assert isinstance(data["id"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["action"], str)
        assert isinstance(data["context"], dict)
        assert isinstance(data["sensitivity"], str)
        assert isinstance(data["device_id"], str)
        assert isinstance(data["sync_status"], str)
        assert isinstance(data["tags"], list)
    
    def test_get_memory_with_complex_context(self, client, memory_manager):
        """Test retrieving a memory entry with complex nested context."""
        complex_context = {
            "form": "registration",
            "fields": {
                "name": "John Doe",
                "email": "john@example.com"
            },
            "metadata": {
                "duration_seconds": 120,
                "validation_errors": 0
            }
        }
        
        entry_id = memory_manager.create_memory(
            action="User completed form",
            context=complex_context,
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["form", "registration"]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify complex context is preserved
        assert data["context"]["form"] == "registration"
        assert data["context"]["fields"]["name"] == "John Doe"
        assert data["context"]["metadata"]["duration_seconds"] == 120
    
    def test_get_memory_timestamp_format(self, client, memory_manager):
        """Test that timestamps are returned in ISO format."""
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"test": "data"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=[]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify timestamp format (ISO 8601 with Z suffix)
        assert data["timestamp"].endswith("Z")
        # Verify it can be parsed as ISO datetime
        datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        
        if data["created_at"]:
            assert data["created_at"].endswith("Z")
            datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        
        if data["updated_at"]:
            assert data["updated_at"].endswith("Z")
            datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))
    
    def test_get_memory_with_empty_tags(self, client, memory_manager):
        """Test retrieving a memory entry with empty tags list."""
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"test": "data"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=[]
        )
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["tags"] == []
    
    def test_get_memory_multiple_entries(self, client, memory_manager):
        """Test retrieving multiple different memory entries."""
        entry_ids = []
        
        for i in range(3):
            entry_id = memory_manager.create_memory(
                action=f"User action {i}",
                context={"index": i},
                device_id="laptop-001",
                sensitivity=SensitivityLevel.PUBLIC,
                tags=[f"tag{i}"]
            )
            entry_ids.append(entry_id)
        
        # Retrieve each entry and verify
        for i, entry_id in enumerate(entry_ids):
            response = client.get(f"/api/v1/memory/{entry_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["action"] == f"User action {i}"
            assert data["context"]["index"] == i
            assert f"tag{i}" in data["tags"]
    
    def test_get_memory_with_summary(self, client, memory_manager):
        """Test retrieving a memory entry that has a summary."""
        # Create an entry
        entry_id = memory_manager.create_memory(
            action="User action",
            context={"test": "data"},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=[]
        )
        
        # Update it with a summary (if update is implemented)
        entry = memory_manager.get_memory(entry_id)
        if hasattr(memory_manager, 'update_memory'):
            memory_manager.update_memory(entry_id, {"summary": "Test summary"})
        
        response = client.get(f"/api/v1/memory/{entry_id}")
        
        assert response.status_code == 200
        data = response.json()
        # Summary field should be present (even if None)
        assert "summary" in data
    
    def test_get_memory_empty_id(self, client):
        """Test retrieving with empty entry ID."""
        response = client.get("/api/v1/memory/")
        
        # Should return 404 or 405 (method not allowed) since it doesn't match the route
        assert response.status_code in [404, 405]
