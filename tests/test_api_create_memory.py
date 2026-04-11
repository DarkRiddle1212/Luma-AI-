"""
Test for POST /api/v1/memory endpoint (Task 11.2).

This test verifies that the REST API endpoint for creating memory entries
works correctly with proper validation, error handling, and response formatting.
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


class TestCreateMemoryEndpoint:
    """Test suite for POST /api/v1/memory endpoint."""
    
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
    
    def test_create_memory_success(self, client):
        """Test creating a memory entry successfully."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf", "page": 1},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["document", "work"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
        assert data["message"] == "Memory entry created successfully"
        assert len(data["entry_id"]) > 0
    
    def test_create_memory_minimal_payload(self, client):
        """Test creating a memory entry with minimal required fields."""
        payload = {
            "action": "User clicked button",
            "context": {"button": "submit"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_with_private_sensitivity(self, client):
        """Test creating a memory entry with private sensitivity."""
        payload = {
            "action": "User entered password",
            "context": {"field": "login_form"},
            "device_id": "laptop-001",
            "sensitivity": "private"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_with_sensitive_data(self, client):
        """Test creating a memory entry with sensitive data."""
        payload = {
            "action": "User viewed medical record",
            "context": {"record_id": "12345"},
            "device_id": "laptop-001",
            "sensitivity": "sensitive",
            "tags": ["health", "private"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_missing_action(self, client):
        """Test creating a memory entry without action field."""
        payload = {
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_missing_context(self, client):
        """Test creating a memory entry without context field."""
        payload = {
            "action": "User opened document",
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_missing_device_id(self, client):
        """Test creating a memory entry without device_id field."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"}
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_empty_action(self, client):
        """Test creating a memory entry with empty action."""
        payload = {
            "action": "",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_invalid_sensitivity(self, client):
        """Test creating a memory entry with invalid sensitivity level."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001",
            "sensitivity": "invalid_level"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "Invalid sensitivity level" in str(data)
    
    def test_create_memory_invalid_context_type(self, client):
        """Test creating a memory entry with non-dict context."""
        payload = {
            "action": "User opened document",
            "context": "not a dictionary",
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_invalid_tags_type(self, client):
        """Test creating a memory entry with non-list tags."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001",
            "tags": "not a list"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_create_memory_with_empty_tags(self, client):
        """Test creating a memory entry with empty tags list."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001",
            "tags": []
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_response_format(self, client):
        """Test that the response format matches the specification."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001"
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        
        # Verify response structure
        assert isinstance(data, dict)
        assert "entry_id" in data
        assert "message" in data
        assert isinstance(data["entry_id"], str)
        assert isinstance(data["message"], str)
    
    def test_create_memory_returns_valid_id(self, client, memory_manager):
        """Test that the returned entry_id can be used to retrieve the entry."""
        payload = {
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "device_id": "laptop-001",
            "tags": ["document"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        assert response.status_code == 201
        
        entry_id = response.json()["entry_id"]
        
        # Verify we can retrieve the entry
        entry = memory_manager.get_memory(entry_id)
        assert entry is not None
        assert entry.action == "User opened document"
        assert entry.context["file"] == "report.pdf"
        assert entry.device_id == "laptop-001"
        assert "document" in entry.tags
    
    def test_create_memory_with_complex_context(self, client):
        """Test creating a memory entry with complex nested context."""
        payload = {
            "action": "User completed form",
            "context": {
                "form": "registration",
                "fields": {
                    "name": "John Doe",
                    "email": "john@example.com"
                },
                "metadata": {
                    "duration_seconds": 120,
                    "validation_errors": 0
                }
            },
            "device_id": "laptop-001",
            "tags": ["form", "registration"]
        }
        
        response = client.post("/api/v1/memory", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "entry_id" in data
    
    def test_create_memory_multiple_entries(self, client):
        """Test creating multiple memory entries in sequence."""
        entry_ids = []
        
        for i in range(5):
            payload = {
                "action": f"User action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            }
            
            response = client.post("/api/v1/memory", json=payload)
            assert response.status_code == 201
            
            entry_id = response.json()["entry_id"]
            entry_ids.append(entry_id)
        
        # Verify all IDs are unique
        assert len(entry_ids) == len(set(entry_ids))
    
    def test_create_memory_case_insensitive_sensitivity(self, client):
        """Test that sensitivity level is case-insensitive."""
        for sensitivity in ["PUBLIC", "Public", "public"]:
            payload = {
                "action": "User action",
                "context": {"test": "data"},
                "device_id": "laptop-001",
                "sensitivity": sensitivity
            }
            
            response = client.post("/api/v1/memory", json=payload)
            assert response.status_code == 201
