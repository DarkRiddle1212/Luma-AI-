"""
Tests for Pydantic request/response models.

This test suite verifies that all Pydantic models used in the API
properly validate input data and serialize/deserialize correctly.
"""

import pytest
from pydantic import ValidationError

from luma_memory.api.routes import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    MemoryEntryResponse,
    QueryMemoryRequest,
    QueryMemoryResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)


class TestPydanticModels:
    """Test suite for Pydantic models validation."""
    
    def test_create_memory_request_valid(self):
        """Test CreateMemoryRequest with valid data."""
        data = {
            "action": "User opened document",
            "context": {"file": "report.pdf", "page": 1},
            "device_id": "laptop-001",
            "sensitivity": "private",
            "tags": ["document", "work"]
        }
        request = CreateMemoryRequest(**data)
        assert request.action == "User opened document"
        assert request.context == {"file": "report.pdf", "page": 1}
        assert request.device_id == "laptop-001"
        assert request.sensitivity == "private"
        assert request.tags == ["document", "work"]
    
    def test_create_memory_request_minimal(self):
        """Test CreateMemoryRequest with minimal required fields."""
        data = {
            "action": "Test action",
            "context": {},
            "device_id": "device-001"
        }
        request = CreateMemoryRequest(**data)
        assert request.action == "Test action"
        assert request.context == {}
        assert request.device_id == "device-001"
        assert request.sensitivity == "public"  # default value
        assert request.tags == []  # default value
    
    def test_create_memory_request_missing_action(self):
        """Test CreateMemoryRequest fails without action."""
        data = {
            "context": {},
            "device_id": "device-001"
        }
        with pytest.raises(ValidationError) as exc_info:
            CreateMemoryRequest(**data)
        assert "action" in str(exc_info.value)
    
    def test_create_memory_request_empty_action(self):
        """Test CreateMemoryRequest fails with empty action."""
        data = {
            "action": "",
            "context": {},
            "device_id": "device-001"
        }
        with pytest.raises(ValidationError) as exc_info:
            CreateMemoryRequest(**data)
        assert "action" in str(exc_info.value)
    
    def test_create_memory_request_missing_context(self):
        """Test CreateMemoryRequest fails without context."""
        data = {
            "action": "Test action",
            "device_id": "device-001"
        }
        with pytest.raises(ValidationError) as exc_info:
            CreateMemoryRequest(**data)
        assert "context" in str(exc_info.value)
    
    def test_create_memory_request_missing_device_id(self):
        """Test CreateMemoryRequest fails without device_id."""
        data = {
            "action": "Test action",
            "context": {}
        }
        with pytest.raises(ValidationError) as exc_info:
            CreateMemoryRequest(**data)
        assert "device_id" in str(exc_info.value)
    
    def test_create_memory_response_valid(self):
        """Test CreateMemoryResponse with valid data."""
        data = {
            "entry_id": "abc-123-def-456",
            "message": "Memory entry created successfully"
        }
        response = CreateMemoryResponse(**data)
        assert response.entry_id == "abc-123-def-456"
        assert response.message == "Memory entry created successfully"
    
    def test_memory_entry_response_valid(self):
        """Test MemoryEntryResponse with valid data."""
        data = {
            "id": "abc-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "action": "User opened document",
            "context": {"file": "report.pdf"},
            "sensitivity": "private",
            "device_id": "laptop-001",
            "sync_status": "pending",
            "tags": ["document", "work"]
        }
        response = MemoryEntryResponse(**data)
        assert response.id == "abc-123"
        assert response.timestamp == "2024-01-15T10:30:00Z"
        assert response.action == "User opened document"
        assert response.context == {"file": "report.pdf"}
        assert response.sensitivity == "private"
        assert response.device_id == "laptop-001"
        assert response.sync_status == "pending"
        assert response.tags == ["document", "work"]
        assert response.summary is None
        assert response.parent_id is None
    
    def test_query_memory_request_valid(self):
        """Test QueryMemoryRequest with valid data."""
        data = {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-31T23:59:59Z",
            "tags": ["work"],
            "action_type": "document",
            "limit": 50,
            "offset": 10
        }
        request = QueryMemoryRequest(**data)
        assert request.start_time == "2024-01-01T00:00:00Z"
        assert request.end_time == "2024-01-31T23:59:59Z"
        assert request.tags == ["work"]
        assert request.action_type == "document"
        assert request.limit == 50
        assert request.offset == 10
    
    def test_query_memory_request_defaults(self):
        """Test QueryMemoryRequest with default values."""
        request = QueryMemoryRequest()
        assert request.start_time is None
        assert request.end_time is None
        assert request.tags is None
        assert request.action_type is None
        assert request.limit == 100
        assert request.offset == 0
    
    def test_query_memory_request_limit_validation(self):
        """Test QueryMemoryRequest limit validation."""
        # Test limit too low
        with pytest.raises(ValidationError) as exc_info:
            QueryMemoryRequest(limit=0)
        assert "limit" in str(exc_info.value)
        
        # Test limit too high
        with pytest.raises(ValidationError) as exc_info:
            QueryMemoryRequest(limit=1001)
        assert "limit" in str(exc_info.value)
    
    def test_query_memory_request_offset_validation(self):
        """Test QueryMemoryRequest offset validation."""
        # Test negative offset
        with pytest.raises(ValidationError) as exc_info:
            QueryMemoryRequest(offset=-1)
        assert "offset" in str(exc_info.value)
    
    def test_query_memory_response_valid(self):
        """Test QueryMemoryResponse with valid data."""
        entry_data = {
            "id": "abc-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "action": "Test action",
            "context": {},
            "sensitivity": "public",
            "device_id": "device-001",
            "sync_status": "pending",
            "tags": []
        }
        data = {
            "entries": [MemoryEntryResponse(**entry_data)],
            "total": 1,
            "limit": 100,
            "offset": 0
        }
        response = QueryMemoryResponse(**data)
        assert len(response.entries) == 1
        assert response.total == 1
        assert response.limit == 100
        assert response.offset == 0
    
    def test_health_response_valid(self):
        """Test HealthResponse with valid data."""
        data = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z"
        }
        response = HealthResponse(**data)
        assert response.status == "healthy"
        assert response.timestamp == "2024-01-15T10:30:00Z"
    
    def test_stats_response_valid(self):
        """Test StatsResponse with valid data."""
        data = {
            "total_entries": 1000,
            "storage_size_bytes": 1048576,
            "encryption_enabled": True,
            "summarizer_enabled": True,
            "config": {"cache_size": 1000}
        }
        response = StatsResponse(**data)
        assert response.total_entries == 1000
        assert response.storage_size_bytes == 1048576
        assert response.encryption_enabled is True
        assert response.summarizer_enabled is True
        assert response.config == {"cache_size": 1000}
        assert response.performance is None
    
    def test_error_response_valid(self):
        """Test ErrorResponse with valid data."""
        data = {
            "error": "Validation failed",
            "detail": "Action cannot be empty"
        }
        response = ErrorResponse(**data)
        assert response.error == "Validation failed"
        assert response.detail == "Action cannot be empty"
    
    def test_error_response_without_detail(self):
        """Test ErrorResponse without detail field."""
        data = {
            "error": "Not found"
        }
        response = ErrorResponse(**data)
        assert response.error == "Not found"
        assert response.detail is None
    
    def test_model_serialization(self):
        """Test that models can be serialized to dict."""
        request = CreateMemoryRequest(
            action="Test",
            context={"key": "value"},
            device_id="device-001"
        )
        data = request.model_dump()
        assert data["action"] == "Test"
        assert data["context"] == {"key": "value"}
        assert data["device_id"] == "device-001"
        assert data["sensitivity"] == "public"
        assert data["tags"] == []
    
    def test_model_json_serialization(self):
        """Test that models can be serialized to JSON."""
        request = CreateMemoryRequest(
            action="Test",
            context={"key": "value"},
            device_id="device-001"
        )
        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test" in json_str
        assert "device-001" in json_str
