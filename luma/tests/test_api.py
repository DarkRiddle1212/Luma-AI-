"""
Unit Tests - API Layer

Tests for FastAPI routes in the Luma system.
Covers all API endpoints including success cases, validation errors, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from luma.memory.service import ValidationError, NotFoundError


# ============================================================================
# 17.1 Test POST /memories endpoint (success case)
# ============================================================================

class TestCreateMemorySuccess:
    """Tests for successful memory creation via POST /memories."""
    
    def test_create_memory_with_all_fields(self, test_client: TestClient):
        """Test creating a memory with content and metadata."""
        payload = {
            "content": "Test memory content",
            "metadata": {"source": "test", "priority": "high"}
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None
        assert data["content"] == "Test memory content"
        assert data["metadata"] == {"source": "test", "priority": "high"}
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_create_memory_with_minimal_fields(self, test_client: TestClient):
        """Test creating a memory with only required content field."""
        payload = {
            "content": "Minimal memory"
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None
        assert data["content"] == "Minimal memory"
        assert data["metadata"] == {}
    
    def test_create_memory_with_empty_metadata(self, test_client: TestClient):
        """Test creating a memory with explicitly empty metadata."""
        payload = {
            "content": "Memory with empty metadata",
            "metadata": {}
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"] == {}
    
    def test_create_memory_returns_timestamps(self, test_client: TestClient):
        """Test that created memory includes created_at and updated_at timestamps."""
        payload = {
            "content": "Timestamp test"
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "created_at" in data
        assert "updated_at" in data
        # Verify timestamps are valid ISO format
        datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))


# ============================================================================
# 17.2 Test POST /memories validation errors
# ============================================================================

class TestCreateMemoryValidation:
    """Tests for validation errors when creating memories."""
    
    def test_create_memory_with_empty_content(self, test_client: TestClient):
        """Test that empty content returns 422 Pydantic validation error."""
        payload = {
            "content": ""
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_create_memory_with_whitespace_only_content(self, test_client: TestClient):
        """Test that whitespace-only content returns 400 validation error."""
        payload = {
            "content": "   "
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_create_memory_with_tabs_and_newlines(self, test_client: TestClient):
        """Test that tabs and newlines only returns 400 validation error."""
        payload = {
            "content": "\t\n\r"
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_create_memory_missing_content_field(self, test_client: TestClient):
        """Test that missing content field returns 422 Pydantic validation error."""
        payload = {
            "metadata": {"test": "data"}
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 422
    
    def test_create_memory_with_invalid_metadata_type(self, test_client: TestClient):
        """Test that non-dict metadata returns 422 Pydantic validation error."""
        payload = {
            "content": "Test content",
            "metadata": "not a dictionary"
        }
        
        response = test_client.post("/api/v1/memories", json=payload)
        
        assert response.status_code == 422


# ============================================================================
# 17.3 Test GET /memories/{id} endpoint
# ============================================================================

class TestGetMemory:
    """Tests for retrieving a memory by ID via GET /memories/{id}."""
    
    def test_get_memory_success(self, test_client: TestClient):
        """Test successfully retrieving an existing memory."""
        # Create a memory first
        create_payload = {
            "content": "Memory to retrieve",
            "metadata": {"key": "value"}
        }
        create_response = test_client.post("/api/v1/memories", json=create_payload)
        memory_id = create_response.json()["id"]
        
        # Retrieve it
        response = test_client.get(f"/api/v1/memories/{memory_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Memory to retrieve"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_get_memory_with_complex_metadata(self, test_client: TestClient):
        """Test retrieving memory with complex nested metadata."""
        create_payload = {
            "content": "Complex metadata test",
            "metadata": {
                "nested": {
                    "level1": {
                        "level2": "deep value"
                    }
                },
                "list": [1, 2, 3],
                "string": "value"
            }
        }
        create_response = test_client.post("/api/v1/memories", json=create_payload)
        memory_id = create_response.json()["id"]
        
        # Retrieve it
        response = test_client.get(f"/api/v1/memories/{memory_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"] == create_payload["metadata"]


# ============================================================================
# 17.4 Test GET /memories/{id} not found error
# ============================================================================

class TestGetMemoryNotFound:
    """Tests for 404 errors when retrieving non-existent memories."""
    
    def test_get_memory_nonexistent_id(self, test_client: TestClient):
        """Test that retrieving non-existent memory returns 404."""
        response = test_client.get("/api/v1/memories/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
        assert "99999" in data["detail"]
    
    def test_get_memory_negative_id(self, test_client: TestClient):
        """Test that retrieving memory with negative ID returns 404."""
        response = test_client.get("/api/v1/memories/-1")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_get_memory_zero_id(self, test_client: TestClient):
        """Test that retrieving memory with ID 0 returns 404."""
        response = test_client.get("/api/v1/memories/0")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


# ============================================================================
# 17.5 Test GET /memories list endpoint
# ============================================================================

class TestListMemories:
    """Tests for listing memories via GET /memories."""
    
    def test_list_memories_empty(self, test_client: TestClient):
        """Test listing memories when none exist."""
        response = test_client.get("/api/v1/memories")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_list_memories_with_data(self, test_client: TestClient):
        """Test listing memories when data exists."""
        # Create multiple memories
        test_client.post("/api/v1/memories", json={"content": "Memory 1"})
        test_client.post("/api/v1/memories", json={"content": "Memory 2"})
        test_client.post("/api/v1/memories", json={"content": "Memory 3"})
        
        # List all
        response = test_client.get("/api/v1/memories")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
    
    def test_list_memories_with_limit(self, test_client: TestClient):
        """Test listing memories with limit parameter."""
        # Create 5 memories
        for i in range(5):
            test_client.post("/api/v1/memories", json={"content": f"Memory {i}"})
        
        # List with limit=2
        response = test_client.get("/api/v1/memories?limit=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
    
    def test_list_memories_with_skip(self, test_client: TestClient):
        """Test listing memories with skip parameter."""
        # Create 5 memories
        for i in range(5):
            test_client.post("/api/v1/memories", json={"content": f"Memory {i}"})
        
        # Skip first 2
        response = test_client.get("/api/v1/memories?skip=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
    
    def test_list_memories_with_skip_and_limit(self, test_client: TestClient):
        """Test listing memories with both skip and limit."""
        # Create 10 memories
        for i in range(10):
            test_client.post("/api/v1/memories", json={"content": f"Memory {i}"})
        
        # Get page 2 (skip 3, limit 3)
        response = test_client.get("/api/v1/memories?skip=3&limit=3")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
    
    def test_list_memories_ordered_by_created_at_desc(self, test_client: TestClient):
        """Test that memories are returned in descending order by created_at."""
        # Create memories
        response1 = test_client.post("/api/v1/memories", json={"content": "First"})
        response2 = test_client.post("/api/v1/memories", json={"content": "Second"})
        response3 = test_client.post("/api/v1/memories", json={"content": "Third"})
        
        id1 = response1.json()["id"]
        id2 = response2.json()["id"]
        id3 = response3.json()["id"]
        
        # List all
        response = test_client.get("/api/v1/memories")
        
        assert response.status_code == 200
        data = response.json()
        # Should be in reverse order (newest first)
        assert data[0]["id"] == id3
        assert data[1]["id"] == id2
        assert data[2]["id"] == id1


# ============================================================================
# 17.6 Test PUT /memories/{id} endpoint
# ============================================================================

class TestUpdateMemory:
    """Tests for updating memories via PUT /memories/{id}."""
    
    def test_update_memory_success(self, test_client: TestClient):
        """Test successfully updating an existing memory."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "Original content",
            "metadata": {"key": "original"}
        })
        memory_id = create_response.json()["id"]
        
        # Update it
        update_payload = {
            "content": "Updated content",
            "metadata": {"key": "updated"}
        }
        response = test_client.put(f"/api/v1/memories/{memory_id}", json=update_payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Updated content"
        assert data["metadata"] == {"key": "updated"}
    
    def test_update_memory_content_only(self, test_client: TestClient):
        """Test updating only the content field."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "Original",
            "metadata": {"keep": "this"}
        })
        memory_id = create_response.json()["id"]
        
        # Update content only
        update_payload = {
            "content": "New content",
            "metadata": {}
        }
        response = test_client.put(f"/api/v1/memories/{memory_id}", json=update_payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "New content"
        assert data["metadata"] == {}
    
    def test_update_memory_validation_error(self, test_client: TestClient):
        """Test that updating with invalid content returns 422."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "Original"
        })
        memory_id = create_response.json()["id"]
        
        # Try to update with empty content
        update_payload = {
            "content": ""
        }
        response = test_client.put(f"/api/v1/memories/{memory_id}", json=update_payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_update_memory_not_found(self, test_client: TestClient):
        """Test that updating non-existent memory returns 404."""
        update_payload = {
            "content": "New content"
        }
        response = test_client.put("/api/v1/memories/99999", json=update_payload)
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_update_memory_with_whitespace_content(self, test_client: TestClient):
        """Test that updating with whitespace-only content returns 400."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "Original"
        })
        memory_id = create_response.json()["id"]
        
        # Try to update with whitespace
        update_payload = {
            "content": "   "
        }
        response = test_client.put(f"/api/v1/memories/{memory_id}", json=update_payload)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data


# ============================================================================
# 17.7 Test DELETE /memories/{id} endpoint
# ============================================================================

class TestDeleteMemory:
    """Tests for deleting memories via DELETE /memories/{id}."""
    
    def test_delete_memory_success(self, test_client: TestClient):
        """Test successfully deleting an existing memory."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "To be deleted"
        })
        memory_id = create_response.json()["id"]
        
        # Delete it
        response = test_client.delete(f"/api/v1/memories/{memory_id}")
        
        assert response.status_code == 204
        assert response.content == b""
        
        # Verify it's gone
        get_response = test_client.get(f"/api/v1/memories/{memory_id}")
        assert get_response.status_code == 404
    
    def test_delete_memory_not_found(self, test_client: TestClient):
        """Test that deleting non-existent memory returns 404."""
        response = test_client.delete("/api/v1/memories/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_delete_memory_idempotency(self, test_client: TestClient):
        """Test that deleting the same memory twice returns 404 on second attempt."""
        # Create a memory
        create_response = test_client.post("/api/v1/memories", json={
            "content": "To be deleted"
        })
        memory_id = create_response.json()["id"]
        
        # Delete it first time
        response1 = test_client.delete(f"/api/v1/memories/{memory_id}")
        assert response1.status_code == 204
        
        # Try to delete again
        response2 = test_client.delete(f"/api/v1/memories/{memory_id}")
        assert response2.status_code == 404
    
    def test_delete_memory_does_not_affect_others(self, test_client: TestClient):
        """Test that deleting one memory doesn't affect others."""
        # Create multiple memories
        response1 = test_client.post("/api/v1/memories", json={"content": "Memory 1"})
        response2 = test_client.post("/api/v1/memories", json={"content": "Memory 2"})
        response3 = test_client.post("/api/v1/memories", json={"content": "Memory 3"})
        
        id1 = response1.json()["id"]
        id2 = response2.json()["id"]
        id3 = response3.json()["id"]
        
        # Delete the middle one
        delete_response = test_client.delete(f"/api/v1/memories/{id2}")
        assert delete_response.status_code == 204
        
        # Verify others still exist
        get_response1 = test_client.get(f"/api/v1/memories/{id1}")
        assert get_response1.status_code == 200
        
        get_response3 = test_client.get(f"/api/v1/memories/{id3}")
        assert get_response3.status_code == 200
        
        # Verify deleted one is gone
        get_response2 = test_client.get(f"/api/v1/memories/{id2}")
        assert get_response2.status_code == 404
