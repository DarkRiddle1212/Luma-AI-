"""
Test for GET /api/v1/health endpoint (Task 11.5).

This test verifies that the health check endpoint returns the correct
status and timestamp information.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from luma_memory.api.routes import app


class TestHealthEndpoint:
    """Test suite for GET /api/v1/health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_health_check_success(self, client):
        """Test that health check returns healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
        # Verify timestamp is in ISO format with Z suffix
        timestamp = data["timestamp"]
        assert timestamp.endswith("Z")
        
        # Verify timestamp can be parsed
        timestamp_without_z = timestamp[:-1]
        parsed_time = datetime.fromisoformat(timestamp_without_z)
        assert isinstance(parsed_time, datetime)
    
    def test_health_check_response_structure(self, client):
        """Test that health check response has correct structure."""
        response = client.get("/api/v1/health")
        
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 2  # Should have exactly 2 fields
        assert "status" in data
        assert "timestamp" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
