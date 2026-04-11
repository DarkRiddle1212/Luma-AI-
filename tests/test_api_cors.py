"""
Test for CORS middleware configuration (Task 11.10).

This test verifies that CORS headers are properly set for cross-origin requests.
"""

import pytest
from fastapi.testclient import TestClient

from luma_memory.api.routes import app


class TestCORSMiddleware:
    """Test suite for CORS middleware configuration."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_cors_headers_on_get_request(self, client):
        """Test that CORS headers are present on GET requests."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://example.com"}
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    def test_cors_preflight_request(self, client):
        """Test that CORS preflight OPTIONS requests are handled correctly."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_cors_headers_on_post_request(self, client):
        """Test that CORS headers are present on POST requests."""
        response = client.post(
            "/api/v1/memory",
            json={
                "action": "test action",
                "context": {"key": "value"},
                "device_id": "test-device"
            },
            headers={"Origin": "http://example.com"}
        )
        
        # Response may be 503 if memory manager not initialized, but CORS headers should still be present
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    def test_cors_allows_credentials(self, client):
        """Test that CORS allows credentials."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://example.com"}
        )
        
        assert "access-control-allow-credentials" in response.headers
        assert response.headers["access-control-allow-credentials"] == "true"
