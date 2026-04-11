"""
Basic smoke test to verify the system works.

Run with: python -m pytest luma/test_basic.py -v
"""

from fastapi.testclient import TestClient
from luma.main import app, get_db, Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Create test database
test_engine = create_engine("sqlite:///:memory:")
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Override dependency
app.dependency_overrides[get_db] = override_get_db

# Create tables
Base.metadata.create_all(bind=test_engine)

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Luma is alive"}


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_memory():
    """Test creating a memory."""
    response = client.post(
        "/api/v1/memories",
        json={"content": "Test memory", "metadata": {"source": "test"}}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "Test memory"
    assert data["metadata"]["source"] == "test"
    assert "id" in data
    assert "created_at" in data


def test_create_memory_validation():
    """Test memory creation validation."""
    # Empty content should fail
    response = client.post(
        "/api/v1/memories",
        json={"content": "", "metadata": {}}
    )
    assert response.status_code == 400
    
    # Whitespace-only content should fail
    response = client.post(
        "/api/v1/memories",
        json={"content": "   ", "metadata": {}}
    )
    assert response.status_code == 400


def test_get_memory():
    """Test retrieving a memory."""
    # Create a memory first
    create_response = client.post(
        "/api/v1/memories",
        json={"content": "Test memory for retrieval", "metadata": {}}
    )
    memory_id = create_response.json()["id"]
    
    # Retrieve it
    response = client.get(f"/api/v1/memories/{memory_id}")
    assert response.status_code == 200
    assert response.json()["content"] == "Test memory for retrieval"


def test_get_nonexistent_memory():
    """Test retrieving a non-existent memory."""
    response = client.get("/api/v1/memories/99999")
    assert response.status_code == 404


def test_list_memories():
    """Test listing memories."""
    # Create a few memories
    for i in range(3):
        client.post(
            "/api/v1/memories",
            json={"content": f"Memory {i}", "metadata": {}}
        )
    
    # List them
    response = client.get("/api/v1/memories")
    assert response.status_code == 200
    memories = response.json()
    assert len(memories) >= 3


def test_update_memory():
    """Test updating a memory."""
    # Create a memory
    create_response = client.post(
        "/api/v1/memories",
        json={"content": "Original content", "metadata": {}}
    )
    memory_id = create_response.json()["id"]
    
    # Update it
    response = client.put(
        f"/api/v1/memories/{memory_id}",
        json={"content": "Updated content", "metadata": {"updated": True}}
    )
    assert response.status_code == 200
    assert response.json()["content"] == "Updated content"
    assert response.json()["metadata"]["updated"] is True


def test_delete_memory():
    """Test deleting a memory."""
    # Create a memory
    create_response = client.post(
        "/api/v1/memories",
        json={"content": "Memory to delete", "metadata": {}}
    )
    memory_id = create_response.json()["id"]
    
    # Delete it
    response = client.delete(f"/api/v1/memories/{memory_id}")
    assert response.status_code == 204
    
    # Verify it's gone
    response = client.get(f"/api/v1/memories/{memory_id}")
    assert response.status_code == 404


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
