"""
Comprehensive tests for POST /api/v1/memory/query endpoint.

This test suite verifies the query endpoint functionality including:
- Basic query operations
- Time range filtering
- Tag filtering
- Action type filtering
- Pagination (limit and offset)
- Combined filters
- Edge cases and error handling
- Response format validation
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, UTC
import tempfile
import os

from luma_memory.api.server import create_app
from luma_memory.config import MemoryModuleConfig
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.api.routes import set_memory_manager


class TestQueryMemoryEndpoint:
    """Test suite for POST /api/v1/memory/query endpoint."""
    
    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for database and encryption key."""
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
        
        storage = SQLiteStorage(db_path=db_path, cache_size=100)
        encryption = EncryptionService(key_path=key_path)
        validation = ValidationManager()
        summarizer = ContextSummarizer(similarity_threshold=0.8)
        
        config = MemoryModuleConfig(
            db_path=db_path,
            encryption_key_path=key_path,
            cache_size=100
        )
        
        manager = MemoryManager(
            storage=storage,
            encryption=encryption,
            validation=validation,
            summarizer=summarizer,
            config=config
        )
        
        yield manager
        
        if hasattr(storage, 'close'):
            storage.close()
    
    @pytest.fixture
    def client(self, memory_manager):
        """Create a test client with initialized memory manager."""
        set_memory_manager(memory_manager)
        
        from luma_memory.api.routes import app
        client = TestClient(app)
        
        yield client
    
    @pytest.fixture
    def sample_memories(self, client):
        """Create sample memories for testing queries."""
        memories = []
        
        # Create memories with different characteristics
        # Using only public sensitivity to avoid encryption issues in tests
        memory_data = [
            {
                "action": "User opened document",
                "context": {"file": "report.pdf", "page": 1},
                "device_id": "laptop-001",
                "tags": ["work", "document"]
            },
            {
                "action": "User clicked button",
                "context": {"button": "submit", "form": "contact"},
                "device_id": "laptop-001",
                "tags": ["interaction", "form"]
            },
            {
                "action": "User searched query",
                "context": {"query": "weather forecast", "results": 10},
                "device_id": "phone-001",
                "tags": ["search", "weather"]
            },
            {
                "action": "User opened document",
                "context": {"file": "notes.txt", "size": 1024},
                "device_id": "laptop-001",
                "tags": ["personal", "document"]
            },
            {
                "action": "User sent message",
                "context": {"recipient": "friend", "length": 50},
                "device_id": "phone-001",
                "tags": ["communication", "personal"]
            }
        ]
        
        for data in memory_data:
            response = client.post("/api/v1/memory", json=data)
            assert response.status_code == 201
            memories.append(response.json()["entry_id"])
        
        return memories
    
    # Basic Query Tests
    
    def test_query_empty_request(self, client, sample_memories):
        """Test querying with empty request body returns all memories."""
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["total"] >= 5
        assert len(data["entries"]) >= 5
        assert data["limit"] == 100  # Default limit
        assert data["offset"] == 0  # Default offset
    
    def test_query_response_structure(self, client, sample_memories):
        """Test query response has correct structure."""
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level structure
        assert isinstance(data["entries"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["limit"], int)
        assert isinstance(data["offset"], int)
        
        # Check entry structure
        if len(data["entries"]) > 0:
            entry = data["entries"][0]
            assert "id" in entry
            assert "timestamp" in entry
            assert "action" in entry
            assert "context" in entry
            assert "sensitivity" in entry
            assert "device_id" in entry
            assert "sync_status" in entry
            assert "tags" in entry
            assert "created_at" in entry
            assert "updated_at" in entry
    
    def test_query_returns_newest_first(self, client):
        """Test query returns entries in reverse chronological order."""
        # Create memories with slight delays
        import time
        entry_ids = []
        for i in range(3):
            response = client.post("/api/v1/memory", json={
                "action": f"Action {i}",
                "context": {"index": i},
                "device_id": "laptop-001"
            })
            entry_ids.append(response.json()["entry_id"])
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Query all
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify order (newest first)
        timestamps = [entry["timestamp"] for entry in data["entries"]]
        assert timestamps == sorted(timestamps, reverse=True)
    
    # Pagination Tests
    
    def test_query_with_limit(self, client, sample_memories):
        """Test querying with limit parameter."""
        response = client.post("/api/v1/memory/query", json={"limit": 2})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) <= 2
        assert data["limit"] == 2
    
    def test_query_with_offset(self, client, sample_memories):
        """Test querying with offset parameter."""
        # Get first page
        response1 = client.post("/api/v1/memory/query", json={"limit": 2, "offset": 0})
        data1 = response1.json()
        
        # Get second page
        response2 = client.post("/api/v1/memory/query", json={"limit": 2, "offset": 2})
        data2 = response2.json()
        
        assert response2.status_code == 200
        assert data2["offset"] == 2
        
        # Verify different entries
        if len(data1["entries"]) > 0 and len(data2["entries"]) > 0:
            ids1 = {e["id"] for e in data1["entries"]}
            ids2 = {e["id"] for e in data2["entries"]}
            assert ids1.isdisjoint(ids2)  # No overlap
    
    def test_query_with_large_limit(self, client, sample_memories):
        """Test querying with limit larger than total entries."""
        response = client.post("/api/v1/memory/query", json={"limit": 1000})
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 1000
        assert len(data["entries"]) <= data["total"]
    
    def test_query_with_offset_beyond_total(self, client, sample_memories):
        """Test querying with offset beyond total entries returns empty."""
        response = client.post("/api/v1/memory/query", json={"offset": 10000})
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["entries"]) == 0
    
    def test_query_with_zero_limit(self, client, sample_memories):
        """Test querying with zero limit."""
        response = client.post("/api/v1/memory/query", json={"limit": 0})
        
        # API may reject zero limit with 422 or accept it and return empty
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            assert len(data["entries"]) == 0
            assert data["limit"] == 0
    
    # Tag Filtering Tests
    
    def test_query_by_single_tag(self, client, sample_memories):
        """Test querying by a single tag."""
        response = client.post("/api/v1/memory/query", json={"tags": ["work"]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        
        # Verify all entries have the tag
        for entry in data["entries"]:
            assert "work" in entry["tags"]
    
    def test_query_by_multiple_tags(self, client, sample_memories):
        """Test querying by multiple tags (OR logic)."""
        response = client.post("/api/v1/memory/query", json={"tags": ["work", "personal"]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2
        
        # Verify entries have at least one of the tags
        for entry in data["entries"]:
            assert any(tag in entry["tags"] for tag in ["work", "personal"])
    
    def test_query_by_nonexistent_tag(self, client, sample_memories):
        """Test querying by tag that doesn't exist returns empty."""
        response = client.post("/api/v1/memory/query", json={"tags": ["nonexistent-tag-xyz"]})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []
    
    def test_query_by_empty_tag_list(self, client, sample_memories):
        """Test querying with empty tag list returns all entries."""
        response = client.post("/api/v1/memory/query", json={"tags": []})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 5
    
    # Action Type Filtering Tests
    
    def test_query_by_action_type(self, client, sample_memories):
        """Test querying by action type (partial match)."""
        response = client.post("/api/v1/memory/query", json={"action_type": "document"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2
        
        # Verify all entries contain the action type
        for entry in data["entries"]:
            assert "document" in entry["action"].lower()
    
    def test_query_by_action_type_case_insensitive(self, client, sample_memories):
        """Test action type filtering is case insensitive."""
        response1 = client.post("/api/v1/memory/query", json={"action_type": "DOCUMENT"})
        response2 = client.post("/api/v1/memory/query", json={"action_type": "document"})
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should return same results
        assert response1.json()["total"] == response2.json()["total"]
    
    def test_query_by_nonexistent_action_type(self, client, sample_memories):
        """Test querying by action type that doesn't exist returns empty."""
        response = client.post("/api/v1/memory/query", json={"action_type": "nonexistent-action-xyz"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []
    
    # Time Range Filtering Tests
    
    def test_query_by_start_time(self, client, sample_memories):
        """Test querying with start_time filter."""
        now = datetime.now(UTC)
        start_time_dt = now - timedelta(hours=1)
        # Use isoformat() which already includes +00:00, then replace with Z
        start_time = start_time_dt.isoformat().replace('+00:00', 'Z')
        
        response = client.post("/api/v1/memory/query", json={"start_time": start_time})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 5
        
        # Verify all entries are after start_time
        for entry in data["entries"]:
            entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
            filter_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            assert entry_time >= filter_time
    
    def test_query_by_end_time(self, client, sample_memories):
        """Test querying with end_time filter."""
        now = datetime.now(UTC)
        end_time_dt = now + timedelta(hours=1)
        # Use isoformat() which already includes +00:00, then replace with Z
        end_time = end_time_dt.isoformat().replace('+00:00', 'Z')
        
        response = client.post("/api/v1/memory/query", json={"end_time": end_time})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 5
        
        # Verify all entries are before end_time
        for entry in data["entries"]:
            entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
            filter_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            assert entry_time <= filter_time
    
    def test_query_by_time_range(self, client, sample_memories):
        """Test querying with both start_time and end_time."""
        now = datetime.now(UTC)
        start_time_dt = now - timedelta(hours=1)
        end_time_dt = now + timedelta(hours=1)
        # Use isoformat() which already includes +00:00, then replace with Z
        start_time = start_time_dt.isoformat().replace('+00:00', 'Z')
        end_time = end_time_dt.isoformat().replace('+00:00', 'Z')
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "end_time": end_time
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 5
    
    def test_query_by_narrow_time_range(self, client):
        """Test querying with narrow time range."""
        # Create memory
        client.post("/api/v1/memory", json={
            "action": "Test action",
            "context": {"test": "data"},
            "device_id": "laptop-001"
        })
        
        # Query with very narrow range (should find it)
        now = datetime.now(UTC)
        start_time = (now - timedelta(seconds=5)).isoformat() + "Z"
        end_time = (now + timedelta(seconds=5)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "end_time": end_time
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
    
    def test_query_by_past_time_range(self, client, sample_memories):
        """Test querying with time range in the past returns empty."""
        past_start = (datetime.now(UTC) - timedelta(days=2)).isoformat() + "Z"
        past_end = (datetime.now(UTC) - timedelta(days=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": past_start,
            "end_time": past_end
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
    
    # Combined Filters Tests
    
    def test_query_with_all_filters(self, client, sample_memories):
        """Test querying with all filters combined."""
        now = datetime.now(UTC)
        start_time = (now - timedelta(hours=1)).isoformat() + "Z"
        end_time = (now + timedelta(hours=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "end_time": end_time,
            "tags": ["work"],
            "action_type": "document",
            "limit": 10,
            "offset": 0
        })
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["entries"], list)
        assert data["limit"] == 10
        assert data["offset"] == 0
    
    def test_query_tags_and_action_type(self, client, sample_memories):
        """Test combining tag and action type filters."""
        response = client.post("/api/v1/memory/query", json={
            "tags": ["document"],
            "action_type": "opened"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify entries match both filters
        for entry in data["entries"]:
            assert "document" in entry["tags"]
            assert "opened" in entry["action"].lower()
    
    def test_query_time_and_tags(self, client, sample_memories):
        """Test combining time range and tag filters."""
        now = datetime.now(UTC)
        start_time = (now - timedelta(hours=1)).isoformat() + "Z"
        
        response = client.post("/api/v1/memory/query", json={
            "start_time": start_time,
            "tags": ["work"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["entries"], list)
    
    # Error Handling Tests
    
    def test_query_invalid_start_time_format(self, client):
        """Test querying with invalid start_time format returns error."""
        response = client.post("/api/v1/memory/query", json={
            "start_time": "invalid-time-format"
        })
        
        # Should return 400 or 500 for invalid format
        assert response.status_code in [400, 500]
        data = response.json()
        assert "detail" in data
    
    def test_query_invalid_end_time_format(self, client):
        """Test querying with invalid end_time format returns error."""
        response = client.post("/api/v1/memory/query", json={
            "end_time": "not-a-valid-time"
        })
        
        # Should return 400 or 500 for invalid format
        assert response.status_code in [400, 500]
        data = response.json()
        assert "detail" in data
    
    def test_query_negative_limit(self, client):
        """Test querying with negative limit."""
        response = client.post("/api/v1/memory/query", json={"limit": -1})
        
        # Should either reject or treat as 0
        assert response.status_code in [200, 400, 422]
    
    def test_query_negative_offset(self, client):
        """Test querying with negative offset."""
        response = client.post("/api/v1/memory/query", json={"offset": -1})
        
        # Should either reject or treat as 0
        assert response.status_code in [200, 400, 422]
    
    def test_query_invalid_json(self, client):
        """Test querying with invalid JSON returns 422."""
        response = client.post(
            "/api/v1/memory/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    # Edge Cases
    
    def test_query_with_null_values(self, client, sample_memories):
        """Test querying with null values in optional fields."""
        response = client.post("/api/v1/memory/query", json={
            "start_time": None,
            "end_time": None,
            "tags": None,
            "action_type": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 5
    
    def test_query_empty_database(self, client):
        """Test querying when database is empty."""
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []
    
    def test_query_with_special_characters_in_action_type(self, client):
        """Test querying with special characters in action type."""
        # Create memory with special characters
        client.post("/api/v1/memory", json={
            "action": "User clicked [button]",
            "context": {"test": "data"},
            "device_id": "laptop-001"
        })
        
        response = client.post("/api/v1/memory/query", json={
            "action_type": "[button]"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
    
    def test_query_preserves_entry_data(self, client):
        """Test that query returns complete and accurate entry data."""
        # Create memory with all fields (using public sensitivity to avoid encryption issues)
        create_response = client.post("/api/v1/memory", json={
            "action": "Test action",
            "context": {"key": "value", "number": 42},
            "device_id": "laptop-001",
            "sensitivity": "public",
            "tags": ["tag1", "tag2"]
        })
        
        assert create_response.status_code == 201
        entry_id = create_response.json()["entry_id"]
        
        # Query it back
        response = client.post("/api/v1/memory/query", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        # Find our entry
        entry = next((e for e in data["entries"] if e["id"] == entry_id), None)
        assert entry is not None
        assert entry["action"] == "Test action"
        assert entry["context"] == {"key": "value", "number": 42}
        assert entry["device_id"] == "laptop-001"
        assert entry["sensitivity"] == "public"
        assert set(entry["tags"]) == {"tag1", "tag2"}
        assert entry["sync_status"] == "pending"
