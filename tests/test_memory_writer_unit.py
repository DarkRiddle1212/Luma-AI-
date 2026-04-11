"""Unit tests for Memory Writer component.

Tests the memory persistence and deduplication logic of the MemoryWriter class.
"""

import pytest
from datetime import datetime
from luma.core.memory_write.memory_writer import MemoryWriter, calculate_similarity
from luma.core.memory_write.schemas import ScoredMemory, StoredMemory
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters, RetrievalResult
from typing import Dict, List, Optional, Any


class MockMemoryStore(MemoryInterface):
    """Mock memory store for testing."""
    
    def __init__(self):
        self.stored_memories: List[MemoryEntry] = []
        self.next_id = 1
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory and return its ID."""
        metadata = metadata or {}
        
        # Check if updating existing memory (id in metadata)
        if "id" in metadata:
            memory_id = metadata["id"]
            # Find and update existing memory
            for i, mem in enumerate(self.stored_memories):
                if mem["id"] == memory_id:
                    self.stored_memories[i] = {
                        "id": memory_id,
                        "content": content,
                        "metadata": metadata,
                        "timestamp": mem["timestamp"],  # Preserve original
                        "category": metadata.get("category", "conversation_memory"),
                        "tags": metadata.get("tags", [])
                    }
                    return memory_id
        
        # Create new memory
        memory_id = f"mem_{self.next_id}"
        self.next_id += 1
        
        memory_entry: MemoryEntry = {
            "id": memory_id,
            "content": content,
            "metadata": metadata,
            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
            "category": metadata.get("category", "conversation_memory"),
            "tags": metadata.get("tags", [])
        }
        
        self.stored_memories.append(memory_entry)
        return memory_id
    
    def retrieve(
        self,
        query: Optional[str] = None,
        params: Optional[QueryParameters] = None,
        limit: int = 10,
        metrics_collector=None,
        logger=None
    ) -> RetrievalResult:
        """Retrieve memories matching the query."""
        # Simple implementation: return all memories for testing
        result: RetrievalResult = {
            "memories": self.stored_memories[:limit],
            "total_count": len(self.stored_memories[:limit]),
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }
        return result


class TestCalculateSimilarity:
    """Tests for the calculate_similarity helper function."""
    
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        text = "I want to build a web application"
        similarity = calculate_similarity(text, text)
        assert similarity == 1.0
    
    def test_completely_different_texts(self):
        """Test similarity of completely different texts."""
        text1 = "I want to build a web application"
        text2 = "The weather is nice today"
        similarity = calculate_similarity(text1, text2)
        assert similarity < 0.4  # Very low similarity
    
    def test_similar_texts(self):
        """Test similarity of similar texts."""
        text1 = "I want to build a web application"
        text2 = "I want to build a web app"
        similarity = calculate_similarity(text1, text2)
        assert similarity > 0.8  # High similarity
    
    def test_case_insensitive(self):
        """Test that similarity is case-insensitive."""
        text1 = "I WANT TO BUILD A WEB APPLICATION"
        text2 = "i want to build a web application"
        similarity = calculate_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        text1 = "  I want to build a web application  "
        text2 = "I want to build a web application"
        similarity = calculate_similarity(text1, text2)
        assert similarity == 1.0


class TestMemoryWriter:
    """Tests for the MemoryWriter class."""
    
    def test_initialization(self):
        """Test MemoryWriter initialization."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        assert writer.memory_store is mock_store
        assert writer.similarity_threshold == 0.9
    
    def test_initialization_with_custom_threshold(self):
        """Test MemoryWriter initialization with custom threshold."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.85)
        
        assert writer.similarity_threshold == 0.85
    
    def test_initialization_invalid_threshold(self):
        """Test MemoryWriter initialization with invalid threshold."""
        mock_store = MockMemoryStore()
        
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            MemoryWriter(memory_store=mock_store, similarity_threshold=1.5)
        
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            MemoryWriter(memory_store=mock_store, similarity_threshold=-0.1)
    
    def test_store_new_memory(self):
        """Test storing a new memory."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        scored_memory = ScoredMemory(
            text="I want to build a web application",
            type="project_goal",
            importance=0.85
        )
        
        result = writer.store_memory(scored_memory)
        
        assert isinstance(result, StoredMemory)
        assert result.text == scored_memory.text
        assert result.type == scored_memory.type
        assert result.importance == scored_memory.importance
        assert result.is_update is False
        assert result.memory_id.startswith("mem_")
        assert len(mock_store.stored_memories) == 1
    
    def test_store_memory_with_none_input(self):
        """Test storing None memory raises ValueError."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        with pytest.raises(ValueError, match="scored_memory cannot be None"):
            writer.store_memory(None)
    
    def test_memory_entry_field_mapping(self):
        """Test MemoryEntry creation with correct field mapping."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        scored_memory = ScoredMemory(
            text="I prefer using Python for development",
            type="user_preference",
            importance=0.75
        )
        
        result = writer.store_memory(scored_memory)
        
        # Check stored memory in mock store
        stored = mock_store.stored_memories[0]
        assert stored["content"] == scored_memory.text
        assert stored["metadata"]["importance"] == scored_memory.importance
        assert stored["category"] == "conversation_memory"
        assert scored_memory.type in stored["tags"]
        assert "timestamp" in stored
    
    def test_unique_id_generation(self):
        """Test that unique IDs are generated for new memories."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        memory1 = ScoredMemory(text="First memory", type="fact", importance=0.6)
        memory2 = ScoredMemory(text="Second memory", type="fact", importance=0.6)
        
        result1 = writer.store_memory(memory1)
        result2 = writer.store_memory(memory2)
        
        assert result1.memory_id != result2.memory_id
        assert len(mock_store.stored_memories) == 2
    
    def test_detect_similar_memory(self):
        """Test detection of similar existing memory."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # Store initial memory
        first_memory = ScoredMemory(
            text="I want to build a web application",
            type="project_goal",
            importance=0.8
        )
        writer.store_memory(first_memory)
        
        # Try to store very similar memory
        similar_memory = ScoredMemory(
            text="I want to build a web application",  # Identical
            type="project_goal",
            importance=0.85
        )
        result = writer.store_memory(similar_memory)
        
        # Should update existing memory, not create new one
        assert result.is_update is True
        assert len(mock_store.stored_memories) == 1
    
    def test_update_existing_memory_when_similarity_high(self):
        """Test updating existing memory when similarity > 0.9."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # Store initial memory
        first_memory = ScoredMemory(
            text="I want to build a web application",
            type="project_goal",
            importance=0.8
        )
        result1 = writer.store_memory(first_memory)
        original_timestamp = result1.created_at
        
        # Store very similar memory (should trigger update)
        similar_memory = ScoredMemory(
            text="I want to build a web application",
            type="project_goal",
            importance=0.85
        )
        result2 = writer.store_memory(similar_memory)
        
        assert result2.is_update is True
        assert result2.created_at == original_timestamp  # Timestamp preserved
        assert len(mock_store.stored_memories) == 1
    
    def test_preserve_original_timestamp_on_update(self):
        """Test that original timestamp is preserved when updating."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # Store initial memory
        first_memory = ScoredMemory(
            text="I prefer Python",
            type="user_preference",
            importance=0.7
        )
        result1 = writer.store_memory(first_memory)
        original_timestamp = result1.created_at
        
        # Update with similar memory
        similar_memory = ScoredMemory(
            text="I prefer Python",
            type="user_preference",
            importance=0.75
        )
        result2 = writer.store_memory(similar_memory)
        
        assert result2.created_at == original_timestamp
        assert result2.is_update is True
    
    def test_use_maximum_importance_score_on_update(self):
        """Test that maximum importance score is used when updating."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # Store initial memory with high importance
        first_memory = ScoredMemory(
            text="I want to build a web app",
            type="project_goal",
            importance=0.9
        )
        writer.store_memory(first_memory)
        
        # Update with lower importance
        similar_memory = ScoredMemory(
            text="I want to build a web app",
            type="project_goal",
            importance=0.7
        )
        result = writer.store_memory(similar_memory)
        
        # Should keep higher score
        assert result.importance == 0.9
        assert result.is_update is True
    
    def test_use_maximum_importance_score_when_new_higher(self):
        """Test that maximum importance score is used when new score is higher."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # Store initial memory with low importance
        first_memory = ScoredMemory(
            text="I want to build a web app",
            type="project_goal",
            importance=0.7
        )
        writer.store_memory(first_memory)
        
        # Update with higher importance
        similar_memory = ScoredMemory(
            text="I want to build a web app",
            type="project_goal",
            importance=0.9
        )
        result = writer.store_memory(similar_memory)
        
        # Should use new higher score
        assert result.importance == 0.9
        assert result.is_update is True
    
    def test_is_update_flag_accuracy(self):
        """Test that is_update flag is set correctly."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store)
        
        # New memory
        new_memory = ScoredMemory(
            text="First memory",
            type="fact",
            importance=0.6
        )
        result1 = writer.store_memory(new_memory)
        assert result1.is_update is False
        
        # Similar memory (update)
        similar_memory = ScoredMemory(
            text="First memory",
            type="fact",
            importance=0.65
        )
        result2 = writer.store_memory(similar_memory)
        assert result2.is_update is True
        
        # Different memory (new)
        different_memory = ScoredMemory(
            text="Completely different memory",
            type="fact",
            importance=0.6
        )
        result3 = writer.store_memory(different_memory)
        assert result3.is_update is False
    
    def test_create_new_when_similarity_below_threshold(self):
        """Test creating new memory when similarity is below threshold."""
        mock_store = MockMemoryStore()
        writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
        
        # Store initial memory
        first_memory = ScoredMemory(
            text="I want to build a web application",
            type="project_goal",
            importance=0.8
        )
        writer.store_memory(first_memory)
        
        # Store somewhat similar but below threshold memory
        different_memory = ScoredMemory(
            text="I want to create a mobile app",
            type="project_goal",
            importance=0.8
        )
        result = writer.store_memory(different_memory)
        
        # Should create new memory
        assert result.is_update is False
        assert len(mock_store.stored_memories) == 2
