"""Property test for deduplication trigger threshold.

**Validates: Requirements 5.2**

Property 4: Deduplication Trigger Threshold
For any new memory being stored, if a similar existing memory exists with
text similarity above 0.9, then the Memory Writer must update the existing
memory instead of creating a new entry.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from luma.core.memory_write.memory_writer import MemoryWriter, calculate_similarity
from luma.core.memory_write.schemas import ScoredMemory
from luma.core.memory_interface import MemoryInterface, MemoryEntry, QueryParameters, RetrievalResult
from typing import Dict, List, Optional, Any
from datetime import datetime


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
        # Return all stored memories for similarity checking
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


# Strategy for generating valid text
valid_text_strategy = st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != "")


@settings(max_examples=10)
@given(
    base_text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_deduplication_trigger_above_threshold(base_text, memory_type, importance1, importance2):
    """
    Property 4: Deduplication Trigger Threshold
    
    When storing a memory that is very similar (>0.9 similarity) to an existing
    memory, the Memory Writer must update the existing memory instead of creating
    a new entry.
    
    **Validates: Requirements 5.2**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=base_text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory (similarity = 1.0, which is > 0.9)
    memory2 = ScoredMemory(text=base_text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Should update existing memory, not create new one
    assert result2.is_update is True, \
        "Memory with similarity > 0.9 should trigger update, not create new entry"
    assert len(mock_store.stored_memories) == 1, \
        f"Should have 1 memory after deduplication, got {len(mock_store.stored_memories)}"
    assert result1.memory_id == result2.memory_id, \
        "Updated memory should have same ID as original"


@settings(max_examples=10)
@given(
    text1=valid_text_strategy,
    text2=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_no_deduplication_below_threshold(text1, text2, memory_type, importance1, importance2):
    """
    Property: When similarity is below threshold, create new memory.
    
    When storing a memory that is dissimilar (<=0.9 similarity) to existing
    memories, the Memory Writer must create a new entry.
    
    **Validates: Requirements 5.2**
    """
    # Calculate similarity and skip if above threshold
    similarity = calculate_similarity(text1, text2)
    assume(similarity <= 0.9)
    
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=text1, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store different memory (similarity <= 0.9)
    memory2 = ScoredMemory(text=text2, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Should create new memory, not update
    assert result2.is_update is False, \
        "Memory with similarity <= 0.9 should create new entry, not update"
    assert len(mock_store.stored_memories) == 2, \
        f"Should have 2 memories when similarity <= 0.9, got {len(mock_store.stored_memories)}"
    assert result1.memory_id != result2.memory_id, \
        "Different memories should have different IDs"


@settings(max_examples=10)
@given(
    base_text=valid_text_strategy,
    # Generate a small variation to the text
    variation=st.integers(min_value=0, max_value=5),
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_deduplication_threshold_boundary(base_text, variation, memory_type, importance1, importance2):
    """
    Property: Deduplication behavior is consistent at threshold boundary.
    
    Tests that the deduplication logic correctly handles cases near the
    similarity threshold (0.9).
    
    **Validates: Requirements 5.2**
    """
    # Create a slightly modified version of the text
    if variation == 0:
        modified_text = base_text  # Identical (similarity = 1.0)
    else:
        # Add some characters to create variation
        modified_text = base_text + " " + "x" * variation
    
    similarity = calculate_similarity(base_text, modified_text)
    
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=base_text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store modified memory
    memory2 = ScoredMemory(text=modified_text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Behavior depends on similarity threshold
    if similarity > 0.9:
        # Should update existing memory
        assert result2.is_update is True, \
            f"Memory with similarity {similarity} > 0.9 should trigger update"
        assert len(mock_store.stored_memories) == 1, \
            f"Should have 1 memory after deduplication, got {len(mock_store.stored_memories)}"
    else:
        # Should create new memory
        assert result2.is_update is False, \
            f"Memory with similarity {similarity} <= 0.9 should create new entry"
        assert len(mock_store.stored_memories) == 2, \
            f"Should have 2 memories when similarity <= 0.9, got {len(mock_store.stored_memories)}"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance=st.floats(min_value=0.0, max_value=1.0),
    custom_threshold=st.floats(min_value=0.5, max_value=0.99)
)
def test_custom_similarity_threshold(text, memory_type, importance, custom_threshold):
    """
    Property: Custom similarity thresholds work correctly.
    
    Tests that the Memory Writer respects custom similarity thresholds
    for deduplication.
    
    **Validates: Requirements 5.2**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=custom_threshold)
    
    # Store initial memory
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory (similarity = 1.0)
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance)
    result2 = writer.store_memory(memory2)
    
    # Property: Identical memory should always trigger update (similarity = 1.0 > any threshold)
    assert result2.is_update is True, \
        f"Identical memory should trigger update with threshold {custom_threshold}"
    assert len(mock_store.stored_memories) == 1, \
        "Should have 1 memory after storing identical memory"
