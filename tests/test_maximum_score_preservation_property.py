"""Property test for maximum score preservation on update.

**Validates: Requirements 5.4**

Property 5: Maximum Score Preservation on Update
For any memory update operation where both the existing memory and new memory
have importance scores, the Memory Writer must use the maximum of the two
scores in the updated memory.
"""

import pytest
from hypothesis import given, strategies as st, settings
from luma.core.memory_write.memory_writer import MemoryWriter
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
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_maximum_score_preservation_property(text, memory_type, importance1, importance2):
    """
    Property 5: Maximum Score Preservation on Update
    
    For any memory update operation, the Memory Writer must use the maximum
    of the existing and new importance scores.
    
    **Validates: Requirements 5.4**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory with different importance (triggers update)
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Updated memory should have maximum of both scores
    expected_max = max(importance1, importance2)
    assert result2.importance == expected_max, \
        f"Updated memory should have max importance {expected_max}, got {result2.importance}"
    assert result2.is_update is True, \
        "Identical memory should trigger update"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_score_preserved_when_first_higher(text, memory_type, importance1, importance2):
    """
    Property: When existing score is higher, it should be preserved.
    
    Tests that the maximum score logic correctly preserves the existing
    score when it's higher than the new score.
    
    **Validates: Requirements 5.4**
    """
    # Ensure first importance is higher
    if importance1 < importance2:
        importance1, importance2 = importance2, importance1
    
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory with higher importance
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory with lower importance
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Should keep the higher (first) score
    assert result2.importance == importance1, \
        f"Should preserve higher existing score {importance1}, got {result2.importance}"
    assert result2.importance >= importance2, \
        "Updated score should be at least as high as new score"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_score_updated_when_second_higher(text, memory_type, importance1, importance2):
    """
    Property: When new score is higher, it should be used.
    
    Tests that the maximum score logic correctly uses the new score
    when it's higher than the existing score.
    
    **Validates: Requirements 5.4**
    """
    # Ensure second importance is higher
    if importance2 < importance1:
        importance1, importance2 = importance2, importance1
    
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory with lower importance
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory with higher importance
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Should use the higher (second) score
    assert result2.importance == importance2, \
        f"Should use higher new score {importance2}, got {result2.importance}"
    assert result2.importance >= importance1, \
        "Updated score should be at least as high as existing score"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance=st.floats(min_value=0.0, max_value=1.0)
)
def test_score_unchanged_when_equal(text, memory_type, importance):
    """
    Property: When scores are equal, the score should remain unchanged.
    
    Tests that the maximum score logic handles equal scores correctly.
    
    **Validates: Requirements 5.4**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory with same importance
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance)
    result2 = writer.store_memory(memory2)
    
    # Property: Score should remain the same
    assert result2.importance == importance, \
        f"Score should remain {importance}, got {result2.importance}"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=5)
)
def test_maximum_score_across_multiple_updates(text, memory_type, scores):
    """
    Property: Maximum score is preserved across multiple updates.
    
    Tests that the maximum score logic works correctly when a memory
    is updated multiple times with different scores.
    
    **Validates: Requirements 5.4**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory = ScoredMemory(text=text, type=memory_type, importance=scores[0])
    result = writer.store_memory(memory)
    
    # Update multiple times with different scores
    for score in scores[1:]:
        memory = ScoredMemory(text=text, type=memory_type, importance=score)
        result = writer.store_memory(memory)
    
    # Property: Final score should be the maximum of all scores
    expected_max = max(scores)
    assert result.importance == expected_max, \
        f"After multiple updates, score should be max {expected_max}, got {result.importance}"
    assert len(mock_store.stored_memories) == 1, \
        "Should still have only 1 memory after multiple updates"


@settings(max_examples=10)
@given(
    text=valid_text_strategy,
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_score_never_decreases_on_update(text, memory_type, importance1, importance2):
    """
    Property: Importance score never decreases on update.
    
    Tests that the maximum score logic ensures scores can only stay the
    same or increase, never decrease.
    
    **Validates: Requirements 5.4**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store, similarity_threshold=0.9)
    
    # Store initial memory
    memory1 = ScoredMemory(text=text, type=memory_type, importance=importance1)
    result1 = writer.store_memory(memory1)
    
    # Store identical memory with different importance
    memory2 = ScoredMemory(text=text, type=memory_type, importance=importance2)
    result2 = writer.store_memory(memory2)
    
    # Property: Score should never decrease
    assert result2.importance >= result1.importance, \
        f"Score decreased from {result1.importance} to {result2.importance}"
    assert result2.importance >= importance1, \
        "Updated score should be at least as high as original score"
    assert result2.importance >= importance2, \
        "Updated score should be at least as high as new score"
