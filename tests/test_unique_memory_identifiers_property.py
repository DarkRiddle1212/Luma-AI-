"""Property test for unique memory identifiers.

**Validates: Requirements 4.6**

Property 3: Unique Memory Identifiers
For any two memories stored by the Memory Writer in the same session,
their generated memory identifiers must be distinct (no duplicate IDs).
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
        # Return empty list to avoid deduplication in property tests
        result: RetrievalResult = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": limit,
                "has_more": False
            }
        }
        return result


# Strategy for generating scored memories
# Filter out whitespace-only strings
valid_text_strategy = st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != "")

scored_memory_strategy = st.builds(
    ScoredMemory,
    text=valid_text_strategy,
    type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance=st.floats(min_value=0.0, max_value=1.0)
)


@settings(max_examples=10)
@given(memories=st.lists(scored_memory_strategy, min_size=2, max_size=5))
def test_unique_memory_identifiers_property(memories):
    """
    Property 3: Unique Memory Identifiers
    
    For any set of memories stored by the Memory Writer in the same session,
    their generated memory identifiers must be distinct (no duplicate IDs).
    
    **Validates: Requirements 4.6**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store)
    
    # Store all memories
    stored_results = []
    for memory in memories:
        try:
            result = writer.store_memory(memory)
            stored_results.append(result)
        except ValueError:
            # Skip invalid memories (empty text after validation)
            continue
    
    # Extract all memory IDs
    memory_ids = [result.memory_id for result in stored_results]
    
    # Property: All IDs must be unique (no duplicates)
    assert len(memory_ids) == len(set(memory_ids)), \
        f"Found duplicate memory IDs: {memory_ids}"


@settings(max_examples=10)
@given(
    text1=st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != ""),
    text2=st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != ""),
    type1=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    type2=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance1=st.floats(min_value=0.0, max_value=1.0),
    importance2=st.floats(min_value=0.0, max_value=1.0)
)
def test_unique_ids_for_different_memories(text1, text2, type1, type2, importance1, importance2):
    """
    Property: Two different memories must have different IDs.
    
    Even if memories have similar content, they should get unique IDs
    when they are sufficiently different (below similarity threshold).
    
    **Validates: Requirements 4.6**
    """
    # Skip if texts are too similar (would trigger deduplication)
    from luma.core.memory_write.memory_writer import calculate_similarity
    if calculate_similarity(text1, text2) > 0.9:
        return
    
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store)
    
    memory1 = ScoredMemory(text=text1, type=type1, importance=importance1)
    memory2 = ScoredMemory(text=text2, type=type2, importance=importance2)
    
    try:
        result1 = writer.store_memory(memory1)
        result2 = writer.store_memory(memory2)
        
        # Property: Different memories must have different IDs
        assert result1.memory_id != result2.memory_id, \
            f"Two different memories got the same ID: {result1.memory_id}"
    except ValueError:
        # Skip invalid memories
        pass


@settings(max_examples=10)
@given(
    text=st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != ""),
    memory_type=st.sampled_from(["project_goal", "user_preference", "fact", "statement"]),
    importance=st.floats(min_value=0.0, max_value=1.0)
)
def test_id_format_consistency(text, memory_type, importance):
    """
    Property: All generated IDs should follow a consistent format.
    
    Memory IDs should be non-empty strings that can be used as unique
    identifiers in the memory store.
    
    **Validates: Requirements 4.6**
    """
    mock_store = MockMemoryStore()
    writer = MemoryWriter(memory_store=mock_store)
    
    memory = ScoredMemory(text=text, type=memory_type, importance=importance)
    
    try:
        result = writer.store_memory(memory)
        
        # Property: ID must be non-empty string
        assert isinstance(result.memory_id, str), \
            f"Memory ID must be a string, got {type(result.memory_id)}"
        assert len(result.memory_id) > 0, \
            "Memory ID must be non-empty"
        assert result.memory_id.strip() == result.memory_id, \
            "Memory ID should not have leading/trailing whitespace"
    except ValueError:
        # Skip invalid memories
        pass
