"""
Unit tests for InjectionResult dataclass.

Tests the InjectionResult output dataclass including:
- to_dict() serialization
- from_dict() deserialization
- Round-trip serialization/deserialization
"""

import json
from datetime import datetime, timezone
from luma.core.injection_engine import InjectedMemory, InjectionResult


def test_injection_result_to_dict():
    """Test InjectionResult.to_dict() serialization."""
    # Create test data
    timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    memory = InjectedMemory(
        memory_id="mem_123",
        content="Test content",
        metadata={"source": "test", "token_count": 10},
        similarity_score=0.85,
        timestamp=timestamp,
        category="test_category"
    )
    
    result = InjectionResult(
        memories=[memory],
        total_tokens=10,
        input_count=100,
        filtered_by_category=20,
        filtered_by_redundancy=30,
        filtered_by_budget=40
    )
    
    # Serialize to dict
    result_dict = result.to_dict()
    
    # Verify structure
    assert 'memories' in result_dict
    assert 'total_tokens' in result_dict
    assert 'input_count' in result_dict
    assert 'filtered_by_category' in result_dict
    assert 'filtered_by_redundancy' in result_dict
    assert 'filtered_by_budget' in result_dict
    
    # Verify values
    assert result_dict['total_tokens'] == 10
    assert result_dict['input_count'] == 100
    assert result_dict['filtered_by_category'] == 20
    assert result_dict['filtered_by_redundancy'] == 30
    assert result_dict['filtered_by_budget'] == 40
    assert len(result_dict['memories']) == 1
    
    # Verify memory serialization
    memory_dict = result_dict['memories'][0]
    assert memory_dict['memory_id'] == "mem_123"
    assert memory_dict['content'] == "Test content"
    assert memory_dict['similarity_score'] == 0.85
    assert memory_dict['category'] == "test_category"
    assert memory_dict['timestamp'] == timestamp.isoformat()


def test_injection_result_from_dict():
    """Test InjectionResult.from_dict() deserialization."""
    # Create test dictionary
    timestamp_str = "2024-01-15T10:30:00+00:00"
    data = {
        'memories': [
            {
                'memory_id': 'mem_123',
                'content': 'Test content',
                'metadata': {'source': 'test', 'token_count': 10},
                'similarity_score': 0.85,
                'timestamp': timestamp_str,
                'category': 'test_category'
            }
        ],
        'total_tokens': 10,
        'input_count': 100,
        'filtered_by_category': 20,
        'filtered_by_redundancy': 30,
        'filtered_by_budget': 40
    }
    
    # Deserialize from dict
    result = InjectionResult.from_dict(data)
    
    # Verify structure
    assert isinstance(result, InjectionResult)
    assert len(result.memories) == 1
    
    # Verify values
    assert result.total_tokens == 10
    assert result.input_count == 100
    assert result.filtered_by_category == 20
    assert result.filtered_by_redundancy == 30
    assert result.filtered_by_budget == 40
    
    # Verify memory deserialization
    memory = result.memories[0]
    assert isinstance(memory, InjectedMemory)
    assert memory.memory_id == "mem_123"
    assert memory.content == "Test content"
    assert memory.similarity_score == 0.85
    assert memory.category == "test_category"
    assert memory.timestamp == datetime.fromisoformat(timestamp_str)
    assert memory.metadata == {'source': 'test', 'token_count': 10}


def test_injection_result_round_trip():
    """Test round-trip serialization/deserialization (Property 9)."""
    # Create original result
    timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    memory1 = InjectedMemory(
        memory_id="mem_123",
        content="Test content 1",
        metadata={"source": "test", "token_count": 10},
        similarity_score=0.85,
        timestamp=timestamp,
        category="test_category"
    )
    memory2 = InjectedMemory(
        memory_id="mem_456",
        content="Test content 2",
        metadata={"source": "test", "token_count": 15},
        similarity_score=0.75,
        timestamp=timestamp,
        category=None
    )
    
    original = InjectionResult(
        memories=[memory1, memory2],
        total_tokens=25,
        input_count=100,
        filtered_by_category=20,
        filtered_by_redundancy=30,
        filtered_by_budget=48
    )
    
    # Serialize to dict and JSON
    result_dict = original.to_dict()
    json_str = json.dumps(result_dict)
    
    # Deserialize back
    parsed_dict = json.loads(json_str)
    restored = InjectionResult.from_dict(parsed_dict)
    
    # Verify equivalence
    assert len(restored.memories) == len(original.memories)
    assert restored.total_tokens == original.total_tokens
    assert restored.input_count == original.input_count
    assert restored.filtered_by_category == original.filtered_by_category
    assert restored.filtered_by_redundancy == original.filtered_by_redundancy
    assert restored.filtered_by_budget == original.filtered_by_budget
    
    # Verify memories
    for orig_mem, rest_mem in zip(original.memories, restored.memories):
        assert rest_mem.memory_id == orig_mem.memory_id
        assert rest_mem.content == orig_mem.content
        assert rest_mem.similarity_score == orig_mem.similarity_score
        assert rest_mem.timestamp == orig_mem.timestamp
        assert rest_mem.category == orig_mem.category
        assert rest_mem.metadata == orig_mem.metadata


def test_injection_result_empty_memories():
    """Test InjectionResult with empty memories list."""
    result = InjectionResult(
        memories=[],
        total_tokens=0,
        input_count=0,
        filtered_by_category=0,
        filtered_by_redundancy=0,
        filtered_by_budget=0
    )
    
    # Serialize and deserialize
    result_dict = result.to_dict()
    restored = InjectionResult.from_dict(result_dict)
    
    # Verify
    assert len(restored.memories) == 0
    assert restored.total_tokens == 0
    assert restored.input_count == 0


def test_injection_result_json_serializable():
    """Test that InjectionResult can be fully serialized to JSON."""
    timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    memory = InjectedMemory(
        memory_id="mem_123",
        content="Test content",
        metadata={"source": "test", "nested": {"key": "value"}},
        similarity_score=0.85,
        timestamp=timestamp,
        category="test_category"
    )
    
    result = InjectionResult(
        memories=[memory],
        total_tokens=10,
        input_count=100,
        filtered_by_category=20,
        filtered_by_redundancy=30,
        filtered_by_budget=40
    )
    
    # This should not raise any exceptions
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)
    
    # Verify it's valid JSON
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Verify it can be parsed back
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
