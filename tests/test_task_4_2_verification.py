"""
Verification test for Task 4.2: Implement memory retrieval call

This test specifically verifies the requirements for Task 4.2:
- Call memory_interface.retrieve(params=params)
- Extract memories list from RetrievalResult
- Apply size limit truncation (slice to max_memories)
- Transform each MemoryEntry to pure dict

Requirements: 2.1, 2.3, 2.4, 3.1, 3.3, 3.4
"""

import pytest
from unittest.mock import Mock, call
from typing import Dict, List, Any

from luma.core.context_injection import inject_memories, InjectionConfig
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryEntry,
    QueryParameters,
    RetrievalResult
)


class TestTask42MemoryRetrievalCall:
    """Test suite verifying Task 4.2 implementation."""
    
    def test_calls_memory_interface_retrieve_with_params(self):
        """
        Verify that inject_memories calls memory_interface.retrieve(params=params).
        
        Requirements: 2.3
        """
        # Create mock memory interface
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        }
        
        # Create config
        config = InjectionConfig(max_memories=10)
        
        # Call inject_memories
        context = inject_memories(
            query="test query",
            memory_interface=mock_interface,
            config=config
        )
        
        # Verify retrieve was called with params
        mock_interface.retrieve.assert_called_once()
        call_args = mock_interface.retrieve.call_args
        
        # Verify params were passed
        assert "params" in call_args.kwargs
        params = call_args.kwargs["params"]
        
        # Verify params structure
        assert "query" in params
        assert params["query"] == "test query"
        assert "limit" in params
        assert params["limit"] == 10
    
    def test_extracts_memories_from_retrieval_result(self):
        """
        Verify that inject_memories extracts memories list from RetrievalResult.
        
        Requirements: 2.1, 2.4
        """
        # Create test memories
        test_memories: List[MemoryEntry] = [
            {
                "id": "mem_1",
                "content": "Test content 1",
                "category": "test",
                "timestamp": "2024-01-01T00:00:00",
                "metadata": {"source": "test"},
                "tags": ["test"]
            },
            {
                "id": "mem_2",
                "content": "Test content 2",
                "category": "test",
                "timestamp": "2024-01-02T00:00:00",
                "metadata": {"source": "test"},
                "tags": ["test"]
            }
        ]
        
        # Create mock memory interface
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve.return_value = {
            "memories": test_memories,
            "total_count": 2,
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        }
        
        # Create config
        config = InjectionConfig(max_memories=10)
        
        # Call inject_memories
        context = inject_memories(
            query="test query",
            memory_interface=mock_interface,
            config=config
        )
        
        # Verify memories were extracted and injected
        assert "memories" in context
        assert len(context["memories"]) == 2
        assert context["memories"][0]["id"] == "mem_1"
        assert context["memories"][1]["id"] == "mem_2"
    
    def test_applies_size_limit_truncation(self):
        """
        Verify that inject_memories applies size limit truncation (slice to max_memories).
        
        Requirements: 3.1, 3.3, 3.4
        """
        # Create more memories than the limit
        test_memories: List[MemoryEntry] = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "category": "test",
                "timestamp": f"2024-01-{i:02d}T00:00:00",
                "metadata": {"source": "test"},
                "tags": ["test"]
            }
            for i in range(1, 16)  # 15 memories
        ]
        
        # Create mock memory interface
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve.return_value = {
            "memories": test_memories,
            "total_count": 15,
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": True
            }
        }
        
        # Create config with limit of 10
        config = InjectionConfig(max_memories=10)
        
        # Call inject_memories
        context = inject_memories(
            query="test query",
            memory_interface=mock_interface,
            config=config
        )
        
        # Verify truncation was applied
        assert "memories" in context
        assert len(context["memories"]) == 10, \
            f"Expected 10 memories after truncation, got {len(context['memories'])}"
        
        # Verify first 10 memories were kept (preserving order)
        for i in range(10):
            assert context["memories"][i]["id"] == f"mem_{i+1}", \
                f"Expected mem_{i+1}, got {context['memories'][i]['id']}"
    
    def test_transforms_memory_entries_to_pure_dicts(self):
        """
        Verify that inject_memories transforms each MemoryEntry to pure dict.
        
        Requirements: 2.1, 2.4
        """
        # Create test memory with all fields
        test_memory: MemoryEntry = {
            "id": "mem_1",
            "content": "Test content",
            "category": "education",
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {
                "source": "user_input",
                "priority": 5,
                "nested": {"key": "value"}
            },
            "tags": ["python", "programming"]
        }
        
        # Create mock memory interface
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve.return_value = {
            "memories": [test_memory],
            "total_count": 1,
            "query_metadata": {
                "execution_time_ms": 10.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        }
        
        # Create config
        config = InjectionConfig(max_memories=10)
        
        # Call inject_memories
        context = inject_memories(
            query="test query",
            memory_interface=mock_interface,
            config=config
        )
        
        # Verify transformation to pure dict
        assert "memories" in context
        assert len(context["memories"]) == 1
        
        transformed = context["memories"][0]
        
        # Verify it's a dict (not MemoryEntry object)
        assert isinstance(transformed, dict)
        
        # Verify all required fields present
        assert "id" in transformed
        assert "content" in transformed
        assert "category" in transformed
        assert "timestamp" in transformed
        assert "metadata" in transformed
        assert "tags" in transformed
        
        # Verify values match original
        assert transformed["id"] == test_memory["id"]
        assert transformed["content"] == test_memory["content"]
        assert transformed["category"] == test_memory["category"]
        assert transformed["timestamp"] == test_memory["timestamp"]
        assert transformed["metadata"] == test_memory["metadata"]
        assert transformed["tags"] == test_memory["tags"]
        
        # Verify only primitive types (no custom objects)
        def is_primitive(value):
            if isinstance(value, (str, int, float, bool, type(None))):
                return True
            if isinstance(value, list):
                return all(is_primitive(v) for v in value)
            if isinstance(value, dict):
                return all(is_primitive(v) for v in value.values())
            return False
        
        for field_name, field_value in transformed.items():
            assert is_primitive(field_value), \
                f"Field '{field_name}' contains non-primitive type: {type(field_value)}"
    
    def test_complete_flow_with_truncation_and_transformation(self):
        """
        Integration test verifying complete Task 4.2 flow:
        1. Call memory_interface.retrieve(params=params)
        2. Extract memories list from RetrievalResult
        3. Apply size limit truncation
        4. Transform each MemoryEntry to pure dict
        
        Requirements: 2.1, 2.3, 2.4, 3.1, 3.3, 3.4
        """
        # Create 8 test memories
        test_memories: List[MemoryEntry] = [
            {
                "id": f"mem_{i}",
                "content": f"Content {i}",
                "category": "test",
                "timestamp": f"2024-01-{i:02d}T00:00:00",
                "metadata": {"index": i, "nested": {"data": f"value_{i}"}},
                "tags": [f"tag_{i}"]
            }
            for i in range(1, 9)
        ]
        
        # Create mock memory interface
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve.return_value = {
            "memories": test_memories,
            "total_count": 8,
            "query_metadata": {
                "execution_time_ms": 15.0,
                "filters_applied": {},
                "limit": 5,
                "has_more": True
            }
        }
        
        # Create config with limit of 5
        config = InjectionConfig(max_memories=5)
        
        # Call inject_memories
        context = inject_memories(
            query="integration test",
            memory_interface=mock_interface,
            config=config
        )
        
        # STEP 1: Verify retrieve was called with params
        mock_interface.retrieve.assert_called_once()
        call_args = mock_interface.retrieve.call_args
        assert "params" in call_args.kwargs
        params = call_args.kwargs["params"]
        assert params["query"] == "integration test"
        assert params["limit"] == 5
        
        # STEP 2: Verify memories were extracted from RetrievalResult
        assert "memories" in context
        assert isinstance(context["memories"], list)
        
        # STEP 3: Verify size limit truncation (8 memories -> 5 memories)
        assert len(context["memories"]) == 5, \
            f"Expected 5 memories after truncation, got {len(context['memories'])}"
        
        # STEP 4: Verify transformation to pure dicts
        for i, memory in enumerate(context["memories"]):
            # Verify it's a dict
            assert isinstance(memory, dict)
            
            # Verify all fields present
            assert "id" in memory
            assert "content" in memory
            assert "category" in memory
            assert "timestamp" in memory
            assert "metadata" in memory
            assert "tags" in memory
            
            # Verify values match original (first 5 memories)
            expected_index = i + 1
            assert memory["id"] == f"mem_{expected_index}"
            assert memory["content"] == f"Content {expected_index}"
            assert memory["metadata"]["index"] == expected_index
            assert memory["metadata"]["nested"]["data"] == f"value_{expected_index}"
            assert memory["tags"] == [f"tag_{expected_index}"]
            
            # Verify only primitive types
            def is_primitive(value):
                if isinstance(value, (str, int, float, bool, type(None))):
                    return True
                if isinstance(value, list):
                    return all(is_primitive(v) for v in value)
                if isinstance(value, dict):
                    return all(is_primitive(v) for v in value.values())
                return False
            
            for field_value in memory.values():
                assert is_primitive(field_value), \
                    f"Non-primitive value found in memory {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
