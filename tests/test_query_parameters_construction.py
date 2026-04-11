"""
Unit tests for QueryParameters construction in inject_memories().

Tests that QueryParameters are correctly constructed from query and InjectionConfig,
including proper handling of optional filters.

Requirements tested:
- 2.3: Use top-ranked retrieval results from the MemoryInterface
"""

from datetime import datetime, UTC
from unittest.mock import Mock, MagicMock
import pytest

from luma.core.context_injection import inject_memories, InjectionConfig
from luma.core.memory_interface import MemoryInterface, QueryParameters, RetrievalResult


class TestQueryParametersConstruction:
    """Test suite for QueryParameters construction in inject_memories()."""
    
    def test_basic_query_parameters_construction(self):
        """Test QueryParameters construction with only required fields."""
        # Setup
        query = "test query"
        config = InjectionConfig(max_memories=10)
        
        # Create mock memory interface that captures the params
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve = MagicMock(return_value={
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        })
        
        # Execute
        inject_memories(query, mock_interface, config)
        
        # Verify params passed to retrieve()
        mock_interface.retrieve.assert_called_once()
        call_args = mock_interface.retrieve.call_args
        params = call_args.kwargs["params"]
        
        # Verify required fields
        assert params["query"] == query
        assert params["limit"] == 10
        
        # Verify optional fields not included when not specified
        assert "category" not in params
        assert "tags" not in params
        assert "start_time" not in params
        assert "end_time" not in params
    
    def test_query_parameters_with_category_filter(self):
        """Test QueryParameters construction with category filter."""
        # Setup
        query = "test query"
        config = InjectionConfig(
            max_memories=10,
            category_filter="education"
        )
        
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve = MagicMock(return_value={
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        })
        
        # Execute
        inject_memories(query, mock_interface, config)
        
        # Verify params
        call_args = mock_interface.retrieve.call_args
        params = call_args.kwargs["params"]
        
        assert params["query"] == query
        assert params["limit"] == 10
        assert params["category"] == "education"
        assert "tags" not in params
        assert "start_time" not in params
        assert "end_time" not in params
    
    def test_query_parameters_with_tag_filters(self):
        """Test QueryParameters construction with tag filters."""
        # Setup
        query = "test query"
        config = InjectionConfig(
            max_memories=15,
            tag_filters=["python", "programming"]
        )
        
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve = MagicMock(return_value={
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": 15,
                "has_more": False
            }
        })
        
        # Execute
        inject_memories(query, mock_interface, config)
        
        # Verify params
        call_args = mock_interface.retrieve.call_args
        params = call_args.kwargs["params"]
        
        assert params["query"] == query
        assert params["limit"] == 15
        assert params["tags"] == ["python", "programming"]
        assert "category" not in params
        assert "start_time" not in params
        assert "end_time" not in params
    
    def test_query_parameters_with_time_range(self):
        """Test QueryParameters construction with time range."""
        # Setup
        query = "test query"
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 12, 31, tzinfo=UTC)
        config = InjectionConfig(
            max_memories=20,
            time_range=(start_time, end_time)
        )
        
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve = MagicMock(return_value={
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": 20,
                "has_more": False
            }
        })
        
        # Execute
        inject_memories(query, mock_interface, config)
        
        # Verify params
        call_args = mock_interface.retrieve.call_args
        params = call_args.kwargs["params"]
        
        assert params["query"] == query
        assert params["limit"] == 20
        assert params["start_time"] == start_time
        assert params["end_time"] == end_time
        assert "category" not in params
        assert "tags" not in params
    
    def test_query_parameters_with_all_filters(self):
        """Test QueryParameters construction with all optional filters."""
        # Setup
        query = "test query"
        start_time = datetime(2024, 1, 1, tzinfo=UTC)
        end_time = datetime(2024, 12, 31, tzinfo=UTC)
        config = InjectionConfig(
            max_memories=10,
            category_filter="education",
            tag_filters=["python", "programming"],
            time_range=(start_time, end_time)
        )
        
        mock_interface = Mock(spec=MemoryInterface)
        mock_interface.retrieve = MagicMock(return_value={
            "memories": [],
            "total_count": 0,
            "query_metadata": {
                "execution_time_ms": 0.0,
                "filters_applied": {},
                "limit": 10,
                "has_more": False
            }
        })
        
        # Execute
        inject_memories(query, mock_interface, config)
        
        # Verify params
        call_args = mock_interface.retrieve.call_args
        params = call_args.kwargs["params"]
        
        # Verify all fields present
        assert params["query"] == query
        assert params["limit"] == 10
        assert params["category"] == "education"
        assert params["tags"] == ["python", "programming"]
        assert params["start_time"] == start_time
        assert params["end_time"] == end_time
    
    def test_query_parameters_respects_max_memories_limit(self):
        """Test that limit in QueryParameters matches config.max_memories."""
        # Test various limit values
        for max_memories in [5, 10, 15, 20]:
            query = "test query"
            config = InjectionConfig(max_memories=max_memories)
            
            mock_interface = Mock(spec=MemoryInterface)
            mock_interface.retrieve = MagicMock(return_value={
                "memories": [],
                "total_count": 0,
                "query_metadata": {
                    "execution_time_ms": 0.0,
                    "filters_applied": {},
                    "limit": max_memories,
                    "has_more": False
                }
            })
            
            # Execute
            inject_memories(query, mock_interface, config)
            
            # Verify limit matches max_memories
            call_args = mock_interface.retrieve.call_args
            params = call_args.kwargs["params"]
            assert params["limit"] == max_memories
