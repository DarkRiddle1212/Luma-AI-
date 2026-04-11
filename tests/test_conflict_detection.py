"""
Unit tests for conflict detection in Memory_Write_Strategy.

Tests the conflict detection functionality including:
- Detection of known contradictory statements
- Conflict metadata attachment
- Marking older memories as potentially_outdated
- Conflict metadata in retrieval

Feature: memory-write-strategy-session-management
Requirements: 5.1, 5.2, 5.3, 5.4
"""

import pytest
from unittest.mock import Mock, MagicMock
from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig


class TestConflictDetection:
    """Test suite for detect_conflict method."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory interface."""
        return Mock()
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        return Mock()
    
    @pytest.fixture
    def strategy(self, mock_memory, mock_session_manager):
        """Create a Memory_Write_Strategy instance with mocks."""
        config = WriteStrategyConfig(enable_conflict_detection=True)
        return Memory_Write_Strategy(config, mock_session_manager, mock_memory)
    
    def test_no_conflict_returns_none(self, strategy, mock_memory):
        """Test that no conflict returns None."""
        # Setup: Empty result from memory
        mock_memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        result = strategy.detect_conflict("User likes Python", "preferences")
        
        assert result is None
        mock_memory.retrieve.assert_called_once()
    
    def test_contradictory_statements_detected(self, strategy, mock_memory):
        """
        Test that known contradictory statements are detected.
        Validates: Requirement 5.1
        """
        # Setup: Existing memory "User likes Python"
        existing_memory = {
            "id": "mem_123",
            "content": "User likes Python",
            "metadata": {"category": "preferences"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "preferences",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        # New memory contradicts existing: "User doesn't like Python"
        result = strategy.detect_conflict("User doesn't like Python", "preferences")
        
        # Should detect conflict
        assert result == "mem_123"
    
    def test_negation_conflict_with_not(self, strategy, mock_memory):
        """Test conflict detection with 'not' negation."""
        existing_memory = {
            "id": "mem_456",
            "content": "The system is stable",
            "metadata": {"category": "status"},
            "category": "status",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.detect_conflict("The system is not stable", "status")
        
        assert result == "mem_456"
    
    def test_negation_conflict_with_never(self, strategy, mock_memory):
        """Test conflict detection with 'never' negation."""
        existing_memory = {
            "id": "mem_789",
            "content": "User visits the gym regularly",
            "metadata": {"category": "habits"},
            "category": "habits",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.detect_conflict("User never visits the gym", "habits")
        
        assert result == "mem_789"
    
    def test_negation_conflict_with_cannot(self, strategy, mock_memory):
        """Test conflict detection with 'cannot' negation."""
        existing_memory = {
            "id": "mem_abc",
            "content": "User can speak French",
            "metadata": {"category": "skills"},
            "category": "skills",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.detect_conflict("User cannot speak French", "skills")
        
        assert result == "mem_abc"
    
    def test_no_conflict_same_negation_status(self, strategy, mock_memory):
        """Test that similar content with same negation status is not a conflict."""
        existing_memory = {
            "id": "mem_xyz",
            "content": "User doesn't like coffee",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        # Both have negation, so not a conflict
        result = strategy.detect_conflict("User doesn't like tea", "preferences")
        
        assert result is None
    
    def test_no_conflict_different_topics(self, strategy, mock_memory):
        """Test that different topics don't trigger conflict even with negation."""
        existing_memory = {
            "id": "mem_diff",
            "content": "User likes Python",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        # Different topic (JavaScript vs Python), low word overlap
        result = strategy.detect_conflict("User doesn't like JavaScript", "preferences")
        
        assert result is None
    
    def test_conflict_detection_disabled(self, mock_memory, mock_session_manager):
        """Test that conflict detection can be disabled via config."""
        config = WriteStrategyConfig(enable_conflict_detection=False)
        strategy = Memory_Write_Strategy(config, mock_session_manager, mock_memory)
        
        # Even with contradictory content, should return None when disabled
        existing_memory = {
            "id": "mem_disabled",
            "content": "User likes Python",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.detect_conflict("User doesn't like Python", "preferences")
        
        assert result is None
        # Should not even call retrieve when disabled
        mock_memory.retrieve.assert_not_called()
    
    def test_retrieval_failure_returns_none(self, strategy, mock_memory):
        """Test that retrieval failure is handled gracefully."""
        # Setup: Memory retrieval raises exception
        mock_memory.retrieve.side_effect = Exception("Database error")
        
        result = strategy.detect_conflict("User likes Python", "preferences")
        
        # Should return None and not raise exception
        assert result is None


class TestConflictMetadataAttachment:
    """Test suite for conflict metadata attachment during storage."""
    
    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory interface."""
        mock = Mock()
        # Default: no duplicates, return generated ID
        mock.store.return_value = "new_mem_id"
        # Add attributes that normalize_metadata accesses
        mock.default_category = None
        mock.default_tags = []
        return mock
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager with no active sessions."""
        mock = Mock()
        mock.sessions = {}
        mock.lock = MagicMock()
        mock.lock.__enter__ = Mock(return_value=None)
        mock.lock.__exit__ = Mock(return_value=None)
        return mock
    
    @pytest.fixture
    def strategy(self, mock_memory, mock_session_manager):
        """Create a Memory_Write_Strategy instance with mocks."""
        config = WriteStrategyConfig(
            enable_conflict_detection=True,
            min_content_length=3
        )
        return Memory_Write_Strategy(config, mock_session_manager, mock_memory)
    
    def test_conflict_metadata_attached_to_new_memory(self, strategy, mock_memory):
        """
        Test that conflict metadata is attached to the new memory.
        Validates: Requirement 5.2
        """
        # Setup: Existing memory that will conflict
        existing_memory = {
            "id": "mem_old",
            "content": "User likes Python",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        
        # First call: check for duplicates (none found)
        # Second call: detect conflicts (found)
        mock_memory.retrieve.side_effect = [
            {"memories": [], "total_count": 0},  # No duplicates
            {"memories": [existing_memory], "total_count": 1}  # Conflict found
        ]
        
        # Store conflicting memory
        memory_id = strategy.store_memory(
            "User doesn't like Python",
            metadata={"category": "preferences"}
        )
        
        # Verify memory was stored
        assert memory_id == "new_mem_id"
        
        # Verify conflict metadata was attached
        store_call_args = mock_memory.store.call_args
        stored_metadata = store_call_args[0][1]  # Second argument is metadata
        
        assert "conflicts_with" in stored_metadata
        assert stored_metadata["conflicts_with"] == "mem_old"
        assert stored_metadata["conflict_detected"] is True
    
    def test_conflicts_with_field_contains_memory_id(self, strategy, mock_memory):
        """
        Test that conflicts_with field references the conflicting memory_id.
        Validates: Requirement 5.3
        """
        existing_memory = {
            "id": "mem_conflict_123",
            "content": "System is online",
            "metadata": {"category": "status"},
            "category": "status",
            "tags": []
        }
        
        mock_memory.retrieve.side_effect = [
            {"memories": [], "total_count": 0},  # No duplicates
            {"memories": [existing_memory], "total_count": 1}  # Conflict found
        ]
        
        strategy.store_memory(
            "System is not online",
            metadata={"category": "status"}
        )
        
        store_call_args = mock_memory.store.call_args
        stored_metadata = store_call_args[0][1]
        
        assert stored_metadata["conflicts_with"] == "mem_conflict_123"
    
    def test_no_conflict_metadata_when_no_conflict(self, strategy, mock_memory):
        """Test that no conflict metadata is added when there's no conflict."""
        # Setup: No duplicates, no conflicts
        mock_memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0
        }
        
        strategy.store_memory(
            "User likes Python",
            metadata={"category": "preferences"}
        )
        
        store_call_args = mock_memory.store.call_args
        stored_metadata = store_call_args[0][1]
        
        assert "conflicts_with" not in stored_metadata
        assert "conflict_detected" not in stored_metadata
    
    def test_multiple_potential_conflicts_returns_first(self, strategy, mock_memory):
        """Test that when multiple conflicts exist, the first one is returned."""
        existing_memories = [
            {
                "id": "mem_first",
                "content": "User likes Python",
                "metadata": {"category": "preferences"},
                "category": "preferences",
                "tags": []
            },
            {
                "id": "mem_second",
                "content": "User enjoys Python programming",
                "metadata": {"category": "preferences"},
                "category": "preferences",
                "tags": []
            }
        ]
        
        mock_memory.retrieve.side_effect = [
            {"memories": [], "total_count": 0},  # No duplicates
            {"memories": existing_memories, "total_count": 2}  # Multiple conflicts
        ]
        
        strategy.store_memory(
            "User doesn't like Python",
            metadata={"category": "preferences"}
        )
        
        store_call_args = mock_memory.store.call_args
        stored_metadata = store_call_args[0][1]
        
        # Should reference the first conflicting memory
        assert stored_metadata["conflicts_with"] == "mem_first"


class TestConflictDetectionEdgeCases:
    """Test edge cases in conflict detection."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance."""
        config = WriteStrategyConfig(enable_conflict_detection=True)
        mock_memory = Mock()
        mock_session_manager = Mock()
        return Memory_Write_Strategy(config, mock_session_manager, mock_memory)
    
    def test_case_insensitive_conflict_detection(self, strategy):
        """Test that conflict detection is case-insensitive."""
        existing_memory = {
            "id": "mem_case",
            "content": "USER LIKES PYTHON",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        strategy.memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1
        }
        
        result = strategy.detect_conflict("user doesn't like python", "preferences")
        
        assert result == "mem_case"
    
    def test_whitespace_normalized_in_conflict_detection(self, strategy):
        """Test that whitespace is normalized in conflict detection."""
        existing_memory = {
            "id": "mem_space",
            "content": "  User   likes   Python  ",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        strategy.memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1
        }
        
        result = strategy.detect_conflict("User doesn't like Python", "preferences")
        
        assert result == "mem_space"
    
    def test_category_normalized_in_conflict_query(self, strategy):
        """Test that category is normalized when querying for conflicts."""
        strategy.memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0
        }
        
        strategy.detect_conflict("User likes Python", "  PREFERENCES  ")
        
        # Verify category was normalized in the query
        call_args = strategy.memory.retrieve.call_args
        assert call_args[1]["params"]["category"] == "preferences"
    
    def test_low_word_overlap_no_conflict(self, strategy):
        """Test that low word overlap doesn't trigger false conflicts."""
        existing_memory = {
            "id": "mem_low_overlap",
            "content": "User likes Python programming language",
            "metadata": {"category": "preferences"},
            "category": "preferences",
            "tags": []
        }
        strategy.memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1
        }
        
        # Only shares "User" and "doesn't" (negation), but different topics
        result = strategy.detect_conflict("User doesn't like coffee", "preferences")
        
        # Should not detect conflict due to low overlap
        assert result is None
    
    def test_empty_content_no_conflict(self, strategy):
        """Test that empty content doesn't cause conflicts."""
        strategy.memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0
        }
        
        result = strategy.detect_conflict("", "preferences")
        
        assert result is None
