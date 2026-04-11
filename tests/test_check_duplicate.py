"""
Unit tests for check_duplicate method in Memory_Write_Strategy.

Tests the duplicate detection functionality including exact duplicates,
near-duplicates, and content normalization.
"""

import pytest
from unittest.mock import Mock
from luma.core.write_strategy import Memory_Write_Strategy, WriteStrategyConfig


class TestCheckDuplicate:
    """Test suite for check_duplicate method."""
    
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
        config = WriteStrategyConfig()
        return Memory_Write_Strategy(config, mock_session_manager, mock_memory)
    
    def test_no_duplicate_returns_none(self, strategy, mock_memory):
        """Test that no duplicate returns None."""
        # Setup: Empty result from memory
        mock_memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        result = strategy.check_duplicate("test content", "test_category")
        
        assert result is None
        mock_memory.retrieve.assert_called_once()
    
    def test_exact_duplicate_returns_memory_id(self, strategy, mock_memory):
        """Test that exact duplicate returns existing memory_id."""
        # Setup: Existing memory with same content
        existing_memory = {
            "id": "mem_123",
            "content": "test content",
            "metadata": {"category": "test_category"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "test_category",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.check_duplicate("test content", "test_category")
        
        assert result == "mem_123"
    
    def test_case_insensitive_duplicate_detection(self, strategy, mock_memory):
        """Test that duplicate detection is case-insensitive."""
        # Setup: Existing memory with different case
        existing_memory = {
            "id": "mem_456",
            "content": "TEST CONTENT",
            "metadata": {"category": "test_category"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "test_category",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.check_duplicate("test content", "test_category")
        
        assert result == "mem_456"
    
    def test_whitespace_normalized_duplicate_detection(self, strategy, mock_memory):
        """Test that duplicate detection normalizes whitespace."""
        # Setup: Existing memory with extra whitespace
        existing_memory = {
            "id": "mem_789",
            "content": "  test content  ",
            "metadata": {"category": "test_category"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "test_category",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.check_duplicate("test content", "test_category")
        
        assert result == "mem_789"
    
    def test_near_duplicate_detection(self, strategy, mock_memory):
        """Test that near-duplicates are detected based on similarity threshold."""
        # Setup: Existing memory with very similar content (only one character different)
        existing_memory = {
            "id": "mem_near",
            "content": "test content example",
            "metadata": {"category": "test_category"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "test_category",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        # Very similar content (only last word differs by one letter)
        # This should have similarity >= 0.9 (the default threshold)
        result = strategy.check_duplicate("test content exampl", "test_category")
        
        # Should detect as near-duplicate (high similarity)
        assert result == "mem_near"
    
    def test_different_content_not_duplicate(self, strategy, mock_memory):
        """Test that different content is not detected as duplicate."""
        # Setup: Existing memory with different content
        existing_memory = {
            "id": "mem_diff",
            "content": "completely different content",
            "metadata": {"category": "test_category"},
            "timestamp": "2024-01-01T00:00:00",
            "category": "test_category",
            "tags": []
        }
        mock_memory.retrieve.return_value = {
            "memories": [existing_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = strategy.check_duplicate("test content", "test_category")
        
        assert result is None
    
    def test_retrieval_failure_returns_none(self, strategy, mock_memory):
        """Test that retrieval failure is handled gracefully."""
        # Setup: Memory retrieval raises exception
        mock_memory.retrieve.side_effect = Exception("Database error")
        
        result = strategy.check_duplicate("test content", "test_category")
        
        # Should return None and not raise exception
        assert result is None
    
    def test_category_normalized_in_query(self, strategy, mock_memory):
        """Test that category is normalized when querying."""
        mock_memory.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        strategy.check_duplicate("test content", "  TEST_Category  ")
        
        # Verify category was normalized in the query
        call_args = mock_memory.retrieve.call_args
        assert call_args[1]["params"]["category"] == "test_category"


class TestCalculateSimilarity:
    """Test suite for _calculate_similarity helper method."""
    
    @pytest.fixture
    def strategy(self):
        """Create a Memory_Write_Strategy instance."""
        config = WriteStrategyConfig()
        mock_memory = Mock()
        mock_session_manager = Mock()
        return Memory_Write_Strategy(config, mock_session_manager, mock_memory)
    
    def test_identical_strings_similarity_one(self, strategy):
        """Test that identical strings have similarity of 1.0."""
        similarity = strategy._calculate_similarity("test", "test")
        assert similarity == 1.0
    
    def test_empty_strings_similarity_one(self, strategy):
        """Test that two empty strings have similarity of 1.0."""
        similarity = strategy._calculate_similarity("", "")
        assert similarity == 1.0
    
    def test_one_empty_string_similarity_zero(self, strategy):
        """Test that one empty string results in similarity of 0.0."""
        similarity = strategy._calculate_similarity("test", "")
        assert similarity == 0.0
        
        similarity = strategy._calculate_similarity("", "test")
        assert similarity == 0.0
    
    def test_completely_different_strings_low_similarity(self, strategy):
        """Test that completely different strings have low similarity."""
        similarity = strategy._calculate_similarity("abc", "xyz")
        assert similarity < 0.3
    
    def test_similar_strings_high_similarity(self, strategy):
        """Test that similar strings have high similarity."""
        similarity = strategy._calculate_similarity("testing", "testing123")
        assert similarity > 0.6  # Adjusted threshold based on bigram similarity
    
    def test_single_character_strings(self, strategy):
        """Test similarity calculation with single character strings."""
        # Single characters can't form bigrams, should fall back to equality
        similarity = strategy._calculate_similarity("a", "a")
        assert similarity == 1.0
        
        similarity = strategy._calculate_similarity("a", "b")
        assert similarity == 0.0
