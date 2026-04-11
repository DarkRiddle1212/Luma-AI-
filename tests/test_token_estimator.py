"""
Unit tests for TokenEstimator.

**Validates: Requirements 2.1**
"""

import pytest
from luma.core.injection_engine import TokenEstimator


class MockMemory:
    """Mock memory object for testing TokenEstimator."""
    
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata if metadata is not None else {}


def test_estimate_tokens_with_precomputed_token_count():
    """Test that TokenEstimator uses precomputed token_count from metadata when available."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with precomputed token_count
    memory = MockMemory(
        content="This is a test with many words",
        metadata={"token_count": 42}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # Should use precomputed value, not word count approximation
    assert result == 42


def test_estimate_tokens_with_precomputed_float_token_count():
    """Test that TokenEstimator converts float token_count to int for determinism."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with float token_count
    memory = MockMemory(
        content="Test content",
        metadata={"token_count": 42.7}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # Should convert to int
    assert result == 42
    assert isinstance(result, int)


def test_estimate_tokens_fallback_to_word_count_approximation():
    """Test that TokenEstimator falls back to word-count approximation when token_count is missing."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory without token_count in metadata
    memory = MockMemory(
        content="Hello world test",  # 3 words
        metadata={"source": "test"}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # Should use approximation: 3 words × 1.3 = 3.9 → 3 (int conversion)
    assert result == 3


def test_estimate_tokens_fallback_with_empty_metadata():
    """Test that TokenEstimator falls back to word-count when metadata is empty."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with empty metadata
    memory = MockMemory(
        content="One two three four five",  # 5 words
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # Should use approximation: 5 words × 1.3 = 6.5 → 6 (int conversion)
    assert result == 6


def test_estimate_tokens_empty_content():
    """Test that TokenEstimator handles empty content correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with empty content
    memory = MockMemory(
        content="",
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # Empty content has 0 words, so 0 × 1.3 = 0
    assert result == 0


def test_estimate_tokens_single_word():
    """Test that TokenEstimator handles single word content."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with single word
    memory = MockMemory(
        content="Hello",
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # 1 word × 1.3 = 1.3 → 1 (int conversion)
    assert result == 1


def test_estimate_tokens_very_long_content():
    """Test that TokenEstimator handles very long content efficiently."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with very long content (1000 words)
    words = ["word"] * 1000
    content = " ".join(words)
    
    memory = MockMemory(
        content=content,
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # 1000 words × 1.3 = 1300
    assert result == 1300


def test_estimate_tokens_with_multiple_spaces():
    """Test that TokenEstimator handles multiple spaces correctly (split() behavior)."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with multiple spaces between words
    memory = MockMemory(
        content="Hello    world    test",  # 3 words with multiple spaces
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # split() handles multiple spaces, so still 3 words × 1.3 = 3.9 → 3
    assert result == 3


def test_estimate_tokens_with_newlines():
    """Test that TokenEstimator handles newlines correctly (split() behavior)."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with newlines
    memory = MockMemory(
        content="Hello\nworld\ntest",  # 3 words separated by newlines
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # split() handles newlines, so 3 words × 1.3 = 3.9 → 3
    assert result == 3


def test_estimate_tokens_with_tabs():
    """Test that TokenEstimator handles tabs correctly (split() behavior)."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with tabs
    memory = MockMemory(
        content="Hello\tworld\ttest",  # 3 words separated by tabs
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # split() handles tabs, so 3 words × 1.3 = 3.9 → 3
    assert result == 3


def test_estimate_tokens_custom_estimation_factor():
    """Test that TokenEstimator respects custom estimation_factor."""
    estimator = TokenEstimator(estimation_factor=2.0)
    
    # Create memory without token_count
    memory = MockMemory(
        content="One two three",  # 3 words
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # 3 words × 2.0 = 6.0 → 6
    assert result == 6


def test_estimate_tokens_small_estimation_factor():
    """Test that TokenEstimator works with small estimation_factor."""
    estimator = TokenEstimator(estimation_factor=0.5)
    
    # Create memory without token_count
    memory = MockMemory(
        content="One two three four",  # 4 words
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # 4 words × 0.5 = 2.0 → 2
    assert result == 2


def test_estimate_tokens_deterministic():
    """Test that TokenEstimator produces deterministic results (same input → same output)."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory
    memory = MockMemory(
        content="Deterministic test content",
        metadata={}
    )
    
    # Call estimate_tokens multiple times
    result1 = estimator.estimate_tokens(memory)
    result2 = estimator.estimate_tokens(memory)
    result3 = estimator.estimate_tokens(memory)
    
    # All results should be identical
    assert result1 == result2 == result3


def test_estimate_tokens_deterministic_with_precomputed():
    """Test that TokenEstimator is deterministic with precomputed token_count."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with precomputed token_count
    memory = MockMemory(
        content="Test content",
        metadata={"token_count": 100}
    )
    
    # Call estimate_tokens multiple times
    result1 = estimator.estimate_tokens(memory)
    result2 = estimator.estimate_tokens(memory)
    result3 = estimator.estimate_tokens(memory)
    
    # All results should be identical
    assert result1 == result2 == result3 == 100


def test_token_estimator_initialization_default_factor():
    """Test that TokenEstimator initializes with default estimation_factor."""
    estimator = TokenEstimator()
    
    assert estimator.estimation_factor == 1.3


def test_token_estimator_initialization_custom_factor():
    """Test that TokenEstimator initializes with custom estimation_factor."""
    estimator = TokenEstimator(estimation_factor=2.5)
    
    assert estimator.estimation_factor == 2.5


def test_token_estimator_initialization_negative_factor_raises_error():
    """Test that TokenEstimator raises ValueError for negative estimation_factor."""
    with pytest.raises(ValueError) as exc_info:
        TokenEstimator(estimation_factor=-1.0)
    
    error_msg = str(exc_info.value)
    assert "estimation_factor" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_token_estimator_initialization_zero_factor_raises_error():
    """Test that TokenEstimator raises ValueError for zero estimation_factor."""
    with pytest.raises(ValueError) as exc_info:
        TokenEstimator(estimation_factor=0.0)
    
    error_msg = str(exc_info.value)
    assert "estimation_factor" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_estimate_tokens_memory_without_metadata_attribute():
    """Test that TokenEstimator handles memory objects without metadata attribute."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create a simple object with only content attribute
    class SimpleMemory:
        def __init__(self, content):
            self.content = content
    
    memory = SimpleMemory(content="Hello world")
    
    result = estimator.estimate_tokens(memory)
    
    # Should fall back to word count: 2 words × 1.3 = 2.6 → 2
    assert result == 2


def test_estimate_tokens_memory_with_non_dict_metadata():
    """Test that TokenEstimator handles memory with non-dict metadata gracefully."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with non-dict metadata
    class BadMemory:
        def __init__(self, content, metadata):
            self.content = content
            self.metadata = metadata
    
    memory = BadMemory(content="Hello world test", metadata="not a dict")
    
    result = estimator.estimate_tokens(memory)
    
    # Should fall back to word count: 3 words × 1.3 = 3.9 → 3
    assert result == 3


def test_estimate_tokens_whitespace_only_content():
    """Test that TokenEstimator handles whitespace-only content."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with only whitespace
    memory = MockMemory(
        content="   \t\n   ",
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # split() on whitespace-only string returns empty list, so 0 words
    assert result == 0


def test_estimate_tokens_punctuation_handling():
    """Test that TokenEstimator treats punctuation as part of words (split() behavior)."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    # Create memory with punctuation
    memory = MockMemory(
        content="Hello, world! How are you?",  # 5 words (punctuation attached)
        metadata={}
    )
    
    result = estimator.estimate_tokens(memory)
    
    # split() treats "Hello," "world!" "How" "are" "you?" as 5 words
    # 5 words × 1.3 = 6.5 → 6
    assert result == 6
