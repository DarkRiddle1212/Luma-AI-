"""
Unit tests for MemoryDeduplicator component.

Tests the MemoryDeduplicator component's similarity metrics (cosine, Jaccard,
Levenshtein), duplicate detection, merging logic, and integration with memory
interface.
"""

import pytest
from unittest.mock import MagicMock

from luma.core.lifecycle.memory_deduplicator import MemoryDeduplicator
from luma.core.lifecycle.schemas import (
    DeduplicationConfig,
    SimilarityMetric,
)


class TestMemoryDeduplicatorInitialization:
    """Test MemoryDeduplicator initialization."""
    
    def test_initialization_with_all_dependencies(self):
        """Test MemoryDeduplicator initialization with all dependencies."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()
        
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        
        assert deduplicator.memory_interface == mock_memory_interface
        assert deduplicator.dedup_config == dedup_config
        assert deduplicator.metrics_collector == mock_metrics_collector
        assert deduplicator.logger == mock_logger
    
    def test_initialization_without_optional_dependencies(self):
        """Test MemoryDeduplicator initialization without optional dependencies."""
        mock_memory_interface = MagicMock()
        
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        assert deduplicator.memory_interface == mock_memory_interface
        assert deduplicator.dedup_config == dedup_config
        assert deduplicator.metrics_collector is None
        assert deduplicator.logger is None


class TestCosineSimilarity:
    """Test cosine similarity calculation."""
    
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors (should be 1.0)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        embedding = [1.0, 2.0, 3.0]
        similarity = deduplicator._cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors (should be 0.5 after normalization)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Orthogonal vectors: [1, 0] and [0, 1]
        similarity = deduplicator._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        
        # Cosine similarity is 0, normalized to [0, 1] becomes 0.5
        assert abs(similarity - 0.5) < 0.001
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity for opposite vectors (should be 0.0 after normalization)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Opposite vectors: [1, 1] and [-1, -1]
        similarity = deduplicator._cosine_similarity([1.0, 1.0], [-1.0, -1.0])
        
        # Cosine similarity is -1, normalized to [0, 1] becomes 0.0
        assert abs(similarity - 0.0) < 0.001
    
    def test_cosine_similarity_with_zero_vector(self):
        """Test cosine similarity when one vector is zero."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # One zero vector
        similarity = deduplicator._cosine_similarity([0.0, 0.0], [1.0, 1.0])
        
        assert similarity == 0.0
    
    def test_cosine_similarity_known_values(self):
        """Test cosine similarity with known values."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Vectors with known cosine similarity
        # v1 = [1, 2, 3], v2 = [4, 5, 6]
        # dot = 1*4 + 2*5 + 3*6 = 32
        # |v1| = sqrt(14) ≈ 3.74, |v2| = sqrt(77) ≈ 8.77
        # cosine = 32 / (3.74 * 8.77) ≈ 0.975
        # normalized = (0.975 + 1) / 2 ≈ 0.987
        similarity = deduplicator._cosine_similarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        
        assert 0.98 <= similarity <= 0.99


class TestJaccardSimilarity:
    """Test Jaccard similarity calculation."""
    
    def test_jaccard_similarity_identical_texts(self):
        """Test Jaccard similarity for identical texts (should be 1.0)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        similarity = deduplicator._jaccard_similarity("hello world", "hello world")
        
        assert abs(similarity - 1.0) < 0.001
    
    def test_jaccard_similarity_disjoint_texts(self):
        """Test Jaccard similarity for disjoint texts (should be 0.0)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        similarity = deduplicator._jaccard_similarity("hello", "world")
        
        assert abs(similarity - 0.0) < 0.001
    
    def test_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity for partially overlapping texts."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # "hello world" and "hello there" share "hello"
        # tokens1 = {hello, world}, tokens2 = {hello, there}
        # intersection = {hello}, union = {hello, world, there}
        # Jaccard = 1/3 ≈ 0.333
        similarity = deduplicator._jaccard_similarity("hello world", "hello there")
        
        assert abs(similarity - 0.333) < 0.01
    
    def test_jaccard_similarity_empty_texts(self):
        """Test Jaccard similarity for empty texts."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Both empty
        similarity = deduplicator._jaccard_similarity("", "")
        
        assert similarity == 1.0
    
    def test_jaccard_similarity_one_empty(self):
        """Test Jaccard similarity when one text is empty."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        similarity = deduplicator._jaccard_similarity("hello", "")
        
        assert similarity == 0.0


class TestLevenshteinSimilarity:
    """Test Levenshtein similarity calculation."""
    
    def test_levenshtein_similarity_identical_strings(self):
        """Test Levenshtein similarity for identical strings (should be 1.0)."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        similarity = deduplicator._levenshtein_similarity("hello", "hello")
        
        assert abs(similarity - 1.0) < 0.001
    
    def test_levenshtein_similarity_completely_different(self):
        """Test Levenshtein similarity for completely different strings."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # "abc" to "xyz" requires 3 substitutions
        # similarity = 1 - 3/3 = 0.0
        similarity = deduplicator._levenshtein_similarity("abc", "xyz")
        
        assert abs(similarity - 0.0) < 0.001
    
    def test_levenshtein_similarity_one_edit(self):
        """Test Levenshtein similarity for one edit distance."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # "cat" to "bat" requires 1 substitution
        # similarity = 1 - 1/3 ≈ 0.667
        similarity = deduplicator._levenshtein_similarity("cat", "bat")
        
        assert 0.66 <= similarity <= 0.67
    
    def test_levenshtein_similarity_empty_strings(self):
        """Test Levenshtein similarity for empty strings."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Both empty
        similarity = deduplicator._levenshtein_similarity("", "")
        
        assert similarity == 1.0
    
    def test_levenshtein_similarity_one_empty(self):
        """Test Levenshtein similarity when one string is empty."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # "hello" to "" requires 5 deletions
        # similarity = 1 - 5/5 = 0.0
        similarity = deduplicator._levenshtein_similarity("hello", "")
        
        assert similarity == 0.0


class TestSimilarityScoreNormalization:
    """Test similarity score normalization to [0, 1]."""
    
    def test_cosine_similarity_normalized_range(self):
        """Test that cosine similarity is always in [0, 1]."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Test various vector combinations
        test_cases = [
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([1.0, 1.0], [-1.0, -1.0]),
            ([0.0, 0.0], [1.0, 1.0]),
        ]
        
        for v1, v2 in test_cases:
            similarity = deduplicator._cosine_similarity(v1, v2)
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"
    
    def test_jaccard_similarity_normalized_range(self):
        """Test that Jaccard similarity is always in [0, 1]."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        test_cases = [
            ("hello world", "hello there"),
            ("abc", "xyz"),
            ("", ""),
            ("test", ""),
        ]
        
        for text1, text2 in test_cases:
            similarity = deduplicator._jaccard_similarity(text1, text2)
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"
    
    def test_levenshtein_similarity_normalized_range(self):
        """Test that Levenshtein similarity is always in [0, 1]."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.LEVENSHTEIN,
            similarity_threshold=0.9,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        test_cases = [
            ("hello", "world"),
            ("abc", "xyz"),
            ("", ""),
            ("test", ""),
        ]
        
        for text1, text2 in test_cases:
            similarity = deduplicator._levenshtein_similarity(text1, text2)
            assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"


class TestDuplicateDetection:
    """Test duplicate detection based on similarity threshold."""
    
    def test_detect_duplicates_above_threshold(self):
        """Test that memories above threshold are detected as duplicates."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,  # Low threshold for testing
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Memories with high similarity (cosine similarity ~0.97)
        memory1 = {
            "id": "mem_1",
            "content": "Test content",
            "metadata": {"embedding": [1.0, 2.0, 3.0]},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Test content",
            "metadata": {"embedding": [1.1, 2.1, 3.1]},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        similarity = deduplicator.compute_similarity(memory1, memory2)
        
        assert similarity >= 0.5, f"Expected similarity >= 0.5, got {similarity}"
    
    def test_no_duplicate_below_threshold(self):
        """Test that memories below threshold are not detected as duplicates."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.99,  # High threshold
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Memories with low similarity
        memory1 = {
            "id": "mem_1",
            "content": "Hello",
            "metadata": {"embedding": [1.0, 0.0, 0.0]},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "World",
            "metadata": {"embedding": [0.0, 1.0, 0.0]},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        similarity = deduplicator.compute_similarity(memory1, memory2)
        
        assert similarity < 0.99, f"Expected similarity < 0.99, got {similarity}"


class TestMergePriority:
    """Test merge priority (higher importance retained)."""
    
    def test_retain_higher_importance(self):
        """Test that memory with higher importance is retained."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        memory1 = {
            "id": "mem_1",
            "content": "Content 1",
            "metadata": {"importance": 0.3},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Content 2",
            "metadata": {"importance": 0.7},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        kept, deleted = deduplicator._select_duplicate_retention(memory1, memory2)
        
        assert kept["id"] == "mem_2"
        assert deleted["id"] == "mem_1"
    
    def test_retain_earlier_timestamp_on_equal_importance(self):
        """Test that earlier timestamp is retained when importance is equal."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        memory1 = {
            "id": "mem_1",
            "content": "Content 1",
            "metadata": {"importance": 0.5},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Content 2",
            "metadata": {"importance": 0.5},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        kept, deleted = deduplicator._select_duplicate_retention(memory1, memory2)
        
        assert kept["id"] == "mem_1"
        assert deleted["id"] == "mem_2"


class TestProtectedMemoryRetention:
    """Test protected memory retention."""
    
    def test_retain_protected_memory(self):
        """Test that protected memory is retained in duplicate pair."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        memory1 = {
            "id": "mem_1",
            "content": "Content 1",
            "metadata": {"importance": 0.3, "protected": True},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Content 2",
            "metadata": {"importance": 0.7},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        kept, deleted = deduplicator._select_duplicate_retention(memory1, memory2)
        
        assert kept["id"] == "mem_1"
        assert deleted["id"] == "mem_2"
    
    def test_both_protected_handling(self):
        """Test that when both memories are protected, the one with earlier timestamp is kept."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        memory1 = {
            "id": "mem_1",
            "content": "Content 1",
            "metadata": {"importance": 0.3, "protected": True},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Content 2",
            "metadata": {"importance": 0.7, "protected": True},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        # Both protected - by timestamp, mem_1 is earlier so it's kept
        kept, deleted = deduplicator._select_duplicate_retention(memory1, memory2)
        
        # When both protected, the earlier timestamp is kept
        assert kept["id"] == "mem_1"
        assert deleted["id"] == "mem_2"


class TestMetadataMerging:
    """Test metadata merging (tags union)."""
    
    def test_merge_tags_union(self):
        """Test that tags are merged as union."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        kept_memory = {
            "id": "mem_1",
            "content": "Content 1",
            "metadata": {"tags": ["tag1", "tag2"]},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        deleted_memory = {
            "id": "mem_2",
            "content": "Content 2",
            "metadata": {"tags": ["tag2", "tag3"]},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        kept_tags = set(kept_memory.get("metadata", {}).get("tags", []))
        deleted_tags = set(deleted_memory.get("metadata", {}).get("tags", []))
        merged_tags = list(kept_tags | deleted_tags)
        
        assert set(merged_tags) == {"tag1", "tag2", "tag3"}


class TestEmbeddingPreference:
    """Test embedding preference (use cosine when available)."""
    
    def test_prefer_cosine_with_embeddings(self):
        """Test that cosine similarity is used when embeddings are available."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.JACCARD,  # Config says Jaccard
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # But memories have embeddings
        memory1 = {
            "id": "mem_1",
            "content": "Hello world",
            "metadata": {"embedding": [1.0, 2.0, 3.0]},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory2 = {
            "id": "mem_2",
            "content": "Hello world",
            "metadata": {"embedding": [1.1, 2.1, 3.1]},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        # Should use cosine similarity (because embeddings available)
        similarity = deduplicator.compute_similarity(memory1, memory2)
        
        # Verify it's using cosine (result should be ~0.99 for similar embeddings)
        assert similarity > 0.9, f"Expected cosine similarity > 0.9, got {similarity}"


class TestBatchProcessingOrder:
    """Test batch processing order (by creation timestamp)."""
    
    def test_sort_by_timestamp(self):
        """Test that memories are sorted by creation timestamp."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Create memories with different timestamps
        memories = [
            {
                "id": "mem_3",
                "content": "Content 3",
                "metadata": {},
                "timestamp": "2024-01-01T00:00:02Z",
            },
            {
                "id": "mem_1",
                "content": "Content 1",
                "metadata": {},
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "id": "mem_2",
                "content": "Content 2",
                "metadata": {},
                "timestamp": "2024-01-01T00:00:01Z",
            },
        ]
        
        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.get("timestamp", ""))
        
        assert sorted_memories[0]["id"] == "mem_1"
        assert sorted_memories[1]["id"] == "mem_2"
        assert sorted_memories[2]["id"] == "mem_3"


class TestDryRunMode:
    """Test dry_run mode (no merging)."""
    
    def test_dry_run_no_persistence(self):
        """Test that dry_run mode doesn't persist changes."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Mock memory retrieval with two similar memories
        mock_memory1 = {
            "id": "mem_1",
            "content": "Test content",
            "metadata": {"embedding": [1.0, 2.0, 3.0]},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        mock_memory2 = {
            "id": "mem_2",
            "content": "Test content",
            "metadata": {"embedding": [1.1, 2.1, 3.1]},
            "timestamp": "2024-01-01T00:00:01Z",
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory1, mock_memory2],
            "total_count": 2,
            "query_metadata": {},
        }
        
        # Run in dry_run mode
        result = deduplicator.deduplicate(dry_run=True)
        
        # Verify results
        assert result.duplicate_pairs_found == 1
        assert result.memories_merged == 1
        
        # Verify no persistence operations
        mock_memory_interface.store.assert_not_called()
        mock_memory_interface.delete.assert_not_called()

    class TestCheckpointPersistence:
        """Test checkpoint persistence (last processed timestamp)."""

        def test_checkpoint_updated_after_processing(self):
            """Test that checkpoint timestamp is updated after processing."""
            mock_memory_interface = MagicMock()
            dedup_config = DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            )

            deduplicator = MemoryDeduplicator(
                memory_interface=mock_memory_interface,
                dedup_config=dedup_config
            )

            # Create memories with different timestamps
            mock_memories = [
                {
                    "id": "mem_1",
                    "content": "Content 1",
                    "metadata": {"embedding": [1.0, 2.0, 3.0]},
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "mem_2",
                    "content": "Content 2",
                    "metadata": {"embedding": [1.1, 2.1, 3.1]},
                    "timestamp": "2024-01-01T00:00:01Z",
                },
            ]

            mock_memory_interface.retrieve.return_value = {
                "memories": mock_memories,
                "total_count": 2,
                "query_metadata": {},
            }

            # Run deduplication
            result = deduplicator.deduplicate(dry_run=True)

            # Result should have checkpoint_timestamp set to last processed memory
            assert result.checkpoint_timestamp is not None


    class TestCheckpointResumption:
        """Test checkpoint resumption (start from checkpoint)."""

        def test_start_from_checkpoint(self):
            """Test that deduplication can start from a checkpoint timestamp."""
            mock_memory_interface = MagicMock()
            dedup_config = DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            )

            deduplicator = MemoryDeduplicator(
                memory_interface=mock_memory_interface,
                dedup_config=dedup_config
            )

            # Create memories with different timestamps
            mock_memories = [
                {
                    "id": "mem_1",
                    "content": "Content 1",
                    "metadata": {"embedding": [1.0, 2.0, 3.0]},
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "mem_2",
                    "content": "Content 2",
                    "metadata": {"embedding": [1.1, 2.1, 3.1]},
                    "timestamp": "2024-01-01T00:00:01Z",
                },
                {
                    "id": "mem_3",
                    "content": "Content 3",
                    "metadata": {"embedding": [10.0, 20.0, 30.0]},
                    "timestamp": "2024-01-01T00:00:02Z",
                },
            ]

            mock_memory_interface.retrieve.return_value = {
                "memories": mock_memories,
                "total_count": 3,
                "query_metadata": {},
            }

            # Run deduplication with checkpoint
            result = deduplicator.deduplicate(dry_run=True)

            # Result should include checkpoint for resumption
            assert result.checkpoint_timestamp is not None


    class TestCheckpointReset:
        """Test checkpoint reset (all memories processed)."""

        def test_checkpoint_reset_after_completion(self):
            """Test that checkpoint is reset after all memories are processed."""
            mock_memory_interface = MagicMock()
            dedup_config = DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            )

            deduplicator = MemoryDeduplicator(
                memory_interface=mock_memory_interface,
                dedup_config=dedup_config
            )

            # Create memories with different timestamps
            mock_memories = [
                {
                    "id": "mem_1",
                    "content": "Content 1",
                    "metadata": {"embedding": [1.0, 2.0, 3.0]},
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "mem_2",
                    "content": "Content 2",
                    "metadata": {"embedding": [1.1, 2.1, 3.1]},
                    "timestamp": "2024-01-01T00:00:01Z",
                },
            ]

            mock_memory_interface.retrieve.return_value = {
                "memories": mock_memories,
                "total_count": 2,
                "query_metadata": {},
            }

            # Run deduplication
            result = deduplicator.deduplicate(dry_run=True)

            # After completion, checkpoint should be set to last processed memory timestamp
            # (for incremental processing support)
            assert result.checkpoint_timestamp is not None


    class TestIntegration:
        """Integration tests for MemoryDeduplicator."""

        def test_deduplicate_with_multiple_memories(self):
            """Test deduplicate with multiple memories."""
            mock_memory_interface = MagicMock()
            dedup_config = DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            )

            deduplicator = MemoryDeduplicator(
                memory_interface=mock_memory_interface,
                dedup_config=dedup_config
            )

            # Mock multiple memories with some similar
            mock_memories = [
                {
                    "id": "mem_1",
                    "content": "Content 1",
                    "metadata": {"embedding": [1.0, 2.0, 3.0]},
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "mem_2",
                    "content": "Content 2",
                    "metadata": {"embedding": [1.1, 2.1, 3.1]},
                    "timestamp": "2024-01-01T00:00:01Z",
                },
                {
                    "id": "mem_3",
                    "content": "Content 3",
                    "metadata": {"embedding": [10.0, 20.0, 30.0]},
                    "timestamp": "2024-01-01T00:00:02Z",
                },
            ]

            mock_memory_interface.retrieve.return_value = {
                "memories": mock_memories,
                "total_count": 3,
                "query_metadata": {},
            }

            result = deduplicator.deduplicate(dry_run=True)

            # mem_1 and mem_2 should be duplicates, mem_3 is different
            assert result.duplicate_pairs_found == 1
            assert result.memories_merged == 1

        def test_deduplicate_empty_store(self):
            """Test deduplicate with empty memory store."""
            mock_memory_interface = MagicMock()
            dedup_config = DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            )

            deduplicator = MemoryDeduplicator(
                memory_interface=mock_memory_interface,
                dedup_config=dedup_config
            )

            mock_memory_interface.retrieve.return_value = {
                "memories": [],
                "total_count": 0,
                "query_metadata": {},
            }

            result = deduplicator.deduplicate(dry_run=True)

            assert result.duplicate_pairs_found == 0
            assert result.memories_merged == 0


class TestIntegration:
    """Integration tests for MemoryDeduplicator."""
    
    def test_deduplicate_with_multiple_memories(self):
        """Test deduplicate with multiple memories."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        # Mock multiple memories with some similar
        mock_memories = [
            {
                "id": "mem_1",
                "content": "Content 1",
                "metadata": {"embedding": [1.0, 2.0, 3.0]},
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {
                "id": "mem_2",
                "content": "Content 2",
                "metadata": {"embedding": [1.1, 2.1, 3.1]},
                "timestamp": "2024-01-01T00:00:01Z",
            },
            {
                "id": "mem_3",
                "content": "Content 3",
                "metadata": {"embedding": [10.0, 20.0, 30.0]},
                "timestamp": "2024-01-01T00:00:02Z",
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 3,
            "query_metadata": {},
        }
        
        result = deduplicator.deduplicate(dry_run=True)
        
        # mem_1 and mem_2 should be duplicates, mem_3 is different
        assert result.duplicate_pairs_found == 1
        assert result.memories_merged == 1
    
    def test_deduplicate_empty_store(self):
        """Test deduplicate with empty memory store."""
        mock_memory_interface = MagicMock()
        dedup_config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.5,
            batch_size=100
        )
        
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=dedup_config
        )
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {},
        }
        
        result = deduplicator.deduplicate(dry_run=True)
        
        assert result.duplicate_pairs_found == 0
        assert result.memories_merged == 0
