"""
Unit tests for MemoryPruner component.

Tests the MemoryPruner component's pruning strategies (threshold, percentile,
capacity), protected memory filtering, and integration with memory interface.
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.schemas import (
    PruningConfig,
    PruningStrategy,
    PruningResult,
)


class TestMemoryPrunerInitialization:
    """Test MemoryPruner initialization."""
    
    def test_initialization_with_all_dependencies(self):
        """Test MemoryPruner initialization with all dependencies."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()
        
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.3,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        
        assert pruner.memory_interface == mock_memory_interface
        assert pruner.pruning_config == pruning_config
        assert pruner.metrics_collector == mock_metrics_collector
        assert pruner.logger == mock_logger
    
    def test_initialization_without_optional_dependencies(self):
        """Test MemoryPruner initialization without optional dependencies."""
        mock_memory_interface = MagicMock()
        
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.3,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        assert pruner.memory_interface == mock_memory_interface
        assert pruner.pruning_config == pruning_config
        assert pruner.metrics_collector is None
        assert pruner.logger is None


class TestThresholdPruning:
    """Test threshold-based pruning strategy."""
    
    def test_prune_threshold_deletes_below_threshold(self):
        """Test that memories below threshold are deleted."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock memories with varying importance scores
        mock_memories = [
            {
                "id": "mem_low",
                "content": "Low importance",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_medium",
                "content": "Medium importance",
                "metadata": {"importance": 0.6},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_high",
                "content": "High importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 3,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert isinstance(result, PruningResult)
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_called_once_with("mem_low")
    
    def test_prune_threshold_excludes_protected(self):
        """Test that protected memories are excluded from threshold pruning."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock memories with one protected
        mock_memories = [
            {
                "id": "mem_low_protected",
                "content": "Low importance but protected",
                "metadata": {"importance": 0.3, "protected": True},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_low_unprotected",
                "content": "Low importance and unprotected",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 2,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        # Only the unprotected low importance memory should be deleted
        mock_memory_interface.delete.assert_called_once_with("mem_low_unprotected")
    
    def test_prune_threshold_empty_store(self):
        """Test threshold pruning with empty memory store."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert result.memories_deleted == 0
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_not_called()
    
    def test_prune_threshold_dry_run(self):
        """Test threshold pruning in dry_run mode."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memories = [
            {
                "id": "mem_low",
                "content": "Low importance",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = pruner.prune(dry_run=True)
        
        assert result.memories_deleted == 1
        # In dry_run, delete should not be called
        mock_memory_interface.delete.assert_not_called()


class TestPercentilePruning:
    """Test percentile-based pruning strategy."""
    
    def test_prune_percentile_bottom_n_percent(self):
        """Test that bottom N% of memories are deleted."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.PERCENTILE,
            percentile=20.0,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock 5 memories with varying importance
        mock_memories = [
            {
                "id": "mem_1",
                "content": "Lowest importance",
                "metadata": {"importance": 0.1},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_2",
                "content": "Low importance",
                "metadata": {"importance": 0.2},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_3",
                "content": "Medium importance",
                "metadata": {"importance": 0.5},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_4",
                "content": "High importance",
                "metadata": {"importance": 0.8},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_5",
                "content": "Highest importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 5,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # 20% of 5 = 1 memory should be deleted
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_called_once_with("mem_1")
    
    def test_prune_percentile_excludes_protected(self):
        """Test that protected memories are excluded from percentile pruning."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.PERCENTILE,
            percentile=50.0,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock 4 memories with one protected at lowest importance
        mock_memories = [
            {
                "id": "mem_protected",
                "content": "Protected but lowest importance",
                "metadata": {"importance": 0.1, "protected": True},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_2",
                "content": "Low importance",
                "metadata": {"importance": 0.2},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_3",
                "content": "Medium importance",
                "metadata": {"importance": 0.5},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_4",
                "content": "High importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 4,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # 50% of 3 unprotected = 1 memory should be deleted (mem_2, not protected)
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_called_once_with("mem_2")
    
    def test_prune_percentile_empty_store(self):
        """Test percentile pruning with empty memory store."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.PERCENTILE,
            percentile=10.0,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert result.memories_deleted == 0
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_not_called()


class TestCapacityPruning:
    """Test capacity-based pruning strategy."""
    
    def test_prune_capacity_reduces_to_limit(self):
        """Test that memories are deleted to meet capacity limit."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=3,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock 5 memories
        mock_memories = [
            {
                "id": "mem_1",
                "content": "Lowest importance",
                "metadata": {"importance": 0.1},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_2",
                "content": "Low importance",
                "metadata": {"importance": 0.2},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_3",
                "content": "Medium importance",
                "metadata": {"importance": 0.5},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_4",
                "content": "High importance",
                "metadata": {"importance": 0.8},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_5",
                "content": "Highest importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 5,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # 5 - 3 = 2 memories should be deleted
        assert result.memories_deleted == 2
        assert result.deletion_failures == 0
        # Should delete mem_1 and mem_2 (lowest importance)
        assert mock_memory_interface.delete.call_count == 2
    
    def test_prune_capacity_excludes_protected(self):
        """Test that protected memories are excluded from capacity pruning."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=2,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock 4 memories with one protected at lowest importance
        # Capacity limit is 2, so we need to delete 2 memories total
        # But mem_protected is excluded, so only 1 unprotected gets deleted
        mock_memories = [
            {
                "id": "mem_protected",
                "content": "Protected but lowest importance",
                "metadata": {"importance": 0.1, "protected": True},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_2",
                "content": "Low importance",
                "metadata": {"importance": 0.2},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_3",
                "content": "Medium importance",
                "metadata": {"importance": 0.5},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_4",
                "content": "High importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 4,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # 4 - 2 = 2 should be deleted, but mem_protected is excluded
        # Only 1 unprotected memory (mem_2) gets deleted
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        # Verify protected memory was not deleted
        delete_calls = [call[0][0] for call in mock_memory_interface.delete.call_args_list]
        assert "mem_protected" not in delete_calls
    
    def test_prune_capacity_under_limit(self):
        """Test no deletion when under capacity limit."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=10,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Mock 5 memories (under limit of 10)
        mock_memories = [
            {
                "id": f"mem_{i}",
                "content": f"Content {i}",
                "metadata": {"importance": 0.5},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            }
            for i in range(5)
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 5,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert result.memories_deleted == 0
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_not_called()
    
    def test_prune_capacity_empty_store(self):
        """Test capacity pruning with empty memory store."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=10,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [],
            "total_count": 0,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        assert result.memories_deleted == 0
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_not_called()


class TestDeterministicOrdering:
    """Test deterministic ordering of memories for deletion."""
    
    def test_prune_sorts_by_importance_timestamp_id(self):
        """Test that memories are sorted by (importance, timestamp, id)."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=1,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        # Create memories with identical importance but different timestamps and IDs
        base_time = datetime.now(UTC)
        mock_memories = [
            {
                "id": "mem_c",
                "content": "Content C",
                "metadata": {"importance": 0.5},
                "timestamp": (base_time - timedelta(days=1)).isoformat() + "Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_a",
                "content": "Content A",
                "metadata": {"importance": 0.5},
                "timestamp": (base_time - timedelta(days=3)).isoformat() + "Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_b",
                "content": "Content B",
                "metadata": {"importance": 0.5},
                "timestamp": (base_time - timedelta(days=2)).isoformat() + "Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 3,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # Should delete mem_a (oldest), then mem_b (next oldest)
        assert result.memories_deleted == 2
        delete_calls = [call[0][0] for call in mock_memory_interface.delete.call_args_list]
        assert delete_calls[0] == "mem_a"
        assert delete_calls[1] == "mem_b"


class TestDeletionFailureHandling:
    """Test handling of individual deletion failures."""
    
    def test_prune_continues_on_deletion_failure(self):
        """Test that pruning continues after a deletion failure."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memories = [
            {
                "id": "mem_fail",
                "content": "Will fail",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_succeed",
                "content": "Will succeed",
                "metadata": {"importance": 0.4},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 2,
            "query_metadata": {}
        }
        
        # Make first deletion fail
        mock_memory_interface.delete.side_effect = [
            Exception("Simulated failure"),
            None,
        ]
        
        result = pruner.prune()
        
        assert result.memories_deleted == 1
        assert result.deletion_failures == 1
        # Should have attempted both deletions
        assert mock_memory_interface.delete.call_count == 2
    
    def test_prune_logs_deletion_errors(self):
        """Test that deletion errors are logged."""
        mock_memory_interface = MagicMock()
        mock_logger = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config,
            logger=mock_logger
        )
        
        mock_memories = [
            {
                "id": "mem_fail",
                "content": "Will fail",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 1,
            "query_metadata": {}
        }
        
        mock_memory_interface.delete.side_effect = Exception("Simulated failure")
        
        result = pruner.prune()
        
        # Logger should be called with error
        mock_logger.log.assert_called()
        assert result.deletion_failures == 1


class TestIntegration:
    """Integration tests for MemoryPruner."""
    
    def test_prune_with_mixed_protected_unprotected(self):
        """Test pruning with mix of protected and unprotected memories."""
        mock_memory_interface = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config
        )
        
        mock_memories = [
            {
                "id": "protected_high",
                "content": "Protected high importance",
                "metadata": {"importance": 0.9, "protected": True},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "protected_low",
                "content": "Protected low importance",
                "metadata": {"importance": 0.2, "protected": True},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "unprotected_high",
                "content": "Unprotected high importance",
                "metadata": {"importance": 0.9},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
            {
                "id": "unprotected_low",
                "content": "Unprotected low importance",
                "metadata": {"importance": 0.2},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 4,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # Only unprotected_low should be deleted
        assert result.memories_deleted == 1
        assert result.deletion_failures == 0
        mock_memory_interface.delete.assert_called_once_with("unprotected_low")
        
        # Verify protected memories were not deleted
        delete_calls = [call[0][0] for call in mock_memory_interface.delete.call_args_list]
        assert "protected_high" not in delete_calls
        assert "protected_low" not in delete_calls
    
    def test_prune_records_metrics(self):
        """Test that metrics are recorded."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config,
            metrics_collector=mock_metrics_collector
        )
        
        mock_memories = [
            {
                "id": "mem_low",
                "content": "Low importance",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # Metrics should be recorded
        mock_metrics_collector.increment.assert_any_call("memory_prune.deleted", 1)
        mock_metrics_collector.increment.assert_any_call("memory_prune.failures", 0)
        mock_metrics_collector.record_duration.assert_called()
    
    def test_prune_logs_completion(self):
        """Test that completion is logged."""
        mock_memory_interface = MagicMock()
        mock_logger = MagicMock()
        pruning_config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.5,
            min_importance_protected=0.8
        )
        
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=pruning_config,
            logger=mock_logger
        )
        
        mock_memories = [
            {
                "id": "mem_low",
                "content": "Low importance",
                "metadata": {"importance": 0.3},
                "timestamp": "2024-01-01T00:00:00Z",
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = pruner.prune()
        
        # Logger should be called with completion event
        mock_logger.log.assert_called()
        log_call = mock_logger.log.call_args
        assert log_call[1]["event"] == "memory_prune_completed"
        assert log_call[1]["payload"]["strategy"] == "threshold"
        assert log_call[1]["payload"]["memories_deleted"] == 1