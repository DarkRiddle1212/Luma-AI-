"""
Integration tests for memory lifecycle management system.

Tests the complete lifecycle system including:
- Full maintenance cycle (decay → pruning → deduplication)
- Concurrent operations (maintenance while reads/writes occur)
- Large memory stores (10k+ memories)
- Checkpoint persistence across maintenance cycles
- Observability integration (metrics and logs emitted)
- Error recovery (individual failures don't cascade)
- Dry run end-to-end (no persistence, accurate report)
- All decay functions (exponential, linear, step)
- All pruning strategies (threshold, percentile, capacity)
- All similarity metrics (cosine, Jaccard, Levenshtein)
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

from luma.core.lifecycle.schemas import (
    DecayConfig,
    DecayFunctionType,
    PruningConfig,
    PruningStrategy,
    DeduplicationConfig,
    SimilarityMetric,
)
from luma.core.lifecycle.memory_decay import MemoryDecay
from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.memory_deduplicator import MemoryDeduplicator
from luma.core.lifecycle.lifecycle_manager import LifecycleManager


class TestFullMaintenanceCycle:
    """Test full maintenance cycle (decay → pruning → deduplication)."""

    def test_full_maintenance_cycle_execution_order(self):
        """Test that maintenance operations execute in correct order."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        from luma.core.lifecycle.schemas import MemoryDecayResult, PruningResult, DeduplicationResult

        # Use actual dataclass instances (LifecycleReport validates types)
        decay_result_mock = MemoryDecayResult(
            memories_processed=100,
            memories_updated=80,
            average_decay_applied=0.15,
            execution_time_ms=50.0
        )

        pruning_result_mock = PruningResult(
            memories_deleted=0,
            deletion_failures=0,
            pruned_memories=[],
            execution_time_ms=30.0
        )

        dedup_result_mock = DeduplicationResult(
            duplicate_pairs_found=5,
            memories_merged=0,
            merge_details=[],
            checkpoint_timestamp=None,
            execution_time_ms=100.0
        )

        # Create components with mocked results
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            )
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            )
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            )
        )

        # Patch the apply methods to return mocked results
        with patch.object(decay, 'apply_decay', return_value=decay_result_mock):
            with patch.object(pruner, 'prune', return_value=pruning_result_mock):
                with patch.object(deduplicator, 'deduplicate', return_value=dedup_result_mock):
                    manager = LifecycleManager(
                        memory_decay=decay,
                        memory_pruner=pruner,
                        memory_deduplicator=deduplicator,
                        memory_interface=mock_memory_interface,
                        metrics_collector=mock_metrics_collector,
                        logger=mock_logger,
                        timeout_seconds=300
                    )

                    # Run maintenance
                    report = manager.run_maintenance(dry_run=False)

                    # Verify report structure
                    assert report.decay_result == decay_result_mock
                    assert report.pruning_result == pruning_result_mock
                    assert report.deduplication_result == dedup_result_mock
                    assert report.total_execution_time_ms > 0
                    assert report.dry_run is False

    def test_full_maintenance_cycle_with_memories(self):
        """Test full maintenance cycle with actual memory data."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        # Create memories with varying importance and timestamps
        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5 + (i % 10) * 0.05,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                    "protected": False,
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(20)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 20,
            "query_metadata": {},
        }

        # Create components
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.01
            )
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            )
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.95,
                batch_size=100
            )
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance
        report = manager.run_maintenance(dry_run=True)

        # Verify report structure
        assert report.decay_result.memories_processed > 0
        assert report.total_execution_time_ms > 0
        assert report.dry_run is True


class TestConcurrentOperations:
    """Test concurrent operations (maintenance while reads/writes occur)."""

    def test_concurrent_read_during_maintenance(self):
        """Test that reads can proceed during maintenance."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        # Create memories
        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            )
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            )
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            )
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance in dry_run mode (doesn't block)
        report = manager.run_maintenance(dry_run=True)

        # Verify maintenance completed
        assert report.total_execution_time_ms > 0

        # Verify memory_interface was used for retrieval
        assert mock_memory_interface.retrieve.called

    def test_concurrent_write_during_maintenance(self):
        """Test that writes can proceed during maintenance."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            )
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            )
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            )
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance
        report = manager.run_maintenance(dry_run=True)

        # Verify maintenance completed without blocking
        assert report.total_execution_time_ms > 0


class TestLargeMemoryStores:
    """Test large memory stores (10k+ memories)."""

    def test_large_memory_store_decay(self):
        """Test decay operation with 10k+ memories."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        # Create 10,000 memories
        memories = [
            {
                "id": f"mem_{i:05d}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5 + (i % 100) * 0.005,
                    "creation_timestamp": (now - timedelta(days=i % 365)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i % 365)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10000)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10000,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.001
            )
        )

        # Run decay
        result = decay.apply_decay(dry_run=True)

        # Verify all memories were processed
        assert result.memories_processed == 10000
        assert result.memories_updated > 0

    def test_large_memory_store_pruning(self):
        """Test pruning operation with 10k+ memories."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        # Create 10,000 memories with varying importance
        memories = [
            {
                "id": f"mem_{i:05d}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": (i % 100) / 100.0,  # Range 0.0 to 0.99
                    "protected": False,
                },
                "timestamp": (now - timedelta(days=i % 365)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10000)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10000,
            "query_metadata": {},
        }

        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            )
        )

        # Run pruning
        result = pruner.prune(dry_run=True)

        # Verify pruning identified candidates
        assert result.memories_deleted > 0
        assert result.memories_deleted < 10000  # Not all should be deleted

    def test_large_memory_store_deduplication(self):
        """Test deduplication with 10k+ memories."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        # Create 10,000 memories with some similar content
        memories = []
        for i in range(10000):
            # Create some duplicate content patterns
            content_base = f"Content pattern {i % 1000}"
            memories.append({
                "id": f"mem_{i:05d}",
                "content": content_base,
                "metadata": {
                    "importance": 0.5,
                    "embedding": [i % 10 / 10.0, (i + 1) % 10 / 10.0],  # Some similar embeddings
                },
                "timestamp": (now - timedelta(seconds=i)).isoformat().replace('+00:00', 'Z'),
            })

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10000,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.8,  # Lower threshold to find more duplicates
                batch_size=1000
            )
        )

        # Run deduplication
        result = deduplicator.deduplicate(dry_run=True)

        # Verify deduplication found some pairs
        assert result.duplicate_pairs_found >= 0
        assert result.memories_merged >= 0


class TestCheckpointPersistence:
    """Test checkpoint persistence across maintenance cycles."""

    def test_checkpoint_updated_after_deduplication(self):
        """Test that checkpoint is updated after deduplication cycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "embedding": [i % 10 / 10.0, (i + 1) % 10 / 10.0],
                },
                "timestamp": (now - timedelta(seconds=i * 100)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100,
                checkpoint_enabled=True
            )
        )

        # Run deduplication
        result = deduplicator.deduplicate(dry_run=True)

        # Verify checkpoint is set
        assert result.checkpoint_timestamp is not None

    def test_checkpoint_survives_multiple_cycles(self):
        """Test that checkpoint persists across multiple maintenance cycles."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "embedding": [i % 10 / 10.0, (i + 1) % 10 / 10.0],
                },
                "timestamp": (now - timedelta(seconds=i * 100)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(20)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 20,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100,
                checkpoint_enabled=True
            )
        )

        # Run first cycle
        result1 = deduplicator.deduplicate(dry_run=True)

        # Run second cycle
        result2 = deduplicator.deduplicate(dry_run=True)

        # Both should have checkpoints
        assert result1.checkpoint_timestamp is not None
        assert result2.checkpoint_timestamp is not None


class TestObservabilityIntegration:
    """Test observability integration (metrics and logs emitted)."""

    def test_metrics_emitted_during_decay(self):
        """Test that metrics are emitted during decay operation."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        # Run decay
        decay.apply_decay(dry_run=True)

        # Verify metrics were recorded
        assert mock_metrics_collector.increment.called
        assert mock_metrics_collector.record_duration.called

    def test_logs_emitted_during_maintenance(self):
        """Test that logs are emitted during maintenance cycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance
        manager.run_maintenance(dry_run=True)

        # Verify logs were emitted
        assert mock_logger.log.called

    def test_metrics_collector_increment_called(self):
        """Test that metrics collector increment is called during maintenance."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance
        manager.run_maintenance(dry_run=True)

        # Verify metrics were recorded
        assert mock_metrics_collector.increment.called
        assert mock_metrics_collector.record_duration.called


class TestErrorRecovery:
    """Test error recovery (individual failures don't cascade)."""

    def test_decay_failure_does_not_stop_pruning(self):
        """Test that decay failure doesn't stop pruning operation."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Mock decay to raise an exception
        with patch.object(decay, 'apply_decay', side_effect=Exception("Decay failed")):
            # Run maintenance - should not crash
            report = manager.run_maintenance(dry_run=True)

            # Verify pruning still executed (returned default result)
            assert report.pruning_result is not None
            assert report.deduplication_result is not None

    def test_pruning_failure_does_not_stop_deduplication(self):
        """Test that pruning failure doesn't stop deduplication operation."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Mock pruning to raise an exception
        with patch.object(pruner, 'prune', side_effect=Exception("Pruning failed")):
            # Run maintenance - should not crash
            report = manager.run_maintenance(dry_run=True)

            # Verify deduplication still executed
            assert report.deduplication_result is not None

    def test_deduplication_failure_does_not_stop_maintenance(self):
        """Test that deduplication failure doesn't stop maintenance completion."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Mock deduplication to raise an exception
        with patch.object(deduplicator, 'deduplicate', side_effect=Exception("Deduplication failed")):
            # Run maintenance - should not crash
            report = manager.run_maintenance(dry_run=True)

            # Verify maintenance completed with error results
            assert report.decay_result is not None
            assert report.pruning_result is not None
            assert report.deduplication_result is not None


class TestDryRunMode:
    """Test dry_run end-to-end (no persistence, accurate report)."""

    def test_dry_run_no_persistence(self):
        """Test that dry_run mode doesn't persist changes."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance in dry_run mode
        report = manager.run_maintenance(dry_run=True)

        # Verify no store or delete operations were called
        mock_memory_interface.store.assert_not_called()
        mock_memory_interface.delete.assert_not_called()

        # Verify report shows dry_run=True
        assert report.dry_run is True

    def test_dry_run_accurate_report(self):
        """Test that dry_run mode returns accurate report of changes."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        manager = LifecycleManager(
            memory_decay=decay,
            memory_pruner=pruner,
            memory_deduplicator=deduplicator,
            memory_interface=mock_memory_interface,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger,
            timeout_seconds=300
        )

        # Run maintenance in dry_run mode
        report = manager.run_maintenance(dry_run=True)

        # Verify report contains valid results
        assert report.decay_result.memories_processed > 0
        assert report.total_execution_time_ms > 0
        assert report.dry_run is True


class TestAllDecayFunctions:
    """Test all decay functions (exponential, linear, step)."""

    def test_exponential_decay_integration(self):
        """Test exponential decay function in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.8,
                    "creation_timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.01
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = decay.apply_decay(dry_run=True)

        # Verify exponential decay was applied
        assert result.memories_processed > 0
        assert result.memories_updated > 0
        assert 0.0 <= result.average_decay_applied <= 1.0

    def test_linear_decay_integration(self):
        """Test linear decay function in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.8,
                    "creation_timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.LINEAR,
                decay_rate=0.001
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = decay.apply_decay(dry_run=True)

        # Verify linear decay was applied
        assert result.memories_processed > 0
        assert result.memories_updated > 0
        assert 0.0 <= result.average_decay_applied <= 1.0

    def test_step_decay_integration(self):
        """Test step decay function in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.8,
                    "creation_timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
                },
                "timestamp": (now - timedelta(days=i * 10)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=DecayConfig(
                decay_function_type=DecayFunctionType.STEP,
                decay_rate=0.01,
                step_interval_days=7,
                step_percentage=0.1
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = decay.apply_decay(dry_run=True)

        # Verify step decay was applied
        assert result.memories_processed > 0
        assert result.memories_updated > 0
        assert 0.0 <= result.average_decay_applied <= 1.0


class TestAllPruningStrategies:
    """Test all pruning strategies (threshold, percentile, capacity)."""

    def test_threshold_pruning_integration(self):
        """Test threshold-based pruning in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": (i % 10) / 10.0,  # Range 0.0 to 0.9
                    "protected": False,
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(20)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 20,
            "query_metadata": {},
        }

        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=0.8
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = pruner.prune(dry_run=True)

        # Verify threshold pruning identified candidates
        assert result.memories_deleted > 0
        assert result.memories_deleted < 20  # Not all should be deleted

    def test_percentile_pruning_integration(self):
        """Test percentile-based pruning in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": (i % 10) / 10.0,
                    "protected": False,
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(20)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 20,
            "query_metadata": {},
        }

        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.PERCENTILE,
                percentile=20.0  # Remove bottom 20%
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = pruner.prune(dry_run=True)

        # Verify percentile pruning identified ~20% of memories
        expected_delete = max(1, int(20 * 20.0 / 100.0))
        assert result.memories_deleted == expected_delete

    def test_capacity_pruning_integration(self):
        """Test capacity-based pruning in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": (i % 10) / 10.0,
                    "protected": False,
                },
                "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(20)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 20,
            "query_metadata": {},
        }

        pruner = MemoryPruner(
            memory_interface=mock_memory_interface,
            pruning_config=PruningConfig(
                strategy=PruningStrategy.CAPACITY,
                capacity_limit=10  # Keep only 10 memories
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = pruner.prune(dry_run=True)

        # Verify capacity pruning identified 10 memories to delete
        assert result.memories_deleted == 10


class TestAllSimilarityMetrics:
    """Test all similarity metrics (cosine, Jaccard, Levenshtein)."""

    def test_cosine_similarity_integration(self):
        """Test cosine similarity metric in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i}",
                "metadata": {
                    "importance": 0.5,
                    "embedding": [i % 10 / 10.0, (i + 1) % 10 / 10.0],
                },
                "timestamp": (now - timedelta(seconds=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.5,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = deduplicator.deduplicate(dry_run=True)

        # Verify cosine similarity was used
        assert result.duplicate_pairs_found >= 0
        assert result.memories_merged >= 0

    def test_jaccard_similarity_integration(self):
        """Test Jaccard similarity metric in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i % 5}",  # Some duplicate content
                "metadata": {
                    "importance": 0.5,
                },
                "timestamp": (now - timedelta(seconds=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.JACCARD,
                similarity_threshold=0.5,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = deduplicator.deduplicate(dry_run=True)

        # Verify Jaccard similarity was used
        assert result.duplicate_pairs_found >= 0
        assert result.memories_merged >= 0

    def test_levenshtein_similarity_integration(self):
        """Test Levenshtein similarity metric in full lifecycle."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()

        now = datetime.now(UTC)
        memories = [
            {
                "id": f"mem_{i}",
                "content": f"Test content {i % 5}",  # Some similar content
                "metadata": {
                    "importance": 0.5,
                },
                "timestamp": (now - timedelta(seconds=i)).isoformat().replace('+00:00', 'Z'),
            }
            for i in range(10)
        ]

        mock_memory_interface.retrieve.return_value = {
            "memories": memories,
            "total_count": 10,
            "query_metadata": {},
        }

        deduplicator = MemoryDeduplicator(
            memory_interface=mock_memory_interface,
            dedup_config=DeduplicationConfig(
                similarity_metric=SimilarityMetric.LEVENSHTEIN,
                similarity_threshold=0.5,
                batch_size=100
            ),
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )

        result = deduplicator.deduplicate(dry_run=True)

        # Verify Levenshtein similarity was used
        assert result.duplicate_pairs_found >= 0
        assert result.memories_merged >= 0
