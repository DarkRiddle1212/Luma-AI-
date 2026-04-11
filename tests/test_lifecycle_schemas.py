"""
Unit tests for lifecycle schemas validation.

Tests all schema dataclasses and configuration classes for proper validation
of fields, ranges, and error conditions as specified in the requirements.
"""

import pytest
from datetime import datetime
from luma.core.lifecycle.schemas import (
    MemoryDecayResult,
    PrunedMemory,
    PruningResult,
    MergeDetail,
    DeduplicationResult,
    LifecycleReport,
    DecayConfig,
    PruningConfig,
    DeduplicationConfig,
    DecayFunctionType,
    PruningStrategy,
    SimilarityMetric,
)


class TestMemoryDecayResult:
    """Test MemoryDecayResult validation."""
    
    def test_valid_memory_decay_result(self):
        """Test creating valid MemoryDecayResult."""
        result = MemoryDecayResult(
            memories_processed=1000,
            memories_updated=800,
            average_decay_applied=0.15,
            execution_time_ms=250.5
        )
        assert result.memories_processed == 1000
        assert result.memories_updated == 800
        assert result.average_decay_applied == 0.15
        assert result.execution_time_ms == 250.5
    
    def test_negative_memories_processed(self):
        """Test negative memories_processed raises ValueError."""
        with pytest.raises(ValueError, match="memories_processed must be non-negative"):
            MemoryDecayResult(
                memories_processed=-1,
                memories_updated=0,
                average_decay_applied=0.0,
                execution_time_ms=0.0
            )
    
    def test_negative_memories_updated(self):
        """Test negative memories_updated raises ValueError."""
        with pytest.raises(ValueError, match="memories_updated must be non-negative"):
            MemoryDecayResult(
                memories_processed=100,
                memories_updated=-1,
                average_decay_applied=0.0,
                execution_time_ms=0.0
            )
    
    def test_memories_updated_exceeds_processed(self):
        """Test memories_updated > memories_processed raises ValueError."""
        with pytest.raises(ValueError, match="memories_updated .* cannot exceed memories_processed"):
            MemoryDecayResult(
                memories_processed=100,
                memories_updated=150,
                average_decay_applied=0.0,
                execution_time_ms=0.0
            )
    
    def test_average_decay_below_range(self):
        """Test average_decay_applied < 0 raises ValueError."""
        with pytest.raises(ValueError, match="average_decay_applied must be in \\[0, 1\\]"):
            MemoryDecayResult(
                memories_processed=100,
                memories_updated=50,
                average_decay_applied=-0.1,
                execution_time_ms=0.0
            )
    
    def test_average_decay_above_range(self):
        """Test average_decay_applied > 1 raises ValueError."""
        with pytest.raises(ValueError, match="average_decay_applied must be in \\[0, 1\\]"):
            MemoryDecayResult(
                memories_processed=100,
                memories_updated=50,
                average_decay_applied=1.5,
                execution_time_ms=0.0
            )
    
    def test_negative_execution_time(self):
        """Test negative execution_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            MemoryDecayResult(
                memories_processed=100,
                memories_updated=50,
                average_decay_applied=0.5,
                execution_time_ms=-10.0
            )


class TestPrunedMemory:
    """Test PrunedMemory validation."""
    
    def test_valid_pruned_memory(self):
        """Test creating valid PrunedMemory."""
        timestamp = datetime.now()
        pruned = PrunedMemory(
            memory_id="mem_123",
            importance_score=0.2,
            final_score=0.15,
            deletion_timestamp=timestamp,
            reason="score"
        )
        assert pruned.memory_id == "mem_123"
        assert pruned.importance_score == 0.2
        assert pruned.final_score == 0.15
        assert pruned.deletion_timestamp == timestamp
        assert pruned.reason == "score"
    
    def test_empty_memory_id(self):
        """Test empty memory_id raises ValueError."""
        with pytest.raises(ValueError, match="memory_id must be a non-empty string"):
            PrunedMemory(
                memory_id="",
                importance_score=0.2,
                final_score=0.15,
                deletion_timestamp=datetime.now(),
                reason="score"
            )
    
    def test_importance_score_below_range(self):
        """Test importance_score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="importance_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=-0.1,
                final_score=0.15,
                deletion_timestamp=datetime.now(),
                reason="score"
            )
    
    def test_importance_score_above_range(self):
        """Test importance_score > 1 raises ValueError."""
        with pytest.raises(ValueError, match="importance_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=1.5,
                final_score=0.15,
                deletion_timestamp=datetime.now(),
                reason="score"
            )
    
    def test_final_score_below_range(self):
        """Test final_score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="final_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=0.2,
                final_score=-0.1,
                deletion_timestamp=datetime.now(),
                reason="score"
            )
    
    def test_final_score_above_range(self):
        """Test final_score > 1 raises ValueError."""
        with pytest.raises(ValueError, match="final_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=0.2,
                final_score=1.5,
                deletion_timestamp=datetime.now(),
                reason="score"
            )
    
    def test_invalid_deletion_timestamp(self):
        """Test non-datetime deletion_timestamp raises ValueError."""
        with pytest.raises(ValueError, match="deletion_timestamp must be a datetime object"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=0.2,
                final_score=0.15,
                deletion_timestamp="2023-01-01",  # String instead of datetime
                reason="score"
            )
    
    def test_empty_reason(self):
        """Test empty reason raises ValueError."""
        with pytest.raises(ValueError, match="reason must be a non-empty string"):
            PrunedMemory(
                memory_id="mem_123",
                importance_score=0.2,
                final_score=0.15,
                deletion_timestamp=datetime.now(),
                reason=""
            )


class TestPruningResult:
    """Test PruningResult validation."""
    
    def test_valid_pruning_result(self):
        """Test creating valid PruningResult."""
        pruned_memories = [
            PrunedMemory("mem_1", 0.1, 0.05, datetime.now(), "score"),
            PrunedMemory("mem_2", 0.2, 0.15, datetime.now(), "score")
        ]
        result = PruningResult(
            memories_deleted=2,
            deletion_failures=1,
            pruned_memories=pruned_memories,
            execution_time_ms=150.0
        )
        assert result.memories_deleted == 2
        assert result.deletion_failures == 1
        assert len(result.pruned_memories) == 2
        assert result.execution_time_ms == 150.0
    
    def test_negative_memories_deleted(self):
        """Test negative memories_deleted raises ValueError."""
        with pytest.raises(ValueError, match="memories_deleted must be non-negative"):
            PruningResult(
                memories_deleted=-1,
                deletion_failures=0,
                pruned_memories=[],
                execution_time_ms=0.0
            )
    
    def test_negative_deletion_failures(self):
        """Test negative deletion_failures raises ValueError."""
        with pytest.raises(ValueError, match="deletion_failures must be non-negative"):
            PruningResult(
                memories_deleted=0,
                deletion_failures=-1,
                pruned_memories=[],
                execution_time_ms=0.0
            )
    
    def test_pruned_memories_length_mismatch(self):
        """Test pruned_memories length != memories_deleted raises ValueError."""
        pruned_memories = [
            PrunedMemory("mem_1", 0.1, 0.05, datetime.now(), "score")
        ]
        with pytest.raises(ValueError, match="pruned_memories length .* must equal memories_deleted"):
            PruningResult(
                memories_deleted=2,  # Mismatch: 2 != 1
                deletion_failures=0,
                pruned_memories=pruned_memories,
                execution_time_ms=0.0
            )
    
    def test_negative_execution_time(self):
        """Test negative execution_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            PruningResult(
                memories_deleted=0,
                deletion_failures=0,
                pruned_memories=[],
                execution_time_ms=-10.0
            )


class TestMergeDetail:
    """Test MergeDetail validation."""
    
    def test_valid_merge_detail(self):
        """Test creating valid MergeDetail."""
        timestamp = datetime.now()
        detail = MergeDetail(
            kept_memory_id="mem_123",
            deleted_memory_id="mem_456",
            similarity_score=0.95,
            merged_tags=["tag1", "tag2"],
            merge_timestamp=timestamp
        )
        assert detail.kept_memory_id == "mem_123"
        assert detail.deleted_memory_id == "mem_456"
        assert detail.similarity_score == 0.95
        assert detail.merged_tags == ["tag1", "tag2"]
        assert detail.merge_timestamp == timestamp
    
    def test_empty_kept_memory_id(self):
        """Test empty kept_memory_id raises ValueError."""
        with pytest.raises(ValueError, match="kept_memory_id must be a non-empty string"):
            MergeDetail(
                kept_memory_id="",
                deleted_memory_id="mem_456",
                similarity_score=0.95,
                merged_tags=[],
                merge_timestamp=datetime.now()
            )
    
    def test_empty_deleted_memory_id(self):
        """Test empty deleted_memory_id raises ValueError."""
        with pytest.raises(ValueError, match="deleted_memory_id must be a non-empty string"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="",
                similarity_score=0.95,
                merged_tags=[],
                merge_timestamp=datetime.now()
            )
    
    def test_same_memory_ids(self):
        """Test kept_memory_id == deleted_memory_id raises ValueError."""
        with pytest.raises(ValueError, match="kept_memory_id and deleted_memory_id must be different"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="mem_123",  # Same as kept_memory_id
                similarity_score=0.95,
                merged_tags=[],
                merge_timestamp=datetime.now()
            )
    
    def test_similarity_score_below_range(self):
        """Test similarity_score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_score must be in \\[0, 1\\]"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="mem_456",
                similarity_score=-0.1,
                merged_tags=[],
                merge_timestamp=datetime.now()
            )
    
    def test_similarity_score_above_range(self):
        """Test similarity_score > 1 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_score must be in \\[0, 1\\]"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="mem_456",
                similarity_score=1.5,
                merged_tags=[],
                merge_timestamp=datetime.now()
            )
    
    def test_invalid_merge_timestamp(self):
        """Test non-datetime merge_timestamp raises ValueError."""
        with pytest.raises(ValueError, match="merge_timestamp must be a datetime object"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="mem_456",
                similarity_score=0.95,
                merged_tags=[],
                merge_timestamp="2023-01-01"  # String instead of datetime
            )
    
    def test_invalid_merged_tags(self):
        """Test non-list merged_tags raises ValueError."""
        with pytest.raises(ValueError, match="merged_tags must be a list"):
            MergeDetail(
                kept_memory_id="mem_123",
                deleted_memory_id="mem_456",
                similarity_score=0.95,
                merged_tags="tag1,tag2",  # String instead of list
                merge_timestamp=datetime.now()
            )


class TestDeduplicationResult:
    """Test DeduplicationResult validation."""
    
    def test_valid_deduplication_result(self):
        """Test creating valid DeduplicationResult."""
        merge_details = [
            MergeDetail("mem_1", "mem_2", 0.95, ["tag1"], datetime.now()),
            MergeDetail("mem_3", "mem_4", 0.92, ["tag2"], datetime.now())
        ]
        result = DeduplicationResult(
            duplicate_pairs_found=3,
            memories_merged=2,
            merge_details=merge_details,
            checkpoint_timestamp=datetime.now(),
            execution_time_ms=500.0
        )
        assert result.duplicate_pairs_found == 3
        assert result.memories_merged == 2
        assert len(result.merge_details) == 2
        assert result.execution_time_ms == 500.0
    
    def test_negative_duplicate_pairs_found(self):
        """Test negative duplicate_pairs_found raises ValueError."""
        with pytest.raises(ValueError, match="duplicate_pairs_found must be non-negative"):
            DeduplicationResult(
                duplicate_pairs_found=-1,
                memories_merged=0,
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=0.0
            )
    
    def test_negative_memories_merged(self):
        """Test negative memories_merged raises ValueError."""
        with pytest.raises(ValueError, match="memories_merged must be non-negative"):
            DeduplicationResult(
                duplicate_pairs_found=0,
                memories_merged=-1,
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=0.0
            )
    
    def test_memories_merged_exceeds_pairs_found(self):
        """Test memories_merged > duplicate_pairs_found raises ValueError."""
        with pytest.raises(ValueError, match="memories_merged .* cannot exceed duplicate_pairs_found"):
            DeduplicationResult(
                duplicate_pairs_found=2,
                memories_merged=5,  # More than pairs found
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=0.0
            )
    
    def test_merge_details_length_mismatch(self):
        """Test merge_details length != memories_merged raises ValueError."""
        merge_details = [
            MergeDetail("mem_1", "mem_2", 0.95, ["tag1"], datetime.now())
        ]
        with pytest.raises(ValueError, match="merge_details length .* must equal memories_merged"):
            DeduplicationResult(
                duplicate_pairs_found=3,
                memories_merged=2,  # Mismatch: 2 != 1
                merge_details=merge_details,
                checkpoint_timestamp=None,
                execution_time_ms=0.0
            )
    
    def test_negative_execution_time(self):
        """Test negative execution_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            DeduplicationResult(
                duplicate_pairs_found=0,
                memories_merged=0,
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=-10.0
            )


class TestLifecycleReport:
    """Test LifecycleReport validation."""
    
    def test_valid_lifecycle_report(self):
        """Test creating valid LifecycleReport."""
        decay_result = MemoryDecayResult(100, 80, 0.1, 100.0)
        pruning_result = PruningResult(0, 1, [], 50.0)  # 0 deleted, so empty list is valid
        dedup_result = DeduplicationResult(5, 0, [], None, 200.0)  # 0 merged, so empty list is valid
        
        report = LifecycleReport(
            decay_result=decay_result,
            pruning_result=pruning_result,
            deduplication_result=dedup_result,
            total_execution_time_ms=350.0,
            maintenance_timestamp=datetime.now(),
            dry_run=False
        )
        assert report.decay_result == decay_result
        assert report.pruning_result == pruning_result
        assert report.deduplication_result == dedup_result
        assert report.total_execution_time_ms == 350.0
        assert report.dry_run is False
    
    def test_invalid_decay_result(self):
        """Test non-MemoryDecayResult decay_result raises ValueError."""
        with pytest.raises(ValueError, match="decay_result must be a MemoryDecayResult instance"):
            LifecycleReport(
                decay_result="invalid",  # String instead of MemoryDecayResult
                pruning_result=PruningResult(0, 0, [], 0.0),
                deduplication_result=DeduplicationResult(0, 0, [], None, 0.0),
                total_execution_time_ms=0.0,
                maintenance_timestamp=datetime.now(),
                dry_run=False
            )
    
    def test_invalid_pruning_result(self):
        """Test non-PruningResult pruning_result raises ValueError."""
        with pytest.raises(ValueError, match="pruning_result must be a PruningResult instance"):
            LifecycleReport(
                decay_result=MemoryDecayResult(0, 0, 0.0, 0.0),
                pruning_result="invalid",  # String instead of PruningResult
                deduplication_result=DeduplicationResult(0, 0, [], None, 0.0),
                total_execution_time_ms=0.0,
                maintenance_timestamp=datetime.now(),
                dry_run=False
            )
    
    def test_invalid_deduplication_result(self):
        """Test non-DeduplicationResult deduplication_result raises ValueError."""
        with pytest.raises(ValueError, match="deduplication_result must be a DeduplicationResult instance"):
            LifecycleReport(
                decay_result=MemoryDecayResult(0, 0, 0.0, 0.0),
                pruning_result=PruningResult(0, 0, [], 0.0),
                deduplication_result="invalid",  # String instead of DeduplicationResult
                total_execution_time_ms=0.0,
                maintenance_timestamp=datetime.now(),
                dry_run=False
            )
    
    def test_negative_total_execution_time(self):
        """Test negative total_execution_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="total_execution_time_ms must be non-negative"):
            LifecycleReport(
                decay_result=MemoryDecayResult(0, 0, 0.0, 0.0),
                pruning_result=PruningResult(0, 0, [], 0.0),
                deduplication_result=DeduplicationResult(0, 0, [], None, 0.0),
                total_execution_time_ms=-10.0,
                maintenance_timestamp=datetime.now(),
                dry_run=False
            )
    
    def test_invalid_maintenance_timestamp(self):
        """Test non-datetime maintenance_timestamp raises ValueError."""
        with pytest.raises(ValueError, match="maintenance_timestamp must be a datetime object"):
            LifecycleReport(
                decay_result=MemoryDecayResult(0, 0, 0.0, 0.0),
                pruning_result=PruningResult(0, 0, [], 0.0),
                deduplication_result=DeduplicationResult(0, 0, [], None, 0.0),
                total_execution_time_ms=0.0,
                maintenance_timestamp="2023-01-01",  # String instead of datetime
                dry_run=False
            )
    
    def test_invalid_dry_run(self):
        """Test non-boolean dry_run raises ValueError."""
        with pytest.raises(ValueError, match="dry_run must be a boolean"):
            LifecycleReport(
                decay_result=MemoryDecayResult(0, 0, 0.0, 0.0),
                pruning_result=PruningResult(0, 0, [], 0.0),
                deduplication_result=DeduplicationResult(0, 0, [], None, 0.0),
                total_execution_time_ms=0.0,
                maintenance_timestamp=datetime.now(),
                dry_run="false"  # String instead of boolean
            )


class TestDecayConfig:
    """Test DecayConfig validation."""
    
    def test_valid_exponential_decay_config(self):
        """Test creating valid exponential DecayConfig."""
        config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        assert config.decay_function_type == DecayFunctionType.EXPONENTIAL
        assert config.decay_rate == 0.1
        assert config.step_interval_days is None
        assert config.step_percentage is None
    
    def test_valid_linear_decay_config(self):
        """Test creating valid linear DecayConfig."""
        config = DecayConfig(
            decay_function_type=DecayFunctionType.LINEAR,
            decay_rate=0.05
        )
        assert config.decay_function_type == DecayFunctionType.LINEAR
        assert config.decay_rate == 0.05
    
    def test_valid_step_decay_config(self):
        """Test creating valid step DecayConfig."""
        config = DecayConfig(
            decay_function_type=DecayFunctionType.STEP,
            decay_rate=0.05,
            step_interval_days=7,
            step_percentage=0.1
        )
        assert config.decay_function_type == DecayFunctionType.STEP
        assert config.decay_rate == 0.05
        assert config.step_interval_days == 7
        assert config.step_percentage == 0.1
    
    def test_negative_decay_rate(self):
        """Test negative decay_rate raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=-0.1
            )
    
    def test_zero_decay_rate(self):
        """Test zero decay_rate raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.0
            )
    
    def test_step_decay_missing_interval(self):
        """Test STEP decay without step_interval_days raises ValueError."""
        with pytest.raises(ValueError, match="step_interval_days must be provided and positive for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.STEP,
                decay_rate=0.05,
                step_percentage=0.1
            )
    
    def test_step_decay_negative_interval(self):
        """Test STEP decay with negative step_interval_days raises ValueError."""
        with pytest.raises(ValueError, match="step_interval_days must be provided and positive for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.STEP,
                decay_rate=0.05,
                step_interval_days=-1,
                step_percentage=0.1
            )
    
    def test_step_decay_missing_percentage(self):
        """Test STEP decay without step_percentage raises ValueError."""
        with pytest.raises(ValueError, match="step_percentage must be provided and in range \\(0, 1\\) for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.STEP,
                decay_rate=0.05,
                step_interval_days=7
            )
    
    def test_step_decay_invalid_percentage(self):
        """Test STEP decay with invalid step_percentage raises ValueError."""
        with pytest.raises(ValueError, match="step_percentage must be provided and in range \\(0, 1\\) for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.STEP,
                decay_rate=0.05,
                step_interval_days=7,
                step_percentage=1.5  # > 1
            )
    
    def test_non_step_decay_with_interval(self):
        """Test non-STEP decay with step_interval_days raises ValueError."""
        with pytest.raises(ValueError, match="step_interval_days should only be provided for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=0.1,
                step_interval_days=7
            )
    
    def test_non_step_decay_with_percentage(self):
        """Test non-STEP decay with step_percentage raises ValueError."""
        with pytest.raises(ValueError, match="step_percentage should only be provided for STEP decay function"):
            DecayConfig(
                decay_function_type=DecayFunctionType.LINEAR,
                decay_rate=0.1,
                step_percentage=0.1
            )


class TestPruningConfig:
    """Test PruningConfig validation."""
    
    def test_valid_threshold_pruning_config(self):
        """Test creating valid threshold PruningConfig."""
        config = PruningConfig(
            strategy=PruningStrategy.THRESHOLD,
            threshold=0.3,
            min_importance_protected=0.8
        )
        assert config.strategy == PruningStrategy.THRESHOLD
        assert config.threshold == 0.3
        assert config.min_importance_protected == 0.8
        assert config.percentile is None
        assert config.capacity_limit is None
    
    def test_valid_percentile_pruning_config(self):
        """Test creating valid percentile PruningConfig."""
        config = PruningConfig(
            strategy=PruningStrategy.PERCENTILE,
            percentile=10.0,
            min_importance_protected=0.8
        )
        assert config.strategy == PruningStrategy.PERCENTILE
        assert config.percentile == 10.0
        assert config.min_importance_protected == 0.8
    
    def test_valid_capacity_pruning_config(self):
        """Test creating valid capacity PruningConfig."""
        config = PruningConfig(
            strategy=PruningStrategy.CAPACITY,
            capacity_limit=1000,
            min_importance_protected=0.8
        )
        assert config.strategy == PruningStrategy.CAPACITY
        assert config.capacity_limit == 1000
        assert config.min_importance_protected == 0.8
    
    def test_invalid_min_importance_protected_below_range(self):
        """Test min_importance_protected < 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_importance_protected must be in \\[0, 1\\]"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=-0.1
            )
    
    def test_invalid_min_importance_protected_above_range(self):
        """Test min_importance_protected > 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_importance_protected must be in \\[0, 1\\]"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                min_importance_protected=1.5
            )
    
    def test_threshold_strategy_missing_threshold(self):
        """Test THRESHOLD strategy without threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be provided and in \\[0, 1\\] for THRESHOLD strategy"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                min_importance_protected=0.8
            )
    
    def test_threshold_strategy_invalid_threshold(self):
        """Test THRESHOLD strategy with invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be provided and in \\[0, 1\\] for THRESHOLD strategy"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=1.5,  # > 1
                min_importance_protected=0.8
            )
    
    def test_threshold_strategy_with_percentile(self):
        """Test THRESHOLD strategy with percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile should not be provided for THRESHOLD strategy"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.3,
                percentile=10.0,
                min_importance_protected=0.8
            )
    
    def test_percentile_strategy_missing_percentile(self):
        """Test PERCENTILE strategy without percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be provided and in \\(0, 100\\) for PERCENTILE strategy"):
            PruningConfig(
                strategy=PruningStrategy.PERCENTILE,
                min_importance_protected=0.8
            )
    
    def test_percentile_strategy_invalid_percentile(self):
        """Test PERCENTILE strategy with invalid percentile raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be provided and in \\(0, 100\\) for PERCENTILE strategy"):
            PruningConfig(
                strategy=PruningStrategy.PERCENTILE,
                percentile=150.0,  # > 100
                min_importance_protected=0.8
            )
    
    def test_capacity_strategy_missing_capacity(self):
        """Test CAPACITY strategy without capacity_limit raises ValueError."""
        with pytest.raises(ValueError, match="capacity_limit must be provided and positive for CAPACITY strategy"):
            PruningConfig(
                strategy=PruningStrategy.CAPACITY,
                min_importance_protected=0.8
            )
    
    def test_capacity_strategy_invalid_capacity(self):
        """Test CAPACITY strategy with invalid capacity_limit raises ValueError."""
        with pytest.raises(ValueError, match="capacity_limit must be provided and positive for CAPACITY strategy"):
            PruningConfig(
                strategy=PruningStrategy.CAPACITY,
                capacity_limit=0,  # Not positive
                min_importance_protected=0.8
            )


class TestDeduplicationConfig:
    """Test DeduplicationConfig validation."""
    
    def test_valid_deduplication_config(self):
        """Test creating valid DeduplicationConfig."""
        config = DeduplicationConfig(
            similarity_metric=SimilarityMetric.COSINE,
            similarity_threshold=0.9,
            batch_size=1000,
            checkpoint_enabled=True
        )
        assert config.similarity_metric == SimilarityMetric.COSINE
        assert config.similarity_threshold == 0.9
        assert config.batch_size == 1000
        assert config.checkpoint_enabled is True
    
    def test_similarity_threshold_below_range(self):
        """Test similarity_threshold < 0 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be in \\[0, 1\\]"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=-0.1,
                batch_size=1000
            )
    
    def test_similarity_threshold_above_range(self):
        """Test similarity_threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be in \\[0, 1\\]"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=1.5,
                batch_size=1000
            )
    
    def test_negative_batch_size(self):
        """Test negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=-1
            )
    
    def test_zero_batch_size(self):
        """Test zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=0
            )
    
    def test_invalid_checkpoint_enabled(self):
        """Test non-boolean checkpoint_enabled raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_enabled must be a boolean"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=1000,
                checkpoint_enabled="true"  # String instead of boolean
            )