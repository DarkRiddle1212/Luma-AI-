"""
Property test for schema validation (Property 29).

Feature: memory-lifecycle-management, Property 29: Schema Validation

For any schema instance (MemoryDecayResult, PruningResult, DeduplicationResult, 
LifecycleReport) with invalid values (negative counts, scores outside [0,1]), 
validation should fail.

Validates: Requirements 5.6
"""

import pytest
from hypothesis import given, strategies as st
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


class TestProperty29SchemaValidation:
    """
    Feature: memory-lifecycle-management, Property 29: Schema Validation
    
    For any schema instance with invalid values (negative counts, scores outside [0,1]), 
    validation should fail.
    """
    
    @given(
        memories_processed=st.integers(max_value=-1),  # Negative values
        memories_updated=st.integers(min_value=0, max_value=1000),
        average_decay_applied=st.floats(min_value=0.0, max_value=1.0),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_memory_decay_result_negative_memories_processed(
        self, memories_processed, memories_updated, average_decay_applied, execution_time_ms
    ):
        """Test MemoryDecayResult with negative memories_processed fails validation."""
        with pytest.raises(ValueError, match="memories_processed must be non-negative"):
            MemoryDecayResult(
                memories_processed=memories_processed,
                memories_updated=memories_updated,
                average_decay_applied=average_decay_applied,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        memories_processed=st.integers(min_value=0, max_value=1000),
        memories_updated=st.integers(max_value=-1),  # Negative values
        average_decay_applied=st.floats(min_value=0.0, max_value=1.0),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_memory_decay_result_negative_memories_updated(
        self, memories_processed, memories_updated, average_decay_applied, execution_time_ms
    ):
        """Test MemoryDecayResult with negative memories_updated fails validation."""
        with pytest.raises(ValueError, match="memories_updated must be non-negative"):
            MemoryDecayResult(
                memories_processed=memories_processed,
                memories_updated=memories_updated,
                average_decay_applied=average_decay_applied,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        memories_processed=st.integers(min_value=1, max_value=1000),
        average_decay_applied=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        ),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_memory_decay_result_invalid_average_decay(
        self, memories_processed, average_decay_applied, execution_time_ms
    ):
        """Test MemoryDecayResult with average_decay_applied outside [0,1] fails validation."""
        # Ensure memories_updated <= memories_processed to avoid that validation error
        memories_updated = min(memories_processed, memories_processed // 2)
        
        with pytest.raises(ValueError, match="average_decay_applied must be in \\[0, 1\\]"):
            MemoryDecayResult(
                memories_processed=memories_processed,
                memories_updated=memories_updated,
                average_decay_applied=average_decay_applied,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        memories_processed=st.integers(min_value=1, max_value=1000),
        average_decay_applied=st.floats(min_value=0.0, max_value=1.0),
        execution_time_ms=st.floats(max_value=-0.001)  # Negative values
    )
    def test_memory_decay_result_negative_execution_time(
        self, memories_processed, average_decay_applied, execution_time_ms
    ):
        """Test MemoryDecayResult with negative execution_time_ms fails validation."""
        # Ensure memories_updated <= memories_processed to avoid that validation error
        memories_updated = min(memories_processed, memories_processed // 2)
        
        with pytest.raises(ValueError, match="execution_time_ms must be non-negative"):
            MemoryDecayResult(
                memories_processed=memories_processed,
                memories_updated=memories_updated,
                average_decay_applied=average_decay_applied,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        memory_id=st.text(min_size=1, max_size=50),
        importance_score=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        ),
        final_score=st.floats(min_value=0.0, max_value=1.0),
        reason=st.text(min_size=1, max_size=20)
    )
    def test_pruned_memory_invalid_importance_score(
        self, memory_id, importance_score, final_score, reason
    ):
        """Test PrunedMemory with importance_score outside [0,1] fails validation."""
        with pytest.raises(ValueError, match="importance_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id=memory_id,
                importance_score=importance_score,
                final_score=final_score,
                deletion_timestamp=datetime.now(),
                reason=reason
            )
    
    @given(
        memory_id=st.text(min_size=1, max_size=50),
        importance_score=st.floats(min_value=0.0, max_value=1.0),
        final_score=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        ),
        reason=st.text(min_size=1, max_size=20)
    )
    def test_pruned_memory_invalid_final_score(
        self, memory_id, importance_score, final_score, reason
    ):
        """Test PrunedMemory with final_score outside [0,1] fails validation."""
        with pytest.raises(ValueError, match="final_score must be in \\[0, 1\\]"):
            PrunedMemory(
                memory_id=memory_id,
                importance_score=importance_score,
                final_score=final_score,
                deletion_timestamp=datetime.now(),
                reason=reason
            )
    
    @given(
        memories_deleted=st.integers(max_value=-1),  # Negative values
        deletion_failures=st.integers(min_value=0, max_value=100),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_pruning_result_negative_memories_deleted(
        self, memories_deleted, deletion_failures, execution_time_ms
    ):
        """Test PruningResult with negative memories_deleted fails validation."""
        with pytest.raises(ValueError, match="memories_deleted must be non-negative"):
            PruningResult(
                memories_deleted=memories_deleted,
                deletion_failures=deletion_failures,
                pruned_memories=[],
                execution_time_ms=execution_time_ms
            )
    
    @given(
        memories_deleted=st.integers(min_value=0, max_value=100),
        deletion_failures=st.integers(max_value=-1),  # Negative values
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_pruning_result_negative_deletion_failures(
        self, memories_deleted, deletion_failures, execution_time_ms
    ):
        """Test PruningResult with negative deletion_failures fails validation."""
        with pytest.raises(ValueError, match="deletion_failures must be non-negative"):
            PruningResult(
                memories_deleted=memories_deleted,
                deletion_failures=deletion_failures,
                pruned_memories=[],
                execution_time_ms=execution_time_ms
            )
    
    @given(
        kept_memory_id=st.text(min_size=1, max_size=50),
        deleted_memory_id=st.text(min_size=1, max_size=50),
        similarity_score=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        ),
        merged_tags=st.lists(st.text(min_size=1, max_size=10), max_size=5)
    )
    def test_merge_detail_invalid_similarity_score(
        self, kept_memory_id, deleted_memory_id, similarity_score, merged_tags
    ):
        """Test MergeDetail with similarity_score outside [0,1] fails validation."""
        # Ensure different memory IDs
        if kept_memory_id == deleted_memory_id:
            deleted_memory_id = kept_memory_id + "_different"
        
        with pytest.raises(ValueError, match="similarity_score must be in \\[0, 1\\]"):
            MergeDetail(
                kept_memory_id=kept_memory_id,
                deleted_memory_id=deleted_memory_id,
                similarity_score=similarity_score,
                merged_tags=merged_tags,
                merge_timestamp=datetime.now()
            )
    
    @given(
        duplicate_pairs_found=st.integers(max_value=-1),  # Negative values
        memories_merged=st.integers(min_value=0, max_value=100),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_deduplication_result_negative_duplicate_pairs(
        self, duplicate_pairs_found, memories_merged, execution_time_ms
    ):
        """Test DeduplicationResult with negative duplicate_pairs_found fails validation."""
        with pytest.raises(ValueError, match="duplicate_pairs_found must be non-negative"):
            DeduplicationResult(
                duplicate_pairs_found=duplicate_pairs_found,
                memories_merged=memories_merged,
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        duplicate_pairs_found=st.integers(min_value=0, max_value=100),
        memories_merged=st.integers(max_value=-1),  # Negative values
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0)
    )
    def test_deduplication_result_negative_memories_merged(
        self, duplicate_pairs_found, memories_merged, execution_time_ms
    ):
        """Test DeduplicationResult with negative memories_merged fails validation."""
        with pytest.raises(ValueError, match="memories_merged must be non-negative"):
            DeduplicationResult(
                duplicate_pairs_found=duplicate_pairs_found,
                memories_merged=memories_merged,
                merge_details=[],
                checkpoint_timestamp=None,
                execution_time_ms=execution_time_ms
            )
    
    @given(
        total_execution_time_ms=st.floats(max_value=-0.001)  # Negative values
    )
    def test_lifecycle_report_negative_total_execution_time(self, total_execution_time_ms):
        """Test LifecycleReport with negative total_execution_time_ms fails validation."""
        # Create valid sub-results
        decay_result = MemoryDecayResult(0, 0, 0.0, 0.0)
        pruning_result = PruningResult(0, 0, [], 0.0)
        dedup_result = DeduplicationResult(0, 0, [], None, 0.0)
        
        with pytest.raises(ValueError, match="total_execution_time_ms must be non-negative"):
            LifecycleReport(
                decay_result=decay_result,
                pruning_result=pruning_result,
                deduplication_result=dedup_result,
                total_execution_time_ms=total_execution_time_ms,
                maintenance_timestamp=datetime.now(),
                dry_run=False
            )
    
    @given(
        decay_rate=st.floats(max_value=0.0)  # Non-positive values
    )
    def test_decay_config_invalid_decay_rate(self, decay_rate):
        """Test DecayConfig with non-positive decay_rate fails validation."""
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            DecayConfig(
                decay_function_type=DecayFunctionType.EXPONENTIAL,
                decay_rate=decay_rate
            )
    
    @given(
        min_importance_protected=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        )
    )
    def test_pruning_config_invalid_min_importance_protected(self, min_importance_protected):
        """Test PruningConfig with min_importance_protected outside [0,1] fails validation."""
        with pytest.raises(ValueError, match="min_importance_protected must be in \\[0, 1\\]"):
            PruningConfig(
                strategy=PruningStrategy.THRESHOLD,
                threshold=0.5,
                min_importance_protected=min_importance_protected
            )
    
    @given(
        similarity_threshold=st.one_of(
            st.floats(max_value=-0.001),  # Below 0
            st.floats(min_value=1.001)    # Above 1
        ),
        batch_size=st.integers(min_value=1, max_value=1000)
    )
    def test_deduplication_config_invalid_similarity_threshold(self, similarity_threshold, batch_size):
        """Test DeduplicationConfig with similarity_threshold outside [0,1] fails validation."""
        with pytest.raises(ValueError, match="similarity_threshold must be in \\[0, 1\\]"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=similarity_threshold,
                batch_size=batch_size
            )
    
    @given(
        batch_size=st.integers(max_value=0)  # Non-positive values
    )
    def test_deduplication_config_invalid_batch_size(self, batch_size):
        """Test DeduplicationConfig with non-positive batch_size fails validation."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DeduplicationConfig(
                similarity_metric=SimilarityMetric.COSINE,
                similarity_threshold=0.9,
                batch_size=batch_size
            )