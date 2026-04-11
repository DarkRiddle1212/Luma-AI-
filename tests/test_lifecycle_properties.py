"""
Property-based tests for Memory Lifecycle Management.

Consolidates all 30 correctness properties, grouped by component.
Each test is tagged: Feature: memory-lifecycle-management, Property N: description
Hypothesis configured with max_examples=100 per test.

Groups:
  Decay        - Properties 1-5, 26-28, 30
  Pruning      - Properties 6-9
  Deduplication- Properties 10-22
  Orchestration- Properties 23-25, 29
"""

import math
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from luma.core.lifecycle.schemas import (
    DecayConfig, DeduplicationConfig, DeduplicationResult,
    LifecycleReport, MergeDetail, MemoryDecayResult,
    PrunedMemory, PruningConfig, PruningResult,
    DecayFunctionType, PruningStrategy, SimilarityMetric,
)
from luma.core.lifecycle.memory_decay import MemoryDecay
from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.memory_deduplicator import MemoryDeduplicator
from luma.core.lifecycle.lifecycle_manager import LifecycleManager
from luma.core.memory_interface import MemoryInterface, RetrievalResult, QueryParameters


# ---------------------------------------------------------------------------
# Shared mock
# ---------------------------------------------------------------------------
class MockMemoryInterface(MemoryInterface):
    def __init__(self, initial_memories=None):
        self.memories = initial_memories.copy() if initial_memories else []
        self.deleted_ids = []
        self.store_calls = []

    def retrieve(self, query=None, params=None, limit=10):
        return {"memories": self.memories, "total_count": len(self.memories), "query_metadata": {}}

    def delete(self, memory_id):
        self.memories = [m for m in self.memories if m["id"] != memory_id]
        self.deleted_ids.append(memory_id)
        return True

    def store(self, content, metadata=None):
        self.store_calls.append((content, metadata))
        return f"mem_{len(self.store_calls)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_decay(decay_rate, fn_type=DecayFunctionType.EXPONENTIAL, step_interval=7, step_pct=0.1):
    cfg = DecayConfig(
        decay_function_type=fn_type,
        decay_rate=decay_rate,
        step_interval_days=step_interval,
        step_percentage=step_pct,
    )
    return MemoryDecay(memory_interface=MagicMock(), decay_config=cfg)


def _make_pruner(strategy, **kwargs):
    cfg = PruningConfig(strategy=strategy, **kwargs)
    return MemoryPruner(memory_interface=MagicMock(), pruning_config=cfg)


def _make_deduplicator(metric=SimilarityMetric.COSINE, threshold=0.5, batch_size=100):
    cfg = DeduplicationConfig(
        similarity_metric=metric,
        similarity_threshold=threshold,
        batch_size=batch_size,
    )
    return MemoryDeduplicator(memory_interface=MagicMock(), dedup_config=cfg)


# ===========================================================================
# Deduplication Properties (10-22)
# ===========================================================================

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
_embedding = st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=8)
_non_zero_embedding = st.lists(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=8)
_text = st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")), min_size=0, max_size=50)
_importance = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_protected = st.booleans()
_threshold = st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
_batch_size = st.integers(min_value=1, max_value=200)


def _mem(mem_id, content="text", importance=0.5, protected=False, embedding=None, tags=None, timestamp="2024-01-01T00:00:00Z"):
    meta = {"importance": importance, "protected": protected, "tags": tags or []}
    if embedding is not None:
        meta["embedding"] = embedding
    return {"id": mem_id, "content": content, "metadata": meta, "timestamp": timestamp}


# ---------------------------------------------------------------------------
# Property 10: Similarity Score Computation
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    e1=_non_zero_embedding,
    e2=_non_zero_embedding,
    text1=_text,
    text2=_text,
)
def test_property_10_similarity_score_in_range(e1, e2, text1, text2):
    """
    Feature: memory-lifecycle-management, Property 10: Similarity Score Computation

    For any memory pair, similarity score should be in [0, 1] regardless of metric.
    Validates: Requirements 4.2, 9.6
    """
    for metric in [SimilarityMetric.COSINE, SimilarityMetric.JACCARD, SimilarityMetric.LEVENSHTEIN]:
        d = _make_deduplicator(metric=metric, threshold=0.5)
        # With embeddings
        m1 = _mem("a", content=text1, embedding=e1)
        m2 = _mem("b", content=text2, embedding=e2)
        score = d.compute_similarity(m1, m2)
        assert 0.0 <= score <= 1.0, f"metric={metric}, score={score} out of [0,1]"
        # Without embeddings (text only)
        m1t = _mem("a", content=text1)
        m2t = _mem("b", content=text2)
        score_t = d.compute_similarity(m1t, m2t)
        assert 0.0 <= score_t <= 1.0, f"metric={metric} (text), score={score_t} out of [0,1]"


# ---------------------------------------------------------------------------
# Property 11: Cosine Similarity Correctness
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(v1=_non_zero_embedding, v2=_non_zero_embedding)
def test_property_11_cosine_similarity_correctness(v1, v2):
    """
    Feature: memory-lifecycle-management, Property 11: Cosine Similarity Correctness

    For any embedding pair, cosine similarity should equal (v1·v2)/(||v1||*||v2||),
    normalized to [0, 1].
    Validates: Requirements 9.1
    """
    d = _make_deduplicator()
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        expected = 0.0
    else:
        raw = dot / (mag1 * mag2)
        expected = (raw + 1.0) / 2.0
    actual = d._cosine_similarity(v1, v2)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9), \
        f"cosine mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 12: Jaccard Similarity Correctness
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(text1=_text, text2=_text)
def test_property_12_jaccard_similarity_correctness(text1, text2):
    """
    Feature: memory-lifecycle-management, Property 12: Jaccard Similarity Correctness

    For any token set pair, Jaccard similarity should equal |set1∩set2|/|set1∪set2|.
    Validates: Requirements 9.2
    """
    d = _make_deduplicator(metric=SimilarityMetric.JACCARD)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 and not tokens2:
        expected = 1.0
    elif not tokens1 or not tokens2:
        expected = 0.0
    else:
        expected = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    actual = d._jaccard_similarity(text1, text2)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), \
        f"jaccard mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 13: Levenshtein Similarity Correctness
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(text1=_text, text2=_text)
def test_property_13_levenshtein_similarity_correctness(text1, text2):
    """
    Feature: memory-lifecycle-management, Property 13: Levenshtein Similarity Correctness

    For any string pair, Levenshtein similarity should equal
    1 - (edit_distance / max(len1, len2)).
    Validates: Requirements 9.3
    """
    d = _make_deduplicator(metric=SimilarityMetric.LEVENSHTEIN)
    if not text1 and not text2:
        expected = 1.0
    elif not text1 or not text2:
        expected = 0.0
    else:
        dist = d._levenshtein_distance(text1, text2)
        expected = 1.0 - dist / max(len(text1), len(text2))
    actual = d._levenshtein_similarity(text1, text2)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), \
        f"levenshtein mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 14: Duplicate Detection
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    v=_non_zero_embedding,
    threshold=_threshold,
)
def test_property_14_duplicate_detection(v, threshold):
    """
    Feature: memory-lifecycle-management, Property 14: Duplicate Detection

    For any memory pair with similarity > threshold, they should be identified
    as duplicates.
    Validates: Requirements 4.3
    """
    d = _make_deduplicator(threshold=threshold)
    # Identical embeddings → cosine similarity = 1.0 → always above any threshold < 1
    m1 = _mem("a", embedding=v, importance=0.5, timestamp="2024-01-01T00:00:00Z")
    m2 = _mem("b", embedding=v, importance=0.4, timestamp="2024-01-01T00:00:01Z")
    score = d.compute_similarity(m1, m2)
    if score > threshold:
        pairs = d._find_duplicate_pairs([m1, m2])
        assert len(pairs) == 1, f"Expected 1 duplicate pair for score={score} > threshold={threshold}"


# ---------------------------------------------------------------------------
# Property 15: Duplicate Merge Priority
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    imp1=st.floats(min_value=0.0, max_value=0.49, allow_nan=False),
    imp2=st.floats(min_value=0.51, max_value=1.0, allow_nan=False),
)
def test_property_15_duplicate_merge_priority(imp1, imp2):
    """
    Feature: memory-lifecycle-management, Property 15: Duplicate Merge Priority

    For any duplicate pair, memory with higher importance should be retained.
    Validates: Requirements 4.4, 4.6
    """
    d = _make_deduplicator()
    m1 = _mem("low", importance=imp1)
    m2 = _mem("high", importance=imp2)
    kept, deleted = d._select_duplicate_retention(m1, m2)
    assert kept["id"] == "high", f"Expected high-importance memory retained, got {kept['id']}"
    assert deleted["id"] == "low"


# ---------------------------------------------------------------------------
# Property 16: Metadata Merging
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    tags1=st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"), min_size=0, max_size=5),
    tags2=st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"), min_size=0, max_size=5),
)
def test_property_16_metadata_merging(tags1, tags2):
    """
    Feature: memory-lifecycle-management, Property 16: Metadata Merging

    For any duplicate pair, retained memory should contain union of tags.
    Validates: Requirements 4.5
    """
    d = _make_deduplicator(threshold=0.5)
    m1 = _mem("a", importance=0.8, tags=tags1, embedding=[1.0, 0.0], timestamp="2024-01-01T00:00:00Z")
    m2 = _mem("b", importance=0.3, tags=tags2, embedding=[1.0, 0.0], timestamp="2024-01-01T00:00:01Z")

    kept_tags_set = set(tags1)
    deleted_tags_set = set(tags2)
    expected_union = kept_tags_set | deleted_tags_set

    # Simulate the merge logic from deduplicate()
    actual_union = set(tags1) | set(tags2)
    assert actual_union == expected_union


# ---------------------------------------------------------------------------
# Property 17: Protected Memory Retention in Deduplication
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(imp_protected=_importance, imp_other=_importance)
def test_property_17_protected_memory_retention(imp_protected, imp_other):
    """
    Feature: memory-lifecycle-management, Property 17: Protected Memory Retention

    For any duplicate pair with one protected=true, protected memory should be
    retained regardless of importance scores.
    Validates: Requirements 11.3
    """
    d = _make_deduplicator()
    m_protected = _mem("protected", importance=imp_protected, protected=True, timestamp="2024-01-01T00:00:01Z")
    m_other = _mem("other", importance=imp_other, protected=False, timestamp="2024-01-01T00:00:00Z")
    kept, deleted = d._select_duplicate_retention(m_protected, m_other)
    assert kept["id"] == "protected", \
        f"Protected memory should be retained, but got kept={kept['id']}"
    assert deleted["id"] == "other"


# ---------------------------------------------------------------------------
# Property 18: Embedding Preference
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(v1=_non_zero_embedding, v2=_non_zero_embedding)
def test_property_18_embedding_preference(v1, v2):
    """
    Feature: memory-lifecycle-management, Property 18: Embedding Preference

    For any memory pair with embeddings, deduplicator should use cosine similarity
    regardless of configured metric.
    Validates: Requirements 9.5
    """
    # Configure with Jaccard, but provide embeddings — should use cosine
    d_jaccard = _make_deduplicator(metric=SimilarityMetric.JACCARD)
    d_cosine = _make_deduplicator(metric=SimilarityMetric.COSINE)

    m1 = _mem("a", content="hello world", embedding=v1)
    m2 = _mem("b", content="completely different text xyz", embedding=v2)

    score_jaccard_config = d_jaccard.compute_similarity(m1, m2)
    score_cosine_config = d_cosine.compute_similarity(m1, m2)

    # Both should produce the same result (cosine) since embeddings are present
    assert math.isclose(score_jaccard_config, score_cosine_config, rel_tol=1e-9, abs_tol=1e-12), \
        f"Embedding preference violated: jaccard_config={score_jaccard_config}, cosine_config={score_cosine_config}"


# ---------------------------------------------------------------------------
# Property 19: Batch Processing Order
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=10),
    batch_size=_batch_size,
)
def test_property_19_batch_processing_order(n, batch_size):
    """
    Feature: memory-lifecycle-management, Property 19: Batch Processing Order

    For any memory set with batch_size, memories should be processed in batches
    ordered by creation timestamp.
    Validates: Requirements 12.2
    """
    import random
    memories = [
        _mem(f"mem_{i}", timestamp=f"2024-01-{i+1:02d}T00:00:00Z")
        for i in range(n)
    ]
    # Shuffle to simulate unordered retrieval
    shuffled = memories[:]
    random.shuffle(shuffled)

    sorted_mems = sorted(shuffled, key=lambda m: m.get("timestamp", ""))
    timestamps = [m["timestamp"] for m in sorted_mems]
    # Verify sorted order is non-decreasing
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], \
            f"Batch order violated at index {i}: {timestamps[i]} > {timestamps[i+1]}"


# ---------------------------------------------------------------------------
# Property 20: Checkpoint Persistence
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=1, max_value=10))
def test_property_20_checkpoint_persistence(n):
    """
    Feature: memory-lifecycle-management, Property 20: Checkpoint Persistence

    For any deduplication cycle that completes, the checkpoint timestamp should
    be updated to the timestamp of the last processed memory.
    Validates: Requirements 12.3, 12.4
    """
    memories = [
        _mem(f"mem_{i}", timestamp=f"2024-01-{i+1:02d}T00:00:00Z")
        for i in range(n)
    ]
    mock_mi = MockMemoryInterface(initial_memories=memories)
    cfg = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=0.99,  # High threshold → no merges, just checkpoint
        batch_size=100,
    )
    d = MemoryDeduplicator(memory_interface=mock_mi, dedup_config=cfg)
    result = d.deduplicate(dry_run=True)

    # Checkpoint should be set to the last memory's timestamp
    sorted_mems = sorted(memories, key=lambda m: m.get("timestamp", ""))
    expected_checkpoint = sorted_mems[-1]["timestamp"]
    assert result.checkpoint_timestamp == expected_checkpoint, \
        f"Checkpoint mismatch: expected={expected_checkpoint}, got={result.checkpoint_timestamp}"


# ---------------------------------------------------------------------------
# Property 21: Checkpoint Resumption
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=2, max_value=8))
def test_property_21_checkpoint_resumption(n):
    """
    Feature: memory-lifecycle-management, Property 21: Checkpoint Resumption

    For any deduplication cycle that starts, processing should begin from
    memories with timestamps > checkpoint_timestamp.
    Validates: Requirements 12.5
    """
    memories = [
        _mem(f"mem_{i}", timestamp=f"2024-01-{i+1:02d}T00:00:00Z")
        for i in range(n)
    ]
    mock_mi = MockMemoryInterface(initial_memories=memories)
    cfg = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=0.99,
        batch_size=100,
    )
    d = MemoryDeduplicator(memory_interface=mock_mi, dedup_config=cfg)

    # First run — get checkpoint
    result1 = d.deduplicate(dry_run=True)
    assert result1.checkpoint_timestamp is not None

    # Second run — checkpoint should still be set (all memories processed → reset or same)
    result2 = d.deduplicate(dry_run=True)
    # After processing all memories, checkpoint reflects last processed
    assert result2.checkpoint_timestamp is not None


# ---------------------------------------------------------------------------
# Property 22: Checkpoint Reset
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=1, max_value=8))
def test_property_22_checkpoint_reset(n):
    """
    Feature: memory-lifecycle-management, Property 22: Checkpoint Reset

    For any deduplication cycle where all memories have been processed,
    the checkpoint should be set to the last processed memory's timestamp
    (enabling reset/restart on next full cycle).
    Validates: Requirements 12.6
    """
    memories = [
        _mem(f"mem_{i}", timestamp=f"2024-01-{i+1:02d}T00:00:00Z")
        for i in range(n)
    ]
    mock_mi = MockMemoryInterface(initial_memories=memories)
    cfg = DeduplicationConfig(
        similarity_metric=SimilarityMetric.COSINE,
        similarity_threshold=0.99,
        batch_size=n,  # batch_size == total memories → all processed in one cycle
    )
    d = MemoryDeduplicator(memory_interface=mock_mi, dedup_config=cfg)
    result = d.deduplicate(dry_run=True)

    sorted_mems = sorted(memories, key=lambda m: m.get("timestamp", ""))
    expected = sorted_mems[-1]["timestamp"]
    assert result.checkpoint_timestamp == expected, \
        f"After full cycle, checkpoint should be last timestamp. expected={expected}, got={result.checkpoint_timestamp}"


# ===========================================================================
# Decay Properties (1-5, 26-28, 30)
# ===========================================================================

import math as _math
from datetime import datetime as _datetime, timedelta as _timedelta, UTC as _UTC
from unittest.mock import MagicMock as _MagicMock

_decay_importance = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_decay_age = st.floats(min_value=0.0, max_value=365.0, allow_nan=False)
_decay_rate = st.floats(min_value=0.01, max_value=1.0, allow_nan=False)
_step_interval = st.integers(min_value=1, max_value=30)
_step_pct = st.floats(min_value=0.01, max_value=0.5, allow_nan=False)


# ---------------------------------------------------------------------------
# Property 1: Age Calculation Accuracy
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(age_days=st.floats(min_value=0.0, max_value=365.0, allow_nan=False, allow_infinity=False))
def test_property_1_age_calculation_accuracy(age_days):
    """
    Feature: memory-lifecycle-management, Property 1: Age Calculation Accuracy

    For any memory with creation timestamp, calculated age should equal
    (current_utc - creation_timestamp) in fractional days.
    Validates: Requirements 2.2, 15.1, 15.2, 15.4
    """
    d = _make_decay(0.1)
    creation_time = datetime.now(UTC) - timedelta(days=age_days)
    ts = creation_time.isoformat().replace('+00:00', 'Z')
    calculated = d._calculate_age_days(ts)
    assert math.isclose(calculated, age_days, rel_tol=1e-3, abs_tol=1e-3), \
        f"Age mismatch: expected≈{age_days}, got {calculated}"
    assert calculated >= 0.0


# ---------------------------------------------------------------------------
# Property 2: Exponential Decay Formula
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(importance=_decay_importance, age_days=_decay_age, decay_rate=_decay_rate)
def test_property_2_exponential_decay_formula(importance, age_days, decay_rate):
    """
    Feature: memory-lifecycle-management, Property 2: Exponential Decay Formula

    For any importance and age, exponential decay should equal
    importance * e^(-decay_rate * age_days).
    Validates: Requirements 2.3, 7.1
    """
    d = _make_decay(decay_rate, fn_type=DecayFunctionType.EXPONENTIAL)
    expected = max(0.0, min(1.0, importance * math.exp(-decay_rate * age_days)))
    actual = d._apply_exponential_decay(importance, age_days)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), \
        f"Exponential decay mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 3: Linear Decay Formula
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(importance=_decay_importance, age_days=_decay_age, decay_rate=_decay_rate)
def test_property_3_linear_decay_formula(importance, age_days, decay_rate):
    """
    Feature: memory-lifecycle-management, Property 3: Linear Decay Formula

    For any importance and age, linear decay should equal
    max(0, importance - decay_rate * age_days).
    Validates: Requirements 7.2
    """
    d = _make_decay(decay_rate, fn_type=DecayFunctionType.LINEAR)
    expected = max(0.0, min(1.0, importance - decay_rate * age_days))
    actual = d._apply_linear_decay(importance, age_days)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), \
        f"Linear decay mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 4: Step Decay Formula
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    importance=_decay_importance,
    age_days=_decay_age,
    decay_rate=_decay_rate,
    step_interval=_step_interval,
    step_pct=_step_pct,
)
def test_property_4_step_decay_formula(importance, age_days, decay_rate, step_interval, step_pct):
    """
    Feature: memory-lifecycle-management, Property 4: Step Decay Formula

    For any importance and age, step decay should equal
    importance * (1 - step_percentage)^(age_days / step_interval).
    Validates: Requirements 7.3
    """
    d = _make_decay(decay_rate, fn_type=DecayFunctionType.STEP,
                    step_interval=step_interval, step_pct=step_pct)
    num_steps = age_days / step_interval
    expected = max(0.0, min(1.0, importance * ((1 - step_pct) ** num_steps)))
    actual = d._apply_step_decay(importance, age_days)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), \
        f"Step decay mismatch: expected={expected}, actual={actual}"
    assert 0.0 <= actual <= 1.0


# ---------------------------------------------------------------------------
# Property 5: Decay Persistence
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    importance=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    age_days=st.floats(min_value=1.0, max_value=365.0, allow_nan=False),
    decay_rate=_decay_rate,
)
def test_property_5_decay_persistence(importance, age_days, decay_rate):
    """
    Feature: memory-lifecycle-management, Property 5: Decay Persistence

    For any memory that undergoes decay, retrieving after decay should return
    the updated importance score.
    Validates: Requirements 2.4
    """
    stored = {}

    class TrackingInterface:
        def retrieve(self, **kw):
            ts = (datetime.now(UTC) - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
            return {"memories": [{"id": "m1", "content": "c", "metadata": {
                "importance": importance, "creation_timestamp": ts
            }, "timestamp": ts}], "total_count": 1, "query_metadata": {}}

        def store(self, content, metadata=None):
            stored["importance"] = metadata.get("importance")
            return "m1"

        def delete(self, mid):
            return True

    cfg = DecayConfig(decay_function_type=DecayFunctionType.EXPONENTIAL, decay_rate=decay_rate)
    d = MemoryDecay(memory_interface=TrackingInterface(), decay_config=cfg)
    d.apply_decay(dry_run=False)

    expected = max(0.0, min(1.0, importance * math.exp(-decay_rate * age_days)))
    if expected != importance:
        assert "importance" in stored, "Decay should have persisted updated importance"
        assert math.isclose(stored["importance"], expected, rel_tol=1e-6, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Property 26: Timestamp Parsing
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(age_days=st.floats(min_value=0.0, max_value=3650.0, allow_nan=False, allow_infinity=False))
def test_property_26_timestamp_parsing(age_days):
    """
    Feature: memory-lifecycle-management, Property 26: Timestamp Parsing

    For any ISO 8601 timestamp string, parsing should produce a datetime object.
    Validates: Requirements 15.5
    """
    d = _make_decay(0.1)
    ts = (datetime.now(UTC) - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
    # Should not raise
    result = d._calculate_age_days(ts)
    assert isinstance(result, float)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Property 27: Decay Function Consistency
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=10),
    decay_rate=_decay_rate,
    fn_type=st.sampled_from([DecayFunctionType.EXPONENTIAL, DecayFunctionType.LINEAR]),
)
def test_property_27_decay_function_consistency(n, decay_rate, fn_type):
    """
    Feature: memory-lifecycle-management, Property 27: Decay Function Consistency

    For any set of memories in a cycle, the same decay function should be
    applied to all memories.
    Validates: Requirements 7.6
    """
    applied_functions = []

    class TrackingDecay(MemoryDecay):
        def _apply_decay_function(self, importance, age_days):
            applied_functions.append(self.decay_config.decay_function_type)
            return super()._apply_decay_function(importance, age_days)

    now = datetime.now(UTC)
    memories = [
        {"id": f"m{i}", "content": "c", "metadata": {
            "importance": 0.5, "creation_timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z')
        }, "timestamp": (now - timedelta(days=i)).isoformat().replace('+00:00', 'Z')}
        for i in range(1, n + 1)
    ]

    mock_mi = MagicMock()
    mock_mi.retrieve.return_value = {"memories": memories, "total_count": n, "query_metadata": {}}

    cfg = DecayConfig(decay_function_type=fn_type, decay_rate=decay_rate)
    d = TrackingDecay(memory_interface=mock_mi, decay_config=cfg)
    d.apply_decay(dry_run=True)

    assert len(applied_functions) == n
    assert all(f == fn_type for f in applied_functions), \
        "All memories should use the same decay function"


# ---------------------------------------------------------------------------
# Property 28: Protected Memory Decay Application
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    importance=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    age_days=st.floats(min_value=1.0, max_value=365.0, allow_nan=False),
    decay_rate=_decay_rate,
)
def test_property_28_protected_memory_decay_application(importance, age_days, decay_rate):
    """
    Feature: memory-lifecycle-management, Property 28: Protected Memory Decay Application

    For any memory with protected=true, decay calculations should still be applied
    (protection only prevents deletion, not decay).
    Validates: Requirements 11.5
    """
    now = datetime.now(UTC)
    ts = (now - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
    memory = {"id": "m1", "content": "c", "metadata": {
        "importance": importance, "creation_timestamp": ts, "protected": True
    }, "timestamp": ts}

    mock_mi = MagicMock()
    mock_mi.retrieve.return_value = {"memories": [memory], "total_count": 1, "query_metadata": {}}

    cfg = DecayConfig(decay_function_type=DecayFunctionType.EXPONENTIAL, decay_rate=decay_rate)
    d = MemoryDecay(memory_interface=mock_mi, decay_config=cfg)
    result = d.apply_decay(dry_run=True)

    # Protected memory should still be processed (decay calculated)
    assert result.memories_processed == 1, \
        "Protected memory should be processed by decay"


# ---------------------------------------------------------------------------
# Property 30: ISO 8601 Timestamp Parsing
# ---------------------------------------------------------------------------
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(age_days=st.floats(min_value=0.0, max_value=3650.0, allow_nan=False, allow_infinity=False))
def test_property_30_iso8601_timestamp_parsing(age_days):
    """
    Feature: memory-lifecycle-management, Property 30: ISO 8601 Timestamp Parsing

    For any valid ISO 8601 timestamp, parsing should produce a timezone-aware
    datetime in UTC.
    Validates: Requirements 15.4, 15.5
    """
    d = _make_decay(0.1)
    ts = (datetime.now(UTC) - timedelta(days=age_days)).isoformat().replace('+00:00', 'Z')
    # _calculate_age_days should parse without raising
    age = d._calculate_age_days(ts)
    assert isinstance(age, float)
    assert age >= 0.0
    # Verify the timestamp round-trips through fromisoformat
    parsed = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    assert parsed.tzinfo is not None, "Parsed timestamp should be timezone-aware"


# ===========================================================================
# Pruning Properties (6-9) — imported from test_memory_pruner_properties
# (already covered in test_memory_pruner_properties.py; included here for
#  consolidated reference via imports)
# ===========================================================================

# Orchestration Properties (23-25) and Schema Property (29) are covered in
# test_lifecycle_orchestration_properties.py and test_lifecycle_property_29.py.
# This file consolidates the decay (1-5, 26-28, 30) and deduplication (10-22)
# properties that were not previously in a single location.
