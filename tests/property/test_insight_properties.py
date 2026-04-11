"""
Property-based tests for the Pattern Recognition & Insight Engine.

Properties 1-5 cover PatternDetector behaviour.
Each test is tagged: Feature: pattern-recognition-insight-engine, Property N: description
Hypothesis configured with max_examples=100 per test.
"""

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import List

from luma.core.insight.pattern_detector import PatternDetector
from luma.core.memory_interface import MemoryEntry


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_mem_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=12,
)

_content = st.one_of(
    st.none(),
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_size=0,
        max_size=80,
    ),
)

_category = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=1,
    max_size=12,
)

_timestamp = st.just("2024-01-01T00:00:00")


def memory_entry_strategy() -> st.SearchStrategy:
    """Generate valid MemoryEntry dicts with unique-ish IDs."""
    return st.fixed_dictionaries(
        {
            "id": _mem_id,
            "content": _content,
            "metadata": st.just({}),
            "timestamp": _timestamp,
            "category": _category,
            "tags": st.just([]),
        }
    )


def _unique_memories(memories: List[MemoryEntry]) -> List[MemoryEntry]:
    """Deduplicate by ID so evidence subset checks are unambiguous."""
    seen = set()
    result = []
    for m in memories:
        if m["id"] not in seen:
            seen.add(m["id"])
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# Property 1: Confidence scores are always in [0.0, 1.0]
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 1: Confidence scores are always in [0.0, 1.0]
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_confidence_always_in_range(memories):
    """
    **Validates: Requirements 2.4, 12.3**

    For any list of memory entries, every PatternResult.confidence returned by
    PatternDetector.detect() must be in [0.0, 1.0].
    """
    memories = _unique_memories(memories)
    detector = PatternDetector(min_frequency=1, min_confidence=0.0)
    results = detector.detect(memories)
    for r in results:
        assert 0.0 <= r.confidence <= 1.0, (
            f"confidence {r.confidence} out of [0.0, 1.0] for pattern {r.pattern!r}"
        )


# ---------------------------------------------------------------------------
# Property 2: Evidence IDs are a subset of input memory IDs
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 2: Evidence IDs are a subset of input memory IDs
@given(memories=st.lists(memory_entry_strategy(), min_size=1, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_evidence_ids_subset_of_inputs(memories):
    """
    **Validates: Requirements 2.5, 9.2, 12.5**

    For any list of memory entries, every ID in PatternResult.evidence must be
    present in the input memory list.
    """
    memories = _unique_memories(memories)
    input_ids = {m["id"] for m in memories}
    detector = PatternDetector(min_frequency=1, min_confidence=0.0)
    results = detector.detect(memories)
    for r in results:
        for eid in r.evidence:
            assert eid in input_ids, (
                f"Evidence ID {eid!r} not found in input memory IDs for pattern {r.pattern!r}"
            )


# ---------------------------------------------------------------------------
# Property 3: Frequency threshold filtering
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 3: Frequency threshold filtering
@given(
    memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50),
    threshold=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_frequency_threshold_respected(memories, threshold):
    """
    **Validates: Requirements 2.8, 12.2**

    For any list of memory entries and any min_frequency threshold, every
    PatternResult returned must have frequency >= threshold.
    """
    memories = _unique_memories(memories)
    detector = PatternDetector(min_frequency=threshold, min_confidence=0.0)
    results = detector.detect(memories)
    for r in results:
        assert r.frequency >= threshold, (
            f"Pattern {r.pattern!r} has frequency {r.frequency} < threshold {threshold}"
        )


# ---------------------------------------------------------------------------
# Property 4: Frequency equals evidence count
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 4: Frequency equals evidence count
@given(memories=st.lists(memory_entry_strategy(), min_size=1, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_frequency_equals_evidence_count(memories):
    """
    **Validates: Requirements 2.3, 12.4**

    For any PatternResult returned by PatternDetector.detect(), the frequency
    field must equal len(evidence).
    """
    memories = _unique_memories(memories)
    detector = PatternDetector(min_frequency=1, min_confidence=0.0)
    results = detector.detect(memories)
    for r in results:
        assert r.frequency == len(r.evidence), (
            f"Pattern {r.pattern!r}: frequency={r.frequency} != len(evidence)={len(r.evidence)}"
        )


# ---------------------------------------------------------------------------
# Property 5: Determinism — same inputs produce same outputs
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 5: Determinism — same inputs produce same outputs
@given(memories=st.lists(memory_entry_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_detector_is_deterministic(memories):
    """
    **Validates: Requirements 2.9, 12.6**

    Calling PatternDetector.detect() twice with identical inputs must produce
    identical outputs (same objects in same order).
    """
    memories = _unique_memories(memories)
    detector = PatternDetector(min_frequency=1, min_confidence=0.0)
    results1 = detector.detect(memories)
    results2 = detector.detect(memories)

    assert len(results1) == len(results2), (
        f"Non-deterministic result count: {len(results1)} vs {len(results2)}"
    )
    for r1, r2 in zip(results1, results2):
        assert r1.pattern_type == r2.pattern_type
        assert r1.pattern == r2.pattern
        assert r1.frequency == r2.frequency
        assert r1.confidence == r2.confidence
        assert r1.evidence == r2.evidence


# ---------------------------------------------------------------------------
# TrendAnalyzer imports and strategies
# ---------------------------------------------------------------------------

from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.insight.schemas import PatternResult as _PatternResult
from datetime import datetime, timedelta


def memory_entry_strategy_with_timestamps() -> st.SearchStrategy:
    """
    Generate MemoryEntry dicts with timestamps spread across a 365-day range
    so that memories are not all in the same time window.
    """
    base_date = datetime(2024, 1, 1)

    def make_entry(mem_id, content, category, day_offset):
        ts = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "id": mem_id,
            "content": content,
            "metadata": {},
            "timestamp": ts,
            "category": category,
            "tags": [],
        }

    return st.builds(
        make_entry,
        mem_id=_mem_id,
        content=_content,
        category=_category,
        day_offset=st.integers(min_value=0, max_value=364),
    )


def _unique_memories_ts(memories):
    """Deduplicate by ID."""
    seen = set()
    result = []
    for m in memories:
        if m["id"] not in seen:
            seen.add(m["id"])
            result.append(m)
    return result


def _make_patterns_from_memories(memories) -> List[_PatternResult]:
    """
    Build a minimal set of PatternResult objects from the given memories
    so TrendAnalyzer has something to evaluate.
    Each memory contributes its ID as evidence for a single 'all' pattern.
    """
    if not memories:
        return []
    ids = [m["id"] for m in memories]
    return [
        _PatternResult(
            pattern_type="keyword",
            pattern="all",
            frequency=len(ids),
            confidence=1.0,
            evidence=ids,
        )
    ]


# ---------------------------------------------------------------------------
# Property 5 (TrendAnalyzer): Determinism — same inputs produce same outputs
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 5: Determinism — same inputs produce same outputs (TrendAnalyzer)
@given(
    memories=st.lists(memory_entry_strategy_with_timestamps(), min_size=0, max_size=30),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_trend_analyzer_is_deterministic(memories):
    """
    **Validates: Requirements 3.9**

    Calling TrendAnalyzer.analyze() twice with identical inputs must produce
    identical outputs (same objects in same order).
    """
    memories = _unique_memories_ts(memories)
    patterns = _make_patterns_from_memories(memories)

    analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)
    results1 = analyzer.analyze(patterns, memories)
    results2 = analyzer.analyze(patterns, memories)

    assert len(results1) == len(results2), (
        f"Non-deterministic result count: {len(results1)} vs {len(results2)}"
    )
    for r1, r2 in zip(results1, results2):
        assert r1.trend == r2.trend
        assert r1.topic == r2.topic
        assert r1.confidence == r2.confidence
        assert r1.time_window == r2.time_window


# ---------------------------------------------------------------------------
# InsightGenerator imports and strategies
# ---------------------------------------------------------------------------

from luma.core.insight.insight_generator import InsightGenerator
from luma.core.insight.schemas import TrendResult as _TrendResult


def pattern_result_strategy(min_confidence: float = 0.0) -> st.SearchStrategy:
    """Generate valid PatternResult objects above a given confidence floor."""
    return st.builds(
        lambda pattern, frequency, confidence, evidence: _PatternResult(
            pattern_type="keyword",
            pattern=pattern,
            frequency=frequency,
            confidence=confidence,
            evidence=evidence,
        ),
        pattern=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=1,
            max_size=12,
        ),
        frequency=st.integers(min_value=1, max_value=20),
        confidence=st.floats(min_value=min_confidence, max_value=1.0, allow_nan=False),
        evidence=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
            min_size=1,
            max_size=10,
        ),
    )


# ---------------------------------------------------------------------------
# Property 6: Insight coverage — every qualifying pattern produces an insight
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 6: Insight coverage — every qualifying pattern produces an insight
@given(
    patterns=st.lists(pattern_result_strategy(min_confidence=0.0), min_size=0, max_size=20),
    min_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_qualifying_patterns_produce_insights(patterns, min_confidence):
    """
    **Validates: Requirements 4.2, 4.3**

    For any list of PatternResult objects where confidence >= min_confidence,
    the InsightGenerator SHALL produce at least one Insight referencing that
    pattern's topic.
    """
    generator = InsightGenerator(min_confidence=min_confidence)
    results = generator.generate(patterns, [])

    # Collect all topics referenced in insights
    result_texts = [i.text for i in results]

    for pattern in patterns:
        if pattern.confidence >= min_confidence and pattern.evidence:
            # At least one insight must reference this pattern's topic
            assert any(pattern.pattern in text for text in result_texts), (
                f"No insight found for qualifying pattern {pattern.pattern!r} "
                f"(confidence={pattern.confidence}, min_confidence={min_confidence})"
            )


# ---------------------------------------------------------------------------
# Property 8: All insights have non-empty evidence
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 8: All insights have non-empty evidence
@given(
    patterns=st.lists(pattern_result_strategy(min_confidence=0.0), min_size=0, max_size=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_all_insights_have_non_empty_evidence(patterns):
    """
    **Validates: Requirements 4.6, 9.1, 9.5, 13.6**

    For any InsightGenerator output, every Insight in the result list SHALL
    have a non-empty evidence list.
    """
    generator = InsightGenerator(min_confidence=0.0)
    results = generator.generate(patterns, [])

    for insight in results:
        assert len(insight.evidence) > 0, (
            f"Insight has empty evidence: {insight.text!r}"
        )


# ---------------------------------------------------------------------------
# InsightEngine imports
# ---------------------------------------------------------------------------

import copy
from luma.core.insight.insight_engine import InsightEngine
from luma.core.memory_interface import MemoryInterface, MemoryRetrievalError, RetrievalResult
from luma.core.insight.schemas import InsightReport


class _MockMemoryInterface(MemoryInterface):
    """Local mock for property tests — returns a fixed list of memories."""

    def __init__(self, memories):
        self._memories = memories

    def store(self, content, metadata=None):
        raise AssertionError("InsightEngine must never call store()")

    def retrieve(self, query=None, params=None, limit=10, **kwargs):
        return {
            "memories": self._memories,
            "total_count": len(self._memories),
            "query_metadata": {},
        }


def memory_list_strategy():
    """Strategy producing lists of memories with timestamps spread across a year."""
    return st.lists(memory_entry_strategy_with_timestamps(), min_size=0, max_size=30).map(
        _unique_memories_ts
    )


def memory_list_strategy_with_timestamps():
    """Strategy producing non-empty lists of memories with timestamps spread across a year."""
    return st.lists(memory_entry_strategy_with_timestamps(), min_size=1, max_size=30).map(
        _unique_memories_ts
    )


# ---------------------------------------------------------------------------
# Property 7: InsightReport metadata counts are accurate
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 7: InsightReport metadata counts are accurate
@given(memories=memory_list_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_report_counts_are_accurate(memories):
    """
    **Validates: Requirements 1.6, 13.3, 13.4, 13.5**

    InsightReport.pattern_count == number of PatternResult objects detected
    InsightReport.trend_count == number of TrendResult objects detected
    InsightReport.memory_count == number of memories retrieved
    """
    mock = _MockMemoryInterface(memories)
    engine = InsightEngine(
        memory_interface=mock,
        pattern_detector=PatternDetector(min_frequency=1, min_confidence=0.0),
        trend_analyzer=TrendAnalyzer(trend_ratio_threshold=1.5),
        insight_generator=InsightGenerator(min_confidence=0.0),
    )

    report = engine.generate_insights()

    # Independently compute expected counts using the same components
    detector = PatternDetector(min_frequency=1, min_confidence=0.0)
    analyzer = TrendAnalyzer(trend_ratio_threshold=1.5)

    expected_patterns = detector.detect(memories)
    expected_trends = analyzer.analyze(expected_patterns, memories)

    assert report.memory_count == len(memories), (
        f"memory_count {report.memory_count} != len(memories) {len(memories)}"
    )
    assert report.pattern_count == len(expected_patterns), (
        f"pattern_count {report.pattern_count} != expected {len(expected_patterns)}"
    )
    assert report.trend_count == len(expected_trends), (
        f"trend_count {report.trend_count} != expected {len(expected_trends)}"
    )


# ---------------------------------------------------------------------------
# Property 9: Memory immutability
# ---------------------------------------------------------------------------

# Feature: pattern-recognition-insight-engine, Property 9: Memory immutability
@given(memories=memory_list_strategy_with_timestamps())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_memories_not_mutated(memories):
    """
    **Validates: Requirements 1.9, 3.10, 8.5**

    After calling generate_insights(), every MemoryEntry field
    (id, content, metadata, timestamp, category, tags) must be identical
    to before the call.
    """
    # Deep copy before the call
    memories_before = copy.deepcopy(memories)

    mock = _MockMemoryInterface(memories)
    engine = InsightEngine(
        memory_interface=mock,
        pattern_detector=PatternDetector(min_frequency=1, min_confidence=0.0),
        trend_analyzer=TrendAnalyzer(trend_ratio_threshold=1.5),
        insight_generator=InsightGenerator(min_confidence=0.0),
    )

    engine.generate_insights()

    # Compare field by field
    assert len(memories) == len(memories_before), (
        "Memory list length changed after generate_insights()"
    )
    for i, (after, before) in enumerate(zip(memories, memories_before)):
        assert after["id"] == before["id"], (
            f"Memory[{i}].id mutated: {before['id']!r} → {after['id']!r}"
        )
        assert after["content"] == before["content"], (
            f"Memory[{i}].content mutated: {before['content']!r} → {after['content']!r}"
        )
        assert after["metadata"] == before["metadata"], (
            f"Memory[{i}].metadata mutated: {before['metadata']!r} → {after['metadata']!r}"
        )
        assert after["timestamp"] == before["timestamp"], (
            f"Memory[{i}].timestamp mutated: {before['timestamp']!r} → {after['timestamp']!r}"
        )
        assert after["category"] == before["category"], (
            f"Memory[{i}].category mutated: {before['category']!r} → {after['category']!r}"
        )
        assert after["tags"] == before["tags"], (
            f"Memory[{i}].tags mutated: {before['tags']!r} → {after['tags']!r}"
        )
