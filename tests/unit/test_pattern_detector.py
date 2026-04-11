"""
Unit tests for luma.core.insight.pattern_detector.PatternDetector.

Covers:
- Empty input returns []
- Keyword at exactly min_frequency is included; below threshold is excluded
- Patterns below min_confidence are excluded
- Evidence contains only IDs from input memories
- Output is sorted deterministically (frequency DESC, pattern ASC)
- ValueError raised for min_frequency < 1
- ValueError raised for min_confidence outside [0.0, 1.0]
- Memories with None content are skipped gracefully
"""

import pytest
from luma.core.insight.pattern_detector import PatternDetector
from luma.core.memory_interface import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mem(mem_id: str, content: str | None, category: str = "general") -> MemoryEntry:
    return {
        "id": mem_id,
        "content": content,
        "metadata": {},
        "timestamp": "2024-01-01T00:00:00",
        "category": category,
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_min_frequency_zero_raises(self):
        with pytest.raises(ValueError, match="min_frequency"):
            PatternDetector(min_frequency=0)

    def test_min_frequency_negative_raises(self):
        with pytest.raises(ValueError, match="min_frequency"):
            PatternDetector(min_frequency=-1)

    def test_min_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            PatternDetector(min_confidence=-0.1)

    def test_min_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            PatternDetector(min_confidence=1.1)

    def test_valid_defaults_do_not_raise(self):
        detector = PatternDetector()
        assert detector is not None

    def test_boundary_values_valid(self):
        # min_frequency=1, min_confidence at boundaries
        PatternDetector(min_frequency=1, min_confidence=0.0)
        PatternDetector(min_frequency=1, min_confidence=1.0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list_returns_empty(self):
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        assert detector.detect([]) == []


# ---------------------------------------------------------------------------
# Frequency threshold
# ---------------------------------------------------------------------------

class TestFrequencyThreshold:
    def test_keyword_at_exactly_min_frequency_is_included(self):
        # "python" appears in exactly 2 memories; min_frequency=2
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" in patterns

    def test_keyword_below_min_frequency_is_excluded(self):
        # "python" appears only once; min_frequency=2
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "java rocks"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" not in patterns

    def test_min_frequency_one_includes_single_occurrence(self):
        memories = [_mem("m1", "unique keyword here")]
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "unique" in patterns


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    def test_pattern_below_min_confidence_excluded(self):
        # 3 memories, keyword appears in 1 → confidence = 1/3 ≈ 0.33
        # min_confidence=0.5 → should be excluded
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "java rocks"),
            _mem("m3", "go is fast"),
        ]
        detector = PatternDetector(min_frequency=1, min_confidence=0.5)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" not in patterns

    def test_pattern_at_min_confidence_included(self):
        # 2 memories, keyword appears in 1 → confidence = 0.5
        # min_confidence=0.5 → should be included (>= threshold)
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "java rocks"),
        ]
        detector = PatternDetector(min_frequency=1, min_confidence=0.5)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" in patterns

    def test_min_confidence_zero_includes_all_above_frequency(self):
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
            _mem("m3", "java is fine"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" in patterns


# ---------------------------------------------------------------------------
# Evidence correctness
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_contains_only_input_ids(self):
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
            _mem("m3", "java is fine"),
        ]
        input_ids = {m["id"] for m in memories}
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        results = detector.detect(memories)
        for result in results:
            for eid in result.evidence:
                assert eid in input_ids, f"Evidence ID {eid!r} not in input memories"

    def test_evidence_non_empty_for_all_results(self):
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
        ]
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        results = detector.detect(memories)
        for result in results:
            assert len(result.evidence) > 0

    def test_each_memory_id_appears_at_most_once_per_pattern(self):
        # A memory should contribute its ID only once per pattern even if
        # the keyword appears multiple times in the content.
        memories = [
            _mem("m1", "python python python"),
            _mem("m2", "python is great"),
        ]
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        results = detector.detect(memories)
        python_results = [r for r in results if r.pattern == "python"]
        assert len(python_results) == 1
        evidence = python_results[0].evidence
        assert len(evidence) == len(set(evidence)), "Duplicate IDs in evidence"


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------

class TestSorting:
    def test_output_sorted_by_frequency_desc(self):
        # "python" appears 3 times, "java" appears 2 times
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
            _mem("m3", "python and java"),
            _mem("m4", "java is fine"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        freqs = [r.frequency for r in results]
        assert freqs == sorted(freqs, reverse=True), "Results not sorted by frequency DESC"

    def test_output_sorted_by_pattern_asc_for_equal_frequency(self):
        # Both "alpha" and "beta" appear in exactly 2 memories
        memories = [
            _mem("m1", "alpha beta"),
            _mem("m2", "alpha beta"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        # Filter to just the two keywords
        kw_results = [r for r in results if r.pattern in ("alpha", "beta")]
        assert len(kw_results) == 2
        assert kw_results[0].pattern == "alpha"
        assert kw_results[1].pattern == "beta"

    def test_same_input_produces_same_output(self):
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
            _mem("m3", "java is fine"),
        ]
        detector = PatternDetector(min_frequency=1, min_confidence=0.0)
        results1 = detector.detect(memories)
        results2 = detector.detect(memories)
        assert [(r.pattern, r.frequency) for r in results1] == [
            (r.pattern, r.frequency) for r in results2
        ]


# ---------------------------------------------------------------------------
# None content handling
# ---------------------------------------------------------------------------

class TestNoneContent:
    def test_memory_with_none_content_is_skipped(self):
        memories = [
            _mem("m1", None),
            _mem("m2", "python is great"),
            _mem("m3", "python rocks"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        # Should not raise; m1 is skipped
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "python" in patterns

    def test_all_none_content_returns_empty_or_category_only(self):
        memories = [
            _mem("m1", None, category="work"),
            _mem("m2", None, category="work"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        # No keywords, but category "work" should appear
        patterns = [r.pattern for r in results]
        assert "work" in patterns

    def test_no_crash_on_mixed_none_and_valid_content(self):
        memories = [
            _mem("m1", None),
            _mem("m2", "hello world"),
            _mem("m3", None),
            _mem("m4", "hello there"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        patterns = [r.pattern for r in results]
        assert "hello" in patterns


# ---------------------------------------------------------------------------
# Category patterns
# ---------------------------------------------------------------------------

class TestCategoryPatterns:
    def test_category_pattern_detected(self):
        memories = [
            _mem("m1", "some content", category="work"),
            _mem("m2", "other content", category="work"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        category_results = [r for r in results if r.pattern_type == "category"]
        assert any(r.pattern == "work" for r in category_results)

    def test_pattern_type_keyword_for_keywords(self):
        memories = [
            _mem("m1", "python is great"),
            _mem("m2", "python rocks"),
        ]
        detector = PatternDetector(min_frequency=2, min_confidence=0.0)
        results = detector.detect(memories)
        python_results = [r for r in results if r.pattern == "python"]
        assert len(python_results) == 1
        assert python_results[0].pattern_type == "keyword"
