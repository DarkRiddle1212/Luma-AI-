"""Integration tests for Memory Write Engine instrumentation.

Tests the observability instrumentation of MemoryWriteEngine to verify:
- Metrics are recorded correctly when metrics_collector is provided
- No exceptions occur when observability dependencies are None
- Write results are identical with and without instrumentation
- Failure metrics are recorded on exceptions

Requirements: 14.4, 14.5, 14.6
"""

import pytest
import time
from unittest.mock import Mock
from luma.core.memory_write import (
    MemoryWriteEngine,
    MemoryCandidate,
    ScoredMemory,
    StoredMemory,
    MemoryWriteResult,
)
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


# Mock implementations for testing

class MockMemoryExtractor:
    """Mock Memory Extractor for testing."""
    
    def __init__(self, candidates):
        self.candidates = candidates
    
    def extract_candidates(self, user_query, system_response):
        return self.candidates


class MockImportanceScorer:
    """Mock Importance Scorer for testing."""
    
    def __init__(self, scores):
        self.scores = scores
    
    def score_memory(self, candidate):
        score = self.scores.get(candidate.text)
        if score is None:
            return None
        return ScoredMemory(
            text=candidate.text,
            type=candidate.type,
            importance=score
        )


class MockMemoryWriter:
    """Mock Memory Writer for testing."""
    
    def __init__(self, stored_memories):
        self.stored_memories = stored_memories
    
    def store_memory(self, scored_memory):
        return self.stored_memories[scored_memory.text]


class FailingMemoryExtractor:
    """Mock extractor that raises exceptions for testing failure scenarios."""
    
    def extract_candidates(self, user_query, system_response):
        raise RuntimeError("Extraction failed")


# Test fixtures

@pytest.fixture
def sample_candidates():
    """Sample candidate memories for testing."""
    return [
        MemoryCandidate(text="I want to build a web app", type="project_goal"),
        MemoryCandidate(text="I prefer Python", type="user_preference"),
    ]


@pytest.fixture
def sample_scores():
    """Sample importance scores for testing."""
    return {
        "I want to build a web app": 0.85,
        "I prefer Python": 0.80,
    }


@pytest.fixture
def sample_stored():
    """Sample stored memories for testing."""
    return {
        "I want to build a web app": StoredMemory(
            memory_id="mem_001",
            text="I want to build a web app",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        ),
        "I prefer Python": StoredMemory(
            memory_id="mem_002",
            text="I prefer Python",
            type="user_preference",
            importance=0.80,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        ),
    }


# Integration Tests

def test_memory_write_latency_recorded_with_metrics_collector(
    sample_candidates, sample_scores, sample_stored
):
    """Test that memory_write_latency_ms is recorded when metrics_collector is provided."""
    metrics_collector = MetricsCollector()
    
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=metrics_collector
    )
    
    # Process a memory write operation
    result = engine.process(
        user_query="I want to build a web app. I prefer Python.",
        system_response="Great! Let's get started."
    )
    
    # Verify metrics were recorded
    snapshot = metrics_collector.get_snapshot()
    
    # Check that latency was recorded
    assert "memory_write_latency_ms" in snapshot["timers"]
    timer_stats = snapshot["timers"]["memory_write_latency_ms"]
    assert timer_stats["count"] == 1
    assert timer_stats["sum"] > 0  # Should have some duration
    assert timer_stats["min"] > 0
    assert timer_stats["max"] > 0
    assert timer_stats["mean"] > 0
    
    # Verify result is still correct
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 2


def test_memory_write_count_incremented_with_metrics_collector(
    sample_candidates, sample_scores, sample_stored
):
    """Test that memory_write_count is incremented when metrics_collector is provided."""
    metrics_collector = MetricsCollector()
    
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=metrics_collector
    )
    
    # Process multiple memory write operations
    engine.process(
        user_query="First query",
        system_response="First response"
    )
    
    engine.process(
        user_query="Second query", 
        system_response="Second response"
    )
    
    # Verify counter was incremented
    snapshot = metrics_collector.get_snapshot()
    assert "memory_write_count" in snapshot["counters"]
    assert snapshot["counters"]["memory_write_count"] == 2


def test_memory_write_failures_incremented_on_exceptions():
    """Test that memory_write_failures is incremented on exceptions."""
    metrics_collector = MetricsCollector()
    
    # Use failing extractor to trigger exception
    extractor = FailingMemoryExtractor()
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=metrics_collector
    )
    
    # Process should raise exception
    with pytest.raises(RuntimeError, match="Extraction failed"):
        engine.process(
            user_query="Test query",
            system_response="Test response"
        )
    
    # Verify failure metric was incremented
    snapshot = metrics_collector.get_snapshot()
    assert "memory_write_failures" in snapshot["counters"]
    assert snapshot["counters"]["memory_write_failures"] == 1
    
    # Verify latency was still recorded for failed operation
    assert "memory_write_latency_ms" in snapshot["timers"]
    timer_stats = snapshot["timers"]["memory_write_latency_ms"]
    assert timer_stats["count"] == 1


def test_no_exceptions_when_metrics_collector_is_none(
    sample_candidates, sample_scores, sample_stored
):
    """Test that no exceptions occur when metrics_collector is None."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    # Create engine without metrics_collector
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=None  # Explicitly None
    )
    
    # Should not raise any exceptions
    result = engine.process(
        user_query="I want to build a web app",
        system_response="Great idea!"
    )
    
    # Verify result is still correct
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 2


def test_no_exceptions_when_logger_is_none(
    sample_candidates, sample_scores, sample_stored
):
    """Test that no exceptions occur when logger is None."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    # Create engine without logger
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        logger=None  # Explicitly None
    )
    
    # Should not raise any exceptions
    result = engine.process(
        user_query="I want to build a web app",
        system_response="Great idea!"
    )
    
    # Verify result is still correct
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 2


def test_no_exceptions_when_both_observability_dependencies_are_none(
    sample_candidates, sample_scores, sample_stored
):
    """Test that no exceptions occur when both metrics_collector and logger are None."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    # Create engine without any observability dependencies
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=None,
        logger=None
    )
    
    # Should not raise any exceptions
    result = engine.process(
        user_query="I want to build a web app",
        system_response="Great idea!"
    )
    
    # Verify result is still correct
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 2


def test_write_results_identical_with_and_without_instrumentation(
    sample_candidates, sample_scores, sample_stored
):
    """Test that write results are identical with and without instrumentation."""
    extractor1 = MockMemoryExtractor(sample_candidates)
    scorer1 = MockImportanceScorer(sample_scores)
    writer1 = MockMemoryWriter(sample_stored)
    
    extractor2 = MockMemoryExtractor(sample_candidates)
    scorer2 = MockImportanceScorer(sample_scores)
    writer2 = MockMemoryWriter(sample_stored)
    
    # Engine with instrumentation
    instrumented_engine = MemoryWriteEngine(
        extractor=extractor1,
        scorer=scorer1,
        writer=writer1,
        metrics_collector=MetricsCollector(),
        logger=StructuredLogger()
    )
    
    # Engine without instrumentation
    uninstrumented_engine = MemoryWriteEngine(
        extractor=extractor2,
        scorer=scorer2,
        writer=writer2,
        metrics_collector=None,
        logger=None
    )
    
    # Process same inputs
    user_query = "I want to build a web app. I prefer Python."
    system_response = "Great! Let's get started."
    
    instrumented_result = instrumented_engine.process(user_query, system_response)
    uninstrumented_result = uninstrumented_engine.process(user_query, system_response)
    
    # Results should be identical
    assert len(instrumented_result.stored_memories) == len(uninstrumented_result.stored_memories)
    assert len(instrumented_result.ignored_memories) == len(uninstrumented_result.ignored_memories)
    
    # Compare stored memories
    for i, (instrumented_mem, uninstrumented_mem) in enumerate(
        zip(instrumented_result.stored_memories, uninstrumented_result.stored_memories)
    ):
        assert instrumented_mem.memory_id == uninstrumented_mem.memory_id
        assert instrumented_mem.text == uninstrumented_mem.text
        assert instrumented_mem.type == uninstrumented_mem.type
        assert instrumented_mem.importance == uninstrumented_mem.importance
        assert instrumented_mem.created_at == uninstrumented_mem.created_at
        assert instrumented_mem.is_update == uninstrumented_mem.is_update
    
    # Compare ignored memories
    for i, (instrumented_mem, uninstrumented_mem) in enumerate(
        zip(instrumented_result.ignored_memories, uninstrumented_result.ignored_memories)
    ):
        assert instrumented_mem.text == uninstrumented_mem.text
        assert instrumented_mem.type == uninstrumented_mem.type


def test_logger_receives_success_events():
    """Test that logger receives structured events on successful operations."""
    # Use a mock logger to capture log calls
    mock_logger = Mock(spec=StructuredLogger)
    
    sample_candidates = [
        MemoryCandidate(text="I want to build a web app", type="project_goal")
    ]
    sample_scores = {"I want to build a web app": 0.85}
    sample_stored = {
        "I want to build a web app": StoredMemory(
            memory_id="mem_001",
            text="I want to build a web app",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        )
    }
    
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        logger=mock_logger
    )
    
    # Process operation
    engine.process(
        user_query="I want to build a web app",
        system_response="Great idea!"
    )
    
    # Verify logger was called with success event
    mock_logger.log.assert_called_once()
    call_args = mock_logger.log.call_args
    assert call_args[0][0] == "memory_write_completed"  # event name
    
    payload = call_args[0][1]  # payload
    assert "duration_ms" in payload
    assert "candidates_extracted" in payload
    assert "memories_stored" in payload
    assert "memories_ignored" in payload
    assert payload["candidates_extracted"] == 1
    assert payload["memories_stored"] == 1
    assert payload["memories_ignored"] == 0


def test_logger_receives_failure_events():
    """Test that logger receives structured events on failed operations."""
    # Use a mock logger to capture log calls
    mock_logger = Mock(spec=StructuredLogger)
    
    extractor = FailingMemoryExtractor()
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        logger=mock_logger
    )
    
    # Process should raise exception
    with pytest.raises(RuntimeError, match="Extraction failed"):
        engine.process(
            user_query="Test query",
            system_response="Test response"
        )
    
    # Verify logger was called with failure event
    mock_logger.log.assert_called_once()
    call_args = mock_logger.log.call_args
    assert call_args[0][0] == "memory_write_failed"  # event name
    
    payload = call_args[0][1]  # payload
    assert "duration_ms" in payload
    assert "error_type" in payload
    assert "error_message" in payload
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "Extraction failed"


def test_metrics_and_logging_work_together():
    """Test that metrics collection and logging work together without interference."""
    metrics_collector = MetricsCollector()
    mock_logger = Mock(spec=StructuredLogger)
    
    sample_candidates = [
        MemoryCandidate(text="I want to build a web app", type="project_goal")
    ]
    sample_scores = {"I want to build a web app": 0.85}
    sample_stored = {
        "I want to build a web app": StoredMemory(
            memory_id="mem_001",
            text="I want to build a web app",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        )
    }
    
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=metrics_collector,
        logger=mock_logger
    )
    
    # Process operation
    result = engine.process(
        user_query="I want to build a web app",
        system_response="Great idea!"
    )
    
    # Verify metrics were recorded
    snapshot = metrics_collector.get_snapshot()
    assert "memory_write_latency_ms" in snapshot["timers"]
    assert "memory_write_count" in snapshot["counters"]
    assert snapshot["counters"]["memory_write_count"] == 1
    
    # Verify logger was called
    mock_logger.log.assert_called_once()
    call_args = mock_logger.log.call_args
    assert call_args[0][0] == "memory_write_completed"
    
    # Verify result is still correct
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 1


def test_multiple_operations_accumulate_metrics():
    """Test that multiple operations correctly accumulate metrics."""
    metrics_collector = MetricsCollector()
    
    sample_candidates = [
        MemoryCandidate(text="Goal 1", type="project_goal")
    ]
    sample_scores = {"Goal 1": 0.85}
    sample_stored = {
        "Goal 1": StoredMemory(
            memory_id="mem_001",
            text="Goal 1",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        )
    }
    
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer,
        metrics_collector=metrics_collector
    )
    
    # Process multiple operations
    for i in range(3):
        engine.process(
            user_query=f"Query {i}",
            system_response=f"Response {i}"
        )
    
    # Verify metrics accumulated correctly
    snapshot = metrics_collector.get_snapshot()
    
    # Counter should accumulate
    assert snapshot["counters"]["memory_write_count"] == 3
    
    # Timer should have multiple measurements
    timer_stats = snapshot["timers"]["memory_write_latency_ms"]
    assert timer_stats["count"] == 3
    assert timer_stats["sum"] > 0
    assert timer_stats["min"] > 0
    assert timer_stats["max"] > 0
    assert timer_stats["mean"] > 0


def test_failure_and_success_metrics_tracked_separately():
    """Test that failure and success operations are tracked separately."""
    metrics_collector = MetricsCollector()
    
    # Setup for successful operation
    sample_candidates = [MemoryCandidate(text="Goal", type="project_goal")]
    sample_scores = {"Goal": 0.85}
    sample_stored = {
        "Goal": StoredMemory(
            memory_id="mem_001",
            text="Goal",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00Z",
            is_update=False
        )
    }
    
    # Successful engine
    success_extractor = MockMemoryExtractor(sample_candidates)
    success_scorer = MockImportanceScorer(sample_scores)
    success_writer = MockMemoryWriter(sample_stored)
    
    success_engine = MemoryWriteEngine(
        extractor=success_extractor,
        scorer=success_scorer,
        writer=success_writer,
        metrics_collector=metrics_collector
    )
    
    # Failing engine
    fail_engine = MemoryWriteEngine(
        extractor=FailingMemoryExtractor(),
        scorer=MockImportanceScorer({}),
        writer=MockMemoryWriter({}),
        metrics_collector=metrics_collector
    )
    
    # Process successful operation
    success_engine.process("Query", "Response")
    
    # Process failing operation
    with pytest.raises(RuntimeError):
        fail_engine.process("Query", "Response")
    
    # Verify metrics
    snapshot = metrics_collector.get_snapshot()
    
    # Should have 1 success count and 1 failure count
    assert snapshot["counters"]["memory_write_count"] == 1
    assert snapshot["counters"]["memory_write_failures"] == 1
    
    # Should have 2 latency measurements (both success and failure record latency)
    timer_stats = snapshot["timers"]["memory_write_latency_ms"]
    assert timer_stats["count"] == 2