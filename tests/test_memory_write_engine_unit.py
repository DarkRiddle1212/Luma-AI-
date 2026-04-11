"""Unit tests for Memory Write Engine orchestrator.

Tests the MemoryWriteEngine class which coordinates the memory processing
pipeline: extraction, scoring, and persistence.
"""

import pytest
from luma.core.memory_write import (
    MemoryWriteEngine,
    MemoryCandidate,
    ScoredMemory,
    StoredMemory,
    MemoryWriteResult,
)


# Mock implementations for testing

class MockMemoryExtractor:
    """Mock Memory Extractor for testing."""
    
    def __init__(self, candidates):
        """Initialize with predefined candidates.
        
        Args:
            candidates: List of MemoryCandidate objects to return
        """
        self.candidates = candidates
        self.extract_calls = []
    
    def extract_candidates(self, user_query, system_response):
        """Mock extract_candidates method."""
        self.extract_calls.append((user_query, system_response))
        return self.candidates


class MockImportanceScorer:
    """Mock Importance Scorer for testing."""
    
    def __init__(self, scores):
        """Initialize with predefined scores.
        
        Args:
            scores: Dict mapping candidate text to importance score
                   (None means filtered out)
        """
        self.scores = scores
        self.score_calls = []
    
    def score_memory(self, candidate):
        """Mock score_memory method."""
        self.score_calls.append(candidate)
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
        """Initialize with predefined stored memories.
        
        Args:
            stored_memories: Dict mapping scored memory text to StoredMemory
        """
        self.stored_memories = stored_memories
        self.store_calls = []
    
    def store_memory(self, scored_memory):
        """Mock store_memory method."""
        self.store_calls.append(scored_memory)
        return self.stored_memories[scored_memory.text]


# Test fixtures

@pytest.fixture
def sample_candidates():
    """Sample candidate memories for testing."""
    return [
        MemoryCandidate(text="I want to build a web app", type="project_goal"),
        MemoryCandidate(text="I prefer Python", type="user_preference"),
        MemoryCandidate(text="Hello there", type="statement"),
    ]


@pytest.fixture
def sample_scores():
    """Sample importance scores for testing."""
    return {
        "I want to build a web app": 0.85,
        "I prefer Python": 0.80,
        "Hello there": None,  # Filtered out (below threshold)
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
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
        "I prefer Python": StoredMemory(
            memory_id="mem_002",
            text="I prefer Python",
            type="user_preference",
            importance=0.80,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
    }


# Tests

def test_memory_write_engine_initialization():
    """Test MemoryWriteEngine initialization with dependencies."""
    extractor = MockMemoryExtractor([])
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    assert engine.extractor is extractor
    assert engine.scorer is scorer
    assert engine.writer is writer


def test_process_full_pipeline(sample_candidates, sample_scores, sample_stored):
    """Test full pipeline orchestration with mock dependencies."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="I want to build a web app. I prefer Python.",
        system_response="Hello there! I can help you with that."
    )
    
    # Verify result structure
    assert isinstance(result, MemoryWriteResult)
    assert len(result.stored_memories) == 2
    assert len(result.ignored_memories) == 1
    
    # Verify stored memories
    stored_texts = [m.text for m in result.stored_memories]
    assert "I want to build a web app" in stored_texts
    assert "I prefer Python" in stored_texts
    
    # Verify ignored memories
    ignored_texts = [m.text for m in result.ignored_memories]
    assert "Hello there" in ignored_texts


def test_extractor_invocation_with_correct_arguments(sample_candidates):
    """Test that extractor is called with correct arguments."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    user_query = "I want to build a web app"
    system_response = "Great! Let's get started."
    
    engine.process(user_query, system_response)
    
    # Verify extractor was called with correct arguments
    assert len(extractor.extract_calls) == 1
    assert extractor.extract_calls[0] == (user_query, system_response)


def test_scorer_invocation_for_each_candidate(sample_candidates, sample_scores, sample_stored):
    """Test that scorer is called for each candidate."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify scorer was called for each candidate
    assert len(scorer.score_calls) == len(sample_candidates)
    scored_texts = [c.text for c in scorer.score_calls]
    for candidate in sample_candidates:
        assert candidate.text in scored_texts


def test_writer_invocation_only_for_above_threshold(
    sample_candidates, sample_scores, sample_stored
):
    """Test that writer is called only for memories above threshold."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify writer was called only for memories above threshold
    # (2 out of 3 candidates have scores above threshold)
    assert len(writer.store_calls) == 2
    stored_texts = [m.text for m in writer.store_calls]
    assert "I want to build a web app" in stored_texts
    assert "I prefer Python" in stored_texts
    assert "Hello there" not in stored_texts


def test_memory_write_result_structure(sample_candidates, sample_scores, sample_stored):
    """Test that MemoryWriteResult has correct structure."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify result structure
    assert isinstance(result, MemoryWriteResult)
    assert hasattr(result, 'stored_memories')
    assert hasattr(result, 'ignored_memories')
    assert isinstance(result.stored_memories, list)
    assert isinstance(result.ignored_memories, list)


def test_stored_memories_included_in_result(
    sample_candidates, sample_scores, sample_stored
):
    """Test that stored memories are correctly included in result."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify stored memories
    assert len(result.stored_memories) == 2
    for memory in result.stored_memories:
        assert isinstance(memory, StoredMemory)
        assert memory.memory_id is not None
        assert memory.text is not None
        assert memory.type is not None
        assert 0.0 <= memory.importance <= 1.0


def test_ignored_memories_included_in_result(
    sample_candidates, sample_scores, sample_stored
):
    """Test that ignored memories are correctly included in result."""
    extractor = MockMemoryExtractor(sample_candidates)
    scorer = MockImportanceScorer(sample_scores)
    writer = MockMemoryWriter(sample_stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify ignored memories
    assert len(result.ignored_memories) == 1
    for memory in result.ignored_memories:
        assert isinstance(memory, MemoryCandidate)
        assert memory.text is not None
        assert memory.type is not None


def test_empty_candidate_list():
    """Test handling of empty candidate list from extractor."""
    extractor = MockMemoryExtractor([])
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify empty result
    assert len(result.stored_memories) == 0
    assert len(result.ignored_memories) == 0


def test_all_candidates_below_threshold():
    """Test handling when all candidates are below threshold."""
    candidates = [
        MemoryCandidate(text="Hello", type="statement"),
        MemoryCandidate(text="Thanks", type="statement"),
    ]
    
    scores = {
        "Hello": None,  # Below threshold
        "Thanks": None,  # Below threshold
    }
    
    extractor = MockMemoryExtractor(candidates)
    scorer = MockImportanceScorer(scores)
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify no memories stored, all ignored
    assert len(result.stored_memories) == 0
    assert len(result.ignored_memories) == 2


def test_invalid_user_query_raises_error():
    """Test that invalid user_query raises ValueError."""
    extractor = MockMemoryExtractor([])
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    # Test None user_query
    with pytest.raises(ValueError, match="user_query must be non-empty"):
        engine.process(user_query=None, system_response="Test response")
    
    # Test empty user_query
    with pytest.raises(ValueError, match="user_query must be non-empty"):
        engine.process(user_query="", system_response="Test response")
    
    # Test whitespace-only user_query
    with pytest.raises(ValueError, match="user_query must be non-empty"):
        engine.process(user_query="   ", system_response="Test response")


def test_invalid_system_response_raises_error():
    """Test that invalid system_response raises ValueError."""
    extractor = MockMemoryExtractor([])
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    # Test None system_response
    with pytest.raises(ValueError, match="system_response must be non-empty"):
        engine.process(user_query="Test query", system_response=None)
    
    # Test empty system_response
    with pytest.raises(ValueError, match="system_response must be non-empty"):
        engine.process(user_query="Test query", system_response="")
    
    # Test whitespace-only system_response
    with pytest.raises(ValueError, match="system_response must be non-empty"):
        engine.process(user_query="Test query", system_response="   ")


def test_component_error_propagation():
    """Test that component errors are propagated to caller."""
    
    class FailingExtractor:
        def extract_candidates(self, user_query, system_response):
            raise RuntimeError("Extraction failed")
    
    extractor = FailingExtractor()
    scorer = MockImportanceScorer({})
    writer = MockMemoryWriter({})
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    # Verify error is propagated
    with pytest.raises(RuntimeError, match="Extraction failed"):
        engine.process(
            user_query="Test query",
            system_response="Test response"
        )


def test_multiple_memories_stored():
    """Test storing multiple memories in a single interaction."""
    candidates = [
        MemoryCandidate(text="Goal 1", type="project_goal"),
        MemoryCandidate(text="Goal 2", type="project_goal"),
        MemoryCandidate(text="Preference 1", type="user_preference"),
    ]
    
    scores = {
        "Goal 1": 0.85,
        "Goal 2": 0.90,
        "Preference 1": 0.80,
    }
    
    stored = {
        "Goal 1": StoredMemory(
            memory_id="mem_001",
            text="Goal 1",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
        "Goal 2": StoredMemory(
            memory_id="mem_002",
            text="Goal 2",
            type="project_goal",
            importance=0.90,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
        "Preference 1": StoredMemory(
            memory_id="mem_003",
            text="Preference 1",
            type="user_preference",
            importance=0.80,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
    }
    
    extractor = MockMemoryExtractor(candidates)
    scorer = MockImportanceScorer(scores)
    writer = MockMemoryWriter(stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify all memories stored
    assert len(result.stored_memories) == 3
    assert len(result.ignored_memories) == 0


def test_mixed_stored_and_ignored():
    """Test interaction with both stored and ignored memories."""
    candidates = [
        MemoryCandidate(text="Important goal", type="project_goal"),
        MemoryCandidate(text="Hello", type="statement"),
        MemoryCandidate(text="User preference", type="user_preference"),
        MemoryCandidate(text="Thanks", type="statement"),
    ]
    
    scores = {
        "Important goal": 0.85,
        "Hello": None,  # Filtered
        "User preference": 0.80,
        "Thanks": None,  # Filtered
    }
    
    stored = {
        "Important goal": StoredMemory(
            memory_id="mem_001",
            text="Important goal",
            type="project_goal",
            importance=0.85,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
        "User preference": StoredMemory(
            memory_id="mem_002",
            text="User preference",
            type="user_preference",
            importance=0.80,
            created_at="2024-01-01T00:00:00",
            is_update=False
        ),
    }
    
    extractor = MockMemoryExtractor(candidates)
    scorer = MockImportanceScorer(scores)
    writer = MockMemoryWriter(stored)
    
    engine = MemoryWriteEngine(
        extractor=extractor,
        scorer=scorer,
        writer=writer
    )
    
    result = engine.process(
        user_query="Test query",
        system_response="Test response"
    )
    
    # Verify mixed result
    assert len(result.stored_memories) == 2
    assert len(result.ignored_memories) == 2
    
    stored_texts = [m.text for m in result.stored_memories]
    assert "Important goal" in stored_texts
    assert "User preference" in stored_texts
    
    ignored_texts = [m.text for m in result.ignored_memories]
    assert "Hello" in ignored_texts
    assert "Thanks" in ignored_texts
