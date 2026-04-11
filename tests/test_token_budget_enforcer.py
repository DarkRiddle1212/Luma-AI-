"""
Unit tests for TokenBudgetEnforcer.

**Validates: Requirements 2.3, 2.4, 8.2, 8.4**
"""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Optional
from luma.core.injection_engine import TokenBudgetEnforcer, TokenEstimator


@dataclass
class MockRankedMemory:
    """Mock RankedMemory object for testing TokenBudgetEnforcer."""
    
    memory_id: str
    timestamp: datetime
    content: str
    namespace: Optional[str]
    similarity_score: float
    importance_score: float
    recency_score: float
    final_score: float
    memory_entry: Any
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_mock_memory(
    memory_id: str,
    content: str,
    token_count: Optional[int] = None,
    final_score: float = 1.0
) -> MockRankedMemory:
    """Helper function to create mock memories for testing."""
    metadata = {}
    if token_count is not None:
        metadata['token_count'] = token_count
    
    return MockRankedMemory(
        memory_id=memory_id,
        timestamp=datetime.now(timezone.utc),
        content=content,
        namespace="test",
        similarity_score=0.9,
        importance_score=0.8,
        recency_score=0.7,
        final_score=final_score,
        memory_entry=None,
        metadata=metadata
    )


# Test initialization and validation

def test_token_budget_enforcer_initialization():
    """Test that TokenBudgetEnforcer initializes correctly with valid parameters."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=2048,
        max_memory_count=50,
        token_estimator=estimator
    )
    
    assert enforcer.max_token_budget == 2048
    assert enforcer.max_memory_count == 50
    assert enforcer.token_estimator is estimator


def test_token_budget_enforcer_negative_budget_raises_error():
    """Test that TokenBudgetEnforcer raises ValueError for negative max_token_budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    with pytest.raises(ValueError) as exc_info:
        TokenBudgetEnforcer(
            max_token_budget=-100,
            max_memory_count=50,
            token_estimator=estimator
        )
    
    error_msg = str(exc_info.value)
    assert "max_token_budget" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_token_budget_enforcer_zero_budget_raises_error():
    """Test that TokenBudgetEnforcer raises ValueError for zero max_token_budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    with pytest.raises(ValueError) as exc_info:
        TokenBudgetEnforcer(
            max_token_budget=0,
            max_memory_count=50,
            token_estimator=estimator
        )
    
    error_msg = str(exc_info.value)
    assert "max_token_budget" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_token_budget_enforcer_negative_memory_count_raises_error():
    """Test that TokenBudgetEnforcer raises ValueError for negative max_memory_count."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    with pytest.raises(ValueError) as exc_info:
        TokenBudgetEnforcer(
            max_token_budget=2048,
            max_memory_count=-10,
            token_estimator=estimator
        )
    
    error_msg = str(exc_info.value)
    assert "max_memory_count" in error_msg.lower()
    assert "positive" in error_msg.lower()


def test_token_budget_enforcer_zero_memory_count_raises_error():
    """Test that TokenBudgetEnforcer raises ValueError for zero max_memory_count."""
    estimator = TokenEstimator(estimation_factor=1.3)
    
    with pytest.raises(ValueError) as exc_info:
        TokenBudgetEnforcer(
            max_token_budget=2048,
            max_memory_count=0,
            token_estimator=estimator
        )
    
    error_msg = str(exc_info.value)
    assert "max_memory_count" in error_msg.lower()
    assert "positive" in error_msg.lower()


# Test token budget cutoff

def test_enforce_token_budget_cutoff():
    """Test that TokenBudgetEnforcer stops selecting memories when token budget would be exceeded.
    
    **Validates: Requirements 2.3, 2.4**
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,  # High enough to not be limiting factor
        token_estimator=estimator
    )
    
    # Create memories with known token counts
    memories = [
        create_mock_memory("mem1", "content", token_count=30),  # Total: 30
        create_mock_memory("mem2", "content", token_count=40),  # Total: 70
        create_mock_memory("mem3", "content", token_count=25),  # Total: 95
        create_mock_memory("mem4", "content", token_count=20),  # Would exceed: 115 > 100
        create_mock_memory("mem5", "content", token_count=10),  # Would exceed: 125 > 100
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 3 memories (total: 95 tokens)
    assert len(selected) == 3
    assert total_tokens == 95
    assert filtered_count == 2
    assert selected[0].memory_id == "mem1"
    assert selected[1].memory_id == "mem2"
    assert selected[2].memory_id == "mem3"
    # Verify budget constraint is satisfied
    assert total_tokens <= 100


def test_enforce_token_budget_exact_fit():
    """Test that TokenBudgetEnforcer selects memories that exactly fit the budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories that exactly fit the budget
    memories = [
        create_mock_memory("mem1", "content", token_count=50),  # Total: 50
        create_mock_memory("mem2", "content", token_count=50),  # Total: 100 (exact fit)
        create_mock_memory("mem3", "content", token_count=10),  # Would exceed
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 2 memories (total: 100 tokens, exact fit)
    assert len(selected) == 2
    assert total_tokens == 100
    assert filtered_count == 1
    assert selected[0].memory_id == "mem1"
    assert selected[1].memory_id == "mem2"


def test_enforce_token_budget_tight_budget():
    """Test that TokenBudgetEnforcer handles very tight budgets correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories with various token counts
    memories = [
        create_mock_memory("mem1", "content", token_count=5),   # Total: 5
        create_mock_memory("mem2", "content", token_count=3),   # Total: 8
        create_mock_memory("mem3", "content", token_count=1),   # Total: 9
        create_mock_memory("mem4", "content", token_count=2),   # Would exceed: 11 > 10
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 3 memories (total: 9 tokens)
    assert len(selected) == 3
    assert total_tokens == 9
    assert filtered_count == 1
    assert total_tokens <= 10


# Test memory count limit

def test_enforce_memory_count_limit():
    """Test that TokenBudgetEnforcer stops selecting memories when memory count limit is reached.
    
    **Validates: Requirements 8.2, 8.4**
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10000,  # High enough to not be limiting factor
        max_memory_count=3,
        token_estimator=estimator
    )
    
    # Create memories with small token counts
    memories = [
        create_mock_memory("mem1", "content", token_count=10),
        create_mock_memory("mem2", "content", token_count=10),
        create_mock_memory("mem3", "content", token_count=10),
        create_mock_memory("mem4", "content", token_count=10),
        create_mock_memory("mem5", "content", token_count=10),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select exactly 3 memories (memory count limit)
    assert len(selected) == 3
    assert total_tokens == 30
    assert filtered_count == 2
    assert selected[0].memory_id == "mem1"
    assert selected[1].memory_id == "mem2"
    assert selected[2].memory_id == "mem3"
    # Verify memory count constraint is satisfied
    assert len(selected) <= 3


def test_enforce_memory_count_limit_one():
    """Test that TokenBudgetEnforcer handles max_memory_count=1 correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10000,
        max_memory_count=1,
        token_estimator=estimator
    )
    
    # Create multiple memories
    memories = [
        create_mock_memory("mem1", "content", token_count=10),
        create_mock_memory("mem2", "content", token_count=10),
        create_mock_memory("mem3", "content", token_count=10),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select exactly 1 memory
    assert len(selected) == 1
    assert total_tokens == 10
    assert filtered_count == 2
    assert selected[0].memory_id == "mem1"


# Test both constraints simultaneously

def test_enforce_both_constraints_token_budget_limiting():
    """Test that TokenBudgetEnforcer applies both constraints when token budget is limiting factor.
    
    **Validates: Requirements 8.3**
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=10,  # Higher than what budget allows
        token_estimator=estimator
    )
    
    # Create memories where token budget is limiting
    memories = [
        create_mock_memory("mem1", "content", token_count=30),  # Total: 30
        create_mock_memory("mem2", "content", token_count=40),  # Total: 70
        create_mock_memory("mem3", "content", token_count=25),  # Total: 95
        create_mock_memory("mem4", "content", token_count=20),  # Would exceed budget
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Token budget is limiting factor (3 memories, 95 tokens)
    assert len(selected) == 3
    assert total_tokens == 95
    assert filtered_count == 1
    assert len(selected) <= 10  # Memory count constraint satisfied
    assert total_tokens <= 100  # Token budget constraint satisfied


def test_enforce_both_constraints_memory_count_limiting():
    """Test that TokenBudgetEnforcer applies both constraints when memory count is limiting factor.
    
    **Validates: Requirements 8.3, 8.4**
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10000,  # Higher than what memory count allows
        max_memory_count=3,
        token_estimator=estimator
    )
    
    # Create memories where memory count is limiting
    memories = [
        create_mock_memory("mem1", "content", token_count=10),
        create_mock_memory("mem2", "content", token_count=10),
        create_mock_memory("mem3", "content", token_count=10),
        create_mock_memory("mem4", "content", token_count=10),
        create_mock_memory("mem5", "content", token_count=10),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Memory count is limiting factor (3 memories, 30 tokens)
    assert len(selected) == 3
    assert total_tokens == 30
    assert filtered_count == 2
    assert len(selected) <= 3  # Memory count constraint satisfied
    assert total_tokens <= 10000  # Token budget constraint satisfied


def test_enforce_both_constraints_equal_limits():
    """Test that TokenBudgetEnforcer handles equal constraint limits correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=50,
        max_memory_count=2,
        token_estimator=estimator
    )
    
    # Create memories where both constraints are reached simultaneously
    memories = [
        create_mock_memory("mem1", "content", token_count=25),  # Total: 25, count: 1
        create_mock_memory("mem2", "content", token_count=25),  # Total: 50, count: 2 (both limits)
        create_mock_memory("mem3", "content", token_count=10),  # Would exceed both
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Both constraints are satisfied
    assert len(selected) == 2
    assert total_tokens == 50
    assert filtered_count == 1


# Test edge case: budget exhausted on first memory

def test_enforce_budget_exhausted_on_first_memory():
    """Test that TokenBudgetEnforcer handles case where first memory exceeds budget.
    
    **Validates: Requirements 2.3, 2.4**
    """
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=50,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories where first memory exceeds budget
    memories = [
        create_mock_memory("mem1", "content", token_count=100),  # Exceeds budget
        create_mock_memory("mem2", "content", token_count=10),   # Fits
        create_mock_memory("mem3", "content", token_count=10),   # Fits
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # First memory exceeds budget, but subsequent memories can still be selected
    # Should select mem2 and mem3 (total: 20 tokens)
    assert len(selected) == 2
    assert total_tokens == 20
    assert filtered_count == 1
    assert selected[0].memory_id == "mem2"
    assert selected[1].memory_id == "mem3"


def test_enforce_budget_exhausted_on_first_memory_exact_budget():
    """Test that TokenBudgetEnforcer selects first memory when it exactly matches budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories where first memory exactly matches budget
    memories = [
        create_mock_memory("mem1", "content", token_count=100),  # Exact match
        create_mock_memory("mem2", "content", token_count=10),   # Would exceed
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first memory (exact budget match)
    assert len(selected) == 1
    assert total_tokens == 100
    assert filtered_count == 1
    assert selected[0].memory_id == "mem1"


def test_enforce_budget_exhausted_on_first_memory_just_under():
    """Test that TokenBudgetEnforcer selects first memory when it's just under budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories where first memory is just under budget
    memories = [
        create_mock_memory("mem1", "content", token_count=99),  # Just under
        create_mock_memory("mem2", "content", token_count=1),   # Would reach 100 (exact fit)
        create_mock_memory("mem3", "content", token_count=1),   # Would exceed
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first two memories (total: 100 tokens)
    assert len(selected) == 2
    assert total_tokens == 100
    assert filtered_count == 1


# Test empty input

def test_enforce_empty_input():
    """Test that TokenBudgetEnforcer handles empty input list correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=10,
        token_estimator=estimator
    )
    
    selected, total_tokens, filtered_count = enforcer.enforce([])
    
    # Should return empty results
    assert len(selected) == 0
    assert total_tokens == 0
    assert filtered_count == 0


# Test single memory

def test_enforce_single_memory_within_budget():
    """Test that TokenBudgetEnforcer handles single memory within budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=10,
        token_estimator=estimator
    )
    
    memories = [
        create_mock_memory("mem1", "content", token_count=50),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select the single memory
    assert len(selected) == 1
    assert total_tokens == 50
    assert filtered_count == 0
    assert selected[0].memory_id == "mem1"


def test_enforce_single_memory_exceeds_budget():
    """Test that TokenBudgetEnforcer handles single memory exceeding budget."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=10,
        token_estimator=estimator
    )
    
    memories = [
        create_mock_memory("mem1", "content", token_count=150),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should not select the memory (exceeds budget)
    assert len(selected) == 0
    assert total_tokens == 0
    assert filtered_count == 1


# Test with word count approximation (no precomputed token_count)

def test_enforce_with_word_count_approximation():
    """Test that TokenBudgetEnforcer works with word count approximation when token_count is missing."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories without precomputed token_count (will use word count approximation)
    memories = [
        create_mock_memory("mem1", "one two"),           # 2 words × 1.3 = 2 tokens
        create_mock_memory("mem2", "three four five"),   # 3 words × 1.3 = 3 tokens
        create_mock_memory("mem3", "six seven eight"),   # 3 words × 1.3 = 3 tokens (total: 8)
        create_mock_memory("mem4", "nine ten eleven"),   # 3 words × 1.3 = 3 tokens (would exceed: 11 > 10)
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 3 memories (total: 8 tokens)
    assert len(selected) == 3
    assert total_tokens == 8
    assert filtered_count == 1
    assert total_tokens <= 10


# Test order preservation

def test_enforce_preserves_input_order():
    """Test that TokenBudgetEnforcer preserves the order of input memories."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories in specific order
    memories = [
        create_mock_memory("mem1", "content", token_count=10),
        create_mock_memory("mem2", "content", token_count=10),
        create_mock_memory("mem3", "content", token_count=10),
        create_mock_memory("mem4", "content", token_count=10),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should preserve order
    assert len(selected) == 4
    assert selected[0].memory_id == "mem1"
    assert selected[1].memory_id == "mem2"
    assert selected[2].memory_id == "mem3"
    assert selected[3].memory_id == "mem4"


# Test cumulative token tracking

def test_enforce_cumulative_token_tracking():
    """Test that TokenBudgetEnforcer correctly tracks cumulative token count."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=100,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories with varying token counts
    memories = [
        create_mock_memory("mem1", "content", token_count=15),  # Cumulative: 15
        create_mock_memory("mem2", "content", token_count=25),  # Cumulative: 40
        create_mock_memory("mem3", "content", token_count=35),  # Cumulative: 75
        create_mock_memory("mem4", "content", token_count=20),  # Cumulative: 95
        create_mock_memory("mem5", "content", token_count=10),  # Would be: 105 > 100
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 4 memories (cumulative: 95 tokens)
    assert len(selected) == 4
    assert total_tokens == 95
    assert filtered_count == 1
    # Verify cumulative tracking worked correctly
    assert total_tokens == 15 + 25 + 35 + 20


# Test with zero token memories

def test_enforce_with_zero_token_memories():
    """Test that TokenBudgetEnforcer handles memories with zero tokens correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=50,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories including some with zero tokens
    memories = [
        create_mock_memory("mem1", "content", token_count=0),   # Zero tokens
        create_mock_memory("mem2", "content", token_count=20),  # Total: 20
        create_mock_memory("mem3", "content", token_count=0),   # Total: 20
        create_mock_memory("mem4", "content", token_count=30),  # Total: 50
        create_mock_memory("mem5", "content", token_count=10),  # Would exceed
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 4 memories (total: 50 tokens)
    assert len(selected) == 4
    assert total_tokens == 50
    assert filtered_count == 1


# Test large token counts

def test_enforce_with_large_token_counts():
    """Test that TokenBudgetEnforcer handles large token counts correctly."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=1000000,  # 1 million tokens
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories with large token counts
    memories = [
        create_mock_memory("mem1", "content", token_count=250000),  # Total: 250k
        create_mock_memory("mem2", "content", token_count=250000),  # Total: 500k
        create_mock_memory("mem3", "content", token_count=250000),  # Total: 750k
        create_mock_memory("mem4", "content", token_count=250000),  # Total: 1M (exact fit)
        create_mock_memory("mem5", "content", token_count=100000),  # Would exceed
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # Should select first 4 memories (total: 1M tokens)
    assert len(selected) == 4
    assert total_tokens == 1000000
    assert filtered_count == 1


# Test filtered count accuracy

def test_enforce_filtered_count_all_filtered():
    """Test that filtered_count is accurate when all memories are filtered."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=10,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories that all exceed budget
    memories = [
        create_mock_memory("mem1", "content", token_count=100),
        create_mock_memory("mem2", "content", token_count=100),
        create_mock_memory("mem3", "content", token_count=100),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # All memories should be filtered
    assert len(selected) == 0
    assert total_tokens == 0
    assert filtered_count == 3


def test_enforce_filtered_count_none_filtered():
    """Test that filtered_count is accurate when no memories are filtered."""
    estimator = TokenEstimator(estimation_factor=1.3)
    enforcer = TokenBudgetEnforcer(
        max_token_budget=1000,
        max_memory_count=100,
        token_estimator=estimator
    )
    
    # Create memories that all fit within constraints
    memories = [
        create_mock_memory("mem1", "content", token_count=10),
        create_mock_memory("mem2", "content", token_count=10),
        create_mock_memory("mem3", "content", token_count=10),
    ]
    
    selected, total_tokens, filtered_count = enforcer.enforce(memories)
    
    # No memories should be filtered
    assert len(selected) == 3
    assert total_tokens == 30
    assert filtered_count == 0
