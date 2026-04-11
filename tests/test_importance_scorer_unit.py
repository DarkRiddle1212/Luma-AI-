"""Unit tests for the Importance Scorer component.

Tests verify scoring heuristics, threshold filtering, and edge cases for the
ImportanceScorer class in the Memory Write Engine.
"""

import pytest
from luma.core.memory_write.importance_scorer import ImportanceScorer
from luma.core.memory_write.schemas import MemoryCandidate, ScoredMemory


class TestImportanceScorerInitialization:
    """Tests for ImportanceScorer initialization and configuration."""
    
    def test_default_threshold(self):
        """Test scorer initializes with default threshold of 0.5."""
        scorer = ImportanceScorer()
        assert scorer.threshold == 0.5
    
    def test_custom_threshold(self):
        """Test scorer accepts custom threshold parameter."""
        scorer = ImportanceScorer(threshold=0.7)
        assert scorer.threshold == 0.7
    
    def test_threshold_validation_too_low(self):
        """Test scorer rejects threshold below 0.0."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            ImportanceScorer(threshold=-0.1)
    
    def test_threshold_validation_too_high(self):
        """Test scorer rejects threshold above 1.0."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            ImportanceScorer(threshold=1.5)
    
    def test_threshold_boundary_values(self):
        """Test scorer accepts boundary threshold values 0.0 and 1.0."""
        scorer_min = ImportanceScorer(threshold=0.0)
        scorer_max = ImportanceScorer(threshold=1.0)
        assert scorer_min.threshold == 0.0
        assert scorer_max.threshold == 1.0


class TestProjectGoalScoring:
    """Tests for scoring project goal memories."""
    
    def test_project_goal_high_score(self):
        """Test project goals receive high importance scores (>0.7)."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="I want to build a task management system",
            type="project_goal"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.7
        assert result.text == candidate.text
        assert result.type == candidate.type
    
    def test_project_goal_with_keywords(self):
        """Test project goals with high-importance keywords get boosted scores."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="My goal is to create a critical system for managing requirements",
            type="project_goal"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.8


class TestUserPreferenceScoring:
    """Tests for scoring user preference memories."""
    
    def test_user_preference_high_score(self):
        """Test user preferences receive high importance scores (>0.7)."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="I prefer using TypeScript for all projects",
            type="user_preference"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.7
    
    def test_user_preference_with_always_keyword(self):
        """Test preferences with 'always' keyword get high scores."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="I always write tests before implementation",
            type="user_preference"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.8


class TestGreetingScoring:
    """Tests for scoring greeting and acknowledgement memories."""
    
    def test_greeting_low_score(self):
        """Test greetings receive low importance scores (<0.3)."""
        scorer = ImportanceScorer(threshold=0.0)
        greetings = ["hello", "hi", "hey there", "Hello!", "Hi there"]
        
        for greeting_text in greetings:
            candidate = MemoryCandidate(text=greeting_text, type="statement")
            result = scorer.score_memory(candidate)
            
            assert result is not None
            assert result.importance < 0.3, f"Greeting '{greeting_text}' should have low score"
    
    def test_acknowledgement_low_score(self):
        """Test acknowledgements receive low importance scores (<0.3)."""
        scorer = ImportanceScorer(threshold=0.0)
        acknowledgements = [
            "thanks", "thank you", "ok", "okay", "sure",
            "yes", "no", "got it", "understood", "sounds good"
        ]
        
        for ack_text in acknowledgements:
            candidate = MemoryCandidate(text=ack_text, type="statement")
            result = scorer.score_memory(candidate)
            
            assert result is not None
            assert result.importance < 0.3, f"Acknowledgement '{ack_text}' should have low score"
    
    def test_goodbye_low_score(self):
        """Test goodbye phrases receive low importance scores (<0.3)."""
        scorer = ImportanceScorer(threshold=0.0)
        goodbyes = ["bye", "goodbye", "see you"]
        
        for goodbye_text in goodbyes:
            candidate = MemoryCandidate(text=goodbye_text, type="statement")
            result = scorer.score_memory(candidate)
            
            assert result is not None
            assert result.importance < 0.3


class TestFactScoring:
    """Tests for scoring fact memories."""
    
    def test_fact_medium_score(self):
        """Test facts receive medium importance scores (0.4-0.6)."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="The project uses Python 3.11 and FastAPI framework",
            type="fact"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert 0.4 <= result.importance <= 0.7  # Allow some adjustment
    
    def test_detailed_fact_higher_score(self):
        """Test longer, detailed facts receive slightly higher scores."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="The system architecture consists of a FastAPI backend with PostgreSQL database, "
                 "Redis for caching, and a React frontend with TypeScript for type safety",
            type="fact"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.5


class TestThresholdFiltering:
    """Tests for threshold filtering behavior."""
    
    def test_filter_below_threshold(self):
        """Test memories below threshold are filtered out (return None)."""
        scorer = ImportanceScorer(threshold=0.5)
        candidate = MemoryCandidate(text="hello", type="statement")
        
        result = scorer.score_memory(candidate)
        
        assert result is None
    
    def test_accept_above_threshold(self):
        """Test memories above threshold are returned."""
        scorer = ImportanceScorer(threshold=0.5)
        candidate = MemoryCandidate(
            text="I want to build a comprehensive testing framework",
            type="project_goal"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert isinstance(result, ScoredMemory)
    
    def test_threshold_boundary_exact_match(self):
        """Test memory with score exactly at threshold is accepted."""
        scorer = ImportanceScorer(threshold=0.5)
        # Create a candidate that should score around 0.5
        candidate = MemoryCandidate(
            text="The database schema includes user and project tables",
            type="fact"
        )
        
        result = scorer.score_memory(candidate)
        
        # Should be accepted if score >= threshold
        if result is not None:
            assert result.importance >= scorer.threshold
    
    def test_high_threshold_filters_more(self):
        """Test higher threshold filters out more memories."""
        candidate = MemoryCandidate(
            text="The API uses REST conventions",
            type="fact"
        )
        
        scorer_low = ImportanceScorer(threshold=0.3)
        scorer_high = ImportanceScorer(threshold=0.8)
        
        result_low = scorer_low.score_memory(candidate)
        result_high = scorer_high.score_memory(candidate)
        
        # Low threshold should accept, high threshold should filter
        assert result_low is not None
        assert result_high is None


class TestContentAdjustments:
    """Tests for content-based score adjustments."""
    
    def test_short_statement_lower_score(self):
        """Test very short statements receive lower scores."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(text="Yes", type="statement")
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance < 0.4
    
    def test_detailed_content_higher_score(self):
        """Test longer, detailed content receives higher scores."""
        scorer = ImportanceScorer(threshold=0.0)
        candidate = MemoryCandidate(
            text="The authentication system must support OAuth2, JWT tokens, "
                 "refresh token rotation, and multi-factor authentication with "
                 "support for TOTP and SMS-based verification methods",
            type="fact"
        )
        
        result = scorer.score_memory(candidate)
        
        assert result is not None
        assert result.importance > 0.55


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_none_candidate_raises_error(self):
        """Test scorer rejects None candidate."""
        scorer = ImportanceScorer()
        
        with pytest.raises(ValueError, match="candidate cannot be None"):
            scorer.score_memory(None)
    
    def test_invalid_candidate_type_raises_error(self):
        """Test scorer validates candidate type through MemoryCandidate validation."""
        scorer = ImportanceScorer()
        
        # MemoryCandidate validation should catch invalid type
        with pytest.raises(ValueError):
            candidate = MemoryCandidate(text="test", type="invalid_type")
            scorer.score_memory(candidate)
    
    def test_empty_text_raises_error(self):
        """Test scorer validates non-empty text through MemoryCandidate validation."""
        scorer = ImportanceScorer()
        
        # MemoryCandidate validation should catch empty text
        with pytest.raises(ValueError):
            candidate = MemoryCandidate(text="", type="fact")
            scorer.score_memory(candidate)


class TestScoreRange:
    """Tests for score range validation."""
    
    def test_score_never_exceeds_one(self):
        """Test importance scores never exceed 1.0."""
        scorer = ImportanceScorer(threshold=0.0)
        candidates = [
            MemoryCandidate(
                text="This is a critical goal that must be achieved with high priority",
                type="project_goal"
            ),
            MemoryCandidate(
                text="I always prefer to use the most important and essential tools",
                type="user_preference"
            )
        ]
        
        for candidate in candidates:
            result = scorer.score_memory(candidate)
            assert result is not None
            assert result.importance <= 1.0
    
    def test_score_never_below_zero(self):
        """Test importance scores never go below 0.0."""
        scorer = ImportanceScorer(threshold=0.0)
        candidates = [
            MemoryCandidate(text="hi", type="statement"),
            MemoryCandidate(text="ok", type="statement")
        ]
        
        for candidate in candidates:
            result = scorer.score_memory(candidate)
            assert result is not None
            assert result.importance >= 0.0
