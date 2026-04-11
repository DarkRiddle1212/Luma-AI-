"""Importance scoring component for the Memory Write Engine.

This module evaluates candidate memories and assigns importance scores based on
memory type and content patterns. Memories below the configured threshold are
filtered out before storage.
"""

from typing import Optional
from .schemas import MemoryCandidate, ScoredMemory


class ImportanceScorer:
    """Evaluates importance of candidate memories and filters by threshold.
    
    The scorer assigns importance scores (0.0 to 1.0) based on memory type and
    content patterns, then filters out memories below the configured threshold.
    
    Scoring heuristics:
    - High importance (0.7-1.0): Project goals, user preferences, personal info
    - Medium importance (0.4-0.6): Technical facts, project details
    - Low importance (0.0-0.3): Greetings, acknowledgements, temporary statements
    
    Attributes:
        threshold: Minimum importance score (0.0 to 1.0) for storage
    """
    
    def __init__(self, threshold: float = 0.5):
        """Initialize with importance threshold.
        
        Args:
            threshold: Minimum importance score (0.0 to 1.0) for storage
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
    
    def score_memory(self, candidate: MemoryCandidate) -> Optional[ScoredMemory]:
        """Score a candidate memory and filter by threshold.
        
        Assigns an importance score based on memory type and content patterns.
        Returns None if the score is below the configured threshold.
        
        Args:
            candidate: MemoryCandidate to score
            
        Returns:
            ScoredMemory if above threshold, None otherwise
            
        Raises:
            ValueError: If candidate is None or has invalid fields
        """
        if candidate is None:
            raise ValueError("candidate cannot be None")
        
        # Calculate base score from memory type
        base_score = self._get_base_score(candidate.type)
        
        # Adjust score based on content patterns
        adjusted_score = self._adjust_score_by_content(base_score, candidate.text)
        
        # Ensure score is within valid range
        final_score = max(0.0, min(1.0, adjusted_score))
        
        # Filter by threshold
        if final_score < self.threshold:
            return None
        
        return ScoredMemory(
            text=candidate.text,
            type=candidate.type,
            importance=final_score
        )
    
    def _get_base_score(self, memory_type: str) -> float:
        """Get base importance score for memory type.
        
        Args:
            memory_type: Type of memory (project_goal, user_preference, fact, statement)
            
        Returns:
            Base importance score (0.0 to 1.0)
        """
        base_scores = {
            "project_goal": 0.85,      # High importance
            "user_preference": 0.80,   # High importance
            "fact": 0.55,              # Medium importance
            "statement": 0.45          # Medium-low importance
        }
        return base_scores.get(memory_type, 0.5)
    
    def _adjust_score_by_content(self, base_score: float, text: str) -> float:
        """Adjust score based on content patterns.
        
        Analyzes the text content for patterns that indicate higher or lower
        importance, such as greetings, acknowledgements, or specific keywords.
        
        Args:
            base_score: Initial score based on memory type
            text: Memory content text
            
        Returns:
            Adjusted importance score
        """
        text_lower = text.lower().strip()
        
        # Low importance patterns (greetings, acknowledgements)
        low_importance_patterns = [
            "hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
            "sure", "yes", "no", "got it", "understood", "sounds good",
            "great", "cool", "nice", "bye", "goodbye", "see you"
        ]
        
        # Check if text is primarily a low-importance phrase
        if any(text_lower == pattern or text_lower.startswith(pattern + " ") 
               for pattern in low_importance_patterns):
            return 0.2  # Very low importance
        
        # High importance keywords (goals, preferences, constraints)
        high_importance_keywords = [
            "goal", "objective", "want to", "need to", "must", "require",
            "prefer", "always", "never", "important", "critical", "essential",
            "constraint", "requirement", "decision", "plan", "strategy"
        ]
        
        # Boost score if high-importance keywords are present
        if any(keyword in text_lower for keyword in high_importance_keywords):
            return min(1.0, base_score + 0.1)
        
        # Check for very short statements (likely low importance)
        if len(text.split()) <= 3:
            return max(0.2, base_score - 0.2)
        
        # Check for longer, detailed content (likely higher importance)
        if len(text.split()) > 20:
            return min(1.0, base_score + 0.05)
        
        return base_score
