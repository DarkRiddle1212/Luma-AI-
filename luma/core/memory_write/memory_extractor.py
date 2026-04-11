"""Memory extraction component for the Memory Write Engine.

This module analyzes conversation interactions to identify candidate memories
that should be considered for long-term storage.
"""

import re
from typing import List, Optional
from .schemas import MemoryCandidate, MemoryType


class MemoryExtractor:
    """Extracts candidate memories from conversation interactions.
    
    Analyzes both user queries and system responses to identify information
    worth storing as long-term memory. Classifies extracted memories by type
    (project goals, user preferences, facts, or statements).
    """
    
    # Pattern definitions for memory extraction
    PROJECT_GOAL_PATTERNS = [
        r"(?:I want to|I'd like to|I need to|My goal is to|I'm trying to|I plan to)\s+(.+?)(?:\.|$)",
        r"(?:I'm building|I'm creating|I'm developing|I'm working on)\s+(.+?)(?:\.|$)",
        r"(?:The goal is to|The objective is to|The aim is to)\s+(.+?)(?:\.|$)",
    ]
    
    USER_PREFERENCE_PATTERNS = [
        r"(?:I prefer|I like|I love|I enjoy|I always|I usually|I typically)\s+(.+?)(?:\.|$)",
        r"(?:I don't like|I dislike|I hate|I avoid|I never)\s+(.+?)(?:\.|$)",
        r"(?:My preference is|My style is)\s+(.+?)(?:\.|$)",
    ]
    
    FACT_PATTERNS = [
        r"(?:I am|I'm)\s+(?:a|an)\s+(.+?)(?:\.|$)",
        r"(?:My name is|I work at|I live in|I use)\s+(.+?)(?:\.|$)",
        r"(?:The project is|This is|This project is)\s+(.+?)(?:\.|$)",
    ]
    
    # Keywords that indicate low-value content (greetings, acknowledgements)
    LOW_VALUE_KEYWORDS = [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay",
        "sure", "yes", "no", "bye", "goodbye", "see you"
    ]
    
    def extract_candidates(
        self,
        user_query: str,
        system_response: str
    ) -> List[MemoryCandidate]:
        """Extract candidate memories from an interaction.
        
        Analyzes both the user query and system response to identify information
        worth storing as long-term memory. Returns an empty list if inputs are
        invalid or no extractable information is found.
        
        Args:
            user_query: The user's query text
            system_response: The system's response text
            
        Returns:
            List of MemoryCandidate objects with text and type classification
        """
        # Handle empty/None inputs gracefully
        if not user_query or not system_response:
            return []
        
        # Normalize inputs
        user_query = user_query.strip()
        system_response = system_response.strip()
        
        if not user_query or not system_response:
            return []
        
        candidates: List[MemoryCandidate] = []
        
        # Extract from user query
        candidates.extend(self._extract_from_text(user_query, source="user"))
        
        # Extract from system response
        candidates.extend(self._extract_from_text(system_response, source="system"))
        
        # Filter out low-value content
        candidates = self._filter_low_value(candidates)
        
        return candidates
    
    def _extract_from_text(
        self,
        text: str,
        source: str
    ) -> List[MemoryCandidate]:
        """Extract memories from a single text (user query or system response).
        
        Args:
            text: The text to analyze
            source: Either "user" or "system" to indicate the source
            
        Returns:
            List of MemoryCandidate objects extracted from the text
        """
        candidates: List[MemoryCandidate] = []
        
        # Try to extract project goals
        for pattern in self.PROJECT_GOAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted_text = match.group(1).strip()
                if extracted_text and len(extracted_text) > 5:  # Minimum length filter
                    # Reconstruct full sentence for context
                    full_text = match.group(0).strip()
                    candidates.append(MemoryCandidate(
                        text=full_text,
                        type="project_goal"
                    ))
        
        # Try to extract user preferences
        for pattern in self.USER_PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted_text = match.group(1).strip()
                if extracted_text and len(extracted_text) > 5:
                    full_text = match.group(0).strip()
                    candidates.append(MemoryCandidate(
                        text=full_text,
                        type="user_preference"
                    ))
        
        # Try to extract facts
        for pattern in self.FACT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                extracted_text = match.group(1).strip()
                if extracted_text and len(extracted_text) > 3:
                    full_text = match.group(0).strip()
                    candidates.append(MemoryCandidate(
                        text=full_text,
                        type="fact"
                    ))
        
        # Extract important statements (sentences that don't match other patterns)
        # Look for declarative sentences that might be important
        if source == "user":
            sentences = self._split_sentences(text)
            for sentence in sentences:
                # Skip if already extracted by other patterns
                if any(candidate.text in sentence for candidate in candidates):
                    continue
                
                # Check if sentence looks important (contains key decision words)
                if self._is_important_statement(sentence):
                    candidates.append(MemoryCandidate(
                        text=sentence.strip(),
                        type="statement"
                    ))
        
        return candidates
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: The text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_important_statement(self, sentence: str) -> bool:
        """Check if a sentence is an important statement worth storing.
        
        Args:
            sentence: The sentence to evaluate
            
        Returns:
            True if the sentence appears to be important, False otherwise
        """
        # Minimum length requirement
        if len(sentence) < 10:
            return False
        
        # Keywords that indicate important statements
        important_keywords = [
            "must", "should", "need", "require", "important", "critical",
            "constraint", "requirement", "decision", "choose", "selected"
        ]
        
        sentence_lower = sentence.lower()
        
        # Check for important keywords
        if any(keyword in sentence_lower for keyword in important_keywords):
            return True
        
        # Check if it's a declarative statement (not a question)
        if sentence.strip().endswith("?"):
            return False
        
        return False
    
    def _filter_low_value(
        self,
        candidates: List[MemoryCandidate]
    ) -> List[MemoryCandidate]:
        """Filter out low-value content like greetings and acknowledgements.
        
        Args:
            candidates: List of candidate memories to filter
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            text_lower = candidate.text.lower()
            
            # Check if the entire text is just a low-value keyword
            if text_lower.strip() in self.LOW_VALUE_KEYWORDS:
                continue
            
            # Check if text is very short and contains only low-value keywords
            words = text_lower.split()
            if len(words) <= 3 and all(word in self.LOW_VALUE_KEYWORDS for word in words):
                continue
            
            filtered.append(candidate)
        
        return filtered
