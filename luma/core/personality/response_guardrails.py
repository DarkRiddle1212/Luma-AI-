"""
Response Guardrails Component.

Validates LLM responses against quality constraints and detects weak responses
(rambling, repetition, contradiction, vague filler). Stateless and deterministic.
"""

import re
from typing import List

from luma.core.personality.schemas import GuardrailResult


class ResponseGuardrails:
    """
    Stateless component that validates LLM responses against quality constraints.
    
    Checks for:
    - Rambling: >500 words without clear structure
    - Repetition: >2 occurrences of identical 5+ word sequences
    - Contradiction: presence of contradictory phrases
    - Vague filler: >3 filler phrases without concrete examples
    - Length constraints: e.g., "concise" → max 200 words
    """

    def validate(
        self,
        response_text: str,
        constraints: List[str],
    ) -> GuardrailResult:
        """
        Validate a response against quality constraints.
        
        Args:
            response_text: The LLM response text to validate
            constraints: List of constraint strings (e.g., ["concise"])
        
        Returns:
            GuardrailResult with passed status, violations list, score, and notes
        """
        # Handle empty response
        stripped_text = response_text.strip()
        if not stripped_text:
            return GuardrailResult(
                passed=False,
                violations=["empty response"],
                score=0.0,
                notes="Response is empty after stripping whitespace",
            )
        
        violations: List[str] = []
        
        # Check for rambling (>500 words without clear structure)
        if self._check_rambling(stripped_text):
            violations.append("rambling")
        
        # Check for repetition (>2 occurrences of identical 5+ word sequences)
        if self._check_repetition(stripped_text):
            violations.append("repetition")
        
        # Check for contradiction
        if self._check_contradiction(stripped_text):
            violations.append("contradiction")
        
        # Check for vague filler (>3 filler phrases without concrete examples)
        if self._check_vague_filler(stripped_text):
            violations.append("vague filler")
        
        # Check length constraint
        if "concise" in constraints:
            word_count = len(stripped_text.split())
            if word_count > 200:
                violations.append("exceeds concise length constraint")
        
        # Calculate score: 1.0 - (violation_count * 0.25), clamped to [0.0, 1.0]
        score = max(0.0, min(1.0, 1.0 - (len(violations) * 0.25)))
        
        # Determine pass/fail: passed if score >= 0.75 (0-1 violations)
        passed = score >= 0.75
        
        # Generate notes
        if violations:
            notes = f"Detected {len(violations)} violation(s): {', '.join(violations)}"
        else:
            notes = "Response passed all quality checks"
        
        return GuardrailResult(
            passed=passed,
            violations=violations,
            score=score,
            notes=notes,
        )
    
    def _check_rambling(self, text: str) -> bool:
        """
        Check if response is rambling (>500 words without clear structure).
        
        Heuristic: Count words and check for structural markers (bullet points,
        numbered lists, clear paragraphs).
        """
        words = text.split()
        word_count = len(words)
        
        if word_count <= 500:
            return False
        
        # Check for structural markers
        has_bullets = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))
        has_paragraphs = text.count('\n\n') >= 2
        
        # If >500 words and no clear structure, it's rambling
        return not (has_bullets or has_numbers or has_paragraphs)
    
    def _check_repetition(self, text: str) -> bool:
        """
        Check for repetition (>2 occurrences of identical 5+ word sequences).
        
        Heuristic: Extract all 5-word sequences and count occurrences.
        """
        words = text.lower().split()
        
        if len(words) < 5:
            return False
        
        # Extract all 5-word sequences
        sequences = []
        for i in range(len(words) - 4):
            sequence = ' '.join(words[i:i+5])
            sequences.append(sequence)
        
        # Count occurrences of each sequence
        sequence_counts = {}
        for seq in sequences:
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
        
        # Check if any sequence appears more than 2 times
        for count in sequence_counts.values():
            if count > 2:
                return True
        
        return False
    
    def _check_contradiction(self, text: str) -> bool:
        """
        Check for contradiction (presence of contradictory phrases).
        
        Heuristic: Look for phrases like "but actually", "on the other hand"
        followed by contradictory claims.
        """
        contradiction_markers = [
            r'but actually',
            r'on the other hand',
            r'however,?\s+(?:the|this|that)',
            r'in contrast',
            r'conversely',
        ]
        
        text_lower = text.lower()
        
        for marker in contradiction_markers:
            if re.search(marker, text_lower):
                return True
        
        return False
    
    def _check_vague_filler(self, text: str) -> bool:
        """
        Check for vague filler (>3 filler phrases without concrete examples).
        
        Heuristic: Count filler phrases and check for concrete examples
        (numbers, specific terms, code snippets).
        """
        filler_phrases = [
            r'it depends',
            r'generally speaking',
            r'in most cases',
            r'typically',
            r'usually',
            r'often',
            r'sometimes',
            r'may or may not',
            r'could be',
            r'might be',
        ]
        
        text_lower = text.lower()
        
        # Count filler phrases
        filler_count = 0
        for phrase in filler_phrases:
            filler_count += len(re.findall(phrase, text_lower))
        
        if filler_count <= 3:
            return False
        
        # Check for concrete examples
        has_numbers = bool(re.search(r'\d+', text))
        has_code = bool(re.search(r'`[^`]+`', text))
        has_specific_terms = bool(re.search(r'\b[A-Z][a-z]+[A-Z]\w*\b', text))  # CamelCase
        
        # If >3 filler phrases and no concrete examples, it's vague
        return not (has_numbers or has_code or has_specific_terms)
