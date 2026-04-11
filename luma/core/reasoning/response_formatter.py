"""
Response Formatter for the Reasoning Engine.

This module provides the Response_Formatter class that transforms raw LLM output
into structured Reasoning_Result objects with answer, used memories, and confidence.
"""

import re
from typing import List, Optional
from .schemas import Reasoning_Result


class Response_Formatter:
    """
    Component that transforms raw LLM output into structured result objects.
    
    The Response_Formatter parses LLM responses to extract:
    - Answer text
    - Used memory identifiers
    - Optional confidence score
    
    When parsing fails, it gracefully returns the raw output as the answer.
    
    Example:
        >>> formatter = Response_Formatter()
        >>> raw_output = "Answer: Python is great\\nUsed Memories: mem_1, mem_2\\nConfidence: 0.95"
        >>> result = formatter.format_response(raw_output)
        >>> result.answer
        'Python is great'
        >>> result.used_memories
        ['mem_1', 'mem_2']
        >>> result.confidence
        0.95
    """
    
    def format_response(self, raw_output: str) -> Reasoning_Result:
        """
        Format raw LLM output into a structured Reasoning_Result.
        
        This method attempts to parse the LLM output to extract structured information.
        If parsing fails, it returns a Reasoning_Result with the raw output as the answer.
        
        Expected LLM output format (flexible):
        - Answer section containing the response text
        - Used Memories section with comma-separated memory IDs
        - Optional Confidence section with a float value
        
        Args:
            raw_output (str): The raw string output from the LLM.
        
        Returns:
            Reasoning_Result: Structured result with answer, used_memories, and confidence.
        
        Examples:
            >>> formatter = Response_Formatter()
            >>> result = formatter.format_response("Answer: Hello\\nUsed Memories: mem_1")
            >>> result.answer
            'Hello'
            
            >>> result = formatter.format_response("Unparseable output")
            >>> result.answer
            'Unparseable output'
            >>> result.used_memories
            []
        """
        try:
            answer = self._extract_answer(raw_output)
            used_memories = self._extract_used_memories(raw_output)
            confidence = self._extract_confidence(raw_output)
            
            return Reasoning_Result(
                answer=answer,
                used_memories=used_memories,
                confidence=confidence
            )
        except Exception:
            # Gracefully handle unparseable output
            return Reasoning_Result(
                answer=raw_output,
                used_memories=[],
                confidence=None
            )
    
    def _extract_answer(self, raw_output: str) -> str:
        """
        Extract the answer text from LLM output.
        
        Looks for patterns like:
        - "Answer: <text>"
        - "Response: <text>"
        - Or returns the entire output if no pattern matches
        
        Args:
            raw_output (str): The raw LLM output.
        
        Returns:
            str: The extracted answer text.
        """
        # Try to find answer section with various patterns
        answer_patterns = [
            r'(?:Answer|Response):\s*(.+?)(?:\n(?:Used Memories|Confidence|$))',
            r'(?:Answer|Response):\s*(.+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, check if there are any structured sections
        # If not, return the entire output as the answer
        if not re.search(r'(?:Used Memories|Confidence):', raw_output, re.IGNORECASE):
            return raw_output.strip()
        
        # If there are structured sections but no answer section, raise to trigger fallback
        raise ValueError("Could not extract answer from structured output")
    
    def _extract_used_memories(self, raw_output: str) -> List[str]:
        """
        Extract used memory identifiers from LLM output.
        
        Looks for patterns like:
        - "Used Memories: mem_1, mem_2, mem_3"
        - "Memories Used: [mem_1, mem_2]"
        
        Args:
            raw_output (str): The raw LLM output.
        
        Returns:
            List[str]: List of memory identifiers, or empty list if none found.
        """
        # Try to find used memories section
        memory_patterns = [
            r'(?:Used Memories|Memories Used):\s*\[?([^\]\n]+)\]?',
            r'(?:Used Memories|Memories Used):\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in memory_patterns:
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                memories_str = match.group(1).strip()
                # Split by comma and clean up whitespace
                memories = [m.strip() for m in memories_str.split(',') if m.strip()]
                # Filter out empty strings and "none" values
                memories = [m for m in memories if m and m.lower() != 'none']
                return memories
        
        return []
    
    def _extract_confidence(self, raw_output: str) -> Optional[float]:
        """
        Extract confidence score from LLM output.
        
        Looks for patterns like:
        - "Confidence: 0.95"
        - "Confidence Score: 0.8"
        
        Args:
            raw_output (str): The raw LLM output.
        
        Returns:
            Optional[float]: Confidence score between 0.0 and 1.0, or None if not found.
        """
        # Try to find confidence section
        confidence_pattern = r'(?:Confidence|Confidence Score):\s*([0-9]*\.?[0-9]+)'
        
        match = re.search(confidence_pattern, raw_output, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                # Ensure confidence is between 0.0 and 1.0
                if 0.0 <= confidence <= 1.0:
                    return confidence
            except ValueError:
                pass
        
        return None


__all__ = ['Response_Formatter']
