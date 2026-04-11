"""
LLM Interface Module

This module defines the abstract interface for language model implementations
and provides a stub implementation for testing and development.

The LLMInterface allows swapping between different LLM providers (local models,
cloud APIs, stubs) without changing orchestration code.
"""

from abc import ABC, abstractmethod
from typing import Dict


class LLMInterface(ABC):
    """
    Abstract interface for language model implementations.
    
    This interface allows swapping between different LLM providers
    (local models, cloud APIs, stubs) without changing orchestration code.
    
    All LLM implementations must inherit from this class and implement
    the generate_response method.
    
    Example:
        >>> class CustomLLM(LLMInterface):
        ...     def generate_response(self, prompt: str, context: Dict) -> str:
        ...         return f"Custom response to: {prompt}"
        >>> 
        >>> llm = CustomLLM()
        >>> response = llm.generate_response("Hello", {"intent": "general"})
    """
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict) -> str:
        """
        Generate a response given a prompt and context.
        
        This method must be implemented by all concrete LLM classes.
        It takes a formatted prompt and context dictionary, and returns
        a generated response string.
        
        Args:
            prompt: The formatted input text for the LLM. This is the
                   primary text that the LLM should respond to.
            context: Dictionary containing relevant information for response
                    generation. May include user_message, timestamp, intent,
                    memory data, system state, and other metadata.
            
        Returns:
            str: The generated response text from the language model.
            
        Raises:
            Exception: Implementation-specific exceptions may be raised
                      if response generation fails (e.g., API errors,
                      timeout, invalid input).
        
        Example:
            >>> llm = StubLLM()
            >>> context = {"intent": "general", "user_message": "Hello"}
            >>> response = llm.generate_response("Hello", context)
            >>> print(response)
        """
        pass


class StubLLM(LLMInterface):
    """
    Stub implementation of LLM interface for testing and development.
    
    This class simulates intelligent responses without actual AI computation.
    It echoes back the received inputs in a formatted string, making it
    useful for:
    - Early development and testing
    - CI/CD pipelines without AI dependencies
    - Demonstrating system architecture
    - Unit testing orchestration logic
    
    The stub response includes the detected intent, the prompt text,
    and the context keys to verify that data flows correctly through
    the system.
    
    Example:
        >>> from luma.core.llm_interface import StubLLM
        >>> 
        >>> llm = StubLLM()
        >>> context = {
        ...     "intent": "education",
        ...     "user_message": "Teach me Python",
        ...     "timestamp": "2024-01-15T10:30:00"
        ... }
        >>> response = llm.generate_response("Teach me Python", context)
        >>> print(response)
        [StubLLM Response]
        Intent: education
        Prompt: 'Teach me Python'
        Context Keys: ['intent', 'user_message', 'timestamp']
        ---
        This is a simulated response. Replace StubLLM with a real LLM implementation.
    """
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        """
        Generate a simulated response echoing inputs.
        
        Creates a formatted string that includes the intent, prompt,
        and context keys to demonstrate that the LLM interface received
        the correct inputs.
        
        Args:
            prompt: The input prompt text
            context: The context dictionary containing metadata
            
        Returns:
            str: Formatted stub response showing received inputs
            
        Example:
            >>> llm = StubLLM()
            >>> response = llm.generate_response(
            ...     "Hello",
            ...     {"intent": "general", "user_message": "Hello"}
            ... )
            >>> "StubLLM Response" in response
            True
        """
        # Extract context information for the stub response
        context_keys = list(context.keys())  # Get all keys to show what data was received
        intent = context.get("intent", "unknown")  # Get intent, default to "unknown" if not present
        
        # Format a response that demonstrates the stub received all inputs correctly
        # This helps verify data flow through the system during testing
        return (
            f"[StubLLM Response]\n"
            f"Intent: {intent}\n"
            f"Prompt: '{prompt}'\n"
            f"Context Keys: {context_keys}\n"
            f"---\n"
            f"This is a simulated response. Replace StubLLM with a real LLM implementation."
        )
