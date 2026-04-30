"""
Provider Interface for LLM Providers.

Defines the abstract base class that all LLM providers must implement,
and the ProviderError exception class for provider-specific errors.
"""

from abc import ABC, abstractmethod
from typing import Dict


class ProviderError(Exception):
    """
    Exception raised by providers when generation fails.
    
    Attributes:
        message: Descriptive error message
        is_transient: Whether the error is eligible for retry (e.g., network timeout, rate limit)
    """
    
    def __init__(self, message: str, is_transient: bool = False):
        """
        Initialize a ProviderError.
        
        Args:
            message: Descriptive error message
            is_transient: True if the error is transient (network timeout, HTTP 429, 5xx),
                         False if permanent (HTTP 400, 401, 403, invalid response)
        """
        super().__init__(message)
        self.is_transient = is_transient


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must subclass this interface and implement
    the generate() and validate_config() methods.
    """
    
    @abstractmethod
    def generate(self, prompt: str, options: Dict) -> Dict:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt string to send to the provider
            options: Generation parameters dictionary containing:
                - temperature (float): Sampling temperature (0.0-2.0)
                - max_tokens (int): Maximum tokens to generate
                - model (str, optional): Model name override
        
        Returns:
            Dictionary with keys:
                - text (str): Generated text
                - model (str): Model name used
                - prompt_tokens (int): Input tokens consumed
                - completion_tokens (int): Output tokens generated
                - provider (str): Provider identifier (e.g., "gemini", "openai")
        
        Raises:
            ProviderError: On API failure, timeout, or invalid response.
                          Set is_transient=True for retryable errors (network, rate limit).
        """
        ...
    
    @abstractmethod
    def validate_config(self, config: Dict) -> bool:
        """
        Validate that required configuration fields are present.
        
        Args:
            config: Provider-specific configuration dictionary
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        ...
