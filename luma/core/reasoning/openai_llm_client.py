"""
OpenAI LLM Client Implementation.

This module provides a concrete implementation of the LLM_Client_Interface
for the OpenAI API. It handles API authentication, request formatting,
response parsing, and error handling for OpenAI's chat completion models.

Requirements:
- 3.3: OpenAI_LLM_Client implements LLM_Client_Interface
- 3.4: Sends prompts to OpenAI API
- 3.5: Returns model response as string
- 10.4: Supports cloud API implementations
"""

import os
from typing import Optional
from luma.core.reasoning.llm_client_interface import LLM_Client_Interface


class OpenAI_LLM_Client(LLM_Client_Interface):
    """
    OpenAI API implementation of the LLM_Client_Interface.
    
    This client sends prompts to OpenAI's chat completion API and returns
    the generated responses. It handles API key configuration, request
    formatting, and error handling specific to the OpenAI platform.
    
    The client uses the chat completions endpoint with configurable model
    and generation parameters. By default, it uses gpt-3.5-turbo for
    cost-effective reasoning operations.
    
    Attributes:
        api_key (str): OpenAI API key for authentication
        model (str): Model identifier (default: gpt-3.5-turbo)
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (Optional[int]): Maximum tokens in response (default: None)
    
    Example:
        >>> client = OpenAI_LLM_Client(api_key="sk-...")
        >>> response = client.generate("What is Python?")
        >>> print(response)
        "Python is a high-level programming language..."
        
        >>> # Using environment variable for API key
        >>> os.environ['OPENAI_API_KEY'] = 'sk-...'
        >>> client = OpenAI_LLM_Client()
        >>> response = client.generate("Hello!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the OpenAI LLM client.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will
                                    attempt to read from OPENAI_API_KEY
                                    environment variable.
            model (str): OpenAI model identifier. Defaults to gpt-3.5-turbo.
            temperature (float): Sampling temperature between 0.0 and 2.0.
                               Higher values make output more random.
                               Defaults to 0.7.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
                                       If None, uses model's default.
        
        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY
                       environment variable is not set.
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as a parameter "
                "or via the OPENAI_API_KEY environment variable"
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import OpenAI library here to avoid import errors if not installed
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use OpenAI_LLM_Client. "
                "Install it with: pip install openai"
            )
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from OpenAI's API for the given prompt.
        
        This method sends the prompt to OpenAI's chat completion endpoint
        and returns the generated response. The prompt is formatted as a
        user message in the chat format.
        
        Args:
            prompt (str): The input prompt to send to the language model.
                         This typically includes system instructions, context,
                         and the user's query formatted by the Prompt_Builder.
        
        Returns:
            str: The generated response from the OpenAI model.
        
        Raises:
            ValueError: If the prompt is empty or invalid.
            Exception: For API errors including authentication failures,
                      rate limits, network issues, or invalid requests.
                      The original OpenAI exception is re-raised with context.
        
        Example:
            >>> client = OpenAI_LLM_Client(api_key="sk-...")
            >>> prompt = "System: You are helpful.\\n\\nUser: Hello!"
            >>> response = client.generate(prompt)
            >>> print(response)
            "Hello! How can I help you today?"
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Send request to OpenAI API
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
            
        except Exception as e:
            # Re-raise with context about the operation
            raise Exception(
                f"OpenAI API request failed: {str(e)}"
            ) from e


__all__ = ['OpenAI_LLM_Client']
