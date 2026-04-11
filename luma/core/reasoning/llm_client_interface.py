"""
Abstract LLM Client Interface for the Reasoning Engine.

This module defines the abstract interface that all LLM client implementations
must follow. This abstraction enables provider-agnostic design, allowing the
Reasoning Engine to work with different language model providers (OpenAI, local
models, cloud APIs, etc.) without code changes.

The interface defines a minimal contract sufficient for reasoning operations,
focusing on the core generate method that accepts a prompt and returns a response.
"""

from abc import ABC, abstractmethod


class LLM_Client_Interface(ABC):
    """
    Abstract interface for language model client implementations.
    
    This interface defines the contract that all LLM providers must implement
    to work with the Reasoning Engine. By depending only on this abstraction,
    the Reasoning Engine remains decoupled from specific provider implementations.
    
    Implementations should handle:
    - API authentication and configuration
    - Request formatting for the specific provider
    - Response parsing and error handling
    - Provider-specific features (temperature, max tokens, etc.)
    
    Example implementations:
    - OpenAI_LLM_Client: Uses OpenAI API
    - Local_LLM_Client: Uses local models (Ollama, llama.cpp, etc.)
    - Mock_LLM_Client: Returns predefined responses for testing
    
    Extension Points:
    
    Tool-Calling Support:
        The interface can be extended to support tool-calling capabilities where
        the LLM can request external tool invocations during reasoning.
        
        To implement tool-calling:
        1. Add a generate_with_tools() method to the interface:
           
           @abstractmethod
           def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]]) -> str:
               '''Generate response with access to external tools.
               
               Args:
                   prompt: The input prompt
                   tools: List of tool schemas with name, description, and parameters
               
               Returns:
                   Response that may include tool call requests in structured format
               '''
               pass
        
        2. Implementations should format tool schemas according to provider requirements
           (e.g., OpenAI function calling format, Anthropic tool use format)
        
        3. The response should indicate tool calls in a parseable format:
           - Tool name to invoke
           - Arguments for the tool
           - Whether to continue reasoning after tool execution
        
        4. Providers that don't support native tool-calling can implement this via
           prompt engineering, instructing the LLM to output structured tool requests
        
        Example tool-calling flow:
            # Define available tools
            tools = [
                {
                    "name": "search_memories",
                    "description": "Search for additional relevant memories",
                    "parameters": {"query": "string"}
                },
                {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {"expression": "string"}
                }
            ]
            
            # LLM client with tool support
            response = client.generate_with_tools(prompt, tools)
            
            # Response might contain:
            # "I need to search for more information.
            #  TOOL_CALL: search_memories(query='user preferences')
            #  Please execute this tool and provide the results."
    
    Example:
        >>> class My_LLM_Client(LLM_Client_Interface):
        ...     def generate(self, prompt: str) -> str:
        ...         # Implementation-specific logic
        ...         return "Generated response"
        ...
        >>> client = My_LLM_Client()
        >>> response = client.generate("What is Python?")
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the language model for the given prompt.
        
        This method sends the prompt to the language model provider and returns
        the generated response as a string. Implementations should handle all
        provider-specific details including authentication, request formatting,
        and error handling.
        
        Args:
            prompt (str): The input prompt to send to the language model.
                         This typically includes system instructions, context,
                         and the user's query formatted by the Prompt_Builder.
        
        Returns:
            str: The generated response from the language model.
                 The response should be the raw text output that will be
                 processed by the Response_Formatter.
        
        Raises:
            Exception: Implementations may raise provider-specific exceptions
                      for API errors, authentication failures, rate limits, etc.
                      Callers should handle these appropriately.
        
        Example:
            >>> client = OpenAI_LLM_Client(api_key="...")
            >>> prompt = "System: You are a helpful assistant.\\n\\nUser: Hello!"
            >>> response = client.generate(prompt)
            >>> print(response)
            "Hello! How can I help you today?"
        """
        pass


__all__ = ['LLM_Client_Interface']
