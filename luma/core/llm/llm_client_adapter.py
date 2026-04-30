"""
LLM Client Adapter Module

This module provides an adapter that makes LLMClient implementations
compatible with the LLMInterface used by ReasoningEngine.

The adapter bridges the gap between:
- LLMInterface: Used by ReasoningEngine (generate_response(prompt, context) -> str)
- LLMClient: New provider-based system (complete(request) -> LLMResponse)
"""

import uuid
from typing import Dict

from luma.core.llm_interface import LLMInterface
from luma.core.llm.llm_client import LLMClient
from luma.core.llm.config import LLMConfig
from luma.core.llm.schemas import LLMRequest, PromptContext
from luma.core.structured_logger import StructuredLogger


class LLMClientAdapter(LLMInterface):
    """
    Adapter that makes an LLMClient compatible with LLMInterface.
    
    This class implements the LLMInterface contract by delegating to an
    LLMClient instance. It converts between the two interfaces:
    
    - LLMInterface.generate_response(prompt: str, context: Dict) -> str
    - LLMClient.complete(request: LLMRequest) -> LLMResponse
    
    The adapter extracts relevant information from the context dictionary
    to construct an LLMRequest with appropriate PromptContext.
    
    Example:
        >>> from luma.core.llm.config import load_llm_config_from_env
        >>> from luma.core.llm.providers.provider_factory import ProviderFactory
        >>> from luma.core.llm.llm_client import ProviderLLMClient
        >>> from luma.core.llm.llm_client_adapter import LLMClientAdapter
        >>> 
        >>> # Load configuration from environment
        >>> config = load_llm_config_from_env()
        >>> 
        >>> # Create provider using factory
        >>> provider = ProviderFactory.create(
        >>>     provider_name=config.provider_name,
        >>>     config=config.provider_config
        >>> )
        >>> 
        >>> # Create LLM client with provider
        >>> llm_client = ProviderLLMClient(
        >>>     provider=provider,
        >>>     config=config,
        >>>     logger=StructuredLogger("llm_client")
        >>> )
        >>> 
        >>> # Create adapter for ReasoningEngine
        >>> llm_adapter = LLMClientAdapter(llm_client, config)
        >>> 
        >>> # Use with ReasoningEngine
        >>> from luma.core.reasoning import ReasoningEngine
        >>> engine = ReasoningEngine(llm=llm_adapter)
    """
    
    def __init__(self, llm_client: LLMClient, config: LLMConfig):
        """
        Initialize the adapter with an LLMClient instance.
        
        Args:
            llm_client: The LLMClient instance to delegate to
            config: LLM configuration for default values
        """
        self._llm_client = llm_client
        self._config = config
    
    def generate_response(self, prompt: str, context: Dict) -> str:
        """
        Generate a response by delegating to the LLMClient.
        
        Converts the prompt and context into an LLMRequest and calls
        llm_client.complete(). Extracts relevant information from context:
        
        - model: Uses config.model as default, can be overridden by context
        - temperature: Uses config.temperature as default
        - max_tokens: Uses config.max_tokens as default
        - request_id: Generates a unique ID if not in context
        
        The PromptContext is constructed with minimal information since
        the ReasoningEngine's context dictionary doesn't map directly to
        PromptContext fields. The prompt string is used as current_input.
        
        Args:
            prompt: The prompt string to send to the LLM
            context: Context dictionary with metadata
            
        Returns:
            str: The generated response text
            
        Raises:
            Exception: If the LLMClient raises an exception
        """
        # Extract or generate request ID
        request_id = context.get("request_id", str(uuid.uuid4()))
        
        # Extract model from context or use default from config
        model = context.get("model", self._config.model)
        
        # Extract temperature from context or use default from config
        temperature = context.get("temperature", self._config.temperature)
        
        # Extract max_tokens from context or use default from config
        max_tokens = context.get("max_tokens", self._config.max_tokens)
        
        # Construct minimal PromptContext
        # The ReasoningEngine context doesn't map directly to PromptContext fields,
        # so we use reasonable defaults for the fields we don't have
        prompt_context = PromptContext(
            system_instructions=context.get("system_instructions", "You are a helpful assistant."),
            user_profile=context.get("user_profile", ""),
            relevant_memories=context.get("relevant_memories", []),
            current_input=prompt,
            output_constraints=context.get("output_constraints", "")
        )
        
        # Create LLMRequest
        request = LLMRequest(
            prompt_context=prompt_context,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_id=request_id
        )
        
        # Delegate to LLMClient
        response = self._llm_client.complete(request)
        
        # Return the raw text from the response
        return response.raw_text