"""
Gemini Provider Implementation.

Implements the LLMProvider interface for Google Gemini API.
Handles authentication, API communication, response normalization,
and error mapping for Gemini-specific interactions.
"""

from typing import Dict
import google.generativeai as genai

from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError
from luma.core.structured_logger import StructuredLogger


class GeminiProvider(LLMProvider):
    """
    Gemini-specific implementation of the LLMProvider interface.
    
    This provider handles authentication with Google's Gemini API,
    sends generation requests, normalizes responses to the common format,
    and maps Gemini-specific errors to ProviderError.
    
    Attributes:
        _config: Provider configuration dictionary
        _logger: StructuredLogger instance for observability
        _api_key: Gemini API key for authentication
        _model_name: Default model name to use
        _timeout: Request timeout in seconds
        _log_prompts: Whether to log full prompt text (default False)
        _model: Configured GenerativeModel instance
    """
    
    def __init__(self, config: Dict, logger: StructuredLogger):
        """
        Initialize the GeminiProvider.
        
        Args:
            config: Configuration dictionary containing:
                - api_key (str, required): Gemini API key
                - model (str, optional): Model name (default: "gemini-2.5-flash")
                - timeout (float, optional): Request timeout in seconds (default: 30.0)
                - max_tokens (int, optional): Default max tokens (default: 1024)
                - temperature (float, optional): Default temperature (default: 0.4)
                - log_prompts (bool, optional): Enable full prompt logging (default: False)
            logger: StructuredLogger instance for structured logging
        
        Raises:
            ValueError: If required configuration fields are missing
        """
        self._config = config
        self._logger = logger
        
        # Validate configuration
        self.validate_config(config)
        
        # Extract configuration values
        self._api_key = config["api_key"]
        self._model_name = config.get("model", "gemini-2.5-flash")
        self._timeout = config.get("timeout", 30.0)
        self._log_prompts = config.get("log_prompts", False)
        
        # Configure Gemini SDK with API key
        genai.configure(api_key=self._api_key)
        
        # Initialize the generative model
        self._model = genai.GenerativeModel(self._model_name)
    
    def generate(self, prompt: str, options: Dict) -> Dict:
        """
        Generate text from a prompt using Gemini API.
        
        Args:
            prompt: The prompt string to send to Gemini
            options: Generation parameters dictionary containing:
                - temperature (float, optional): Sampling temperature
                - max_tokens (int, optional): Maximum tokens to generate
                - model (str, optional): Model name override
                - request_id (str, optional): Request identifier for logging
        
        Returns:
            Dictionary with keys:
                - text (str): Generated text
                - model (str): Model name used
                - prompt_tokens (int): Input tokens consumed
                - completion_tokens (int): Output tokens generated
                - provider (str): "gemini"
        
        Raises:
            ProviderError: On API failure, timeout, or invalid response
        """
        # Extract options with fallbacks to config defaults
        model_name = options.get("model", self._model_name)
        temperature = options.get("temperature", self._config.get("temperature", 0.4))
        max_tokens = options.get("max_tokens", self._config.get("max_tokens", 1024))
        request_id = options.get("request_id", "unknown")
        
        # Log request start
        self._log_request_start(request_id, model_name, len(prompt))
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Create model instance if model name differs from default
            model = self._model
            if model_name != self._model_name:
                model = genai.GenerativeModel(model_name)
            
            # Make API call with timeout handling
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": self._timeout}
            )
            
            # Normalize response to common dictionary format
            result = self._normalize_response(response, model_name, request_id)
            
            # Log success
            self._log_request_success(request_id, result)
            
            return result
            
        except Exception as e:
            # Map to ProviderError with transient flag
            provider_error = self._map_error(e, request_id)
            self._log_request_error(request_id, provider_error)
            raise provider_error
    
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
        required_fields = ["api_key"]
        
        for field in required_fields:
            if field not in config or not config[field]:
                raise ValueError(
                    f"Gemini provider requires '{field}' in configuration"
                )
        
        # Validate API key is non-empty string
        if not isinstance(config["api_key"], str) or not config["api_key"].strip():
            raise ValueError("api_key must be a non-empty, non-whitespace string")
        
        return True
    
    def _mask_api_key(self, key: str) -> str:
        """
        Mask API key for logging (show only last 4 characters).
        
        Args:
            key: The API key to mask
        
        Returns:
            Masked API key string (e.g., "****abcd")
        """
        if len(key) <= 4:
            return "****"
        return "*" * (len(key) - 4) + key[-4:]
    
    def _log_request_start(self, request_id: str, model: str, prompt_length: int) -> None:
        """
        Log the start of a provider request.
        
        Args:
            request_id: Request identifier
            model: Model name being used
            prompt_length: Length of the prompt in characters
        """
        self._logger.log(
            "provider_request_start",
            {
                "provider": "gemini",
                "model": model,
                "prompt_length": prompt_length,
                "request_id": request_id
            }
        )
    
    def _log_request_success(self, request_id: str, result: Dict) -> None:
        """
        Log a successful provider request.
        
        Args:
            request_id: Request identifier
            result: The normalized response dictionary
        """
        self._logger.log(
            "provider_request_success",
            {
                "provider": "gemini",
                "model": result["model"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "request_id": request_id
            }
        )
    
    def _log_request_error(self, request_id: str, error: ProviderError) -> None:
        """
        Log a provider request error.
        
        Args:
            request_id: Request identifier
            error: The ProviderError that occurred
        """
        self._logger.log(
            "provider_request_error",
            {
                "provider": "gemini",
                "error_message": str(error),
                "is_transient": error.is_transient,
                "request_id": request_id
            }
        )
    
    def _normalize_response(self, response, model_name: str, request_id: str) -> Dict:
        """
        Normalize Gemini API response to common dictionary format.
        
        Args:
            response: The Gemini API response object
            model_name: The model name used for the request
            request_id: Request identifier for error logging
        
        Returns:
            Dictionary with keys: text, model, prompt_tokens, completion_tokens, provider
        
        Raises:
            ProviderError: If response is empty or missing required content
        """
        # Check if response has candidates
        if not response.candidates:
            raise ProviderError(
                "empty response: no candidates returned",
                is_transient=False
            )
        
        # Extract first candidate
        candidate = response.candidates[0]
        
        # Check if candidate has content
        if not candidate.content or not candidate.content.parts:
            raise ProviderError(
                "empty response: no content in candidate",
                is_transient=False
            )
        
        # Extract text from first part
        text = candidate.content.parts[0].text
        
        # Extract token usage from metadata
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        
        return {
            "text": text,
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "provider": "gemini"
        }
    
    def _map_error(self, error: Exception, request_id: str) -> ProviderError:
        """
        Map Gemini-specific errors to ProviderError with transient flag.
        
        Args:
            error: The exception raised by Gemini API
            request_id: Request identifier for logging
        
        Returns:
            ProviderError with appropriate message and transient flag
        """
        error_str = str(error)
        
        # Check HTTP status codes first (more specific than keyword matching)
        # Rate limit errors (HTTP 429)
        if "429" in error_str or "quota" in error_str.lower():
            return ProviderError(
                f"rate limit exceeded: {error}",
                is_transient=True
            )
        
        # Server errors (HTTP 5xx)
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return ProviderError(
                f"server error: {error}",
                is_transient=True
            )
        
        # Authentication errors (HTTP 401, 403)
        if any(code in error_str for code in ["401", "403"]):
            return ProviderError(
                f"authentication error: {error}",
                is_transient=False
            )
        
        # Bad request errors (HTTP 400)
        if "400" in error_str:
            return ProviderError(
                f"bad request: {error}",
                is_transient=False
            )
        
        # Timeout errors (checked after HTTP codes to avoid "Gateway timeout" false match)
        if "timeout" in error_str.lower():
            return ProviderError(
                f"timeout after {self._timeout}s: {error}",
                is_transient=True
            )
        
        # Default: treat as non-transient
        return ProviderError(
            f"unexpected error: {error}",
            is_transient=False
        )
