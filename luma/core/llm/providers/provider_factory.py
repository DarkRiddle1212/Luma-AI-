"""
Provider Factory for LLM Provider instantiation.

Centralizes provider selection and instantiation based on configuration.
Supports extensibility via a registry pattern.
"""

from typing import Dict, Type

from luma.core.llm.providers.provider_interface import LLMProvider
from luma.core.llm.providers.mock_provider import MockProvider

# Try to import GeminiProvider, but make it optional for environments
# where google.generativeai is not installed (e.g., testing)
try:
    from luma.core.llm.providers.gemini_provider import GeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiProvider = None  # type: ignore


class ProviderFactory:
    """
    Factory for creating LLM provider instances.

    Maintains an internal registry mapping provider names to provider classes.
    New providers can be added via the register() class method without modifying
    the create() logic.
    """

    _REGISTRY: Dict[str, Type[LLMProvider]] = {
        "mock": MockProvider,
    }
    
    # Only register GeminiProvider if it's available
    if GEMINI_AVAILABLE and GeminiProvider:
        _REGISTRY["gemini"] = GeminiProvider

    @classmethod
    def create(cls, provider_name: str, config: Dict, logger=None) -> LLMProvider:
        """
        Instantiate and return a provider by name.

        Validates configuration before instantiation.

        Args:
            provider_name: Name of the provider (e.g., "gemini", "mock")
            config: Provider-specific configuration dictionary
            logger: Optional StructuredLogger instance (required for GeminiProvider)

        Returns:
            Configured LLMProvider instance

        Raises:
            ValueError: If provider_name is not in the registry, or config is invalid
        """
        if provider_name not in cls._REGISTRY:
            supported = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Supported providers: {supported}"
            )

        provider_class = cls._REGISTRY[provider_name]

        # Validate configuration before instantiation using a temporary instance
        temp_instance = provider_class.__new__(provider_class)
        temp_instance.validate_config(config)

        if logger:
            return provider_class(config, logger)
        else:
            return provider_class(config)

    @classmethod
    def register(cls, provider_name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a new provider class in the factory registry.

        Args:
            provider_name: Name to register the provider under
            provider_class: Provider class implementing LLMProvider
        """
        cls._REGISTRY[provider_name] = provider_class
