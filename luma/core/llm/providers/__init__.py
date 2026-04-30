"""
Provider abstraction layer for LLM backends.

This module provides a common interface for swappable LLM providers (Gemini, OpenAI, etc.)
and a factory for provider instantiation.
"""

from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError
from luma.core.llm.providers.provider_factory import ProviderFactory
from luma.core.llm.providers.mock_provider import MockProvider

# Try to import GeminiProvider, but make it optional for environments
# where google.generativeai is not installed (e.g., testing)
try:
    from luma.core.llm.providers.gemini_provider import GeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiProvider = None  # type: ignore

__all__ = [
    "LLMProvider",
    "ProviderError",
    "ProviderFactory",
    "MockProvider",
]

# Conditionally add GeminiProvider to __all__ if available
if GEMINI_AVAILABLE:
    __all__.append("GeminiProvider")
