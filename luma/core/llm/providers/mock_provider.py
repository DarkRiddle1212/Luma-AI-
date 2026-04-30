"""
Mock LLM Provider for testing.

Provides a configurable mock that returns pre-defined responses,
simulates latency, and can simulate error conditions.
"""

import time
from typing import Dict

from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing purposes.

    Accepts a config dict with:
        - responses (list): Pre-defined responses to return in order
        - delay (float): Simulated latency in seconds (default 0.0)
        - error_mode (str|None): If set, raises ProviderError with this message
    """

    def __init__(self, config: Dict, logger=None):
        self._config = config
        self._logger = logger
        self._responses = config.get("responses", [])
        self._response_index = 0
        self._delay = config.get("delay", 0.0)
        self._error_mode = config.get("error_mode", None)

    def generate(self, prompt: str, options: Dict) -> Dict:
        """
        Return the next pre-configured response.

        Simulates delay and error conditions as configured.

        Raises:
            ProviderError: If error_mode is set, or responses are exhausted.
        """
        if self._delay > 0:
            time.sleep(self._delay)

        if self._error_mode:
            is_transient = "transient" in self._error_mode.lower()
            raise ProviderError(self._error_mode, is_transient=is_transient)

        if self._response_index >= len(self._responses):
            raise ProviderError("no more mock responses available", is_transient=False)

        response = self._responses[self._response_index]
        self._response_index += 1
        return response

    def validate_config(self, config: Dict) -> bool:
        """Mock provider has no required fields — always returns True."""
        return True
