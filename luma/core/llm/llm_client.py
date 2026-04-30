"""
LLM Client — Provider API Abstraction.

Defines the abstract LLMClient base class and the concrete OpenAILLMClient
implementation that handles OpenAI (and OpenAI-compatible) API calls with
retry logic, exponential back-off, and structured error handling.
"""

import time
from abc import ABC, abstractmethod

import openai

from luma.core.llm.config import LLMConfig
from luma.core.llm.providers.provider_interface import LLMProvider, ProviderError
from luma.core.llm.schemas import LLMClientError, LLMRequest, LLMResponse
from luma.core.structured_logger import StructuredLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NON_TRANSIENT_STATUS_CODES = {400, 401, 403}
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract base class for LLM provider clients."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send a completion request to the LLM provider.

        Args:
            request: The LLMRequest containing prompt context and parameters.

        Returns:
            LLMResponse with the provider's raw text and metadata.

        Raises:
            LLMClientError: On API failure, timeout, or empty response.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    """
    LLMClient implementation for OpenAI (and OpenAI-compatible) endpoints.

    Handles retries with exponential back-off for transient errors and raises
    LLMClientError immediately for non-transient errors.
    """

    def __init__(self, config: LLMConfig, logger: StructuredLogger) -> None:
        self._config = config
        self._logger = logger
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute the completion request with retry logic.

        Transient errors (network issues, HTTP 429/5xx) are retried up to
        config.max_retries times with exponential back-off (2^attempt seconds).
        Non-transient errors (HTTP 400/401/403) raise immediately.
        Timeouts raise immediately with "timeout" in the message.
        """
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return self._call_api(request)

            except openai.APITimeoutError as e:
                raise LLMClientError(
                    f"timeout after {self._config.timeout_seconds}s: {e}"
                ) from e

            except openai.APIStatusError as e:
                if e.status_code in NON_TRANSIENT_STATUS_CODES:
                    raise LLMClientError(
                        f"non-transient HTTP {e.status_code}: {e}"
                    ) from e
                # Transient HTTP error — log and retry
                last_error = e
                if attempt < self._config.max_retries:
                    self._logger.log(
                        "llm_client_retry",
                        {
                            "request_id": request.request_id,
                            "attempt_number": attempt + 1,
                            "error": str(e),
                        },
                    )
                    time.sleep(2 ** attempt)

            except Exception as e:  # noqa: BLE001 — network/other transient errors
                last_error = e
                if attempt < self._config.max_retries:
                    self._logger.log(
                        "llm_client_retry",
                        {
                            "request_id": request.request_id,
                            "attempt_number": attempt + 1,
                            "error": str(e),
                        },
                    )
                    time.sleep(2 ** attempt)

        raise LLMClientError(
            f"exhausted {self._config.max_retries} retries: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, request: LLMRequest) -> LLMResponse:
        """
        Make a single API call to the OpenAI Chat Completions endpoint.

        Raises:
            LLMClientError: If the provider returns an empty response body.
        """
        # Build the prompt string from the PromptContext
        prompt_ctx = request.prompt_context
        messages = [
            {"role": "system", "content": prompt_ctx.system_instructions},
            {"role": "user", "content": prompt_ctx.current_input},
        ]

        completion = self._client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Guard against empty response body
        if not completion.choices:
            raise LLMClientError("empty response: provider returned no choices")

        raw_text = completion.choices[0].message.content or ""
        if not raw_text:
            raise LLMClientError("empty response: provider returned empty content")

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            request_id=request.request_id,
            raw_text=raw_text,
            model=completion.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider="openai",
        )


# ---------------------------------------------------------------------------
# Provider-based implementation
# ---------------------------------------------------------------------------

class ProviderLLMClient(LLMClient):
    """
    LLMClient implementation that delegates to an LLMProvider instance.

    Handles retries with exponential back-off for transient ProviderErrors
    and raises LLMClientError immediately for non-transient errors.
    """

    def __init__(self, provider: LLMProvider, config: LLMConfig, logger: StructuredLogger) -> None:
        self._provider = provider
        self._config = config
        self._logger = logger

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Execute the completion request via the provider with retry logic.

        Transient ProviderErrors are retried up to config.max_retries times
        with exponential back-off (2^attempt seconds).
        Non-transient ProviderErrors raise LLMClientError immediately.
        """
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            try:
                prompt = self._build_prompt(request)
                options = {
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "request_id": request.request_id,
                }
                result = self._provider.generate(prompt, options)
                return LLMResponse(
                    request_id=request.request_id,
                    raw_text=result["text"],
                    model=result["model"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    provider=result["provider"],
                )
            except ProviderError as e:
                if not e.is_transient:
                    raise LLMClientError(str(e)) from e
                last_error = e
                if attempt < self._config.max_retries:
                    self._logger.log("llm_client_retry", {
                        "request_id": request.request_id,
                        "attempt_number": attempt + 1,
                        "provider": self._provider.__class__.__name__,
                        "error": str(e),
                    })
                    time.sleep(2 ** attempt)

        raise LLMClientError(
            f"exhausted {self._config.max_retries} retries: {last_error}"
        ) from last_error

    def _build_prompt(self, request: LLMRequest) -> str:
        """Build a structured prompt string from the request's PromptContext."""
        from luma.core.llm.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        return builder.build(request.prompt_context)
