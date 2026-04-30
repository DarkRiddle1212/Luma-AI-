"""
LLMEngine — Orchestration pipeline for LLM generation.

Wires together PromptBuilder, LLMClient, and ResponseParser into a single
clean interface. Implements LLMInterface for backward compatibility.
"""

import time
import uuid
from typing import Dict

from luma.core.llm_interface import LLMInterface
from luma.core.llm.schemas import (
    LLMRequest,
    LLMGenerationError,
    LLMClientError,
    PromptBuildError,
    ResponseParseError,
    ParsedResponse,
    PromptContext,
)
from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.llm_client import LLMClient
from luma.core.llm.response_parser import ResponseParser
from luma.core.structured_logger import StructuredLogger

_DEFAULT_FALLBACK = (
    "I'm having trouble generating a response right now. "
    "Please try again in a moment."
)
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024


class LLMEngine(LLMInterface):
    """
    Orchestrates PromptBuilder → LLMClient → ResponseParser pipeline.

    Implements LLMInterface for backward compatibility via generate_response().
    Primary interface is generate(request: LLMRequest) -> ParsedResponse.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        response_parser: ResponseParser,
        logger: StructuredLogger,
        fallback_response: str = _DEFAULT_FALLBACK,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._response_parser = response_parser
        self._logger = logger
        self._fallback_response = fallback_response

    def generate(self, request: LLMRequest) -> ParsedResponse:
        """
        Execute the full generation pipeline for the given request.

        Pipeline: PromptBuilder.build() → LLMClient.complete() → ResponseParser.parse()

        On LLMClientError: logs warning, returns fallback ParsedResponse (is_valid=False).
        On PromptBuildError or ResponseParseError: logs error, raises LLMGenerationError.

        Args:
            request: The LLMRequest containing prompt context and parameters.

        Returns:
            ParsedResponse with generated text and metadata.

        Raises:
            LLMGenerationError: When prompt building or response parsing fails.
        """
        start_time = time.monotonic()
        self._logger.log("llm_generate_start", {
            "request_id": request.request_id,
            "model": request.model,
            "max_tokens": request.max_tokens,
        })

        # Step 1: Build prompt
        try:
            prompt = self._prompt_builder.build(request.prompt_context)
        except PromptBuildError as e:
            self._logger.log("llm_generate_error", {"error": str(e), "stage": "prompt_build"})
            raise LLMGenerationError("Prompt build failed") from e

        # Step 2: Call LLM (at most once)
        try:
            llm_response = self._llm_client.complete(request)
        except LLMClientError as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            self._logger.log("llm_generate_fallback", {
                "request_id": request.request_id,
                "model": request.model,
                "error": str(e),
                "duration_ms": duration_ms,
            })
            return ParsedResponse(
                request_id=request.request_id,
                text=self._fallback_response,
                is_valid=False,
                validation_notes=["llm_client_error"],
                token_usage={"prompt": 0, "completion": 0},
                truncated=False,
            )

        # Step 3: Parse response
        try:
            parsed = self._response_parser.parse(llm_response)
        except ResponseParseError as e:
            self._logger.log("llm_generate_error", {"error": str(e), "stage": "response_parse"})
            raise LLMGenerationError("Response parse failed") from e

        duration_ms = int((time.monotonic() - start_time) * 1000)
        self._logger.log("llm_generate_complete", {
            "request_id": request.request_id,
            "model": request.model,
            "prompt_tokens": parsed.token_usage.get("prompt", 0),
            "completion_tokens": parsed.token_usage.get("completion", 0),
            "duration_ms": duration_ms,
        })
        return parsed

    def generate_response(self, prompt: str, context: Dict) -> str:
        """
        Backward-compatible wrapper implementing LLMInterface.

        Builds a minimal PromptContext and LLMRequest from the legacy
        prompt string and context dict, then delegates to generate().

        Args:
            prompt: The user's input text.
            context: Dictionary with optional keys like 'system_instructions',
                     'user_profile', 'output_constraints', 'model',
                     'temperature', 'max_tokens'.

        Returns:
            str: The generated response text.
        """
        prompt_context = PromptContext(
            system_instructions=context.get(
                "system_instructions", "You are a helpful assistant."
            ),
            user_profile=context.get("user_profile", ""),
            relevant_memories=context.get("relevant_memories", []),
            current_input=prompt,
            output_constraints=context.get("output_constraints", ""),
        )
        request = LLMRequest(
            prompt_context=prompt_context,
            model=context.get("model", _DEFAULT_MODEL),
            temperature=context.get("temperature", _DEFAULT_TEMPERATURE),
            max_tokens=context.get("max_tokens", _DEFAULT_MAX_TOKENS),
            request_id=context.get("request_id", str(uuid.uuid4())),
        )
        result = self.generate(request)
        return result.text
