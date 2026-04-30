"""
Application Factory / Dependency Injection Wiring

Constructs the application-scoped LumaService with all core dependencies,
including LLMEngine wired with OpenAILLMClient.

LLMConfig is loaded from application settings — the api_key is sourced from
the OPENAI_API_KEY environment variable via the config loader (Settings),
never read directly by LLMClient.
"""

from luma.config import settings
from luma.core.llm import (
    LLMConfig,
    LLMEngine,
    OpenAILLMClient,
    PromptBuilder,
    ResponseParser,
)
from luma.core.structured_logger import StructuredLogger
from luma.api.services.luma_service import LumaService


def build_llm_engine(logger: StructuredLogger) -> LLMEngine:
    """
    Construct an LLMEngine backed by OpenAILLMClient.

    LLMConfig is populated from application settings so that the api_key
    is sourced from the environment (OPENAI_API_KEY) via the config loader,
    not read directly by LLMClient.

    Returns None when OPENAI_API_KEY is not configured, allowing the
    application to start without LLM support (fallback mode).
    """
    api_key = settings.openai_api_key
    if not api_key or not api_key.strip():
        return None

    llm_config = LLMConfig(
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_response_chars=settings.llm_max_response_chars,
        base_url=settings.llm_base_url,
    )

    llm_client = OpenAILLMClient(config=llm_config, logger=logger)
    prompt_builder = PromptBuilder()
    response_parser = ResponseParser(max_response_chars=llm_config.max_response_chars)

    return LLMEngine(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        response_parser=response_parser,
        logger=logger,
        fallback_response=llm_config.fallback_response,
    )


def build_luma_service(
    memory_interface,
    insight_engine,
    insight_moments_engine,
    personalization_engine,
    teacher_mode,
    logger: StructuredLogger,
) -> LumaService:
    """
    Construct a fully-wired LumaService with LLMEngine injected.

    LLMEngine is constructed with OpenAILLMClient when OPENAI_API_KEY is
    present in the environment.  When the key is absent the service starts
    in fallback mode (no LLM calls).
    """
    llm_engine = build_llm_engine(logger)

    return LumaService(
        memory_interface=memory_interface,
        insight_engine=insight_engine,
        insight_moments_engine=insight_moments_engine,
        personalization_engine=personalization_engine,
        teacher_mode=teacher_mode,
        llm_engine=llm_engine,
        logger=logger,
    )
