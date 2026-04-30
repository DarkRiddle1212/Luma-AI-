"""
Unit tests for LLMClient abstract base class and OpenAILLMClient.

Tests cover:
- Abstract base class extensibility via MockLLMClient
- Transient error retry behaviour (up to max_retries, then LLMClientError)
- Non-transient 4xx errors raise LLMClientError immediately (no retry)
- Timeout raises LLMClientError with "timeout" in message
- Empty response body raises LLMClientError
"""

from unittest.mock import MagicMock, patch, call
import pytest
import openai

from luma.core.llm.config import LLMConfig
from luma.core.llm.llm_client import LLMClient, OpenAILLMClient, NON_TRANSIENT_STATUS_CODES
from luma.core.llm.schemas import LLMClientError, LLMRequest, LLMResponse, PromptContext
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_config(max_retries: int = 2, timeout_seconds: float = 5.0) -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        model="gpt-4o-mini",
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )


def make_request(request_id: str = "req-001") -> LLMRequest:
    ctx = PromptContext(
        system_instructions="You are a helpful assistant.",
        user_profile="beginner",
        relevant_memories=[],
        current_input="Hello",
        output_constraints="Be concise.",
    )
    return LLMRequest(
        prompt_context=ctx,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=256,
        request_id=request_id,
    )


def make_logger() -> StructuredLogger:
    logger = MagicMock(spec=StructuredLogger)
    return logger


def make_openai_client(config=None, logger=None) -> OpenAILLMClient:
    config = config or make_config()
    logger = logger or make_logger()
    with patch("luma.core.llm.llm_client.openai.OpenAI"):
        client = OpenAILLMClient(config=config, logger=logger)
    return client


def make_api_status_error(status_code: int) -> openai.APIStatusError:
    """Create a mock APIStatusError with the given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    err = openai.APIStatusError(
        message=f"HTTP {status_code}",
        response=mock_response,
        body=None,
    )
    return err


# ---------------------------------------------------------------------------
# MockLLMClient — verifies abstract base class extensibility
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Concrete subclass used to verify LLMClient is properly extensible."""

    def __init__(self, response: LLMResponse):
        self._response = response

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._response


class TestMockLLMClient:
    def test_mock_client_is_valid_subclass(self):
        """MockLLMClient can be instantiated and used without real HTTP calls."""
        expected = LLMResponse(
            request_id="r1",
            raw_text="Hello!",
            model="mock-model",
            prompt_tokens=10,
            completion_tokens=5,
            provider="mock",
        )
        client = MockLLMClient(response=expected)
        result = client.complete(make_request())
        assert result is expected

    def test_llm_client_is_abstract(self):
        """LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OpenAILLMClient — successful call
# ---------------------------------------------------------------------------

class TestOpenAILLMClientSuccess:
    def test_successful_call_returns_llm_response(self):
        """A successful API call returns a populated LLMResponse."""
        config = make_config()
        logger = make_logger()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10

        mock_message = MagicMock()
        mock_message.content = "Hello, world!"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model = "gpt-4o-mini"

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.return_value = mock_completion

            client = OpenAILLMClient(config=config, logger=logger)
            result = client.complete(make_request())

        assert result.raw_text == "Hello, world!"
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10
        assert result.provider == "openai"
        assert result.request_id == "req-001"

    def test_base_url_passed_to_openai_client(self):
        """base_url from config is forwarded to the openai.OpenAI constructor."""
        config = LLMConfig(
            api_key="key",
            model="llama3",
            base_url="http://localhost:11434/v1",
        )
        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            OpenAILLMClient(config=config, logger=make_logger())
            MockOpenAI.assert_called_once_with(
                api_key="key",
                base_url="http://localhost:11434/v1",
                timeout=config.timeout_seconds,
            )


# ---------------------------------------------------------------------------
# OpenAILLMClient — timeout
# ---------------------------------------------------------------------------

class TestOpenAILLMClientTimeout:
    def test_timeout_raises_llm_client_error_with_timeout_in_message(self):
        """APITimeoutError is converted to LLMClientError with 'timeout' in message."""
        config = make_config(timeout_seconds=10.0)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                openai.APITimeoutError(request=MagicMock())
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert "timeout" in str(exc_info.value).lower()

    def test_timeout_does_not_retry(self):
        """Timeout errors are raised immediately without retrying."""
        config = make_config(max_retries=3)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                openai.APITimeoutError(request=MagicMock())
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # Only called once — no retries
        assert mock_openai_instance.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# OpenAILLMClient — non-transient errors (immediate raise, no retry)
# ---------------------------------------------------------------------------

class TestOpenAILLMClientNonTransientErrors:
    @pytest.mark.parametrize("status_code", [400, 401, 403])
    def test_non_transient_status_raises_immediately(self, status_code):
        """HTTP 400/401/403 raise LLMClientError immediately without retrying."""
        config = make_config(max_retries=3)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                make_api_status_error(status_code)
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        # Only called once — no retries
        assert mock_openai_instance.chat.completions.create.call_count == 1
        assert str(status_code) in str(exc_info.value)

    @pytest.mark.parametrize("status_code", [400, 401, 403])
    def test_non_transient_error_does_not_log_retry(self, status_code):
        """No retry log event is emitted for non-transient errors."""
        config = make_config(max_retries=3)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                make_api_status_error(status_code)
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError):
                client.complete(make_request())

        logger.log.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAILLMClient — transient errors (retry up to max_retries)
# ---------------------------------------------------------------------------

class TestOpenAILLMClientTransientErrors:
    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
    def test_transient_status_retries_then_raises(self, status_code):
        """Transient HTTP errors are retried max_retries times then raise LLMClientError."""
        config = make_config(max_retries=2)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI, \
             patch("luma.core.llm.llm_client.time.sleep"):
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                make_api_status_error(status_code)
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        # Called max_retries + 1 times (initial + retries)
        assert mock_openai_instance.chat.completions.create.call_count == 3
        assert "exhausted" in str(exc_info.value)

    def test_transient_error_logs_each_retry(self):
        """A retry log event is emitted for each retry attempt."""
        config = make_config(max_retries=2)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI, \
             patch("luma.core.llm.llm_client.time.sleep"):
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                make_api_status_error(429)
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # 2 retries → 2 log calls
        assert logger.log.call_count == 2
        for log_call in logger.log.call_args_list:
            event_name = log_call[0][0]
            payload = log_call[0][1]
            assert event_name == "llm_client_retry"
            assert "request_id" in payload
            assert "attempt_number" in payload
            assert "error" in payload

    def test_transient_error_succeeds_on_retry(self):
        """If a transient error occurs then the next attempt succeeds, returns LLMResponse."""
        config = make_config(max_retries=2)
        logger = make_logger()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 3

        mock_message = MagicMock()
        mock_message.content = "Retry success"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model = "gpt-4o-mini"

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI, \
             patch("luma.core.llm.llm_client.time.sleep"):
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = [
                make_api_status_error(503),
                mock_completion,
            ]
            client = OpenAILLMClient(config=config, logger=logger)
            result = client.complete(make_request())

        assert result.raw_text == "Retry success"
        assert logger.log.call_count == 1  # one retry log

    def test_network_error_retries_then_raises(self):
        """Generic network exceptions are treated as transient and retried."""
        config = make_config(max_retries=1)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI, \
             patch("luma.core.llm.llm_client.time.sleep"):
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                ConnectionError("network unreachable")
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert mock_openai_instance.chat.completions.create.call_count == 2
        assert "exhausted" in str(exc_info.value)

    def test_exponential_backoff_sleep_values(self):
        """Exponential back-off sleeps 2^attempt seconds between retries."""
        config = make_config(max_retries=2)
        logger = make_logger()

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI, \
             patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.side_effect = (
                make_api_status_error(500)
            )
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # attempt 0 → sleep(1), attempt 1 → sleep(2)
        mock_sleep.assert_has_calls([call(1), call(2)])


# ---------------------------------------------------------------------------
# OpenAILLMClient — empty response body
# ---------------------------------------------------------------------------

class TestOpenAILLMClientEmptyResponse:
    def test_empty_choices_raises_llm_client_error(self):
        """Provider returning no choices raises LLMClientError."""
        config = make_config(max_retries=0)
        logger = make_logger()

        mock_completion = MagicMock()
        mock_completion.choices = []

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.return_value = mock_completion
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert "empty response" in str(exc_info.value).lower()

    def test_empty_content_raises_llm_client_error(self):
        """Provider returning empty string content raises LLMClientError."""
        config = make_config(max_retries=0)
        logger = make_logger()

        mock_message = MagicMock()
        mock_message.content = ""

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.return_value = mock_completion
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert "empty response" in str(exc_info.value).lower()

    def test_none_content_raises_llm_client_error(self):
        """Provider returning None content raises LLMClientError."""
        config = make_config(max_retries=0)
        logger = make_logger()

        mock_message = MagicMock()
        mock_message.content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch("luma.core.llm.llm_client.openai.OpenAI") as MockOpenAI:
            mock_openai_instance = MockOpenAI.return_value
            mock_openai_instance.chat.completions.create.return_value = mock_completion
            client = OpenAILLMClient(config=config, logger=logger)

            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert "empty response" in str(exc_info.value).lower()
