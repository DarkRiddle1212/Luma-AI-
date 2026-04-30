"""
Unit tests for LLMClient integration with providers.

Tests cover all requirements specified in task 7.4:
- Delegation to provider.generate()
- Response dictionary to LLMResponse conversion
- Retry logic with mock transient errors
- Immediate failure with mock non-transient errors
- Exponential backoff timing
- Retry logging

Requirements validated: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 10.1, 10.2, 10.3, 10.4, 10.5

This test file avoids import issues by mocking google.generativeai at module level.
"""

import sys
from unittest.mock import MagicMock, patch, call
import pytest

# Mock google.generativeai before importing any luma modules
mock_genai = MagicMock()
sys.modules['google.generativeai'] = mock_genai
sys.modules['google.generativeai.types'] = MagicMock()

from luma.core.llm.config import LLMConfig
from luma.core.llm.llm_client import ProviderLLMClient
from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.providers.provider_interface import ProviderError
from luma.core.llm.schemas import LLMClientError, LLMRequest, LLMResponse, PromptContext
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_config(max_retries: int = 2) -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        model="gemini-2.5-flash",
        max_retries=max_retries,
        provider_name="mock",
    )


def make_request(request_id: str = "req-001", model: str = "gemini-2.5-flash") -> LLMRequest:
    ctx = PromptContext(
        system_instructions="You are a helpful assistant.",
        user_profile="beginner",
        relevant_memories=[],
        current_input="Hello",
        output_constraints="Be concise.",
    )
    return LLMRequest(
        prompt_context=ctx,
        model=model,
        temperature=0.7,
        max_tokens=256,
        request_id=request_id,
    )


def make_logger() -> StructuredLogger:
    return MagicMock(spec=StructuredLogger)


def make_mock_response(
    text: str = "Hello from mock",
    model: str = "gemini-2.5-flash",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    provider: str = "mock",
) -> dict:
    return {
        "text": text,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "provider": provider,
    }


def make_provider_client(
    provider: MockProvider,
    config: LLMConfig = None,
    logger: StructuredLogger = None,
) -> ProviderLLMClient:
    return ProviderLLMClient(
        provider=provider,
        config=config or make_config(),
        logger=logger or make_logger(),
    )


# ---------------------------------------------------------------------------
# Test: delegation to provider.generate()
# ---------------------------------------------------------------------------

class TestDelegationToProvider:
    """Tests for delegation to provider.generate() method."""
    
    def test_complete_calls_provider_generate(self):
        """complete() delegates to provider.generate() exactly once on success."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider)
        client.complete(make_request())

        assert spy.call_count == 1

    def test_generate_called_with_prompt_string(self):
        """provider.generate() receives a non-empty prompt string."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider)
        client.complete(make_request())

        prompt_arg = spy.call_args[0][0]
        assert isinstance(prompt_arg, str)
        assert len(prompt_arg) > 0

    def test_generate_called_with_correct_options(self):
        """provider.generate() receives options dict with model, temperature, max_tokens, request_id."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        request = make_request(request_id="req-xyz", model="gemini-2.5-flash")
        client = make_provider_client(provider)
        client.complete(request)

        options_arg = spy.call_args[0][1]
        assert options_arg["model"] == "gemini-2.5-flash"
        assert options_arg["temperature"] == 0.7
        assert options_arg["max_tokens"] == 256
        assert options_arg["request_id"] == "req-xyz"

    def test_prompt_contains_system_instructions(self):
        """The prompt passed to generate() includes the system instructions."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        request = make_request()
        client = make_provider_client(provider)
        client.complete(request)

        prompt_arg = spy.call_args[0][0]
        assert "You are a helpful assistant." in prompt_arg

    def test_prompt_contains_current_input(self):
        """The prompt passed to generate() includes the current input."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider)
        client.complete(make_request())

        prompt_arg = spy.call_args[0][0]
        assert "Hello" in prompt_arg


# ---------------------------------------------------------------------------
# Test: response dictionary to LLMResponse conversion
# ---------------------------------------------------------------------------

class TestResponseConversion:
    """Tests for converting provider response dictionaries to LLMResponse objects."""
    
    def test_response_text_mapped_to_raw_text(self):
        """Provider dict 'text' field maps to LLMResponse.raw_text."""
        response_dict = make_mock_response(text="Generated content")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.raw_text == "Generated content"

    def test_response_model_preserved(self):
        """Provider dict 'model' field maps to LLMResponse.model."""
        response_dict = make_mock_response(model="gemini-2.5-flash")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.model == "gemini-2.5-flash"

    def test_response_prompt_tokens_preserved(self):
        """Provider dict 'prompt_tokens' maps to LLMResponse.prompt_tokens."""
        response_dict = make_mock_response(prompt_tokens=42)
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.prompt_tokens == 42

    def test_response_completion_tokens_preserved(self):
        """Provider dict 'completion_tokens' maps to LLMResponse.completion_tokens."""
        response_dict = make_mock_response(completion_tokens=17)
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.completion_tokens == 17

    def test_response_provider_field_preserved(self):
        """Provider dict 'provider' field maps to LLMResponse.provider."""
        response_dict = make_mock_response(provider="gemini")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.provider == "gemini"

    def test_response_request_id_preserved(self):
        """LLMResponse.request_id matches the original request's request_id."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request(request_id="req-abc"))

        assert result.request_id == "req-abc"

    def test_result_is_llm_response_instance(self):
        """complete() returns an LLMResponse instance."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert isinstance(result, LLMResponse)

    def test_all_fields_preserved_together(self):
        """All provider response fields are correctly mapped to LLMResponse."""
        response_dict = make_mock_response(
            text="Full response",
            model="gemini-pro",
            prompt_tokens=100,
            completion_tokens=50,
            provider="gemini",
        )
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request(request_id="req-full"))

        assert result.raw_text == "Full response"
        assert result.model == "gemini-pro"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.provider == "gemini"
        assert result.request_id == "req-full"


# ---------------------------------------------------------------------------
# Test: retry logic with mock transient errors
# ---------------------------------------------------------------------------

class TestRetryLogicTransientErrors:
    """Tests for retry logic with transient errors."""
    
    def test_transient_error_retries_up_to_max_retries(self):
        """Transient ProviderError triggers retries up to max_retries times."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # initial attempt + 2 retries = 3 total calls
        assert spy.call_count == 3

    def test_transient_error_raises_llm_client_error_after_exhaustion(self):
        """After exhausting retries, LLMClientError is raised."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

    def test_transient_error_exhausted_message_contains_retry_count(self):
        """LLMClientError message after exhaustion mentions the retry count."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError) as exc_info:
                client.complete(make_request())

        assert "exhausted" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)

    def test_transient_error_succeeds_on_retry(self):
        """If transient error occurs then next attempt succeeds, returns LLMResponse."""
        config = make_config(max_retries=2)
        response_dict = make_mock_response(text="Retry success")

        # First call raises transient error, second returns response
        call_count = 0
        original_generate = MockProvider(config={"responses": [response_dict]}).generate

        def generate_with_first_failure(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("transient error", is_transient=True)
            return response_dict

        provider = MockProvider(config={"responses": [response_dict]})
        provider.generate = generate_with_first_failure

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep"):
            result = client.complete(make_request())

        assert result.raw_text == "Retry success"

    def test_zero_max_retries_raises_immediately_on_transient(self):
        """With max_retries=0, transient error raises LLMClientError after single attempt."""
        config = make_config(max_retries=0)
        provider = MockProvider(config={"error_mode": "transient network error"})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # Only 1 attempt (no retries)
        assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Test: immediate failure with mock non-transient errors
# ---------------------------------------------------------------------------

class TestImmediateFailureNonTransientErrors:
    """Tests for immediate failure with non-transient errors."""
    
    def test_non_transient_error_raises_llm_client_error_immediately(self):
        """Non-transient ProviderError raises LLMClientError without retrying."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "permanent auth error"})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        client = make_provider_client(provider, config=config)

        with pytest.raises(LLMClientError):
            client.complete(make_request())

        # Only 1 call — no retries for non-transient errors
        assert spy.call_count == 1

    def test_non_transient_error_message_preserved(self):
        """LLMClientError message contains the original ProviderError message."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "permanent auth error"})

        client = make_provider_client(provider, config=config)

        with pytest.raises(LLMClientError) as exc_info:
            client.complete(make_request())

        assert "permanent auth error" in str(exc_info.value)

    def test_non_transient_error_does_not_sleep(self):
        """No sleep is called for non-transient errors (immediate raise)."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "permanent auth error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        mock_sleep.assert_not_called()

    def test_non_transient_error_does_not_log_retry(self):
        """No retry log event is emitted for non-transient errors."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "permanent auth error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with pytest.raises(LLMClientError):
            client.complete(make_request())

        logger.log.assert_not_called()

    def test_explicit_non_transient_provider_error(self):
        """Directly raised ProviderError(is_transient=False) raises immediately."""
        config = make_config(max_retries=3)

        def always_non_transient(prompt, options):
            raise ProviderError("bad request: invalid parameter", is_transient=False)

        provider = MockProvider(config={})
        provider.generate = always_non_transient

        client = make_provider_client(provider, config=config)

        with pytest.raises(LLMClientError) as exc_info:
            client.complete(make_request())

        assert "bad request" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: exponential backoff timing
# ---------------------------------------------------------------------------

class TestExponentialBackoffTiming:
    """Tests for exponential backoff timing between retries."""
    
    def test_backoff_sleeps_with_exponential_values(self):
        """Exponential backoff sleeps 2^0, 2^1 seconds between retries (max_retries=2)."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # attempt 0 → sleep(2^0=1), attempt 1 → sleep(2^1=2)
        mock_sleep.assert_has_calls([call(1), call(2)])

    def test_backoff_sleep_count_matches_retry_count(self):
        """Number of sleep calls equals max_retries."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        assert mock_sleep.call_count == 3

    def test_backoff_sleep_values_are_powers_of_two(self):
        """Sleep values follow 2^attempt pattern: 1, 2, 4 for max_retries=3."""
        config = make_config(max_retries=3)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        sleep_values = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_values == [1, 2, 4]

    def test_no_sleep_after_final_retry(self):
        """No sleep is called after the last retry attempt (only between retries)."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})

        client = make_provider_client(provider, config=config)

        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # max_retries=2 → 2 sleeps (not 3)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Test: retry logging
# ---------------------------------------------------------------------------

class TestRetryLogging:
    """Tests for retry logging behavior."""
    
    def test_retry_event_logged_for_each_retry(self):
        """A 'llm_client_retry' log event is emitted for each retry attempt."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        # 2 retries → 2 log calls
        assert logger.log.call_count == 2

    def test_retry_log_event_name_is_llm_client_retry(self):
        """Each retry log call uses event name 'llm_client_retry'."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        for log_call in logger.log.call_args_list:
            event_name = log_call[0][0]
            assert event_name == "llm_client_retry"

    def test_retry_log_payload_contains_request_id(self):
        """Retry log payload includes 'request_id' field."""
        config = make_config(max_retries=1)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request(request_id="req-log-test"))

        payload = logger.log.call_args_list[0][0][1]
        assert "request_id" in payload
        assert payload["request_id"] == "req-log-test"

    def test_retry_log_payload_contains_attempt_number(self):
        """Retry log payload includes 'attempt_number' field."""
        config = make_config(max_retries=2)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        payloads = [c[0][1] for c in logger.log.call_args_list]
        assert payloads[0]["attempt_number"] == 1
        assert payloads[1]["attempt_number"] == 2

    def test_retry_log_payload_contains_error(self):
        """Retry log payload includes 'error' field with error message."""
        config = make_config(max_retries=1)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        payload = logger.log.call_args_list[0][0][1]
        assert "error" in payload
        assert "transient network error" in payload["error"]

    def test_retry_log_payload_contains_provider(self):
        """Retry log payload includes 'provider' field."""
        config = make_config(max_retries=1)
        provider = MockProvider(config={"error_mode": "transient network error"})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)

        with patch("luma.core.llm.llm_client.time.sleep"):
            with pytest.raises(LLMClientError):
                client.complete(make_request())

        payload = logger.log.call_args_list[0][0][1]
        assert "provider" in payload

    def test_no_log_on_success(self):
        """No log events are emitted when the first attempt succeeds."""
        config = make_config(max_retries=2)
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        logger = make_logger()

        client = make_provider_client(provider, config=config, logger=logger)
        client.complete(make_request())

        logger.log.assert_not_called()


# ---------------------------------------------------------------------------
# Test: edge cases and additional coverage
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge case tests for LLMClient integration."""
    
    def test_empty_response_text_handling(self):
        """Empty text in provider response should be preserved (not filtered)."""
        response_dict = make_mock_response(text="")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.raw_text == ""
        assert result.prompt_tokens == 10  # Default from make_mock_response
        assert result.completion_tokens == 5  # Default from make_mock_response

    def test_large_token_counts(self):
        """Large token counts should be preserved correctly."""
        response_dict = make_mock_response(
            prompt_tokens=1000000,
            completion_tokens=500000
        )
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.prompt_tokens == 1000000
        assert result.completion_tokens == 500000

    def test_special_characters_in_text(self):
        """Special characters in response text should be preserved."""
        special_text = "Line 1\nLine 2\tTabbed\n\"Quotes\" & 'Apostrophes'"
        response_dict = make_mock_response(text=special_text)
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.raw_text == special_text

    def test_provider_name_with_spaces(self):
        """Provider names with spaces should be preserved."""
        response_dict = make_mock_response(provider="google gemini")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.provider == "google gemini"

    def test_model_name_with_special_chars(self):
        """Model names with special characters should be preserved."""
        response_dict = make_mock_response(model="gemini-2.5-flash-001")
        provider = MockProvider(config={"responses": [response_dict]})
        client = make_provider_client(provider)

        result = client.complete(make_request())

        assert result.model == "gemini-2.5-flash-001"

    def test_zero_max_tokens_in_request(self):
        """Request with max_tokens=0 should be passed correctly to provider."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        request = make_request()
        request.max_tokens = 0
        client = make_provider_client(provider)
        client.complete(request)

        options_arg = spy.call_args[0][1]
        assert options_arg["max_tokens"] == 0

    def test_zero_temperature_in_request(self):
        """Request with temperature=0.0 should be passed correctly to provider."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        request = make_request()
        request.temperature = 0.0
        client = make_provider_client(provider)
        client.complete(request)

        options_arg = spy.call_args[0][1]
        assert options_arg["temperature"] == 0.0

    def test_high_temperature_in_request(self):
        """Request with high temperature should be passed correctly to provider."""
        response_dict = make_mock_response()
        provider = MockProvider(config={"responses": [response_dict]})
        spy = MagicMock(wraps=provider.generate)
        provider.generate = spy

        request = make_request()
        request.temperature = 2.0
        client = make_provider_client(provider)
        client.complete(request)

        options_arg = spy.call_args[0][1]
        assert options_arg["temperature"] == 2.0