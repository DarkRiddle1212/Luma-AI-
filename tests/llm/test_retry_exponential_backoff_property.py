"""
Property-based test for LLMClient retry with exponential backoff.

**Validates: Requirements 5.4, 10.1, 10.2**

Property: For transient ProviderErrors, LLMClient retries up to max_retries times
with exponential backoff timing (2^0, 2^1, 2^2 seconds).

Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
"""

import pytest
from hypothesis import given, settings, strategies as st
from typing import List, Tuple
from unittest.mock import MagicMock, patch, call

from luma.core.llm.llm_client import ProviderLLMClient
from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.providers.provider_interface import ProviderError
from luma.core.llm.config import LLMConfig
from luma.core.llm.schemas import LLMRequest, PromptContext
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating max_retries values
max_retries_strategy = st.integers(min_value=0, max_value=5)

# Strategy for generating sequences of transient errors
@st.composite
def transient_error_sequence_strategy(draw):
    """Generate sequences of transient errors for testing retry logic."""
    max_retries = draw(max_retries_strategy)
    
    # Generate error messages
    error_messages = []
    for i in range(max_retries + 1):  # +1 for the initial attempt
        error_msg = draw(st.text(min_size=1, max_size=100))
        error_messages.append(f"transient error {i}: {error_msg}")
    
    return max_retries, error_messages

# Strategy for generating LLMRequest objects
@st.composite
def llm_request_strategy(draw):
    """Generate LLMRequest objects for testing."""
    ctx = PromptContext(
        system_instructions="Test system instructions",
        user_profile="test_user",
        relevant_memories=[],
        current_input="Test input",
        output_constraints="Be concise"
    )
    
    return LLMRequest(
        prompt_context=ctx,
        model=draw(st.text(min_size=1, max_size=50)),
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.integers(min_value=1, max_value=4000)),
        request_id=draw(st.text(min_size=1, max_size=50))
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_config(max_retries: int) -> LLMConfig:
    """Create a test LLMConfig with specified max_retries."""
    return LLMConfig(
        api_key="test-key",
        model="test-model",
        max_retries=max_retries,
        provider_name="mock",
    )


def make_logger() -> StructuredLogger:
    """Create a mock logger."""
    return MagicMock(spec=StructuredLogger)


def create_failing_provider(error_messages: List[str], succeed_on_last: bool = False) -> MockProvider:
    """
    Create a MockProvider that fails with the given error messages.
    
    Args:
        error_messages: List of error messages to raise in sequence
        succeed_on_last: If True, succeed on the last attempt instead of failing
    
    Returns:
        MockProvider configured to fail with the given errors
    """
    call_count = 0
    
    def generate_with_failures(prompt, options):
        nonlocal call_count
        if call_count < len(error_messages):
            error_msg = error_messages[call_count]
            call_count += 1
            
            # Check if we should succeed on the last attempt
            if succeed_on_last and call_count == len(error_messages):
                # Return a success response instead of raising an error
                return {
                    "text": f"Success after {call_count} attempts",
                    "model": "test-model",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "provider": "mock"
                }
            
            raise ProviderError(error_msg, is_transient=True)
        else:
            # After exhausting all errors, raise a generic error
            raise ProviderError("Unexpected call after error sequence", is_transient=False)
    
    provider = MockProvider(config={})
    provider.generate = generate_with_failures
    return provider


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestRetryExponentialBackoffProperty:
    """
    Property: Retry with exponential backoff for transient errors.
    
    **Validates: Requirements 5.4, 10.1, 10.2**
    
    Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
    """
    
    @pytest.mark.property_test
    @given(
        max_retries_and_errors=transient_error_sequence_strategy(),
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_retry_count_matches_max_retries(self, max_retries_and_errors: Tuple[int, List[str]], request: LLMRequest):
        """
        For any max_retries value and sequence of transient errors,
        LLMClient retries exactly max_retries times before giving up.
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        max_retries, error_messages = max_retries_and_errors
        
        # Create provider that fails with all errors
        provider = create_failing_provider(error_messages)
        
        # Create LLMClient
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=make_logger()
        )
        
        # Mock time.sleep to track sleep calls
        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            # Should raise LLMClientError after exhausting retries
            with pytest.raises(Exception) as exc_info:
                client.complete(request)
            
            # Verify sleep was called max_retries times (once between each retry)
            assert mock_sleep.call_count == max_retries
    
    @pytest.mark.property_test
    @given(
        max_retries_and_errors=transient_error_sequence_strategy(),
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_exponential_backoff_timing(self, max_retries_and_errors: Tuple[int, List[str]], request: LLMRequest):
        """
        For any max_retries value, sleep intervals follow exponential pattern: 2^0, 2^1, 2^2, ...
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        max_retries, error_messages = max_retries_and_errors
        
        # Create provider that fails with all errors
        provider = create_failing_provider(error_messages)
        
        # Create LLMClient
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=make_logger()
        )
        
        # Mock time.sleep to track sleep calls and values
        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            # Should raise LLMClientError after exhausting retries
            with pytest.raises(Exception):
                client.complete(request)
            
            # Verify sleep values follow exponential pattern: 2^0, 2^1, 2^2, ...
            expected_sleeps = [2 ** i for i in range(max_retries)]
            actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
            
            assert actual_sleeps == expected_sleeps, (
                f"Expected sleep values {expected_sleeps}, got {actual_sleeps}"
            )
    
    @pytest.mark.property_test
    @given(
        max_retries_and_errors=transient_error_sequence_strategy(),
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_success_on_last_retry(self, max_retries_and_errors: Tuple[int, List[str]], request: LLMRequest):
        """
        If the last retry succeeds, LLMClient returns LLMResponse without raising.
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        max_retries, error_messages = max_retries_and_errors
        
        # Skip if max_retries is 0 (no retries)
        if max_retries == 0:
            return
        
        # Create provider that fails then succeeds on the last attempt
        provider = create_failing_provider(error_messages, succeed_on_last=True)
        
        # Create LLMClient
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=make_logger()
        )
        
        # Mock time.sleep
        with patch("luma.core.llm.llm_client.time.sleep"):
            # Should succeed without raising
            result = client.complete(request)
            
            # Verify we got a valid response
            assert result is not None
            assert hasattr(result, "raw_text")
            assert hasattr(result, "model")
    
    @pytest.mark.property_test
    @given(
        max_retries=max_retries_strategy,
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_zero_max_retries_no_sleep(self, max_retries: int, request: LLMRequest):
        """
        With max_retries=0, no sleep should occur (immediate failure).
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        # Skip if max_retries is not 0 (we're testing the zero case)
        if max_retries != 0:
            return
        
        # Create provider that fails immediately
        provider = create_failing_provider(["immediate failure"])
        
        # Create LLMClient with max_retries=0
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=make_logger()
        )
        
        # Mock time.sleep
        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            # Should raise immediately
            with pytest.raises(Exception):
                client.complete(request)
            
            # Verify no sleep was called
            mock_sleep.assert_not_called()
    
    @pytest.mark.property_test
    @given(
        max_retries_and_errors=transient_error_sequence_strategy(),
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_retry_logging_events(self, max_retries_and_errors: Tuple[int, List[str]], request: LLMRequest):
        """
        Each retry attempt should log a 'llm_client_retry' event.
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        max_retries, error_messages = max_retries_and_errors
        
        # Create provider that fails with all errors
        provider = create_failing_provider(error_messages)
        
        # Create mock logger
        logger = make_logger()
        
        # Create LLMClient
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=logger
        )
        
        # Mock time.sleep
        with patch("luma.core.llm.llm_client.time.sleep"):
            # Should raise after exhausting retries
            with pytest.raises(Exception):
                client.complete(request)
            
            # Verify logger was called for each retry attempt
            assert logger.log.call_count == max_retries
            
            # Verify each log call has the correct event name
            for call in logger.log.call_args_list:
                event_name = call[0][0]
                assert event_name == "llm_client_retry"
    
    @pytest.mark.property_test
    @given(
        max_retries_and_errors=transient_error_sequence_strategy(),
        request=llm_request_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_no_retry_on_non_transient_error(self, max_retries_and_errors: Tuple[int, List[str]], request: LLMRequest):
        """
        Non-transient errors should not trigger retries.
        
        **Validates: Requirements 5.4, 10.1, 10.2**
        
        Feature: gemini-provider-integration, Property 9: Retry with Exponential Backoff
        """
        max_retries, _ = max_retries_and_errors
        
        # Create a provider that raises a non-transient error
        def generate_with_non_transient_error(prompt, options):
            raise ProviderError("non-transient error", is_transient=False)
        
        provider = MockProvider(config={})
        provider.generate = generate_with_non_transient_error
        
        # Create LLMClient
        config = make_config(max_retries)
        client = ProviderLLMClient(
            provider=provider,
            config=config,
            logger=make_logger()
        )
        
        # Mock time.sleep
        with patch("luma.core.llm.llm_client.time.sleep") as mock_sleep:
            # Should raise immediately
            with pytest.raises(Exception):
                client.complete(request)
            
            # Verify no sleep was called (no retries for non-transient errors)
            mock_sleep.assert_not_called()