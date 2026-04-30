"""
Integration tests for LLM pipeline with provider system.

Tests the full pipeline: LLMEngine → LLMClient → Provider → ResponseParser
Validates that the provider integration preserves existing pipeline behavior.

**Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5**

Feature: gemini-provider-integration
"""

import pytest
from unittest.mock import MagicMock

from luma.core.llm.llm_engine import LLMEngine
from luma.core.llm.llm_client import ProviderLLMClient
from luma.core.llm.prompt_builder import PromptBuilder
from luma.core.llm.response_parser import ResponseParser
from luma.core.llm.providers.mock_provider import MockProvider
from luma.core.llm.config import LLMConfig
from luma.core.llm.schemas import LLMRequest, PromptContext, ParsedResponse
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    """Create a mock StructuredLogger."""
    return MagicMock(spec=StructuredLogger)


@pytest.fixture
def llm_config():
    """Create a test LLMConfig."""
    return LLMConfig(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=1024,
        max_retries=2,
        provider_name="mock",
    )


@pytest.fixture
def prompt_builder():
    """Create a PromptBuilder instance."""
    return PromptBuilder()


@pytest.fixture
def response_parser():
    """Create a ResponseParser instance."""
    return ResponseParser(max_response_chars=4000)


@pytest.fixture
def mock_provider_with_response():
    """Create a MockProvider with a predefined response."""
    response = {
        "text": "This is a test response from the mock provider.",
        "model": "test-model",
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "provider": "mock"
    }
    return MockProvider(config={"responses": [response]})


@pytest.fixture
def llm_client(mock_provider_with_response, llm_config, mock_logger):
    """Create a ProviderLLMClient with MockProvider."""
    return ProviderLLMClient(
        provider=mock_provider_with_response,
        config=llm_config,
        logger=mock_logger
    )


@pytest.fixture
def llm_engine(prompt_builder, llm_client, response_parser, mock_logger):
    """Create an LLMEngine with all components."""
    return LLMEngine(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
        response_parser=response_parser,
        logger=mock_logger
    )


@pytest.fixture
def sample_request():
    """Create a sample LLMRequest."""
    context = PromptContext(
        system_instructions="You are a helpful assistant.",
        user_profile="beginner",
        relevant_memories=["User prefers concise answers"],
        current_input="What is Python?",
        output_constraints="Keep it under 100 words."
    )
    return LLMRequest(
        prompt_context=context,
        model="test-model",
        temperature=0.7,
        max_tokens=256,
        request_id="test-req-001"
    )


# ---------------------------------------------------------------------------
# Test: Full pipeline integration
# ---------------------------------------------------------------------------

class TestLLMPipelineIntegration:
    """Integration tests for the full LLM pipeline with provider system."""
    
    def test_full_pipeline_with_mock_provider(self, llm_engine, sample_request):
        """
        Test complete pipeline: LLMEngine → LLMClient → MockProvider → ResponseParser.
        
        **Validates: Requirements 14.1, 14.4**
        """
        result = llm_engine.generate(sample_request)
        
        # Verify result is a ParsedResponse
        assert isinstance(result, ParsedResponse)
        
        # Verify response contains expected text
        assert result.text == "This is a test response from the mock provider."
        
        # Verify response is valid
        assert result.is_valid is True
        
        # Verify token usage is preserved
        assert result.token_usage["prompt"] == 50
        assert result.token_usage["completion"] == 20
        
        # Verify request ID is preserved
        assert result.request_id == "test-req-001"
    
    def test_prompt_builder_unchanged(self, prompt_builder, sample_request):
        """
        Verify PromptBuilder interface and behavior unchanged.
        
        **Validates: Requirement 14.2**
        """
        # PromptBuilder should work independently of provider system
        prompt = prompt_builder.build(sample_request.prompt_context)
        
        # Verify prompt structure
        assert "[System Instructions]" in prompt
        assert "You are a helpful assistant." in prompt
        assert "[User Profile]" in prompt
        assert "beginner" in prompt
        assert "[Relevant Memories]" in prompt
        assert "User prefers concise answers" in prompt
        assert "[Current Input]" in prompt
        assert "What is Python?" in prompt
        assert "[Output Constraints]" in prompt
        assert "Keep it under 100 words." in prompt
    
    def test_response_parser_unchanged(self, response_parser):
        """
        Verify ResponseParser interface and behavior unchanged.
        
        **Validates: Requirement 14.3**
        """
        from luma.core.llm.schemas import LLMResponse
        
        # ResponseParser should work independently of provider system
        llm_response = LLMResponse(
            request_id="test-req-002",
            raw_text="  Test response with whitespace  ",
            model="test-model",
            prompt_tokens=30,
            completion_tokens=10,
            provider="mock"
        )
        
        parsed = response_parser.parse(llm_response)
        
        # Verify parsing behavior
        assert parsed.text == "Test response with whitespace"
        assert parsed.is_valid is True
        assert parsed.token_usage["prompt"] == 30
        assert parsed.token_usage["completion"] == 10
        assert parsed.request_id == "test-req-002"
    
    def test_llm_engine_interface_unchanged(self, llm_engine):
        """
        Verify LLMEngine interface unchanged (implements LLMInterface).
        
        **Validates: Requirement 14.1**
        """
        from luma.core.llm_interface import LLMInterface
        
        # LLMEngine should still implement LLMInterface
        assert isinstance(llm_engine, LLMInterface)
        
        # Test backward-compatible generate_response method
        result = llm_engine.generate_response(
            prompt="Hello",
            context={
                "system_instructions": "You are helpful.",
                "user_profile": "test",
                "relevant_memories": [],
                "output_constraints": "Be brief."
            }
        )
        
        # Verify result is a string (backward compatibility)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_multiple_requests_sequential(self, llm_engine):
        """
        Test multiple sequential requests through the pipeline.
        
        **Validates: Requirements 14.1, 14.4**
        """
        # Create multiple requests
        requests = []
        for i in range(3):
            context = PromptContext(
                system_instructions="You are a helpful assistant.",
                user_profile="test",
                relevant_memories=[],
                current_input=f"Request {i}",
                output_constraints="Be concise."
            )
            requests.append(LLMRequest(
                prompt_context=context,
                model="test-model",
                temperature=0.7,
                max_tokens=256,
                request_id=f"req-{i}"
            ))
        
        # Process first request (should succeed)
        result1 = llm_engine.generate(requests[0])
        assert result1.is_valid is True
        assert result1.request_id == "req-0"
        
        # Subsequent requests will fail (MockProvider exhausted)
        # but should return fallback response
        result2 = llm_engine.generate(requests[1])
        assert result2.is_valid is False
        assert "trouble generating" in result2.text.lower()
    
    def test_provider_error_handling(self, prompt_builder, response_parser, llm_config, mock_logger):
        """
        Test error handling when provider raises errors.
        
        **Validates: Requirements 14.1, 14.4**
        """
        # Create provider that raises errors
        error_provider = MockProvider(config={"error_mode": "transient network error"})
        
        # Create client with error provider
        error_client = ProviderLLMClient(
            provider=error_provider,
            config=llm_config,
            logger=mock_logger
        )
        
        # Create engine with error client
        error_engine = LLMEngine(
            prompt_builder=prompt_builder,
            llm_client=error_client,
            response_parser=response_parser,
            logger=mock_logger
        )
        
        # Create request
        context = PromptContext(
            system_instructions="You are helpful.",
            user_profile="test",
            relevant_memories=[],
            current_input="Test",
            output_constraints="Brief."
        )
        request = LLMRequest(
            prompt_context=context,
            model="test-model",
            temperature=0.7,
            max_tokens=256,
            request_id="error-req"
        )
        
        # Should return fallback response (not raise exception)
        result = error_engine.generate(request)
        assert result.is_valid is False
        assert "trouble generating" in result.text.lower()


# ---------------------------------------------------------------------------
# Test: Component isolation
# ---------------------------------------------------------------------------

class TestComponentIsolation:
    """Test that components work independently of each other."""
    
    def test_prompt_builder_independent(self):
        """PromptBuilder works without LLMClient or provider."""
        builder = PromptBuilder()
        context = PromptContext(
            system_instructions="Test",
            user_profile="user",
            relevant_memories=[],
            current_input="Input",
            output_constraints="Constraints"
        )
        
        prompt = builder.build(context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    def test_response_parser_independent(self):
        """ResponseParser works without LLMClient or provider."""
        from luma.core.llm.schemas import LLMResponse
        
        parser = ResponseParser()
        response = LLMResponse(
            request_id="test",
            raw_text="Test response",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
            provider="mock"
        )
        
        parsed = parser.parse(response)
        assert isinstance(parsed, ParsedResponse)
        assert parsed.text == "Test response"
    
    def test_llm_client_with_different_providers(self, llm_config, mock_logger):
        """LLMClient works with different provider implementations."""
        # Test with MockProvider
        mock_provider = MockProvider(config={
            "responses": [{
                "text": "Mock response",
                "model": "test-model",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "provider": "mock"
            }]
        })
        
        client = ProviderLLMClient(
            provider=mock_provider,
            config=llm_config,
            logger=mock_logger
        )
        
        context = PromptContext(
            system_instructions="Test",
            user_profile="user",
            relevant_memories=[],
            current_input="Input",
            output_constraints="Constraints"
        )
        request = LLMRequest(
            prompt_context=context,
            model="test-model",
            temperature=0.7,
            max_tokens=256,
            request_id="test-req"
        )
        
        result = client.complete(request)
        assert result.raw_text == "Mock response"
        assert result.provider == "mock"


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_llm_interface_compatibility(self, llm_engine):
        """LLMEngine still implements LLMInterface for backward compatibility."""
        from luma.core.llm_interface import LLMInterface
        
        # Should be instance of LLMInterface
        assert isinstance(llm_engine, LLMInterface)
        
        # Should have generate_response method
        assert hasattr(llm_engine, "generate_response")
        assert callable(llm_engine.generate_response)
    
    def test_generate_response_method(self, llm_engine):
        """generate_response() method works as before."""
        result = llm_engine.generate_response(
            prompt="Test prompt",
            context={}
        )
        
        # Should return string
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_existing_code_patterns(self, prompt_builder, llm_client, response_parser, mock_logger):
        """Existing code patterns continue to work."""
        # Pattern 1: Direct component usage
        context = PromptContext(
            system_instructions="Test",
            user_profile="user",
            relevant_memories=[],
            current_input="Input",
            output_constraints="Constraints"
        )
        
        # Build prompt
        prompt = prompt_builder.build(context)
        assert isinstance(prompt, str)
        
        # Create request
        request = LLMRequest(
            prompt_context=context,
            model="test-model",
            temperature=0.7,
            max_tokens=256,
            request_id="test"
        )
        
        # Call client
        llm_response = llm_client.complete(request)
        assert hasattr(llm_response, "raw_text")
        
        # Parse response
        parsed = response_parser.parse(llm_response)
        assert isinstance(parsed, ParsedResponse)
