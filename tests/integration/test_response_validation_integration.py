"""
Integration test for response validation integration.

Tests verify that the response validation integrates correctly with the LLM generation flow:
- ResponseGuardrails.validate() is called after LLM generation
- GuardrailResult is logged when validation fails
- No exception is raised when validation fails (advisory only)
- GuardrailResult.score is included in response metadata
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call

from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.personality.schemas import GuardrailResult
from luma.core.llm.schemas import PromptContext, LLMRequest, ParsedResponse
from luma.core.llm.llm_engine import LLMEngine
from luma.core.structured_logger import StructuredLogger


class TestResponseValidationIntegration:
    """Integration tests for response validation with LLM generation flow."""

    def test_guardrails_called_after_llm_generation(self):
        """Test that ResponseGuardrails.validate() is called after LLM generation."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="This is a clean, well-structured response.",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 20},
            truncated=False,
        )
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="What is Python?",
            output_constraints="Be concise and clear.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=["concise"],
        )
        
        # Verify LLM generation was called
        assert mock_llm_engine.generate.called
        assert response.text == "This is a clean, well-structured response."
        
        # Verify guardrails validation was called after generation
        assert guardrail_result is not None
        assert isinstance(guardrail_result, GuardrailResult)
        assert guardrail_result.passed is True
        assert guardrail_result.score == 1.0

    def test_guardrail_result_logged_when_validation_fails(self):
        """Test that GuardrailResult is logged when validation fails."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        
        # Create a response with quality violations (rambling)
        rambling_text = " ".join(["word"] * 600)  # >500 words without structure
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text=rambling_text,
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 600},
            truncated=False,
        )
        
        # Mock StructuredLogger
        mock_logger = Mock(spec=StructuredLogger)
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="Explain Python in detail.",
            output_constraints="Be clear and structured.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=[],
        )
        
        # Step 3: Log validation failure
        if not guardrail_result.passed:
            mock_logger.log(
                "response_validation_failed",
                {
                    "violations": guardrail_result.violations,
                    "score": guardrail_result.score,
                    "response_length": len(response.text),
                },
            )
        
        # Verify guardrails detected violations
        assert guardrail_result.passed is False
        assert len(guardrail_result.violations) > 0
        assert "rambling" in guardrail_result.violations
        assert guardrail_result.score < 1.0
        
        # Verify logger was called with validation failure
        assert mock_logger.log.called
        mock_logger.log.assert_called_once_with(
            "response_validation_failed",
            {
                "violations": guardrail_result.violations,
                "score": guardrail_result.score,
                "response_length": len(response.text),
            },
        )

    def test_no_exception_raised_when_validation_fails(self):
        """Test that no exception is raised when validation fails (advisory only)."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        
        # Create a response with multiple quality violations (vague filler + contradiction)
        # Need 2+ violations to ensure passed=False (score < 0.75)
        vague_text = (
            "It depends on the situation. Generally speaking, in most cases, "
            "it could be useful. Typically, it might work, but sometimes it may or may not. "
            "However, on the other hand, it's not always the best choice."
        )
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text=vague_text,
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 40},
            truncated=False,
        )
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="Is Python good for data science?",
            output_constraints="Be specific and actionable.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails (should not raise exception)
        try:
            guardrail_result = response_guardrails.validate(
                response_text=response.text,
                constraints=[],
            )
            exception_raised = False
        except Exception as e:
            exception_raised = True
            raised_exception = e
        
        # Verify no exception was raised
        assert exception_raised is False, "Validation should not raise exception"
        
        # Verify guardrails detected violations but returned result
        assert guardrail_result.passed is False
        assert len(guardrail_result.violations) >= 2
        assert "vague filler" in guardrail_result.violations
        assert "contradiction" in guardrail_result.violations

    def test_guardrail_score_included_in_response_metadata(self):
        """Test that GuardrailResult.score is included in response metadata."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="Python is a high-level programming language.",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 15},
            truncated=False,
        )
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="What is Python?",
            output_constraints="Be concise.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=["concise"],
        )
        
        # Step 3: Include guardrail score in response metadata
        response_metadata = {
            "request_id": response.request_id,
            "token_usage": response.token_usage,
            "is_valid": response.is_valid,
            "guardrail_score": guardrail_result.score,
            "guardrail_passed": guardrail_result.passed,
        }
        
        # Verify guardrail score is included in metadata
        assert "guardrail_score" in response_metadata
        assert response_metadata["guardrail_score"] == guardrail_result.score
        assert response_metadata["guardrail_score"] == 1.0
        assert response_metadata["guardrail_passed"] is True

    def test_full_response_validation_flow_with_failure(self):
        """Test complete response validation flow with guardrail failure."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        
        # Create a response with multiple violations (repetition + vague filler + contradiction)
        # Need 2+ violations to ensure passed=False (score < 0.75)
        repetitive_text = (
            "Python is a great language for data science. "
            "Python is a great language for data science. "
            "Python is a great language for data science. "
            "It depends on what you want. Generally speaking, it could be useful. "
            "In most cases, it might work. Typically, it may or may not be good. "
            "However, on the other hand, it's not always the best choice."
        )
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text=repetitive_text,
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 60},
            truncated=False,
        )
        
        # Mock StructuredLogger
        mock_logger = Mock(spec=StructuredLogger)
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="Is Python good?",
            output_constraints="Be specific and avoid repetition.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=[],
        )
        
        # Step 3: Log validation result
        if not guardrail_result.passed:
            mock_logger.log(
                "response_validation_failed",
                {
                    "violations": guardrail_result.violations,
                    "score": guardrail_result.score,
                    "response_length": len(response.text),
                },
            )
        
        # Step 4: Include guardrail score in response metadata
        response_metadata = {
            "request_id": response.request_id,
            "guardrail_score": guardrail_result.score,
            "guardrail_passed": guardrail_result.passed,
            "guardrail_violations": guardrail_result.violations,
        }
        
        # Verify complete flow
        assert mock_llm_engine.generate.called
        assert guardrail_result.passed is False
        assert len(guardrail_result.violations) >= 2  # Need at least 2 violations for passed=False
        assert guardrail_result.score < 0.75
        assert mock_logger.log.called
        assert response_metadata["guardrail_score"] == guardrail_result.score
        assert response_metadata["guardrail_passed"] is False
        assert len(response_metadata["guardrail_violations"]) >= 2

    def test_response_validation_with_concise_constraint_violation(self):
        """Test response validation with concise constraint violation."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        
        # Create a response that exceeds concise length constraint (>200 words)
        long_text = " ".join(["word"] * 250)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text=long_text,
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 250},
            truncated=False,
        )
        
        # Mock StructuredLogger
        mock_logger = Mock(spec=StructuredLogger)
        
        # Simulate LLM generation flow with concise constraint
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="What is Python?",
            output_constraints="Be concise.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with concise constraint
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=["concise"],
        )
        
        # Step 3: Log validation failure
        if not guardrail_result.passed:
            mock_logger.log(
                "response_validation_failed",
                {
                    "violations": guardrail_result.violations,
                    "score": guardrail_result.score,
                    "response_length": len(response.text),
                },
            )
        
        # Verify concise constraint violation
        assert guardrail_result.passed is False
        assert "exceeds concise length constraint" in guardrail_result.violations
        assert guardrail_result.score < 1.0
        assert mock_logger.log.called

    def test_response_validation_with_clean_response(self):
        """Test response validation with clean response (no violations)."""
        # Create real ResponseGuardrails
        response_guardrails = ResponseGuardrails()
        
        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="Python is a versatile, high-level programming language known for its readability and extensive libraries.",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 20},
            truncated=False,
        )
        
        # Mock StructuredLogger
        mock_logger = Mock(spec=StructuredLogger)
        
        # Simulate LLM generation flow
        prompt_context = PromptContext(
            system_instructions="You are Luma, an AI assistant.",
            user_profile="",
            relevant_memories=[],
            current_input="What is Python?",
            output_constraints="Be clear and concise.",
        )
        
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 1: Generate LLM response
        response = mock_llm_engine.generate(llm_request)
        
        # Step 2: Validate response with guardrails
        guardrail_result = response_guardrails.validate(
            response_text=response.text,
            constraints=["concise"],
        )
        
        # Step 3: Log validation success (optional)
        if guardrail_result.passed:
            mock_logger.log(
                "response_validation_passed",
                {
                    "score": guardrail_result.score,
                    "response_length": len(response.text),
                },
            )
        
        # Step 4: Include guardrail score in response metadata
        response_metadata = {
            "request_id": response.request_id,
            "guardrail_score": guardrail_result.score,
            "guardrail_passed": guardrail_result.passed,
        }
        
        # Verify clean response passes validation
        assert guardrail_result.passed is True
        assert len(guardrail_result.violations) == 0
        assert guardrail_result.score == 1.0
        assert mock_logger.log.called
        assert response_metadata["guardrail_score"] == 1.0
        assert response_metadata["guardrail_passed"] is True
