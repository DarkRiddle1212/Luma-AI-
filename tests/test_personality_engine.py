"""
Unit tests for PersonalityEngine.

Tests verify:
- Constructor accepts all dependencies
- build_instructions() orchestrates all sub-components
- Output completeness (non-empty fields)
- Determinism (identical inputs → identical outputs)
- Exception handling (raises PersonalityError)
- Logging integration
"""

import pytest
from unittest.mock import Mock, MagicMock

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.schemas import (
    ToneSelection,
    StylePreference,
    PromptInstructions,
    PersonalityError,
)
from luma.core.personality.system_prompt import SystemPrompt
from luma.core.personality.tone_manager import ToneManager
from luma.core.personality.style_profiles import StyleProfiles
from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.personality.personality_engine import PersonalityEngine
from luma.core.structured_logger import StructuredLogger


class TestPersonalityEngineConstruction:
    """Test PersonalityEngine constructor and dependency injection."""

    def test_constructor_accepts_all_dependencies(self):
        """Test that constructor accepts all required dependencies."""
        system_prompt = Mock(spec=SystemPrompt)
        tone_manager = Mock(spec=ToneManager)
        style_profiles = Mock(spec=StyleProfiles)
        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        assert engine._system_prompt is system_prompt
        assert engine._tone_manager is tone_manager
        assert engine._style_profiles is style_profiles
        assert engine._response_guardrails is response_guardrails
        assert engine._logger is logger

    def test_constructor_creates_noop_logger_when_none_provided(self):
        """Test that constructor creates a no-op logger when logger is None."""
        system_prompt = Mock(spec=SystemPrompt)
        tone_manager = Mock(spec=ToneManager)
        style_profiles = Mock(spec=StyleProfiles)
        response_guardrails = Mock(spec=ResponseGuardrails)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=None,
        )

        assert engine._logger is not None
        assert isinstance(engine._logger, StructuredLogger)


class TestPersonalityEngineOrchestration:
    """Test PersonalityEngine orchestration logic."""

    def test_build_instructions_calls_all_subcomponents(self):
        """Test that build_instructions() calls all sub-components in order."""
        # Create mocks
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default tone",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {
            "friendly": "Use conversational language",
        }

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced communication",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user123",
            context=context,
            mode="chat",
        )

        # Verify all sub-components were called
        system_prompt.get_identity.assert_called_once()
        tone_manager.select_tone.assert_called_once_with(
            context=context,
            mode="chat",
            user_preference=None,
        )
        style_profiles.get_style.assert_called_once_with("user123")

        # Verify result is a PromptInstructions object
        assert isinstance(result, PromptInstructions)

    def test_build_instructions_output_completeness(self):
        """Test that build_instructions() returns complete PromptInstructions."""
        # Create mocks with real return values
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="professional",
            rationale="formal context",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {
            "professional": "Use formal language",
        }

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="detailed_explanations",
            description="Comprehensive coverage",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="formal",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user456",
            context=context,
            mode="chat",
        )

        # Verify output completeness
        assert result.system_identity != ""
        assert result.tone_guidance != ""
        assert result.style_constraints != ""
        assert len(result.output_rules) > 0
        assert result.metadata["user_id"] == "user456"
        assert result.metadata["mode"] == "chat"
        assert result.metadata["selected_tone"] == "professional"
        assert result.metadata["selected_style"] == "detailed_explanations"

    def test_build_instructions_includes_output_rules(self):
        """Test that build_instructions() includes all required output rules."""
        # Create mocks
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(tone="casual", style="balanced", focus="high-level", reasons={})
        result = engine.build_instructions(
            user_id="user789",
            context=context,
            mode="chat",
        )

        # Verify output rules are present
        assert len(result.output_rules) == 5
        assert any("rambling" in rule.lower() for rule in result.output_rules)
        assert any("repetition" in rule.lower() for rule in result.output_rules)
        assert any("contradiction" in rule.lower() for rule in result.output_rules)
        assert any("vague" in rule.lower() for rule in result.output_rules)
        assert any("length" in rule.lower() for rule in result.output_rules)


class TestPersonalityEngineDeterminism:
    """Test PersonalityEngine determinism."""

    def test_build_instructions_determinism(self):
        """Test that identical inputs produce identical outputs."""
        # Create mocks with deterministic behavior
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="technical",
            rationale="technical context",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"technical": "Use technical terms"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="technical_depth",
            description="Technical details",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions twice with identical inputs
        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )

        result1 = engine.build_instructions(
            user_id="user_determinism",
            context=context,
            mode="chat",
        )

        result2 = engine.build_instructions(
            user_id="user_determinism",
            context=context,
            mode="chat",
        )

        # Verify identical outputs
        assert result1.system_identity == result2.system_identity
        assert result1.tone_guidance == result2.tone_guidance
        assert result1.style_constraints == result2.style_constraints
        assert result1.output_rules == result2.output_rules
        assert result1.metadata == result2.metadata


class TestPersonalityEngineErrorHandling:
    """Test PersonalityEngine error handling."""

    def test_build_instructions_raises_personality_error_on_exception(self):
        """Test that exceptions from sub-components raise PersonalityError."""
        # Create mocks where system_prompt raises an exception
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.side_effect = ValueError("Invalid personality")

        tone_manager = Mock(spec=ToneManager)
        style_profiles = Mock(spec=StyleProfiles)
        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions and expect PersonalityError
        context = AdaptationContext(tone="casual", style="balanced", focus="high-level", reasons={})

        with pytest.raises(PersonalityError) as exc_info:
            engine.build_instructions(
                user_id="user_error",
                context=context,
                mode="chat",
            )

        # Verify the exception message and cause
        assert "Failed to build prompt instructions" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_build_instructions_logs_error_on_exception(self):
        """Test that exceptions are logged before raising PersonalityError."""
        # Create mocks where tone_manager raises an exception
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.side_effect = RuntimeError("Tone selection failed")

        style_profiles = Mock(spec=StyleProfiles)
        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions and expect PersonalityError
        context = AdaptationContext(tone="casual", style="balanced", focus="high-level", reasons={})

        with pytest.raises(PersonalityError):
            engine.build_instructions(
                user_id="user_log_error",
                context=context,
                mode="chat",
            )

        # Verify error was logged
        assert logger.log.call_count >= 2  # Start log + error log
        error_log_call = [
            call for call in logger.log.call_args_list
            if call[0][0] == "personality_engine_error"
        ]
        assert len(error_log_call) == 1


class TestPersonalityEngineCallOrder:
    """Test PersonalityEngine sub-component call order."""

    def test_build_instructions_calls_components_in_correct_order(self):
        """Test that sub-components are called in the correct order: SystemPrompt -> ToneManager -> StyleProfiles."""
        call_order = []

        # Create mocks that track call order
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.side_effect = lambda *args, **kwargs: (
            call_order.append("system_prompt") or "You are Luma"
        )

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.side_effect = lambda *args, **kwargs: (
            call_order.append("tone_manager")
            or ToneSelection(
                tone="friendly",
                rationale="default",
                context_signals={},
            )
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.side_effect = lambda *args, **kwargs: (
            call_order.append("style_profiles")
            or StylePreference(
                style="high_signal_low_noise",
                description="Balanced",
                active=True,
            )
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        engine.build_instructions(
            user_id="user_order",
            context=context,
            mode="chat",
        )

        # Verify call order
        assert call_order == ["system_prompt", "tone_manager", "style_profiles"]

    def test_build_instructions_passes_correct_parameters_to_tone_manager(self):
        """Test that build_instructions passes correct parameters to ToneManager.select_tone."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions with specific parameters
        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={"reason": "test"},
        )
        engine.build_instructions(
            user_id="user_params",
            context=context,
            mode="teacher",
        )

        # Verify ToneManager.select_tone was called with correct parameters
        tone_manager.select_tone.assert_called_once_with(
            context=context,
            mode="teacher",
            user_preference=None,
        )

    def test_build_instructions_passes_correct_user_id_to_style_profiles(self):
        """Test that build_instructions passes correct user_id to StyleProfiles.get_style."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions with specific user_id
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        engine.build_instructions(
            user_id="specific_user_123",
            context=context,
            mode="chat",
        )

        # Verify StyleProfiles.get_style was called with correct user_id
        style_profiles.get_style.assert_called_once_with("specific_user_123")


class TestPersonalityEngineOutputStructure:
    """Test PersonalityEngine output structure and metadata."""

    def test_build_instructions_includes_tone_rationale_in_metadata(self):
        """Test that build_instructions includes tone_rationale in metadata."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="analytical",
            rationale="technical context detected",
            context_signals={"signal": "value"},
        )
        tone_manager.TONE_GUIDANCE = {"analytical": "Use logical reasoning"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="technical_depth",
            description="Technical details",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions with valid AdaptationContext tone
        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user_metadata",
            context=context,
            mode="chat",
        )

        # Verify metadata includes tone_rationale
        assert "tone_rationale" in result.metadata
        assert result.metadata["tone_rationale"] == "technical context detected"

    def test_build_instructions_output_rules_are_non_empty_strings(self):
        """Test that all output rules are non-empty strings."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user_rules",
            context=context,
            mode="chat",
        )

        # Verify all output rules are non-empty strings
        assert len(result.output_rules) > 0
        for rule in result.output_rules:
            assert isinstance(rule, str)
            assert len(rule) > 0

    def test_build_instructions_uses_tone_guidance_from_tone_manager(self):
        """Test that build_instructions uses tone guidance from ToneManager.TONE_GUIDANCE."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="motivational",
            rationale="motivational context",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {
            "motivational": "Use encouraging and positive language",
        }

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="motivational_style",
            description="Motivational",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user_guidance",
            context=context,
            mode="chat",
        )

        # Verify tone_guidance matches TONE_GUIDANCE mapping
        assert result.tone_guidance == "Use encouraging and positive language"

    def test_build_instructions_uses_style_constraints_from_style_profiles(self):
        """Test that build_instructions uses style constraints from STYLE_CONSTRAINTS."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="step_by_step",
            description="Step-by-step guidance",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={},
        )
        result = engine.build_instructions(
            user_id="user_constraints",
            context=context,
            mode="chat",
        )

        # Verify style_constraints is from STYLE_CONSTRAINTS
        from luma.core.personality.style_profiles import STYLE_CONSTRAINTS

        expected_constraints = STYLE_CONSTRAINTS["step_by_step"]
        assert result.style_constraints == expected_constraints


class TestPersonalityEngineErrorHandlingEdgeCases:
    """Test PersonalityEngine error handling edge cases."""

    def test_build_instructions_handles_style_profiles_exception(self):
        """Test that exceptions from StyleProfiles raise PersonalityError."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.side_effect = RuntimeError("Style retrieval failed")

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions and expect PersonalityError
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )

        with pytest.raises(PersonalityError) as exc_info:
            engine.build_instructions(
                user_id="user_style_error",
                context=context,
                mode="chat",
            )

        # Verify the exception message and cause
        assert "Failed to build prompt instructions" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_build_instructions_logs_error_details(self):
        """Test that error logs include user_id, mode, error message, and error type."""
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.side_effect = ValueError("Test error")

        tone_manager = Mock(spec=ToneManager)
        style_profiles = Mock(spec=StyleProfiles)
        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions and expect PersonalityError
        context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )

        with pytest.raises(PersonalityError):
            engine.build_instructions(
                user_id="user_error_details",
                context=context,
                mode="teacher",
            )

        # Verify error log contains all required fields
        error_log_calls = [
            call
            for call in logger.log.call_args_list
            if call[0][0] == "personality_engine_error"
        ]
        assert len(error_log_calls) == 1

        error_payload = error_log_calls[0][0][1]
        assert error_payload["user_id"] == "user_error_details"
        assert error_payload["mode"] == "teacher"
        assert "error" in error_payload
        assert "error_type" in error_payload
        assert error_payload["error_type"] == "ValueError"


class TestPersonalityEngineLogging:
    """Test PersonalityEngine logging integration."""

    def test_build_instructions_logs_start_event(self):
        """Test that build_instructions() logs a start event."""
        # Create mocks
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="friendly",
            rationale="default",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"friendly": "Be friendly"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="high_signal_low_noise",
            description="Balanced",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(tone="casual", style="balanced", focus="high-level", reasons={})
        engine.build_instructions(
            user_id="user_log_start",
            context=context,
            mode="chat",
        )

        # Verify start event was logged
        start_log_calls = [
            call for call in logger.log.call_args_list
            if call[0][0] == "building_prompt_instructions"
        ]
        assert len(start_log_calls) == 1

        # Verify start event payload
        start_payload = start_log_calls[0][0][1]
        assert start_payload["user_id"] == "user_log_start"
        assert start_payload["mode"] == "chat"
        assert start_payload["context_tone"] == "casual"

    def test_build_instructions_logs_end_event(self):
        """Test that build_instructions() logs an end event."""
        # Create mocks
        system_prompt = Mock(spec=SystemPrompt)
        system_prompt.get_identity.return_value = "You are Luma"

        tone_manager = Mock(spec=ToneManager)
        tone_manager.select_tone.return_value = ToneSelection(
            tone="professional",
            rationale="formal context",
            context_signals={},
        )
        tone_manager.TONE_GUIDANCE = {"professional": "Be professional"}

        style_profiles = Mock(spec=StyleProfiles)
        style_profiles.get_style.return_value = StylePreference(
            style="detailed_explanations",
            description="Detailed",
            active=True,
        )

        response_guardrails = Mock(spec=ResponseGuardrails)
        logger = Mock(spec=StructuredLogger)

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Call build_instructions
        context = AdaptationContext(tone="formal", style="detailed", focus="deep-technical", reasons={})
        engine.build_instructions(
            user_id="user_log_end",
            context=context,
            mode="teacher",
        )

        # Verify end event was logged
        end_log_calls = [
            call for call in logger.log.call_args_list
            if call[0][0] == "prompt_instructions_built"
        ]
        assert len(end_log_calls) == 1

        # Verify end event payload
        end_payload = end_log_calls[0][0][1]
        assert end_payload["user_id"] == "user_log_end"
        assert end_payload["selected_tone"] == "professional"
        assert end_payload["selected_style"] == "detailed_explanations"
        assert end_payload["output_rules_count"] == 5
