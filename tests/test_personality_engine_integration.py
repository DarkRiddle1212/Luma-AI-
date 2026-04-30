"""
Integration tests for PersonalityEngine with real components.

Tests verify that PersonalityEngine works correctly when integrated with
real SystemPrompt, ToneManager, StyleProfiles, and ResponseGuardrails components.
"""

import pytest
from unittest.mock import Mock

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.schemas import PromptInstructions
from luma.core.personality.system_prompt import SystemPrompt
from luma.core.personality.tone_manager import ToneManager
from luma.core.personality.style_profiles import StyleProfiles
from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.personality.personality_engine import PersonalityEngine
from luma.core.structured_logger import StructuredLogger


class TestPersonalityEngineIntegration:
    """Integration tests with real components."""

    def test_build_instructions_with_real_components(self):
        """Test PersonalityEngine with real sub-components."""
        # Create real components
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        # Mock storage backend for StyleProfiles
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        logger = Mock(spec=StructuredLogger)

        # Create PersonalityEngine
        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
            logger=logger,
        )

        # Build instructions
        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={"tone": "user is asking technical questions"},
        )

        result = engine.build_instructions(
            user_id="test_user",
            context=context,
            mode="chat",
        )

        # Verify result
        assert isinstance(result, PromptInstructions)
        assert "Luma" in result.system_identity
        assert result.tone_guidance != ""
        assert result.style_constraints != ""
        assert len(result.output_rules) == 5
        assert result.metadata["selected_tone"] == "technical"
        assert result.metadata["selected_style"] == "high_signal_low_noise"

    def test_build_instructions_teacher_mode(self):
        """Test PersonalityEngine in teacher mode."""
        # Create real components
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        # Mock storage backend for StyleProfiles
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()

        # Create PersonalityEngine without logger
        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Build instructions for teacher mode
        context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={},
        )

        result = engine.build_instructions(
            user_id="student_user",
            context=context,
            mode="teacher",
        )

        # Verify teacher tone is selected
        assert result.metadata["selected_tone"] == "teacher"
        assert "teacher" in result.tone_guidance.lower() or "explanatory" in result.tone_guidance.lower()

    def test_build_instructions_with_different_personalities(self):
        """Test PersonalityEngine with different personality profiles."""
        # Create components
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        context = AdaptationContext(
            tone="formal",
            style="concise",
            focus="high-level",
            reasons={},
        )

        # Test default personality
        result = engine.build_instructions(
            user_id="user1",
            context=context,
            mode="chat",
        )

        assert "Luma" in result.system_identity
        assert "intelligent" in result.system_identity.lower() or "practical" in result.system_identity.lower()

    def test_determinism_across_multiple_calls(self):
        """Test that multiple calls with identical inputs produce identical outputs."""
        # Create components
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()

        engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )

        # Call multiple times
        results = []
        for _ in range(3):
            result = engine.build_instructions(
                user_id="determinism_test",
                context=context,
                mode="chat",
            )
            results.append(result)

        # Verify all results are identical
        for i in range(1, len(results)):
            assert results[i].system_identity == results[0].system_identity
            assert results[i].tone_guidance == results[0].tone_guidance
            assert results[i].style_constraints == results[0].style_constraints
            assert results[i].output_rules == results[0].output_rules
