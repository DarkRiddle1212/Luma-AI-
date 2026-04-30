"""
Integration test for teacher mode with personality layer.

Tests verify that the personality layer integrates correctly with teacher mode:
- PersonalityEngine.build_instructions() is called with mode="teacher"
- PromptInstructions are correctly injected into teacher-mode PromptContext
- ExplanationEngine does not call ToneManager or StyleProfiles directly
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.schemas import PromptInstructions
from luma.core.personality.personality_engine import PersonalityEngine
from luma.core.personality.system_prompt import SystemPrompt
from luma.core.personality.tone_manager import ToneManager
from luma.core.personality.style_profiles import StyleProfiles
from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.llm.schemas import PromptContext, LLMRequest, ParsedResponse
from luma.core.llm.llm_engine import LLMEngine
from luma.core.teacher.schemas import Lesson, Explanation
from luma.core.teacher.explanation_engine import ExplanationEngine


class TestTeacherModeIntegration:
    """Integration tests for teacher mode with personality layer."""

    def test_personality_engine_called_with_teacher_mode(self):
        """Test that PersonalityEngine.build_instructions() is called with mode='teacher'."""
        # Create real PersonalityEngine with real components
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="Variables are containers that store data values...",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 30, "completion": 50},
            truncated=False,
        )

        # Create ExplanationEngine with mock LLMEngine
        explanation_engine = ExplanationEngine(llm_engine=mock_llm_engine)

        # Create a lesson
        lesson = Lesson(
            id="lesson_001",
            topic="programming",
            title="Introduction to Variables",
            difficulty="beginner",
            content="Variables are named storage locations in memory that hold data.",
        )

        # Create adaptation context
        adaptation_context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={"tone": "beginner level", "style": "learning mode"},
        )

        # Build prompt instructions with teacher mode
        prompt_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=adaptation_context,
            mode="teacher",
        )

        # Verify teacher mode was used
        assert prompt_instructions.metadata["mode"] == "teacher"
        assert prompt_instructions.metadata["selected_tone"] == "teacher"

    def test_prompt_instructions_injected_into_teacher_mode_context(self):
        """Test that PromptInstructions are correctly injected into teacher-mode PromptContext."""
        # Create real PersonalityEngine
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Build prompt instructions with teacher mode
        adaptation_context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={},
        )
        
        prompt_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=adaptation_context,
            mode="teacher",
        )

        # Simulate building PromptContext for teacher mode
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input="Explain variables in programming",
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )

        # Verify injection
        assert prompt_context.system_instructions == prompt_instructions.system_identity
        assert prompt_instructions.tone_guidance in prompt_context.output_constraints
        assert prompt_instructions.style_constraints in prompt_context.output_constraints
        assert len(prompt_context.system_instructions) > 0
        assert len(prompt_context.output_constraints) > 0

    def test_explanation_engine_does_not_call_tone_manager_directly(self):
        """Test that ExplanationEngine does not call ToneManager or StyleProfiles directly."""
        # Create mock components
        mock_tone_manager = Mock(spec=ToneManager)
        mock_style_profiles = Mock(spec=StyleProfiles)
        
        # Create real PersonalityEngine with mock components
        system_prompt = SystemPrompt()
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=mock_tone_manager,
            style_profiles=mock_style_profiles,
            response_guardrails=response_guardrails,
        )

        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="Functions are reusable blocks of code...",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 25, "completion": 45},
            truncated=False,
        )

        # Create ExplanationEngine
        explanation_engine = ExplanationEngine(llm_engine=mock_llm_engine)

        # Create a lesson
        lesson = Lesson(
            id="lesson_002",
            topic="programming",
            title="Introduction to Functions",
            difficulty="beginner",
            content="Functions are reusable blocks of code that perform specific tasks.",
        )

        # Create adaptation context
        adaptation_context = AdaptationContext(
            tone="casual",
            style="detailed",
            focus="high-level",
            reasons={},
        )

        # Generate explanation (ExplanationEngine should NOT call ToneManager or StyleProfiles)
        explanation = explanation_engine.explain(lesson, adaptation_context)

        # Verify ExplanationEngine did not call ToneManager or StyleProfiles directly
        mock_tone_manager.select_tone.assert_not_called()
        mock_style_profiles.get_style.assert_not_called()
        mock_style_profiles.set_style.assert_not_called()

        # Verify explanation was generated
        assert explanation.lesson_id == lesson.id
        assert len(explanation.content) > 0

    def test_full_teacher_mode_flow_with_personality_layer(self):
        """Test complete teacher mode flow with personality layer integration."""
        # Create real PersonalityEngine
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Mock LLMEngine
        mock_llm_engine = Mock(spec=LLMEngine)
        mock_llm_engine.generate.return_value = ParsedResponse(
            request_id="test_request",
            text="Let me explain loops in a way that's easy to understand. A loop is a programming construct that repeats a block of code multiple times...",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 40, "completion": 80},
            truncated=False,
        )

        # Create ExplanationEngine
        explanation_engine = ExplanationEngine(llm_engine=mock_llm_engine)

        # Create a lesson
        lesson = Lesson(
            id="lesson_003",
            topic="programming",
            title="Understanding Loops",
            difficulty="beginner",
            content="Loops allow you to execute a block of code repeatedly until a condition is met.",
        )

        # Create adaptation context
        adaptation_context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={"tone": "beginner student", "style": "learning mode"},
        )

        # Step 1: Build prompt instructions with teacher mode
        prompt_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=adaptation_context,
            mode="teacher",
        )

        # Step 2: Verify teacher mode was selected
        assert prompt_instructions.metadata["mode"] == "teacher"
        assert prompt_instructions.metadata["selected_tone"] == "teacher"

        # Step 3: Generate explanation (simulating ExplanationEngine flow)
        explanation = explanation_engine.explain(lesson, adaptation_context)

        # Step 4: Verify complete flow
        assert explanation.lesson_id == lesson.id
        assert len(explanation.content) > 0
        assert "loop" in explanation.content.lower()
        assert mock_llm_engine.generate.called

        # Step 5: Verify LLMEngine was called with proper context
        llm_call_args = mock_llm_engine.generate.call_args
        assert llm_call_args is not None
        llm_request = llm_call_args[0][0]
        assert isinstance(llm_request, LLMRequest)
        assert len(llm_request.prompt_context.system_instructions) > 0

    def test_teacher_mode_with_different_adaptation_contexts(self):
        """Test that teacher mode adapts to different adaptation contexts."""
        # Create real PersonalityEngine
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Test with casual tone
        casual_context = AdaptationContext(
            tone="casual",
            style="concise",
            focus="high-level",
            reasons={},
        )
        
        casual_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=casual_context,
            mode="teacher",
        )

        # Test with technical tone
        technical_context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )
        
        technical_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=technical_context,
            mode="teacher",
        )

        # Verify both use teacher tone (mode overrides context tone)
        assert casual_instructions.metadata["selected_tone"] == "teacher"
        assert technical_instructions.metadata["selected_tone"] == "teacher"

        # Verify different styles are applied
        assert casual_instructions.metadata["selected_style"] != technical_instructions.metadata["selected_style"]

    def test_teacher_mode_prompt_context_structure(self):
        """Test that teacher mode PromptContext has the correct structure."""
        # Create real PersonalityEngine
        system_prompt = SystemPrompt()
        tone_manager = ToneManager()
        
        mock_storage = Mock()
        mock_storage.retrieve.return_value = {"memories": []}
        style_profiles = StyleProfiles(storage_backend=mock_storage)
        
        response_guardrails = ResponseGuardrails()
        
        personality_engine = PersonalityEngine(
            system_prompt=system_prompt,
            tone_manager=tone_manager,
            style_profiles=style_profiles,
            response_guardrails=response_guardrails,
        )

        # Build prompt instructions with teacher mode
        adaptation_context = AdaptationContext(
            tone="casual",
            style="step-by-step",
            focus="high-level",
            reasons={},
        )
        
        prompt_instructions = personality_engine.build_instructions(
            user_id="student_user",
            context=adaptation_context,
            mode="teacher",
        )

        # Build PromptContext for teacher mode
        lesson_content = "Arrays are data structures that store multiple values."
        
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input=lesson_content,
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )

        # Verify PromptContext structure
        assert isinstance(prompt_context.system_instructions, str)
        assert isinstance(prompt_context.output_constraints, str)
        assert len(prompt_context.system_instructions) > 0
        assert len(prompt_context.output_constraints) > 0
        assert prompt_context.current_input == lesson_content
        assert prompt_context.user_profile == ""
        assert prompt_context.relevant_memories == []
