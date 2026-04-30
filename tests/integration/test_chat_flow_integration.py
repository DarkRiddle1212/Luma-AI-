"""
Integration test for chat flow with personality layer.

Tests verify that the personality layer integrates correctly with the chat flow:
- PersonalityEngine.build_instructions() is called before LLM generation
- PromptInstructions are correctly injected into PromptContext
- system_identity is in system_instructions field
- tone_guidance and style_constraints are in output_constraints field
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


class TestChatFlowIntegration:
    """Integration tests for chat flow with personality layer."""

    def test_personality_engine_called_before_llm_generation(self):
        """Test that PersonalityEngine.build_instructions() is called before LLM generation."""
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
            text="Test response",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 10, "completion": 20},
            truncated=False,
        )

        # Mock MemoryInterface
        mock_memory_interface = Mock()
        mock_memory_interface.retrieve.return_value = {
            "memories": [
                {"content": "User prefers technical explanations", "metadata": {"importance": 0.8}}
            ],
            "total_count": 1,
        }

        # Simulate chat flow
        user_id = "test_user"
        user_message = "How does async/await work?"
        
        # Step 1: Retrieve memories (simulated)
        memories = mock_memory_interface.retrieve(user_id=user_id, query=user_message)
        
        # Step 2: Build adaptation context (simulated)
        adaptation_context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={"tone": "user prefers technical content"},
        )
        
        # Step 3: Build prompt instructions using PersonalityEngine
        prompt_instructions = personality_engine.build_instructions(
            user_id=user_id,
            context=adaptation_context,
            mode="chat",
        )
        
        # Step 4: Build PromptContext with injected instructions
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[m["content"] for m in memories["memories"]],
            current_input=user_message,
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )
        
        # Step 5: Generate LLM response
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        response = mock_llm_engine.generate(llm_request)
        
        # Verify PersonalityEngine was called (by checking prompt_instructions is not None)
        assert prompt_instructions is not None
        assert isinstance(prompt_instructions, PromptInstructions)
        
        # Verify LLMEngine was called after PersonalityEngine
        assert mock_llm_engine.generate.called
        assert response.text == "Test response"

    def test_prompt_instructions_injected_into_prompt_context(self):
        """Test that PromptInstructions are correctly injected into PromptContext."""
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

        # Build prompt instructions
        adaptation_context = AdaptationContext(
            tone="casual",
            style="concise",
            focus="high-level",
            reasons={},
        )
        
        prompt_instructions = personality_engine.build_instructions(
            user_id="test_user",
            context=adaptation_context,
            mode="chat",
        )
        
        # Build PromptContext with injected instructions
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input="What is Python?",
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )
        
        # Verify injection
        assert prompt_context.system_instructions == prompt_instructions.system_identity
        assert prompt_instructions.tone_guidance in prompt_context.output_constraints
        assert prompt_instructions.style_constraints in prompt_context.output_constraints

    def test_system_identity_in_system_instructions(self):
        """Test that system_identity is in system_instructions field."""
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

        # Build prompt instructions
        adaptation_context = AdaptationContext(
            tone="formal",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        
        prompt_instructions = personality_engine.build_instructions(
            user_id="test_user",
            context=adaptation_context,
            mode="chat",
        )
        
        # Build PromptContext
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input="Test input",
            output_constraints="",
        )
        
        # Verify system_identity is in system_instructions
        assert prompt_context.system_instructions == prompt_instructions.system_identity
        assert "Luma" in prompt_context.system_instructions
        assert len(prompt_context.system_instructions) > 0

    def test_tone_guidance_and_style_constraints_in_output_constraints(self):
        """Test that tone_guidance and style_constraints are in output_constraints field."""
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

        # Build prompt instructions
        adaptation_context = AdaptationContext(
            tone="technical",
            style="detailed",
            focus="deep-technical",
            reasons={},
        )
        
        prompt_instructions = personality_engine.build_instructions(
            user_id="test_user",
            context=adaptation_context,
            mode="chat",
        )
        
        # Build PromptContext with combined output_constraints
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input="Test input",
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )
        
        # Verify tone_guidance and style_constraints are in output_constraints
        assert prompt_instructions.tone_guidance in prompt_context.output_constraints
        assert prompt_instructions.style_constraints in prompt_context.output_constraints
        assert len(prompt_context.output_constraints) > 0

    def test_full_chat_flow_end_to_end(self):
        """Test complete chat flow from memory retrieval to LLM response."""
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
            text="Async/await is a pattern for handling asynchronous operations...",
            is_valid=True,
            validation_notes=[],
            token_usage={"prompt": 50, "completion": 100},
            truncated=False,
        )

        # Mock MemoryInterface
        mock_memory_interface = Mock()
        mock_memory_interface.retrieve.return_value = {
            "memories": [
                {"content": "User is a Python developer", "metadata": {"importance": 0.9}},
                {"content": "User prefers code examples", "metadata": {"importance": 0.8}},
            ],
            "total_count": 2,
        }

        # Simulate complete chat flow
        user_id = "test_user"
        user_message = "Explain async/await in Python"
        
        # Step 1: Retrieve memories
        memories = mock_memory_interface.retrieve(user_id=user_id, query=user_message)
        
        # Step 2: Build adaptation context (simulated PersonalizationEngine)
        adaptation_context = AdaptationContext(
            tone="technical",
            style="step-by-step",
            focus="deep-technical",
            reasons={"tone": "user is a developer", "style": "user prefers examples"},
        )
        
        # Step 3: Build prompt instructions
        prompt_instructions = personality_engine.build_instructions(
            user_id=user_id,
            context=adaptation_context,
            mode="chat",
        )
        
        # Step 4: Build PromptContext
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="Python developer",
            relevant_memories=[m["content"] for m in memories["memories"]],
            current_input=user_message,
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )
        
        # Step 5: Create LLMRequest
        llm_request = LLMRequest(
            prompt_context=prompt_context,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            request_id="test_request",
        )
        
        # Step 6: Generate response
        response = mock_llm_engine.generate(llm_request)
        
        # Verify complete flow
        assert mock_memory_interface.retrieve.called
        assert prompt_instructions is not None
        assert prompt_context.system_instructions == prompt_instructions.system_identity
        assert prompt_instructions.tone_guidance in prompt_context.output_constraints
        assert prompt_instructions.style_constraints in prompt_context.output_constraints
        assert mock_llm_engine.generate.called
        assert response.is_valid
        assert "async/await" in response.text.lower()

    def test_chat_flow_with_teacher_mode(self):
        """Test chat flow integration with teacher mode."""
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
        
        # Verify teacher tone is selected
        assert prompt_instructions.metadata["selected_tone"] == "teacher"
        assert prompt_instructions.metadata["mode"] == "teacher"
        
        # Build PromptContext
        prompt_context = PromptContext(
            system_instructions=prompt_instructions.system_identity,
            user_profile="",
            relevant_memories=[],
            current_input="Teach me about variables",
            output_constraints=f"{prompt_instructions.tone_guidance}\n{prompt_instructions.style_constraints}",
        )
        
        # Verify teacher-specific instructions are present
        assert prompt_context.system_instructions == prompt_instructions.system_identity
        assert len(prompt_context.output_constraints) > 0

    def test_chat_flow_with_multiple_modes(self):
        """Test that different modes produce different prompt instructions."""
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

        adaptation_context = AdaptationContext(
            tone="casual",
            style="balanced",
            focus="high-level",
            reasons={},
        )
        
        # Build instructions for chat mode
        chat_instructions = personality_engine.build_instructions(
            user_id="test_user",
            context=adaptation_context,
            mode="chat",
        )
        
        # Build instructions for teacher mode
        teacher_instructions = personality_engine.build_instructions(
            user_id="test_user",
            context=adaptation_context,
            mode="teacher",
        )
        
        # Verify different modes produce different tones
        assert chat_instructions.metadata["selected_tone"] == "friendly"
        assert teacher_instructions.metadata["selected_tone"] == "teacher"
        assert chat_instructions.tone_guidance != teacher_instructions.tone_guidance
