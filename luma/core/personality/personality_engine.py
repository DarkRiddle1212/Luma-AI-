"""
PersonalityEngine: Orchestrator for the personality layer.

Combines system prompt, tone selection, style profile, and guardrails to produce
final prompt instructions for the LLM layer. Stateless component with deterministic
behavior.
"""

from typing import Optional

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.schemas import (
    PromptInstructions,
    PersonalityError,
)
from luma.core.personality.system_prompt import SystemPrompt
from luma.core.personality.tone_manager import ToneManager
from luma.core.personality.style_profiles import StyleProfiles
from luma.core.personality.response_guardrails import ResponseGuardrails
from luma.core.structured_logger import StructuredLogger


class PersonalityEngine:
    """
    Orchestrator for the personality layer.

    Combines system prompt, tone selection, style profile, and guardrails to
    produce final prompt instructions for the LLM layer. All dependencies are
    injected via constructor for testability.

    Stateless component: identical inputs produce identical outputs.
    """

    def __init__(
        self,
        system_prompt: SystemPrompt,
        tone_manager: ToneManager,
        style_profiles: StyleProfiles,
        response_guardrails: ResponseGuardrails,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """
        Initialize PersonalityEngine with all dependencies.

        Args:
            system_prompt: SystemPrompt component for base identity
            tone_manager: ToneManager component for tone selection
            style_profiles: StyleProfiles component for style retrieval
            response_guardrails: ResponseGuardrails component for validation
            logger: Optional StructuredLogger for observability (defaults to no-op)
        """
        self._system_prompt = system_prompt
        self._tone_manager = tone_manager
        self._style_profiles = style_profiles
        self._response_guardrails = response_guardrails
        self._logger = logger or self._create_noop_logger()

    def build_instructions(
        self,
        user_id: str,
        context: AdaptationContext,
        mode: str,
    ) -> PromptInstructions:
        """
        Build prompt instructions by orchestrating all personality components.

        Orchestration flow:
        1. Log start of instruction building
        2. Get base identity from SystemPrompt
        3. Select tone from ToneManager
        4. Get style preference from StyleProfiles
        5. Combine into PromptInstructions with output rules
        6. Log end of instruction building
        7. Handle exceptions by raising PersonalityError

        Args:
            user_id: User identifier
            context: AdaptationContext from PersonalizationEngine
            mode: Interaction mode (e.g., "chat", "teacher")

        Returns:
            PromptInstructions with system_identity, tone_guidance,
            style_constraints, output_rules, and metadata

        Raises:
            PersonalityError: If any sub-component raises an exception
        """
        # Log start of instruction building
        self._logger.log(
            "building_prompt_instructions",
            {
                "user_id": user_id,
                "mode": mode,
                "context_tone": context.tone,
            },
        )

        try:
            # Get base identity from SystemPrompt
            system_identity = self._system_prompt.get_identity()

            # Select tone from ToneManager
            tone_selection = self._tone_manager.select_tone(
                context=context,
                mode=mode,
                user_preference=None,
            )

            # Get style preference from StyleProfiles
            style_preference = self._style_profiles.get_style(user_id)

            # Get tone guidance from ToneManager's mapping
            tone_guidance = self._tone_manager.TONE_GUIDANCE.get(
                tone_selection.tone,
                "Use clear and appropriate communication",
            )

            # Get style constraints from StyleProfiles
            from luma.core.personality.style_profiles import STYLE_CONSTRAINTS

            style_constraints = STYLE_CONSTRAINTS.get(
                style_preference.style,
                "Balance detail and brevity",
            )

            # Define output rules
            output_rules = [
                "No rambling: keep responses focused and structured",
                "No repetition: avoid repeating the same information",
                "No contradiction: ensure consistency throughout the response",
                "No vague filler: provide concrete, actionable information",
                "Respect requested length: match the user's preferred response length",
            ]

            # Combine into PromptInstructions
            prompt_instructions = PromptInstructions(
                system_identity=system_identity,
                tone_guidance=tone_guidance,
                style_constraints=style_constraints,
                output_rules=output_rules,
                metadata={
                    "user_id": user_id,
                    "mode": mode,
                    "selected_tone": tone_selection.tone,
                    "selected_style": style_preference.style,
                    "tone_rationale": tone_selection.rationale,
                },
            )

            # Log end of instruction building
            self._logger.log(
                "prompt_instructions_built",
                {
                    "user_id": user_id,
                    "selected_tone": tone_selection.tone,
                    "selected_style": style_preference.style,
                    "output_rules_count": len(output_rules),
                },
            )

            return prompt_instructions

        except Exception as exc:
            # Log the exception
            self._logger.log(
                "personality_engine_error",
                {
                    "user_id": user_id,
                    "mode": mode,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

            # Raise PersonalityError with original exception as cause
            raise PersonalityError(
                f"Failed to build prompt instructions: {exc}"
            ) from exc

    def _create_noop_logger(self) -> StructuredLogger:
        """
        Create a no-op logger that doesn't output anything.

        Returns:
            StructuredLogger instance with a NullHandler
        """
        import logging

        logger = StructuredLogger(name="personality_engine_noop")
        logger._logger.handlers.clear()
        logger._logger.addHandler(logging.NullHandler())
        return logger
