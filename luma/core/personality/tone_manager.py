"""
ToneManager: Context-aware tone selection for the personality layer.

Selects tone based on user preference, mode, and context signals.
Stateless component with deterministic behavior.
"""

from typing import Optional

from luma.core.personalization.schemas import AdaptationContext
from luma.core.personality.schemas import ToneSelection, VALID_TONES


class ToneManager:
    """
    Selects tone based on context, mode, and user preference.

    Tone selection priority (highest to lowest):
    1. user_preference (if provided and valid)
    2. mode == "teacher" → tone "teacher"
    3. context.tone == "technical" → tone "technical"
    4. context.tone == "formal" → tone "professional"
    5. context.tone == "casual" → tone "friendly"
    6. Default → tone "friendly"
    """

    # Tone → guidance mapping
    TONE_GUIDANCE = {
        "professional": "Use formal language, structured responses, avoid colloquialisms",
        "friendly": "Use conversational language, warm tone, approachable phrasing",
        "concise": "Prioritize brevity, avoid elaboration, get to the point",
        "technical": "Use domain-specific terminology, include technical details",
        "teacher": "Use explanatory language, break down concepts, encourage learning",
        "motivational": "Use encouraging language, positive framing, growth mindset",
        "analytical": "Use logical structure, evidence-based reasoning, objective tone",
    }

    def select_tone(
        self,
        context: AdaptationContext,
        mode: str,
        user_preference: Optional[str] = None,
    ) -> ToneSelection:
        """
        Select tone based on user preference, mode, and context.

        Args:
            context: AdaptationContext from PersonalizationEngine
            mode: Interaction mode (e.g., "chat", "teacher")
            user_preference: Optional user-specified tone preference

        Returns:
            ToneSelection with selected tone, rationale, and context signals
        """
        context_signals = {
            "context_tone": context.tone,
            "mode": mode,
            "user_preference": user_preference,
        }

        # Priority 1: Valid user preference
        if user_preference and user_preference in VALID_TONES:
            return ToneSelection(
                tone=user_preference,
                rationale="user preference",
                context_signals=context_signals,
            )

        # Priority 2: Teacher mode
        if mode == "teacher":
            return ToneSelection(
                tone="teacher",
                rationale="teacher mode active",
                context_signals=context_signals,
            )

        # Priority 3: Technical context
        if context.tone == "technical":
            return ToneSelection(
                tone="technical",
                rationale="technical context detected",
                context_signals=context_signals,
            )

        # Priority 4: Formal context
        if context.tone == "formal":
            return ToneSelection(
                tone="professional",
                rationale="formal context detected",
                context_signals=context_signals,
            )

        # Priority 5: Casual context
        if context.tone == "casual":
            return ToneSelection(
                tone="friendly",
                rationale="casual context detected",
                context_signals=context_signals,
            )

        # Priority 6: Default
        return ToneSelection(
            tone="friendly",
            rationale="default tone",
            context_signals=context_signals,
        )
