"""
Personalization Engine – AdaptationEngine.

Derives a concrete AdaptationContext (tone, style, focus) from a UserProfile
and a list of Preferences.  The engine is stateless and deterministic: it
never calls MemoryInterface or InsightEngine, and it never mutates its inputs.
"""

from typing import List, Optional

from luma.core.personalization.schemas import AdaptationContext, Preference, UserProfile

# Keywords that indicate a technical domain interest
_TECHNICAL_DOMAIN_KEYWORDS = {
    "python", "javascript", "typescript", "java", "rust", "golang", "cpp",
    "sql", "api", "algorithm", "database", "framework", "library",
    "programming", "software", "engineering", "machine", "learning",
    "neural", "network", "data", "science", "cloud", "devops",
    "kubernetes", "docker",
}


class AdaptationEngine:
    """Derive an AdaptationContext from a UserProfile and detected Preferences."""

    def adapt(
        self,
        profile: UserProfile,
        preferences: Optional[List[Preference]],
    ) -> AdaptationContext:
        """
        Produce an AdaptationContext for the given profile and preferences.

        Parameters
        ----------
        profile:
            The aggregated user profile.
        preferences:
            Detected preferences.  ``None`` is treated as an empty list.

        Returns
        -------
        AdaptationContext
            Immutable context with tone, style, focus, and per-field reasons.
        """
        if preferences is None:
            preferences = []

        pref_labels = {p.preference for p in preferences}

        # ------------------------------------------------------------------
        # Tone
        # ------------------------------------------------------------------
        tone: str
        tone_reason: str

        if "technical" in pref_labels:
            tone = "technical"
            tone_reason = "technical preference detected"
        else:
            # Check whether any interest matches a technical domain keyword
            matching_interest: Optional[str] = None
            for interest in profile.interests:
                if interest.lower() in _TECHNICAL_DOMAIN_KEYWORDS:
                    matching_interest = interest
                    break

            if matching_interest is not None:
                tone = "technical"
                tone_reason = (
                    f"Interest '{matching_interest}' matches a technical domain keyword"
                )
            elif "formal" in pref_labels:
                tone = "formal"
                tone_reason = "formal preference detected"
            else:
                tone = "casual"
                tone_reason = "No strong tone signal detected; defaulting to casual"

        # ------------------------------------------------------------------
        # Style
        # ------------------------------------------------------------------
        style: str
        style_reason: str

        if "step-by-step" in pref_labels:
            style = "step-by-step"
            style_reason = "step-by-step preference detected"
        elif profile.interaction_style == "concise":
            style = "concise"
            style_reason = "User interaction style is concise"
        elif profile.interaction_style == "detailed":
            style = "detailed"
            style_reason = "User interaction style is detailed"
        else:
            style = "balanced"
            style_reason = "No strong style signal; defaulting to balanced"

        # ------------------------------------------------------------------
        # Focus
        # ------------------------------------------------------------------
        focus: str
        focus_reason: str

        if "deep-technical" in pref_labels:
            focus = "deep-technical"
            focus_reason = "deep-technical preference detected"
        elif len(profile.strengths) >= 3:
            focus = "deep-technical"
            focus_reason = "User has 3+ strengths indicating deep expertise"
        else:
            focus = "high-level"
            focus_reason = "Insufficient depth signals; defaulting to high-level"

        return AdaptationContext(
            tone=tone,
            style=style,
            focus=focus,
            reasons={
                "tone": tone_reason,
                "style": style_reason,
                "focus": focus_reason,
            },
        )
