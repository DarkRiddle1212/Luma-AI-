"""
StyleProfiles — User Communication Style Management.

Stores and retrieves user-preferred communication styles and integrates
with the personalization profile. Accepts a storage backend via constructor
for dependency injection.
"""

import json
from typing import Any

from luma.core.personality.schemas import StylePreference, VALID_STYLES
from luma.core.personalization.schemas import AdaptationContext


# Style → constraints mapping
STYLE_CONSTRAINTS = {
    "short_answers": "Limit responses to 1-2 sentences, avoid elaboration",
    "step_by_step": "Use numbered steps, break down processes sequentially",
    "detailed_explanations": "Provide comprehensive coverage, include examples and context",
    "high_signal_low_noise": "Balance detail and brevity, focus on actionable information",
    "motivational_style": "Include encouragement, positive framing, growth mindset language",
    "technical_depth": "Include implementation details, technical terminology, edge cases",
}

# AdaptationContext.style → StylePreference.style mapping
ADAPTATION_STYLE_MAP = {
    "concise": "short_answers",
    "step-by-step": "step_by_step",
    "detailed": "detailed_explanations",
    "balanced": "high_signal_low_noise",
}


class StyleProfiles:
    """Manages user communication style preferences via storage backend."""

    def __init__(self, storage_backend: Any) -> None:
        """
        Initialize StyleProfiles with a storage backend.

        Args:
            storage_backend: Storage backend for persisting user preferences.
                            Expected to have store() and retrieve() methods.
        """
        self._storage = storage_backend

    def get_style(self, user_id: str) -> StylePreference:
        """
        Retrieve the user's preferred communication style.

        Args:
            user_id: User identifier.

        Returns:
            StylePreference with the user's preferred style, or default
            "high_signal_low_noise" if no preference is stored.
        """
        try:
            result = self._storage.retrieve(
                params={"category": "style_preference", "limit": 100}
            )
        except Exception:
            # If retrieval fails, return default
            return self._default_style()

        # Find the user's style preference
        for entry in result.get("memories", []):
            meta = entry.get("metadata") or {}
            if (
                meta.get("category") == "style_preference"
                and meta.get("user_id") == user_id
            ):
                data = json.loads(entry["content"])
                style = data.get("style", "high_signal_low_noise")
                description = STYLE_CONSTRAINTS.get(
                    style, "Balanced communication style"
                )
                return StylePreference(
                    style=style,
                    description=description,
                    active=True,
                )

        # No preference found, return default
        return self._default_style()

    def set_style(self, user_id: str, style: str) -> None:
        """
        Store the user's preferred communication style.

        Args:
            user_id: User identifier.
            style: Communication style. Must be one of VALID_STYLES.

        Raises:
            ValueError: If style is not in VALID_STYLES.
        """
        if style not in VALID_STYLES:
            raise ValueError(
                f"style must be one of {VALID_STYLES}, got {style!r}"
            )

        content = json.dumps(
            {
                "user_id": user_id,
                "style": style,
            }
        )
        metadata = {
            "user_id": user_id,
            "category": "style_preference",
        }

        try:
            self._storage.store(content, metadata)
        except Exception as exc:
            raise ValueError(f"Failed to store style preference: {exc}") from exc

    def get_style_from_context(
        self, context: AdaptationContext
    ) -> StylePreference:
        """
        Map AdaptationContext.style to StylePreference.

        Args:
            context: AdaptationContext from PersonalizationEngine.

        Returns:
            StylePreference mapped from context.style.
        """
        mapped_style = ADAPTATION_STYLE_MAP.get(
            context.style, "high_signal_low_noise"
        )
        description = STYLE_CONSTRAINTS.get(
            mapped_style, "Balanced communication style"
        )
        return StylePreference(
            style=mapped_style,
            description=description,
            active=True,
        )

    def _default_style(self) -> StylePreference:
        """Return default StylePreference."""
        return StylePreference(
            style="high_signal_low_noise",
            description=STYLE_CONSTRAINTS["high_signal_low_noise"],
            active=True,
        )
