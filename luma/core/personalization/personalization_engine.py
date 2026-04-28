"""
PersonalizationEngine: Orchestrates the full personalization pipeline.

Retrieves memories from the memory interface, builds a user profile,
detects preferences, and derives an adaptation context. This class is
the top-level entry point for the personalization subsystem.

Design principles:
- Dependency injection for all collaborators
- Read-only memory consumption (never calls store())
- Never mutates input MemoryEntry or Insight objects
- Propagates MemoryRetrievalError with context message
"""

from typing import Any, Dict, List, Optional

from luma.core.memory_interface import MemoryInterface, MemoryRetrievalError
from luma.core.personalization.schemas import PersonalizationResult
from luma.core.personalization.profile_builder import ProfileBuilder
from luma.core.personalization.preference_detector import PreferenceDetector
from luma.core.personalization.adaptation_engine import AdaptationEngine


class PersonalizationEngine:
    """
    Orchestrates the personalization pipeline.

    Parameters
    ----------
    memory_interface : MemoryInterface
        Used to retrieve stored memories. Never called for storage.
    profile_builder : ProfileBuilder
        Builds a UserProfile from memories and insights.
    preference_detector : PreferenceDetector
        Detects user preferences from memories and profile.
    adaptation_engine : AdaptationEngine
        Derives an AdaptationContext from profile and preferences.
    """

    def __init__(
        self,
        memory_interface: MemoryInterface,
        profile_builder: ProfileBuilder,
        preference_detector: PreferenceDetector,
        adaptation_engine: AdaptationEngine,
    ) -> None:
        self._memory_interface = memory_interface
        self._profile_builder = profile_builder
        self._preference_detector = preference_detector
        self._adaptation_engine = adaptation_engine

    def personalize(
        self,
        input_data: Any,
        context: Any,
        insights: Optional[List[Any]] = None,
    ) -> PersonalizationResult:
        """
        Run the full personalization pipeline.

        Parameters
        ----------
        input_data : Any
            Current input data for the session. Must not be None.
        context : Any
            Current session context. Must not be None.
        insights : Optional[List[Any]]
            Optional list of insight objects to supplement profile building.
            Defaults to an empty list when not provided.

        Returns
        -------
        PersonalizationResult
            Contains the built UserProfile, detected Preferences, and
            derived AdaptationContext.

        Raises
        ------
        TypeError
            If ``input_data`` or ``context`` is None.
        MemoryRetrievalError
            If the memory retrieval operation fails.
        """
        if input_data is None:
            raise TypeError("input_data must not be None")
        if context is None:
            raise TypeError("context must not be None")

        # Step 1: Retrieve memories (read-only — never call store())
        try:
            result = self._memory_interface.retrieve(params={"limit": 500})
        except MemoryRetrievalError as exc:
            raise MemoryRetrievalError(
                f"PersonalizationEngine failed to retrieve memories: {exc}"
            ) from exc

        memories = result["memories"]

        # Step 2: Build user profile
        profile = self._profile_builder.build(memories, insights or [])

        # Step 3: Detect preferences
        preferences = self._preference_detector.detect(memories, profile)

        # Step 4: Derive adaptation context
        adaptation = self._adaptation_engine.adapt(profile, preferences)

        return PersonalizationResult(
            profile=profile,
            preferences=preferences,
            adaptation=adaptation,
        )
