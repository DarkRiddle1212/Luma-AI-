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
- Optional PersonalizationRepository for durable profile persistence
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
    personalization_repository : Optional[Any]
        Optional repository for persisting user profiles. When provided,
        ``get_by_user()`` is called before computing the adaptation context
        and ``upsert()`` is called after updating the profile. When ``None``,
        the engine operates in in-memory mode. The repository is injected via
        the constructor; this module never imports from ``luma.storage``.
    """

    def __init__(
        self,
        memory_interface: MemoryInterface,
        profile_builder: ProfileBuilder,
        preference_detector: PreferenceDetector,
        adaptation_engine: AdaptationEngine,
        personalization_repository: Optional[Any] = None,
    ) -> None:
        self._memory_interface = memory_interface
        self._profile_builder = profile_builder
        self._preference_detector = preference_detector
        self._adaptation_engine = adaptation_engine
        self._personalization_repository = personalization_repository

    def personalize(
        self,
        input_data: Any,
        context: Any,
        insights: Optional[List[Any]] = None,
        user_id: str = "default",
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
        user_id : str
            Identifier for the current user. Used when a
            ``personalization_repository`` is injected to load and persist
            the user profile. Defaults to ``"default"``.

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

        # Step 0 (persistence mode): load persisted profile before adapting
        # Requirement 12.3 — call get_by_user() before computing adaptation context
        persisted_record = None
        if self._personalization_repository is not None:
            persisted_record = self._personalization_repository.get_by_user(user_id)

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

        # Step 5 (persistence mode): persist the updated profile
        # Requirement 12.2 — call upsert() after updating profile
        if self._personalization_repository is not None:
            self._personalization_repository.upsert(
                user_id,
                interests=profile.interests,
                preferences=dict(profile.evidence) if profile.evidence else {},
                strengths=profile.strengths,
            )

        return PersonalizationResult(
            profile=profile,
            preferences=preferences,
            adaptation=adaptation,
        )
