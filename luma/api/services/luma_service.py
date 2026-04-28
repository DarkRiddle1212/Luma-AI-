"""
LumaService — orchestration layer between API controllers and core modules.

Accepts all core module dependencies via constructor injection and exposes
one async method per endpoint group. Never instantiates core modules directly.
"""

import time
from typing import List, Optional

from luma.core.insight_moments.schemas import TimingContext
from luma.core.personalization.schemas import AdaptationContext
from luma.core.teacher.schemas import TeachingSession


class LumaService:
    """
    Orchestrates calls to Luma's core modules on behalf of API controllers.

    All dependencies are injected via the constructor — this class never
    instantiates MemoryInterface, InsightEngine, InsightMomentsEngine,
    PersonalizationEngine, or TeacherMode directly.

    Parameters
    ----------
    memory_interface :
        MemoryInterface implementation for storing and retrieving memories.
    insight_engine :
        InsightEngine for generating insight reports.
    insight_moments_engine :
        InsightMomentsEngine for surfacing triggered insight moments.
    personalization_engine :
        PersonalizationEngine for deriving AdaptationContext.
    teacher_mode :
        TeacherMode for running teaching sessions.
    """

    def __init__(
        self,
        memory_interface,
        insight_engine,
        insight_moments_engine,
        personalization_engine,
        teacher_mode,
    ) -> None:
        self._memory_interface = memory_interface
        self._insight_engine = insight_engine
        self._insight_moments_engine = insight_moments_engine
        self._personalization_engine = personalization_engine
        self._teacher_mode = teacher_mode

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_timing_context() -> TimingContext:
        """Create a minimal TimingContext for the current moment."""
        return TimingContext(
            session_ended=False,
            repeated_behavior=False,
            current_timestamp=time.time(),
        )

    @staticmethod
    def _adaptation_ctx_to_dict(adaptation_ctx) -> dict:
        """Serialize an AdaptationContext to a plain dict."""
        if hasattr(adaptation_ctx, "model_dump"):
            return adaptation_ctx.model_dump()
        return {
            "tone": adaptation_ctx.tone,
            "style": adaptation_ctx.style,
            "focus": adaptation_ctx.focus,
            "reasons": adaptation_ctx.reasons,
        }

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def process_chat(self, user_id: str, message: str) -> dict:
        """
        Orchestrate a full chat turn.

        Orchestration order (per Requirement 1.2):
        1. Retrieve relevant memories.
        2. Obtain AdaptationContext via PersonalizationEngine.
        3. Construct context-injected prompt.
        4. Generate response string.
        5. Surface triggered insight moments.
        6. Persist the new memory.
        7. Return structured dict.

        Parameters
        ----------
        user_id : str
            Identifier of the user.
        message : str
            The user's chat message.

        Returns
        -------
        dict
            Keys: ``response`` (str), ``insight_moments`` (list),
            ``personalization`` (dict with tone/style/focus/reasons).
        """
        # Step 1 — Retrieve relevant memories
        retrieval_result = self._memory_interface.retrieve(
            params={"query": message, "limit": 10}
        )
        memories = retrieval_result["memories"]

        # Step 2 — Personalize
        personalization_result = self._personalization_engine.personalize(
            user_id, message
        )
        adaptation_ctx = personalization_result.adaptation

        # Step 3 — Construct context-injected prompt (simple string construction)
        memory_texts = [m["content"] for m in memories]
        context_str = "; ".join(memory_texts) if memory_texts else ""
        prompt = (
            f"Context: {context_str}\n"
            f"Tone: {adaptation_ctx.tone}, Style: {adaptation_ctx.style}\n"
            f"User: {message}"
        )

        # Step 4 — Generate response (no separate reasoning engine injected)
        response_str = f"Response to '{message}' for user '{user_id}'"

        # Step 5 — Surface triggered insight moments
        timing_ctx = self._make_timing_context()
        moments = self._insight_moments_engine.generate_moments(
            insights=[], context=timing_ctx
        )

        # Step 6 — Persist new memory
        self._memory_interface.store(
            message,
            metadata={"user_id": user_id, "category": "chat"},
        )

        # Step 7 — Return structured dict
        return {
            "response": response_str,
            "insight_moments": moments,
            "personalization": self._adaptation_ctx_to_dict(adaptation_ctx),
        }

    async def get_insights(self, namespace: Optional[str] = None) -> list:
        """
        Generate insights, optionally filtered by namespace.

        Calls ``InsightEngine.generate_insights(namespace=namespace)`` when
        namespace is not None, otherwise calls without the argument.

        Parameters
        ----------
        namespace : Optional[str]
            Category filter forwarded unchanged to InsightEngine.

        Returns
        -------
        list
            The ``insights`` list from the InsightReport.
        """
        if namespace is not None:
            report = self._insight_engine.generate_insights(namespace=namespace)
        else:
            report = self._insight_engine.generate_insights()
        return report.insights

    async def get_insight_moments(self) -> list:
        """
        Surface triggered insight moments for the current session.

        Returns
        -------
        list
            List of DeliveryPayload objects.
        """
        timing_ctx = self._make_timing_context()
        return self._insight_moments_engine.generate_moments(
            insights=[], context=timing_ctx
        )

    async def start_teacher_mode(
        self, user_id: str, topic: str
    ) -> TeachingSession:
        """
        Start a new teaching session.

        Parameters
        ----------
        user_id : str
        topic : str

        Returns
        -------
        TeachingSession
        """
        return self._teacher_mode.teach(user_id, topic)

    async def continue_teacher_mode(
        self, user_id: str, topic: str
    ) -> TeachingSession:
        """
        Continue an existing teaching session.

        TeacherMode handles progress tracking internally; both start and
        continue use the same ``teach()`` call — the distinction is surfaced
        via ``TeachingSession.status``.

        Parameters
        ----------
        user_id : str
        topic : str

        Returns
        -------
        TeachingSession
        """
        return self._teacher_mode.teach(user_id, topic)

    async def get_personalization(self, user_id: str) -> AdaptationContext:
        """
        Retrieve the current adaptation context for a user.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        AdaptationContext
        """
        result = self._personalization_engine.personalize(user_id, "")
        return result.adaptation
