"""
Timing Strategy for the Insight Moments Engine.

Evaluates whether the current context is an appropriate moment to deliver
a given insight, taking into account cooldown windows and trigger signals.
"""

from typing import Dict, Optional

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.schemas import TimingContext


class TimingStrategy:
    """
    Determines whether an insight should be delivered based on timing signals.

    Cooldown always takes precedence: if an insight was recently delivered
    within the cooldown window, it will not be delivered again regardless of
    session_ended or repeated_behavior signals.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3600.0,
        delivery_timestamps: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialise TimingStrategy.

        Args:
            cooldown_seconds: Minimum seconds between deliveries of the same insight.
                              Must be >= 0. Defaults to 3600.0 (one hour).
            delivery_timestamps: Mapping of insight text → last delivery Unix timestamp.
                                  Defaults to empty dict. Never modified by this class.

        Raises:
            ValueError: If cooldown_seconds < 0.
        """
        if cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be >= 0, got {cooldown_seconds}"
            )
        self._cooldown_seconds = cooldown_seconds
        self._delivery_timestamps: Dict[str, float] = (
            delivery_timestamps if delivery_timestamps is not None else {}
        )

    def should_deliver(self, insight: Insight, context: TimingContext) -> bool:
        """
        Decide whether to deliver the given insight in the current context.

        Algorithm:
        1. Cooldown check (always first): if the insight was delivered within
           cooldown_seconds, return False.
        2. If context.session_ended is True → return True.
        3. If context.repeated_behavior is True → return True.
        4. Default → return False.

        Args:
            insight: The insight to evaluate.
            context: The current session/timing context.

        Returns:
            True if the insight should be delivered now, False otherwise.
        """
        # Step 1: Cooldown check — takes precedence over all trigger signals.
        # A negative elapsed time means the stored timestamp is in the future relative
        # to current_timestamp; treat this as "not in cooldown" (no cooldown applies).
        if insight.text in self._delivery_timestamps:
            elapsed = context.current_timestamp - self._delivery_timestamps[insight.text]
            if elapsed >= 0 and elapsed < self._cooldown_seconds:
                return False

        # Step 2: Session-ended trigger signal.
        if context.session_ended:
            return True

        # Step 3: Repeated-behaviour trigger signal.
        if context.repeated_behavior:
            return True

        # Step 4: Default — no trigger signal fired.
        return False
