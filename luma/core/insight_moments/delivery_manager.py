"""
Insight Moments Engine — DeliveryManager.

Formats an approved Insight into a structured DeliveryPayload ready for
user-facing rendering. Classifies the delivery type based on confidence
thresholds and preserves all insight fields without modification.
"""

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.schemas import DeliveryPayload


class DeliveryManager:
    """
    Formats approved insights into DeliveryPayload objects.

    Args:
        highlighted_threshold: Confidence >= this value → "highlighted_insight".
            Defaults to 0.85.
        inline_threshold: Confidence >= this value (and < highlighted_threshold)
            → "inline_suggestion". Defaults to 0.70.

    Raises:
        ValueError: If highlighted_threshold <= inline_threshold at construction.
    """

    def __init__(
        self,
        highlighted_threshold: float = 0.85,
        inline_threshold: float = 0.70,
    ) -> None:
        if highlighted_threshold <= inline_threshold:
            raise ValueError(
                f"highlighted_threshold must be > inline_threshold, "
                f"got highlighted_threshold={highlighted_threshold}, "
                f"inline_threshold={inline_threshold}"
            )
        self._highlighted_threshold = highlighted_threshold
        self._inline_threshold = inline_threshold

    def format_delivery(self, insight: Insight) -> DeliveryPayload:
        """
        Format an approved insight into a DeliveryPayload.

        Classification rules (>= comparisons, so boundary values are included):
          - confidence >= highlighted_threshold → "highlighted_insight"
          - confidence >= inline_threshold      → "inline_suggestion"
          - otherwise                           → "subtle_notification"

        The input Insight is never modified.

        Args:
            insight: The approved Insight to format.

        Returns:
            A DeliveryPayload with payload_type="insight_moment", message set to
            insight.text, confidence preserved exactly, and type classified above.
        """
        if insight.confidence >= self._highlighted_threshold:
            delivery_type = "highlighted_insight"
        elif insight.confidence >= self._inline_threshold:
            delivery_type = "inline_suggestion"
        else:
            delivery_type = "subtle_notification"

        return DeliveryPayload(
            payload_type="insight_moment",
            message=insight.text,
            type=delivery_type,
            confidence=insight.confidence,
        )
