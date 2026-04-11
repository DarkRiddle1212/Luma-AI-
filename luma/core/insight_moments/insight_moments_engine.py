"""
InsightMomentsEngine — orchestrates the full filter → timing → delivery pipeline.

Transforms raw Insight objects into user-facing DeliveryPayload objects by
coordinating InsightTrigger, TimingStrategy, and DeliveryManager via
dependency injection.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 8.5, 9.1, 9.5
"""

from typing import List

from luma.core.insight.schemas import Insight
from luma.core.insight_moments.delivery_manager import DeliveryManager
from luma.core.insight_moments.insight_trigger import InsightTrigger
from luma.core.insight_moments.schemas import DeliveryPayload, TimingContext
from luma.core.insight_moments.timing_strategy import TimingStrategy


class InsightMomentsEngine:
    """
    Orchestrates the full insight delivery pipeline.

    Accepts three sub-components via dependency injection and exposes a single
    public method, generate_moments(), that runs the filter → timing → delivery
    pipeline and returns a list of DeliveryPayload objects.

    Args:
        insight_trigger: Filters insights by confidence and deduplication.
        timing_strategy: Decides whether each filtered insight should be
            delivered in the current context.
        delivery_manager: Formats approved insights into DeliveryPayload objects.
    """

    def __init__(
        self,
        insight_trigger: InsightTrigger,
        timing_strategy: TimingStrategy,
        delivery_manager: DeliveryManager,
    ) -> None:
        self._insight_trigger = insight_trigger
        self._timing_strategy = timing_strategy
        self._delivery_manager = delivery_manager

    def generate_moments(
        self,
        insights: List[Insight],
        context: TimingContext,
    ) -> List[DeliveryPayload]:
        """
        Run the full filter → timing → delivery pipeline.

        Algorithm:
        1. If insights is empty, return [] immediately (no sub-component calls).
        2. Pass insights to insight_trigger.filter_insights(insights) → filtered.
        3. For each insight in filtered, call timing_strategy.should_deliver(insight, context).
        4. For each approved insight (should_deliver returned True), call
           delivery_manager.format_delivery(insight) → DeliveryPayload.
        5. Return the list of DeliveryPayload objects in the order they were approved.

        Never modifies any field of any Insight object.
        Never calls InsightEngine or MemoryInterface directly.

        Args:
            insights: List of Insight objects to process.
            context: Current session/timing context for delivery decisions.

        Returns:
            List of DeliveryPayload objects for insights that passed all filters,
            in the order they were approved.
        """
        if not insights:
            return []

        filtered = self._insight_trigger.filter_insights(insights)

        payloads: List[DeliveryPayload] = []
        for insight in filtered:
            if self._timing_strategy.should_deliver(insight, context):
                payload = self._delivery_manager.format_delivery(insight)
                payloads.append(payload)

        return payloads
