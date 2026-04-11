"""
InsightTrigger — filters a list of Insight objects to only those that meet
confidence and novelty (deduplication) criteria.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 6.1, 6.4,
              7.1, 7.4, 9.2, 10.1
"""

from typing import List, Optional, Set

from luma.core.insight.schemas import Insight


class InsightTrigger:
    """
    Filters insights by confidence threshold and surfaced-history deduplication.

    Args:
        confidence_threshold: Minimum confidence required for an insight to pass.
            Must be in [0.0, 1.0].  Defaults to 0.7.
        surfaced_history: Set of insight texts that have already been shown to
            the user.  Insights whose text is in this set are excluded.
            Defaults to an empty set when None is passed.

    Raises:
        ValueError: If confidence_threshold is outside [0.0, 1.0].
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        surfaced_history: Optional[Set[str]] = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )
        self._confidence_threshold = confidence_threshold
        self._surfaced_history: Set[str] = surfaced_history if surfaced_history is not None else set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        Return the subset of *insights* that pass both filters, preserving
        input order and returning the same object references (not copies).

        An insight is excluded if:
        - ``insight.confidence < self._confidence_threshold``, OR
        - ``insight.text`` is present in ``self._surfaced_history``
          (case-sensitive exact match).

        Args:
            insights: List of Insight objects to evaluate.

        Returns:
            Filtered list of Insight objects (same references, same order).
        """
        result: List[Insight] = []
        for insight in insights:
            if insight.confidence < self._confidence_threshold:
                continue
            if insight.text in self._surfaced_history:
                continue
            result.append(insight)
        return result
