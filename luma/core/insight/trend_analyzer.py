"""
Trend Analyzer Module.

Detects how pattern frequencies change across two time windows by partitioning
memories at the midpoint of their timestamp range.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from luma.core.memory_interface import MemoryEntry
from luma.core.insight.schemas import PatternResult, TrendResult

logger = logging.getLogger(__name__)

_TIME_WINDOW_LABEL = "recent_half"


class TrendAnalyzer:
    """
    Detects increasing or decreasing trends in pattern frequencies over time.

    Algorithm:
      1. Parse ISO 8601 timestamps; compute t_min and t_max.
      2. Partition at midpoint into earlier_window [t_min, midpoint) and
         recent_window [midpoint, t_max].
      3. For each PatternResult, count evidence IDs in each window.
      4. Emit "increasing" if recent/earlier >= threshold,
         "decreasing" if earlier/recent >= threshold.
      5. confidence = min(1.0, max(recent_count, earlier_count) / len(memories))
      6. Sort by (confidence DESC, topic ASC).
    """

    def __init__(self, trend_ratio_threshold: float = 1.5) -> None:
        if trend_ratio_threshold <= 1.0:
            raise ValueError(
                f"trend_ratio_threshold must be > 1.0, got {trend_ratio_threshold}"
            )
        self._threshold = trend_ratio_threshold

    def analyze(
        self,
        patterns: List[PatternResult],
        memories: List[MemoryEntry],
    ) -> List[TrendResult]:
        """
        Analyze patterns against memories to detect temporal trends.

        Args:
            patterns: List of PatternResult objects from PatternDetector.
            memories: List of MemoryEntry objects with ISO 8601 timestamps.

        Returns:
            Sorted list of TrendResult objects (confidence DESC, topic ASC).
        """
        if len(memories) < 2:
            return []

        # Step 1 — Parse timestamps; build id → datetime mapping
        id_to_dt: Dict[str, datetime] = {}
        for memory in memories:
            mem_id = memory["id"]
            ts_str = memory.get("timestamp")
            if not ts_str:
                logger.warning("Memory %s has no timestamp; skipping.", mem_id)
                continue
            try:
                dt = _parse_iso8601(ts_str)
                id_to_dt[mem_id] = dt
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Memory %s has malformed timestamp %r; skipping. (%s)",
                    mem_id,
                    ts_str,
                    exc,
                )

        if len(id_to_dt) < 2:
            return []

        t_min = min(id_to_dt.values())
        t_max = max(id_to_dt.values())

        # Step 2 — Partition at midpoint
        # Use float arithmetic to avoid overflow with large timestamps
        midpoint_ts = t_min.timestamp() + (t_max.timestamp() - t_min.timestamp()) / 2.0
        midpoint = datetime.fromtimestamp(midpoint_ts, tz=t_min.tzinfo)

        # Classify each memory ID into a window
        # earlier: [t_min, midpoint)   recent: [midpoint, t_max]
        earlier_ids: set = set()
        recent_ids: set = set()
        for mem_id, dt in id_to_dt.items():
            if dt < midpoint:
                earlier_ids.add(mem_id)
            else:
                recent_ids.add(mem_id)

        # Edge case: all memories in one window → no trends possible
        if not earlier_ids or not recent_ids:
            return []

        total = len(memories)
        results: List[TrendResult] = []

        # Steps 3–5 — Evaluate each pattern
        for pattern in patterns:
            evidence_ids = set(pattern.evidence)

            earlier_count = len(evidence_ids & earlier_ids)
            recent_count = len(evidence_ids & recent_ids)

            # Skip if no data in earlier window (avoid division by zero)
            if earlier_count == 0:
                continue

            trend: Optional[str] = None
            if recent_count / earlier_count >= self._threshold:
                trend = "increasing"
            elif recent_count == 0 or earlier_count / recent_count >= self._threshold:
                trend = "decreasing"

            if trend is None:
                continue

            confidence = min(1.0, max(recent_count, earlier_count) / total)
            results.append(
                TrendResult(
                    trend=trend,
                    topic=pattern.pattern,
                    confidence=confidence,
                    time_window=_TIME_WINDOW_LABEL,
                )
            )

        # Step 6 — Sort by (confidence DESC, topic ASC)
        results.sort(key=lambda r: (-r.confidence, r.topic))

        return results


def _parse_iso8601(ts_str: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime."""
    # Python 3.7+ fromisoformat doesn't handle 'Z' suffix; replace it.
    normalized = ts_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    # If naive, treat as UTC so comparisons are consistent
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
