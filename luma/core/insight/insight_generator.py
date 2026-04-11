"""
Insight Generator Module.

Converts PatternResult and TrendResult objects into human-readable Insight objects.
"""

from typing import Dict, List, Optional, Set

from luma.core.insight.schemas import Insight, PatternResult, TrendResult

# Text templates
_PATTERN_ONLY_TMPL = "You frequently engage with '{pattern}' (mentioned {frequency} times)."
_TREND_ONLY_TMPL = "Your interest in '{topic}' is {trend} ({time_window})."
_COMBINED_TMPL = "You frequently engage with '{pattern}' and your interest is {trend} ({time_window})."


class InsightGenerator:
    """
    Generates human-readable Insight objects from PatternResult and TrendResult inputs.

    Algorithm:
      1. Index trends by topic.
      2. Process patterns above min_confidence:
         - If matching trend exists: produce combined insight, mark trend consumed.
         - Otherwise: produce pattern-only insight.
      3. Process unconsumed trends above min_confidence:
         - Derive evidence from corresponding PatternResult; skip if no evidence.
      4. Drop insights with empty evidence.
      5. Sort by (confidence DESC, text ASC).
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        self._min_confidence = min_confidence

    def generate(
        self,
        patterns: List[PatternResult],
        trends: List[TrendResult],
    ) -> List[Insight]:
        """
        Generate insights from patterns and trends.

        Args:
            patterns: List of PatternResult objects from PatternDetector.
            trends: List of TrendResult objects from TrendAnalyzer.

        Returns:
            Sorted list of Insight objects (confidence DESC, text ASC).
        """
        if not patterns and not trends:
            return []

        # Step 1 — Index trends by topic
        trend_index: Dict[str, TrendResult] = {t.topic: t for t in trends}
        consumed_topics: Set[str] = set()

        # Build a pattern index for evidence lookup during trend-only processing
        pattern_index: Dict[str, PatternResult] = {p.pattern: p for p in patterns}

        insights: List[Insight] = []

        # Step 2 — Process patterns above min_confidence
        for pattern in patterns:
            if pattern.confidence < self._min_confidence:
                continue

            evidence = pattern.evidence
            if not evidence:
                continue

            matching_trend: Optional[TrendResult] = trend_index.get(pattern.pattern)

            if matching_trend is not None and matching_trend.confidence >= self._min_confidence:
                # Combined insight
                text = _COMBINED_TMPL.format(
                    pattern=pattern.pattern,
                    trend=matching_trend.trend,
                    time_window=matching_trend.time_window,
                )
                confidence = (pattern.confidence + matching_trend.confidence) / 2.0
                consumed_topics.add(pattern.pattern)
            else:
                # Pattern-only insight
                text = _PATTERN_ONLY_TMPL.format(
                    pattern=pattern.pattern,
                    frequency=pattern.frequency,
                )
                confidence = pattern.confidence

            insights.append(Insight(text=text, confidence=confidence, evidence=list(evidence)))

        # Step 3 — Process unconsumed trends above min_confidence
        for trend in trends:
            if trend.topic in consumed_topics:
                continue
            if trend.confidence < self._min_confidence:
                continue

            # Derive evidence from corresponding PatternResult
            matching_pattern: Optional[PatternResult] = pattern_index.get(trend.topic)
            if matching_pattern is None or not matching_pattern.evidence:
                continue

            evidence = matching_pattern.evidence
            text = _TREND_ONLY_TMPL.format(
                topic=trend.topic,
                trend=trend.trend,
                time_window=trend.time_window,
            )
            insights.append(
                Insight(text=text, confidence=trend.confidence, evidence=list(evidence))
            )

        # Step 4 — Filter: drop insights with empty evidence (already guarded above)
        insights = [i for i in insights if i.evidence]

        # Step 5 — Sort by (confidence DESC, text ASC)
        insights.sort(key=lambda i: (-i.confidence, i.text))

        return insights
