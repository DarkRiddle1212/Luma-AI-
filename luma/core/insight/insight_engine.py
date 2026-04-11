"""
Insight Engine Module.

Orchestrates the full analysis pipeline: retrieve memories → detect patterns
→ analyze trends → generate insights → return InsightReport.
"""

from typing import List, Optional

from luma.core.memory_interface import (
    MemoryInterface,
    QueryParameters,
    MemoryRetrievalError,
    MemoryEntry,
)
from luma.core.insight.schemas import InsightReport
from luma.core.insight.pattern_detector import PatternDetector
from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.insight.insight_generator import InsightGenerator


class InsightEngine:
    """
    Orchestrates the pattern recognition and insight generation pipeline.

    The engine is read-only: it only calls retrieve() on MemoryInterface and
    never calls store() or mutates any MemoryEntry field.

    Pipeline:
      1. Build QueryParameters with optional namespace (→ category) and limit.
      2. Call memory_interface.retrieve(params=params).
      3. Extract List[MemoryEntry] from the RetrievalResult.
      4. Pass memories to pattern_detector.detect(memories).
      5. Pass patterns + memories to trend_analyzer.analyze(patterns, memories).
      6. Pass patterns + trends to insight_generator.generate(patterns, trends).
      7. Return InsightReport with counts.
    """

    def __init__(
        self,
        memory_interface: MemoryInterface,
        pattern_detector: PatternDetector,
        trend_analyzer: TrendAnalyzer,
        insight_generator: InsightGenerator,
    ) -> None:
        self._memory_interface = memory_interface
        self._pattern_detector = pattern_detector
        self._trend_analyzer = trend_analyzer
        self._insight_generator = insight_generator

    def generate_insights(
        self,
        namespace: Optional[str] = None,
        limit: int = 500,
    ) -> InsightReport:
        """
        Run the full insight generation pipeline.

        Args:
            namespace: Optional namespace to filter memories by category.
                       When provided, only memories with this category are
                       retrieved. When None, all memories up to limit are used.
            limit: Maximum number of memories to retrieve. Defaults to 500.

        Returns:
            InsightReport containing insights, pattern_count, trend_count,
            and memory_count. Returns an empty report when no memories are
            retrieved.

        Raises:
            MemoryRetrievalError: If the memory retrieval operation fails.
        """
        # Step 1 — Build QueryParameters
        params: QueryParameters = {"limit": limit}
        if namespace is not None:
            params["category"] = namespace

        # Step 2 — Retrieve memories (the only memory call)
        try:
            result = self._memory_interface.retrieve(params=params)
        except MemoryRetrievalError:
            raise
        except Exception as exc:
            raise MemoryRetrievalError(
                f"Unexpected error during memory retrieval: {exc}"
            ) from exc

        # Step 3 — Extract memories
        memories: List[MemoryEntry] = result["memories"]

        # Early return for empty memory list
        if not memories:
            return InsightReport(
                insights=[],
                pattern_count=0,
                trend_count=0,
                memory_count=0,
            )

        # Step 4 — Detect patterns
        patterns = self._pattern_detector.detect(memories)

        # Step 5 — Analyze trends
        trends = self._trend_analyzer.analyze(patterns, memories)

        # Step 6 — Generate insights
        insights = self._insight_generator.generate(patterns, trends)

        # Step 7 — Return InsightReport
        return InsightReport(
            insights=insights,
            pattern_count=len(patterns),
            trend_count=len(trends),
            memory_count=len(memories),
        )
