"""
Pattern Detector Module.

Identifies recurring keywords and categories across a list of memories
using an O(n log n) pipeline: tokenize → filter → score → sort.
"""

import logging
import re
from typing import Dict, List

from luma.core.memory_interface import MemoryEntry
from luma.core.insight.schemas import PatternResult

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects recurring patterns (keywords and categories) in a list of memories.

    Algorithm (O(n log n)):
      1. Tokenize: extract lowercase tokens from content; record category.
      2. Filter: discard entries with frequency < min_frequency.
      3. Score: confidence = min(1.0, freq / len(memories)); discard if < min_confidence.
      4. Sort: by (frequency DESC, pattern ASC) for deterministic output.
      5. Return List[PatternResult].
    """

    def __init__(self, min_frequency: int = 2, min_confidence: float = 0.5) -> None:
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        self._min_frequency = min_frequency
        self._min_confidence = min_confidence

    def detect(self, memories: List[MemoryEntry]) -> List[PatternResult]:
        """
        Detect recurring patterns in the provided memories.

        Args:
            memories: List of MemoryEntry objects to analyse.

        Returns:
            Sorted list of PatternResult objects (frequency DESC, pattern ASC).
        """
        if not memories:
            return []

        # Step 1 — Tokenize
        # keyword_freq: keyword -> set of memory IDs (set avoids duplicates per memory)
        keyword_freq: Dict[str, set] = {}
        # category_freq: category -> set of memory IDs
        category_freq: Dict[str, set] = {}

        for memory in memories:
            mem_id = memory["id"]
            content = memory.get("content")
            category = memory.get("category")

            if content is None:
                logger.warning(
                    "Memory %s has None content; skipping token extraction.", mem_id
                )
            else:
                tokens = re.split(r"[^a-z0-9]+", content.lower())
                seen_tokens: set = set()
                for token in tokens:
                    if token and token not in seen_tokens:
                        seen_tokens.add(token)
                        keyword_freq.setdefault(token, set()).add(mem_id)

            if category:
                category_freq.setdefault(category, set()).add(mem_id)

        total = len(memories)
        results: List[PatternResult] = []

        # Steps 2 & 3 — Filter and score keywords
        for keyword, ids in keyword_freq.items():
            freq = len(ids)
            if freq < self._min_frequency:
                continue
            confidence = min(1.0, freq / total)
            if confidence < self._min_confidence:
                continue
            results.append(
                PatternResult(
                    pattern_type="keyword",
                    pattern=keyword,
                    frequency=freq,
                    confidence=confidence,
                    evidence=sorted(ids),
                )
            )

        # Steps 2 & 3 — Filter and score categories
        for category, ids in category_freq.items():
            freq = len(ids)
            if freq < self._min_frequency:
                continue
            confidence = min(1.0, freq / total)
            if confidence < self._min_confidence:
                continue
            results.append(
                PatternResult(
                    pattern_type="category",
                    pattern=category,
                    frequency=freq,
                    confidence=confidence,
                    evidence=sorted(ids),
                )
            )

        # Step 4 — Sort: frequency DESC, pattern ASC (deterministic total order)
        results.sort(key=lambda r: (-r.frequency, r.pattern))

        return results
