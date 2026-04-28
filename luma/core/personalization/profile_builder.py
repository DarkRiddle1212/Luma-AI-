"""
ProfileBuilder: Aggregates long-term signals from stored memories and insights
to construct a UserProfile.

Algorithm:
1. Keyword extraction (interests) from memory content + high-confidence insight text
2. Behavior pattern extraction from recurring activity phrases
3. Interaction style inference from average word count
4. Strengths extraction from high-confidence insights
5. Evidence mapping
"""

import re
from typing import Any, Dict, List, Union

from luma.core.memory_interface import MemoryEntry
from luma.core.personalization.schemas import UserProfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "its", "let", "put", "say", "she", "too", "use",
}

_BEHAVIOR_INDICATORS = [
    "i always",
    "i prefer",
    "i tend to",
    "i usually",
    "i like to",
]

# Sentence-ending characters used to truncate behavior phrases
_SENTENCE_END_RE = re.compile(r"[.!?;]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, drop stop words and tokens < 3 chars."""
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [
        t for t in tokens
        if len(t) >= 3 and t not in _STOP_WORDS
    ]


def _extract_behavior_phrase(text: str, indicator: str) -> str:
    """
    Extract the phrase following *indicator* in *text*.
    Returns the phrase stripped, lowercased, up to 50 chars or until sentence end.
    Returns empty string if indicator not found.
    """
    lower_text = text.lower()
    idx = lower_text.find(indicator)
    if idx == -1:
        return ""
    start = idx + len(indicator)
    remainder = text[start:].strip()
    # Truncate at sentence end
    m = _SENTENCE_END_RE.search(remainder)
    if m:
        remainder = remainder[: m.start()]
    phrase = remainder.strip().lower()[:50]
    return phrase


def _get_insight_attr(insight: Any, attr: str) -> Any:
    """Access an attribute from either a Pydantic/dataclass model or a plain dict."""
    if isinstance(insight, dict):
        return insight.get(attr)
    return getattr(insight, attr, None)


# ---------------------------------------------------------------------------
# ProfileBuilder
# ---------------------------------------------------------------------------

class ProfileBuilder:
    """
    Builds a UserProfile from a list of MemoryEntry objects and insight objects.

    Parameters
    ----------
    min_keyword_frequency : int
        Minimum number of distinct memories a keyword must appear in to be
        included in interests / behavior_patterns. Must be >= 1.
    """

    def __init__(self, min_keyword_frequency: int = 2) -> None:
        if min_keyword_frequency < 1:
            raise ValueError(
                f"min_keyword_frequency must be >= 1, got {min_keyword_frequency}"
            )
        self._min_freq = min_keyword_frequency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        memories: List[MemoryEntry],
        insights: List[Any],
    ) -> UserProfile:
        """
        Build a UserProfile from memories and insights.

        Parameters
        ----------
        memories : List[MemoryEntry]
            Retrieved memory entries. Entries with missing/None content are skipped.
        insights : List[Any]
            Insight objects (Pydantic models or plain dicts). Insights with
            confidence outside [0.0, 1.0] are skipped.

        Returns
        -------
        UserProfile
        """
        # Filter valid memories (non-None, non-missing content)
        valid_memories = [
            m for m in memories
            if m.get("content") is not None
        ]

        # Filter valid insights (confidence in [0.0, 1.0])
        valid_insights = []
        for ins in insights:
            conf = _get_insight_attr(ins, "confidence")
            if conf is not None and 0.0 <= conf <= 1.0:
                valid_insights.append(ins)

        # Empty memories → return default profile
        if not valid_memories:
            return UserProfile(
                interests=[],
                behavior_patterns=[],
                interaction_style="balanced",
                strengths=[],
                evidence={},
            )

        interests, interests_evidence = self._extract_interests(
            valid_memories, valid_insights
        )
        behavior_patterns, bp_evidence = self._extract_behavior_patterns(valid_memories)
        interaction_style = self._infer_interaction_style(valid_memories)
        strengths, strengths_evidence = self._extract_strengths(valid_insights)

        evidence: Dict[str, List[str]] = {}
        if interests_evidence:
            evidence["interests"] = sorted(interests_evidence)
        if bp_evidence:
            evidence["behavior_patterns"] = sorted(bp_evidence)
        if strengths_evidence:
            evidence["strengths"] = sorted(strengths_evidence)

        return UserProfile(
            interests=interests,
            behavior_patterns=behavior_patterns,
            interaction_style=interaction_style,
            strengths=strengths,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Step 1: Keyword extraction (interests)
    # ------------------------------------------------------------------

    def _extract_interests(
        self,
        memories: List[MemoryEntry],
        insights: List[Any],
    ):
        """Return (interests_list, contributing_memory_ids_set)."""
        # Build keyword → list of memory IDs
        keyword_freq: Dict[str, List[str]] = {}
        for mem in memories:
            content = mem.get("content") or ""
            mem_id = mem["id"]
            for token in _tokenize(content):
                if token not in keyword_freq:
                    keyword_freq[token] = []
                if mem_id not in keyword_freq[token]:
                    keyword_freq[token].append(mem_id)

        # Retain keywords meeting frequency threshold
        retained = {
            kw: ids
            for kw, ids in keyword_freq.items()
            if len(ids) >= self._min_freq
        }

        # Sort by (frequency DESC, keyword ASC)
        sorted_keywords = sorted(
            retained.keys(),
            key=lambda kw: (-len(retained[kw]), kw),
        )

        # Supplement from high-confidence insight text
        insight_keywords: List[str] = []
        for ins in insights:
            conf = _get_insight_attr(ins, "confidence")
            text = _get_insight_attr(ins, "text") or ""
            if conf is not None and conf >= 0.6:
                for token in _tokenize(text):
                    if token not in retained and token not in insight_keywords:
                        insight_keywords.append(token)

        # Deduplicate: sorted_keywords first, then insight supplements
        seen = set(sorted_keywords)
        interests = list(sorted_keywords)
        for kw in insight_keywords:
            if kw not in seen:
                interests.append(kw)
                seen.add(kw)

        # Collect contributing memory IDs
        contributing_ids: set = set()
        for kw in sorted_keywords:
            contributing_ids.update(retained[kw])

        return interests, contributing_ids

    # ------------------------------------------------------------------
    # Step 2: Behavior pattern extraction
    # ------------------------------------------------------------------

    def _extract_behavior_patterns(
        self,
        memories: List[MemoryEntry],
    ):
        """Return (behavior_patterns_list, contributing_memory_ids_set)."""
        # phrase → set of memory IDs
        phrase_memories: Dict[str, set] = {}

        for mem in memories:
            content = mem.get("content") or ""
            mem_id = mem["id"]
            for indicator in _BEHAVIOR_INDICATORS:
                phrase = _extract_behavior_phrase(content, indicator)
                if phrase:
                    if phrase not in phrase_memories:
                        phrase_memories[phrase] = set()
                    phrase_memories[phrase].add(mem_id)

        # Retain phrases meeting frequency threshold
        retained = {
            phrase: ids
            for phrase, ids in phrase_memories.items()
            if len(ids) >= self._min_freq
        }

        # Sort by (frequency DESC, phrase ASC)
        sorted_phrases = sorted(
            retained.keys(),
            key=lambda p: (-len(retained[p]), p),
        )

        # Collect contributing memory IDs
        contributing_ids: set = set()
        for phrase in sorted_phrases:
            contributing_ids.update(retained[phrase])

        return sorted_phrases, contributing_ids

    # ------------------------------------------------------------------
    # Step 3: Interaction style inference
    # ------------------------------------------------------------------

    def _infer_interaction_style(self, memories: List[MemoryEntry]) -> str:
        """Infer interaction style from average word count of memory content."""
        if not memories:
            return "balanced"

        total_words = 0
        for mem in memories:
            content = mem.get("content") or ""
            total_words += len(content.split())

        avg = total_words / len(memories)

        if avg < 15:
            return "concise"
        elif avg > 50:
            return "detailed"
        else:
            return "balanced"

    # ------------------------------------------------------------------
    # Step 4: Strengths extraction
    # ------------------------------------------------------------------

    def _extract_strengths(self, insights: List[Any]):
        """Return (strengths_list, contributing_insight_evidence_ids_set)."""
        strengths_set: set = set()
        evidence_ids: set = set()

        for ins in insights:
            conf = _get_insight_attr(ins, "confidence")
            text = _get_insight_attr(ins, "text") or ""
            evidence = _get_insight_attr(ins, "evidence") or []

            if conf is None or conf < 0.6:
                continue

            # First noun phrase = first 1-3 words of insight.text, normalized
            words = text.strip().split()
            if not words:
                continue
            noun_phrase = " ".join(words[:3]).lower()
            # Strip trailing punctuation
            noun_phrase = noun_phrase.rstrip(".,;:!?")
            if noun_phrase:
                strengths_set.add(noun_phrase)
                if isinstance(evidence, list):
                    evidence_ids.update(evidence)

        strengths = sorted(strengths_set)
        return strengths, evidence_ids
