"""
PreferenceDetector: Identifies user preferences from interaction history and
memory patterns with evidence-backed confidence scores.

Algorithm:
1. Signal extraction from memory tags, categories, and behavior phrases
2. Frequency counting across distinct memories
3. Confidence scoring with optional profile-based boost
4. Threshold filtering
5. Reason generation
6. Deterministic sorting
"""

from typing import Dict, List

from luma.core.memory_interface import MemoryEntry
from luma.core.personalization.schemas import Preference, UserProfile


class PreferenceDetector:
    """
    Detects user preferences from a list of memory entries and a UserProfile.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence threshold for a preference to be emitted.
        Must be in [0.0, 1.0]. Defaults to 0.6.
    min_frequency : int
        Minimum number of distinct memories a signal must appear in to be
        considered. Must be >= 1. Defaults to 2.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_frequency: int = 2,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {min_confidence}"
            )
        if min_frequency < 1:
            raise ValueError(
                f"min_frequency must be >= 1, got {min_frequency}"
            )
        self._min_confidence = min_confidence
        self._min_frequency = min_frequency

    def detect(
        self,
        memories: List[MemoryEntry],
        profile: UserProfile,
    ) -> List[Preference]:
        """
        Detect preferences from memories and a user profile.

        Parameters
        ----------
        memories : List[MemoryEntry]
            Memory entries to analyse.
        profile : UserProfile
            The current user profile providing interests and behavior_patterns
            for signal boosting.

        Returns
        -------
        List[Preference]
            Detected preferences sorted by (confidence DESC, preference ASC).
            Returns [] when memories is empty.
        """
        if not memories:
            return []

        # Step 1 & 2: Signal extraction + frequency counting
        # signal → list of memory IDs (each ID appears at most once per signal)
        signal_freq: Dict[str, List[str]] = {}

        for memory in memories:
            mem_id = memory["id"]
            signals_in_this_memory: set = set()

            # Tags
            for tag in memory.get("tags", []):
                if tag:
                    signals_in_this_memory.add(tag)

            # Category
            category = memory.get("category", "")
            if category:
                signals_in_this_memory.add(category)

            # Behavior phrases from profile that appear in content
            content = memory.get("content", "") or ""
            content_lower = content.lower()
            for phrase in profile.behavior_patterns:
                if phrase and phrase.lower() in content_lower:
                    signals_in_this_memory.add(phrase)

            # Record each signal once per memory
            for signal in signals_in_this_memory:
                if signal not in signal_freq:
                    signal_freq[signal] = []
                if mem_id not in signal_freq[signal]:
                    signal_freq[signal].append(mem_id)

        # Build profile lookup sets for boost check
        profile_signals = set(profile.interests) | set(profile.behavior_patterns)

        total_memories = len(memories)
        preferences: List[Preference] = []

        # Step 3 & 4: Confidence scoring + threshold filtering
        for signal, ids in signal_freq.items():
            if len(ids) < self._min_frequency:
                continue

            confidence = min(1.0, len(ids) / max(1, total_memories))

            # Boost if signal appears in profile interests or behavior_patterns
            if signal in profile_signals:
                confidence = min(1.0, confidence + 0.1)

            if confidence < self._min_confidence:
                continue

            # Step 5: Reason generation
            id_preview = ", ".join(ids[:3])
            ellipsis = "..." if len(ids) > 3 else ""
            reason = f"Detected in {len(ids)} memories: {id_preview}{ellipsis}"

            preferences.append(
                Preference(
                    preference=signal,
                    confidence=confidence,
                    reason=reason,
                )
            )

        # Step 6: Sort by (confidence DESC, preference ASC)
        preferences.sort(key=lambda p: (-p.confidence, p.preference))

        return preferences
