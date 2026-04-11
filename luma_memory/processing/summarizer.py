"""
Context summarization for Luma Memory Module.

This module provides functionality to identify redundant memory entries
and create consolidated summaries to reduce storage overhead while
preserving essential information.
"""

from typing import List, Tuple, Dict, Any, TYPE_CHECKING
from datetime import datetime, UTC
import uuid

from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus

if TYPE_CHECKING:
    from luma_memory.config import MemoryModuleConfig


class ContextSummarizer:
    """
    Identifies and summarizes redundant memory entries.
    
    The ContextSummarizer analyzes memory entries to detect similar or
    redundant context information and creates consolidated summary entries
    that preserve essential information while reducing storage overhead.
    
    Attributes:
        similarity_threshold: Threshold (0.0-1.0) for considering entries similar.
                            Higher values require more similarity.
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.8,
        entry_count_threshold: int = 1000,
        storage_size_threshold_mb: int = 100
    ):
        """
        Initialize the ContextSummarizer.

        Args:
            similarity_threshold: Similarity threshold for grouping entries (0.0-1.0).
                                Default is 0.8 (80% similarity).
            entry_count_threshold: Number of entries before triggering summarization.
                                 Default is 1000 entries.
            storage_size_threshold_mb: Storage size in MB before triggering summarization.
                                      Default is 100 MB.

        Raises:
            ValueError: If similarity_threshold is not between 0.0 and 1.0.
            ValueError: If thresholds are not positive integers.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if entry_count_threshold <= 0:
            raise ValueError("entry_count_threshold must be a positive integer")

        if storage_size_threshold_mb <= 0:
            raise ValueError("storage_size_threshold_mb must be a positive integer")

        self.similarity_threshold = similarity_threshold
        self.entry_count_threshold = entry_count_threshold
        self.storage_size_threshold_bytes = storage_size_threshold_mb * 1_000_000  # Convert MB to bytes

    @classmethod
    def from_config(cls, config: "MemoryModuleConfig") -> "ContextSummarizer":
        """
        Create a ContextSummarizer instance from a MemoryModuleConfig.
        
        This factory method initializes the summarizer with configuration values
        from the provided config object, ensuring consistency across the application.
        
        Args:
            config: MemoryModuleConfig instance containing summarization settings.
        
        Returns:
            ContextSummarizer: Configured summarizer instance.
        
        Example:
            >>> from luma_memory.config import MemoryModuleConfig
            >>> config = MemoryModuleConfig.load_config()
            >>> summarizer = ContextSummarizer.from_config(config)
            >>> print(summarizer.similarity_threshold)
            0.8
        """
        return cls(
            similarity_threshold=config.similarity_threshold,
            entry_count_threshold=config.summarization_threshold,
            storage_size_threshold_mb=config.max_storage_size_mb
        )


    
    def identify_redundant_entries(
        self,
        entries: List[MemoryEntry]
    ) -> List[Tuple[str, List[str]]]:
        """
        Identify groups of entries with redundant context.
        
        Analyzes a list of memory entries and groups those with similar
        context information. Each group can be summarized into a single entry.
        
        Args:
            entries: List of MemoryEntry objects to analyze.
        
        Returns:
            List of tuples, where each tuple contains:
            - summary_id: Generated ID for the summary entry
            - entry_ids: List of entry IDs that should be summarized together
        
        Example:
            >>> summarizer = ContextSummarizer(similarity_threshold=0.8)
            >>> entries = [entry1, entry2, entry3]
            >>> groups = summarizer.identify_redundant_entries(entries)
            >>> # Returns: [('summary-uuid-1', ['id1', 'id2']), ...]
        """
        if not entries:
            return []
        
        # Groups of similar entries: {group_index: [entry_ids]}
        groups: Dict[int, List[str]] = {}
        # Track which entries have been assigned to groups
        assigned_entries = set()
        
        # Compare each entry with others to find similar ones
        for i, entry1 in enumerate(entries):
            if entry1.id in assigned_entries:
                continue
            
            # Start a new group with this entry
            current_group = [entry1.id]
            assigned_entries.add(entry1.id)
            
            # Find similar entries
            for j, entry2 in enumerate(entries[i + 1:], start=i + 1):
                if entry2.id in assigned_entries:
                    continue
                
                # Calculate similarity between entries
                similarity = self._calculate_similarity(entry1, entry2)
                
                if similarity >= self.similarity_threshold:
                    current_group.append(entry2.id)
                    assigned_entries.add(entry2.id)
            
            # Only create groups with multiple entries
            if len(current_group) > 1:
                groups[i] = current_group
        
        # Convert groups to list of tuples with generated summary IDs
        result = []
        for group_entries in groups.values():
            summary_id = f"summary-{uuid.uuid4()}"
            result.append((summary_id, group_entries))
        
        return result
    
    def summarize_entries(
        self,
        entries: List[MemoryEntry]
    ) -> tuple[MemoryEntry, List[str]]:
        """
        Create a summary entry from multiple entries.
        
        Consolidates multiple memory entries into a single summary entry
        that preserves essential information while discarding redundant details.
        
        The method returns both the summary entry and a list of entry IDs that
        should have their parent_id field updated to link to the summary.
        
        Essential information preserved:
        - All unique tags from all entries
        - Merged context with value history tracking
        - Highest sensitivity level
        - Time range (earliest to latest)
        - All unique actions
        - Device information
        - Entry IDs for traceability
        
        Args:
            entries: List of MemoryEntry objects to summarize.
        
        Returns:
            A tuple of (summary_entry, entry_ids_to_link) where:
                - summary_entry: A new MemoryEntry representing the summary
                - entry_ids_to_link: List of entry IDs that should be linked to this summary
        
        Raises:
            ValueError: If entries list is empty.
        
        Example:
            >>> summarizer = ContextSummarizer()
            >>> summary, entry_ids = summarizer.summarize_entries([entry1, entry2, entry3])
            >>> print(summary.summary)  # Contains consolidated information
            >>> print(entry_ids)  # ['id1', 'id2', 'id3'] - entries to link to summary
        """
        if not entries:
            raise ValueError("Cannot summarize empty list of entries")
        
        # Sort entries by timestamp to maintain chronological order
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Collect entry IDs that will be linked to this summary
        entry_ids_to_link = [entry.id for entry in sorted_entries]
        
        # Use the earliest timestamp
        earliest_timestamp = sorted_entries[0].timestamp
        
        # Use the latest timestamp for created_at
        latest_timestamp = sorted_entries[-1].timestamp
        
        # Collect all unique actions
        actions = [entry.action for entry in sorted_entries]
        
        # Merge contexts, preserving unique information
        merged_context = self._merge_contexts([entry.context for entry in sorted_entries])
        
        # Add metadata about summarized entries to preserve essential information
        merged_context["_summary_metadata"] = {
            "entry_count": len(entries),
            "entry_ids": [entry.id for entry in sorted_entries],
            "time_range": {
                "start": earliest_timestamp.isoformat(),
                "end": latest_timestamp.isoformat()
            },
            "unique_actions": list(dict.fromkeys(actions)),
            "devices": list(dict.fromkeys(entry.device_id for entry in sorted_entries)),
            "sensitivity_levels": list(dict.fromkeys(entry.sensitivity.value for entry in sorted_entries))
        }
        
        # Collect all unique tags
        all_tags = set()
        for entry in sorted_entries:
            all_tags.update(entry.tags)
        
        # Use the highest sensitivity level among entries
        max_sensitivity = max(
            (entry.sensitivity for entry in sorted_entries),
            key=lambda s: list(SensitivityLevel).index(s)
        )
        
        # Use the first device_id (could be enhanced to track all devices)
        device_id = sorted_entries[0].device_id
        
        # Create summary text
        summary_text = self._create_summary_text(sorted_entries)
        
        # Create the summary entry
        summary_entry = MemoryEntry(
            id=f"summary-{uuid.uuid4()}",
            timestamp=earliest_timestamp,
            action=f"Summary of {len(entries)} entries: {', '.join(set(actions))}",
            context=merged_context,
            sensitivity=max_sensitivity,
            device_id=device_id,
            sync_status=SyncStatus.PENDING,
            tags=sorted(list(all_tags)),
            summary=summary_text,
            parent_id=None,  # This is a parent summary
            created_at=latest_timestamp,
            updated_at=datetime.now(UTC) if hasattr(datetime, 'UTC') else datetime.now(UTC)
        )
        
        return summary_entry, entry_ids_to_link
    
    def should_trigger_summarization(
        self,
        entry_count: int,
        storage_size: int
    ) -> bool:
        """
        Determine if summarization should be triggered.
        
        Evaluates whether automatic summarization should be performed based
        on configurable thresholds for entry count and storage size.
        
        Args:
            entry_count: Current number of memory entries.
            storage_size: Current storage size in bytes.
        
        Returns:
            True if summarization should be triggered, False otherwise.
        
        Example:
            >>> summarizer = ContextSummarizer()
            >>> if summarizer.should_trigger_summarization(1500, 50_000_000):
            ...     # Perform summarization
            ...     pass
        """
        return (entry_count >= self.entry_count_threshold or 
                storage_size >= self.storage_size_threshold_bytes)
    
    def _calculate_similarity(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        """
        Calculate similarity score between two memory entries.
        
        Uses multiple factors to determine similarity:
        - Action similarity
        - Context overlap
        - Tag overlap
        - Temporal proximity
        
        Args:
            entry1: First memory entry.
            entry2: Second memory entry.
        
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        scores = []
        
        # Action similarity (exact match or substring)
        action_score = 1.0 if entry1.action == entry2.action else 0.0
        if action_score == 0.0:
            # Check for substring match
            if entry1.action in entry2.action or entry2.action in entry1.action:
                action_score = 0.5
        scores.append(action_score)
        
        # Context similarity (Jaccard similarity of keys)
        context1_keys = set(entry1.context.keys())
        context2_keys = set(entry2.context.keys())
        if context1_keys or context2_keys:
            context_score = len(context1_keys & context2_keys) / len(context1_keys | context2_keys)
        else:
            context_score = 1.0  # Both empty
        scores.append(context_score)
        
        # Tag similarity (Jaccard similarity)
        tags1 = set(entry1.tags)
        tags2 = set(entry2.tags)
        if tags1 or tags2:
            tag_score = len(tags1 & tags2) / len(tags1 | tags2)
        else:
            tag_score = 1.0  # Both empty
        scores.append(tag_score)
        
        # Temporal proximity (entries within 1 hour get bonus)
        time_diff = abs((entry1.timestamp - entry2.timestamp).total_seconds())
        temporal_score = 1.0 if time_diff <= 3600 else 0.5 if time_diff <= 86400 else 0.0
        scores.append(temporal_score)
        
        # Weighted average (action and context are more important)
        weights = [0.3, 0.4, 0.2, 0.1]  # action, context, tags, temporal
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_score
    
    def _merge_contexts(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple context dictionaries, preserving essential information.

        Combines context information from multiple entries, preserving
        unique values and handling conflicts intelligently to avoid data loss.

        Essential information preservation strategy:
        - Lists: Merge and deduplicate while preserving order
        - Dicts: Recursively merge, keeping all unique keys
        - Scalars: Keep most recent value but track history if values differ
        - Special keys: Preserve all occurrences of critical fields

        Args:
            contexts: List of context dictionaries.

        Returns:
            Merged context dictionary with essential information preserved.
        """
        if not contexts:
            return {}

        merged = {}

        # Track value history for important scalar fields to preserve changes
        value_history = {}

        for context in contexts:
            for key, value in context.items():
                if key not in merged:
                    # First occurrence - just add it
                    merged[key] = value
                    if not isinstance(value, (list, dict)):
                        value_history[key] = [value]
                elif isinstance(value, list) and isinstance(merged[key], list):
                    # Merge lists, preserving unique values while maintaining order
                    existing_set = set(str(v) for v in merged[key])
                    for item in value:
                        if str(item) not in existing_set:
                            merged[key].append(item)
                            existing_set.add(str(item))
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    # Recursively merge dictionaries
                    merged[key] = self._merge_contexts([merged[key], value])
                else:
                    # For scalar values, track history if values differ
                    if key in value_history:
                        if value not in value_history[key]:
                            value_history[key].append(value)
                    else:
                        value_history[key] = [merged[key], value]

                    # Keep the most recent (last) value
                    merged[key] = value

        # Add value history for fields that changed, preserving essential information
        changed_fields = {k: v for k, v in value_history.items() if len(v) > 1}
        if changed_fields:
            merged["_value_history"] = changed_fields

        return merged


    
    def _create_summary_text(self, entries: List[MemoryEntry]) -> str:
        """
        Create a human-readable summary text from entries, preserving essential information.
        
        Generates a comprehensive summary that includes:
        - Entry count and time range
        - All unique actions performed
        - Devices involved
        - Sensitivity levels
        - Key context information
        
        Args:
            entries: List of memory entries to summarize.
        
        Returns:
            Summary text string with essential information preserved.
        """
        if not entries:
            return ""
        
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Extract key information
        entry_count = len(entries)
        actions = [entry.action for entry in sorted_entries]
        unique_actions = list(dict.fromkeys(actions))  # Preserve order
        
        # Time range
        start_time = sorted_entries[0].timestamp.strftime("%Y-%m-%d %H:%M:%S")
        end_time = sorted_entries[-1].timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Collect unique devices
        unique_devices = list(dict.fromkeys(entry.device_id for entry in sorted_entries))
        
        # Collect sensitivity levels
        sensitivity_levels = list(dict.fromkeys(entry.sensitivity.value for entry in sorted_entries))
        
        # Extract key context fields that appear frequently
        context_keys = {}
        for entry in sorted_entries:
            for key in entry.context.keys():
                context_keys[key] = context_keys.get(key, 0) + 1
        
        # Get most common context keys (appearing in >50% of entries)
        common_keys = [k for k, count in context_keys.items() 
                      if count > len(sorted_entries) * 0.5]
        
        # Build comprehensive summary
        summary_parts = [
            f"Summary of {entry_count} related entries",
            f"Time range: {start_time} to {end_time}",
            f"Actions: {', '.join(unique_actions[:5])}"
        ]
        
        if len(unique_actions) > 5:
            summary_parts.append(f"... and {len(unique_actions) - 5} more actions")
        
        # Add device information if multiple devices
        if len(unique_devices) > 1:
            summary_parts.append(f"Devices: {', '.join(unique_devices)}")
        elif unique_devices:
            summary_parts.append(f"Device: {unique_devices[0]}")
        
        # Add sensitivity information if varied
        if len(sensitivity_levels) > 1:
            summary_parts.append(f"Sensitivity levels: {', '.join(sensitivity_levels)}")
        
        # Add common context keys to preserve essential information
        if common_keys:
            summary_parts.append(f"Common context: {', '.join(common_keys[:3])}")
        
        return " | ".join(summary_parts)
