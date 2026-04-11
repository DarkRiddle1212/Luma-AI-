"""
Cleanup Result Module.

This module provides the CleanupResult dataclass for representing the results
of memory lifecycle cleanup operations. It tracks statistics about pruning
operations and their outcomes.

Example:
    >>> from luma.core.cleanup_result import CleanupResult, CleanupStatus
    >>> 
    >>> result = CleanupResult(
    ...     age_pruned=10,
    ...     score_pruned=5,
    ...     cap_pruned=3,
    ...     total_deleted=18,
    ...     failed_deletions=0,
    ...     final_count=9982,
    ...     status=CleanupStatus.SUCCESS
    ... )
"""

from dataclasses import dataclass
from enum import Enum


class CleanupStatus(Enum):
    """
    Status of a cleanup operation.
    
    Indicates the overall outcome of a memory lifecycle cleanup operation.
    
    Attributes:
        SUCCESS: All operations completed successfully without errors
        PARTIAL: Some operations failed but cleanup partially completed
        FAILED: Cleanup operation failed completely
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class CleanupResult:
    """
    Result of a memory lifecycle cleanup operation.
    
    Contains statistics about the cleanup operation including counts of
    memories deleted by each pruning phase, error counts, and final state.
    
    Attributes:
        age_pruned: Number of memories deleted by age-based pruning.
                   Memories older than max_age_days threshold that were
                   not protected by importance score.
        
        score_pruned: Number of memories deleted by score-based pruning.
                     Memories with final_score below pruning_score_threshold
                     that were not protected by importance score.
        
        cap_pruned: Number of memories deleted by hard cap enforcement.
                   Lowest-ranked memories deleted to ensure total count
                   does not exceed max_total_memories.
        
        total_deleted: Total number of memories successfully deleted across
                      all pruning phases. Equal to age_pruned + score_pruned
                      + cap_pruned.
        
        failed_deletions: Number of deletion operations that failed.
                         Errors are logged but don't stop cleanup processing.
        
        final_count: Total number of memories remaining after cleanup
                    across all namespaces.
        
        status: Overall status of the cleanup operation.
               - SUCCESS: No errors occurred
               - PARTIAL: Some deletions failed but cleanup partially completed
               - FAILED: Cleanup operation failed completely
    
    Example:
        >>> result = CleanupResult(
        ...     age_pruned=10,
        ...     score_pruned=5,
        ...     cap_pruned=3,
        ...     total_deleted=18,
        ...     failed_deletions=0,
        ...     final_count=9982,
        ...     status=CleanupStatus.SUCCESS
        ... )
        >>> print(f"Deleted {result.total_deleted} memories")
        Deleted 18 memories
        
        >>> # Partial completion with errors
        >>> result = CleanupResult(
        ...     age_pruned=10,
        ...     score_pruned=5,
        ...     cap_pruned=3,
        ...     total_deleted=16,
        ...     failed_deletions=2,
        ...     final_count=9984,
        ...     status=CleanupStatus.PARTIAL
        ... )
        >>> print(f"Status: {result.status.value}, Errors: {result.failed_deletions}")
        Status: partial, Errors: 2
    """
    age_pruned: int
    score_pruned: int
    cap_pruned: int
    total_deleted: int
    failed_deletions: int
    final_count: int
    status: CleanupStatus
    
    def __getitem__(self, key: str):
        """
        Enable dictionary-style access for backward compatibility.
        
        Args:
            key: Attribute name to access
        
        Returns:
            Value of the attribute
        
        Raises:
            KeyError: If key doesn't exist
        """
        # Map 'errors' to 'failed_deletions' for backward compatibility
        if key == "errors":
            return self.failed_deletions
        
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in CleanupResult")
    
    def __contains__(self, key: str) -> bool:
        """
        Enable 'in' operator for checking if key exists.
        
        Args:
            key: Attribute name to check
        
        Returns:
            True if attribute exists, False otherwise
        """
        # Map 'errors' to 'failed_deletions' for backward compatibility
        if key == "errors":
            return True
        
        return hasattr(self, key)
