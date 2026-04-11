"""
Utility functions for memory lifecycle management.

This module provides helper functions for extracting metadata from memory entries,
specifically importance scores and final scores used in pruning decisions.
"""

from typing import Any, Dict
from luma.core.memory_interface import MemoryEntry


def extract_importance(memory_entry: MemoryEntry) -> float:
    """
    Extract importance score from memory entry.
    
    Looks for importance in metadata["importance"] or metadata["context"]["importance"].
    Defaults to 0.0 if not found. Values are clamped to [0, 1] range.
    
    The importance score is used to protect critical memories from deletion during
    pruning operations. Memories with importance >= min_importance_protected are
    preserved regardless of age or relevance score.
    
    Args:
        memory_entry: Memory entry from MemoryInterface containing metadata
    
    Returns:
        Importance score in [0, 1] range, defaults to 0.0 if not found
    
    Example:
        >>> entry = {
        ...     "id": "mem_123",
        ...     "content": "Important data",
        ...     "metadata": {"importance": 0.9},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "system",
        ...     "tags": []
        ... }
        >>> extract_importance(entry)
        0.9
        
        >>> entry_with_context = {
        ...     "id": "mem_456",
        ...     "content": "Data",
        ...     "metadata": {"context": {"importance": 0.5}},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "user",
        ...     "tags": []
        ... }
        >>> extract_importance(entry_with_context)
        0.5
    """
    metadata = memory_entry.get("metadata", {})
    
    # Check direct importance field
    if "importance" in metadata:
        importance = metadata["importance"]
        if isinstance(importance, (int, float)):
            return max(0.0, min(1.0, float(importance)))
    
    # Check context.importance field
    if "context" in metadata and isinstance(metadata["context"], dict):
        importance = metadata["context"].get("importance", 0.0)
        if isinstance(importance, (int, float)):
            return max(0.0, min(1.0, float(importance)))
    
    return 0.0


def extract_final_score(memory_entry: MemoryEntry) -> float:
    """
    Extract final score from memory entry.
    
    Looks for final_score in metadata["final_score"] or metadata["score"].
    Defaults to 0.0 if not found. Values are clamped to [0, 1] range.
    
    The final score represents the computed relevance score used for ranking
    and pruning decisions. Memories with scores below pruning_score_threshold
    are candidates for deletion (unless protected by importance).
    
    Args:
        memory_entry: Memory entry from MemoryInterface containing metadata
    
    Returns:
        Final score in [0, 1] range, defaults to 0.0 if not found
    
    Example:
        >>> entry = {
        ...     "id": "mem_123",
        ...     "content": "Relevant data",
        ...     "metadata": {"final_score": 0.85},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "system",
        ...     "tags": []
        ... }
        >>> extract_final_score(entry)
        0.85
        
        >>> entry_with_score = {
        ...     "id": "mem_456",
        ...     "content": "Data",
        ...     "metadata": {"score": 0.3},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "user",
        ...     "tags": []
        ... }
        >>> extract_final_score(entry_with_score)
        0.3
    """
    metadata = memory_entry.get("metadata", {})
    
    # Check final_score field
    if "final_score" in metadata:
        score = metadata["final_score"]
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    
    # Check score field
    if "score" in metadata:
        score = metadata["score"]
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    
    return 0.0


def extract_namespace(memory_entry: MemoryEntry) -> str:
    """
    Extract namespace from memory entry.
    
    Looks for namespace in metadata["namespace"] or metadata["context"]["namespace"].
    Defaults to "default" if not found. This allows grouping memories by namespace
    for independent cap enforcement.
    
    The namespace is used to isolate memory collections and enforce per-namespace
    limits independently. Memories without an explicit namespace are assigned to
    the "default" namespace.
    
    Args:
        memory_entry: Memory entry from MemoryInterface containing metadata
    
    Returns:
        Namespace string, defaults to "default" if not found
    
    Example:
        >>> entry = {
        ...     "id": "mem_123",
        ...     "content": "Data",
        ...     "metadata": {"namespace": "conversation"},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "system",
        ...     "tags": []
        ... }
        >>> extract_namespace(entry)
        'conversation'
        
        >>> entry_with_context = {
        ...     "id": "mem_456",
        ...     "content": "Data",
        ...     "metadata": {"context": {"namespace": "system"}},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "user",
        ...     "tags": []
        ... }
        >>> extract_namespace(entry_with_context)
        'system'
        
        >>> entry_no_namespace = {
        ...     "id": "mem_789",
        ...     "content": "Data",
        ...     "metadata": {},
        ...     "timestamp": "2024-01-01T00:00:00Z",
        ...     "category": "user",
        ...     "tags": []
        ... }
        >>> extract_namespace(entry_no_namespace)
        'default'
    """
    metadata = memory_entry.get("metadata", {})
    
    # Check direct namespace field
    if "namespace" in metadata:
        namespace = metadata["namespace"]
        if isinstance(namespace, str):
            return namespace
    
    # Check context.namespace field
    if "context" in metadata and isinstance(metadata["context"], dict):
        namespace = metadata["context"].get("namespace")
        if isinstance(namespace, str):
            return namespace
    
    return "default"
