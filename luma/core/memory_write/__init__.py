"""Memory Write Engine - Intelligent memory persistence for conversation interactions.

The Memory Write Engine analyzes interactions after the Reasoning Engine produces
responses and decides whether new information should be stored as long-term memory.
The engine extracts candidate memories, scores their importance, and persists
valuable information following clean architecture principles.

Main Components:
    - MemoryWriteEngine: Main orchestrator coordinating the pipeline
    - MemoryExtractor: Identifies candidate memories from interactions
    - ImportanceScorer: Evaluates importance and filters by threshold
    - MemoryWriter: Persists validated memories with deduplication

Data Models:
    - MemoryCandidate: Potential memory before scoring
    - ScoredMemory: Candidate with importance score
    - StoredMemory: Persisted memory with metadata
    - MemoryWriteResult: Result of processing operation
"""

from luma.core.memory_write.schemas import (
    MemoryCandidate,
    MemoryType,
    ScoredMemory,
    StoredMemory,
    MemoryWriteResult,
)
from luma.core.memory_write.importance_scorer import ImportanceScorer
from luma.core.memory_write.memory_extractor import MemoryExtractor
from luma.core.memory_write.memory_writer import MemoryWriter
from luma.core.memory_write.memory_write_engine import MemoryWriteEngine

__all__ = [
    "MemoryCandidate",
    "MemoryType",
    "ScoredMemory",
    "StoredMemory",
    "MemoryWriteResult",
    "ImportanceScorer",
    "MemoryExtractor",
    "MemoryWriter",
    "MemoryWriteEngine",
]
