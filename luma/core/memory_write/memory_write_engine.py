"""Memory Write Engine orchestrator.

This module coordinates the memory processing pipeline: extraction, scoring,
and persistence. It accepts conversation interactions and returns structured
results indicating which memories were stored and which were filtered out.
"""

import time
from typing import List, Optional
from .schemas import MemoryCandidate, ScoredMemory, MemoryWriteResult
from .memory_extractor import MemoryExtractor
from .importance_scorer import ImportanceScorer
from .memory_writer import MemoryWriter
from ..metrics_collector import MetricsCollector
from ..structured_logger import StructuredLogger


class MemoryWriteEngine:
    """Orchestrates the memory processing pipeline.
    
    The MemoryWriteEngine coordinates the three-stage memory processing pipeline:
    1. Extraction: Identify candidate memories from interactions
    2. Scoring: Evaluate importance and filter by threshold
    3. Persistence: Store validated memories with deduplication
    
    The engine uses dependency injection for all components, enabling modular
    testing and future extensibility.
    
    Attributes:
        extractor: MemoryExtractor for candidate identification
        scorer: ImportanceScorer for importance evaluation
        writer: MemoryWriter for persistence operations
        metrics_collector: Optional MetricsCollector for observability
        logger: Optional StructuredLogger for structured logging
    """
    
    def __init__(
        self,
        extractor: MemoryExtractor,
        scorer: ImportanceScorer,
        writer: MemoryWriter,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """Initialize with injected dependencies.
        
        Args:
            extractor: MemoryExtractor instance for candidate extraction
            scorer: ImportanceScorer instance for importance evaluation
            writer: MemoryWriter instance for memory persistence
            metrics_collector: Optional MetricsCollector for observability
            logger: Optional StructuredLogger for structured logging
        """
        self.extractor = extractor
        self.scorer = scorer
        self.writer = writer
        self.metrics_collector = metrics_collector
        self.logger = logger
    
    def process(
        self,
        user_query: str,
        system_response: str
    ) -> MemoryWriteResult:
        """Process an interaction and store valuable memories.

        Orchestrates the complete memory processing pipeline:
        1. Extract candidate memories from the interaction
        2. Score each candidate for importance
        3. Filter candidates below threshold
        4. Store memories above threshold with deduplication
        5. Return structured result with stored and ignored memories

        Args:
            user_query: The user's query text
            system_response: The system's response text

        Returns:
            MemoryWriteResult with stored and ignored memories

        Raises:
            ValueError: If user_query or system_response is None or empty
        """
        # Start timing measurement
        start_time = time.perf_counter()

        try:
            # Validate inputs
            if not user_query or not user_query.strip():
                raise ValueError("user_query must be non-empty")
            if not system_response or not system_response.strip():
                raise ValueError("system_response must be non-empty")

            # Stage 1: Extract candidate memories
            candidates = self.extractor.extract_candidates(user_query, system_response)

            # Stage 2: Score each candidate and collect results
            scored_memories: List[ScoredMemory] = []
            ignored_candidates: List[MemoryCandidate] = []

            for candidate in candidates:
                scored = self.scorer.score_memory(candidate)
                if scored is not None:
                    # Memory above threshold
                    scored_memories.append(scored)
                else:
                    # Memory below threshold - filtered out
                    ignored_candidates.append(candidate)

            # Stage 3: Store each scored memory above threshold
            stored_memories = []
            for scored_memory in scored_memories:
                stored = self.writer.store_memory(scored_memory)
                stored_memories.append(stored)

            # Build result
            result = MemoryWriteResult(
                stored_memories=stored_memories,
                ignored_memories=ignored_candidates
            )

            # Record successful operation metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            if self.metrics_collector is not None:
                self.metrics_collector.record_duration("memory_write_latency_ms", duration_ms)
                self.metrics_collector.increment("memory_write_count")

            # Log successful write event
            if self.logger is not None:
                self.logger.log("memory_write_completed", {
                    "duration_ms": duration_ms,
                    "candidates_extracted": len(candidates),
                    "memories_stored": len(stored_memories),
                    "memories_ignored": len(ignored_candidates),
                    "user_query_length": len(user_query),
                    "system_response_length": len(system_response)
                })

            return result

        except Exception as e:
            # Record failure metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            if self.metrics_collector is not None:
                self.metrics_collector.increment("memory_write_failures")
                # Still record latency for failed operations
                self.metrics_collector.record_duration("memory_write_latency_ms", duration_ms)

            # Log failure event
            if self.logger is not None:
                self.logger.log("memory_write_failed", {
                    "duration_ms": duration_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "user_query_length": len(user_query) if user_query else 0,
                    "system_response_length": len(system_response) if system_response else 0
                })

            # Re-raise the exception to maintain identical behavior
            raise

