"""
Memory Pruner Module.

This module provides the MemoryPruner component responsible for removing
memories based on configurable pruning strategies to prevent unbounded
growth. The system supports threshold-based, percentile-based, and
capacity-based pruning strategies for flexible memory retention.

Key Features:
- Multiple pruning strategies (threshold, percentile, capacity)
- Protected memory filtering (protected=true excluded from deletion)
- Deterministic ordering by (importance, timestamp, id)
- Dry run mode for testing without persistence
- Metrics and logging integration
"""

from datetime import datetime, UTC
from typing import Optional

from luma.core.lifecycle.schemas import (
    PruningResult,
    PrunedMemory,
    PruningConfig,
    PruningStrategy,
)
from luma.core.memory_interface import MemoryInterface

try:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
except ImportError:
    # Allow optional dependency injection
    MetricsCollector = None
    StructuredLogger = None


class MemoryPruner:
    """
    Memory pruner component for threshold-based, percentile-based, and capacity-based removal.
    
    This component removes memories based on configurable pruning strategies
    to prevent unbounded growth of the memory store. It supports three pruning
    strategies:
    
    - THRESHOLD: Remove memories with importance below a fixed threshold
    - PERCENTILE: Remove the bottom N% of memories by importance
    - CAPACITY: Remove lowest-importance memories when count exceeds limit
    
    The component respects protected memory flags, ensuring protected memories
    are never deleted regardless of importance or score.
    
    The component integrates with the existing Luma infrastructure including
    MemoryInterface for storage operations, MetricsCollector for observability,
    and StructuredLogger for event logging.
    
    Attributes:
        memory_interface: MemoryInterface instance for storage operations
        pruning_config: PruningConfig with strategy and parameters
        metrics_collector: Optional MetricsCollector for metrics recording
        logger: Optional StructuredLogger for event logging
    """
    
    def __init__(
        self,
        memory_interface: MemoryInterface,
        pruning_config: PruningConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
    ):
        """
        Initialize the MemoryPruner component.
        
        Args:
            memory_interface: MemoryInterface instance for storage operations
            pruning_config: PruningConfig with strategy and parameters
            metrics_collector: Optional MetricsCollector for metrics recording
            logger: Optional StructuredLogger for event logging
        """
        self.memory_interface = memory_interface
        self.pruning_config = pruning_config
        self.metrics_collector = metrics_collector
        self.logger = logger
    
    def prune(self, dry_run: bool = False) -> PruningResult:
        """
        Execute pruning strategy on all memories in the store.
        
        This method retrieves all memories, applies the configured pruning
        strategy to identify candidates for deletion, and deletes qualifying
        memories from the store. In dry_run mode, candidates are identified
        but no changes are persisted.
        
        Args:
            dry_run: If True, identify candidates without deleting
            
        Returns:
            PruningResult with deletion statistics
        """
        start_time = datetime.now(UTC)
        
        # Retrieve all memories
        result = self.memory_interface.retrieve()
        memories = result["memories"]
        
        # Filter out protected memories
        unprotected_memories = [
            m for m in memories
            if not m.get("metadata", {}).get("protected", False)
        ]
        
        # Sort by (importance, timestamp, id) for deterministic ordering
        sorted_memories = sorted(
            unprotected_memories,
            key=lambda m: (
                m.get("metadata", {}).get("importance", 1.0),
                m.get("timestamp", ""),
                m.get("id", ""),
            )
        )
        
        # Apply pruning strategy
        if self.pruning_config.strategy == PruningStrategy.THRESHOLD:
            to_delete = self._prune_threshold(sorted_memories)
        elif self.pruning_config.strategy == PruningStrategy.PERCENTILE:
            to_delete = self._prune_percentile(sorted_memories)
        elif self.pruning_config.strategy == PruningStrategy.CAPACITY:
            to_delete = self._prune_capacity(sorted_memories)
        else:
            # Default to threshold strategy
            to_delete = self._prune_threshold(sorted_memories)
        
        # Delete qualifying memories
        memories_deleted = 0
        deletion_failures = 0
        pruned_memories = []
        
        for memory in to_delete:
            memory_id = memory.get("id")
            importance = memory.get("metadata", {}).get("importance", 1.0)
            
            try:
                if not dry_run:
                    self.memory_interface.delete(memory_id)
                
                memories_deleted += 1
                pruned_memories.append(
                    PrunedMemory(
                        memory_id=memory_id,
                        importance_score=importance,
                        final_score=importance,
                        deletion_timestamp=start_time,
                        reason=self._get_pruning_reason(),
                    )
                )
                
            except Exception as e:
                deletion_failures += 1
                if self.logger:
                    self.logger.log(
                        event="memory_prune_deletion_error",
                        payload={
                            "memory_id": memory_id,
                            "error": str(e),
                        },
                    )
        
        # Calculate execution time
        execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        # Record metrics if collector available
        if self.metrics_collector:
            self.metrics_collector.increment("memory_prune.deleted", memories_deleted)
            self.metrics_collector.increment("memory_prune.failures", deletion_failures)
            self.metrics_collector.record_duration("memory_prune.duration_ms", execution_time_ms)
        
        # Log completion if logger available
        if self.logger:
            self.logger.log(
                event="memory_prune_completed",
                payload={
                    "memories_deleted": memories_deleted,
                    "deletion_failures": deletion_failures,
                    "strategy": self.pruning_config.strategy.value,
                    "execution_time_ms": execution_time_ms,
                    "dry_run": dry_run,
                },
            )
        
        return PruningResult(
            memories_deleted=memories_deleted,
            deletion_failures=deletion_failures,
            pruned_memories=pruned_memories,
            execution_time_ms=execution_time_ms,
        )
    
    def _prune_threshold(self, memories: list) -> list:
        """
        Identify memories for threshold-based pruning.
        
        This method identifies memories with importance scores below the
        configured threshold for deletion.
        
        Args:
            memories: List of memories sorted by (importance, timestamp, id)
            
        Returns:
            List of memories to delete (all with importance < threshold)
        """
        threshold = self.pruning_config.threshold
        return [m for m in memories if m.get("metadata", {}).get("importance", 1.0) < threshold]
    
    def _prune_percentile(self, memories: list) -> list:
        """
        Identify memories for percentile-based pruning.
        
        This method identifies the bottom N% of memories by importance
        for deletion, where N is the configured percentile.
        
        Args:
            memories: List of memories sorted by (importance, timestamp, id)
            
        Returns:
            List of memories to delete (bottom N% by importance)
        """
        percentile = self.pruning_config.percentile
        total_count = len(memories)
        
        if total_count == 0:
            return []
        
        # Calculate number of memories to delete
        delete_count = int(total_count * percentile / 100.0)
        delete_count = max(1, delete_count)  # At least one memory
        
        return memories[:delete_count]
    
    def _prune_capacity(self, memories: list) -> list:
        """
        Identify memories for capacity-based pruning.
        
        This method identifies the lowest-importance memories for deletion
        when the total count exceeds the configured capacity limit.
        
        Args:
            memories: List of memories sorted by (importance, timestamp, id)
            
        Returns:
            List of memories to delete (to bring count under capacity)
        """
        capacity_limit = self.pruning_config.capacity_limit
        total_count = len(memories)
        
        if total_count <= capacity_limit:
            return []
        
        # Delete enough memories to get under capacity
        delete_count = total_count - capacity_limit
        
        return memories[:delete_count]
    
    def _get_pruning_reason(self) -> str:
        """
        Get the pruning reason based on current strategy.
        
        Returns:
            String reason for pruning (e.g., "score", "percentile", "capacity")
        """
        if self.pruning_config.strategy == PruningStrategy.THRESHOLD:
            return "score"
        elif self.pruning_config.strategy == PruningStrategy.PERCENTILE:
            return "percentile"
        elif self.pruning_config.strategy == PruningStrategy.CAPACITY:
            return "capacity"
        else:
            return "score"
