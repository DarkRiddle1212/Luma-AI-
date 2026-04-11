"""
Memory Lifecycle Manager - Production-grade memory retention and pruning.

This module provides deterministic memory lifecycle management through
age-based expiration, score-based filtering, and hard cap enforcement.
All operations respect importance-based protection and maintain clean
architecture by operating exclusively through the MemoryInterface abstraction.

The component is designed to be:
- Deterministic: Identical inputs produce identical outputs
- Idempotent: Running cleanup multiple times produces same final state
- Resilient: Errors are logged but don't crash the system
- Scalable: Handles 100k+ entries with O(n log n) complexity

Example:
    >>> from luma.core.lifecycle_config import LifecycleConfig
    >>> from luma.core.lifecycle_manager import MemoryLifecycleManager
    >>> from luma.core.memory_interface import MemoryInterface
    >>> 
    >>> config = LifecycleConfig(
    ...     max_total_memories=10000,
    ...     max_age_days=90,
    ...     pruning_score_threshold=0.3,
    ...     min_importance_protected=0.8
    ... )
    >>> manager = MemoryLifecycleManager(config, memory_interface)
    >>> stats = manager.cleanup()
    >>> print(f"Deleted {stats['total_deleted']} memories")
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import logging
import time

from luma.core.lifecycle_config import LifecycleConfig, ConfigValidator
from luma.core.memory_interface import MemoryInterface, MemoryEntry
from luma.core.lifecycle_utils import extract_importance, extract_final_score, extract_namespace
from luma.core.cleanup_result import CleanupResult, CleanupStatus
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger

_logger = logging.getLogger(__name__)


class MemoryLifecycleManager:
    """
    Manages memory lifecycle through deterministic pruning policies.
    
    Enforces retention rules via three-phase pruning:
    1. Age-based pruning (if max_age_days configured)
    2. Score-based pruning (if pruning_score_threshold configured)
    3. Hard cap enforcement (always applied)
    
    All operations are:
    - Deterministic: Identical inputs produce identical outputs
    - Idempotent: Running cleanup multiple times produces same final state
    - Resilient: Errors are logged but don't crash the system
    
    The manager operates exclusively through the MemoryInterface abstraction,
    maintaining clean architecture principles and enabling testability.
    
    Thread Safety:
        This class is thread-safe when used with a thread-safe MemoryInterface
        implementation. All operations are stateless except for logging.
    
    Example:
        >>> config = LifecycleConfig(
        ...     max_total_memories=10000,
        ...     max_age_days=90,
        ...     pruning_score_threshold=0.3,
        ...     min_importance_protected=0.8
        ... )
        >>> manager = MemoryLifecycleManager(config, memory_interface)
        >>> stats = manager.cleanup()
        >>> print(f"Deleted {stats['total_deleted']} memories")
    """
    
    def __init__(
        self,
        config: LifecycleConfig,
        memory_interface: MemoryInterface,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize the Memory Lifecycle Manager.
        
        Args:
            config: Validated lifecycle configuration. Validation occurs
                   automatically in LifecycleConfig.__post_init__.
            memory_interface: Interface for memory operations (retrieve, delete).
                            Must implement MemoryInterface protocol.
            metrics_collector: Optional metrics collector for observability.
                             If provided, cleanup operations will be instrumented.
            logger: Optional structured logger for observability.
                   If provided, cleanup events will be logged.
        
        Raises:
            ValueError: If config validation fails
        
        Example:
            >>> config = LifecycleConfig(max_total_memories=10000)
            >>> manager = MemoryLifecycleManager(config, memory_interface)
            >>> # With observability
            >>> manager = MemoryLifecycleManager(
            ...     config, memory_interface,
            ...     metrics_collector=collector, logger=structured_logger
            ... )
        """
        # Config validation happens in LifecycleConfig.__post_init__
        self.config = config
        self.memory = memory_interface
        self.metrics_collector = metrics_collector
        self.logger = logger
        
        # Initialize pruning components
        self.age_pruner = AgePruner(config, memory_interface)
        self.score_pruner = ScorePruner(config, memory_interface)
        self.namespace_cap_enforcer = NamespaceCapEnforcer(config, memory_interface)
        self.cap_enforcer = HardCapEnforcer(config, memory_interface)
        
        _logger.info(
            f"MemoryLifecycleManager initialized: "
            f"max_total={config.max_total_memories}, "
            f"max_age_days={config.max_age_days}, "
            f"score_threshold={config.pruning_score_threshold}, "
            f"min_importance={config.min_importance_protected}"
        )
    
    def cleanup(self) -> CleanupResult:
        """
        Execute the complete pruning pipeline.
        
        Runs all four pruning phases in deterministic order:
        1. Age-based pruning (if max_age_days configured)
        2. Score-based pruning (if pruning_score_threshold configured)
        3. Namespace cap enforcement (if max_memories_per_namespace configured)
        4. Global hard cap enforcement (always runs)
        
        Returns:
            CleanupResult with cleanup statistics:
            - age_pruned: Count of memories deleted by age pruning
            - score_pruned: Count of memories deleted by score pruning
            - cap_pruned: Count of memories deleted by hard cap enforcement
            - total_deleted: Total memories deleted
            - failed_deletions: Count of deletion errors encountered
            - final_count: Total memories remaining after cleanup
            - status: SUCCESS (no errors), PARTIAL (some errors), or FAILED (all errors)
        
        Note:
            This method is idempotent. Running it multiple times without
            intervening changes produces the same final state.
            
            This method never propagates exceptions to callers. All errors
            are logged and reflected in the returned CleanupResult.
        
        Example:
            >>> result = manager.cleanup()
            >>> print(f"Deleted {result.total_deleted} memories")
            >>> print(f"Status: {result.status.value}")
            >>> print(f"Final count: {result.final_count}")
        """
        # Increment cleanup_runs counter at start
        if self.metrics_collector is not None:
            self.metrics_collector.increment("cleanup_runs")
        
        # Start timing measurement
        start_time = time.perf_counter()
        
        _logger.info("Starting memory lifecycle cleanup")
        
        # Log cleanup start event
        if self.logger is not None:
            self.logger.log("cleanup_started", {})
        
        age_pruned = 0
        score_pruned = 0
        namespace_cap_pruned = 0
        cap_pruned = 0
        failed_deletions = 0
        
        try:
            # Phase 1: Age-based pruning
            if self.config.max_age_days is not None:
                age_deleted, age_failed = self.age_pruner.prune()
                age_pruned = age_deleted
                failed_deletions += age_failed
                _logger.info(f"Age-based pruning: deleted {age_pruned} memories")
            
            # Phase 2: Score-based pruning
            if self.config.pruning_score_threshold is not None:
                score_deleted, score_failed = self.score_pruner.prune()
                score_pruned = score_deleted
                failed_deletions += score_failed
                _logger.info(f"Score-based pruning: deleted {score_pruned} memories")
            
            # Phase 3: Namespace cap enforcement (if configured)
            if self.config.max_memories_per_namespace is not None:
                namespace_deleted, namespace_failed = self.namespace_cap_enforcer.enforce()
                namespace_cap_pruned = namespace_deleted
                failed_deletions += namespace_failed
                _logger.info(f"Namespace cap enforcement: deleted {namespace_cap_pruned} memories")
            
            # Phase 4: Global hard cap enforcement (always runs)
            cap_deleted, cap_failed = self.cap_enforcer.enforce()
            cap_pruned = cap_deleted
            failed_deletions += cap_failed
            _logger.info(f"Hard cap enforcement: deleted {cap_pruned} memories")
            
        except Exception as e:
            # Catch any unexpected errors and log them
            _logger.error(f"Cleanup failed with unexpected error: {e}", exc_info=True)
            
            # Increment cleanup_failures on exceptions
            if self.metrics_collector is not None:
                self.metrics_collector.increment("cleanup_failures")
            
            # Log cleanup failure event
            if self.logger is not None:
                self.logger.log("cleanup_failed", {"error": str(e)})
            
            # Don't propagate the exception - return partial results
        
        # Calculate totals
        total_deleted = age_pruned + score_pruned + namespace_cap_pruned + cap_pruned
        
        # Increment memories_deleted_total by deletion count
        if self.metrics_collector is not None:
            self.metrics_collector.increment("memories_deleted_total", total_deleted)
        
        # Get final count
        final_count = self._count_total_memories()
        
        # Determine status based on errors
        if failed_deletions == 0:
            status = CleanupStatus.SUCCESS
        elif total_deleted > 0:
            status = CleanupStatus.PARTIAL
        else:
            status = CleanupStatus.FAILED
        
        # Record cleanup duration
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        if self.metrics_collector is not None:
            self.metrics_collector.record_duration("cleanup_duration_ms", duration_ms)
        
        _logger.info(
            f"Cleanup completed: total_deleted={total_deleted}, "
            f"failed_deletions={failed_deletions}, "
            f"final_count={final_count}, "
            f"status={status.value}"
        )
        
        # Log cleanup completion event
        if self.logger is not None:
            self.logger.log("cleanup_completed", {
                "total_deleted": total_deleted,
                "failed_deletions": failed_deletions,
                "final_count": final_count,
                "status": status.value,
                "duration_ms": duration_ms
            })
        
        return CleanupResult(
            age_pruned=age_pruned,
            score_pruned=score_pruned,
            cap_pruned=cap_pruned,
            total_deleted=total_deleted,
            failed_deletions=failed_deletions,
            final_count=final_count,
            status=status
        )
    
    def _count_total_memories(self) -> int:
        """
        Count total memories across all namespaces.
        
        Returns:
            Total count of memories
        
        Complexity:
            O(1) if MemoryInterface provides count, O(n) otherwise
        """
        try:
            # Retrieve all memories with minimal data
            result = self.memory.retrieve(params={"limit": 1000000})
            return result["total_count"]
        except Exception as e:
            _logger.error(f"Failed to count memories: {e}", exc_info=True)
            return 0



class AgePruner:
    """Handles age-based memory pruning."""
    
    def __init__(self, config: LifecycleConfig, memory_interface: MemoryInterface):
        """
        Initialize the Age Pruner.
        
        Args:
            config: Lifecycle configuration with max_age_days setting
            memory_interface: Interface for memory operations
        """
        self.config = config
        self.memory = memory_interface
    
    def prune(self) -> tuple[int, int]:
            """
            Prune memories based on age threshold.

            Retrieves all memories, calculates age for each, and deletes memories
            that exceed max_age_days and are not protected by importance threshold.

            Returns:
                Tuple of (deleted_count, failed_count) where:
                - deleted_count: Count of successfully deleted memories
                - failed_count: Count of deletion failures

            Complexity:
                O(n) where n is the number of memories

            Note:
                Errors during deletion are logged but don't stop processing.
                Protected memories (importance >= min_importance_protected) are
                never deleted regardless of age.
            """
            if self.config.max_age_days is None:
                return (0, 0)

            current_time = datetime.now(UTC)
            to_delete = []
            protected_skipped = 0

            try:
                # Retrieve all memories - O(n)
                result = self.memory.retrieve(params={"limit": 1000000})
                memories = result["memories"]

                # Filter by age - O(n)
                for memory in memories:
                    age_days = self._calculate_age_days(memory["timestamp"], current_time)
                    importance = extract_importance(memory)

                    # Check pruning criteria
                    if age_days > self.config.max_age_days:
                        if importance < self.config.min_importance_protected:
                            to_delete.append(memory["id"])
                            _logger.debug(
                                f"Age pruning candidate: id={memory['id']}, "
                                f"age_days={age_days}, importance={importance}"
                            )
                        else:
                            # Memory is old but protected by importance
                            protected_skipped += 1

                # Delete memories - O(k) where k is number to delete
                deleted_count, failed_count = self._delete_memories(to_delete)
                return (deleted_count, failed_count)

            except Exception as e:
                _logger.error(f"Age pruning failed: {e}", exc_info=True)
                return (0, 0)

    
    def _calculate_age_days(self, timestamp_str: str, current_time: datetime) -> int:
        """
        Calculate age in days from ISO 8601 timestamp.
        
        Args:
            timestamp_str: ISO 8601 formatted timestamp
            current_time: Current time for comparison
        
        Returns:
            Age in days (integer)
        """
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        age = current_time - timestamp
        return age.days
    
    def _delete_memories(self, memory_ids: List[str]) -> tuple[int, int]:
        """
        Delete memories by ID, handling errors gracefully.

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
        """
        deleted_count = 0
        failed_count = 0
        for memory_id in memory_ids:
            try:
                self.memory.delete(memory_id)
                deleted_count += 1
            except Exception as e:
                _logger.error(
                    f"Failed to delete memory {memory_id}: {e}",
                    exc_info=True
                )
                failed_count += 1
        return (deleted_count, failed_count)




class ScorePruner:
    """Handles score-based memory pruning."""
    
    def __init__(self, config: LifecycleConfig, memory_interface: MemoryInterface):
        """
        Initialize the Score Pruner.
        
        Args:
            config: Lifecycle configuration with pruning_score_threshold setting
            memory_interface: Interface for memory operations
        """
        self.config = config
        self.memory = memory_interface
    
    def prune(self) -> tuple[int, int]:
        """
        Prune memories based on score threshold.
        
        Retrieves all memories, evaluates final_score for each, and deletes
        memories below pruning_score_threshold that are not protected by
        importance threshold.
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
        
        Complexity:
            O(n) where n is the number of memories
        
        Note:
            Errors during deletion are logged but don't stop processing.
            Protected memories (importance >= min_importance_protected) are
            never deleted regardless of score.
        """
        if self.config.pruning_score_threshold is None:
            return (0, 0)
        
        to_delete = []
        protected_skipped = 0
        
        try:
            # Retrieve all memories - O(n)
            result = self.memory.retrieve(params={"limit": 1000000})
            memories = result["memories"]
            
            # Filter by score - O(n)
            for memory in memories:
                final_score = extract_final_score(memory)
                importance = extract_importance(memory)
                
                # Check pruning criteria
                if final_score < self.config.pruning_score_threshold:
                    if importance < self.config.min_importance_protected:
                        to_delete.append(memory["id"])
                        _logger.debug(
                            f"Score pruning candidate: id={memory['id']}, "
                            f"final_score={final_score}, importance={importance}"
                        )
                    else:
                        # Memory has low score but protected by importance
                        protected_skipped += 1
            
            # Delete memories - O(k) where k is number to delete
            deleted_count, failed_count = self._delete_memories(to_delete)
            return (deleted_count, failed_count)
            
        except Exception as e:
            _logger.error(f"Score pruning failed: {e}", exc_info=True)
            return (0, 0)
    
    def _delete_memories(self, memory_ids: List[str]) -> tuple[int, int]:
        """
        Delete memories by ID, handling errors gracefully.
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
        """
        deleted_count = 0
        failed_count = 0
        for memory_id in memory_ids:
            try:
                self.memory.delete(memory_id)
                deleted_count += 1
            except Exception as e:
                _logger.error(
                    f"Failed to delete memory {memory_id}: {e}",
                    exc_info=True
                )
                failed_count += 1
        return (deleted_count, failed_count)





class NamespaceCapEnforcer:
    """Enforces per-namespace memory caps."""
    
    def __init__(self, config: LifecycleConfig, memory_interface: MemoryInterface):
        """
        Initialize the Namespace Cap Enforcer.
        
        Args:
            config: Lifecycle configuration with max_memories_per_namespace setting
            memory_interface: Interface for memory operations
        """
        self.config = config
        self.memory = memory_interface
    
    def enforce(self) -> tuple[int, int]:
        """
        Enforce per-namespace memory caps.
        
        Groups memories by namespace and enforces max_memories_per_namespace
        limit independently for each namespace. Uses deterministic sorting
        to ensure predictable deletion order within each namespace.
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories across all namespaces
            - failed_count: Count of deletion failures
        
        Complexity:
            O(n log n) where n is the number of memories (due to sorting per namespace)
        
        Note:
            Errors during deletion are logged but don't stop processing.
            Protected memories (importance >= min_importance_protected) are
            never deleted regardless of namespace count.
        """
        if self.config.max_memories_per_namespace is None:
            return (0, 0)
        
        try:
            # Retrieve all memories - O(n)
            result = self.memory.retrieve(params={"limit": 1000000})
            memories = result["memories"]
            
            # Group memories by namespace - O(n)
            namespace_groups: Dict[str, List[MemoryEntry]] = {}
            for memory in memories:
                namespace = extract_namespace(memory)
                if namespace not in namespace_groups:
                    namespace_groups[namespace] = []
                namespace_groups[namespace].append(memory)
            
            total_deleted = 0
            total_failed = 0
            total_protected_skipped = 0
            
            # Process each namespace independently - O(k * n_i log n_i) where k is number of namespaces
            for namespace, namespace_memories in namespace_groups.items():
                namespace_count = len(namespace_memories)
                
                # Check if namespace cap is exceeded
                if namespace_count <= self.config.max_memories_per_namespace:
                    _logger.debug(
                        f"Namespace '{namespace}' cap not exceeded: "
                        f"{namespace_count} <= {self.config.max_memories_per_namespace}"
                    )
                    continue
                
                excess = namespace_count - self.config.max_memories_per_namespace
                _logger.info(
                    f"Namespace '{namespace}' cap exceeded: "
                    f"{namespace_count} > {self.config.max_memories_per_namespace}, "
                    f"need to delete {excess} memories"
                )
                
                # Filter protected memories - O(n_i)
                unprotected = []
                protected_count = 0
                for memory in namespace_memories:
                    importance = extract_importance(memory)
                    if importance < self.config.min_importance_protected:
                        unprotected.append(memory)
                    else:
                        protected_count += 1
                
                # Check if we have enough unprotected memories to delete
                if len(unprotected) < excess:
                    _logger.warning(
                        f"Cannot enforce namespace cap for '{namespace}': "
                        f"only {len(unprotected)} unprotected memories available, "
                        f"need to delete {excess}"
                    )
                    # Count how many protected memories prevented full enforcement
                    total_protected_skipped += (excess - len(unprotected))
                    excess = len(unprotected)
                
                # Sort by deterministic criteria - O(n_i log n_i)
                sorted_memories = self._sort_for_deletion(unprotected)
                
                # Select memories to delete
                to_delete = [m["id"] for m in sorted_memories[:excess]]
                
                # Delete excess memories - O(excess)
                deleted_count, failed_count = self._delete_memories(to_delete)
                total_deleted += deleted_count
                total_failed += failed_count
                
                _logger.info(
                    f"Namespace '{namespace}' cap enforcement: deleted {deleted_count} memories"
                )
            
            return (total_deleted, total_failed)
            
        except Exception as e:
            _logger.error(f"Namespace cap enforcement failed: {e}", exc_info=True)
            return (0, 0)
    
    def _sort_for_deletion(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Sort memories for deletion using deterministic criteria.
        
        Sort order (ascending):
        1. final_score (lowest first)
        2. timestamp (oldest first)
        3. memory_id (lexicographical)
        
        Args:
            memories: List of memory entries to sort
        
        Returns:
            Sorted list (lowest priority first)
        
        Complexity:
            O(n log n) using Python's Timsort (stable sort algorithm)
        """
        def sort_key(memory: MemoryEntry):
            final_score = extract_final_score(memory)
            timestamp_str = memory["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            memory_id = memory["id"]
            
            return (
                final_score,              # Ascending: lowest scores first
                timestamp.timestamp(),    # Ascending: oldest first
                memory_id                 # Ascending: lexicographical
            )
        
        # Deterministic sort - O(n log n) using Timsort
        return sorted(memories, key=sort_key)
    
    def _delete_memories(self, memory_ids: List[str]) -> tuple[int, int]:
        """
        Delete memories by ID, handling errors gracefully.
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
        """
        deleted_count = 0
        failed_count = 0
        for memory_id in memory_ids:
            try:
                self.memory.delete(memory_id)
                deleted_count += 1
            except Exception as e:
                _logger.error(
                    f"Failed to delete memory {memory_id}: {e}",
                    exc_info=True
                )
                failed_count += 1
        return (deleted_count, failed_count)


class HardCapEnforcer:
    """Enforces hard cap on total memory count."""
    
    def __init__(self, config: LifecycleConfig, memory_interface: MemoryInterface):
        """
        Initialize the Hard Cap Enforcer.
        
        Args:
            config: Lifecycle configuration with max_total_memories setting
            memory_interface: Interface for memory operations
        """
        self.config = config
        self.memory = memory_interface
    
    def enforce(self) -> tuple[int, int]:
        """
        Enforce hard cap on total memories.
        
        Counts total memories and deletes lowest-ranked unprotected memories
        if the count exceeds max_total_memories. Uses deterministic sorting
        to ensure predictable deletion order.
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failuresipped) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
            - protected_skipped: Count of memories skipped due to protection
        
        Complexity:
            O(n log n) where n is the number of memories (due to sorting)
        
        Note:
            Errors during deletion are logged but don't stop processing.
            Protected memories (importance >= min_importance_protected) are
            never deleted regardless of total count.
        """
        try:
            # Count total memories - O(1) or O(n)
            result = self.memory.retrieve(params={"limit": 1000000})
            memories = result["memories"]
            total_count = len(memories)
            
            # Check if cap is exceeded
            if total_count <= self.config.max_total_memories:
                _logger.debug(
                    f"Hard cap not exceeded: {total_count} <= {self.config.max_total_memories}"
                )
                return (0, 0)
            
            excess = total_count - self.config.max_total_memories
            _logger.info(
                f"Hard cap exceeded: {total_count} > {self.config.max_total_memories}, "
                f"need to delete {excess} memories"
            )
            
            # Filter protected memories - O(n)
            unprotected = []
            protected_count = 0
            for memory in memories:
                importance = extract_importance(memory)
                if importance < self.config.min_importance_protected:
                    unprotected.append(memory)
                else:
                    protected_count += 1
            
            # Check if we have enough unprotected memories to delete
            protected_skipped = 0
            if len(unprotected) < excess:
                _logger.warning(
                    f"Cannot enforce hard cap: only {len(unprotected)} unprotected memories "
                    f"available, need to delete {excess}"
                )
                # Count how many protected memories prevented full enforcement
                protected_skipped = excess - len(unprotected)
                excess = len(unprotected)
            
            # Sort remaining memories - O(n log n)
            sorted_memories = self._sort_for_deletion(unprotected)
            
            # Select memories to delete
            to_delete = [m["id"] for m in sorted_memories[:excess]]
            
            # Delete excess memories - O(k) where k is number to delete
            deleted_count, failed_count = self._delete_memories(to_delete)
            return (deleted_count, failed_count)
            
        except Exception as e:
            _logger.error(f"Hard cap enforcement failed: {e}", exc_info=True)
            return (0, 0)
    
    def _sort_for_deletion(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Sort memories for deletion using deterministic criteria.
        
        Sort order (ascending):
        1. final_score (lowest first)
        2. timestamp (oldest first)
        3. memory_id (lexicographical)
        
        Args:
            memories: List of memory entries to sort
        
        Returns:
            Sorted list (lowest priority first)
        
        Complexity:
            O(n log n) using Python's Timsort (stable sort algorithm)
        """
        def sort_key(memory: MemoryEntry):
            final_score = extract_final_score(memory)
            timestamp_str = memory["timestamp"]
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            memory_id = memory["id"]
            
            return (
                final_score,              # Ascending: lowest scores first
                timestamp.timestamp(),    # Ascending: oldest first
                memory_id                 # Ascending: lexicographical
            )
        
        # Deterministic sort - O(n log n) using Timsort
        return sorted(memories, key=sort_key)
    
    def _delete_memories(self, memory_ids: List[str]) -> tuple[int, int]:
        """
        Delete memories by ID, handling errors gracefully.
        
        Args:
            memory_ids: List of memory IDs to delete
        
        Returns:
            Tuple of (deleted_count, failed_count) where:
            - deleted_count: Count of successfully deleted memories
            - failed_count: Count of deletion failures
        """
        deleted_count = 0
        failed_count = 0
        for memory_id in memory_ids:
            try:
                self.memory.delete(memory_id)
                deleted_count += 1
            except Exception as e:
                _logger.error(
                    f"Failed to delete memory {memory_id}: {e}",
                    exc_info=True
                )
                failed_count += 1
        return (deleted_count, failed_count)


# Re-export LifecycleConfig and ConfigValidator for backward compatibility
__all__ = [
    'LifecycleConfig',
    'ConfigValidator',
    'MemoryLifecycleManager',
    'AgePruner',
    'ScorePruner',
    'NamespaceCapEnforcer',
    'HardCapEnforcer',
    'extract_importance',
    'extract_final_score',
    'extract_namespace'
]

