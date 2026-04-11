"""
Lifecycle Manager Module.

This module provides the LifecycleManager orchestrator component that coordinates
all memory maintenance operations including decay, pruning, and deduplication.
The system executes operations in a fixed order (decay → pruning → deduplication)
with error isolation to prevent cascading failures.

Key Features:
- Sequential execution of lifecycle operations
- Error isolation (one failure doesn't stop others)
- Timeout enforcement to prevent runaway operations
- Dry run mode for testing without persistence
- Metrics and logging integration
- Comprehensive lifecycle reports
"""

from datetime import datetime, UTC
from typing import Optional

from luma.core.lifecycle.schemas import (
    LifecycleReport,
    MemoryDecayResult,
    PruningResult,
    DeduplicationResult,
)
from luma.core.lifecycle.memory_decay import MemoryDecay
from luma.core.lifecycle.memory_pruner import MemoryPruner
from luma.core.lifecycle.memory_deduplicator import MemoryDeduplicator
from luma.core.memory_interface import MemoryInterface

try:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
except ImportError:
    MetricsCollector = None
    StructuredLogger = None


class LifecycleManager:
    """
    Lifecycle manager orchestrator for memory maintenance operations.
    
    This component coordinates all memory lifecycle operations (decay, pruning,
    deduplication) in a deterministic sequence. It provides:
    
    - Sequential execution: decay → pruning → deduplication
    - Error isolation: failures in one operation don't stop others
    - Timeout enforcement: prevents runaway maintenance operations
    - Dry run mode: preview changes without persistence
    - Comprehensive reporting: unified results from all operations
    
    The component integrates with the existing Luma infrastructure including
    MemoryInterface for storage operations, MetricsCollector for observability,
    and StructuredLogger for event logging.
    
    Attributes:
        memory_decay: MemoryDecay component for time-based decay
        memory_pruner: MemoryPruner component for threshold-based removal
        memory_deduplicator: MemoryDeduplicator component for duplicate detection
        memory_interface: MemoryInterface for storage operations
        metrics_collector: Optional MetricsCollector for metrics recording
        logger: Optional StructuredLogger for event logging
        timeout_seconds: Maximum execution time in seconds
    """
    
    def __init__(
        self,
        memory_decay: MemoryDecay,
        memory_pruner: MemoryPruner,
        memory_deduplicator: MemoryDeduplicator,
        memory_interface: MemoryInterface,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
        timeout_seconds: int = 300,
    ):
        """
        Initialize the LifecycleManager orchestrator.
        
        Args:
            memory_decay: MemoryDecay component for time-based decay
            memory_pruner: MemoryPruner component for threshold-based removal
            memory_deduplicator: MemoryDeduplicator component for duplicate detection
            memory_interface: MemoryInterface for storage operations
            metrics_collector: Optional MetricsCollector for metrics recording
            logger: Optional StructuredLogger for event logging
            timeout_seconds: Maximum execution time in seconds (default: 300)
        """
        self.memory_decay = memory_decay
        self.memory_pruner = memory_pruner
        self.memory_deduplicator = memory_deduplicator
        self.memory_interface = memory_interface
        self.metrics_collector = metrics_collector
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        
        # Validate configuration parameters
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters at initialization."""
        # Validate timeout
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
    
    def run_maintenance(self, dry_run: bool = False) -> LifecycleReport:
        """
        Execute complete memory maintenance pipeline.
        
        This method executes all lifecycle operations in sequence:
        1. Memory decay (time-based importance reduction)
        2. Memory pruning (threshold-based removal)
        3. Memory deduplication (similarity-based merging)
        
        Each operation is wrapped in try-except to prevent cascading failures.
        The method enforces a timeout to prevent runaway operations.
        
        Args:
            dry_run: If True, simulate operations without persisting changes
            
        Returns:
            LifecycleReport with statistics from all operations
        """
        start_time = datetime.now(UTC)
        
        # Log start of maintenance cycle
        if self.logger:
            self.logger.log(
                event="maintenance_cycle_started",
                payload={
                    "start_timestamp": start_time.isoformat().replace('+00:00', 'Z'),
                    "dry_run": dry_run,
                },
            )
        
        # Execute operations with error isolation
        decay_result = self._execute_decay(dry_run, start_time)
        pruning_result = self._execute_pruning(dry_run, start_time)
        dedup_result = self._execute_deduplication(dry_run, start_time)
        
        # Calculate total execution time
        total_execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        # Log completion
        if self.logger:
            self.logger.log(
                event="maintenance_cycle_completed",
                payload={
                    "total_execution_time_ms": total_execution_time_ms,
                    "dry_run": dry_run,
                },
            )
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.increment("maintenance_cycle.count", 1)
            self.metrics_collector.record_duration("maintenance_cycle.duration_ms", total_execution_time_ms)
        
        return LifecycleReport(
            decay_result=decay_result,
            pruning_result=pruning_result,
            deduplication_result=dedup_result,
            total_execution_time_ms=total_execution_time_ms,
            maintenance_timestamp=start_time,
            dry_run=dry_run,
        )
    
    def _execute_decay(
        self,
        dry_run: bool,
        start_time: datetime
    ) -> MemoryDecayResult:
        """
        Execute memory decay operation with error isolation.
        
        Args:
            dry_run: If True, simulate without persisting
            start_time: Start time for timeout calculation
            
        Returns:
            MemoryDecayResult with decay statistics
        """
        try:
            # Check timeout
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > self.timeout_seconds:
                if self.logger:
                    self.logger.log(
                        event="maintenance_timeout",
                        payload={"operation": "decay", "elapsed_seconds": elapsed},
                    )
                return MemoryDecayResult(0, 0, 0.0, 0.0)
            
            return self.memory_decay.apply_decay(dry_run=dry_run)
            
        except Exception as e:
            if self.logger:
                self.logger.log(
                    event="decay_operation_failed",
                    payload={"error": str(e)},
                    level="error",
                )
            return MemoryDecayResult(0, 0, 0.0, 0.0)
    
    def _execute_pruning(
        self,
        dry_run: bool,
        start_time: datetime
    ) -> PruningResult:
        """
        Execute memory pruning operation with error isolation.
        
        Args:
            dry_run: If True, simulate without persisting
            start_time: Start time for timeout calculation
            
        Returns:
            PruningResult with pruning statistics
        """
        try:
            # Check timeout
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > self.timeout_seconds:
                if self.logger:
                    self.logger.log(
                        event="maintenance_timeout",
                        payload={"operation": "pruning", "elapsed_seconds": elapsed},
                    )
                return PruningResult(0, 0, [], 0.0)
            
            return self.memory_pruner.prune(dry_run=dry_run)
            
        except Exception as e:
            if self.logger:
                self.logger.log(
                    event="pruning_operation_failed",
                    payload={"error": str(e)},
                    level="error",
                )
            return PruningResult(0, 0, [], 0.0)
    
    def _execute_deduplication(
        self,
        dry_run: bool,
        start_time: datetime
    ) -> DeduplicationResult:
        """
        Execute memory deduplication operation with error isolation.
        
        Args:
            dry_run: If True, simulate without persisting
            start_time: Start time for timeout calculation
            
        Returns:
            DeduplicationResult with deduplication statistics
        """
        try:
            # Check timeout
            elapsed = (datetime.now(UTC) - start_time).total_seconds()
            if elapsed > self.timeout_seconds:
                if self.logger:
                    self.logger.log(
                        event="maintenance_timeout",
                        payload={"operation": "deduplication", "elapsed_seconds": elapsed},
                    )
                return DeduplicationResult(0, 0, [], None, 0.0)
            
            return self.memory_deduplicator.deduplicate(dry_run=dry_run)
            
        except Exception as e:
            if self.logger:
                self.logger.log(
                    event="deduplication_operation_failed",
                    payload={"error": str(e)},
                    level="error",
                )
            return DeduplicationResult(0, 0, [], None, 0.0)
