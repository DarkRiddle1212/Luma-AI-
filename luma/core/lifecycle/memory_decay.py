"""
Memory Decay Module.

This module provides the MemoryDecay component responsible for gradually
reducing memory importance scores based on age using configurable decay functions.
The system supports exponential, linear, and step decay functions for flexible
memory aging behavior.

Key Features:
- Time-based importance score reduction
- Multiple decay function types (exponential, linear, step)
- ISO 8601 timestamp parsing with timezone handling
- Dry run mode for testing without persistence
- Metrics and logging integration
"""

import math
from datetime import datetime, UTC
from typing import Optional

from luma.core.lifecycle.schemas import (
    MemoryDecayResult,
    DecayConfig,
    DecayFunctionType,
)
from luma.core.memory_interface import MemoryInterface

try:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
except ImportError:
    # Allow optional dependency injection
    MetricsCollector = None
    StructuredLogger = None


class MemoryDecay:
    """
    Memory decay component for time-based importance score reduction.
    
    This component gradually reduces memory importance scores based on their
    age using configurable decay functions. It supports exponential, linear,
    and step decay functions for flexible memory aging behavior.
    
    The component integrates with the existing Luma infrastructure including
    MemoryInterface for storage operations, MetricsCollector for observability,
    and StructuredLogger for event logging.
    
    Attributes:
        memory_interface: MemoryInterface instance for storage operations
        decay_config: DecayConfig with decay function parameters
        metrics_collector: Optional MetricsCollector for metrics recording
        logger: Optional StructuredLogger for event logging
    """
    
    def __init__(
        self,
        memory_interface: MemoryInterface,
        decay_config: DecayConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        logger: Optional[StructuredLogger] = None,
    ):
        """
        Initialize the MemoryDecay component.
        
        Args:
            memory_interface: MemoryInterface instance for storage operations
            decay_config: DecayConfig with decay function parameters
            metrics_collector: Optional MetricsCollector for metrics recording
            logger: Optional StructuredLogger for event logging
        """
        self.memory_interface = memory_interface
        self.decay_config = decay_config
        self.metrics_collector = metrics_collector
        self.logger = logger
    
    def apply_decay(self, dry_run: bool = False) -> MemoryDecayResult:
        """
        Apply decay function to all memories in the store.
        
        This method retrieves all memories, calculates their age, applies
        the configured decay function to their importance scores, and updates
        the memories in the store. In dry_run mode, calculations are performed
        but no changes are persisted.
        
        Args:
            dry_run: If True, calculate decay without persisting changes
            
        Returns:
            MemoryDecayResult with processing statistics
        """
        start_time = datetime.now(UTC)
        
        # Retrieve all memories
        result = self.memory_interface.retrieve()
        memories = result["memories"]
        
        memories_processed = 0
        memories_updated = 0
        total_decay_applied = 0.0
        
        for memory in memories:
            try:
                # Parse creation timestamp from memory metadata
                creation_timestamp = memory.get("metadata", {}).get("creation_timestamp")
                if not creation_timestamp:
                    # Skip memories without timestamp
                    continue
                
                # Calculate age in days
                age_days = self._calculate_age_days(creation_timestamp)
                
                # Get current importance score
                importance = memory.get("metadata", {}).get("importance", 1.0)
                
                # Skip memories with importance=0
                if importance == 0:
                    continue
                
                # Apply decay function
                new_importance = self._apply_decay_function(importance, age_days)
                
                # Update memory if importance changed
                if new_importance != importance:
                    memories_updated += 1
                    total_decay_applied += (importance - new_importance)
                    
                    if not dry_run:
                        # Update memory metadata with new importance
                        updated_metadata = memory.get("metadata", {}).copy()
                        updated_metadata["importance"] = new_importance
                        updated_metadata["last_decay_timestamp"] = start_time.isoformat().replace('+00:00', 'Z')
                        
                        # Update the memory in the store
                        self.memory_interface.store(
                            content=memory["content"],
                            metadata=updated_metadata
                        )
                
                memories_processed += 1
                
            except ValueError as e:
                # Log warning for unparseable timestamps and skip
                if self.logger:
                    self.logger.log(
                        event="memory_decay_timestamp_error",
                        payload={
                            "memory_id": memory.get("id", "unknown"),
                            "error": str(e),
                        },
                    )
                continue
        
        # Calculate average decay applied
        average_decay = (
            total_decay_applied / memories_updated
            if memories_updated > 0
            else 0.0
        )
        
        # Calculate execution time
        execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        # Record metrics if collector available
        if self.metrics_collector:
            self.metrics_collector.increment("memory_decay.processed", memories_processed)
            self.metrics_collector.increment("memory_decay.updated", memories_updated)
            self.metrics_collector.record_duration("memory_decay.duration_ms", execution_time_ms)
        
        # Log completion if logger available
        if self.logger:
            self.logger.log(
                event="memory_decay_completed",
                payload={
                    "memories_processed": memories_processed,
                    "memories_updated": memories_updated,
                    "average_decay_applied": average_decay,
                    "execution_time_ms": execution_time_ms,
                    "dry_run": dry_run,
                },
            )
        
        return MemoryDecayResult(
            memories_processed=memories_processed,
            memories_updated=memories_updated,
            average_decay_applied=average_decay,
            execution_time_ms=execution_time_ms,
        )
    
    def calculate_decay_factor(self, age_days: float) -> float:
        """
        Calculate decay multiplier for given age.
        
        This method calculates the decay factor (multiplier) that should be
        applied to importance scores based on the configured decay function
        and the memory's age in days.
        
        Args:
            age_days: Memory age in days (fractional)
            
        Returns:
            Decay factor in range [0, 1]
        """
        return self._apply_decay_function(1.0, age_days)
    
    def _calculate_age_days(self, creation_timestamp: str) -> float:
        """
        Calculate memory age in days from creation timestamp.
        
        This method parses ISO 8601 formatted timestamps and calculates
        the age in days. It handles both timezone-aware and timezone-naive
        timestamps by converting to UTC.
        
        Args:
            creation_timestamp: ISO 8601 formatted timestamp string
            
        Returns:
            Age in days (fractional)
            
        Raises:
            ValueError: If timestamp cannot be parsed
        """
        try:
            # Parse ISO 8601 timestamp
            timestamp = datetime.fromisoformat(creation_timestamp.replace('Z', '+00:00'))
            
            # Convert to UTC if timezone-aware
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(UTC)
            else:
                # Treat naive timestamps as UTC
                timestamp = timestamp.replace(tzinfo=UTC)
            
            # Calculate age in days
            now = datetime.now(UTC)
            age_delta = now - timestamp
            age_days = age_delta.total_seconds() / (24 * 60 * 60)
            
            # Treat future timestamps as age=0
            return max(0.0, age_days)
            
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Failed to parse timestamp: {creation_timestamp}") from e
    
    def _apply_decay_function(self, importance: float, age_days: float) -> float:
        """
        Apply configured decay function to importance score.
        
        This method selects and applies the appropriate decay function
        based on the configured decay_function_type.
        
        Args:
            importance: Current importance score [0, 1]
            age_days: Memory age in days
            
        Returns:
            New importance score after decay
        """
        decay_function_type = self.decay_config.decay_function_type
        
        if decay_function_type == DecayFunctionType.EXPONENTIAL:
            return self._apply_exponential_decay(importance, age_days)
        elif decay_function_type == DecayFunctionType.LINEAR:
            return self._apply_linear_decay(importance, age_days)
        elif decay_function_type == DecayFunctionType.STEP:
            return self._apply_step_decay(importance, age_days)
        else:
            # Default to exponential decay
            return self._apply_exponential_decay(importance, age_days)
    
    def _apply_exponential_decay(self, importance: float, age_days: float) -> float:
        """
        Apply exponential decay to importance score.
        
        Exponential decay uses the formula:
            importance_new = importance_old * e^(-decay_rate * age_days)
        
        This provides smooth, continuous decay that is rapid initially
        and slows over time.
        
        Args:
            importance: Current importance score [0, 1]
            age_days: Memory age in days
            
        Returns:
            New importance score after exponential decay
        """
        decay_rate = self.decay_config.decay_rate
        new_importance = importance * math.exp(-decay_rate * age_days)
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, new_importance))
    
    def _apply_linear_decay(self, importance: float, age_days: float) -> float:
        """
        Apply linear decay to importance score.
        
        Linear decay uses the formula:
            importance_new = max(0, importance_old - decay_rate * age_days)
        
        This provides constant decay rate regardless of current importance.
        
        Args:
            importance: Current importance score [0, 1]
            age_days: Memory age in days
            
        Returns:
            New importance score after linear decay
        """
        decay_rate = self.decay_config.decay_rate
        new_importance = importance - decay_rate * age_days
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, new_importance))
    
    def _apply_step_decay(self, importance: float, age_days: float) -> float:
        """
        Apply step decay to importance score.
        
        Step decay reduces importance by a fixed percentage at configured
        age intervals:
            importance_new = importance_old * (1 - step_percentage)^(age_days / step_interval)
        
        This provides discrete decay events at regular intervals.
        
        Args:
            importance: Current importance score [0, 1]
            age_days: Memory age in days
            
        Returns:
            New importance score after step decay
        """
        decay_rate = self.decay_config.decay_rate
        step_interval_days = self.decay_config.step_interval_days
        step_percentage = self.decay_config.step_percentage
        
        # Calculate number of steps
        num_steps = age_days / step_interval_days
        
        # Apply step decay
        new_importance = importance * ((1 - step_percentage) ** num_steps)
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, new_importance))
