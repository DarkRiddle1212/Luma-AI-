"""
Core Module

Provides reasoning, scheduling, and memory lifecycle management capabilities for the Luma system.
"""

from luma.core.reasoning import ReasoningEngine  # Imports from reasoning.py (old module)
from luma.core.scheduler import TaskScheduler, Task
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger

# Import lifecycle management components
from luma.core.lifecycle import (
    LifecycleManager,
    MemoryDecay,
    MemoryPruner,
    MemoryDeduplicator,
    DecayConfig,
    PruningConfig,
    DeduplicationConfig,
    DecayFunctionType,
    PruningStrategy,
    SimilarityMetric,
)

__all__ = [
    "ReasoningEngine",
    "TaskScheduler",
    "Task",
    "MetricsCollector",
    "StructuredLogger",
    # Lifecycle management
    "LifecycleManager",
    "MemoryDecay",
    "MemoryPruner",
    "MemoryDeduplicator",
    "DecayConfig",
    "PruningConfig",
    "DeduplicationConfig",
    "DecayFunctionType",
    "PruningStrategy",
    "SimilarityMetric",
]
