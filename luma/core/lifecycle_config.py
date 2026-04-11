"""
Lifecycle Configuration Module.

This module provides the LifecycleConfig dataclass for configuring memory
lifecycle management policies including retention rules, pruning thresholds,
and importance-based protection.

Example:
    >>> from luma.core.lifecycle_config import LifecycleConfig
    >>> 
    >>> config = LifecycleConfig(
    ...     max_total_memories=10000,
    ...     max_age_days=90,
    ...     pruning_score_threshold=0.3,
    ...     min_importance_protected=0.8
    ... )
"""

from dataclasses import dataclass
from typing import Optional


class ConfigValidator:
    """
    Validates LifecycleConfig parameters.
    
    This class provides static validation methods for LifecycleConfig.
    Validation is automatically called in LifecycleConfig.__post_init__.
    """
    
    @staticmethod
    def validate(config: 'LifecycleConfig') -> None:
        """
        Validate all configuration parameters.
        
        Checks all configuration values against their constraints and raises
        ValueError with a descriptive message if any validation fails.
        
        Args:
            config: Configuration to validate
        
        Raises:
            ValueError: With descriptive message identifying invalid parameter
        
        Example:
            >>> config = LifecycleConfig(max_total_memories=10000)
            >>> ConfigValidator.validate(config)  # Passes
            
            >>> config = LifecycleConfig(max_total_memories=-1)
            >>> ConfigValidator.validate(config)  # Raises ValueError
        """
        # Validate max_total_memories > 0
        if config.max_total_memories <= 0:
            raise ValueError("max_total_memories must be greater than 0")
        
        # Validate max_memories_per_namespace must be greater than 0
        if config.max_memories_per_namespace is not None:
            if config.max_memories_per_namespace <= 0:
                raise ValueError("max_memories_per_namespace must be greater than 0")
        
        # Validate max_age_days must be greater than 0
        if config.max_age_days is not None:
            if config.max_age_days <= 0:
                raise ValueError("max_age_days must be greater than 0")
        
        # Validate pruning_score_threshold in [0, 1] if provided
        if config.pruning_score_threshold is not None:
            if not 0.0 <= config.pruning_score_threshold <= 1.0:
                raise ValueError("pruning_score_threshold must be between 0 and 1")
        
        # Validate min_importance_protected in [0, 1]
        if not 0.0 <= config.min_importance_protected <= 1.0:
            raise ValueError("min_importance_protected must be between 0 and 1")


@dataclass
class LifecycleConfig:
    """
    Configuration for memory lifecycle management.
    
    Defines retention policies, pruning thresholds, and protection rules
    for memory lifecycle operations. All parameters are validated on
    initialization to ensure correctness.
    
    Attributes:
        max_total_memories: Maximum total memories across all namespaces.
                          Must be greater than 0. This is the hard cap that
                          will never be exceeded after cleanup.
        
        max_memories_per_namespace: Maximum memories per namespace (optional).
                                   Must be greater than 0 if provided.
                                   Currently not enforced but reserved for
                                   future namespace-level limits.
        
        max_age_days: Maximum age in days before pruning (optional).
                     Must be greater than 0 if provided. Memories older
                     than this threshold will be deleted unless protected
                     by importance score.
        
        pruning_score_threshold: Minimum final_score to retain (optional).
                                Must be in range [0, 1] if provided.
                                Memories with final_score below this threshold
                                will be deleted unless protected by importance.
        
        min_importance_protected: Importance threshold for protection (required).
                                 Must be in range [0, 1]. Memories with
                                 importance >= this value are never deleted
                                 by any pruning operation. Defaults to 0.8.
    
    Validation Rules:
        - max_total_memories must be > 0
        - max_memories_per_namespace must be > 0 if provided
        - max_age_days must be > 0 if provided
        - pruning_score_threshold must be in [0, 1] if provided
        - min_importance_protected must be in [0, 1]
    
    Raises:
        ValueError: If any validation rule is violated, with a descriptive
                   message identifying the invalid parameter.
    
    Example:
        >>> # Valid configuration
        >>> config = LifecycleConfig(
        ...     max_total_memories=10000,
        ...     max_age_days=90,
        ...     pruning_score_threshold=0.3,
        ...     min_importance_protected=0.8
        ... )
        
        >>> # Invalid configuration - raises ValueError
        >>> try:
        ...     config = LifecycleConfig(max_total_memories=0)
        ... except ValueError as e:
        ...     print(e)  # "max_total_memories must be greater than 0"
    """
    max_total_memories: int
    max_memories_per_namespace: Optional[int] = None
    max_age_days: Optional[int] = None
    pruning_score_threshold: Optional[float] = None
    min_importance_protected: float = 0.8
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Raises:
            ValueError: If any validation rule is violated
        """
        ConfigValidator.validate(self)
