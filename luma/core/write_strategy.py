"""
Memory Write Strategy Module

This module provides configuration and logic for intelligent memory persistence,
determining when and how user messages should be stored as memories.

The write strategy evaluates write triggers, validates content, handles deduplication
and conflicts, and coordinates with session management for buffered storage.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
import os
import json
from pathlib import Path
import logging

from luma.core.memory_interface import MemoryStorageError

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Structures
# ============================================================================


@dataclass
class WriteDecision:
    """
    Result of write trigger evaluation.
    
    Represents whether a message should be stored as a memory and why.
    
    Attributes:
        should_write: Whether the message should be stored
        reason: Reason for the decision (e.g., "approved", "trivial", "duplicate", "repetitive")
        metadata: Additional context about the decision (e.g., duplicate_id, similarity_score)
    """
    should_write: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteStrategyConfig:
    """
    Configuration for write strategy behavior.
    
    Controls when and how memories are persisted, including trigger patterns,
    validation rules, and deduplication settings.
    
    Attributes:
        trivial_patterns: List of patterns to reject (e.g., greetings, acknowledgments)
        min_content_length: Minimum character length for storage (default: 3)
        repetition_window: Number of recent messages to check for duplicates (default: 5)
        immediate_persist_patterns: Patterns requiring immediate persistence (bypass buffering)
        similarity_threshold: Threshold for near-duplicate detection, 0.0-1.0 (default: 0.9)
        enable_conflict_detection: Whether to detect conflicting memories (default: True)
    """
    trivial_patterns: List[str] = field(default_factory=lambda: [
        "hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"
    ])
    min_content_length: int = 3
    repetition_window: int = 5
    immediate_persist_patterns: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.9
    enable_conflict_detection: bool = True
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate min_content_length
        if not isinstance(self.min_content_length, int) or self.min_content_length < 0:
            raise ValueError(
                f"min_content_length must be a non-negative integer, got {self.min_content_length}"
            )
        
        # Validate repetition_window
        if not isinstance(self.repetition_window, int) or self.repetition_window < 0:
            raise ValueError(
                f"repetition_window must be a non-negative integer, got {self.repetition_window}"
            )
        
        # Validate similarity_threshold
        if not isinstance(self.similarity_threshold, (int, float)):
            raise ValueError(
                f"similarity_threshold must be a number, got {type(self.similarity_threshold).__name__}"
            )
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )
        
        # Validate trivial_patterns
        if not isinstance(self.trivial_patterns, list):
            raise ValueError(
                f"trivial_patterns must be a list, got {type(self.trivial_patterns).__name__}"
            )
        if not all(isinstance(p, str) for p in self.trivial_patterns):
            raise ValueError("all trivial_patterns must be strings")
        
        # Validate immediate_persist_patterns
        if not isinstance(self.immediate_persist_patterns, list):
            raise ValueError(
                f"immediate_persist_patterns must be a list, got {type(self.immediate_persist_patterns).__name__}"
            )
        if not all(isinstance(p, str) for p in self.immediate_persist_patterns):
            raise ValueError("all immediate_persist_patterns must be strings")
        
        # Validate enable_conflict_detection
        if not isinstance(self.enable_conflict_detection, bool):
            raise ValueError(
                f"enable_conflict_detection must be a boolean, got {type(self.enable_conflict_detection).__name__}"
            )


@dataclass
class SessionConfig:
    """
    Configuration for session management.
    
    Controls session lifecycle, buffering behavior, and cleanup operations.
    
    Attributes:
        timeout_seconds: Session timeout in seconds (default: 1800 = 30 minutes)
        cleanup_interval_seconds: Cleanup task interval in seconds (default: 300 = 5 minutes)
        max_buffer_size: Maximum buffered memories per session (default: 100)
        enable_buffering: Whether to buffer memories during sessions (default: True)
    """
    timeout_seconds: int = 1800  # 30 minutes
    cleanup_interval_seconds: int = 300  # 5 minutes
    max_buffer_size: int = 100
    enable_buffering: bool = True
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate timeout_seconds
        if not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be a positive integer, got {self.timeout_seconds}"
            )
        
        # Validate cleanup_interval_seconds
        if not isinstance(self.cleanup_interval_seconds, int) or self.cleanup_interval_seconds <= 0:
            raise ValueError(
                f"cleanup_interval_seconds must be a positive integer, got {self.cleanup_interval_seconds}"
            )
        
        # Validate max_buffer_size
        if not isinstance(self.max_buffer_size, int) or self.max_buffer_size <= 0:
            raise ValueError(
                f"max_buffer_size must be a positive integer, got {self.max_buffer_size}"
            )
        
        # Validate enable_buffering
        if not isinstance(self.enable_buffering, bool):
            raise ValueError(
                f"enable_buffering must be a boolean, got {type(self.enable_buffering).__name__}"
            )


# ============================================================================
# Configuration Loading
# ============================================================================


def load_write_strategy_config(
    config_file: Optional[str] = None,
    env_prefix: str = "LUMA_WRITE_STRATEGY_"
) -> WriteStrategyConfig:
    """
    Load WriteStrategyConfig from environment variables or config file.
    
    Priority order:
    1. Environment variables (highest priority)
    2. Config file (if provided)
    3. Default values (lowest priority)
    
    Environment variables:
        LUMA_WRITE_STRATEGY_TRIVIAL_PATTERNS: Comma-separated list of patterns
        LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH: Integer
        LUMA_WRITE_STRATEGY_REPETITION_WINDOW: Integer
        LUMA_WRITE_STRATEGY_IMMEDIATE_PERSIST_PATTERNS: Comma-separated list
        LUMA_WRITE_STRATEGY_SIMILARITY_THRESHOLD: Float between 0.0 and 1.0
        LUMA_WRITE_STRATEGY_ENABLE_CONFLICT_DETECTION: Boolean (true/false)
    
    Args:
        config_file: Optional path to JSON configuration file
        env_prefix: Prefix for environment variables (default: "LUMA_WRITE_STRATEGY_")
    
    Returns:
        WriteStrategyConfig instance with loaded configuration
    
    Raises:
        ValueError: If configuration values are invalid
        FileNotFoundError: If config_file is specified but doesn't exist
        json.JSONDecodeError: If config_file contains invalid JSON
    
    Example:
        >>> # Load from defaults
        >>> config = load_write_strategy_config()
        >>> 
        >>> # Load from file
        >>> config = load_write_strategy_config(config_file="config/write_strategy.json")
        >>> 
        >>> # Load from environment
        >>> os.environ["LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH"] = "5"
        >>> config = load_write_strategy_config()
        >>> config.min_content_length
        5
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {}
    
    # Load from config file if provided
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config_dict.update(file_config)
    
    # Override with environment variables
    env_trivial = os.environ.get(f"{env_prefix}TRIVIAL_PATTERNS")
    if env_trivial:
        config_dict["trivial_patterns"] = [p.strip() for p in env_trivial.split(",")]
    
    env_min_length = os.environ.get(f"{env_prefix}MIN_CONTENT_LENGTH")
    if env_min_length:
        config_dict["min_content_length"] = int(env_min_length)
    
    env_rep_window = os.environ.get(f"{env_prefix}REPETITION_WINDOW")
    if env_rep_window:
        config_dict["repetition_window"] = int(env_rep_window)
    
    env_immediate = os.environ.get(f"{env_prefix}IMMEDIATE_PERSIST_PATTERNS")
    if env_immediate:
        config_dict["immediate_persist_patterns"] = [p.strip() for p in env_immediate.split(",")]
    
    env_similarity = os.environ.get(f"{env_prefix}SIMILARITY_THRESHOLD")
    if env_similarity:
        config_dict["similarity_threshold"] = float(env_similarity)
    
    env_conflict = os.environ.get(f"{env_prefix}ENABLE_CONFLICT_DETECTION")
    if env_conflict:
        config_dict["enable_conflict_detection"] = env_conflict.lower() in ("true", "1", "yes")
    
    # Create and return config (validation happens in __post_init__)
    return WriteStrategyConfig(**config_dict)


def load_session_config(
    config_file: Optional[str] = None,
    env_prefix: str = "LUMA_SESSION_"
) -> SessionConfig:
    """
    Load SessionConfig from environment variables or config file.
    
    Priority order:
    1. Environment variables (highest priority)
    2. Config file (if provided)
    3. Default values (lowest priority)
    
    Environment variables:
        LUMA_SESSION_TIMEOUT_SECONDS: Integer (session timeout in seconds)
        LUMA_SESSION_CLEANUP_INTERVAL_SECONDS: Integer (cleanup interval in seconds)
        LUMA_SESSION_MAX_BUFFER_SIZE: Integer (max buffered memories per session)
        LUMA_SESSION_ENABLE_BUFFERING: Boolean (true/false)
    
    Args:
        config_file: Optional path to JSON configuration file
        env_prefix: Prefix for environment variables (default: "LUMA_SESSION_")
    
    Returns:
        SessionConfig instance with loaded configuration
    
    Raises:
        ValueError: If configuration values are invalid
        FileNotFoundError: If config_file is specified but doesn't exist
        json.JSONDecodeError: If config_file contains invalid JSON
    
    Example:
        >>> # Load from defaults
        >>> config = load_session_config()
        >>> 
        >>> # Load from file
        >>> config = load_session_config(config_file="config/session.json")
        >>> 
        >>> # Load from environment
        >>> os.environ["LUMA_SESSION_TIMEOUT_SECONDS"] = "3600"
        >>> config = load_session_config()
        >>> config.timeout_seconds
        3600
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {}
    
    # Load from config file if provided
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config_dict.update(file_config)
    
    # Override with environment variables
    env_timeout = os.environ.get(f"{env_prefix}TIMEOUT_SECONDS")
    if env_timeout:
        config_dict["timeout_seconds"] = int(env_timeout)
    
    env_cleanup = os.environ.get(f"{env_prefix}CLEANUP_INTERVAL_SECONDS")
    if env_cleanup:
        config_dict["cleanup_interval_seconds"] = int(env_cleanup)
    
    env_buffer_size = os.environ.get(f"{env_prefix}MAX_BUFFER_SIZE")
    if env_buffer_size:
        config_dict["max_buffer_size"] = int(env_buffer_size)
    
    env_buffering = os.environ.get(f"{env_prefix}ENABLE_BUFFERING")
    if env_buffering:
        config_dict["enable_buffering"] = env_buffering.lower() in ("true", "1", "yes")
    
    # Create and return config (validation happens in __post_init__)
    return SessionConfig(**config_dict)


# ============================================================================
# Memory Write Strategy Core
# ============================================================================


class Memory_Write_Strategy:
    """
    Determines when and how to persist user messages as memories.
    
    This component encapsulates all logic for intelligent memory persistence:
    - Evaluates write triggers to determine if a message should be stored
    - Validates content and metadata before storage
    - Detects and prevents duplicate memories
    - Detects and handles conflicting memories
    - Normalizes categories and tags
    - Coordinates with Session_Manager for buffering decisions
    
    The Memory_Write_Strategy maintains a history of recent messages to support
    repetition detection and provides configurable rules for write behavior.
    
    Thread Safety:
        This class is designed to be thread-safe when used with thread-safe
        Session_Manager and MemoryInterface implementations. The recent_messages
        list is protected by the Session_Manager's lock when accessed through
        the store_memory method.
    """
    
    def __init__(
        self,
        config: WriteStrategyConfig,
        session_manager: 'Session_Manager',
        memory_interface: 'MemoryInterface'
    ):
        """
        Initialize the Memory_Write_Strategy.
        
        Args:
            config: Configuration for write strategy behavior
            session_manager: Session_Manager for session-based buffering
            memory_interface: Interface for persisting memories to long-term storage
        """
        self.config = config
        self.session_manager = session_manager
        self.memory = memory_interface
        self.recent_messages: List[str] = []  # For repetition detection
    
    def evaluate_write_trigger(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> WriteDecision:
        """
        Evaluate whether content should be stored as a memory.
        
        Checks the content against various write trigger rules:
        1. Non-empty and non-whitespace
        2. Not a trivial message (greeting, acknowledgment)
        3. Not repetitive (not identical to recent messages)
        4. Meets minimum length requirement
        5. Contains substantive content
        
        Args:
            content: The user message content to evaluate
            metadata: Optional metadata associated with the message
        
        Returns:
            WriteDecision with should_write flag, reason, and metadata
        
        Example:
            >>> config = WriteStrategyConfig()
            >>> strategy = Memory_Write_Strategy(config, session_manager, memory)
            >>> decision = strategy.evaluate_write_trigger("Hello, how are you?")
            >>> if decision.should_write:
            ...     print(f"Approved: {decision.reason}")
        """
        # Check for empty or whitespace-only content
        if not content or not content.strip():
            logger.info(
                "Write trigger rejected: empty or whitespace content",
                extra={
                    "reason": "empty_or_whitespace",
                    "content_length": len(content) if content else 0
                }
            )
            return WriteDecision(
                should_write=False,
                reason="empty_or_whitespace",
                metadata={"content_length": len(content) if content else 0}
            )
        
        # Check against trivial patterns
        normalized_content = content.strip().casefold()
        for pattern in self.config.trivial_patterns:
            if normalized_content == pattern.casefold():
                logger.info(
                    f"Write trigger rejected: trivial pattern matched '{pattern}'",
                    extra={
                        "reason": "trivial_pattern",
                        "matched_pattern": pattern,
                        "content_length": len(content)
                    }
                )
                return WriteDecision(
                    should_write=False,
                    reason="trivial_pattern",
                    metadata={"matched_pattern": pattern}
                )
        
        # Check for repetition in recent messages window
        if self._is_repetitive(normalized_content):
            logger.info(
                "Write trigger rejected: repetitive content",
                extra={
                    "reason": "repetitive",
                    "repetition_window": len(self.recent_messages),
                    "content_length": len(content)
                }
            )
            return WriteDecision(
                should_write=False,
                reason="repetitive",
                metadata={"repetition_window": len(self.recent_messages)}
            )
        
        # Check minimum content length
        if len(content.strip()) < self.config.min_content_length:
            logger.info(
                f"Write trigger rejected: content below minimum length ({len(content.strip())} < {self.config.min_content_length})",
                extra={
                    "reason": "below_min_length",
                    "content_length": len(content.strip()),
                    "min_required": self.config.min_content_length
                }
            )
            return WriteDecision(
                should_write=False,
                reason="below_min_length",
                metadata={
                    "content_length": len(content.strip()),
                    "min_required": self.config.min_content_length
                }
            )
        
        # Content passes all checks - approve for storage
        # Add to recent messages for repetition detection
        self.recent_messages.append(normalized_content)
        
        logger.debug(
            "Write trigger approved",
            extra={
                "reason": "approved",
                "content_length": len(content)
            }
        )
        
        return WriteDecision(
            should_write=True,
            reason="approved",
            metadata={}
        )
    
    def _is_repetitive(self, normalized_content: str) -> bool:
        """
        Check if content is repetitive (identical to recent messages).
        
        Args:
            normalized_content: The content to check (already normalized)
        
        Returns:
            True if content matches one of the last N messages, False otherwise
        """
        if not self.recent_messages:
            return False
        
        # Check against the repetition window
        window = self.config.repetition_window
        recent = self.recent_messages[-window:] if len(self.recent_messages) > window else self.recent_messages
        
        return normalized_content in recent
    
    def validate_content(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Validate content and metadata before storage.
        
        Checks:
        - content is a non-empty string
        - content length does not exceed maximum limits
        - metadata is a dictionary if provided
        - tags is a list of strings if provided in metadata
        
        Raises:
            MemoryStorageError: If validation fails with a descriptive message
        
        Example:
            >>> try:
            ...     strategy.validate_content("test content", {"tags": ["test"]})
            ... except MemoryStorageError as e:
            ...     print(f"Validation failed: {e}")
        """
        # Validate content is a non-empty string
        if not isinstance(content, str):
            error_msg = f"Content must be a string, got {type(content).__name__}"
            logger.warning(
                f"Content validation failed: {error_msg}",
                extra={"validation_error": "invalid_type", "type": type(content).__name__}
            )
            raise MemoryStorageError(error_msg)
        
        if not content:
            error_msg = "Content cannot be empty"
            logger.warning(
                f"Content validation failed: {error_msg}",
                extra={"validation_error": "empty_content"}
            )
            raise MemoryStorageError(error_msg)
        
        # Validate content length (no explicit max in requirements, but check for reasonable limits)
        # Using a reasonable maximum of 100,000 characters
        max_length = 100000
        if len(content) > max_length:
            error_msg = f"Content length ({len(content)}) exceeds maximum ({max_length})"
            logger.warning(
                f"Content validation failed: {error_msg}",
                extra={
                    "validation_error": "content_too_long",
                    "content_length": len(content),
                    "max_length": max_length
                }
            )
            raise MemoryStorageError(error_msg)
        
        # Validate metadata is dict if provided
        if metadata is not None and not isinstance(metadata, dict):
            error_msg = f"Metadata must be a dictionary, got {type(metadata).__name__}"
            logger.warning(
                f"Content validation failed: {error_msg}",
                extra={"validation_error": "invalid_metadata_type", "type": type(metadata).__name__}
            )
            raise MemoryStorageError(error_msg)
        
        # Validate tags is list of strings if provided in metadata
        if metadata and "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list):
                error_msg = f"Tags must be a list, got {type(tags).__name__}"
                logger.warning(
                    f"Content validation failed: {error_msg}",
                    extra={"validation_error": "invalid_tags_type", "type": type(tags).__name__}
                )
                raise MemoryStorageError(error_msg)
            if not all(isinstance(tag, str) for tag in tags):
                error_msg = "All tags must be strings"
                logger.warning(
                    f"Content validation failed: {error_msg}",
                    extra={"validation_error": "invalid_tag_types", "tags": tags}
                )
                raise MemoryStorageError(error_msg)
        
        logger.debug(
            "Content validation passed",
            extra={"content_length": len(content), "has_metadata": metadata is not None}
        )
    
    def check_duplicate(
            self,
            content: str,
            category: str
        ) -> Optional[str]:
            """
            Check for duplicate or near-duplicate memories.

            This method checks if a memory with identical or similar content
            already exists in the same category. Exact duplicates are rejected,
            while near-duplicates may trigger metadata merging.

            Process:
            1. Normalize content (trim, lowercase, Unicode normalization for case-insensitive comparison)
            2. Query memory for existing entries in same category
            3. Compare normalized content for exact matches
            4. Calculate similarity for near-duplicates
            5. Return existing memory_id if duplicate found

            Args:
                content: The content to check for duplicates
                category: The category to check in

            Returns:
                None if no duplicate found
                memory_id of existing duplicate if found
            """
            import unicodedata

            # Normalize content for comparison using casefold() and Unicode NFC normalization
            # This handles cases like 'ß' vs 'SS', 'µ' vs 'Μ', etc.
            normalized_content = unicodedata.normalize('NFC', content.strip().casefold())

            # Normalize category for query
            normalized_category = unicodedata.normalize('NFC', category.strip().casefold())

            try:
                # Query memory for existing entries in the same category
                result = self.memory.retrieve(
                    params={
                        "category": normalized_category,
                        "limit": 100  # Check up to 100 recent memories in category
                    }
                )

                # Handle both dict and RetrievalResult return types
                if isinstance(result, dict):
                    memories = result.get("memories", [])
                else:
                    memories = result.memories

                # Check each retrieved memory for duplicates
                for memory_entry in memories:
                    # Handle both dict and MemoryEntry types
                    if isinstance(memory_entry, dict):
                        existing_content = memory_entry.get("content", "")
                        memory_id = memory_entry.get("id")
                    else:
                        existing_content = memory_entry.content
                        memory_id = memory_entry.id

                    # Normalize existing content with same approach
                    existing_normalized = unicodedata.normalize('NFC', existing_content.strip().casefold())

                    # Check for exact duplicate
                    if existing_normalized == normalized_content:
                        logger.info(
                            f"Exact duplicate detected: memory_id={memory_id}",
                            extra={
                                "reason": "exact_duplicate",
                                "memory_id": memory_id,
                                "content_length": len(content),
                                "category": category
                            }
                        )
                        return memory_id

                    # Check for near-duplicate using similarity threshold
                    similarity = self._calculate_similarity(normalized_content, existing_normalized)
                    if similarity >= self.config.similarity_threshold:
                        logger.info(
                            f"Near-duplicate detected: memory_id={memory_id}, similarity={similarity:.2f}",
                            extra={
                                "reason": "near_duplicate",
                                "memory_id": memory_id,
                                "similarity": similarity,
                                "threshold": self.config.similarity_threshold,
                                "content_length": len(content),
                                "category": category
                            }
                        )
                        return memory_id

                # No duplicate found
                return None

            except Exception as e:
                # If retrieval fails, log warning and skip duplicate check
                # This allows storage to continue even if duplicate detection fails
                logger.warning(
                    f"Duplicate check failed, skipping: {e}",
                    exc_info=True
                )
                return None

    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Uses a simple character-based similarity metric (Jaccard similarity
        on character bigrams). This is a basic implementation suitable for
        detecting near-duplicates with minor variations.
        
        Args:
            text1: First text string (normalized)
            text2: Second text string (normalized)
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Handle edge cases
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Create character bigrams for each text
        def get_bigrams(text: str) -> set:
            return set(text[i:i+2] for i in range(len(text) - 1))
        
        bigrams1 = get_bigrams(text1)
        bigrams2 = get_bigrams(text2)
        
        # Handle edge case where texts are too short for bigrams
        if not bigrams1 and not bigrams2:
            return 1.0 if text1 == text2 else 0.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def detect_conflict(
        self,
        content: str,
        category: str
    ) -> Optional[str]:
        """
        Detect if a new memory conflicts with existing memories.
        
        This method checks if the new memory contradicts existing memories
        in the same category using basic keyword-based conflict detection.
        
        The basic approach looks for:
        1. Memories in the same category with similar content
        2. Presence of negation keywords (not, don't, never, etc.) that
           might indicate contradiction
        
        Note: This is a basic implementation. Full conflict detection
        requires semantic analysis (future: LLM-based).
        
        Args:
            content: The content to check for conflicts
            category: The category to check in
        
        Returns:
            None if no conflict detected
            memory_id of conflicting memory if detected
        """
        # Skip conflict detection if disabled in config
        if not self.config.enable_conflict_detection:
            return None
        
        # Normalize content and category for comparison
        normalized_content = content.strip().casefold()
        normalized_category = category.strip().casefold()
        
        # Define negation keywords that might indicate contradiction
        negation_keywords = {
            'not', 'no', 'never', "don't", "doesn't", "didn't", 
            "won't", "wouldn't", "can't", "cannot", "shouldn't",
            "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
            "hadn't", "neither", "nor", "none", "nobody", "nothing"
        }
        
        # Extract words from new content
        new_words = set(normalized_content.lower().split())
        new_has_negation = bool(new_words & negation_keywords)
        
        try:
            # Query memory for existing entries in the same category
            result = self.memory.retrieve(
                params={
                    "category": normalized_category,
                    "limit": 50  # Check up to 50 recent memories in category
                }
            )
            
            # Handle both dict and RetrievalResult return types
            if isinstance(result, dict):
                memories = result.get("memories", [])
            else:
                memories = result.memories
            
            # Check each retrieved memory for potential conflicts
            for memory_entry in memories:
                # Handle both dict and MemoryEntry types
                if isinstance(memory_entry, dict):
                    existing_content = memory_entry.get("content", "").strip().casefold()
                    memory_id = memory_entry.get("id")
                else:
                    existing_content = memory_entry.content.strip().casefold()
                    memory_id = memory_entry.id
                
                # Extract words from existing content
                existing_words = set(existing_content.casefold().split())
                existing_has_negation = bool(existing_words & negation_keywords)
                
                # Calculate content similarity (excluding negation keywords)
                content_words_new = new_words - negation_keywords
                content_words_existing = existing_words - negation_keywords
                
                # Check if there's significant word overlap (potential same topic)
                if content_words_new and content_words_existing:
                    common_words = content_words_new & content_words_existing
                    overlap_ratio = len(common_words) / min(len(content_words_new), len(content_words_existing))
                    
                    # If significant overlap (>40%) and opposite negation status, likely conflict
                    if overlap_ratio > 0.4 and new_has_negation != existing_has_negation:
                        logger.info(
                            f"Potential conflict detected: memory_id={memory_id}",
                            extra={
                                "reason": "conflict",
                                "memory_id": memory_id,
                                "overlap_ratio": overlap_ratio,
                                "new_has_negation": new_has_negation,
                                "existing_has_negation": existing_has_negation,
                                "content_length": len(content),
                                "category": category
                            }
                        )
                        return memory_id
            
            # No conflict found
            return None
            
        except Exception as e:
            # If retrieval fails, log warning and skip conflict detection
            # This allows storage to continue even if conflict detection fails
            logger.warning(
                f"Conflict detection failed, skipping: {e}",
                exc_info=True
            )
            return None
    
    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata fields for storage.
        
        Normalization steps:
        - Trim and lowercase category
        - Trim, lowercase, and deduplicate tags
        - Add timestamp if not present
        - Add session_id from active session if available
        - Apply default_category if no category provided
        - Merge with default_tags from adapter configuration
        
        Args:
            metadata: The metadata dictionary to normalize
        
        Returns:
            Normalized metadata dictionary
        
        Requirements:
            6.1: Attach timestamp in ISO 8601 format
            6.2: Attach session_id if within active session
            6.4: Normalize category (trim and lowercase)
            6.5: Normalize tags (trim, lowercase, deduplicate)
            6.6: Apply default_category if no category provided
            6.7: Merge with default_tags
        """
        from datetime import datetime, UTC
        
        normalized = {}
        
        # Copy metadata fields
        for key, value in metadata.items():
            normalized[key] = value
        
        # Add timestamp if not present (Requirement 6.1)
        if "timestamp" not in normalized:
            normalized["timestamp"] = datetime.now(UTC).isoformat()
        
        # Add session_id from active session if available (Requirement 6.2)
        if self.session_manager:
            # Get the current active session from the session manager
            # We need to check if there's an active session
            with self.session_manager.lock:
                if self.session_manager.sessions:
                    # In a real scenario, we'd track the current session ID
                    # For now, we'll check if there's exactly one active session
                    # or use a current_session_id attribute if available
                    if hasattr(self.session_manager, 'current_session_id') and self.session_manager.current_session_id:
                        normalized["session_id"] = self.session_manager.current_session_id
                    elif len(self.session_manager.sessions) == 1:
                        # If there's only one session, use it
                        session_id = next(iter(self.session_manager.sessions.keys()))
                        normalized["session_id"] = session_id
        
        # Normalize category: trim and casefold (Requirement 6.4)
        if "category" in normalized and normalized["category"]:
            if isinstance(normalized["category"], str):
                normalized["category"] = normalized["category"].strip().casefold()
        
        # Apply default_category if no category provided (Requirement 6.6)
        if "category" not in normalized or not normalized["category"]:
            # Try to get default_category from the memory adapter if it has one
            if hasattr(self.memory, 'default_category') and self.memory.default_category:
                # Normalize the default_category (trim and casefold) for consistency
                normalized["category"] = self.memory.default_category.strip().casefold()
            else:
                # Fallback to a sensible default
                normalized["category"] = "general"
        
        # Normalize tags: trim, casefold, and deduplicate (Requirement 6.5)
        tags_to_normalize = []
        
        # Merge with default_tags from adapter configuration FIRST (Requirement 6.7)
        if hasattr(self.memory, 'default_tags') and self.memory.default_tags:
            tags_to_normalize.extend(self.memory.default_tags)
        
        # Get tags from metadata SECOND
        if "tags" in normalized and normalized["tags"]:
            if isinstance(normalized["tags"], list):
                tags_to_normalize.extend(normalized["tags"])
        
        # Normalize: trim, casefold, and deduplicate
        if tags_to_normalize:
            normalized_tags = []
            seen = set()
            for tag in tags_to_normalize:
                if isinstance(tag, str):
                    normalized_tag = tag.strip().casefold()
                    if normalized_tag and normalized_tag not in seen:
                        normalized_tags.append(normalized_tag)
                        seen.add(normalized_tag)
            normalized["tags"] = normalized_tags
        elif "tags" not in normalized:
            # Ensure tags field exists even if empty
            normalized["tags"] = []
        
        return normalized
    
    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        immediate: bool = False
    ) -> str:
        """
        Store memory with full write strategy logic.
        
        Process:
        1. Evaluate write trigger
        2. Validate content and metadata
        3. Check for duplicates
        4. Detect conflicts
        5. Normalize metadata
        6. Decide buffering vs immediate persistence
        7. Store via Session_Manager or MemoryInterface
        
        Args:
            content: The content to store
            metadata: Optional metadata to associate with the memory
            immediate: Whether to bypass buffering and persist immediately
        
        Returns:
            memory_id of stored memory
        
        Raises:
            MemoryStorageError: If storage fails or validation fails
        
        Example:
            >>> try:
            ...     memory_id = strategy.store_memory(
            ...         "Important information",
            ...         metadata={"tags": ["important"]}
            ...     )
            ...     print(f"Stored with ID: {memory_id}")
            ... except MemoryStorageError as e:
            ...     print(f"Storage failed: {e}")
        """
        # Step 1: Evaluate write trigger
        decision = self.evaluate_write_trigger(content, metadata)
        
        if not decision.should_write:
            logger.info(
                f"Memory write rejected: {decision.reason}",
                extra={
                    "reason": decision.reason,
                    "content_length": len(content),
                    "decision_metadata": decision.metadata
                }
            )
            raise MemoryStorageError(
                f"Memory write rejected: {decision.reason}"
            )
        
        # Step 2: Validate content and metadata
        try:
            self.validate_content(content, metadata)
        except MemoryStorageError as e:
            logger.error(
                f"Content validation failed: {e}",
                extra={"content_length": len(content)},
                exc_info=True
            )
            raise
        
        # Step 3: Normalize metadata early (needed for duplicate/conflict checks)
        normalized_metadata = self.normalize_metadata(metadata or {})
        
        # Extract category for duplicate and conflict checks
        category = normalized_metadata.get("category", "general")
        
        # Step 4: Check for duplicates
        duplicate_id = self.check_duplicate(content, category)
        if duplicate_id:
            logger.info(
                f"Duplicate memory detected, returning existing ID: {duplicate_id}",
                extra={
                    "reason": "duplicate",
                    "duplicate_id": duplicate_id,
                    "content_length": len(content),
                    "category": category,
                    "tags": normalized_metadata.get("tags", [])
                }
            )
            return duplicate_id
        
        # Step 5: Detect conflicts
        conflict_id = self.detect_conflict(content, category)
        if conflict_id:
            logger.info(
                f"Conflict detected with memory: {conflict_id}",
                extra={
                    "conflict_id": conflict_id,
                    "content_length": len(content),
                    "category": category
                }
            )
            # Add conflict metadata to the new memory
            normalized_metadata["conflicts_with"] = conflict_id
            normalized_metadata["conflict_detected"] = True
            
            # Note: Marking the older memory as potentially_outdated would require
            # updating the existing memory, which is beyond the scope of this method.
            # This would typically be handled by a separate conflict resolution process.
        
        # Step 6: Decide buffering vs immediate persistence
        # Immediate persistence if:
        # - immediate flag is True
        # - No active session exists
        # - Content matches immediate_persist_patterns
        should_persist_immediately = immediate
        
        # Check if content matches immediate persistence patterns
        if not should_persist_immediately and self.config.immediate_persist_patterns:
            content_lower = content.lower()
            for pattern in self.config.immediate_persist_patterns:
                if pattern.lower() in content_lower:
                    should_persist_immediately = True
                    logger.debug(
                        f"Content matches immediate persist pattern: {pattern}",
                        extra={"pattern": pattern}
                    )
                    break
        
        # Check if there's an active session for buffering
        active_session_id = None
        if self.session_manager:
            with self.session_manager.lock:
                if self.session_manager.sessions:
                    # Try to get current_session_id if available
                    if hasattr(self.session_manager, 'current_session_id') and self.session_manager.current_session_id:
                        active_session_id = self.session_manager.current_session_id
                    elif len(self.session_manager.sessions) == 1:
                        # If there's only one session, use it
                        active_session_id = next(iter(self.session_manager.sessions.keys()))
        
        # If no active session, must persist immediately
        if not active_session_id:
            should_persist_immediately = True
        
        # Step 7: Store via Session_Manager or MemoryInterface
        if should_persist_immediately:
            # Immediate persistence
            try:
                memory_id = self.memory.store(content, normalized_metadata)
                logger.info(
                    f"Memory stored immediately: memory_id={memory_id}",
                    extra={
                        "memory_id": memory_id,
                        "content_length": len(content),
                        "category": normalized_metadata.get("category", "unknown"),
                        "tags": normalized_metadata.get("tags", []),
                        "has_conflict": "conflicts_with" in normalized_metadata,
                        "storage_type": "immediate"
                    }
                )
                return memory_id
            except MemoryStorageError as e:
                logger.error(
                    f"Memory storage failed: {e}",
                    extra={
                        "content_length": len(content),
                        "category": normalized_metadata.get("category", "unknown"),
                        "tags": normalized_metadata.get("tags", []),
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise
        else:
            # Buffer for session persistence
            try:
                self.session_manager.buffer_memory(active_session_id, content, normalized_metadata)
                # Generate a buffer entry ID
                buffer_index = len(self.session_manager.sessions[active_session_id].buffered_memories)
                buffer_id = f"buffered:{active_session_id}:{buffer_index}"
                logger.info(
                    f"Memory buffered for session: buffer_id={buffer_id}",
                    extra={
                        "session_id": active_session_id,
                        "buffer_id": buffer_id,
                        "content_length": len(content),
                        "category": normalized_metadata.get("category", "unknown"),
                        "tags": normalized_metadata.get("tags", []),
                        "has_conflict": "conflicts_with" in normalized_metadata,
                        "storage_type": "buffered"
                    }
                )
                return buffer_id
            except Exception as e:
                # If buffering fails, fall back to immediate persistence
                logger.warning(
                    f"Session buffering failed, falling back to immediate persistence: {e}",
                    extra={
                        "session_id": active_session_id,
                        "content_length": len(content),
                        "error": str(e)
                    },
                    exc_info=True
                )
                try:
                    memory_id = self.memory.store(content, normalized_metadata)
                    logger.info(
                        f"Memory stored immediately (buffering fallback): memory_id={memory_id}",
                        extra={
                            "memory_id": memory_id,
                            "content_length": len(content),
                            "category": normalized_metadata.get("category", "unknown"),
                            "tags": normalized_metadata.get("tags", []),
                            "storage_type": "immediate_fallback"
                        }
                    )
                    return memory_id
                except MemoryStorageError as storage_error:
                    logger.error(
                        f"Memory storage failed: {storage_error}",
                        extra={
                            "content_length": len(content),
                            "category": normalized_metadata.get("category", "unknown"),
                            "tags": normalized_metadata.get("tags", []),
                            "error": str(storage_error)
                        },
                        exc_info=True
                    )
                    raise
