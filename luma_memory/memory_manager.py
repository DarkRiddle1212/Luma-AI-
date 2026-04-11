"""
Memory Manager for Luma Memory Module.

This module provides the central MemoryManager class that coordinates all
memory operations including storage, encryption, validation, and summarization.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus, create_memory_entry
from luma_memory.storage.backend import StorageBackend, StorageError
from luma_memory.processing.encryption import EncryptionService
from luma_memory.processing.validation import ValidationManager, ValidationError
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.config import MemoryModuleConfig
from luma_memory.utils.error_tracker import ErrorTracker, ErrorCategory, ErrorSeverity


logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory manager that coordinates all memory operations.
    
    The MemoryManager orchestrates the full pipeline for memory operations:
    1. Validation of input data
    2. Encryption of sensitive data
    3. Storage operations
    4. Decryption on retrieval
    5. Automatic summarization triggers
    
    It uses dependency injection for all components, making it easy to test
    and extend with different implementations.
    
    Attributes:
        storage: Storage backend for persistence
        encryption: Encryption service for sensitive data
        validation: Validation manager for data integrity
        summarizer: Context summarizer for reducing storage overhead
        config: Configuration settings
    
    Example:
        >>> from luma_memory.storage.sqlite_storage import SQLiteStorage
        >>> storage = SQLiteStorage("./data/memory.db")
        >>> encryption = EncryptionService("./keys/encryption.key")
        >>> validation = ValidationManager()
        >>> summarizer = ContextSummarizer()
        >>> config = MemoryModuleConfig()
        >>> 
        >>> manager = MemoryManager(
        ...     storage=storage,
        ...     encryption=encryption,
        ...     validation=validation,
        ...     summarizer=summarizer,
        ...     config=config
        ... )
        >>> 
        >>> # Create a memory
        >>> entry_id = manager.create_memory(
        ...     action="User opened document",
        ...     context={"file": "report.pdf"},
        ...     device_id="laptop-001"
        ... )
        >>> 
        >>> # Retrieve a memory
        >>> entry = manager.get_memory(entry_id)
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        encryption: Optional[EncryptionService] = None,
        validation: Optional[ValidationManager] = None,
        summarizer: Optional[ContextSummarizer] = None,
        config: Optional[MemoryModuleConfig] = None,
        error_tracker: Optional[ErrorTracker] = None
    ):
        """
        Initialize the memory manager with dependencies.
        
        Args:
            storage: Storage backend for persistence (required)
            encryption: Encryption service for sensitive data (optional)
            validation: Validation manager for data integrity (optional)
            summarizer: Context summarizer for reducing storage (optional)
            config: Configuration settings (optional, uses defaults if not provided)
            error_tracker: Error tracker for monitoring errors (optional)
        
        Raises:
            ValueError: If storage is None
        """
        if storage is None:
            raise ValueError("Storage backend is required")
        
        self.storage = storage
        self.encryption = encryption
        self.validation = validation or ValidationManager()
        self.summarizer = summarizer
        self.config = config or MemoryModuleConfig()
        self.error_tracker = error_tracker or ErrorTracker()
        
        # Performance monitoring metrics
        self._metrics = {
            'create_memory': {'count': 0, 'total_time_ms': 0, 'min_time_ms': float('inf'), 'max_time_ms': 0, 'errors': 0},
            'get_memory': {'count': 0, 'total_time_ms': 0, 'min_time_ms': float('inf'), 'max_time_ms': 0, 'errors': 0},
            'query_memories': {'count': 0, 'total_time_ms': 0, 'min_time_ms': float('inf'), 'max_time_ms': 0, 'errors': 0},
            'update_memory': {'count': 0, 'total_time_ms': 0, 'min_time_ms': float('inf'), 'max_time_ms': 0, 'errors': 0},
            'delete_memory': {'count': 0, 'total_time_ms': 0, 'min_time_ms': float('inf'), 'max_time_ms': 0, 'errors': 0},
        }
        
        logger.info(
            f"MemoryManager initialized with storage={type(storage).__name__}, "
            f"encryption={'enabled' if encryption else 'disabled'}, "
            f"validation=enabled, "
            f"summarizer={'enabled' if summarizer else 'disabled'}, "
            f"metrics={'enabled' if self.config.enable_metrics else 'disabled'}, "
            f"error_tracking=enabled"
        )
    
    def create_memory(
        self,
        action: str,
        context: Dict[str, Any],
        device_id: str,
        sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC,
        tags: Optional[List[str]] = None,
        entry_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Create a new memory entry with full pipeline processing.
        
        Pipeline:
        1. Sanitize input data
        2. Create MemoryEntry object
        3. Validate entry
        4. Encrypt sensitive data if needed
        5. Store entry
        6. Check if summarization should be triggered
        7. Log operation
        
        Args:
            action: Description of the user action
            context: Dictionary containing contextual information
            device_id: Identifier of the device creating the entry
            sensitivity: Privacy level (defaults to PUBLIC)
            tags: Optional list of tags for categorization
            entry_id: Optional custom ID (generates UUID if not provided)
            timestamp: Optional timestamp (uses current time if not provided)
        
        Returns:
            The ID of the created memory entry
        
        Raises:
            ValidationError: If the entry fails validation
            StorageError: If storage operation fails
            Exception: If encryption or other operations fail
        
        Example:
            >>> entry_id = manager.create_memory(
            ...     action="User opened document",
            ...     context={"file": "report.pdf", "page": 1},
            ...     device_id="laptop-001",
            ...     sensitivity=SensitivityLevel.PRIVATE,
            ...     tags=["document", "work"]
            ... )
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            # Step 1: Sanitize input data
            sanitized_context = self.validation.sanitize_input(context)
            logger.debug(f"Sanitized input context with {len(sanitized_context)} keys")
            
            # Step 2: Create MemoryEntry object
            entry = create_memory_entry(
                action=action,
                context=sanitized_context,
                device_id=device_id,
                sensitivity=sensitivity,
                tags=tags or [],
                entry_id=entry_id,
                timestamp=timestamp
            )
            logger.debug(f"Created MemoryEntry with id={entry.id}")
            
            # Step 3: Validate entry
            self.validation.validate_and_raise(entry)
            logger.debug(f"Validated MemoryEntry id={entry.id}")
            
            # Step 4: Encrypt sensitive data if needed
            if self.encryption and sensitivity in [SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
                entry = self._encrypt_entry(entry)
                logger.debug(f"Encrypted MemoryEntry id={entry.id}")
            
            # Step 5: Store entry
            stored_id = self.storage.create_entry(entry)
            logger.info(f"Stored MemoryEntry id={stored_id}")
            
            # Step 6: Check if summarization should be triggered
            if self.summarizer:
                self._check_summarization_trigger()
            
            # Step 7: Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"create_memory completed in {elapsed_ms:.2f}ms for entry {stored_id}")
            
            # Performance monitoring
            if elapsed_ms > 100:
                logger.warning(
                    f"create_memory exceeded 100ms target: {elapsed_ms:.2f}ms for entry {stored_id}"
                )
            
            # Record metrics
            self._record_metric('create_memory', elapsed_ms, error=False)
            
            return stored_id
            
        except ValidationError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=e,
                operation='create_memory',
                context={'action': action, 'device_id': device_id}
            )
            logger.error(f"Validation failed for create_memory: {e}")
            raise
        except StorageError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='create_memory',
                context={'action': action, 'device_id': device_id}
            )
            logger.error(f"Storage failed for create_memory: {e}")
            raise
        except Exception as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='create_memory',
                context={'action': action, 'device_id': device_id},
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in create_memory: {e}")
            raise
        finally:
            if error_occurred:
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric('create_memory', elapsed_ms, error=True)
    
    def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID with decryption.
        
        Pipeline:
        1. Retrieve entry from storage
        2. Decrypt sensitive data if needed
        3. Log operation
        
        Args:
            entry_id: The unique identifier of the entry to retrieve
        
        Returns:
            The MemoryEntry if found, None otherwise
        
        Raises:
            StorageError: If retrieval operation fails
            Exception: If decryption fails
        
        Example:
            >>> entry = manager.get_memory("abc-123")
            >>> if entry:
            ...     print(f"Action: {entry.action}")
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            # Step 1: Retrieve entry from storage
            entry = self.storage.get_entry(entry_id)
            
            if entry is None:
                logger.debug(f"Entry not found: {entry_id}")
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric('get_memory', elapsed_ms, error=False)
                return None
            
            logger.debug(f"Retrieved MemoryEntry id={entry_id}")
            
            # Step 2: Decrypt sensitive data if needed
            if self.encryption and entry.sensitivity in [SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
                entry = self._decrypt_entry(entry)
                logger.debug(f"Decrypted MemoryEntry id={entry_id}")
            
            # Step 3: Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"get_memory completed in {elapsed_ms:.2f}ms for entry {entry_id}")
            
            # Performance monitoring
            if elapsed_ms > 200:
                logger.warning(
                    f"get_memory exceeded 200ms target: {elapsed_ms:.2f}ms for entry {entry_id}"
                )
            
            # Record metrics
            self._record_metric('get_memory', elapsed_ms, error=False)
            
            return entry
            
        except StorageError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='get_memory',
                context={'entry_id': entry_id}
            )
            logger.error(f"Storage failed for get_memory({entry_id}): {e}")
            raise
        except Exception as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='get_memory',
                context={'entry_id': entry_id},
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in get_memory({entry_id}): {e}")
            raise
        finally:
            if error_occurred:
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric('get_memory', elapsed_ms, error=True)
    
    def query_memories(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        action_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """
        Query memory entries with filters and decryption.
        
        Pipeline:
        1. Query storage with filters
        2. Decrypt sensitive entries if needed
        3. Log operation
        
        Args:
            start_time: Optional start of time range filter (inclusive)
            end_time: Optional end of time range filter (inclusive)
            tags: Optional list of tags to filter by
            action_type: Optional action type to filter by (partial match)
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip for pagination (default: 0)
        
        Returns:
            List of MemoryEntry instances matching the filters
        
        Raises:
            StorageError: If query operation fails
            Exception: If decryption fails
        
        Example:
            >>> from datetime import datetime, timedelta
            >>> yesterday = datetime.now() - timedelta(days=1)
            >>> entries = manager.query_memories(
            ...     start_time=yesterday,
            ...     tags=["work"],
            ...     limit=50
            ... )
        """
        query_start_time = time.time()
        error_occurred = False
        
        try:
            # Step 1: Query storage with filters
            entries = self.storage.query_entries(
                start_time=start_time,
                end_time=end_time,
                tags=tags,
                action_type=action_type,
                limit=limit,
                offset=offset
            )
            
            logger.debug(f"Retrieved {len(entries)} entries from storage")
            
            # Step 2: Decrypt sensitive entries if needed
            if self.encryption:
                decrypted_entries = []
                for entry in entries:
                    if entry.sensitivity in [SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
                        entry = self._decrypt_entry(entry)
                    decrypted_entries.append(entry)
                entries = decrypted_entries
                logger.debug(f"Decrypted {len(entries)} entries")
            
            # Step 3: Log performance
            elapsed_ms = (time.time() - query_start_time) * 1000
            logger.info(
                f"query_memories completed in {elapsed_ms:.2f}ms, "
                f"returned {len(entries)} entries"
            )
            
            # Performance monitoring
            if elapsed_ms > 200:
                logger.warning(
                    f"query_memories exceeded 200ms target: {elapsed_ms:.2f}ms "
                    f"for {len(entries)} entries"
                )
            
            # Record metrics
            self._record_metric('query_memories', elapsed_ms, error=False)
            
            return entries
            
        except StorageError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='query_memories',
                context={'limit': limit, 'offset': offset}
            )
            logger.error(f"Storage failed for query_memories: {e}")
            raise
        except Exception as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='query_memories',
                context={'limit': limit, 'offset': offset},
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in query_memories: {e}")
            raise
        finally:
            if error_occurred:
                elapsed_ms = (time.time() - query_start_time) * 1000
                self._record_metric('query_memories', elapsed_ms, error=True)
    
    def update_memory(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.
        
        Pipeline:
        1. Validate partial update
        2. Sanitize update data
        3. Encrypt sensitive fields if needed
        4. Update storage
        5. Log operation
        
        Args:
            entry_id: The unique identifier of the entry to update
            updates: Dictionary of field names and new values
        
        Returns:
            True if the entry was updated, False if not found
        
        Raises:
            ValidationError: If the updates are invalid
            StorageError: If update operation fails
            Exception: If encryption fails
        
        Example:
            >>> success = manager.update_memory(
            ...     "abc-123",
            ...     {"tags": ["work", "important"], "summary": "Updated summary"}
            ... )
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            # Step 1: Validate partial update
            is_valid, error = self.validation.validate_partial_update(updates)
            if not is_valid:
                raise ValidationError(error)
            logger.debug(f"Validated update for entry {entry_id}")
            
            # Step 2: Sanitize update data
            if 'context' in updates:
                updates['context'] = self.validation.sanitize_input(updates['context'])
            logger.debug(f"Sanitized update data for entry {entry_id}")
            
            # Step 3: Encrypt sensitive fields if needed
            # Note: We need to check the entry's sensitivity level
            # For now, we'll encrypt context if it's being updated and encryption is enabled
            if self.encryption and 'context' in updates:
                # Retrieve the entry to check its sensitivity
                entry = self.storage.get_entry(entry_id)
                if entry and entry.sensitivity in [SensitivityLevel.PRIVATE, SensitivityLevel.SENSITIVE]:
                    # Encrypt the context update
                    encrypted_context = {}
                    for key, value in updates['context'].items():
                        if isinstance(value, str):
                            encrypted_context[key] = self.encryption.encrypt(value)
                        else:
                            encrypted_context[key] = value
                    updates['context'] = encrypted_context
                    logger.debug(f"Encrypted context update for entry {entry_id}")
            
            # Step 4: Update storage
            success = self.storage.update_entry(entry_id, updates)
            
            if success:
                logger.info(f"Updated MemoryEntry id={entry_id}")
            else:
                logger.warning(f"Entry not found for update: {entry_id}")
            
            # Step 5: Log performance
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"update_memory completed in {elapsed_ms:.2f}ms for entry {entry_id}")
            
            # Record metrics
            self._record_metric('update_memory', elapsed_ms, error=False)
            
            return success
            
        except ValidationError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                error=e,
                operation='update_memory',
                context={'entry_id': entry_id, 'updates': list(updates.keys())}
            )
            logger.error(f"Validation failed for update_memory({entry_id}): {e}")
            raise
        except StorageError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='update_memory',
                context={'entry_id': entry_id, 'updates': list(updates.keys())}
            )
            logger.error(f"Storage failed for update_memory({entry_id}): {e}")
            raise
        except Exception as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='update_memory',
                context={'entry_id': entry_id, 'updates': list(updates.keys())},
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in update_memory({entry_id}): {e}")
            raise
        finally:
            if error_occurred:
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric('update_memory', elapsed_ms, error=True)
    
    def delete_memory(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The unique identifier of the entry to delete
        
        Returns:
            True if the entry was deleted, False if not found
        
        Raises:
            StorageError: If delete operation fails
        
        Example:
            >>> success = manager.delete_memory("abc-123")
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            success = self.storage.delete_entry(entry_id)
            
            if success:
                logger.info(f"Deleted MemoryEntry id={entry_id}")
            else:
                logger.warning(f"Entry not found for deletion: {entry_id}")
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"delete_memory completed in {elapsed_ms:.2f}ms for entry {entry_id}")
            
            # Record metrics
            self._record_metric('delete_memory', elapsed_ms, error=False)
            
            return success
            
        except StorageError as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='delete_memory',
                context={'entry_id': entry_id}
            )
            logger.error(f"Storage failed for delete_memory({entry_id}): {e}")
            raise
        except Exception as e:
            error_occurred = True
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                error=e,
                operation='delete_memory',
                context={'entry_id': entry_id},
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in delete_memory({entry_id}): {e}")
            raise
        finally:
            if error_occurred:
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric('delete_memory', elapsed_ms, error=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage and performance statistics.
        
        Returns:
            Dictionary containing statistics including:
                - Storage statistics from backend
                - Configuration settings
                - Component status
                - Performance metrics (if enabled)
                - Error tracking statistics
        
        Raises:
            StorageError: If stats operation fails
        
        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Total entries: {stats['total_entries']}")
            >>> print(f"Average create time: {stats['performance']['create_memory']['avg_time_ms']}ms")
            >>> print(f"Total errors: {stats['error_tracking']['total_errors']}")
        """
        try:
            storage_stats = self.storage.get_storage_stats()
            
            stats = {
                **storage_stats,
                'encryption_enabled': self.encryption is not None,
                'summarizer_enabled': self.summarizer is not None,
                'config': {
                    'cache_size': self.config.cache_size,
                    'max_storage_size_mb': self.config.max_storage_size_mb,
                    'summarization_threshold': self.config.summarization_threshold,
                }
            }
            
            # Add performance metrics if enabled
            if self.config.enable_metrics:
                stats['performance'] = self.get_performance_metrics()
            
            # Add error tracking statistics
            stats['error_tracking'] = self.error_tracker.get_error_stats()
            
            logger.debug(f"Retrieved stats: {len(stats)} fields")
            return stats
            
        except StorageError as e:
            self.error_tracker.track_error(
                category=ErrorCategory.STORAGE,
                severity=ErrorSeverity.MEDIUM,
                error=e,
                operation='get_stats'
            )
            logger.error(f"Storage failed for get_stats: {e}")
            raise
        except Exception as e:
            self.error_tracker.track_error(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                error=e,
                operation='get_stats',
                include_stack_trace=True
            )
            logger.error(f"Unexpected error in get_stats: {e}")
            raise
    
    def _encrypt_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Encrypt sensitive fields in a memory entry.
        
        Currently encrypts string values in the context dictionary.
        
        Args:
            entry: The memory entry to encrypt
        
        Returns:
            Memory entry with encrypted context
        
        Raises:
            Exception: If encryption fails
        """
        if not self.encryption:
            return entry
        
        encrypted_context = {}
        for key, value in entry.context.items():
            if isinstance(value, str):
                encrypted_context[key] = self.encryption.encrypt(value)
            else:
                # Keep non-string values as-is
                encrypted_context[key] = value
        
        entry.context = encrypted_context
        return entry
    
    def _decrypt_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Decrypt sensitive fields in a memory entry.
        
        Currently decrypts bytes values in the context dictionary.
        
        Args:
            entry: The memory entry to decrypt
        
        Returns:
            Memory entry with decrypted context
        
        Raises:
            Exception: If decryption fails
        """
        if not self.encryption:
            return entry
        
        decrypted_context = {}
        for key, value in entry.context.items():
            if isinstance(value, bytes):
                decrypted_context[key] = self.encryption.decrypt(value)
            else:
                # Keep non-bytes values as-is
                decrypted_context[key] = value
        
        entry.context = decrypted_context
        return entry
    
    def _check_summarization_trigger(self) -> None:
        """
        Check if summarization should be triggered and execute if needed.
        
        Triggers summarization based on:
        - Number of entries exceeding threshold
        - Storage size exceeding threshold
        """
        if not self.summarizer:
            return
        
        try:
            stats = self.storage.get_storage_stats()
            total_entries = stats.get('total_entries', 0)
            storage_size = stats.get('storage_size_bytes', 0)
            
            if self.summarizer.should_trigger_summarization(total_entries, storage_size):
                logger.info(
                    f"Summarization triggered: {total_entries} entries, "
                    f"{storage_size} bytes"
                )
                self._perform_summarization()
            
        except Exception as e:
            self.error_tracker.track_error(
                category=ErrorCategory.SUMMARIZATION,
                severity=ErrorSeverity.LOW,
                error=e,
                operation='_check_summarization_trigger'
            )
            logger.error(f"Error checking summarization trigger: {e}")
            # Don't raise - summarization is optional and shouldn't break main operations
    
    def _record_metric(self, operation: str, elapsed_ms: float, error: bool = False) -> None:
        """
        Record performance metrics for an operation.
        
        Args:
            operation: Name of the operation (e.g., 'create_memory')
            elapsed_ms: Time taken in milliseconds
            error: Whether the operation resulted in an error
        """
        if not self.config.enable_metrics:
            return
        
        if operation not in self._metrics:
            return
        
        metrics = self._metrics[operation]
        metrics['count'] += 1
        metrics['total_time_ms'] += elapsed_ms
        metrics['min_time_ms'] = min(metrics['min_time_ms'], elapsed_ms)
        metrics['max_time_ms'] = max(metrics['max_time_ms'], elapsed_ms)
        
        if error:
            metrics['errors'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all operations.
        
        Returns:
            Dictionary containing performance metrics including:
                - Operation counts
                - Average, min, max latencies
                - Error counts
                - System resource usage
        
        Example:
            >>> metrics = manager.get_performance_metrics()
            >>> print(f"Average create time: {metrics['create_memory']['avg_time_ms']:.2f}ms")
        """
        result = {}
        
        for operation, metrics in self._metrics.items():
            count = metrics['count']
            avg_time_ms = metrics['total_time_ms'] / count if count > 0 else 0
            
            result[operation] = {
                'count': count,
                'avg_time_ms': round(avg_time_ms, 2),
                'min_time_ms': round(metrics['min_time_ms'], 2) if count > 0 else 0,
                'max_time_ms': round(metrics['max_time_ms'], 2),
                'errors': metrics['errors'],
                'error_rate': round(metrics['errors'] / count * 100, 2) if count > 0 else 0
            }
        
        # Add system resource metrics
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            result['system_resources'] = {
                'memory_usage_mb': round(memory_info.rss / 1024 / 1024, 2),
                'memory_usage_percent': round(process.memory_percent(), 2),
                'cpu_percent': round(process.cpu_percent(interval=0.1), 2),
                'num_threads': process.num_threads(),
            }
        except ImportError:
            # psutil not available, skip system metrics
            logger.debug("psutil not available, skipping system resource metrics")
        except Exception as e:
            logger.warning(f"Failed to collect system resource metrics: {e}")
        
        return result
    
    def reset_performance_metrics(self) -> None:
        """
        Reset all performance metrics to zero.
        
        Useful for testing or when starting a new monitoring period.
        """
        for metrics in self._metrics.values():
            metrics['count'] = 0
            metrics['total_time_ms'] = 0
            metrics['min_time_ms'] = float('inf')
            metrics['max_time_ms'] = 0
            metrics['errors'] = 0
        
        logger.info("Performance metrics reset")
    
    def _perform_summarization(self) -> None:
        """
        Perform context summarization on redundant entries.
        
        This is a placeholder for the full summarization implementation.
        The actual implementation would:
        1. Query entries that are candidates for summarization
        2. Identify redundant entries
        3. Create summary entries
        4. Update parent references
        5. Optionally delete or archive original entries
        """
        logger.info("Performing context summarization...")
        
        try:
            # Query recent entries for summarization
            # Use a reasonable limit for querying entries
            query_limit = max(1000, self.config.summarization_threshold * 2)
            entries = self.storage.query_entries(limit=query_limit)
            
            # Need at least 2 entries to create a summary
            if len(entries) < 2:
                logger.debug("Not enough entries for summarization (need at least 2)")
                return
            
            # Identify redundant entries
            redundant_groups = self.summarizer.identify_redundant_entries(entries)
            
            if not redundant_groups:
                logger.debug("No redundant entries found")
                return
            
            # Create summaries for each group
            for summary_id, entry_ids in redundant_groups:
                # Get the entries to summarize
                entries_to_summarize = [e for e in entries if e.id in entry_ids]
                
                # Create summary entry (returns tuple of summary_entry and entry_ids_to_link)
                summary_entry, entry_ids_to_link = self.summarizer.summarize_entries(entries_to_summarize)
                
                # Store summary
                self.storage.create_entry(summary_entry)
                
                # Update original entries to reference the summary
                for entry_id in entry_ids_to_link:
                    self.storage.update_entry(entry_id, {'parent_id': summary_entry.id})
                
                logger.info(
                    f"Created summary {summary_entry.id} for {len(entry_ids_to_link)} entries"
                )
            
        except Exception as e:
            self.error_tracker.track_error(
                category=ErrorCategory.SUMMARIZATION,
                severity=ErrorSeverity.LOW,
                error=e,
                operation='_perform_summarization'
            )
            logger.error(f"Error performing summarization: {e}")
            # Don't raise - summarization is optional
