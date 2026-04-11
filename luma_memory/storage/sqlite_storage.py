"""
SQLite storage backend implementation for Luma Memory Module.

This module provides a SQLite-based storage backend with connection pooling,
LRU caching, and transaction support for reliable memory entry persistence.
"""

import sqlite3
import json
import logging
import base64
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC, date
from pathlib import Path
from functools import lru_cache
from threading import Lock
from queue import Queue, Empty
from contextlib import contextmanager
from decimal import Decimal
from uuid import UUID

from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
from luma_memory.storage.backend import StorageBackend, StorageError


logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Connection pool for SQLite database connections.
    
    Manages a pool of reusable database connections to prevent
    resource exhaustion and improve performance.
    """
    
    def __init__(self, db_path: str, pool_size: int = 10):
        """
        Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum number of connections in the pool
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool: Queue = Queue(maxsize=pool_size)
        self.lock = Lock()
        self._connection_count = 0
        
        # Pre-create connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)
        
        logger.info(f"ConnectionPool initialized with {pool_size} connections")
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection.
        
        Returns:
            SQLite connection with row factory configured
        
        Raises:
            StorageError: If connection creation fails
        """
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.Error:
                pass
            with self.lock:
                self._connection_count += 1
            logger.debug(f"Created new database connection (total: {self._connection_count})")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to create database connection to {self.db_path}: {e}")
            raise StorageError(f"Failed to create database connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).
        
        Yields:
            SQLite connection from the pool
        
        Raises:
            StorageError: If no connection is available within timeout
        """
        conn = None
        try:
            # Try to get a connection from the pool (non-blocking)
            try:
                conn = self.pool.get(block=False)
                logger.debug("Retrieved connection from pool (non-blocking)")
            except Empty:
                # Pool is empty, create a new connection if under limit
                with self.lock:
                    if self._connection_count < self.pool_size:
                        logger.debug("Pool empty, creating new connection")
                        conn = self._create_connection()
                    else:
                        # Wait for a connection to become available
                        logger.debug("Pool full, waiting for available connection")
                        conn = self.pool.get(block=True, timeout=5.0)
            
            yield conn
            
        except Empty:
            logger.error("Connection pool timeout: no connections available after 5 seconds")
            raise StorageError("Connection pool timeout: no connections available")
        except (ValueError, StorageError):
            # Re-raise validation and storage errors without wrapping
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting connection from pool: {e}")
            raise StorageError(f"Failed to get database connection: {e}")
        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    self.pool.put(conn, block=False)
                    logger.debug("Returned connection to pool")
                except:
                    # Pool is full, close the connection
                    logger.debug("Pool full, closing excess connection")
                    conn.close()
                    with self.lock:
                        self._connection_count -= 1
    
    def close_all(self) -> None:
        """
        Close all connections in the pool.
        
        Logs the number of connections closed.
        """
        closed_count = 0
        while not self.pool.empty():
            try:
                conn = self.pool.get(block=False)
                conn.close()
                with self.lock:
                    self._connection_count -= 1
                closed_count += 1
            except Empty:
                break
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        
        logger.info(f"Closed {closed_count} database connections")


class LRUCache:
    """
    Simple LRU cache for frequently accessed memory entries.
    
    Thread-safe implementation with size limit.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of entries to cache
        """
        self.capacity = capacity
        self.cache: Dict[str, MemoryEntry] = {}
        self.access_order: List[str] = []
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[MemoryEntry]:
        """
        Get an entry from cache.
        
        Args:
            key: Entry ID
        
        Returns:
            MemoryEntry if found, None otherwise
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                logger.debug(f"Cache hit for entry {key}")
                return self.cache[key]
            logger.debug(f"Cache miss for entry {key}")
            return None
    
    def put(self, key: str, value: MemoryEntry) -> None:
        """
        Put an entry in cache.
        
        Args:
            key: Entry ID
            value: MemoryEntry to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.access_order.remove(key)
                logger.debug(f"Updated cache entry {key}")
            elif len(self.cache) >= self.capacity:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
                logger.debug(f"Evicted LRU entry {lru_key} from cache")
            
            self.cache[key] = value
            self.access_order.append(key)
            logger.debug(f"Added entry {key} to cache (size: {len(self.cache)}/{self.capacity})")
    
    def invalidate(self, key: str) -> None:
        """
        Remove an entry from cache.
        
        Args:
            key: Entry ID to remove
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
                logger.debug(f"Invalidated cache entry {key}")
            else:
                logger.debug(f"Cache entry {key} not found for invalidation")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            cache_size = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            logger.info(f"Cleared {cache_size} entries from cache")


class SQLiteStorage(StorageBackend):
    @staticmethod
    def _to_json_safe(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            return base64.b64encode(bytes(obj)).decode('ascii')
        if isinstance(obj, dict):
            return {k: SQLiteStorage._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [SQLiteStorage._to_json_safe(v) for v in obj]
        return str(obj)
    """
    SQLite implementation of storage backend.
    
    Provides persistent storage with connection pooling, caching,
    and transaction support for reliable memory entry management.
    """
    
    def __init__(self, db_path: str, cache_size: int = 1000, pool_size: int = 10):
        """
        Initialize SQLite storage backend.
        
        Args:
            db_path: Path to SQLite database file
            cache_size: Size of LRU cache for entries
            pool_size: Size of connection pool
        
        Raises:
            StorageError: If initialization fails
        """
        self.db_path = db_path
        self.cache = LRUCache(cache_size)
        self.lock = Lock()
        
        # Performance metrics
        self._metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'total_inserts': 0,
            'total_updates': 0,
            'total_deletes': 0,
        }
        
        logger.info(f"Initializing SQLiteStorage with db_path={db_path}, cache_size={cache_size}, pool_size={pool_size}")
        
        try:
            # Ensure database directory exists
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Database directory ensured: {db_dir}")
        except Exception as e:
            logger.error(f"Failed to create database directory {db_dir}: {e}")
            raise StorageError(f"Failed to create database directory: {e}")
        
        try:
            # Initialize connection pool
            self.connection_pool = ConnectionPool(db_path, pool_size)
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise StorageError(f"Failed to initialize connection pool: {e}")
        
        # Initialize database schema
        self._init_database()
        
        logger.info(f"SQLiteStorage initialized successfully")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all connections."""
        self.close()
        return False
    
    def close(self) -> None:
        """Close all connections in the pool and clean up resources."""
        self.connection_pool.close_all()
        logger.info("SQLiteStorage closed")
    
    def __del__(self):
        """Destructor to ensure connections are closed."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error during SQLiteStorage cleanup: {e}")
    
    def _init_database(self) -> None:
        """
        Initialize database schema if not exists.
        
        Raises:
            StorageError: If schema initialization fails
        """
        logger.info("Initializing database schema")
        with self.connection_pool.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Create memory_entries table
                logger.debug("Creating memory_entries table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id TEXT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        action TEXT NOT NULL,
                        context_json TEXT NOT NULL,
                        sensitivity TEXT NOT NULL,
                        device_id TEXT NOT NULL,
                        sync_status TEXT NOT NULL,
                        tags_json TEXT,
                        summary TEXT,
                        parent_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (parent_id) REFERENCES memory_entries(id)
                    )
                """)
                
                # Create indexes for query performance
                logger.debug("Creating database indexes")
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON memory_entries(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_device_id 
                    ON memory_entries(device_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sync_status 
                    ON memory_entries(sync_status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tags 
                    ON memory_entries(tags_json)
                """)
                
                # Create encryption_keys table
                logger.debug("Creating encryption_keys table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS encryption_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_hash TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        rotated_at DATETIME,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Create sync_queue table for future use
                logger.debug("Creating sync_queue table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sync_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entry_id TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        queued_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        synced_at DATETIME,
                        FOREIGN KEY (entry_id) REFERENCES memory_entries(id)
                    )
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
                raise StorageError(f"Database initialization failed: {e}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Unexpected error during database initialization: {e}", exc_info=True)
                raise StorageError(f"Database initialization failed: {e}")

    
    def create_entry(self, entry: MemoryEntry) -> str:
        """
        Store a new memory entry and return its ID.
        
        Args:
            entry: The MemoryEntry instance to store
        
        Returns:
            The ID of the stored entry
        
        Raises:
            ValueError: If the entry is invalid or already exists
            StorageError: If the storage operation fails
        """
        logger.debug(f"Creating entry {entry.id}")
        
        # Validate entry
        is_valid, error_msg = entry.validate()
        if not is_valid:
            logger.warning(f"Invalid memory entry {entry.id}: {error_msg}")
            raise ValueError(f"Invalid memory entry: {error_msg}")
        
        self._metrics['total_inserts'] += 1
        
        with self.connection_pool.get_connection() as conn:
            try:
                with self.lock:
                    cursor = conn.cursor()
                    
                    # Check if entry already exists
                    cursor.execute("SELECT id FROM memory_entries WHERE id = ?", (entry.id,))
                    if cursor.fetchone():
                        logger.warning(f"Attempted to create duplicate entry {entry.id}")
                        raise ValueError(f"Entry with id {entry.id} already exists")
                    
                    # Insert entry
                    cursor.execute("""
                        INSERT INTO memory_entries (
                            id, timestamp, action, context_json, sensitivity,
                            device_id, sync_status, tags_json, summary, parent_id,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.id,
                        entry.timestamp.isoformat(),
                        entry.action,
                        json.dumps(self._to_json_safe(entry.context)),
                        entry.sensitivity.value,
                        entry.device_id,
                        entry.sync_status.value,
                        json.dumps(entry.tags),
                        entry.summary,
                        entry.parent_id,
                        entry.created_at.isoformat() if entry.created_at else None,
                        entry.updated_at.isoformat() if entry.updated_at else None
                    ))
                    
                    conn.commit()
                    
                    # Cache the entry
                    self.cache.put(entry.id, entry)
                    
                    logger.info(f"Successfully created entry {entry.id} (action: {entry.action}, sensitivity: {entry.sensitivity.value})")
                    return entry.id
                    
            except ValueError:
                # Re-raise validation errors
                raise
            except sqlite3.IntegrityError as e:
                conn.rollback()
                logger.error(f"Integrity constraint violation creating entry {entry.id}: {e}")
                raise StorageError(f"Failed to create entry due to integrity constraint: {e}")
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Database error creating entry {entry.id}: {e}", exc_info=True)
                raise StorageError(f"Failed to create entry: {e}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Unexpected error creating entry {entry.id}: {e}", exc_info=True)
                raise StorageError(f"Failed to create entry: {e}")
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by ID.
        
        Args:
            entry_id: The unique identifier of the entry to retrieve
        
        Returns:
            The MemoryEntry if found, None otherwise
        
        Raises:
            StorageError: If the retrieval operation fails
        """
        logger.debug(f"Retrieving entry {entry_id}")
        
        # Check cache first
        cached_entry = self.cache.get(entry_id)
        if cached_entry:
            self._metrics['cache_hits'] += 1
            logger.info(f"Retrieved entry {entry_id} from cache")
            return cached_entry
        
        self._metrics['cache_misses'] += 1
        self._metrics['total_queries'] += 1
        
        with self.connection_pool.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM memory_entries WHERE id = ?
                """, (entry_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.info(f"Entry {entry_id} not found in database")
                    return None
                
                entry = self._row_to_entry(row)
                
                # Cache the entry
                self.cache.put(entry_id, entry)
                
                logger.info(f"Retrieved entry {entry_id} from database (action: {entry.action})")
                return entry
                    
            except sqlite3.Error as e:
                logger.error(f"Database error retrieving entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to retrieve entry: {e}")
            except Exception as e:
                logger.error(f"Unexpected error retrieving entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to retrieve entry: {e}")
    
    def query_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        action_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """
        Query memory entries with filters.
        
        Returns entries in reverse chronological order (newest first).
        
        Args:
            start_time: Optional start of time range filter (inclusive)
            end_time: Optional end of time range filter (inclusive)
            tags: Optional list of tags to filter by
            action_type: Optional action type to filter by (partial match)
            limit: Maximum number of entries to return
            offset: Number of entries to skip for pagination
        
        Returns:
            List of MemoryEntry instances matching the filters
        
        Raises:
            ValueError: If filter parameters are invalid
            StorageError: If the query operation fails
        """
        logger.debug(f"Querying entries with filters: start_time={start_time}, end_time={end_time}, "
                    f"tags={tags}, action_type={action_type}, limit={limit}, offset={offset}")
        
        # Validate parameters
        if limit <= 0:
            logger.warning(f"Invalid limit value: {limit}")
            raise ValueError("limit must be positive")
        if offset < 0:
            logger.warning(f"Invalid offset value: {offset}")
            raise ValueError("offset must be non-negative")
        
        self._metrics['total_queries'] += 1
        
        with self.connection_pool.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT * FROM memory_entries WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                if action_type:
                    query += " AND action LIKE ?"
                    params.append(f"%{action_type}%")
                
                if tags:
                    # Filter by tags - entry must have at least one matching tag
                    tag_conditions = []
                    for tag in tags:
                        tag_conditions.append("tags_json LIKE ?")
                        params.append(f'%"{tag}"%')
                    query += f" AND ({' OR '.join(tag_conditions)})"
                
                # Order by timestamp descending (newest first), then by id descending for deterministic ordering
                query += " ORDER BY timestamp DESC, id DESC"
                
                # Add pagination
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                logger.debug(f"Executing query: {query}")
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                entries = [self._row_to_entry(row) for row in rows]
                
                logger.info(f"Query returned {len(entries)} entries (limit={limit}, offset={offset})")
                return entries
                    
            except ValueError:
                # Re-raise validation errors
                raise
            except sqlite3.Error as e:
                logger.error(f"Database error querying entries: {e}", exc_info=True)
                raise StorageError(f"Failed to query entries: {e}")
            except Exception as e:
                logger.error(f"Unexpected error querying entries: {e}", exc_info=True)
                raise StorageError(f"Failed to query entries: {e}")
    
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: The unique identifier of the entry to update
            updates: Dictionary of field names and new values
        
        Returns:
            True if the entry was updated, False if not found
        
        Raises:
            ValueError: If the updates are invalid
            StorageError: If the update operation fails
        """
        logger.debug(f"Updating entry {entry_id} with fields: {list(updates.keys())}")
        
        if not updates:
            logger.warning("Attempted to update entry with empty updates dictionary")
            raise ValueError("updates dictionary cannot be empty")
        
        # Validate update fields
        allowed_fields = {
            'action', 'context', 'sensitivity', 'sync_status',
            'tags', 'summary', 'parent_id'
        }
        
        invalid_fields = set(updates.keys()) - allowed_fields
        if invalid_fields:
            logger.warning(f"Attempted to update invalid fields: {invalid_fields}")
            raise ValueError(f"Cannot update fields: {invalid_fields}")
        
        self._metrics['total_updates'] += 1
        
        with self.connection_pool.get_connection() as conn:
            try:
                with self.lock:
                    cursor = conn.cursor()
                    
                    # Check if entry exists
                    cursor.execute("SELECT id FROM memory_entries WHERE id = ?", (entry_id,))
                    if not cursor.fetchone():
                        logger.info(f"Entry {entry_id} not found for update")
                        return False
                    
                    # Build update query
                    set_clauses = []
                    params = []
                    
                    for field, value in updates.items():
                        if field == 'context':
                            set_clauses.append("context_json = ?")
                            params.append(json.dumps(SQLiteStorage._to_json_safe(value)))
                        elif field == 'tags':
                            set_clauses.append("tags_json = ?")
                            params.append(json.dumps(SQLiteStorage._to_json_safe(value)))
                        elif field == 'sensitivity':
                            set_clauses.append("sensitivity = ?")
                            if isinstance(value, SensitivityLevel):
                                params.append(value.value)
                            else:
                                params.append(value)
                        elif field == 'sync_status':
                            set_clauses.append("sync_status = ?")
                            if isinstance(value, SyncStatus):
                                params.append(value.value)
                            else:
                                params.append(value)
                        else:
                            set_clauses.append(f"{field} = ?")
                            params.append(value)
                    
                    # Update updated_at timestamp
                    set_clauses.append("updated_at = ?")
                    now = datetime.now(UTC) if hasattr(datetime, 'UTC') else datetime.now(UTC)
                    params.append(now.isoformat())
                    
                    params.append(entry_id)
                    
                    query = f"UPDATE memory_entries SET {', '.join(set_clauses)} WHERE id = ?"
                    cursor.execute(query, params)
                    
                    conn.commit()
                    
                    # Invalidate cache
                    self.cache.invalidate(entry_id)
                    
                    logger.info(f"Successfully updated entry {entry_id}")
                    return True
                    
            except ValueError:
                # Re-raise validation errors
                raise
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Database error updating entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to update entry: {e}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Unexpected error updating entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to update entry: {e}")
    
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: The unique identifier of the entry to delete
        
        Returns:
            True if the entry was deleted, False if not found
        
        Raises:
            StorageError: If the delete operation fails
        """
        logger.debug(f"Deleting entry {entry_id}")
        
        self._metrics['total_deletes'] += 1
        
        with self.connection_pool.get_connection() as conn:
            try:
                with self.lock:
                    cursor = conn.cursor()
                    
                    # Check if entry exists
                    cursor.execute("SELECT id FROM memory_entries WHERE id = ?", (entry_id,))
                    if not cursor.fetchone():
                        logger.info(f"Entry {entry_id} not found for deletion")
                        return False
                    
                    # Delete entry
                    cursor.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
                    
                    conn.commit()
                    
                    # Invalidate cache
                    self.cache.invalidate(entry_id)
                    
                    logger.info(f"Successfully deleted entry {entry_id}")
                    return True
                    
            except sqlite3.IntegrityError as e:
                conn.rollback()
                logger.error(f"Integrity constraint violation deleting entry {entry_id}: {e}")
                raise StorageError(f"Failed to delete entry due to integrity constraint: {e}")
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Database error deleting entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to delete entry: {e}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Unexpected error deleting entry {entry_id}: {e}", exc_info=True)
                raise StorageError(f"Failed to delete entry: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Return storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        
        Raises:
            StorageError: If the stats operation fails
        """
        logger.debug("Retrieving storage statistics")
        
        with self.connection_pool.get_connection() as conn:
            try:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute("SELECT COUNT(*) as count FROM memory_entries")
                total_entries = cursor.fetchone()['count']
                
                # Storage size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                storage_size = cursor.fetchone()['size']
                
                # Oldest and newest entries
                cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM memory_entries")
                row = cursor.fetchone()
                oldest_entry = datetime.fromisoformat(row['oldest']) if row['oldest'] else None
                newest_entry = datetime.fromisoformat(row['newest']) if row['newest'] else None
                
                # Entries by sensitivity
                cursor.execute("""
                    SELECT sensitivity, COUNT(*) as count 
                    FROM memory_entries 
                    GROUP BY sensitivity
                """)
                entries_by_sensitivity = {row['sensitivity']: row['count'] for row in cursor.fetchall()}
                
                # Entries by sync status
                cursor.execute("""
                    SELECT sync_status, COUNT(*) as count 
                    FROM memory_entries 
                    GROUP BY sync_status
                """)
                entries_by_sync_status = {row['sync_status']: row['count'] for row in cursor.fetchall()}
                
                # Calculate cache hit rate
                total_cache_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
                cache_hit_rate = (self._metrics['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
                
                stats = {
                    'total_entries': total_entries,
                    'storage_size_bytes': storage_size,
                    'oldest_entry': oldest_entry,
                    'newest_entry': newest_entry,
                    'entries_by_sensitivity': entries_by_sensitivity,
                    'entries_by_sync_status': entries_by_sync_status,
                    'storage_metrics': {
                        'cache_hits': self._metrics['cache_hits'],
                        'cache_misses': self._metrics['cache_misses'],
                        'cache_hit_rate': round(cache_hit_rate, 2),
                        'total_queries': self._metrics['total_queries'],
                        'total_inserts': self._metrics['total_inserts'],
                        'total_updates': self._metrics['total_updates'],
                        'total_deletes': self._metrics['total_deletes'],
                    }
                }
                
                logger.info(f"Retrieved storage stats: {total_entries} entries, {storage_size} bytes")
                return stats
                    
            except sqlite3.Error as e:
                logger.error(f"Database error getting storage stats: {e}", exc_info=True)
                raise StorageError(f"Failed to get storage stats: {e}")
            except Exception as e:
                logger.error(f"Unexpected error getting storage stats: {e}", exc_info=True)
                raise StorageError(f"Failed to get storage stats: {e}")
    
    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """
        Convert a database row to a MemoryEntry instance.
        
        Args:
            row: SQLite row object
        
        Returns:
            MemoryEntry instance
        
        Raises:
            StorageError: If row conversion fails
        """
        try:
            return MemoryEntry(
                id=row['id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                action=row['action'],
                context=json.loads(row['context_json']),
                sensitivity=SensitivityLevel(row['sensitivity']),
                device_id=row['device_id'],
                sync_status=SyncStatus(row['sync_status']),
                tags=json.loads(row['tags_json']) if row['tags_json'] else [],
                summary=row['summary'],
                parent_id=row['parent_id'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to convert database row to MemoryEntry: {e}", exc_info=True)
            raise StorageError(f"Failed to convert database row to MemoryEntry: {e}")
