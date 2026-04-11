"""
Storage backends for memory persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
import json
import os
from pathlib import Path

from .models import MemoryEntry, MemoryType


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry. Returns True if successful."""
        pass
    
    @abstractmethod
    def retrieve(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Retrieve memory entries with optional filters."""
        pass
    
    @abstractmethod
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        pass
    
    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry. Returns True if successful."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the storage backend and cleanup resources."""
        pass


class SQLiteStorage(StorageBackend):
    """SQLite-based storage backend for production use."""
    
    def __init__(self, db_path: str = "luma_memory.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                entry_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                tags TEXT,
                encrypted INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON memories(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type 
            ON memories(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source 
            ON memories(source)
        """)
        
        self.conn.commit()
    
    def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in SQLite."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO memories 
                (entry_id, content, memory_type, source, timestamp, metadata, tags, encrypted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.content,
                entry.memory_type.value,
                entry.source,
                entry.timestamp.isoformat(),
                json.dumps(entry.metadata),
                json.dumps(entry.tags),
                1 if entry.encrypted else 0
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    def retrieve(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Retrieve memory entries with filters."""
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        entries = []
        for row in cursor.fetchall():
            entry_dict = dict(row)
            entry_dict['metadata'] = json.loads(entry_dict['metadata'])
            entry_dict['tags'] = json.loads(entry_dict['tags'])
            entry_dict['encrypted'] = bool(entry_dict['encrypted'])
            entries.append(MemoryEntry.from_dict(entry_dict))
        
        # Filter by tags if specified (post-query filtering)
        if tags:
            entries = [e for e in entries if any(tag in e.tags for tag in tags)]
        
        return entries
    
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE entry_id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if row:
            entry_dict = dict(row)
            entry_dict['metadata'] = json.loads(entry_dict['metadata'])
            entry_dict['tags'] = json.loads(entry_dict['tags'])
            entry_dict['encrypted'] = bool(entry_dict['encrypted'])
            return MemoryEntry.from_dict(entry_dict)
        
        return None
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM memories WHERE entry_id = ?", (entry_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


class JSONStorage(StorageBackend):
    """JSON file-based storage backend for lightweight/testing use."""
    
    def __init__(self, file_path: str = "luma_memory.json"):
        """
        Initialize JSON storage.
        
        Args:
            file_path: Path to the JSON file
        """
        self.file_path = file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create the JSON file if it doesn't exist."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)
    
    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all entries from the JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_all(self, entries: List[Dict[str, Any]]):
        """Write all entries to the JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in JSON file."""
        try:
            entries = self._read_all()
            entries.append(entry.to_dict())
            self._write_all(entries)
            return True
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    def retrieve(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        memory_type: Optional[MemoryType] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Retrieve memory entries with filters."""
        entries = self._read_all()
        results = []
        
        for entry_dict in entries:
            entry = MemoryEntry.from_dict(entry_dict)
            
            # Apply filters
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            if source and entry.source != source:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            results.append(entry)
        
        # Sort by timestamp descending and apply limit
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]
    
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        entries = self._read_all()
        for entry_dict in entries:
            if entry_dict['entry_id'] == entry_id:
                return MemoryEntry.from_dict(entry_dict)
        return None
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            entries = self._read_all()
            original_count = len(entries)
            entries = [e for e in entries if e['entry_id'] != entry_id]
            
            if len(entries) < original_count:
                self._write_all(entries)
                return True
            return False
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def close(self):
        """No cleanup needed for JSON storage."""
        pass
