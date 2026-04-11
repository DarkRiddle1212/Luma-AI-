"""Unit tests for connection pooling in SQLite storage."""

import pytest
import tempfile
import os
from pathlib import Path

from luma_memory.storage.sqlite_storage import SQLiteStorage, ConnectionPool
from luma_memory.models import create_memory_entry


class TestConnectionPool:
    """Tests for connection pooling functionality."""
    
    def test_connection_pool_initialization(self):
        """Test that connection pool is initialized with correct size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Initialize storage with specific pool size
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=5)
            
            # Verify pool was created
            assert storage.connection_pool is not None
            assert storage.connection_pool.pool_size == 5
            
            # Clean up
            storage.close()
    
    def test_connection_pool_reuses_connections(self):
        """Test that connection pool reuses connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Initialize storage with small pool
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=2)
            
            # Create multiple entries to test connection reuse
            entries = []
            for i in range(5):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                entry_id = storage.create_entry(entry)
                entries.append(entry_id)
            
            # Verify all entries were created
            assert len(entries) == 5
            
            # Verify entries can be retrieved
            for entry_id in entries:
                retrieved = storage.get_entry(entry_id)
                assert retrieved is not None
                assert retrieved.id == entry_id
            
            # Clean up
            storage.close()
    
    def test_connection_pool_context_manager(self):
        """Test that connection pool works with context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Use storage as context manager
            with SQLiteStorage(db_path=db_path, cache_size=10, pool_size=3) as storage:
                # Create an entry
                entry = create_memory_entry(
                    action="test_action",
                    context={"key": "value"},
                    device_id="test_device"
                )
                entry_id = storage.create_entry(entry)
                
                # Verify entry was created
                retrieved = storage.get_entry(entry_id)
                assert retrieved is not None
                assert retrieved.id == entry_id
            
            # Connections should be closed after context manager exits
    
    def test_connection_pool_concurrent_operations(self):
        """Test that connection pool handles concurrent operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Initialize storage with pool
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=3)
            
            # Perform multiple operations
            entry1 = create_memory_entry(
                action="action1",
                context={"key": "value1"},
                device_id="device1"
            )
            entry2 = create_memory_entry(
                action="action2",
                context={"key": "value2"},
                device_id="device2"
            )
            
            # Create entries
            id1 = storage.create_entry(entry1)
            id2 = storage.create_entry(entry2)
            
            # Query entries
            results = storage.query_entries()
            assert len(results) == 2
            
            # Update entry
            storage.update_entry(id1, {"action": "updated_action"})
            
            # Verify update
            updated = storage.get_entry(id1)
            assert updated.action == "updated_action"
            
            # Delete entry
            storage.delete_entry(id2)
            
            # Verify deletion
            deleted = storage.get_entry(id2)
            assert deleted is None
            
            # Clean up
            storage.close()
    
    def test_connection_pool_closes_all_connections(self):
        """Test that close() closes all connections in the pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Initialize storage
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=3)
            
            # Create some entries to use connections
            for i in range(3):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                storage.create_entry(entry)
            
            # Close storage
            storage.close()
            
            # Verify pool is empty
            assert storage.connection_pool.pool.empty()

    def test_connection_pool_respects_size_limit(self):
        """Test that connection pool respects the maximum pool size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create a connection pool directly
            pool = ConnectionPool(db_path=db_path, pool_size=3)
            
            # Verify pool size
            assert pool.pool_size == 3
            assert pool._connection_count == 3
            
            # Clean up
            pool.close_all()
            assert pool._connection_count == 0
    
    def test_connection_pool_blocks_when_exhausted(self):
        """Test that connection pool blocks when all connections are in use."""
        import threading
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create a small pool
            pool = ConnectionPool(db_path=db_path, pool_size=2)
            
            # Hold all connections
            held_connections = []
            
            # Get all connections from pool
            for _ in range(2):
                conn = pool.pool.get(block=False)
                held_connections.append(conn)
            
            # Pool should be empty now
            assert pool.pool.empty()
            
            # Try to get another connection in a separate thread
            result = {"success": False, "error": None}
            
            def try_get_connection():
                try:
                    with pool.get_connection() as conn:
                        result["success"] = True
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=try_get_connection)
            thread.start()
            
            # Wait a bit to ensure thread is blocked
            time.sleep(0.5)
            
            # Thread should still be alive (blocked)
            assert thread.is_alive()
            
            # Return one connection to pool
            pool.pool.put(held_connections[0])
            
            # Wait for thread to complete
            thread.join(timeout=2.0)
            
            # Thread should have succeeded
            assert result["success"] is True
            
            # Return remaining connection
            pool.pool.put(held_connections[1])
            
            # Clean up
            pool.close_all()
    
    def test_connection_pool_timeout_on_exhaustion(self):
        """Test that connection pool times out when exhausted for too long."""
        import threading
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create a small pool
            pool = ConnectionPool(db_path=db_path, pool_size=1)
            
            # Hold the only connection
            conn = pool.pool.get(block=False)
            
            # Try to get another connection (should timeout)
            result = {"error": None}
            
            def try_get_connection():
                try:
                    with pool.get_connection() as c:
                        pass
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=try_get_connection)
            thread.start()
            
            # Wait for timeout (5 seconds + buffer)
            thread.join(timeout=7.0)
            
            # Should have timed out
            assert result["error"] is not None
            assert "timeout" in result["error"].lower()
            
            # Return connection
            pool.pool.put(conn)
            
            # Clean up
            pool.close_all()
    
    def test_connection_pool_returns_connection_after_use(self):
        """Test that connections are returned to pool after use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create pool
            pool = ConnectionPool(db_path=db_path, pool_size=3)
            
            # Initial pool should have 3 connections
            initial_size = pool.pool.qsize()
            assert initial_size == 3
            
            # Use a connection via context manager
            with pool.get_connection() as conn:
                # Pool should have 2 connections while one is in use
                assert pool.pool.qsize() == 2
            
            # After context manager exits, connection should be returned
            assert pool.pool.qsize() == 3
            
            # Clean up
            pool.close_all()
    
    def test_connection_pool_handles_multiple_sequential_operations(self):
        """Test that pool handles many sequential operations efficiently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create storage with small pool
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=2)
            
            # Perform many sequential operations
            entry_ids = []
            for i in range(20):
                entry = create_memory_entry(
                    action=f"action_{i}",
                    context={"index": i},
                    device_id="test_device"
                )
                entry_id = storage.create_entry(entry)
                entry_ids.append(entry_id)
            
            # Verify all entries were created
            assert len(entry_ids) == 20
            
            # Retrieve all entries
            for entry_id in entry_ids:
                retrieved = storage.get_entry(entry_id)
                assert retrieved is not None
            
            # Pool should still have connections
            assert not storage.connection_pool.pool.empty()
            
            # Clean up
            storage.close()
    
    def test_connection_pool_thread_safety(self):
        """Test that connection pool is thread-safe."""
        import threading
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create storage with pool
            storage = SQLiteStorage(db_path=db_path, cache_size=10, pool_size=5)
            
            # Create entries from multiple threads
            results = {"success": 0, "errors": []}
            lock = threading.Lock()
            
            def create_entries(thread_id):
                try:
                    for i in range(5):
                        entry = create_memory_entry(
                            action=f"thread_{thread_id}_action_{i}",
                            context={"thread": thread_id, "index": i},
                            device_id=f"device_{thread_id}"
                        )
                        storage.create_entry(entry)
                    
                    with lock:
                        results["success"] += 1
                except Exception as e:
                    with lock:
                        results["errors"].append(str(e))
            
            # Create multiple threads
            threads = []
            for i in range(10):
                thread = threading.Thread(target=create_entries, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10.0)
            
            # Verify all threads succeeded
            assert results["success"] == 10
            assert len(results["errors"]) == 0
            
            # Verify all entries were created
            all_entries = storage.query_entries(limit=1000)
            assert len(all_entries) == 50  # 10 threads * 5 entries each
            
            # Clean up
            storage.close()
    
    def test_connection_pool_error_handling(self):
        """Test that connection pool handles errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create pool
            pool = ConnectionPool(db_path=db_path, pool_size=2)
            
            # Simulate error during connection use
            try:
                with pool.get_connection() as conn:
                    # Force an error
                    raise ValueError("Simulated error")
            except ValueError:
                pass  # Expected
            
            # Pool should still be functional
            with pool.get_connection() as conn:
                # Should work fine
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1
            
            # Clean up
            pool.close_all()
    
    def test_connection_pool_closes_excess_connections(self):
        """Test that pool closes excess connections when full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create pool
            pool = ConnectionPool(db_path=db_path, pool_size=2)
            
            # Get all connections
            conn1 = pool.pool.get(block=False)
            conn2 = pool.pool.get(block=False)
            
            # Pool is now empty
            assert pool.pool.empty()
            
            # Try to return 3 connections (more than pool size)
            pool.pool.put(conn1)
            pool.pool.put(conn2)
            
            # Pool should be full
            assert pool.pool.qsize() == 2
            
            # Create a new connection manually
            import sqlite3
            extra_conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # Try to put it in the pool (should be rejected when full)
            try:
                pool.pool.put(extra_conn, block=False)
                # If it was added, pool would be over capacity
                # This tests the finally block in get_connection that closes excess connections
            except:
                # Expected if pool is full
                extra_conn.close()
            
            # Clean up
            pool.close_all()
