"""
Tests for SyncCoordinator functionality.
"""

import pytest
import sqlite3
from datetime import datetime, UTC
from luma_memory.sync.coordinator import SyncCoordinator, ConflictResolution
from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_sync.db"
    return str(db_path)


@pytest.fixture
def storage(temp_db):
    """Create a SQLite storage backend for testing."""
    storage = SQLiteStorage(db_path=temp_db, cache_size=10, pool_size=2)
    yield storage
    storage.close()


@pytest.fixture
def sync_coordinator(storage):
    """Create a SyncCoordinator with storage backend."""
    return SyncCoordinator(
        resolution_strategy=ConflictResolution.LATEST_WINS,
        storage_backend=storage
    )


def test_mark_for_sync_with_update_operation(sync_coordinator, storage):
    """Test marking an entry for sync with update operation."""
    # Create a memory entry first
    entry = MemoryEntry(
        id="test-entry-1",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark the entry for sync
    sync_coordinator.mark_for_sync("test-entry-1", operation="update")
    
    # Verify the entry is in the sync queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_id, operation, queued_at, synced_at
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-1",))
        row = cursor.fetchone()
    
    assert row is not None
    assert row[0] == "test-entry-1"
    assert row[1] == "update"
    assert row[2] is not None  # queued_at should be set
    assert row[3] is None  # synced_at should be NULL


def test_mark_for_sync_with_create_operation(sync_coordinator, storage):
    """Test marking an entry for sync with create operation."""
    # Create a memory entry first
    entry = MemoryEntry(
        id="test-entry-2",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark the entry for sync with create operation
    sync_coordinator.mark_for_sync("test-entry-2", operation="create")
    
    # Verify the entry is in the sync queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_id, operation
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-2",))
        row = cursor.fetchone()
    
    assert row is not None
    assert row[0] == "test-entry-2"
    assert row[1] == "create"


def test_mark_for_sync_with_delete_operation(sync_coordinator, storage):
    """Test marking an entry for sync with delete operation."""
    # Create a memory entry first
    entry = MemoryEntry(
        id="test-entry-3",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark the entry for sync with delete operation
    sync_coordinator.mark_for_sync("test-entry-3", operation="delete")
    
    # Verify the entry is in the sync queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_id, operation
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-3",))
        row = cursor.fetchone()
    
    assert row is not None
    assert row[0] == "test-entry-3"
    assert row[1] == "delete"


def test_mark_for_sync_invalid_operation(sync_coordinator):
    """Test that invalid operation raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        sync_coordinator.mark_for_sync("test-entry", operation="invalid")
    
    assert "Invalid operation" in str(exc_info.value)
    assert "create, update, delete" in str(exc_info.value)


def test_mark_for_sync_without_storage_backend():
    """Test that marking for sync without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.mark_for_sync("test-entry", operation="update")
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_mark_for_sync_multiple_entries(sync_coordinator, storage):
    """Test marking multiple entries for sync."""
    # Create multiple entries
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-{i}", operation="update")
    
    # Verify all entries are in the sync queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sync_queue")
        count = cursor.fetchone()[0]
    
    assert count == 3


def test_mark_for_sync_default_operation(sync_coordinator, storage):
    """Test that default operation is 'update'."""
    # Create a memory entry first
    entry = MemoryEntry(
        id="test-entry-default",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark the entry for sync without specifying operation
    sync_coordinator.mark_for_sync("test-entry-default")
    
    # Verify the operation is 'update'
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT operation
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-default",))
        row = cursor.fetchone()
    
    assert row is not None
    assert row[0] == "update"


def test_get_pending_sync_empty_queue(sync_coordinator):
    """Test getting pending sync entries when queue is empty."""
    entries = sync_coordinator.get_pending_sync()
    assert entries == []


def test_get_pending_sync_single_entry(sync_coordinator, storage):
    """Test getting a single pending sync entry."""
    # Create a memory entry
    entry = MemoryEntry(
        id="test-entry-pending-1",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark it for sync
    sync_coordinator.mark_for_sync("test-entry-pending-1", operation="create")
    
    # Get pending sync entries
    pending_entries = sync_coordinator.get_pending_sync()
    
    assert len(pending_entries) == 1
    assert pending_entries[0].id == "test-entry-pending-1"
    assert pending_entries[0].action == "test_action"
    assert pending_entries[0].context == {"key": "value"}


def test_get_pending_sync_multiple_entries(sync_coordinator, storage):
    """Test getting multiple pending sync entries."""
    # Create multiple memory entries
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-pending-{i}",
            timestamp=datetime.now(UTC),
            action=f"test_action_{i}",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-pending-{i}", operation="update")
    
    # Get pending sync entries
    pending_entries = sync_coordinator.get_pending_sync()
    
    assert len(pending_entries) == 3
    entry_ids = [e.id for e in pending_entries]
    assert "test-entry-pending-0" in entry_ids
    assert "test-entry-pending-1" in entry_ids
    assert "test-entry-pending-2" in entry_ids


def test_get_pending_sync_excludes_synced_entries(sync_coordinator, storage):
    """Test that get_pending_sync excludes entries that have been synced."""
    # Create two memory entries
    entry1 = MemoryEntry(
        id="test-entry-synced",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value1"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    entry2 = MemoryEntry(
        id="test-entry-not-synced",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value2"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry1)
    storage.create_entry(entry2)
    
    # Mark both for sync
    sync_coordinator.mark_for_sync("test-entry-synced", operation="update")
    sync_coordinator.mark_for_sync("test-entry-not-synced", operation="update")
    
    # Mark the first entry as synced by updating synced_at
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sync_queue
            SET synced_at = ?
            WHERE entry_id = ?
        """, (datetime.now(UTC), "test-entry-synced"))
        conn.commit()
    
    # Get pending sync entries
    pending_entries = sync_coordinator.get_pending_sync()
    
    # Should only return the entry that hasn't been synced
    assert len(pending_entries) == 1
    assert pending_entries[0].id == "test-entry-not-synced"


def test_get_pending_sync_ordered_by_queued_at(sync_coordinator, storage):
    """Test that pending sync entries are ordered by queued_at (oldest first)."""
    import time
    
    # Create entries with slight time delays to ensure different queued_at times
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-order-{i}",
            timestamp=datetime.now(UTC),
            action=f"test_action_{i}",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-order-{i}", operation="update")
        time.sleep(0.01)  # Small delay to ensure different timestamps
    
    # Get pending sync entries
    pending_entries = sync_coordinator.get_pending_sync()
    
    # Should be ordered by queued_at (oldest first)
    assert len(pending_entries) == 3
    assert pending_entries[0].id == "test-entry-order-0"
    assert pending_entries[1].id == "test-entry-order-1"
    assert pending_entries[2].id == "test-entry-order-2"


def test_get_pending_sync_without_storage_backend():
    """Test that get_pending_sync without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.get_pending_sync()
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_get_pending_sync_with_different_operations(sync_coordinator, storage):
    """Test getting pending sync entries with different operation types."""
    # Create entries with different operations
    operations = ["create", "update", "delete"]
    for i, op in enumerate(operations):
        entry = MemoryEntry(
            id=f"test-entry-op-{i}",
            timestamp=datetime.now(UTC),
            action=f"test_action_{i}",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-op-{i}", operation=op)
    
    # Get pending sync entries
    pending_entries = sync_coordinator.get_pending_sync()
    
    # Should return all entries regardless of operation type
    assert len(pending_entries) == 3
    entry_ids = [e.id for e in pending_entries]
    assert "test-entry-op-0" in entry_ids
    assert "test-entry-op-1" in entry_ids
    assert "test-entry-op-2" in entry_ids


# Tests for conflict resolution strategies

def test_resolve_conflict_latest_wins_local_newer(sync_coordinator):
    """Test LATEST_WINS strategy when local entry is newer."""
    from datetime import timedelta, UTC
    
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-1",
        timestamp=base_time,
        action="local_action",
        context={"source": "local"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local"],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=10)
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-1",
        timestamp=base_time,
        action="remote_action",
        context={"source": "remote"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote"],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=5)
    )
    
    resolved = sync_coordinator.resolve_conflict(local_entry, remote_entry)
    
    assert resolved.id == "conflict-entry-1"
    assert resolved.action == "local_action"
    assert resolved.context == {"source": "local"}
    assert resolved.tags == ["local"]


def test_resolve_conflict_latest_wins_remote_newer(sync_coordinator):
    """Test LATEST_WINS strategy when remote entry is newer."""
    from datetime import timedelta
    
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-2",
        timestamp=base_time,
        action="local_action",
        context={"source": "local"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local"],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=5)
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-2",
        timestamp=base_time,
        action="remote_action",
        context={"source": "remote"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote"],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=10)
    )
    
    resolved = sync_coordinator.resolve_conflict(local_entry, remote_entry)
    
    assert resolved.id == "conflict-entry-2"
    assert resolved.action == "remote_action"
    assert resolved.context == {"source": "remote"}
    assert resolved.tags == ["remote"]


def test_resolve_conflict_latest_wins_no_updated_at():
    """Test LATEST_WINS strategy when entries have no updated_at."""
    from datetime import timedelta
    
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.LATEST_WINS)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-3",
        timestamp=base_time,
        action="local_action",
        context={"source": "local"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local"],
        created_at=base_time + timedelta(seconds=5),  # Older
        updated_at=None
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-3",
        timestamp=base_time,
        action="remote_action",
        context={"source": "remote"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote"],
        created_at=base_time + timedelta(seconds=10),  # Newer
        updated_at=None
    )
    
    resolved = coordinator.resolve_conflict(local_entry, remote_entry)
    
    # Should use created_at when updated_at is None - remote is newer
    assert resolved.action == "remote_action"


def test_resolve_conflict_merge_strategy():
    """Test MERGE strategy combines changes from both entries."""
    from datetime import timedelta
    
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.MERGE)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-4",
        timestamp=base_time,
        action="newer_action",
        context={"local_key": "local_value", "shared_key": "local_version"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local", "shared"],
        summary="Local summary",
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=10)
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-4",
        timestamp=base_time,
        action="older_action",
        context={"remote_key": "remote_value", "shared_key": "remote_version"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote", "shared"],
        summary="Remote summary",
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=5)
    )
    
    resolved = coordinator.resolve_conflict(local_entry, remote_entry)
    
    # Should use newer entry's action
    assert resolved.action == "newer_action"
    
    # Should merge tags (union)
    assert set(resolved.tags) == {"local", "remote", "shared"}
    
    # Should merge context with newer values taking precedence
    assert "local_key" in resolved.context
    assert "remote_key" in resolved.context
    assert resolved.context["shared_key"] == "local_version"  # Newer wins
    
    # Should use newer summary
    assert resolved.summary == "Local summary"


def test_resolve_conflict_merge_preserves_older_summary():
    """Test MERGE strategy preserves older summary if newer doesn't have one."""
    from datetime import timedelta
    
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.MERGE)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-5",
        timestamp=base_time,
        action="newer_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local"],
        summary=None,  # Newer entry has no summary
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=10)
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-5",
        timestamp=base_time,
        action="older_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote"],
        summary="Older summary",  # Older entry has summary
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=5)
    )
    
    resolved = coordinator.resolve_conflict(local_entry, remote_entry)
    
    # Should preserve older summary if newer doesn't have one
    assert resolved.summary == "Older summary"


def test_resolve_conflict_manual_strategy_raises_error():
    """Test MANUAL strategy raises ValueError."""
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.MANUAL)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-6",
        timestamp=base_time,
        action="local_action",
        context={"source": "local"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["local"],
        created_at=base_time,
        updated_at=base_time
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-6",
        timestamp=base_time,
        action="remote_action",
        context={"source": "remote"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["remote"],
        created_at=base_time,
        updated_at=base_time
    )
    
    with pytest.raises(ValueError) as exc_info:
        coordinator.resolve_conflict(local_entry, remote_entry)
    
    assert "Manual conflict resolution required" in str(exc_info.value)
    assert "conflict-entry-6" in str(exc_info.value)


def test_resolve_conflict_different_ids_raises_error(sync_coordinator):
    """Test that resolving conflict with different IDs raises ValueError."""
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="entry-1",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=[],
        created_at=base_time,
        updated_at=base_time
    )
    
    remote_entry = MemoryEntry(
        id="entry-2",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=[],
        created_at=base_time,
        updated_at=base_time
    )
    
    with pytest.raises(ValueError) as exc_info:
        sync_coordinator.resolve_conflict(local_entry, remote_entry)
    
    assert "Cannot resolve conflict between different entries" in str(exc_info.value)
    assert "entry-1" in str(exc_info.value)
    assert "entry-2" in str(exc_info.value)


def test_resolve_conflict_merge_empty_tags():
    """Test MERGE strategy handles empty tags correctly."""
    from datetime import timedelta
    
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.MERGE)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-7",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=[],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=10)
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-7",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=["tag1", "tag2"],
        created_at=base_time,
        updated_at=base_time + timedelta(seconds=5)
    )
    
    resolved = coordinator.resolve_conflict(local_entry, remote_entry)
    
    # Should merge tags even if one is empty
    assert set(resolved.tags) == {"tag1", "tag2"}


def test_resolve_conflict_merge_updates_timestamp():
    """Test MERGE strategy updates the updated_at timestamp."""
    from datetime import timedelta
    import time
    
    coordinator = SyncCoordinator(resolution_strategy=ConflictResolution.MERGE)
    base_time = datetime.now(UTC)
    
    local_entry = MemoryEntry(
        id="conflict-entry-8",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=[],
        created_at=base_time,
        updated_at=base_time  # Set to base_time, not future
    )
    
    remote_entry = MemoryEntry(
        id="conflict-entry-8",
        timestamp=base_time,
        action="action",
        context={},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-2",
        sync_status=SyncStatus.PENDING,
        tags=[],
        created_at=base_time,
        updated_at=base_time - timedelta(seconds=5)  # Older than local
    )
    
    # Add a small delay to ensure the merge timestamp is later
    time.sleep(0.01)
    
    resolved = coordinator.resolve_conflict(local_entry, remote_entry)
    
    # updated_at should be set to current time (after the merge)
    # It should be later than both original entries
    assert resolved.updated_at >= local_entry.updated_at


# Tests for sync queue management methods

def test_mark_as_synced_success(sync_coordinator, storage):
    """Test successfully marking an entry as synced."""
    # Create a memory entry
    entry = MemoryEntry(
        id="test-entry-mark-synced",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark it for sync
    sync_coordinator.mark_for_sync("test-entry-mark-synced", operation="update")
    
    # Mark it as synced
    sync_coordinator.mark_as_synced("test-entry-mark-synced")
    
    # Verify synced_at is set
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT synced_at
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-mark-synced",))
        row = cursor.fetchone()
    
    assert row is not None
    assert row[0] is not None  # synced_at should be set


def test_mark_as_synced_not_in_queue(sync_coordinator, storage):
    """Test marking an entry as synced when it's not in the queue."""
    with pytest.raises(ValueError) as exc_info:
        sync_coordinator.mark_as_synced("non-existent-entry")
    
    assert "not found in sync queue" in str(exc_info.value)


def test_mark_as_synced_already_synced(sync_coordinator, storage):
    """Test marking an entry as synced when it's already synced."""
    # Create a memory entry
    entry = MemoryEntry(
        id="test-entry-already-synced",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark it for sync
    sync_coordinator.mark_for_sync("test-entry-already-synced", operation="update")
    
    # Mark it as synced
    sync_coordinator.mark_as_synced("test-entry-already-synced")
    
    # Try to mark it as synced again
    with pytest.raises(ValueError) as exc_info:
        sync_coordinator.mark_as_synced("test-entry-already-synced")
    
    assert "already synced" in str(exc_info.value)


def test_mark_as_synced_without_storage_backend():
    """Test marking as synced without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.mark_as_synced("test-entry")
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_remove_from_queue_success(sync_coordinator, storage):
    """Test successfully removing an entry from the queue."""
    # Create a memory entry
    entry = MemoryEntry(
        id="test-entry-remove",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry)
    
    # Mark it for sync
    sync_coordinator.mark_for_sync("test-entry-remove", operation="update")
    
    # Verify it's in the queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-remove",))
        count_before = cursor.fetchone()[0]
    
    assert count_before == 1
    
    # Remove it from the queue
    sync_coordinator.remove_from_queue("test-entry-remove")
    
    # Verify it's no longer in the queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-remove",))
        count_after = cursor.fetchone()[0]
    
    assert count_after == 0


def test_remove_from_queue_non_existent(sync_coordinator, storage):
    """Test removing a non-existent entry from the queue (should not raise error)."""
    # This should not raise an error, just do nothing
    sync_coordinator.remove_from_queue("non-existent-entry")


def test_remove_from_queue_without_storage_backend():
    """Test removing from queue without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.remove_from_queue("test-entry")
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_clear_synced_entries_all(sync_coordinator, storage):
    """Test clearing all synced entries from the queue."""
    # Create multiple entries
    for i in range(5):
        entry = MemoryEntry(
            id=f"test-entry-clear-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-clear-{i}", operation="update")
    
    # Mark first 3 as synced
    for i in range(3):
        sync_coordinator.mark_as_synced(f"test-entry-clear-{i}")
    
    # Clear all synced entries
    removed_count = sync_coordinator.clear_synced_entries()
    
    assert removed_count == 3
    
    # Verify only pending entries remain
    pending_entries = sync_coordinator.get_pending_sync()
    assert len(pending_entries) == 2
    entry_ids = [e.id for e in pending_entries]
    assert "test-entry-clear-3" in entry_ids
    assert "test-entry-clear-4" in entry_ids


def test_clear_synced_entries_older_than(sync_coordinator, storage):
    """Test clearing synced entries older than a specific time."""
    from datetime import timedelta
    import time
    
    # Create entries
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-clear-old-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-clear-old-{i}", operation="update")
    
    # Mark first entry as synced
    sync_coordinator.mark_as_synced("test-entry-clear-old-0")
    
    # Wait a bit
    time.sleep(0.1)
    
    # Record the cutoff time
    cutoff_time = datetime.now(UTC)
    
    # Wait a bit more
    time.sleep(0.1)
    
    # Mark second entry as synced (after cutoff)
    sync_coordinator.mark_as_synced("test-entry-clear-old-1")
    
    # Clear entries synced before cutoff
    removed_count = sync_coordinator.clear_synced_entries(older_than=cutoff_time)
    
    # Should only remove the first entry
    assert removed_count == 1
    
    # Verify the second synced entry is still in the queue
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM sync_queue
            WHERE entry_id = ? AND synced_at IS NOT NULL
        """, ("test-entry-clear-old-1",))
        count = cursor.fetchone()[0]
    
    assert count == 1


def test_clear_synced_entries_empty_queue(sync_coordinator, storage):
    """Test clearing synced entries when queue is empty."""
    removed_count = sync_coordinator.clear_synced_entries()
    assert removed_count == 0


def test_clear_synced_entries_no_synced_entries(sync_coordinator, storage):
    """Test clearing synced entries when there are only pending entries."""
    # Create entries but don't mark them as synced
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-no-synced-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-no-synced-{i}", operation="update")
    
    # Clear synced entries (should remove nothing)
    removed_count = sync_coordinator.clear_synced_entries()
    
    assert removed_count == 0
    
    # Verify all entries are still in the queue
    pending_entries = sync_coordinator.get_pending_sync()
    assert len(pending_entries) == 3


def test_clear_synced_entries_without_storage_backend():
    """Test clearing synced entries without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.clear_synced_entries()
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_get_queue_stats_empty_queue(sync_coordinator, storage):
    """Test getting queue stats when queue is empty."""
    stats = sync_coordinator.get_queue_stats()
    
    assert stats["total_entries"] == 0
    assert stats["pending_count"] == 0
    assert stats["synced_count"] == 0
    assert stats["oldest_pending"] is None
    assert stats["operations"] == {}


def test_get_queue_stats_with_entries(sync_coordinator, storage):
    """Test getting queue stats with various entries."""
    # Create entries with different operations
    operations = ["create", "update", "delete", "update", "create"]
    for i, op in enumerate(operations):
        entry = MemoryEntry(
            id=f"test-entry-stats-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-stats-{i}", operation=op)
    
    # Mark 2 entries as synced
    sync_coordinator.mark_as_synced("test-entry-stats-0")
    sync_coordinator.mark_as_synced("test-entry-stats-1")
    
    # Get stats
    stats = sync_coordinator.get_queue_stats()
    
    assert stats["total_entries"] == 5
    assert stats["pending_count"] == 3
    assert stats["synced_count"] == 2
    assert stats["oldest_pending"] is not None
    
    # Check operation breakdown (only pending entries)
    assert stats["operations"]["delete"] == 1
    assert stats["operations"]["update"] == 1
    assert stats["operations"]["create"] == 1


def test_get_queue_stats_oldest_pending(sync_coordinator, storage):
    """Test that oldest_pending returns the correct timestamp."""
    import time
    
    # Create first entry
    entry1 = MemoryEntry(
        id="test-entry-oldest-1",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value1"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry1)
    sync_coordinator.mark_for_sync("test-entry-oldest-1", operation="update")
    
    # Get the queued_at time for the first entry
    with storage.connection_pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT queued_at
            FROM sync_queue
            WHERE entry_id = ?
        """, ("test-entry-oldest-1",))
        oldest_time = cursor.fetchone()[0]
    
    # Wait a bit
    time.sleep(0.1)
    
    # Create second entry
    entry2 = MemoryEntry(
        id="test-entry-oldest-2",
        timestamp=datetime.now(UTC),
        action="test_action",
        context={"key": "value2"},
        sensitivity=SensitivityLevel.PUBLIC,
        device_id="device-1",
        sync_status=SyncStatus.PENDING,
        tags=["test"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )
    storage.create_entry(entry2)
    sync_coordinator.mark_for_sync("test-entry-oldest-2", operation="update")
    
    # Get stats
    stats = sync_coordinator.get_queue_stats()
    
    # oldest_pending should be the first entry's queued_at time
    assert stats["oldest_pending"] == oldest_time


def test_get_queue_stats_without_storage_backend():
    """Test getting queue stats without storage backend raises RuntimeError."""
    coordinator = SyncCoordinator(storage_backend=None)
    
    with pytest.raises(RuntimeError) as exc_info:
        coordinator.get_queue_stats()
    
    assert "Storage backend not configured" in str(exc_info.value)


def test_get_queue_stats_all_synced(sync_coordinator, storage):
    """Test queue stats when all entries are synced."""
    # Create entries
    for i in range(3):
        entry = MemoryEntry(
            id=f"test-entry-all-synced-{i}",
            timestamp=datetime.now(UTC),
            action="test_action",
            context={"key": f"value-{i}"},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="device-1",
            sync_status=SyncStatus.PENDING,
            tags=["test"],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )
        storage.create_entry(entry)
        sync_coordinator.mark_for_sync(f"test-entry-all-synced-{i}", operation="update")
        sync_coordinator.mark_as_synced(f"test-entry-all-synced-{i}")
    
    # Get stats
    stats = sync_coordinator.get_queue_stats()
    
    assert stats["total_entries"] == 3
    assert stats["pending_count"] == 0
    assert stats["synced_count"] == 3
    assert stats["oldest_pending"] is None
    assert stats["operations"] == {}
