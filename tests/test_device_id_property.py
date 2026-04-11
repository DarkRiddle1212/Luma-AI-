"""
Property-Based Test for Device ID Application

This module implements property-based tests using Hypothesis to verify
that the SQLiteMemoryAdapter correctly attaches device_id to stored memories.

Feature: memory-write-strategy-session-management
Property 16: Device_id attachment
Validates: Requirements 6.3, 8.4
"""

import pytest
import tempfile
import os
from hypothesis import given, strategies as st, settings, HealthCheck
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.sqlite_storage import SQLiteStorage


# ============================================================================
# Property 16: Device_id Attachment
# ============================================================================

# Feature: memory-write-strategy-session-management, Property 16: Device_id attachment
@given(
    device_id=st.text(min_size=3, max_size=50, alphabet=st.characters(
        whitelist_categories=['Ll', 'Lu', 'Nd', 'Pd'],
        whitelist_characters=['-', '_']
    )),
    content=st.text(min_size=10, max_size=200),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_property_16_device_id_attachment(device_id, content, category):
    """
    Property: For any memory stored through an adapter configured with device_id,
    the memory should have that device_id in its metadata.
    
    **Validates: Requirements 6.3, 8.4**
    
    This test verifies that:
    1. The adapter accepts device_id configuration parameter
    2. When storing a memory, the configured device_id is attached
    3. The device_id is retrievable from the stored memory
    4. The device_id matches the configured value exactly
    """
    # Create a temporary database for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        
        # Initialize memory components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        
        # Create adapter with specific device_id
        adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager,
            device_id=device_id
        )
        
        try:
            # Store a memory
            memory_id = adapter.store(
                content=content,
                metadata={"category": category}
            )
            
            # Retrieve the stored memory directly from MemoryManager
            # to verify device_id was attached
            entries = memory_manager.query_memories(limit=1)
            
            # Property 1: Memory should be stored
            assert len(entries) > 0, "Memory should be stored"
            
            # Property 2: The stored memory should have the configured device_id
            stored_entry = entries[0]
            assert hasattr(stored_entry, 'device_id'), "Stored entry should have device_id attribute"
            assert stored_entry.device_id == device_id, \
                f"Stored memory should have device_id={device_id}, got {stored_entry.device_id}"
            
            # Property 3: The memory_id should match
            assert stored_entry.id == memory_id, \
                f"Stored memory ID should match returned ID"
        
        finally:
            adapter.close()


# Feature: memory-write-strategy-session-management, Property 16: Device_id attachment
@given(
    device_id=st.text(min_size=3, max_size=50, alphabet=st.characters(
        whitelist_categories=['Ll', 'Lu', 'Nd', 'Pd'],
        whitelist_characters=['-', '_']
    )),
    content=st.text(min_size=10, max_size=200),
    num_memories=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_property_16_device_id_consistent_across_memories(device_id, content, num_memories):
    """
    Property: For any adapter configured with device_id, all memories stored
    through that adapter should have the same device_id.
    
    **Validates: Requirements 6.3, 8.4**
    
    This test verifies that:
    1. Multiple memories stored through the same adapter all get the same device_id
    2. The device_id is consistently applied across all storage operations
    3. The device_id configuration is persistent for the adapter instance
    """
    # Create a temporary database for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        
        # Initialize memory components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        
        # Create adapter with specific device_id
        adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager,
            device_id=device_id
        )
        
        try:
            # Store multiple memories
            memory_ids = []
            for i in range(num_memories):
                memory_id = adapter.store(
                    content=f"{content} - memory {i}",
                    metadata={"category": f"test_{i}"}
                )
                memory_ids.append(memory_id)
            
            # Retrieve all stored memories
            entries = memory_manager.query_memories(limit=num_memories * 2)
            
            # Property 1: All memories should be stored
            assert len(entries) >= num_memories, \
                f"Should have stored {num_memories} memories, found {len(entries)}"
            
            # Property 2: All memories should have the same device_id
            for entry in entries[:num_memories]:
                assert hasattr(entry, 'device_id'), "Entry should have device_id attribute"
                assert entry.device_id == device_id, \
                    f"All memories should have device_id={device_id}, got {entry.device_id}"
        
        finally:
            adapter.close()


# Feature: memory-write-strategy-session-management, Property 16: Device_id attachment
@given(
    content=st.text(min_size=10, max_size=200),
    category=st.text(min_size=3, max_size=5)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_property_16_default_device_id_when_not_configured(content, category):
    """
    Property: For any memory stored through an adapter without explicit device_id
    configuration, the memory should have the default device_id ("reasoning-engine").
    
    **Validates: Requirements 8.4**
    
    This test verifies that:
    1. When device_id is not provided, a default value is used
    2. The default device_id is "reasoning-engine"
    3. The default device_id is consistently applied
    """
    # Create a temporary database for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        
        # Initialize memory components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        
        # Create adapter WITHOUT device_id (should use default)
        adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager
            # device_id not provided
        )
        
        try:
            # Store a memory
            memory_id = adapter.store(
                content=content,
                metadata={"category": category}
            )
            
            # Retrieve the stored memory
            entries = memory_manager.query_memories(limit=1)
            
            # Property 1: Memory should be stored
            assert len(entries) > 0, "Memory should be stored"
            
            # Property 2: The stored memory should have the default device_id
            stored_entry = entries[0]
            assert hasattr(stored_entry, 'device_id'), "Stored entry should have device_id attribute"
            assert stored_entry.device_id == "reasoning-engine", \
                f"Default device_id should be 'reasoning-engine', got {stored_entry.device_id}"
        
        finally:
            adapter.close()


# Feature: memory-write-strategy-session-management, Property 16: Device_id attachment
@given(
    device_id=st.text(min_size=3, max_size=50, alphabet=st.characters(
        whitelist_categories=['Ll', 'Lu', 'Nd', 'Pd'],
        whitelist_characters=['-', '_']
    )),
    content=st.text(min_size=10, max_size=200),
    metadata_device_id=st.text(min_size=3, max_size=50, alphabet=st.characters(
        whitelist_categories=['Ll', 'Lu', 'Nd', 'Pd'],
        whitelist_characters=['-', '_']
    ))
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_property_16_adapter_device_id_overrides_metadata(device_id, content, metadata_device_id):
    """
    Property: For any memory stored through an adapter configured with device_id,
    the adapter's device_id should be used even if device_id is provided in metadata.
    
    **Validates: Requirements 6.3, 8.4**
    
    This test verifies that:
    1. The adapter's configured device_id takes precedence
    2. device_id in metadata is ignored
    3. The adapter configuration is authoritative for device_id
    """
    # Skip if device_ids are the same (no override to test)
    if device_id == metadata_device_id:
        return
    
    # Create a temporary database for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        
        # Initialize memory components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        
        # Create adapter with specific device_id
        adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager,
            device_id=device_id
        )
        
        try:
            # Store a memory with device_id in metadata (should be ignored)
            memory_id = adapter.store(
                content=content,
                metadata={
                    "category": "test",
                    "device_id": metadata_device_id  # This should be ignored
                }
            )
            
            # Retrieve the stored memory
            entries = memory_manager.query_memories(limit=1)
            
            # Property 1: Memory should be stored
            assert len(entries) > 0, "Memory should be stored"
            
            # Property 2: The stored memory should have the adapter's device_id, not metadata's
            stored_entry = entries[0]
            assert hasattr(stored_entry, 'device_id'), "Stored entry should have device_id attribute"
            assert stored_entry.device_id == device_id, \
                f"Adapter device_id should override metadata device_id. " \
                f"Expected {device_id}, got {stored_entry.device_id}"
            assert stored_entry.device_id != metadata_device_id, \
                f"Metadata device_id should be ignored. " \
                f"Got {stored_entry.device_id}, should not be {metadata_device_id}"
        
        finally:
            adapter.close()


# Feature: memory-write-strategy-session-management, Property 16: Device_id attachment
@given(
    device_id=st.text(min_size=1, max_size=100, alphabet=st.characters(
        min_codepoint=32, max_codepoint=126  # Printable ASCII
    )),
    content=st.text(min_size=10, max_size=200)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.property_test
def test_property_16_device_id_accepts_various_formats(device_id, content):
    """
    Property: For any valid string device_id, the adapter should accept it and
    attach it to stored memories without modification.
    
    **Validates: Requirements 6.3, 8.4**
    
    This test verifies that:
    1. Various device_id formats are accepted (UUIDs, names, IPs, etc.)
    2. The device_id is stored exactly as provided (no normalization)
    3. Special characters in device_id are preserved
    """
    # Skip empty or whitespace-only device_ids
    if not device_id or not device_id.strip():
        return
    
    # Create a temporary database for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        
        # Initialize memory components
        storage = SQLiteStorage(db_path=db_path)
        memory_manager = MemoryManager(storage=storage)
        
        # Create adapter with the device_id
        adapter = SQLiteMemoryAdapter(
            memory_manager=memory_manager,
            device_id=device_id
        )
        
        try:
            # Store a memory
            memory_id = adapter.store(
                content=content,
                metadata={"category": "test"}
            )
            
            # Retrieve the stored memory
            entries = memory_manager.query_memories(limit=1)
            
            # Property 1: Memory should be stored
            assert len(entries) > 0, "Memory should be stored"
            
            # Property 2: The device_id should be stored exactly as provided
            stored_entry = entries[0]
            assert hasattr(stored_entry, 'device_id'), "Stored entry should have device_id attribute"
            assert stored_entry.device_id == device_id, \
                f"Device_id should be stored exactly as provided. " \
                f"Expected '{device_id}', got '{stored_entry.device_id}'"
        
        finally:
            adapter.close()
