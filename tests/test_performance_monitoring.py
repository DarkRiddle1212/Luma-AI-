"""Tests for performance monitoring in MemoryManager."""

import pytest
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.config import MemoryModuleConfig
from luma_memory.models import SensitivityLevel


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    def test_metrics_initialization(self):
        """Test that metrics are initialized correctly."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Check that metrics dictionary exists
        assert hasattr(manager, '_metrics')
        assert 'create_memory' in manager._metrics
        assert 'get_memory' in manager._metrics
        assert 'query_memories' in manager._metrics
        assert 'update_memory' in manager._metrics
        assert 'delete_memory' in manager._metrics
        
        # Check initial values
        for operation, metrics in manager._metrics.items():
            assert metrics['count'] == 0
            assert metrics['total_time_ms'] == 0
            assert metrics['min_time_ms'] == float('inf')
            assert metrics['max_time_ms'] == 0
            assert metrics['errors'] == 0
    
    def test_create_memory_records_metrics(self):
        """Test that create_memory records performance metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create a memory entry
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Check that metrics were recorded
        metrics = manager._metrics['create_memory']
        assert metrics['count'] == 1
        assert metrics['total_time_ms'] > 0
        assert metrics['min_time_ms'] > 0
        assert metrics['max_time_ms'] > 0
        assert metrics['errors'] == 0
    
    def test_get_memory_records_metrics(self):
        """Test that get_memory records performance metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create and retrieve a memory entry
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Reset metrics to isolate get_memory
        manager.reset_performance_metrics()
        
        entry = manager.get_memory(entry_id)
        
        # Check that metrics were recorded
        metrics = manager._metrics['get_memory']
        assert metrics['count'] == 1
        assert metrics['total_time_ms'] > 0
        assert metrics['errors'] == 0
    
    def test_query_memories_records_metrics(self):
        """Test that query_memories records performance metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries
        for i in range(3):
            manager.create_memory(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
        
        # Reset metrics to isolate query_memories
        manager.reset_performance_metrics()
        
        entries = manager.query_memories()
        
        # Check that metrics were recorded
        metrics = manager._metrics['query_memories']
        assert metrics['count'] == 1
        assert metrics['total_time_ms'] > 0
        assert metrics['errors'] == 0
    
    def test_update_memory_records_metrics(self):
        """Test that update_memory records performance metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create an entry
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Reset metrics to isolate update_memory
        manager.reset_performance_metrics()
        
        success = manager.update_memory(entry_id, {"tags": ["test"]})
        
        # Check that metrics were recorded
        metrics = manager._metrics['update_memory']
        assert metrics['count'] == 1
        assert metrics['total_time_ms'] > 0
        assert metrics['errors'] == 0
    
    def test_delete_memory_records_metrics(self):
        """Test that delete_memory records performance metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create an entry
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Reset metrics to isolate delete_memory
        manager.reset_performance_metrics()
        
        success = manager.delete_memory(entry_id)
        
        # Check that metrics were recorded
        metrics = manager._metrics['delete_memory']
        assert metrics['count'] == 1
        assert metrics['total_time_ms'] > 0
        assert metrics['errors'] == 0
    
    def test_get_performance_metrics(self):
        """Test that get_performance_metrics returns correct format."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries to generate metrics
        for i in range(5):
            manager.create_memory(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        
        # Check structure
        assert 'create_memory' in metrics
        create_metrics = metrics['create_memory']
        
        assert 'count' in create_metrics
        assert 'avg_time_ms' in create_metrics
        assert 'min_time_ms' in create_metrics
        assert 'max_time_ms' in create_metrics
        assert 'errors' in create_metrics
        assert 'error_rate' in create_metrics
        
        # Check values
        assert create_metrics['count'] == 5
        assert create_metrics['avg_time_ms'] > 0
        assert create_metrics['min_time_ms'] > 0
        assert create_metrics['max_time_ms'] >= create_metrics['min_time_ms']
        assert create_metrics['errors'] == 0
        assert create_metrics['error_rate'] == 0
    
    def test_reset_performance_metrics(self):
        """Test that reset_performance_metrics clears all metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries to generate metrics
        manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Verify metrics exist
        assert manager._metrics['create_memory']['count'] > 0
        
        # Reset metrics
        manager.reset_performance_metrics()
        
        # Verify metrics are reset
        for operation, metrics in manager._metrics.items():
            assert metrics['count'] == 0
            assert metrics['total_time_ms'] == 0
            assert metrics['min_time_ms'] == float('inf')
            assert metrics['max_time_ms'] == 0
            assert metrics['errors'] == 0
    
    def test_metrics_disabled(self):
        """Test that metrics are not recorded when disabled."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=False)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create an entry
        entry_id = manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Metrics should still be initialized but not updated
        # (the _record_metric method returns early when metrics are disabled)
        metrics = manager._metrics['create_memory']
        assert metrics['count'] == 0  # Should not be incremented
    
    def test_get_stats_includes_performance_metrics(self):
        """Test that get_stats includes performance metrics when enabled."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries
        manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Get stats
        stats = manager.get_stats()
        
        # Check that performance metrics are included
        assert 'performance' in stats
        assert 'create_memory' in stats['performance']
        assert stats['performance']['create_memory']['count'] > 0
    
    def test_error_metrics_recorded(self):
        """Test that errors are recorded in metrics."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Try to create an invalid entry (missing required fields)
        try:
            manager.create_memory(
                action="",  # Empty action should fail validation
                context={},
                device_id="device-001"
            )
        except Exception:
            pass  # Expected to fail
        
        # Check that error was recorded
        metrics = manager._metrics['create_memory']
        assert metrics['errors'] == 1
        assert metrics['count'] == 1  # Count should still increment
    
    def test_multiple_operations_aggregate_metrics(self):
        """Test that multiple operations aggregate metrics correctly."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create multiple entries
        for i in range(10):
            manager.create_memory(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
        
        # Get metrics
        metrics = manager.get_performance_metrics()
        create_metrics = metrics['create_memory']
        
        # Check aggregation
        assert create_metrics['count'] == 10
        assert create_metrics['avg_time_ms'] > 0
        assert create_metrics['min_time_ms'] <= create_metrics['avg_time_ms']
        assert create_metrics['max_time_ms'] >= create_metrics['avg_time_ms']
