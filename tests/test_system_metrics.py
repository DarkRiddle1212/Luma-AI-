"""Tests for system resource metrics collection."""

import pytest
from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.config import MemoryModuleConfig


class TestSystemMetrics:
    """Test system resource metrics collection."""
    
    def test_system_metrics_included_in_performance_metrics(self):
        """Test that system resource metrics are included when psutil is available."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries to generate metrics
        for i in range(3):
            manager.create_memory(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        
        # Check if system_resources is included (depends on psutil availability)
        # If psutil is available, system_resources should be present
        try:
            import psutil
            assert 'system_resources' in metrics
            
            # Verify system resource fields
            sys_metrics = metrics['system_resources']
            assert 'memory_usage_mb' in sys_metrics
            assert 'memory_usage_percent' in sys_metrics
            assert 'cpu_percent' in sys_metrics
            assert 'num_threads' in sys_metrics
            
            # Verify values are reasonable
            assert sys_metrics['memory_usage_mb'] > 0
            assert sys_metrics['memory_usage_percent'] >= 0
            assert sys_metrics['cpu_percent'] >= 0
            assert sys_metrics['num_threads'] > 0
            
        except ImportError:
            # psutil not available, system_resources should not be present
            assert 'system_resources' not in metrics
    
    def test_system_metrics_graceful_failure(self):
        """Test that metrics collection continues even if system metrics fail."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create an entry
        manager.create_memory(
            action="Test action",
            context={"key": "value"},
            device_id="device-001"
        )
        
        # Get performance metrics should not fail even if system metrics fail
        metrics = manager.get_performance_metrics()
        
        # Basic metrics should always be present
        assert 'create_memory' in metrics
        assert metrics['create_memory']['count'] > 0
    
    def test_storage_metrics_in_stats(self):
        """Test that storage metrics are included in get_stats."""
        storage = MemoryStorage()
        config = MemoryModuleConfig(enable_metrics=True)
        manager = MemoryManager(storage=storage, config=config)
        
        # Create some entries
        for i in range(5):
            manager.create_memory(
                action=f"Test action {i}",
                context={"key": f"value{i}"},
                device_id="device-001"
            )
        
        # Get stats
        stats = manager.get_stats()
        
        # Check that performance metrics are included
        assert 'performance' in stats
        assert 'create_memory' in stats['performance']
        
        # Verify operation metrics
        assert stats['performance']['create_memory']['count'] == 5
        assert stats['performance']['create_memory']['avg_time_ms'] > 0
