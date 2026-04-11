"""
Example demonstrating performance monitoring in the Luma Memory Module.

This example shows how to:
1. Enable performance monitoring
2. Perform various memory operations
3. Retrieve and display performance metrics
"""

from luma_memory.memory_manager import MemoryManager
from luma_memory.storage.memory_storage import MemoryStorage
from luma_memory.config import MemoryModuleConfig
from luma_memory.models import SensitivityLevel


def main():
    """Demonstrate performance monitoring functionality."""
    
    # Initialize with metrics enabled
    print("Initializing Memory Manager with performance monitoring enabled...")
    storage = MemoryStorage()
    config = MemoryModuleConfig(enable_metrics=True)
    manager = MemoryManager(storage=storage, config=config)
    
    print("\n" + "="*60)
    print("Creating memory entries...")
    print("="*60)
    
    # Create several memory entries
    entry_ids = []
    for i in range(10):
        entry_id = manager.create_memory(
            action=f"User action {i}",
            context={"data": f"value_{i}", "index": i},
            device_id="laptop-001",
            sensitivity=SensitivityLevel.PUBLIC,
            tags=["example", f"batch_{i // 5}"]
        )
        entry_ids.append(entry_id)
        print(f"  Created entry {i+1}/10: {entry_id}")
    
    print("\n" + "="*60)
    print("Retrieving memory entries...")
    print("="*60)
    
    # Retrieve some entries
    for i, entry_id in enumerate(entry_ids[:5]):
        entry = manager.get_memory(entry_id)
        print(f"  Retrieved entry {i+1}/5: {entry.action}")
    
    print("\n" + "="*60)
    print("Querying memory entries...")
    print("="*60)
    
    # Query entries
    entries = manager.query_memories(tags=["example"], limit=5)
    print(f"  Found {len(entries)} entries with tag 'example'")
    
    # Update an entry
    print("\n" + "="*60)
    print("Updating memory entry...")
    print("="*60)
    
    success = manager.update_memory(entry_ids[0], {"tags": ["example", "updated"]})
    print(f"  Update {'successful' if success else 'failed'}")
    
    # Delete an entry
    print("\n" + "="*60)
    print("Deleting memory entry...")
    print("="*60)
    
    success = manager.delete_memory(entry_ids[-1])
    print(f"  Delete {'successful' if success else 'failed'}")
    
    # Get performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    metrics = manager.get_performance_metrics()
    
    for operation, stats in metrics.items():
        if stats['count'] > 0:
            print(f"\n{operation}:")
            print(f"  Operations:     {stats['count']}")
            print(f"  Avg Time:       {stats['avg_time_ms']:.2f} ms")
            print(f"  Min Time:       {stats['min_time_ms']:.2f} ms")
            print(f"  Max Time:       {stats['max_time_ms']:.2f} ms")
            print(f"  Errors:         {stats['errors']}")
            print(f"  Error Rate:     {stats['error_rate']:.2f}%")
    
    # Get full stats including performance metrics
    print("\n" + "="*60)
    print("FULL STATISTICS")
    print("="*60)
    
    full_stats = manager.get_stats()
    print(f"\nTotal Entries:        {full_stats.get('total_entries', 0)}")
    print(f"Storage Size:         {full_stats.get('storage_size_bytes', 0)} bytes")
    print(f"Encryption Enabled:   {full_stats['encryption_enabled']}")
    print(f"Summarizer Enabled:   {full_stats['summarizer_enabled']}")
    
    if 'performance' in full_stats:
        print("\nPerformance metrics are included in stats!")
    
    # Reset metrics
    print("\n" + "="*60)
    print("Resetting performance metrics...")
    print("="*60)
    
    manager.reset_performance_metrics()
    print("  Metrics reset successfully")
    
    # Verify reset
    metrics_after_reset = manager.get_performance_metrics()
    total_operations = sum(m['count'] for m in metrics_after_reset.values())
    print(f"  Total operations after reset: {total_operations}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
