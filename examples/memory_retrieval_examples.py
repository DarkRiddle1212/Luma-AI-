"""
Memory Retrieval Enhancement Examples

This module demonstrates the enhanced memory retrieval capabilities including:
- Basic retrieval with filters
- Adapter configuration
- Error handling
- Backward compatibility

These examples show how to use the enhanced QueryParameters, typed contracts,
and error handling features added to the memory system.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from luma_memory.storage.sqlite_storage import SQLiteStorage
from luma_memory.memory_manager import MemoryManager
from luma.adapters.sqlite_memory_adapter import SQLiteMemoryAdapter
from luma.core.memory_interface import (
    QueryParameters,
    MemoryEntry,
    RetrievalResult,
    MemoryStorageError,
    MemoryRetrievalError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Retrieval with Filters
# ============================================================================

def example_basic_retrieval_with_filters():
    """
    Demonstrates basic memory retrieval using enhanced query parameters.
    
    Shows how to:
    - Store memories with metadata
    - Retrieve with category filters
    - Retrieve with tag filters
    - Retrieve with time range filters
    - Combine multiple filters
    """
    print("\n" + "="*70)
    print("Example 1: Basic Retrieval with Filters")
    print("="*70 + "\n")
    
    # Initialize storage and adapter
    storage = SQLiteStorage("./data/examples_memory.db")
    memory_manager = MemoryManager(storage=storage)
    adapter = SQLiteMemoryAdapter(memory_manager)
    
    try:
        # Store some example memories
        print("Storing example memories...")
        
        adapter.store(
            "Python is a high-level programming language",
            metadata={"category": "education", "tags": ["programming", "python"]}
        )
        
        adapter.store(
            "JavaScript is used for web development",
            metadata={"category": "education", "tags": ["programming", "javascript", "web"]}
        )
        
        adapter.store(
            "Remember to buy groceries tomorrow",
            metadata={"category": "personal", "tags": ["reminder", "shopping"]}
        )
        
        adapter.store(
            "Meeting scheduled for 3 PM",
            metadata={"category": "work", "tags": ["meeting", "schedule"]}
        )
        
        print("✓ Stored 4 memories\n")
        
        # Example 1a: Retrieve by category
        print("1a. Retrieve by category (education):")
        params: QueryParameters = {
            "category": "education",
            "limit": 10
        }
        result: RetrievalResult = adapter.retrieve(params=params)
        print(f"   Found {result['total_count']} memories")
        for memory in result["memories"]:
            print(f"   - {memory['content']}")
        print()
        
        # Example 1b: Retrieve by tags
        print("1b. Retrieve by tags (programming):")
        params = {
            "tags": ["programming"],
            "limit": 10
        }
        result = adapter.retrieve(params=params)
        print(f"   Found {result['total_count']} memories")
        for memory in result["memories"]:
            print(f"   - {memory['content']}")
            print(f"     Tags: {memory['tags']}")
        print()
        
        # Example 1c: Retrieve by multiple tags (AND logic)
        print("1c. Retrieve by multiple tags (programming AND python):")
        params = {
            "tags": ["programming", "python"],
            "limit": 10
        }
        result = adapter.retrieve(params=params)
        print(f"   Found {result['total_count']} memories")
        for memory in result["memories"]:
            print(f"   - {memory['content']}")
        print()
        
        # Example 1d: Retrieve by time range
        print("1d. Retrieve by time range (last 24 hours):")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "limit": 10
        }
        result = adapter.retrieve(params=params)
        print(f"   Found {result['total_count']} memories")
        print(f"   Execution time: {result['query_metadata']['execution_time_ms']:.2f}ms")
        print()
        
        # Example 1e: Combine multiple filters
        print("1e. Combine multiple filters (category + tags):")
        params = {
            "category": "education",
            "tags": ["programming"],
            "limit": 5
        }
        result = adapter.retrieve(params=params)
        print(f"   Found {result['total_count']} memories")
        print(f"   Filters applied: {result['query_metadata']['filters_applied']}")
        for memory in result["memories"]:
            print(f"   - {memory['content']}")
            print(f"     Category: {memory['category']}, Tags: {memory['tags']}")
        print()
        
    finally:
        adapter.close()
        print("✓ Adapter closed\n")


# ============================================================================
# Example 2: Adapter Configuration
# ============================================================================

def example_adapter_configuration():
    """
    Demonstrates adapter configuration with device_id, default_category, and default_tags.
    
    Shows how to:
    - Configure adapter with defaults
    - Apply default category automatically
    - Merge default tags with metadata tags
    - Use device_id for multi-device scenarios
    """
    print("\n" + "="*70)
    print("Example 2: Adapter Configuration")
    print("="*70 + "\n")
    
    # Initialize storage and memory manager
    storage = SQLiteStorage("./data/examples_memory.db")
    memory_manager = MemoryManager(storage=storage)
    
    # Configure adapter with defaults
    print("Configuring adapter with defaults:")
    print("  - device_id: 'laptop-001'")
    print("  - default_category: 'learning'")
    print("  - default_tags: ['auto-tagged', 'example']")
    print()
    
    adapter = SQLiteMemoryAdapter(
        memory_manager,
        device_id="laptop-001",
        default_category="learning",
        default_tags=["auto-tagged", "example"]
    )
    
    try:
        # Example 2a: Store without category (uses default)
        print("2a. Store without category (uses default):")
        memory_id = adapter.store(
            "Docker is a containerization platform",
            metadata={"tags": ["docker", "devops"]}
        )
        print(f"   ✓ Stored memory {memory_id}")
        
        # Retrieve to verify default category was applied
        result = adapter.retrieve(params={"category": "learning", "limit": 1})
        if result["memories"]:
            memory = result["memories"][0]
            print(f"   Category: {memory['category']} (default applied)")
            print(f"   Tags: {memory['tags']} (merged with defaults)")
        print()
        
        # Example 2b: Store with explicit category (overrides default)
        print("2b. Store with explicit category (overrides default):")
        memory_id = adapter.store(
            "Team meeting notes",
            metadata={"category": "work", "tags": ["meeting"]}
        )
        print(f"   ✓ Stored memory {memory_id}")
        
        # Retrieve to verify explicit category was used
        result = adapter.retrieve(params={"category": "work", "limit": 1})
        if result["memories"]:
            memory = result["memories"][0]
            print(f"   Category: {memory['category']} (explicit)")
            print(f"   Tags: {memory['tags']} (still merged with defaults)")
        print()
        
        # Example 2c: Verify default tags are merged
        print("2c. Verify default tags are merged:")
        result = adapter.retrieve(params={"tags": ["auto-tagged"], "limit": 10})
        print(f"   Found {result['total_count']} memories with 'auto-tagged' tag")
        print(f"   (All memories stored through this adapter have default tags)")
        print()
        
    finally:
        adapter.close()
        print("✓ Adapter closed\n")


# ============================================================================
# Example 3: Error Handling
# ============================================================================

def example_error_handling():
    """
    Demonstrates comprehensive error handling for memory operations.
    
    Shows how to:
    - Handle storage errors gracefully
    - Handle retrieval errors gracefully
    - Validate query parameters
    - Handle invalid inputs
    """
    print("\n" + "="*70)
    print("Example 3: Error Handling")
    print("="*70 + "\n")
    
    # Initialize storage and adapter
    storage = SQLiteStorage("./data/examples_memory.db")
    memory_manager = MemoryManager(storage=storage)
    adapter = SQLiteMemoryAdapter(memory_manager)
    
    try:
        # Example 3a: Handle storage errors
        print("3a. Handle storage errors:")
        try:
            # This would fail if storage is unavailable
            memory_id = adapter.store(
                "Test content",
                metadata={"tags": ["test"]}
            )
            print(f"   ✓ Storage successful: {memory_id}")
        except MemoryStorageError as e:
            print(f"   ✗ Storage failed: {e}")
            print(f"   Application continues running...")
        print()
        
        # Example 3b: Handle retrieval errors
        print("3b. Handle retrieval errors:")
        try:
            result = adapter.retrieve(params={"query": "test", "limit": 5})
            print(f"   ✓ Retrieval successful: {result['total_count']} memories")
        except MemoryRetrievalError as e:
            print(f"   ✗ Retrieval failed: {e}")
            print(f"   Falling back to empty results...")
            result = {"memories": [], "total_count": 0, "query_metadata": {}}
        print()
        
        # Example 3c: Validate query parameters (invalid limit)
        print("3c. Validate query parameters (invalid limit):")
        try:
            params: QueryParameters = {
                "query": "test",
                "limit": -5  # Invalid: must be positive
            }
            result = adapter.retrieve(params=params)
        except ValueError as e:
            print(f"   ✗ Validation error: {e}")
            print(f"   Fix: Use positive limit value")
        print()
        
        # Example 3d: Validate timestamp range
        print("3d. Validate timestamp range (invalid range):")
        try:
            end_time = datetime.now()
            start_time = end_time + timedelta(days=1)  # Invalid: start > end
            params = {
                "start_time": start_time,
                "end_time": end_time,
                "limit": 10
            }
            result = adapter.retrieve(params=params)
        except ValueError as e:
            print(f"   ✗ Validation error: {e}")
            print(f"   Fix: Ensure start_time <= end_time")
        print()
        
        # Example 3e: Handle empty/whitespace queries
        print("3e. Handle empty/whitespace queries:")
        params = {
            "query": "   ",  # Whitespace only
            "limit": 10
        }
        result = adapter.retrieve(params=params)
        print(f"   ✓ Empty query handled gracefully")
        print(f"   Result: {result['total_count']} memories (treats as no query)")
        print()
        
        # Example 3f: Handle invalid tag types
        print("3f. Handle invalid tag types:")
        try:
            params = {
                "tags": ["valid", 123, "another"],  # Invalid: 123 is not a string
                "limit": 10
            }
            result = adapter.retrieve(params=params)
        except ValueError as e:
            print(f"   ✗ Validation error: {e}")
            print(f"   Fix: All tags must be strings")
        print()
        
    finally:
        adapter.close()
        print("✓ Adapter closed\n")


# ============================================================================
# Example 4: Backward Compatibility
# ============================================================================

def example_backward_compatibility():
    """
    Demonstrates backward compatibility with legacy API.
    
    Shows how to:
    - Use legacy retrieve(query, limit) API
    - Use enhanced retrieve(params) API
    - Verify both produce equivalent results
    - Migrate from legacy to enhanced API
    """
    print("\n" + "="*70)
    print("Example 4: Backward Compatibility")
    print("="*70 + "\n")
    
    # Initialize storage and adapter
    storage = SQLiteStorage("./data/examples_memory.db")
    memory_manager = MemoryManager(storage=storage)
    adapter = SQLiteMemoryAdapter(memory_manager)
    
    try:
        # Store some test data
        print("Storing test memories...")
        adapter.store(
            "Python programming tutorial",
            metadata={"category": "education", "tags": ["python", "tutorial"]}
        )
        adapter.store(
            "Python best practices guide",
            metadata={"category": "education", "tags": ["python", "guide"]}
        )
        print("✓ Stored 2 memories\n")
        
        # Example 4a: Legacy API (query string + limit)
        print("4a. Legacy API (query string + limit):")
        result_legacy = adapter.retrieve(query="Python", limit=5)
        print(f"   retrieve(query='Python', limit=5)")
        print(f"   Found {result_legacy['total_count']} memories")
        for memory in result_legacy["memories"]:
            print(f"   - {memory['content']}")
        print()
        
        # Example 4b: Enhanced API (params dictionary)
        print("4b. Enhanced API (params dictionary):")
        params: QueryParameters = {
            "query": "Python",
            "limit": 5
        }
        result_enhanced = adapter.retrieve(params=params)
        print(f"   retrieve(params={{'query': 'Python', 'limit': 5}})")
        print(f"   Found {result_enhanced['total_count']} memories")
        for memory in result_enhanced["memories"]:
            print(f"   - {memory['content']}")
        print()
        
        # Example 4c: Verify equivalence
        print("4c. Verify equivalence:")
        print(f"   Legacy result count: {result_legacy['total_count']}")
        print(f"   Enhanced result count: {result_enhanced['total_count']}")
        print(f"   Results match: {result_legacy['total_count'] == result_enhanced['total_count']}")
        print()
        
        # Example 4d: Enhanced API with additional filters
        print("4d. Enhanced API with additional filters (not possible with legacy):")
        params = {
            "query": "Python",
            "category": "education",
            "tags": ["tutorial"],
            "limit": 5
        }
        result = adapter.retrieve(params=params)
        print(f"   retrieve(params={{'query': 'Python', 'category': 'education', 'tags': ['tutorial'], 'limit': 5}})")
        print(f"   Found {result['total_count']} memories")
        print(f"   Execution time: {result['query_metadata']['execution_time_ms']:.2f}ms")
        print(f"   Filters applied: {result['query_metadata']['filters_applied']}")
        print()
        
        # Example 4e: Migration guide
        print("4e. Migration guide (legacy → enhanced):")
        print("   Legacy:  result = adapter.retrieve('Python', limit=10)")
        print("   Enhanced: result = adapter.retrieve(params={'query': 'Python', 'limit': 10})")
        print()
        print("   Benefits of enhanced API:")
        print("   - Type safety with QueryParameters")
        print("   - Rich filtering (category, tags, time range)")
        print("   - Execution metadata in results")
        print("   - Future-proof for vector search")
        print()
        
    finally:
        adapter.close()
        print("✓ Adapter closed\n")


# ============================================================================
# Example 5: Advanced Usage - Retrieval Result Metadata
# ============================================================================

def example_retrieval_result_metadata():
    """
    Demonstrates using retrieval result metadata for monitoring and debugging.
    
    Shows how to:
    - Access execution time metrics
    - Inspect applied filters
    - Check pagination status
    - Use metadata for performance monitoring
    """
    print("\n" + "="*70)
    print("Example 5: Retrieval Result Metadata")
    print("="*70 + "\n")
    
    # Initialize storage and adapter
    storage = SQLiteStorage("./data/examples_memory.db")
    memory_manager = MemoryManager(storage=storage)
    adapter = SQLiteMemoryAdapter(memory_manager)
    
    try:
        # Store some test data
        print("Storing test memories...")
        for i in range(15):
            adapter.store(
                f"Test memory {i+1}",
                metadata={"category": "test", "tags": ["performance", f"batch-{i//5}"]}
            )
        print("✓ Stored 15 memories\n")
        
        # Example 5a: Access execution time
        print("5a. Access execution time:")
        params: QueryParameters = {
            "category": "test",
            "limit": 10
        }
        result = adapter.retrieve(params=params)
        print(f"   Query executed in {result['query_metadata']['execution_time_ms']:.2f}ms")
        print(f"   Retrieved {result['total_count']} memories")
        print()
        
        # Example 5b: Inspect applied filters
        print("5b. Inspect applied filters:")
        params = {
            "category": "test",
            "tags": ["performance"],
            "limit": 5
        }
        result = adapter.retrieve(params=params)
        print(f"   Filters applied: {result['query_metadata']['filters_applied']}")
        print(f"   Limit used: {result['query_metadata']['limit']}")
        print()
        
        # Example 5c: Check pagination status
        print("5c. Check pagination status:")
        params = {
            "category": "test",
            "limit": 5
        }
        result = adapter.retrieve(params=params)
        print(f"   Retrieved: {result['total_count']} memories")
        print(f"   Limit: {result['query_metadata']['limit']}")
        print(f"   Has more: {result['query_metadata']['has_more']}")
        print(f"   (Pagination support coming in future update)")
        print()
        
        # Example 5d: Performance monitoring
        print("5d. Performance monitoring:")
        import time
        
        # Run multiple queries and track performance
        execution_times = []
        for i in range(5):
            params = {
                "category": "test",
                "tags": ["performance"],
                "limit": 10
            }
            result = adapter.retrieve(params=params)
            execution_times.append(result['query_metadata']['execution_time_ms'])
        
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        print(f"   Ran 5 queries:")
        print(f"   - Average: {avg_time:.2f}ms")
        print(f"   - Min: {min_time:.2f}ms")
        print(f"   - Max: {max_time:.2f}ms")
        print()
        
    finally:
        adapter.close()
        print("✓ Adapter closed\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MEMORY RETRIEVAL ENHANCEMENT EXAMPLES")
    print("="*70)
    
    try:
        example_basic_retrieval_with_filters()
        example_adapter_configuration()
        example_error_handling()
        example_backward_compatibility()
        example_retrieval_result_metadata()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\n✗ Example failed: {e}\n")


if __name__ == "__main__":
    main()
