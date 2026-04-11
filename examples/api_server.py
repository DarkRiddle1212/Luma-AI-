"""
Example API server for Luma Memory Module.
Run this to start the REST API for agent communication.
"""

from luma_memory import MemoryManager, SQLiteStorage
from luma_memory.api import MemoryAPI


def main():
    """Start the Memory API server."""
    
    # Initialize memory manager
    print("Initializing Luma Memory Manager...")
    manager = MemoryManager(
        storage=SQLiteStorage("luma_memory.db"),
        enable_encryption=False  # Set to True for encrypted storage
    )
    
    # Create and start API
    print("Starting Memory API server...")
    api = MemoryAPI(manager, host="0.0.0.0", port=5000)
    
    print("\nMemory API is running!")
    print("Available endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /memory              - Store a memory")
    print("  GET  /memory              - Retrieve memories (with filters)")
    print("  GET  /memory/<id>         - Get specific memory")
    print("  DELETE /memory/<id>       - Delete a memory")
    print("  GET  /memory/summary      - Get context summary")
    print("\nExample curl commands:")
    print("  curl http://localhost:5000/health")
    print('  curl -X POST http://localhost:5000/memory -H "Content-Type: application/json" -d \'{"content":"Test","memory_type":"action","source":"laptop"}\'')
    print("  curl http://localhost:5000/memory?source=laptop")
    print("\n")
    
    api.run(debug=True)


if __name__ == "__main__":
    main()
