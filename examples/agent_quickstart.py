"""
Quick Start Guide for Agent Developers

This is a minimal example to get started quickly with the Luma Memory Module.
For comprehensive examples, see agent_usage_guide.py
"""

import requests
from datetime import datetime, timedelta


# Configuration
API_BASE_URL = "http://localhost:8000"
DEVICE_ID = "my-device-001"


def create_memory(action: str, context: dict, sensitivity: str = "public", tags: list = None):
    """Create a memory entry."""
    response = requests.post(
        f"{API_BASE_URL}/api/v1/memory",
        json={
            "action": action,
            "context": context,
            "device_id": DEVICE_ID,
            "sensitivity": sensitivity,
            "tags": tags or []
        }
    )
    response.raise_for_status()
    return response.json()["entry_id"]


def get_memory(entry_id: str):
    """Get a specific memory entry."""
    response = requests.get(f"{API_BASE_URL}/api/v1/memory/{entry_id}")
    response.raise_for_status()
    return response.json()


def query_memories(tags: list = None, limit: int = 100):
    """Query memory entries."""
    response = requests.post(
        f"{API_BASE_URL}/api/v1/memory/query",
        json={
            "tags": tags,
            "limit": limit
        }
    )
    response.raise_for_status()
    return response.json()


# Quick Start Example
if __name__ == "__main__":
    print("Luma Memory Module - Quick Start\n")
    
    # 1. Create a memory
    print("1. Creating a memory entry...")
    entry_id = create_memory(
        action="User opened document",
        context={"file": "report.pdf", "page": 1},
        sensitivity="private",
        tags=["document", "work"]
    )
    print(f"   Created entry: {entry_id}\n")
    
    # 2. Retrieve the memory
    print("2. Retrieving the memory...")
    memory = get_memory(entry_id)
    print(f"   Action: {memory['action']}")
    print(f"   Context: {memory['context']}\n")
    
    # 3. Query memories
    print("3. Querying memories with 'work' tag...")
    results = query_memories(tags=["work"])
    print(f"   Found {results['total']} entries")
    for entry in results['entries']:
        print(f"   - {entry['action']}")
    
    print("\n✓ Quick start completed!")
