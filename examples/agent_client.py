"""
Example client for laptop/phone agents to communicate with Memory API.
"""

import requests
from datetime import datetime
from typing import Optional, List, Dict, Any


class LumaMemoryClient:
    """Client for agents to interact with Luma Memory API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the Memory API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def store_memory(
        self,
        content: str,
        memory_type: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        encrypt: bool = False
    ) -> Optional[str]:
        """
        Store a memory entry.
        
        Args:
            content: Memory content
            memory_type: Type (action, context, conversation, task, system)
            source: Source device/agent
            metadata: Additional metadata
            tags: Tags for categorization
            encrypt: Whether to encrypt
        
        Returns:
            Entry ID if successful, None otherwise
        """
        payload = {
            "content": content,
            "memory_type": memory_type,
            "source": source,
            "metadata": metadata or {},
            "tags": tags or [],
            "encrypt": encrypt
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/memory",
                json=payload
            )
            
            if response.status_code == 201:
                return response.json()['entry_id']
            else:
                print(f"Error storing memory: {response.json()}")
                return None
        except Exception as e:
            print(f"Failed to store memory: {e}")
            return None
    
    def retrieve_memories(
        self,
        memory_type: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories with filters.
        
        Args:
            memory_type: Filter by type
            source: Filter by source
            tags: Filter by tags
            limit: Maximum entries to return
        
        Returns:
            List of memory entries
        """
        params = {"limit": limit}
        
        if memory_type:
            params["memory_type"] = memory_type
        if source:
            params["source"] = source
        if tags:
            params["tags"] = ",".join(tags)
        
        try:
            response = requests.get(
                f"{self.base_url}/memory",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()['entries']
            else:
                print(f"Error retrieving memories: {response.json()}")
                return []
        except Exception as e:
            print(f"Failed to retrieve memories: {e}")
            return []
    
    def get_memory(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            Memory entry if found, None otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/memory/{entry_id}")
            
            if response.status_code == 200:
                return response.json()['entry']
            else:
                return None
        except Exception as e:
            print(f"Failed to get memory: {e}")
            return None
    
    def delete_memory(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: Entry ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.delete(f"{self.base_url}/memory/{entry_id}")
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to delete memory: {e}")
            return False
    
    def get_summary(self, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get context summary.
        
        Args:
            source: Filter by source
        
        Returns:
            Summary dictionary if successful, None otherwise
        """
        params = {}
        if source:
            params["source"] = source
        
        try:
            response = requests.get(
                f"{self.base_url}/memory/summary",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()['summary']
            else:
                return None
        except Exception as e:
            print(f"Failed to get summary: {e}")
            return None


def main():
    """Demonstrate agent client usage."""
    
    # Initialize client
    client = LumaMemoryClient()
    
    # Check health
    print("Checking API health...")
    if not client.health_check():
        print("API is not available. Make sure the server is running.")
        return
    print("API is healthy!\n")
    
    # Store memories from laptop agent
    print("Laptop Agent: Storing memories...")
    entry_id1 = client.store_memory(
        content="User opened VS Code",
        memory_type="action",
        source="laptop",
        metadata={"app": "vscode", "project": "luma"},
        tags=["coding", "productivity"]
    )
    print(f"  Stored: {entry_id1}")
    
    entry_id2 = client.store_memory(
        content="User is working on Memory Module",
        memory_type="context",
        source="laptop",
        metadata={"task": "development"},
        tags=["work", "coding"]
    )
    print(f"  Stored: {entry_id2}\n")
    
    # Store memory from phone agent
    print("Phone Agent: Storing memory...")
    entry_id3 = client.store_memory(
        content="User received notification from Slack",
        memory_type="action",
        source="phone",
        metadata={"app": "slack", "channel": "engineering"},
        tags=["communication", "work"]
    )
    print(f"  Stored: {entry_id3}\n")
    
    # Retrieve all memories
    print("Retrieving all memories...")
    all_memories = client.retrieve_memories()
    print(f"  Found {len(all_memories)} memories\n")
    
    # Retrieve laptop memories only
    print("Retrieving laptop memories...")
    laptop_memories = client.retrieve_memories(source="laptop")
    print(f"  Found {len(laptop_memories)} laptop memories")
    for mem in laptop_memories:
        print(f"    - {mem['content']}")
    print()
    
    # Retrieve by tags
    print("Retrieving memories with 'work' tag...")
    work_memories = client.retrieve_memories(tags=["work"])
    print(f"  Found {len(work_memories)} work-related memories\n")
    
    # Get summary
    print("Getting context summary...")
    summary = client.get_summary()
    if summary:
        print(f"  Total entries: {summary['total_entries']}")
        print(f"  By type: {summary['by_type']}")
        print(f"  By source: {summary['by_source']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
