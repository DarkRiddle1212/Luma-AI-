"""
Comprehensive Usage Guide for Luma Memory Module - Agent Integration

This guide demonstrates how agents (laptop, phone, or other devices) can
integrate with the Luma Memory Module API to store and retrieve user actions
and context summaries.

Target Audience: Agent developers building on the Luma platform
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time


class LumaMemoryAgent:
    """
    Production-ready client for agents to interact with Luma Memory API.
    
    This client implements best practices including:
    - Retry logic with exponential backoff
    - Input validation
    - Error handling
    - Proper datetime formatting
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", device_id: str = None):
        """
        Initialize the agent client.
        
        Args:
            base_url: Base URL of the Memory API (default: http://localhost:8000)
            device_id: Unique identifier for this device/agent
        """
        self.base_url = base_url.rstrip('/')
        self.device_id = device_id or "unknown-device"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def wait_for_service(self, max_wait: int = 30) -> bool:
        """
        Wait for the API service to become available.
        
        Args:
            max_wait: Maximum seconds to wait
            
        Returns:
            True if service is available, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{self.base_url}/api/v1/health")
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.
        
        Returns:
            Health status dictionary
            
        Raises:
            requests.exceptions.RequestException: If health check fails
        """
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage and performance statistics.
        
        Returns:
            Statistics dictionary
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = self.session.get(f"{self.base_url}/api/v1/stats")
        response.raise_for_status()
        return response.json()
    
    def create_memory(
        self,
        action: str,
        context: Dict[str, Any],
        sensitivity: str = "public",
        tags: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Create a new memory entry with retry logic.
        
        Args:
            action: Description of the user action
            context: Dictionary containing contextual information
            sensitivity: Privacy level (public, private, sensitive)
            tags: List of tags for categorization
            max_retries: Maximum number of retry attempts
            
        Returns:
            Entry ID of the created memory
            
        Raises:
            ValueError: If validation fails
            requests.exceptions.RequestException: If request fails after retries
        """
        # Validate inputs
        if not action or not action.strip():
            raise ValueError("Action cannot be empty")
        
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
        
        if sensitivity not in ["public", "private", "sensitive"]:
            raise ValueError(f"Invalid sensitivity level: {sensitivity}")
        
        payload = {
            "action": action,
            "context": context,
            "device_id": self.device_id,
            "sensitivity": sensitivity,
            "tags": tags or []
        }
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/memory",
                    json=payload
                )
                response.raise_for_status()
                return response.json()["entry_id"]
                
            except requests.exceptions.HTTPError as e:
                # Retry on server errors (500, 503)
                if e.response.status_code in [500, 503] and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                    continue
                raise
    
    def get_memory(self, entry_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific memory entry by ID.
        
        Args:
            entry_id: Unique identifier of the memory entry
            
        Returns:
            Memory entry dictionary
            
        Raises:
            requests.exceptions.HTTPError: If entry not found (404) or other error
        """
        response = self.session.get(f"{self.base_url}/api/v1/memory/{entry_id}")
        response.raise_for_status()
        return response.json()
    
    def query_memories(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        action_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Query memory entries with filters.
        
        Args:
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            tags: Filter by tags (entries must have ALL specified tags)
            action_type: Filter by action type (partial match)
            limit: Maximum number of entries to return (1-1000)
            offset: Number of entries to skip (for pagination)
            
        Returns:
            Dictionary with 'entries', 'total', 'limit', 'offset' keys
            
        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.RequestException: If request fails
        """
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        
        payload = {
            "limit": limit,
            "offset": offset
        }
        
        if start_time:
            payload["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if end_time:
            payload["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        if tags:
            payload["tags"] = tags
        
        if action_type:
            payload["action_type"] = action_type
        
        response = self.session.post(
            f"{self.base_url}/api/v1/memory/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# EXAMPLE 1: Basic Agent Usage - Laptop Agent
# ============================================================================

def example_laptop_agent():
    """
    Example: Laptop agent tracking user productivity activities.
    """
    print("=" * 70)
    print("EXAMPLE 1: Laptop Agent - Productivity Tracking")
    print("=" * 70)
    
    # Initialize agent
    agent = LumaMemoryAgent(device_id="laptop-001")
    
    # Wait for service to be ready
    print("\n1. Waiting for Memory API service...")
    if not agent.wait_for_service():
        print("   ERROR: Service not available")
        return
    print("   ✓ Service is ready")
    
    # Check health
    print("\n2. Checking API health...")
    health = agent.health_check()
    print(f"   Status: {health['status']}")
    
    # Store user actions
    print("\n3. Storing user actions...")
    
    # Action 1: User opened an application
    entry_id1 = agent.create_memory(
        action="User opened VS Code",
        context={
            "application": "vscode",
            "workspace": "/home/user/projects/luma-memory",
            "window_count": 1
        },
        sensitivity="public",
        tags=["productivity", "coding", "vscode"]
    )
    print(f"   ✓ Stored: User opened VS Code (ID: {entry_id1[:8]}...)")
    
    # Action 2: User started working on a file
    entry_id2 = agent.create_memory(
        action="User edited file",
        context={
            "file": "luma_memory/api/routes.py",
            "lines_changed": 45,
            "language": "python"
        },
        sensitivity="private",
        tags=["coding", "python", "editing"]
    )
    print(f"   ✓ Stored: User edited file (ID: {entry_id2[:8]}...)")
    
    # Action 3: User ran tests
    entry_id3 = agent.create_memory(
        action="User ran tests",
        context={
            "command": "pytest tests/",
            "tests_passed": 42,
            "tests_failed": 0,
            "duration_seconds": 3.2
        },
        sensitivity="public",
        tags=["testing", "pytest", "success"]
    )
    print(f"   ✓ Stored: User ran tests (ID: {entry_id3[:8]}...)")
    
    # Retrieve a specific memory
    print("\n4. Retrieving specific memory...")
    memory = agent.get_memory(entry_id1)
    print(f"   Action: {memory['action']}")
    print(f"   Context: {json.dumps(memory['context'], indent=6)}")
    print(f"   Tags: {memory['tags']}")
    
    # Query recent coding activities
    print("\n5. Querying recent coding activities...")
    results = agent.query_memories(
        start_time=datetime.utcnow() - timedelta(hours=1),
        tags=["coding"],
        limit=10
    )
    print(f"   Found {results['total']} coding activities:")
    for entry in results['entries']:
        print(f"   - {entry['action']} ({entry['timestamp']})")
    
    print("\n✓ Laptop agent example completed successfully!\n")


# ============================================================================
# EXAMPLE 2: Phone Agent - Context Awareness
# ============================================================================

def example_phone_agent():
    """
    Example: Phone agent tracking user context and notifications.
    """
    print("=" * 70)
    print("EXAMPLE 2: Phone Agent - Context Awareness")
    print("=" * 70)
    
    # Initialize agent
    agent = LumaMemoryAgent(device_id="phone-001")
    
    print("\n1. Storing context and notifications...")
    
    # Context 1: Location change
    entry_id1 = agent.create_memory(
        action="User arrived at location",
        context={
            "location_type": "home",
            "wifi_network": "HomeNetwork-5G",
            "time_of_day": "evening"
        },
        sensitivity="private",
        tags=["location", "context", "home"]
    )
    print(f"   ✓ Stored: Location context (ID: {entry_id1[:8]}...)")
    
    # Context 2: Notification received
    entry_id2 = agent.create_memory(
        action="User received notification",
        context={
            "app": "Slack",
            "channel": "engineering",
            "priority": "high",
            "preview": "Meeting in 15 minutes"
        },
        sensitivity="private",
        tags=["notification", "slack", "meeting"]
    )
    print(f"   ✓ Stored: Notification (ID: {entry_id2[:8]}...)")
    
    # Context 3: App usage
    entry_id3 = agent.create_memory(
        action="User opened app",
        context={
            "app": "Calendar",
            "duration_seconds": 45,
            "action_taken": "viewed_schedule"
        },
        sensitivity="public",
        tags=["app_usage", "calendar", "productivity"]
    )
    print(f"   ✓ Stored: App usage (ID: {entry_id3[:8]}...)")
    
    # Query notifications
    print("\n2. Querying recent notifications...")
    results = agent.query_memories(
        tags=["notification"],
        limit=5
    )
    print(f"   Found {results['total']} notifications")
    
    # Query by action type
    print("\n3. Querying app-related activities...")
    results = agent.query_memories(
        action_type="app",
        limit=10
    )
    print(f"   Found {results['total']} app-related activities:")
    for entry in results['entries']:
        print(f"   - {entry['action']}")
    
    print("\n✓ Phone agent example completed successfully!\n")


# ============================================================================
# EXAMPLE 3: Sensitive Data Handling
# ============================================================================

def example_sensitive_data():
    """
    Example: Handling sensitive user data with encryption.
    """
    print("=" * 70)
    print("EXAMPLE 3: Sensitive Data Handling")
    print("=" * 70)
    
    agent = LumaMemoryAgent(device_id="laptop-002")
    
    print("\n1. Storing sensitive data (automatically encrypted)...")
    
    # Sensitive action: Authentication
    entry_id1 = agent.create_memory(
        action="User logged into banking app",
        context={
            "app": "mobile_banking",
            "account_type": "checking",
            "authentication_method": "biometric"
        },
        sensitivity="sensitive",  # This triggers automatic encryption
        tags=["banking", "authentication", "security"]
    )
    print(f"   ✓ Stored: Banking login (ID: {entry_id1[:8]}...)")
    print("   Note: Data is encrypted at rest using AES-256")
    
    # Sensitive action: Password manager
    entry_id2 = agent.create_memory(
        action="User accessed password manager",
        context={
            "app": "1Password",
            "vault": "personal",
            "entries_accessed": 3
        },
        sensitivity="sensitive",
        tags=["security", "passwords", "authentication"]
    )
    print(f"   ✓ Stored: Password manager access (ID: {entry_id2[:8]}...)")
    
    # Retrieve sensitive data (automatically decrypted)
    print("\n2. Retrieving sensitive data (automatically decrypted)...")
    memory = agent.get_memory(entry_id1)
    print(f"   Action: {memory['action']}")
    print(f"   Sensitivity: {memory['sensitivity']}")
    print("   Note: Data was decrypted transparently")
    
    print("\n✓ Sensitive data example completed successfully!\n")


# ============================================================================
# EXAMPLE 4: Pagination for Large Datasets
# ============================================================================

def example_pagination():
    """
    Example: Using pagination to retrieve large datasets efficiently.
    """
    print("=" * 70)
    print("EXAMPLE 4: Pagination for Large Datasets")
    print("=" * 70)
    
    agent = LumaMemoryAgent(device_id="laptop-003")
    
    # First, create some test data
    print("\n1. Creating test data...")
    for i in range(10):
        agent.create_memory(
            action=f"Test action {i+1}",
            context={"test": True, "index": i+1},
            tags=["test", "pagination"]
        )
    print("   ✓ Created 10 test entries")
    
    # Paginate through results
    print("\n2. Paginating through results (page size: 3)...")
    page_size = 3
    offset = 0
    page_num = 1
    
    while True:
        results = agent.query_memories(
            tags=["pagination"],
            limit=page_size,
            offset=offset
        )
        
        if not results['entries']:
            break
        
        print(f"\n   Page {page_num} (offset={offset}, limit={page_size}):")
        for entry in results['entries']:
            print(f"   - {entry['action']}")
        
        offset += page_size
        page_num += 1
        
        if offset >= results['total']:
            break
    
    print(f"\n   Total entries: {results['total']}")
    print("\n✓ Pagination example completed successfully!\n")


# ============================================================================
# EXAMPLE 5: Error Handling Best Practices
# ============================================================================

def example_error_handling():
    """
    Example: Proper error handling for production agents.
    """
    print("=" * 70)
    print("EXAMPLE 5: Error Handling Best Practices")
    print("=" * 70)
    
    agent = LumaMemoryAgent(device_id="laptop-004")
    
    # Example 1: Validation error
    print("\n1. Handling validation errors...")
    try:
        agent.create_memory(
            action="",  # Empty action - will fail validation
            context={}
        )
    except ValueError as e:
        print(f"   ✓ Caught validation error: {e}")
    
    # Example 2: Invalid sensitivity level
    print("\n2. Handling invalid sensitivity level...")
    try:
        agent.create_memory(
            action="Test action",
            context={},
            sensitivity="confidential"  # Invalid level
        )
    except ValueError as e:
        print(f"   ✓ Caught validation error: {e}")
    
    # Example 3: Not found error
    print("\n3. Handling not found errors...")
    try:
        agent.get_memory("nonexistent-id-12345")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"   ✓ Caught 404 error: Entry not found")
    
    # Example 4: Invalid query parameters
    print("\n4. Handling invalid query parameters...")
    try:
        agent.query_memories(limit=5000)  # Exceeds maximum
    except ValueError as e:
        print(f"   ✓ Caught validation error: {e}")
    
    print("\n✓ Error handling example completed successfully!\n")


# ============================================================================
# EXAMPLE 6: Time-Based Queries
# ============================================================================

def example_time_queries():
    """
    Example: Querying memories by time range.
    """
    print("=" * 70)
    print("EXAMPLE 6: Time-Based Queries")
    print("=" * 70)
    
    agent = LumaMemoryAgent(device_id="laptop-005")
    
    # Create some timestamped entries
    print("\n1. Creating timestamped entries...")
    agent.create_memory(
        action="Morning activity",
        context={"time_of_day": "morning"},
        tags=["time_test"]
    )
    time.sleep(1)  # Small delay to ensure different timestamps
    agent.create_memory(
        action="Afternoon activity",
        context={"time_of_day": "afternoon"},
        tags=["time_test"]
    )
    print("   ✓ Created timestamped entries")
    
    # Query last hour
    print("\n2. Querying entries from last hour...")
    results = agent.query_memories(
        start_time=datetime.utcnow() - timedelta(hours=1),
        tags=["time_test"]
    )
    print(f"   Found {results['total']} entries from last hour")
    
    # Query last 24 hours
    print("\n3. Querying entries from last 24 hours...")
    results = agent.query_memories(
        start_time=datetime.utcnow() - timedelta(days=1),
        tags=["time_test"]
    )
    print(f"   Found {results['total']} entries from last 24 hours")
    
    # Query specific time range
    print("\n4. Querying specific time range...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)
    results = agent.query_memories(
        start_time=start_time,
        end_time=end_time,
        tags=["time_test"]
    )
    print(f"   Found {results['total']} entries in last 5 minutes")
    
    print("\n✓ Time-based queries example completed successfully!\n")


# ============================================================================
# EXAMPLE 7: Performance Monitoring
# ============================================================================

def example_performance_monitoring():
    """
    Example: Monitoring API performance and statistics.
    """
    print("=" * 70)
    print("EXAMPLE 7: Performance Monitoring")
    print("=" * 70)
    
    agent = LumaMemoryAgent(device_id="laptop-006")
    
    # Get initial stats
    print("\n1. Getting initial statistics...")
    stats = agent.get_stats()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Storage size: {stats['storage_size_bytes']} bytes")
    print(f"   Encryption enabled: {stats['encryption_enabled']}")
    
    # Create some entries and measure performance
    print("\n2. Creating entries and measuring performance...")
    start_time = time.time()
    for i in range(5):
        agent.create_memory(
            action=f"Performance test {i+1}",
            context={"test": "performance", "index": i+1},
            tags=["performance"]
        )
    elapsed = time.time() - start_time
    print(f"   Created 5 entries in {elapsed:.3f} seconds")
    print(f"   Average: {(elapsed/5)*1000:.1f} ms per entry")
    
    # Get updated stats
    print("\n3. Getting updated statistics...")
    stats = agent.get_stats()
    print(f"   Total entries: {stats['total_entries']}")
    if 'performance' in stats:
        perf = stats['performance']
        if 'create_memory' in perf:
            print(f"   Avg create time: {perf['create_memory']['avg_time_ms']:.1f} ms")
    
    print("\n✓ Performance monitoring example completed successfully!\n")


# ============================================================================
# Main Function - Run All Examples
# ============================================================================

def main():
    """
    Run all usage examples.
    
    Note: Make sure the Luma Memory API server is running before executing:
        python -m luma_memory.api.server
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Luma Memory Module - Agent Usage Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    examples = [
        ("Laptop Agent - Productivity Tracking", example_laptop_agent),
        ("Phone Agent - Context Awareness", example_phone_agent),
        ("Sensitive Data Handling", example_sensitive_data),
        ("Pagination for Large Datasets", example_pagination),
        ("Error Handling Best Practices", example_error_handling),
        ("Time-Based Queries", example_time_queries),
        ("Performance Monitoring", example_performance_monitoring),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all examples")
    print()
    
    try:
        choice = input("Select an example to run (1-8): ").strip()
        
        if choice == str(len(examples) + 1):
            # Run all examples
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Example failed: {e}\n")
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            # Run selected example
            _, func = examples[int(choice) - 1]
            func()
        else:
            print("Invalid choice. Please run again and select a valid option.")
    
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        print("\nMake sure the Luma Memory API server is running:")
        print("  python -m luma_memory.api.server")


if __name__ == "__main__":
    main()
