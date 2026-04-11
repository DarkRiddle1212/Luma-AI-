"""
Example demonstrating how to use ContextSummarizer with configuration.

This example shows how to:
1. Load configuration from environment variables or .env file
2. Create a ContextSummarizer using the configuration
3. Use the summarizer with configured thresholds
"""

from luma_memory.config import MemoryModuleConfig
from luma_memory.processing.summarizer import ContextSummarizer
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
from datetime import datetime, timedelta


def main():
    # Example 1: Create summarizer from default configuration
    print("Example 1: Using default configuration")
    print("-" * 50)
    config = MemoryModuleConfig()
    summarizer = ContextSummarizer.from_config(config)
    
    print(f"Similarity threshold: {summarizer.similarity_threshold}")
    print(f"Entry count threshold: {summarizer.entry_count_threshold}")
    print(f"Storage size threshold: {summarizer.storage_size_threshold_bytes / 1_000_000} MB")
    print()
    
    # Example 2: Create summarizer with custom configuration
    print("Example 2: Using custom configuration")
    print("-" * 50)
    custom_config = MemoryModuleConfig(
        similarity_threshold=0.75,
        summarization_threshold=500,
        max_storage_size_mb=50
    )
    custom_summarizer = ContextSummarizer.from_config(custom_config)
    
    print(f"Similarity threshold: {custom_summarizer.similarity_threshold}")
    print(f"Entry count threshold: {custom_summarizer.entry_count_threshold}")
    print(f"Storage size threshold: {custom_summarizer.storage_size_threshold_bytes / 1_000_000} MB")
    print()
    
    # Example 3: Use the summarizer to check if summarization should trigger
    print("Example 3: Checking summarization triggers")
    print("-" * 50)
    
    # Check with default thresholds
    should_trigger = summarizer.should_trigger_summarization(
        entry_count=1500,
        storage_size=50_000_000
    )
    print(f"Should trigger with 1500 entries and 50 MB: {should_trigger}")
    
    should_trigger = summarizer.should_trigger_summarization(
        entry_count=500,
        storage_size=50_000_000
    )
    print(f"Should trigger with 500 entries and 50 MB: {should_trigger}")
    print()
    
    # Example 4: Create and summarize similar entries
    print("Example 4: Summarizing similar entries")
    print("-" * 50)
    
    now = datetime.now()
    entries = [
        MemoryEntry(
            id="1",
            timestamp=now,
            action="open_file",
            context={"file": "test.py", "line": 10},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        ),
        MemoryEntry(
            id="2",
            timestamp=now + timedelta(minutes=5),
            action="open_file",
            context={"file": "test.py", "line": 15},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        ),
        MemoryEntry(
            id="3",
            timestamp=now + timedelta(minutes=10),
            action="open_file",
            context={"file": "test.py", "line": 20},
            sensitivity=SensitivityLevel.PUBLIC,
            device_id="laptop",
            sync_status=SyncStatus.PENDING,
            tags=["coding", "python"]
        )
    ]
    
    # Identify redundant entries
    groups = summarizer.identify_redundant_entries(entries)
    print(f"Found {len(groups)} group(s) of similar entries")
    
    if groups:
        summary_id, entry_ids = groups[0]
        print(f"Group summary ID: {summary_id}")
        print(f"Entry IDs in group: {entry_ids}")
        
        # Create a summary
        entries_to_summarize = [e for e in entries if e.id in entry_ids]
        summary, linked_entry_ids = summarizer.summarize_entries(entries_to_summarize)
        
        print(f"\nSummary entry:")
        print(f"  ID: {summary.id}")
        print(f"  Action: {summary.action}")
        print(f"  Summary text: {summary.summary}")
        print(f"  Tags: {summary.tags}")
        print(f"  Entry IDs to link: {linked_entry_ids}")


if __name__ == "__main__":
    main()
