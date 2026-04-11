# Plugin Development Guide

## Overview

The Luma Memory Module provides a flexible plugin system that allows you to extend functionality by adding custom validation, processing, and metadata handling for specific memory entry types. This guide explains how to create, load, and use plugins.

## Table of Contents

1. [Plugin Architecture](#plugin-architecture)
2. [Creating a Plugin](#creating-a-plugin)
3. [Plugin Interface Reference](#plugin-interface-reference)
4. [Plugin Registration](#plugin-registration)
5. [Loading Plugins](#loading-plugins)
6. [Plugin Registry](#plugin-registry)
7. [Best Practices](#best-practices)
8. [Example Plugin](#example-plugin)
9. [Testing Plugins](#testing-plugins)

---

## Plugin Architecture

The plugin system consists of three main components:

1. **MemoryEntryPlugin**: Abstract base class that defines the plugin interface
2. **PluginRegistry**: Manages registered plugins and routes operations to appropriate plugins
3. **PluginLoader**: Dynamically discovers and loads plugins from files, directories, or packages

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Manager                          │
│  (Coordinates storage, validation, encryption)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plugin Registry                           │
│  • Routes operations to appropriate plugins                 │
│  • Manages plugin lifecycle                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Plugin A   │ │   Plugin B   │ │   Plugin C   │
│ (Social      │ │ (Calendar    │ │ (Custom      │
│  Media)      │ │  Events)     │ │  Type)       │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## Creating a Plugin

### Step 1: Import Required Classes

```python
from luma_memory.plugins.plugin_interface import (
    MemoryEntryPlugin,
    PluginValidationError,
    PluginProcessingError
)
from luma_memory.models import MemoryEntry, SensitivityLevel
from typing import Dict, Any, Optional, List
```

### Step 2: Define Your Plugin Class

Create a class that inherits from `MemoryEntryPlugin` and implements the required abstract properties:

```python
class MyCustomPlugin(MemoryEntryPlugin):
    """Plugin for handling custom memory entry types."""
    
    @property
    def name(self) -> str:
        """Unique identifier for the plugin."""
        return "my_custom_plugin"
    
    @property
    def version(self) -> str:
        """Plugin version string."""
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        """List of action types this plugin handles."""
        return ["custom_action", "another_action"]
```

### Step 3: Implement Plugin Methods

Override the methods you need to customize behavior. All methods have default implementations, so you only need to override what you want to change.

```python
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """Validate context data for your action types."""
        if action == "custom_action":
            if "required_field" not in context:
                return False, "Missing required_field"
            return True, None
        return True, None
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        """Enrich or transform entry before storage."""
        # Add computed fields
        entry.context["processed_at"] = datetime.utcnow().isoformat()
        return entry
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Generate automatic tags."""
        return ["custom", action]
```

---

## Plugin Interface Reference

### Required Properties

#### `name` (property)
- **Type**: `str`
- **Description**: Unique identifier for the plugin
- **Example**: `"social_media"`, `"calendar_events"`

#### `version` (property)
- **Type**: `str`
- **Description**: Plugin version string (semantic versioning recommended)
- **Example**: `"1.0.0"`, `"2.1.3"`

#### `supported_actions` (property)
- **Type**: `List[str]`
- **Description**: List of action types this plugin handles
- **Example**: `["tweet", "facebook_post"]`

### Optional Methods

#### `validate_entry(entry: MemoryEntry) -> tuple[bool, Optional[str]]`
Validate a complete memory entry after core validation passes.

**Parameters:**
- `entry`: The memory entry to validate

**Returns:**
- Tuple of `(is_valid, error_message)`
- If valid, `error_message` should be `None`

**Example:**
```python
def validate_entry(self, entry: MemoryEntry) -> tuple[bool, Optional[str]]:
    if entry.action == "custom_action" and not entry.tags:
        return False, "Custom actions must have at least one tag"
    return True, None
```

#### `validate_context(context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]`
Validate the context dictionary for a specific action type.

**Parameters:**
- `context`: The context dictionary to validate
- `action`: The action type

**Returns:**
- Tuple of `(is_valid, error_message)`

**Example:**
```python
def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
    if action == "email_sent":
        required_fields = ["recipient", "subject"]
        for field in required_fields:
            if field not in context:
                return False, f"Missing required field: {field}"
    return True, None
```

#### `process_before_storage(entry: MemoryEntry) -> MemoryEntry`
Process or enrich an entry before it is stored.

**Parameters:**
- `entry`: The memory entry to process

**Returns:**
- Processed memory entry (can be the same instance or a new one)

**Use Cases:**
- Extract metadata
- Normalize data
- Add computed fields
- Enrich with external data

**Example:**
```python
def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
    # Extract email domain
    if "recipient" in entry.context:
        email = entry.context["recipient"]
        domain = email.split("@")[1] if "@" in email else None
        entry.context["recipient_domain"] = domain
    return entry
```

#### `process_after_retrieval(entry: MemoryEntry) -> MemoryEntry`
Process an entry after it is retrieved from storage.

**Parameters:**
- `entry`: The memory entry to process

**Returns:**
- Processed memory entry

**Use Cases:**
- Resolve references
- Compute derived values
- Format data for display
- Decrypt sensitive fields

**Example:**
```python
def process_after_retrieval(self, entry: MemoryEntry) -> MemoryEntry:
    # Add human-readable timestamp
    entry.context["formatted_time"] = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return entry
```

#### `serialize_context(context: Dict[str, Any]) -> Dict[str, Any]`
Customize how context data is serialized for storage.

**Parameters:**
- `context`: The context dictionary to serialize

**Returns:**
- Serialized context dictionary

**Example:**
```python
def serialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # Convert datetime objects to ISO strings
    if "event_time" in context and isinstance(context["event_time"], datetime):
        context["event_time"] = context["event_time"].isoformat()
    return context
```

#### `deserialize_context(context: Dict[str, Any]) -> Dict[str, Any]`
Customize how context data is deserialized after retrieval.

**Parameters:**
- `context`: The serialized context dictionary

**Returns:**
- Deserialized context dictionary

**Example:**
```python
def deserialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
    # Convert ISO strings back to datetime objects
    if "event_time" in context and isinstance(context["event_time"], str):
        context["event_time"] = datetime.fromisoformat(context["event_time"])
    return context
```

#### `get_default_sensitivity(action: str) -> Optional[SensitivityLevel]`
Specify default sensitivity level for action types.

**Parameters:**
- `action`: The action type

**Returns:**
- Default `SensitivityLevel`, or `None` to use system default

**Example:**
```python
def get_default_sensitivity(self, action: str) -> Optional[SensitivityLevel]:
    # Financial actions are always sensitive
    if action in ["payment", "transaction"]:
        return SensitivityLevel.SENSITIVE
    return SensitivityLevel.PRIVATE
```

#### `get_default_tags(action: str, context: Dict[str, Any]) -> List[str]`
Generate automatic tags based on action type and context.

**Parameters:**
- `action`: The action type
- `context`: The context dictionary

**Returns:**
- List of tag strings

**Example:**
```python
def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
    tags = [action]
    
    # Add category tags
    if "category" in context:
        tags.append(f"category:{context['category']}")
    
    # Add priority tags
    if context.get("priority") == "high":
        tags.append("urgent")
    
    return tags
```

#### `should_summarize(entries: List[MemoryEntry]) -> bool`
Determine if a group of entries should be summarized.

**Parameters:**
- `entries`: List of memory entries to consider

**Returns:**
- `True` if entries should be summarized, `False` otherwise

**Example:**
```python
def should_summarize(self, entries: List[MemoryEntry]) -> bool:
    # Summarize if more than 50 similar entries
    return len(entries) > 50
```

#### `get_metadata() -> Dict[str, Any]`
Provide additional plugin metadata.

**Returns:**
- Dictionary containing plugin metadata

**Example:**
```python
def get_metadata(self) -> Dict[str, Any]:
    metadata = super().get_metadata()
    metadata.update({
        "description": "Plugin for email tracking",
        "author": "Your Name",
        "features": ["Email validation", "Domain extraction"]
    })
    return metadata
```

---

## Plugin Registration

The plugin system provides multiple ways to register plugins with the registry.

### Method 1: Decorator-Based Registration

The simplest way to register a plugin is using the `@register_plugin` decorator:

```python
from luma_memory.plugins import register_plugin, MemoryEntryPlugin

@register_plugin
class MyPlugin(MemoryEntryPlugin):
    @property
    def name(self) -> str:
        return "my_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["my_action"]
```

The plugin is automatically registered when the module is imported. You can also specify a custom registry:

```python
@register_plugin(registry=my_custom_registry)
class MyPlugin(MemoryEntryPlugin):
    # ...
```

### Method 2: Manual Registration

Register a plugin instance manually using the `register()` function:

```python
from luma_memory.plugins import register

plugin = MyPlugin()
register(plugin)  # Uses global registry

# Or with a custom registry:
register(plugin, registry=my_custom_registry)
```

### Method 3: Registry Method

Register directly with the registry:

```python
from luma_memory.plugins import get_global_registry

registry = get_global_registry()
plugin = MyPlugin()
registry.register(plugin)
```

### Convenience Functions

The plugin system provides several convenience functions:

```python
from luma_memory.plugins import (
    register,        # Register a plugin instance
    unregister,      # Unregister a plugin by name
    get_plugin,      # Get a plugin by name
    list_plugins,    # List all registered plugins
)

# Register a plugin
plugin = MyPlugin()
register(plugin)

# Get a plugin
my_plugin = get_plugin("my_plugin")

# List all plugins
all_plugins = list_plugins()
for plugin_info in all_plugins:
    print(f"{plugin_info['name']} v{plugin_info['version']}")

# Unregister a plugin
unregister("my_plugin")
```

---

## Loading Plugins

### Method 1: Load from File

Load a single plugin file:

```python
from luma_memory.plugins.plugin_loader import PluginLoader

loader = PluginLoader()
plugin_classes = loader.load_from_file("path/to/my_plugin.py")
```

### Method 2: Load from Directory

Load all plugins from a directory:

```python
from luma_memory.plugins.plugin_loader import load_plugins_from_directory

# Non-recursive (only files in the directory)
plugins = load_plugins_from_directory("path/to/plugins")

# Recursive (includes subdirectories)
plugins = load_plugins_from_directory("path/to/plugins", recursive=True)
```

### Method 3: Load from Package

Load plugins from an installed Python package:

```python
from luma_memory.plugins.plugin_loader import load_plugins_from_package

plugins = load_plugins_from_package("my_plugin_package")
```

### Method 4: Load Plugin Class Directly

If you have the plugin class already imported:

```python
from luma_memory.plugins.plugin_loader import PluginLoader
from my_module import MyCustomPlugin

loader = PluginLoader()
loader.load_plugin_class(MyCustomPlugin)
```

### Auto-Registration

By default, plugins are automatically registered when loaded. To disable auto-registration:

```python
loader = PluginLoader()
plugin_classes = loader.load_from_file("my_plugin.py", auto_register=False)

# Manually register later
for plugin_class in plugin_classes:
    plugin_instance = plugin_class()
    loader.registry.register(plugin_instance)
```

---

## Plugin Registry

### Getting the Global Registry

```python
from luma_memory.plugins.plugin_interface import get_global_registry

registry = get_global_registry()
```

### Registry Operations

#### List All Plugins

```python
plugins = registry.list_plugins()
for plugin_info in plugins:
    print(f"{plugin_info['name']} v{plugin_info['version']}")
    print(f"  Actions: {plugin_info['supported_actions']}")
```

#### Get Plugin by Name

```python
plugin = registry.get_plugin("social_media")
if plugin:
    print(f"Found plugin: {plugin.name}")
```

#### Get Plugin for Action Type

```python
plugin = registry.get_plugin_for_action("tweet")
if plugin:
    print(f"Action 'tweet' is handled by: {plugin.name}")
```

#### Check if Action is Supported

```python
if registry.is_action_supported("custom_action"):
    print("Action is supported by a plugin")
```

#### Get All Supported Actions

```python
actions = registry.get_supported_actions()
print(f"Supported actions: {actions}")
```

#### Unregister a Plugin

```python
registry.unregister("plugin_name")
```

---

## Best Practices

### 1. Plugin Naming

- Use descriptive, lowercase names with underscores
- Good: `"social_media"`, `"calendar_events"`, `"email_tracker"`
- Avoid: `"plugin1"`, `"MyPlugin"`, `"sm"`

### 2. Action Type Naming

- Use clear, specific action names
- Include the domain/category in the name
- Good: `"email_sent"`, `"calendar_event_created"`, `"tweet_posted"`
- Avoid: `"action1"`, `"do_thing"`, `"process"`

### 3. Validation

- Validate early and provide clear error messages
- Check required fields in `validate_context()`
- Use `validate_entry()` for cross-field validation
- Return descriptive error messages

```python
def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
    if action == "email_sent":
        if "recipient" not in context:
            return False, "Email action requires 'recipient' field"
        
        recipient = context["recipient"]
        if not isinstance(recipient, str) or "@" not in recipient:
            return False, f"Invalid email address: {recipient}"
    
    return True, None
```

### 4. Error Handling

- Catch exceptions in processing methods
- Raise `PluginProcessingError` for processing failures
- Raise `PluginValidationError` for validation failures
- Log errors for debugging

```python
def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
    try:
        # Processing logic
        entry.context["processed"] = True
        return entry
    except Exception as e:
        raise PluginProcessingError(f"Failed to process entry: {e}") from e
```

### 5. Performance

- Keep processing lightweight
- Avoid expensive operations in validation
- Cache computed values when possible
- Use lazy loading for external resources

### 6. Backward Compatibility

- Handle missing context fields gracefully
- Provide defaults for new fields
- Version your plugin and document changes

```python
def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
    # Handle both old and new context formats
    if "old_field" in entry.context:
        entry.context["new_field"] = entry.context["old_field"]
    return entry
```

### 7. Testing

- Write unit tests for each plugin method
- Test validation with valid and invalid data
- Test processing with edge cases
- Test integration with the memory manager

---

## Example Plugin

Here's a complete example plugin for tracking calendar events:

```python
from typing import Dict, Any, Optional, List
from datetime import datetime

from luma_memory.plugins.plugin_interface import (
    MemoryEntryPlugin,
    PluginValidationError
)
from luma_memory.models import MemoryEntry, SensitivityLevel


class CalendarPlugin(MemoryEntryPlugin):
    """Plugin for handling calendar event memory entries."""
    
    @property
    def name(self) -> str:
        return "calendar_events"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["event_created", "event_updated", "event_deleted", "event_attended"]
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """Validate calendar event context."""
        # All calendar actions require event_id
        if "event_id" not in context:
            return False, "Calendar actions require 'event_id' field"
        
        if action in ["event_created", "event_updated"]:
            # These actions require event details
            required_fields = ["title", "start_time", "end_time"]
            for field in required_fields:
                if field not in context:
                    return False, f"Missing required field: {field}"
            
            # Validate time fields
            try:
                start = datetime.fromisoformat(context["start_time"])
                end = datetime.fromisoformat(context["end_time"])
                
                if end <= start:
                    return False, "Event end_time must be after start_time"
            except (ValueError, TypeError) as e:
                return False, f"Invalid datetime format: {e}"
        
        return True, None
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        """Enrich calendar event entry."""
        context = entry.context
        
        # Calculate event duration
        if "start_time" in context and "end_time" in context:
            try:
                start = datetime.fromisoformat(context["start_time"])
                end = datetime.fromisoformat(context["end_time"])
                duration_minutes = int((end - start).total_seconds() / 60)
                context["duration_minutes"] = duration_minutes
            except (ValueError, TypeError):
                pass
        
        # Extract attendee count
        if "attendees" in context and isinstance(context["attendees"], list):
            context["attendee_count"] = len(context["attendees"])
        
        # Add calendar metadata
        context["calendar_type"] = context.get("calendar_type", "personal")
        
        return entry
    
    def get_default_sensitivity(self, action: str) -> Optional[SensitivityLevel]:
        """Calendar events are typically private."""
        return SensitivityLevel.PRIVATE
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Generate tags for calendar events."""
        tags = ["calendar", action]
        
        # Add calendar type tag
        if "calendar_type" in context:
            tags.append(f"calendar:{context['calendar_type']}")
        
        # Add duration-based tags
        if "duration_minutes" in context:
            duration = context["duration_minutes"]
            if duration < 30:
                tags.append("short_event")
            elif duration > 240:  # 4 hours
                tags.append("long_event")
        
        # Add attendee tags
        if "attendee_count" in context:
            count = context["attendee_count"]
            if count == 0:
                tags.append("solo")
            elif count > 10:
                tags.append("large_meeting")
        
        return tags
    
    def should_summarize(self, entries: List[MemoryEntry]) -> bool:
        """Summarize if many events in a short time."""
        # Summarize if more than 30 events
        return len(entries) > 30
```

---

## Testing Plugins

### Unit Test Example

```python
import pytest
from datetime import datetime
from luma_memory.models import MemoryEntry, SensitivityLevel, SyncStatus
from my_plugins import CalendarPlugin


def test_calendar_plugin_validation():
    """Test calendar plugin context validation."""
    plugin = CalendarPlugin()
    
    # Valid context
    valid_context = {
        "event_id": "evt_123",
        "title": "Team Meeting",
        "start_time": "2024-01-15T10:00:00",
        "end_time": "2024-01-15T11:00:00"
    }
    is_valid, error = plugin.validate_context(valid_context, "event_created")
    assert is_valid is True
    assert error is None
    
    # Missing required field
    invalid_context = {
        "event_id": "evt_123",
        "title": "Team Meeting"
    }
    is_valid, error = plugin.validate_context(invalid_context, "event_created")
    assert is_valid is False
    assert "start_time" in error or "end_time" in error


def test_calendar_plugin_processing():
    """Test calendar plugin entry processing."""
    plugin = CalendarPlugin()
    
    entry = MemoryEntry(
        id="mem_123",
        timestamp=datetime.utcnow(),
        action="event_created",
        context={
            "event_id": "evt_123",
            "title": "Team Meeting",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00",
            "attendees": ["alice@example.com", "bob@example.com"]
        },
        sensitivity=SensitivityLevel.PRIVATE,
        device_id="device_1",
        sync_status=SyncStatus.PENDING,
        tags=[]
    )
    
    processed = plugin.process_before_storage(entry)
    
    # Check computed fields
    assert "duration_minutes" in processed.context
    assert processed.context["duration_minutes"] == 60
    assert "attendee_count" in processed.context
    assert processed.context["attendee_count"] == 2


def test_calendar_plugin_tags():
    """Test automatic tag generation."""
    plugin = CalendarPlugin()
    
    context = {
        "event_id": "evt_123",
        "calendar_type": "work",
        "duration_minutes": 15,
        "attendee_count": 0
    }
    
    tags = plugin.get_default_tags("event_created", context)
    
    assert "calendar" in tags
    assert "event_created" in tags
    assert "calendar:work" in tags
    assert "short_event" in tags
    assert "solo" in tags
```

### Integration Test Example

```python
def test_plugin_integration_with_memory_manager():
    """Test plugin integration with memory manager."""
    from luma_memory.memory_manager import MemoryManager
    from luma_memory.plugins.plugin_loader import PluginLoader
    from my_plugins import CalendarPlugin
    
    # Load plugin
    loader = PluginLoader()
    loader.load_plugin_class(CalendarPlugin)
    
    # Create memory manager
    manager = MemoryManager()
    
    # Create entry with calendar action
    entry_id = manager.create_memory(
        action="event_created",
        context={
            "event_id": "evt_123",
            "title": "Team Meeting",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T11:00:00"
        },
        device_id="device_1"
    )
    
    # Retrieve and verify
    entry = manager.get_memory(entry_id)
    assert entry is not None
    assert "duration_minutes" in entry.context
    assert "calendar" in entry.tags
```

---

## Troubleshooting

### Plugin Not Loading

**Problem**: Plugin file is not being loaded

**Solutions**:
- Check that the file has a `.py` extension
- Ensure the plugin class inherits from `MemoryEntryPlugin`
- Verify the plugin class is not abstract
- Check for syntax errors in the plugin file
- Enable debug logging: `logging.getLogger("luma_memory.plugins").setLevel(logging.DEBUG)`

### Action Type Conflicts

**Problem**: `ValueError: Action type 'X' is already handled by plugin 'Y'`

**Solutions**:
- Each action type can only be handled by one plugin
- Choose unique action names for your plugin
- Unregister the conflicting plugin if you want to replace it

### Validation Errors

**Problem**: Entries are being rejected unexpectedly

**Solutions**:
- Check the error message returned by `validate_context()`
- Ensure all required fields are present in the context
- Verify field types match expectations
- Test validation with sample data

### Processing Errors

**Problem**: `PluginProcessingError` is raised

**Solutions**:
- Add try-except blocks in processing methods
- Log errors for debugging
- Return the original entry if processing fails
- Check for missing or malformed context fields

---

## Additional Resources

- [API Documentation](API_DOCUMENTATION.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Example Plugins](../luma_memory/plugins/)
- [Plugin Interface Source](../luma_memory/plugins/plugin_interface.py)
- [Plugin Loader Source](../luma_memory/plugins/plugin_loader.py)

---

## Support

For questions or issues with plugin development:

1. Check the example plugins in `luma_memory/plugins/`
2. Review the plugin interface documentation
3. Enable debug logging to see plugin loading details
4. Open an issue on the project repository

---

**Last Updated**: 2024-01-15  
**Plugin System Version**: 1.0.0
