"""
Example demonstrating plugin registration functionality.

This example shows different ways to register plugins with the Luma Memory Module:
1. Using the @register_plugin decorator
2. Using the register() convenience function
3. Using the PluginLoader for dynamic loading
"""

from typing import List, Dict, Any, Optional
from luma_memory.plugins import (
    MemoryEntryPlugin,
    register_plugin,
    register,
    unregister,
    get_plugin,
    list_plugins,
    get_global_registry,
)
from luma_memory.models import MemoryEntry, SensitivityLevel


# Example 1: Using the @register_plugin decorator
# This automatically registers the plugin when the module is imported
@register_plugin
class EmailPlugin(MemoryEntryPlugin):
    """Plugin for handling email-related memory entries."""
    
    @property
    def name(self) -> str:
        return "email"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["email_sent", "email_received", "email_drafted"]
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """Validate email context."""
        required_fields = ["subject", "recipient"]
        
        for field in required_fields:
            if field not in context:
                return False, f"Email must have '{field}' field"
        
        return True, None
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Generate default tags for email entries."""
        tags = ["email", action]
        
        if "priority" in context:
            tags.append(f"priority:{context['priority']}")
        
        return tags


# Example 2: Plugin without decorator (manual registration)
class CalendarPlugin(MemoryEntryPlugin):
    """Plugin for handling calendar event memory entries."""
    
    @property
    def name(self) -> str:
        return "calendar"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["event_created", "event_updated", "event_deleted", "meeting_scheduled"]
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """Validate calendar event context."""
        if "title" not in context:
            return False, "Calendar event must have 'title' field"
        
        if action == "meeting_scheduled":
            if "attendees" not in context:
                return False, "Meeting must have 'attendees' field"
        
        return True, None
    
    def get_default_sensitivity(self, action: str) -> Optional[SensitivityLevel]:
        """Calendar events are typically private."""
        return SensitivityLevel.PRIVATE
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Generate default tags for calendar entries."""
        tags = ["calendar", action]
        
        if "event_type" in context:
            tags.append(f"type:{context['event_type']}")
        
        if action == "meeting_scheduled":
            tags.append("meeting")
        
        return tags


# Example 3: Plugin with custom processing
class TaskPlugin(MemoryEntryPlugin):
    """Plugin for handling task management memory entries."""
    
    @property
    def name(self) -> str:
        return "task"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["task_created", "task_completed", "task_updated", "task_deleted"]
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        """Add computed fields to task entries."""
        context = entry.context
        
        # Add status field based on action
        if entry.action == "task_completed":
            context["status"] = "completed"
        elif entry.action == "task_created":
            context["status"] = "pending"
        
        # Add priority if not specified
        if "priority" not in context:
            context["priority"] = "medium"
        
        return entry
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Generate default tags for task entries."""
        tags = ["task", action]
        
        if "priority" in context:
            tags.append(f"priority:{context['priority']}")
        
        if "project" in context:
            tags.append(f"project:{context['project']}")
        
        return tags


def main():
    """Demonstrate plugin registration functionality."""
    print("=== Plugin Registration Example ===\n")
    
    # Get the global registry
    registry = get_global_registry()
    
    # Example 1: EmailPlugin is already registered via decorator
    print("1. Decorator-based registration:")
    email_plugin = get_plugin("email")
    if email_plugin:
        print(f"   ✓ {email_plugin.name} v{email_plugin.version} is registered")
        print(f"   Supported actions: {', '.join(email_plugin.supported_actions)}")
    print()
    
    # Example 2: Manually register CalendarPlugin
    print("2. Manual registration using register():")
    calendar_plugin = CalendarPlugin()
    register(calendar_plugin)
    print(f"   ✓ {calendar_plugin.name} v{calendar_plugin.version} registered")
    print(f"   Supported actions: {', '.join(calendar_plugin.supported_actions)}")
    print()
    
    # Example 3: Register TaskPlugin
    print("3. Register TaskPlugin:")
    task_plugin = TaskPlugin()
    register(task_plugin)
    print(f"   ✓ {task_plugin.name} v{task_plugin.version} registered")
    print(f"   Supported actions: {', '.join(task_plugin.supported_actions)}")
    print()
    
    # List all registered plugins
    print("4. List all registered plugins:")
    all_plugins = list_plugins()
    for plugin_info in all_plugins:
        print(f"   - {plugin_info['name']} v{plugin_info['version']}")
        print(f"     Actions: {', '.join(plugin_info['supported_actions'])}")
    print()
    
    # Demonstrate plugin retrieval
    print("5. Retrieve specific plugin:")
    retrieved = get_plugin("calendar")
    if retrieved:
        print(f"   ✓ Retrieved: {retrieved.name}")
        print(f"   Default sensitivity: {retrieved.get_default_sensitivity('event_created')}")
    print()
    
    # Demonstrate unregistration
    print("6. Unregister a plugin:")
    unregister("task")
    print("   ✓ TaskPlugin unregistered")
    
    # Verify it's gone
    task_check = get_plugin("task")
    if task_check is None:
        print("   ✓ Confirmed: TaskPlugin is no longer registered")
    print()
    
    # Show remaining plugins
    print("7. Remaining plugins:")
    remaining = list_plugins()
    for plugin_info in remaining:
        print(f"   - {plugin_info['name']}")
    print()
    
    # Demonstrate plugin validation
    print("8. Test plugin validation:")
    email_plugin = get_plugin("email")
    if email_plugin:
        # Valid context
        valid_context = {"subject": "Test", "recipient": "user@example.com"}
        is_valid, error = email_plugin.validate_context(valid_context, "email_sent")
        print(f"   Valid context: {is_valid} (error: {error})")
        
        # Invalid context
        invalid_context = {"subject": "Test"}  # Missing recipient
        is_valid, error = email_plugin.validate_context(invalid_context, "email_sent")
        print(f"   Invalid context: {is_valid} (error: {error})")
    print()
    
    # Demonstrate tag generation
    print("9. Test automatic tag generation:")
    calendar_plugin = get_plugin("calendar")
    if calendar_plugin:
        context = {"title": "Team Meeting", "event_type": "meeting"}
        tags = calendar_plugin.get_default_tags("meeting_scheduled", context)
        print(f"   Generated tags: {', '.join(tags)}")
    print()
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
