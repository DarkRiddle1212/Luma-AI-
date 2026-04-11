"""
Plugin interface for extending memory entry types.

This module defines the abstract base class for memory entry plugins
and provides a registry for managing plugin instances.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

from ..models import MemoryEntry, SensitivityLevel, SyncStatus


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass


class PluginProcessingError(Exception):
    """Raised when plugin processing fails."""
    pass


class MemoryEntryPlugin(ABC):
    """
    Abstract base class for memory entry plugins.
    
    Plugins extend the functionality of the memory module by providing
    custom validation, processing, and serialization for specific entry types.
    
    Each plugin is associated with one or more action types and can customize
    how entries of those types are validated, processed, and stored.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for the plugin.
        
        Returns:
            Plugin name (e.g., "social_media", "calendar_event")
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Plugin version string.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        pass
    
    @property
    @abstractmethod
    def supported_actions(self) -> List[str]:
        """
        List of action types this plugin handles.
        
        Returns:
            List of action type strings (e.g., ["tweet", "facebook_post"])
        """
        pass
    
    def validate_entry(self, entry: MemoryEntry) -> tuple[bool, Optional[str]]:
        """
        Validate a memory entry according to plugin-specific rules.
        
        This method is called after the core validation passes. Plugins can
        add additional validation logic specific to their entry types.
        
        Args:
            entry: The memory entry to validate
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        
        Raises:
            PluginValidationError: If validation fails critically
        """
        return True, None
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """
        Validate the context dictionary for a specific action type.
        
        Plugins can enforce required fields and data types in the context
        based on the action type.
        
        Args:
            context: The context dictionary to validate
            action: The action type
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        return True, None
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Process an entry before it is stored.
        
        Plugins can modify, enrich, or transform entries before storage.
        For example, extracting metadata, normalizing data, or adding
        computed fields.
        
        Args:
            entry: The memory entry to process
        
        Returns:
            Processed memory entry (may be the same instance or a new one)
        
        Raises:
            PluginProcessingError: If processing fails
        """
        return entry
    
    def process_after_retrieval(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Process an entry after it is retrieved from storage.
        
        Plugins can transform or enrich entries after retrieval.
        For example, resolving references, computing derived values,
        or formatting data.
        
        Args:
            entry: The memory entry to process
        
        Returns:
            Processed memory entry (may be the same instance or a new one)
        
        Raises:
            PluginProcessingError: If processing fails
        """
        return entry
    
    def serialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize context data for storage.
        
        Plugins can customize how context data is serialized, handling
        special data types or structures specific to their entry types.
        
        Args:
            context: The context dictionary to serialize
        
        Returns:
            Serialized context dictionary
        """
        return context
    
    def deserialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize context data after retrieval.
        
        Plugins can customize how context data is deserialized, reconstructing
        special data types or structures specific to their entry types.
        
        Args:
            context: The serialized context dictionary
        
        Returns:
            Deserialized context dictionary
        """
        return context
    
    def get_default_sensitivity(self, action: str) -> Optional[SensitivityLevel]:
        """
        Get the default sensitivity level for an action type.
        
        Plugins can specify default sensitivity levels for their action types.
        
        Args:
            action: The action type
        
        Returns:
            Default sensitivity level, or None to use system default
        """
        return None
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate default tags for an entry.
        
        Plugins can automatically generate tags based on the action type
        and context data.
        
        Args:
            action: The action type
            context: The context dictionary
        
        Returns:
            List of default tags
        """
        return []
    
    def should_summarize(self, entries: List[MemoryEntry]) -> bool:
        """
        Determine if a group of entries should be summarized.
        
        Plugins can provide custom logic for when entries of their types
        should be summarized.
        
        Args:
            entries: List of memory entries to consider
        
        Returns:
            True if entries should be summarized, False otherwise
        """
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Dictionary containing plugin metadata (name, version, description, etc.)
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_actions": self.supported_actions,
        }


class PluginRegistry:
    """
    Registry for managing memory entry plugins.
    
    The registry maintains a collection of plugins and routes operations
    to the appropriate plugin based on action types.
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, MemoryEntryPlugin] = {}
        self._action_to_plugin: Dict[str, str] = {}
    
    def register(self, plugin: MemoryEntryPlugin) -> None:
        """
        Register a plugin with the registry.
        
        Args:
            plugin: The plugin instance to register
        
        Raises:
            ValueError: If a plugin with the same name is already registered
                       or if action types conflict with existing plugins
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already registered")
        
        # Check for action type conflicts
        for action in plugin.supported_actions:
            if action in self._action_to_plugin:
                existing_plugin = self._action_to_plugin[action]
                raise ValueError(
                    f"Action type '{action}' is already handled by plugin '{existing_plugin}'"
                )
        
        # Register the plugin
        self._plugins[plugin.name] = plugin
        
        # Map action types to plugin
        for action in plugin.supported_actions:
            self._action_to_plugin[action] = plugin.name
    
    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_name: Name of the plugin to unregister
        
        Raises:
            KeyError: If the plugin is not registered
        """
        if plugin_name not in self._plugins:
            raise KeyError(f"Plugin '{plugin_name}' is not registered")
        
        plugin = self._plugins[plugin_name]
        
        # Remove action type mappings
        for action in plugin.supported_actions:
            del self._action_to_plugin[action]
        
        # Remove plugin
        del self._plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[MemoryEntryPlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
        
        Returns:
            Plugin instance, or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def get_plugin_for_action(self, action: str) -> Optional[MemoryEntryPlugin]:
        """
        Get the plugin that handles a specific action type.
        
        Args:
            action: The action type
        
        Returns:
            Plugin instance, or None if no plugin handles this action
        """
        plugin_name = self._action_to_plugin.get(action)
        if plugin_name:
            return self._plugins.get(plugin_name)
        return None
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with their metadata.
        
        Returns:
            List of plugin metadata dictionaries
        """
        return [plugin.get_metadata() for plugin in self._plugins.values()]
    
    def is_action_supported(self, action: str) -> bool:
        """
        Check if an action type is supported by any plugin.
        
        Args:
            action: The action type to check
        
        Returns:
            True if a plugin handles this action, False otherwise
        """
        return action in self._action_to_plugin
    
    def get_supported_actions(self) -> List[str]:
        """
        Get all action types supported by registered plugins.
        
        Returns:
            List of supported action types
        """
        return list(self._action_to_plugin.keys())
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._action_to_plugin.clear()


# Global plugin registry instance
_global_registry = PluginRegistry()


def get_global_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        The global PluginRegistry instance
    """
    return _global_registry
