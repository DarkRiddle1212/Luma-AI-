"""
Luma Memory Module Plugin System.

This package provides a plugin system for extending memory entry types
with custom validation, processing, and metadata.
"""

from .plugin_interface import (
    MemoryEntryPlugin,
    PluginRegistry,
    PluginValidationError,
    PluginProcessingError,
    get_global_registry,
)

from .plugin_loader import (
    PluginLoader,
    PluginLoadError,
    load_plugins_from_directory,
    load_plugins_from_package,
)

# Plugin registration decorator and convenience functions
from typing import Type, Optional
import functools


def register_plugin(
    plugin_class: Optional[Type[MemoryEntryPlugin]] = None,
    *,
    registry: Optional[PluginRegistry] = None
):
    """
    Decorator for registering a plugin class.

    This decorator can be used to automatically register a plugin class
    with the global registry (or a specified registry) when the module
    is imported.

    Usage:
        @register_plugin
        class MyPlugin(MemoryEntryPlugin):
            ...

        # Or with a specific registry:
        @register_plugin(registry=my_registry)
        class MyPlugin(MemoryEntryPlugin):
            ...

    Args:
        plugin_class: The plugin class to register (when used without parentheses)
        registry: Optional registry to use. If None, uses the global registry.

    Returns:
        The decorated class (unchanged)

    Raises:
        ValueError: If the plugin cannot be registered
    """
    def decorator(cls: Type[MemoryEntryPlugin]) -> Type[MemoryEntryPlugin]:
        target_registry = registry or get_global_registry()

        # Create an instance and register it
        try:
            plugin_instance = cls()
            target_registry.register(plugin_instance)
        except Exception as e:
            raise ValueError(f"Failed to register plugin {cls.__name__}: {e}") from e

        return cls

    # Handle both @register_plugin and @register_plugin()
    if plugin_class is None:
        # Called with parentheses: @register_plugin()
        return decorator
    else:
        # Called without parentheses: @register_plugin
        return decorator(plugin_class)


def register(
    plugin: MemoryEntryPlugin,
    registry: Optional[PluginRegistry] = None
) -> None:
    """
    Register a plugin instance.

    Convenience function for registering a plugin instance with the
    global registry or a specified registry.

    Args:
        plugin: The plugin instance to register
        registry: Optional registry to use. If None, uses the global registry.

    Raises:
        ValueError: If the plugin cannot be registered

    Example:
        plugin = MyPlugin()
        register(plugin)
    """
    target_registry = registry or get_global_registry()
    target_registry.register(plugin)


def unregister(
    plugin_name: str,
    registry: Optional[PluginRegistry] = None
) -> None:
    """
    Unregister a plugin by name.

    Convenience function for unregistering a plugin from the
    global registry or a specified registry.

    Args:
        plugin_name: Name of the plugin to unregister
        registry: Optional registry to use. If None, uses the global registry.

    Raises:
        KeyError: If the plugin is not registered

    Example:
        unregister("my_plugin")
    """
    target_registry = registry or get_global_registry()
    target_registry.unregister(plugin_name)


def get_plugin(
    plugin_name: str,
    registry: Optional[PluginRegistry] = None
) -> Optional[MemoryEntryPlugin]:
    """
    Get a plugin by name.

    Convenience function for retrieving a plugin from the
    global registry or a specified registry.

    Args:
        plugin_name: Name of the plugin to retrieve
        registry: Optional registry to use. If None, uses the global registry.

    Returns:
        The plugin instance, or None if not found

    Example:
        plugin = get_plugin("my_plugin")
    """
    target_registry = registry or get_global_registry()
    return target_registry.get_plugin(plugin_name)


def list_plugins(
    registry: Optional[PluginRegistry] = None
) -> list:
    """
    List all registered plugins.

    Convenience function for listing all plugins in the
    global registry or a specified registry.

    Args:
        registry: Optional registry to use. If None, uses the global registry.

    Returns:
        List of plugin metadata dictionaries

    Example:
        plugins = list_plugins()
        for plugin_info in plugins:
            print(f"{plugin_info['name']} v{plugin_info['version']}")
    """
    target_registry = registry or get_global_registry()
    return target_registry.list_plugins()


__all__ = [
    # Core classes
    "MemoryEntryPlugin",
    "PluginRegistry",
    "PluginLoader",

    # Exceptions
    "PluginValidationError",
    "PluginProcessingError",
    "PluginLoadError",

    # Registry functions
    "get_global_registry",
    "register_plugin",
    "register",
    "unregister",
    "get_plugin",
    "list_plugins",

    # Loader functions
    "load_plugins_from_directory",
    "load_plugins_from_package",
]
