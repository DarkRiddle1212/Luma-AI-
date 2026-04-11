"""
Plugin loader for dynamically loading memory entry plugins.

This module provides functionality to discover, load, and register
plugins from a specified directory or package.
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import List, Optional, Type, Dict, Any

from .plugin_interface import MemoryEntryPlugin, PluginRegistry, get_global_registry


logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when plugin loading fails."""
    pass


class PluginLoader:
    """
    Loader for dynamically discovering and loading memory entry plugins.
    
    The loader can discover plugins from:
    - A directory containing Python modules
    - A Python package
    - Individual Python files
    
    Plugins are identified by finding classes that inherit from MemoryEntryPlugin
    and are not abstract.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the plugin loader.
        
        Args:
            registry: Plugin registry to use. If None, uses the global registry.
        """
        self.registry = registry or get_global_registry()
        self._loaded_modules: Dict[str, Any] = {}
    
    def load_from_directory(
        self,
        directory: str,
        recursive: bool = False,
        auto_register: bool = True
    ) -> List[Type[MemoryEntryPlugin]]:
        """
        Load plugins from a directory.
        
        Scans the directory for Python files and loads any plugin classes found.
        
        Args:
            directory: Path to the directory containing plugin modules
            recursive: If True, scan subdirectories recursively
            auto_register: If True, automatically register loaded plugins
        
        Returns:
            List of loaded plugin classes
        
        Raises:
            PluginLoadError: If directory doesn't exist or loading fails
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise PluginLoadError(f"Plugin directory does not exist: {directory}")
        
        if not dir_path.is_dir():
            raise PluginLoadError(f"Path is not a directory: {directory}")
        
        logger.info(f"Loading plugins from directory: {directory}")
        
        plugin_classes = []
        
        # Find all Python files
        if recursive:
            python_files = dir_path.rglob("*.py")
        else:
            python_files = dir_path.glob("*.py")
        
        for file_path in python_files:
            # Skip __init__.py and private modules
            if file_path.name.startswith("_"):
                continue
            
            try:
                classes = self.load_from_file(str(file_path), auto_register=auto_register)
                plugin_classes.extend(classes)
            except Exception as e:
                logger.warning(f"Failed to load plugins from {file_path}: {e}")
        
        logger.info(f"Loaded {len(plugin_classes)} plugin(s) from {directory}")
        return plugin_classes
    
    def load_from_file(
        self,
        file_path: str,
        auto_register: bool = True
    ) -> List[Type[MemoryEntryPlugin]]:
        """
        Load plugins from a Python file.
        
        Args:
            file_path: Path to the Python file
            auto_register: If True, automatically register loaded plugins
        
        Returns:
            List of loaded plugin classes
        
        Raises:
            PluginLoadError: If file doesn't exist or loading fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise PluginLoadError(f"Plugin file does not exist: {file_path}")
        
        if not path.is_file():
            raise PluginLoadError(f"Path is not a file: {file_path}")
        
        if path.suffix != ".py":
            raise PluginLoadError(f"File is not a Python file: {file_path}")
        
        logger.debug(f"Loading plugins from file: {file_path}")
        
        # Generate module name from file path
        module_name = f"luma_memory_plugin_{path.stem}_{id(path)}"
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Failed to create module spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            self._loaded_modules[module_name] = module
            
            # Add to sys.modules so imports work
            sys.modules[module_name] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            plugin_classes = self._find_plugin_classes(module)
            
            # Auto-register if requested
            if auto_register:
                for plugin_class in plugin_classes:
                    try:
                        plugin_instance = plugin_class()
                        self.registry.register(plugin_instance)
                        logger.info(f"Registered plugin: {plugin_instance.name}")
                    except Exception as e:
                        logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            
            return plugin_classes
            
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin from {file_path}: {e}") from e
    
    def load_from_package(
        self,
        package_name: str,
        auto_register: bool = True
    ) -> List[Type[MemoryEntryPlugin]]:
        """
        Load plugins from a Python package.
        
        Args:
            package_name: Name of the package to load (e.g., "my_plugins")
            auto_register: If True, automatically register loaded plugins
        
        Returns:
            List of loaded plugin classes
        
        Raises:
            PluginLoadError: If package cannot be imported or loading fails
        """
        logger.debug(f"Loading plugins from package: {package_name}")
        
        try:
            # Import the package
            module = importlib.import_module(package_name)
            self._loaded_modules[package_name] = module
            
            # Find plugin classes in the module
            plugin_classes = self._find_plugin_classes(module)
            
            # Auto-register if requested
            if auto_register:
                for plugin_class in plugin_classes:
                    try:
                        plugin_instance = plugin_class()
                        self.registry.register(plugin_instance)
                        logger.info(f"Registered plugin: {plugin_instance.name}")
                    except Exception as e:
                        logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            
            return plugin_classes
            
        except ImportError as e:
            raise PluginLoadError(f"Failed to import package {package_name}: {e}") from e
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugins from package {package_name}: {e}") from e
    
    def load_plugin_class(
        self,
        plugin_class: Type[MemoryEntryPlugin],
        auto_register: bool = True
    ) -> Type[MemoryEntryPlugin]:
        """
        Load a specific plugin class.
        
        Args:
            plugin_class: The plugin class to load
            auto_register: If True, automatically register the plugin
        
        Returns:
            The plugin class
        
        Raises:
            PluginLoadError: If the class is not a valid plugin or registration fails
        """
        if not self._is_plugin_class(plugin_class):
            raise PluginLoadError(f"{plugin_class.__name__} is not a valid plugin class")
        
        if auto_register:
            try:
                plugin_instance = plugin_class()
                self.registry.register(plugin_instance)
                logger.info(f"Registered plugin: {plugin_instance.name}")
            except Exception as e:
                raise PluginLoadError(f"Failed to register plugin {plugin_class.__name__}: {e}") from e
        
        return plugin_class
    
    def _find_plugin_classes(self, module: Any) -> List[Type[MemoryEntryPlugin]]:
        """
        Find all plugin classes in a module.
        
        Args:
            module: The module to search
        
        Returns:
            List of plugin classes found in the module
        """
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self._is_plugin_class(obj):
                plugin_classes.append(obj)
                logger.debug(f"Found plugin class: {name}")
        
        return plugin_classes
    
    def _is_plugin_class(self, cls: Type) -> bool:
        """
        Check if a class is a valid plugin class.
        
        A valid plugin class:
        - Inherits from MemoryEntryPlugin
        - Is not MemoryEntryPlugin itself
        - Is not abstract
        
        Args:
            cls: The class to check
        
        Returns:
            True if the class is a valid plugin class
        """
        try:
            # Check if it's a subclass of MemoryEntryPlugin
            if not issubclass(cls, MemoryEntryPlugin):
                return False
            
            # Exclude MemoryEntryPlugin itself
            if cls is MemoryEntryPlugin:
                return False
            
            # Check if it's abstract
            if inspect.isabstract(cls):
                return False
            
            return True
            
        except TypeError:
            # issubclass raises TypeError if cls is not a class
            return False
    
    def get_loaded_modules(self) -> Dict[str, Any]:
        """
        Get all modules loaded by this loader.
        
        Returns:
            Dictionary mapping module names to module objects
        """
        return self._loaded_modules.copy()
    
    def unload_all(self) -> None:
        """
        Unload all plugins loaded by this loader.
        
        This removes plugins from the registry and clears loaded modules.
        """
        # Get all plugin names from registry
        plugin_names = [plugin.name for plugin in self.registry._plugins.values()]
        
        # Unregister all plugins
        for plugin_name in plugin_names:
            try:
                self.registry.unregister(plugin_name)
                logger.info(f"Unregistered plugin: {plugin_name}")
            except KeyError:
                pass
        
        # Clear loaded modules
        for module_name in list(self._loaded_modules.keys()):
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        self._loaded_modules.clear()


def load_plugins_from_directory(
    directory: str,
    recursive: bool = False,
    registry: Optional[PluginRegistry] = None
) -> List[Type[MemoryEntryPlugin]]:
    """
    Convenience function to load plugins from a directory.
    
    Args:
        directory: Path to the directory containing plugin modules
        recursive: If True, scan subdirectories recursively
        registry: Plugin registry to use. If None, uses the global registry.
    
    Returns:
        List of loaded plugin classes
    
    Raises:
        PluginLoadError: If loading fails
    """
    loader = PluginLoader(registry=registry)
    return loader.load_from_directory(directory, recursive=recursive, auto_register=True)


def load_plugins_from_package(
    package_name: str,
    registry: Optional[PluginRegistry] = None
) -> List[Type[MemoryEntryPlugin]]:
    """
    Convenience function to load plugins from a package.
    
    Args:
        package_name: Name of the package to load
        registry: Plugin registry to use. If None, uses the global registry.
    
    Returns:
        List of loaded plugin classes
    
    Raises:
        PluginLoadError: If loading fails
    """
    loader = PluginLoader(registry=registry)
    return loader.load_from_package(package_name, auto_register=True)
