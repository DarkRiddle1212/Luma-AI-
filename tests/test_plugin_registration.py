"""
Tests for plugin registration functionality.

This module tests the decorator-based registration and convenience functions
for registering plugins with the plugin registry.
"""

import pytest
from typing import List

from luma_memory.plugins import (
    MemoryEntryPlugin,
    PluginRegistry,
    register_plugin,
    register,
    unregister,
    get_plugin,
    list_plugins,
    get_global_registry,
)


class TestPluginRegistration:
    """Test suite for plugin registration functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh plugin registry for testing."""
        return PluginRegistry()
    
    @pytest.fixture
    def sample_plugin_class(self):
        """Create a sample plugin class for testing."""
        class SamplePlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "sample_plugin"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["sample_action"]
        
        return SamplePlugin
    
    def test_register_plugin_decorator_without_parentheses(self, registry):
        """Test @register_plugin decorator without parentheses."""
        @register_plugin(registry=registry)
        class TestPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "test_plugin"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["test_action"]
        
        # Check that plugin was registered
        plugin = registry.get_plugin("test_plugin")
        assert plugin is not None
        assert plugin.name == "test_plugin"
        assert plugin.version == "1.0.0"
    
    def test_register_plugin_decorator_with_parentheses(self, registry):
        """Test @register_plugin() decorator with parentheses."""
        @register_plugin(registry=registry)
        class TestPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "test_plugin_2"
            
            @property
            def version(self) -> str:
                return "2.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["test_action_2"]
        
        # Check that plugin was registered
        plugin = registry.get_plugin("test_plugin_2")
        assert plugin is not None
        assert plugin.name == "test_plugin_2"
        assert plugin.version == "2.0.0"
    
    def test_register_plugin_decorator_returns_class(self, registry):
        """Test that decorator returns the original class unchanged."""
        @register_plugin(registry=registry)
        class TestPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "test_plugin_3"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return []
        
        # Should be able to instantiate the class
        instance = TestPlugin()
        assert isinstance(instance, MemoryEntryPlugin)
        assert instance.name == "test_plugin_3"
    
    def test_register_plugin_decorator_duplicate_raises_error(self, registry):
        """Test that registering duplicate plugin raises error."""
        @register_plugin(registry=registry)
        class TestPlugin1(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "duplicate_plugin"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return []
        
        # Attempting to register another plugin with same name should fail
        with pytest.raises(ValueError, match="already registered"):
            @register_plugin(registry=registry)
            class TestPlugin2(MemoryEntryPlugin):
                @property
                def name(self) -> str:
                    return "duplicate_plugin"
                
                @property
                def version(self) -> str:
                    return "2.0.0"
                
                @property
                def supported_actions(self) -> List[str]:
                    return []
    
    def test_register_function(self, registry, sample_plugin_class):
        """Test register() convenience function."""
        plugin = sample_plugin_class()
        register(plugin, registry=registry)
        
        # Check that plugin was registered
        retrieved = registry.get_plugin("sample_plugin")
        assert retrieved is not None
        assert retrieved.name == "sample_plugin"
    
    def test_register_function_with_global_registry(self, sample_plugin_class):
        """Test register() with global registry."""
        global_registry = get_global_registry()
        
        # Clear global registry first
        global_registry.clear()
        
        plugin = sample_plugin_class()
        register(plugin)  # Should use global registry
        
        # Check that plugin was registered in global registry
        retrieved = global_registry.get_plugin("sample_plugin")
        assert retrieved is not None
        
        # Clean up
        global_registry.clear()
    
    def test_unregister_function(self, registry, sample_plugin_class):
        """Test unregister() convenience function."""
        plugin = sample_plugin_class()
        register(plugin, registry=registry)
        
        # Verify it's registered
        assert registry.get_plugin("sample_plugin") is not None
        
        # Unregister it
        unregister("sample_plugin", registry=registry)
        
        # Verify it's gone
        assert registry.get_plugin("sample_plugin") is None
    
    def test_unregister_nonexistent_raises_error(self, registry):
        """Test that unregistering nonexistent plugin raises error."""
        with pytest.raises(KeyError, match="not registered"):
            unregister("nonexistent_plugin", registry=registry)
    
    def test_get_plugin_function(self, registry, sample_plugin_class):
        """Test get_plugin() convenience function."""
        plugin = sample_plugin_class()
        register(plugin, registry=registry)
        
        # Retrieve using convenience function
        retrieved = get_plugin("sample_plugin", registry=registry)
        assert retrieved is not None
        assert retrieved.name == "sample_plugin"
    
    def test_get_plugin_nonexistent_returns_none(self, registry):
        """Test that getting nonexistent plugin returns None."""
        result = get_plugin("nonexistent_plugin", registry=registry)
        assert result is None
    
    def test_list_plugins_function(self, registry):
        """Test list_plugins() convenience function."""
        # Register multiple plugins
        class Plugin1(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "plugin_1"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["action_1"]
        
        class Plugin2(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "plugin_2"
            
            @property
            def version(self) -> str:
                return "2.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["action_2"]
        
        register(Plugin1(), registry=registry)
        register(Plugin2(), registry=registry)
        
        # List all plugins
        plugins = list_plugins(registry=registry)
        
        assert len(plugins) == 2
        plugin_names = [p["name"] for p in plugins]
        assert "plugin_1" in plugin_names
        assert "plugin_2" in plugin_names
    
    def test_list_plugins_empty_registry(self, registry):
        """Test listing plugins from empty registry."""
        plugins = list_plugins(registry=registry)
        assert plugins == []
    
    def test_register_plugin_with_conflicting_actions(self, registry):
        """Test that registering plugins with conflicting actions raises error."""
        @register_plugin(registry=registry)
        class Plugin1(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "plugin_1"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["shared_action"]
        
        # Attempting to register another plugin with same action should fail
        with pytest.raises(ValueError, match="already handled"):
            @register_plugin(registry=registry)
            class Plugin2(MemoryEntryPlugin):
                @property
                def name(self) -> str:
                    return "plugin_2"
                
                @property
                def version(self) -> str:
                    return "1.0.0"
                
                @property
                def supported_actions(self) -> List[str]:
                    return ["shared_action"]
    
    def test_decorator_with_initialization_error(self, registry):
        """Test decorator handles plugin initialization errors."""
        with pytest.raises(ValueError, match="Failed to register"):
            @register_plugin(registry=registry)
            class BadPlugin(MemoryEntryPlugin):
                def __init__(self):
                    raise RuntimeError("Initialization failed")
                
                @property
                def name(self) -> str:
                    return "bad_plugin"
                
                @property
                def version(self) -> str:
                    return "1.0.0"
                
                @property
                def supported_actions(self) -> List[str]:
                    return []
    
    def test_multiple_registrations_and_unregistrations(self, registry):
        """Test multiple register/unregister cycles."""
        class TestPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "cycle_plugin"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["cycle_action"]
        
        # Register
        plugin1 = TestPlugin()
        register(plugin1, registry=registry)
        assert get_plugin("cycle_plugin", registry=registry) is not None
        
        # Unregister
        unregister("cycle_plugin", registry=registry)
        assert get_plugin("cycle_plugin", registry=registry) is None
        
        # Register again
        plugin2 = TestPlugin()
        register(plugin2, registry=registry)
        assert get_plugin("cycle_plugin", registry=registry) is not None
        
        # Unregister again
        unregister("cycle_plugin", registry=registry)
        assert get_plugin("cycle_plugin", registry=registry) is None
    
    def test_register_function_validates_duplicate(self, registry, sample_plugin_class):
        """Test that register() function validates duplicate registrations."""
        plugin1 = sample_plugin_class()
        register(plugin1, registry=registry)
        
        # Attempting to register again should fail
        plugin2 = sample_plugin_class()
        with pytest.raises(ValueError, match="already registered"):
            register(plugin2, registry=registry)
