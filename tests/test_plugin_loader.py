"""
Tests for the plugin loader functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path

from luma_memory.plugins import (
    PluginLoader,
    PluginLoadError,
    MemoryEntryPlugin,
    PluginRegistry,
    load_plugins_from_directory,
)
from luma_memory.models import MemoryEntry, SensitivityLevel


class TestPluginLoader:
    """Test suite for PluginLoader."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh plugin registry for testing."""
        return PluginRegistry()
    
    @pytest.fixture
    def loader(self, registry):
        """Create a plugin loader with a test registry."""
        return PluginLoader(registry=registry)
    
    @pytest.fixture
    def sample_plugin_code(self):
        """Sample plugin code for testing."""
        return '''
from luma_memory.plugins import MemoryEntryPlugin
from typing import List

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
'''
    
    def test_loader_initialization(self, loader, registry):
        """Test that loader initializes correctly."""
        assert loader.registry is registry
        assert len(loader._loaded_modules) == 0
    
    def test_load_from_file(self, loader, registry, sample_plugin_code):
        """Test loading plugins from a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_plugin_code)
            f.flush()
            temp_file = f.name
        
        try:
            plugin_classes = loader.load_from_file(temp_file, auto_register=True)
            
            assert len(plugin_classes) == 1
            assert plugin_classes[0].__name__ == "TestPlugin"
            
            # Check that plugin was registered
            plugin = registry.get_plugin("test_plugin")
            assert plugin is not None
            assert plugin.name == "test_plugin"
            assert plugin.version == "1.0.0"
            assert "test_action" in plugin.supported_actions
        finally:
            os.unlink(temp_file)
    
    def test_load_from_file_no_auto_register(self, loader, registry, sample_plugin_code):
        """Test loading plugins without auto-registration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_plugin_code)
            f.flush()
            temp_file = f.name
        
        try:
            plugin_classes = loader.load_from_file(temp_file, auto_register=False)
            
            assert len(plugin_classes) == 1
            
            # Check that plugin was NOT registered
            plugin = registry.get_plugin("test_plugin")
            assert plugin is None
        finally:
            os.unlink(temp_file)
    
    def test_load_from_nonexistent_file(self, loader):
        """Test that loading from nonexistent file raises error."""
        with pytest.raises(PluginLoadError, match="does not exist"):
            loader.load_from_file("/nonexistent/file.py")
    
    def test_load_from_directory(self, loader, registry, sample_plugin_code):
        """Test loading plugins from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a plugin file
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_file.write_text(sample_plugin_code)
            
            plugin_classes = loader.load_from_directory(temp_dir, auto_register=True)
            
            assert len(plugin_classes) == 1
            
            # Check that plugin was registered
            plugin = registry.get_plugin("test_plugin")
            assert plugin is not None
    
    def test_load_from_directory_recursive(self, loader, registry, sample_plugin_code):
        """Test loading plugins from directory recursively."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            
            # Create plugin in subdirectory
            plugin_file = subdir / "test_plugin.py"
            plugin_file.write_text(sample_plugin_code)
            
            # Non-recursive should find nothing
            plugin_classes = loader.load_from_directory(temp_dir, recursive=False)
            assert len(plugin_classes) == 0
            
            # Recursive should find the plugin
            plugin_classes = loader.load_from_directory(temp_dir, recursive=True)
            assert len(plugin_classes) == 1
    
    def test_load_from_directory_skips_private_files(self, loader, sample_plugin_code):
        """Test that loader skips files starting with underscore."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create private file
            private_file = Path(temp_dir) / "_private_plugin.py"
            private_file.write_text(sample_plugin_code)
            
            plugin_classes = loader.load_from_directory(temp_dir)
            assert len(plugin_classes) == 0
    
    def test_load_from_nonexistent_directory(self, loader):
        """Test that loading from nonexistent directory raises error."""
        with pytest.raises(PluginLoadError, match="does not exist"):
            loader.load_from_directory("/nonexistent/directory")
    
    def test_load_plugin_class_directly(self, loader, registry):
        """Test loading a plugin class directly."""
        class DirectPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "direct_plugin"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return ["direct_action"]
        
        plugin_class = loader.load_plugin_class(DirectPlugin, auto_register=True)
        
        assert plugin_class is DirectPlugin
        
        # Check that plugin was registered
        plugin = registry.get_plugin("direct_plugin")
        assert plugin is not None
    
    def test_is_plugin_class(self, loader):
        """Test plugin class validation."""
        class ValidPlugin(MemoryEntryPlugin):
            @property
            def name(self) -> str:
                return "valid"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def supported_actions(self) -> List[str]:
                return []
        
        class NotAPlugin:
            pass
        
        assert loader._is_plugin_class(ValidPlugin) is True
        assert loader._is_plugin_class(NotAPlugin) is False
        assert loader._is_plugin_class(MemoryEntryPlugin) is False
    
    def test_get_loaded_modules(self, loader, sample_plugin_code):
        """Test getting loaded modules."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_plugin_code)
            f.flush()
            temp_file = f.name
        
        try:
            loader.load_from_file(temp_file)
            
            modules = loader.get_loaded_modules()
            assert len(modules) > 0
        finally:
            os.unlink(temp_file)
    
    def test_convenience_function(self, sample_plugin_code):
        """Test convenience function for loading from directory."""
        registry = PluginRegistry()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_file = Path(temp_dir) / "test_plugin.py"
            plugin_file.write_text(sample_plugin_code)
            
            plugin_classes = load_plugins_from_directory(temp_dir, registry=registry)
            
            assert len(plugin_classes) == 1
            assert registry.get_plugin("test_plugin") is not None
