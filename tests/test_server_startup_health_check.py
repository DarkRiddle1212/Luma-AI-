"""Tests for server startup health check."""

import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from luma_memory.api.server import lifespan


@pytest.mark.asyncio
async def test_startup_health_check_success():
    """Test that health check is performed successfully on startup."""
    app = FastAPI()
    
    # Create mock memory manager with get_stats method
    mock_manager = Mock()
    mock_manager.get_stats = Mock(return_value={
        'total_entries': 42,
        'storage_size_mb': 5.2,
        'cache_hit_rate': 0.85
    })
    mock_manager.storage = Mock()
    mock_manager.storage.close = Mock()
    mock_manager.storage.cache = Mock()
    mock_manager.storage.cache.clear = Mock()
    
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.initialize_memory_manager') as mock_init:
            with patch('luma_memory.api.server.set_memory_manager') as mock_set:
                with patch('luma_memory.api.server.logger') as mock_logger:
                    # Setup mocks
                    mock_config = Mock()
                    mock_config.model_dump = Mock(return_value={})
                    mock_config.api_host = "0.0.0.0"
                    mock_config.api_port = 8000
                    mock_config_class.load_config = Mock(return_value=mock_config)
                    mock_init.return_value = mock_manager
                    
                    # Use lifespan context manager
                    async with lifespan(app):
                        # Verify health check was performed
                        mock_manager.get_stats.assert_called_once()
                        
                        # Verify health check result was logged
                        health_check_logged = any(
                            'health check passed' in str(call).lower() and '42' in str(call)
                            for call in mock_logger.info.call_args_list
                        )
                        assert health_check_logged, "Health check result should be logged"


@pytest.mark.asyncio
async def test_startup_health_check_with_empty_database():
    """Test that health check works correctly with empty database."""
    app = FastAPI()
    
    # Create mock memory manager with empty stats
    mock_manager = Mock()
    mock_manager.get_stats = Mock(return_value={
        'total_entries': 0,
        'storage_size_mb': 0.0
    })
    mock_manager.storage = Mock()
    mock_manager.storage.close = Mock()
    mock_manager.storage.cache = Mock()
    mock_manager.storage.cache.clear = Mock()
    
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.initialize_memory_manager') as mock_init:
            with patch('luma_memory.api.server.set_memory_manager'):
                with patch('luma_memory.api.server.logger') as mock_logger:
                    # Setup mocks
                    mock_config = Mock()
                    mock_config.model_dump = Mock(return_value={})
                    mock_config.api_host = "0.0.0.0"
                    mock_config.api_port = 8000
                    mock_config_class.load_config = Mock(return_value=mock_config)
                    mock_init.return_value = mock_manager
                    
                    # Use lifespan context manager
                    async with lifespan(app):
                        # Verify health check was performed
                        mock_manager.get_stats.assert_called_once()
                        
                        # Verify health check with 0 entries was logged
                        health_check_logged = any(
                            'health check passed' in str(call).lower() and 
                            'total entries: 0' in str(call).lower()
                            for call in mock_logger.info.call_args_list
                        )
                        assert health_check_logged, "Health check with 0 entries should be logged"


@pytest.mark.asyncio
async def test_startup_fails_if_health_check_raises_exception():
    """Test that startup fails gracefully if health check raises an exception."""
    app = FastAPI()
    
    # Create mock memory manager that raises exception on get_stats
    mock_manager = Mock()
    mock_manager.get_stats = Mock(side_effect=Exception("Database connection failed"))
    
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.initialize_memory_manager') as mock_init:
            with patch('luma_memory.api.server.set_memory_manager'):
                with patch('luma_memory.api.server.logger') as mock_logger:
                    # Setup mocks
                    mock_config = Mock()
                    mock_config.model_dump = Mock(return_value={})
                    mock_config.api_host = "0.0.0.0"
                    mock_config.api_port = 8000
                    mock_config_class.load_config = Mock(return_value=mock_config)
                    mock_init.return_value = mock_manager
                    
                    # Lifespan should raise exception on startup failure
                    with pytest.raises(Exception, match="Database connection failed"):
                        async with lifespan(app):
                            pass
                    
                    # Verify error was logged
                    assert any('Failed to start server' in str(call) 
                              for call in mock_logger.error.call_args_list)


@pytest.mark.asyncio
async def test_startup_logs_configuration():
    """Test that configuration is logged on startup."""
    app = FastAPI()
    
    # Create mock memory manager
    mock_manager = Mock()
    mock_manager.get_stats = Mock(return_value={'total_entries': 0})
    mock_manager.storage = Mock()
    mock_manager.storage.close = Mock()
    mock_manager.storage.cache = Mock()
    mock_manager.storage.cache.clear = Mock()
    
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.initialize_memory_manager') as mock_init:
            with patch('luma_memory.api.server.set_memory_manager'):
                with patch('luma_memory.api.server.logger') as mock_logger:
                    # Setup mocks with specific config
                    mock_config = Mock()
                    mock_config.model_dump = Mock(return_value={
                        'db_path': './data/test.db',
                        'api_host': '127.0.0.1',
                        'api_port': 9000
                    })
                    mock_config.api_host = "127.0.0.1"
                    mock_config.api_port = 9000
                    mock_config_class.load_config = Mock(return_value=mock_config)
                    mock_init.return_value = mock_manager
                    
                    # Use lifespan context manager
                    async with lifespan(app):
                        # Verify configuration was logged
                        config_logged = any(
                            'configuration loaded' in str(call).lower()
                            for call in mock_logger.info.call_args_list
                        )
                        assert config_logged, "Configuration should be logged on startup"
                        
                        # Verify server started message includes host and port
                        server_started_logged = any(
                            'server started successfully' in str(call).lower() and
                            '127.0.0.1' in str(call) and '9000' in str(call)
                            for call in mock_logger.info.call_args_list
                        )
                        assert server_started_logged, "Server start message should include host and port"
