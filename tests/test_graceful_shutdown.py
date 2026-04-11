"""Tests for graceful shutdown handling in the API server."""

import pytest
import signal
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from luma_memory.api.server import handle_shutdown_signal, lifespan
from luma_memory.api import server as server_module
from fastapi import FastAPI


def test_handle_shutdown_signal_logs_correctly():
    """Test that shutdown signal handler logs the signal correctly."""
    # Reset shutdown flag
    server_module._shutdown_requested = False
    
    with patch('luma_memory.api.server.logger') as mock_logger:
        # Simulate SIGTERM signal
        handle_shutdown_signal(signal.SIGTERM, None)
        
        # Verify logging
        mock_logger.info.assert_called_once()
        assert "SIGTERM" in mock_logger.info.call_args[0][0]
        assert "graceful shutdown" in mock_logger.info.call_args[0][0].lower()
        
        # Verify shutdown flag is set
        assert server_module._shutdown_requested is True


def test_handle_shutdown_signal_ignores_duplicate_signals():
    """Test that duplicate shutdown signals are ignored."""
    # Set shutdown flag to True (already shutting down)
    server_module._shutdown_requested = True
    
    with patch('luma_memory.api.server.logger') as mock_logger:
        # Simulate another SIGTERM signal
        handle_shutdown_signal(signal.SIGTERM, None)
        
        # Verify warning is logged
        mock_logger.warning.assert_called_once()
        assert "already in progress" in mock_logger.warning.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_lifespan_cleanup_closes_storage():
    """Test that lifespan context manager properly closes storage on shutdown."""
    app = FastAPI()
    
    # Create mock storage with close method
    mock_storage = Mock()
    mock_storage.close = Mock()
    mock_storage.cache = Mock()
    mock_storage.cache.clear = Mock()
    
    # Create mock memory manager
    mock_manager = Mock()
    mock_manager.storage = mock_storage
    mock_manager.get_stats = Mock(return_value={'total_entries': 0})
    
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.initialize_memory_manager') as mock_init:
            with patch('luma_memory.api.server.set_memory_manager') as mock_set:
                # Setup mocks
                mock_config = Mock()
                mock_config.model_dump = Mock(return_value={})
                mock_config.api_host = "0.0.0.0"
                mock_config.api_port = 8000
                mock_config_class.load_config = Mock(return_value=mock_config)
                mock_init.return_value = mock_manager
                
                # Use lifespan context manager
                async with lifespan(app):
                    # Verify startup
                    assert mock_init.called
                    assert mock_set.called
                
                # After exiting context, verify cleanup was called
                mock_storage.close.assert_called_once()
                mock_storage.cache.clear.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_handles_cleanup_errors_gracefully():
    """Test that lifespan handles cleanup errors without raising exceptions."""
    app = FastAPI()
    
    # Create mock storage that raises error on close
    mock_storage = Mock()
    mock_storage.close = Mock(side_effect=Exception("Connection error"))
    mock_storage.cache = Mock()
    
    # Create mock memory manager
    mock_manager = Mock()
    mock_manager.storage = mock_storage
    mock_manager.get_stats = Mock(return_value={'total_entries': 0})
    
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
                    
                    # Use lifespan context manager - should not raise exception
                    async with lifespan(app):
                        pass
                    
                    # Verify error was logged but not raised
                    assert any('Error during shutdown' in str(call) for call in mock_logger.error.call_args_list)


def test_signal_handlers_registered_in_run_server():
    """Test that signal handlers are registered when run_server is called."""
    with patch('luma_memory.api.server.MemoryModuleConfig') as mock_config_class:
        with patch('luma_memory.api.server.setup_logging'):
            with patch('luma_memory.api.server.signal.signal') as mock_signal:
                with patch('uvicorn.run'):  # Patch uvicorn directly, not through server module
                    # Setup mock config
                    mock_config = Mock()
                    mock_config.api_host = "0.0.0.0"
                    mock_config.api_port = 8000
                    mock_config.api_workers = 1
                    mock_config.log_level = "INFO"
                    mock_config_class.load_config = Mock(return_value=mock_config)
                    
                    # Import and call run_server
                    from luma_memory.api.server import run_server
                    
                    # This will register signal handlers but not actually start the server
                    # because we've mocked uvicorn.run
                    run_server()
                    
                    # Verify signal handlers were registered
                    assert mock_signal.call_count >= 2
                    
                    # Check that SIGTERM and SIGINT were registered
                    signal_calls = [call[0] for call in mock_signal.call_args_list]
                    assert any(signal.SIGTERM in call for call in signal_calls)
                    assert any(signal.SIGINT in call for call in signal_calls)
