"""
Tests for Write Strategy Configuration

Tests configuration loading, validation, and defaults for WriteStrategyConfig
and SessionConfig.

Feature: memory-write-strategy-session-management
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

from luma.core.write_strategy import (
    WriteStrategyConfig,
    SessionConfig,
    load_write_strategy_config,
    load_session_config
)


# ============================================================================
# WriteStrategyConfig Tests
# ============================================================================


class TestWriteStrategyConfigDefaults:
    """Test default values for WriteStrategyConfig."""
    
    def test_default_trivial_patterns(self):
        """Test that default trivial patterns are set correctly."""
        config = WriteStrategyConfig()
        assert isinstance(config.trivial_patterns, list)
        assert len(config.trivial_patterns) > 0
        assert "hello" in config.trivial_patterns
        assert "hi" in config.trivial_patterns
        assert "thanks" in config.trivial_patterns
    
    def test_default_min_content_length(self):
        """Test that default min_content_length is 3."""
        config = WriteStrategyConfig()
        assert config.min_content_length == 3
    
    def test_default_repetition_window(self):
        """Test that default repetition_window is 5."""
        config = WriteStrategyConfig()
        assert config.repetition_window == 5
    
    def test_default_immediate_persist_patterns(self):
        """Test that default immediate_persist_patterns is empty list."""
        config = WriteStrategyConfig()
        assert isinstance(config.immediate_persist_patterns, list)
        assert len(config.immediate_persist_patterns) == 0
    
    def test_default_similarity_threshold(self):
        """Test that default similarity_threshold is 0.9."""
        config = WriteStrategyConfig()
        assert config.similarity_threshold == 0.9
    
    def test_default_enable_conflict_detection(self):
        """Test that default enable_conflict_detection is True."""
        config = WriteStrategyConfig()
        assert config.enable_conflict_detection is True


class TestWriteStrategyConfigValidation:
    """Test validation for WriteStrategyConfig."""
    
    def test_negative_min_content_length_raises_error(self):
        """Test that negative min_content_length raises ValueError."""
        with pytest.raises(ValueError, match="min_content_length must be a non-negative integer"):
            WriteStrategyConfig(min_content_length=-1)
    
    def test_non_integer_min_content_length_raises_error(self):
        """Test that non-integer min_content_length raises ValueError."""
        with pytest.raises(ValueError, match="min_content_length must be a non-negative integer"):
            WriteStrategyConfig(min_content_length="5")
    
    def test_negative_repetition_window_raises_error(self):
        """Test that negative repetition_window raises ValueError."""
        with pytest.raises(ValueError, match="repetition_window must be a non-negative integer"):
            WriteStrategyConfig(repetition_window=-1)
    
    def test_non_integer_repetition_window_raises_error(self):
        """Test that non-integer repetition_window raises ValueError."""
        with pytest.raises(ValueError, match="repetition_window must be a non-negative integer"):
            WriteStrategyConfig(repetition_window=5.5)
    
    def test_similarity_threshold_below_zero_raises_error(self):
        """Test that similarity_threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            WriteStrategyConfig(similarity_threshold=-0.1)
    
    def test_similarity_threshold_above_one_raises_error(self):
        """Test that similarity_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
            WriteStrategyConfig(similarity_threshold=1.5)
    
    def test_non_numeric_similarity_threshold_raises_error(self):
        """Test that non-numeric similarity_threshold raises ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be a number"):
            WriteStrategyConfig(similarity_threshold="0.9")
    
    def test_non_list_trivial_patterns_raises_error(self):
        """Test that non-list trivial_patterns raises ValueError."""
        with pytest.raises(ValueError, match="trivial_patterns must be a list"):
            WriteStrategyConfig(trivial_patterns="hello,hi")
    
    def test_non_string_trivial_pattern_raises_error(self):
        """Test that non-string elements in trivial_patterns raise ValueError."""
        with pytest.raises(ValueError, match="all trivial_patterns must be strings"):
            WriteStrategyConfig(trivial_patterns=["hello", 123, "hi"])
    
    def test_non_list_immediate_persist_patterns_raises_error(self):
        """Test that non-list immediate_persist_patterns raises ValueError."""
        with pytest.raises(ValueError, match="immediate_persist_patterns must be a list"):
            WriteStrategyConfig(immediate_persist_patterns="pattern1,pattern2")
    
    def test_non_string_immediate_persist_pattern_raises_error(self):
        """Test that non-string elements in immediate_persist_patterns raise ValueError."""
        with pytest.raises(ValueError, match="all immediate_persist_patterns must be strings"):
            WriteStrategyConfig(immediate_persist_patterns=["pattern1", None])
    
    def test_non_boolean_enable_conflict_detection_raises_error(self):
        """Test that non-boolean enable_conflict_detection raises ValueError."""
        with pytest.raises(ValueError, match="enable_conflict_detection must be a boolean"):
            WriteStrategyConfig(enable_conflict_detection="true")
    
    def test_valid_custom_config(self):
        """Test that valid custom configuration is accepted."""
        config = WriteStrategyConfig(
            trivial_patterns=["custom1", "custom2"],
            min_content_length=10,
            repetition_window=3,
            immediate_persist_patterns=["important"],
            similarity_threshold=0.85,
            enable_conflict_detection=False
        )
        assert config.trivial_patterns == ["custom1", "custom2"]
        assert config.min_content_length == 10
        assert config.repetition_window == 3
        assert config.immediate_persist_patterns == ["important"]
        assert config.similarity_threshold == 0.85
        assert config.enable_conflict_detection is False
    
    def test_zero_values_are_valid(self):
        """Test that zero values are valid for numeric fields."""
        config = WriteStrategyConfig(
            min_content_length=0,
            repetition_window=0,
            similarity_threshold=0.0
        )
        assert config.min_content_length == 0
        assert config.repetition_window == 0
        assert config.similarity_threshold == 0.0


# ============================================================================
# SessionConfig Tests
# ============================================================================


class TestSessionConfigDefaults:
    """Test default values for SessionConfig."""
    
    def test_default_timeout_seconds(self):
        """Test that default timeout_seconds is 1800 (30 minutes)."""
        config = SessionConfig()
        assert config.timeout_seconds == 1800
    
    def test_default_cleanup_interval_seconds(self):
        """Test that default cleanup_interval_seconds is 300 (5 minutes)."""
        config = SessionConfig()
        assert config.cleanup_interval_seconds == 300
    
    def test_default_max_buffer_size(self):
        """Test that default max_buffer_size is 100."""
        config = SessionConfig()
        assert config.max_buffer_size == 100
    
    def test_default_enable_buffering(self):
        """Test that default enable_buffering is True."""
        config = SessionConfig()
        assert config.enable_buffering is True


class TestSessionConfigValidation:
    """Test validation for SessionConfig."""
    
    def test_zero_timeout_seconds_raises_error(self):
        """Test that zero timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be a positive integer"):
            SessionConfig(timeout_seconds=0)
    
    def test_negative_timeout_seconds_raises_error(self):
        """Test that negative timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be a positive integer"):
            SessionConfig(timeout_seconds=-100)
    
    def test_non_integer_timeout_seconds_raises_error(self):
        """Test that non-integer timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be a positive integer"):
            SessionConfig(timeout_seconds=30.5)
    
    def test_zero_cleanup_interval_seconds_raises_error(self):
        """Test that zero cleanup_interval_seconds raises ValueError."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be a positive integer"):
            SessionConfig(cleanup_interval_seconds=0)
    
    def test_negative_cleanup_interval_seconds_raises_error(self):
        """Test that negative cleanup_interval_seconds raises ValueError."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be a positive integer"):
            SessionConfig(cleanup_interval_seconds=-50)
    
    def test_non_integer_cleanup_interval_seconds_raises_error(self):
        """Test that non-integer cleanup_interval_seconds raises ValueError."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be a positive integer"):
            SessionConfig(cleanup_interval_seconds="300")
    
    def test_zero_max_buffer_size_raises_error(self):
        """Test that zero max_buffer_size raises ValueError."""
        with pytest.raises(ValueError, match="max_buffer_size must be a positive integer"):
            SessionConfig(max_buffer_size=0)
    
    def test_negative_max_buffer_size_raises_error(self):
        """Test that negative max_buffer_size raises ValueError."""
        with pytest.raises(ValueError, match="max_buffer_size must be a positive integer"):
            SessionConfig(max_buffer_size=-10)
    
    def test_non_integer_max_buffer_size_raises_error(self):
        """Test that non-integer max_buffer_size raises ValueError."""
        with pytest.raises(ValueError, match="max_buffer_size must be a positive integer"):
            SessionConfig(max_buffer_size=100.5)
    
    def test_non_boolean_enable_buffering_raises_error(self):
        """Test that non-boolean enable_buffering raises ValueError."""
        with pytest.raises(ValueError, match="enable_buffering must be a boolean"):
            SessionConfig(enable_buffering="true")
    
    def test_valid_custom_config(self):
        """Test that valid custom configuration is accepted."""
        config = SessionConfig(
            timeout_seconds=3600,
            cleanup_interval_seconds=600,
            max_buffer_size=50,
            enable_buffering=False
        )
        assert config.timeout_seconds == 3600
        assert config.cleanup_interval_seconds == 600
        assert config.max_buffer_size == 50
        assert config.enable_buffering is False


# ============================================================================
# Configuration Loading Tests
# ============================================================================


class TestLoadWriteStrategyConfig:
    """Test loading WriteStrategyConfig from various sources."""
    
    def test_load_with_defaults(self):
        """Test loading with no config file or environment variables."""
        config = load_write_strategy_config()
        assert config.min_content_length == 3
        assert config.repetition_window == 5
        assert config.similarity_threshold == 0.9
        assert config.enable_conflict_detection is True
    
    def test_load_from_environment_variables(self):
        """Test loading from environment variables."""
        # Set environment variables
        os.environ["LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH"] = "10"
        os.environ["LUMA_WRITE_STRATEGY_REPETITION_WINDOW"] = "7"
        os.environ["LUMA_WRITE_STRATEGY_SIMILARITY_THRESHOLD"] = "0.85"
        os.environ["LUMA_WRITE_STRATEGY_ENABLE_CONFLICT_DETECTION"] = "false"
        os.environ["LUMA_WRITE_STRATEGY_TRIVIAL_PATTERNS"] = "test1,test2,test3"
        os.environ["LUMA_WRITE_STRATEGY_IMMEDIATE_PERSIST_PATTERNS"] = "urgent,critical"
        
        try:
            config = load_write_strategy_config()
            assert config.min_content_length == 10
            assert config.repetition_window == 7
            assert config.similarity_threshold == 0.85
            assert config.enable_conflict_detection is False
            assert config.trivial_patterns == ["test1", "test2", "test3"]
            assert config.immediate_persist_patterns == ["urgent", "critical"]
        finally:
            # Clean up environment variables
            for key in [
                "LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH",
                "LUMA_WRITE_STRATEGY_REPETITION_WINDOW",
                "LUMA_WRITE_STRATEGY_SIMILARITY_THRESHOLD",
                "LUMA_WRITE_STRATEGY_ENABLE_CONFLICT_DETECTION",
                "LUMA_WRITE_STRATEGY_TRIVIAL_PATTERNS",
                "LUMA_WRITE_STRATEGY_IMMEDIATE_PERSIST_PATTERNS"
            ]:
                os.environ.pop(key, None)
    
    def test_load_from_config_file(self):
        """Test loading from JSON config file."""
        config_data = {
            "min_content_length": 15,
            "repetition_window": 10,
            "similarity_threshold": 0.95,
            "enable_conflict_detection": False,
            "trivial_patterns": ["file1", "file2"],
            "immediate_persist_patterns": ["file_urgent"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_write_strategy_config(config_file=temp_file)
            assert config.min_content_length == 15
            assert config.repetition_window == 10
            assert config.similarity_threshold == 0.95
            assert config.enable_conflict_detection is False
            assert config.trivial_patterns == ["file1", "file2"]
            assert config.immediate_persist_patterns == ["file_urgent"]
        finally:
            # Ensure file is closed before deletion (Windows compatibility)
            try:
                Path(temp_file).unlink()
            except PermissionError:
                # On Windows, ensure file handle is released
                import gc
                gc.collect()
                Path(temp_file).unlink()
    
    def test_environment_overrides_config_file(self):
        """Test that environment variables override config file values."""
        config_data = {
            "min_content_length": 15,
            "repetition_window": 10
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        os.environ["LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH"] = "20"
        
        try:
            config = load_write_strategy_config(config_file=temp_file)
            assert config.min_content_length == 20  # From environment
            assert config.repetition_window == 10  # From file
        finally:
            # Ensure file is closed before deletion (Windows compatibility)
            try:
                Path(temp_file).unlink()
            except PermissionError:
                # On Windows, ensure file handle is released
                import gc
                gc.collect()
                Path(temp_file).unlink()
            os.environ.pop("LUMA_WRITE_STRATEGY_MIN_CONTENT_LENGTH", None)
    
    def test_nonexistent_config_file_raises_error(self):
        """Test that nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_write_strategy_config(config_file="/nonexistent/path/config.json")
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON in config file raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_write_strategy_config(config_file=temp_file)
        finally:
            # Ensure file is closed before deletion (Windows compatibility)
            try:
                Path(temp_file).unlink()
            except PermissionError:
                # On Windows, ensure file handle is released
                import gc
                gc.collect()
                Path(temp_file).unlink()
    
    def test_boolean_environment_variable_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("anything_else", False)
        ]
        
        for env_value, expected in test_cases:
            os.environ["LUMA_WRITE_STRATEGY_ENABLE_CONFLICT_DETECTION"] = env_value
            try:
                config = load_write_strategy_config()
                assert config.enable_conflict_detection == expected, \
                    f"Expected {expected} for env value '{env_value}', got {config.enable_conflict_detection}"
            finally:
                os.environ.pop("LUMA_WRITE_STRATEGY_ENABLE_CONFLICT_DETECTION", None)


class TestLoadSessionConfig:
    """Test loading SessionConfig from various sources."""
    
    def test_load_with_defaults(self):
        """Test loading with no config file or environment variables."""
        config = load_session_config()
        assert config.timeout_seconds == 1800
        assert config.cleanup_interval_seconds == 300
        assert config.max_buffer_size == 100
        assert config.enable_buffering is True
    
    def test_load_from_environment_variables(self):
        """Test loading from environment variables."""
        os.environ["LUMA_SESSION_TIMEOUT_SECONDS"] = "3600"
        os.environ["LUMA_SESSION_CLEANUP_INTERVAL_SECONDS"] = "600"
        os.environ["LUMA_SESSION_MAX_BUFFER_SIZE"] = "50"
        os.environ["LUMA_SESSION_ENABLE_BUFFERING"] = "false"
        
        try:
            config = load_session_config()
            assert config.timeout_seconds == 3600
            assert config.cleanup_interval_seconds == 600
            assert config.max_buffer_size == 50
            assert config.enable_buffering is False
        finally:
            for key in [
                "LUMA_SESSION_TIMEOUT_SECONDS",
                "LUMA_SESSION_CLEANUP_INTERVAL_SECONDS",
                "LUMA_SESSION_MAX_BUFFER_SIZE",
                "LUMA_SESSION_ENABLE_BUFFERING"
            ]:
                os.environ.pop(key, None)
    
    def test_load_from_config_file(self):
        """Test loading from JSON config file."""
        config_data = {
            "timeout_seconds": 7200,
            "cleanup_interval_seconds": 900,
            "max_buffer_size": 200,
            "enable_buffering": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_session_config(config_file=temp_file)
            assert config.timeout_seconds == 7200
            assert config.cleanup_interval_seconds == 900
            assert config.max_buffer_size == 200
            assert config.enable_buffering is False
        finally:
            # Ensure file is closed before deletion (Windows compatibility)
            try:
                Path(temp_file).unlink()
            except PermissionError:
                # On Windows, ensure file handle is released
                import gc
                gc.collect()
                Path(temp_file).unlink()
    
    def test_environment_overrides_config_file(self):
        """Test that environment variables override config file values."""
        config_data = {
            "timeout_seconds": 7200,
            "max_buffer_size": 200
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        os.environ["LUMA_SESSION_TIMEOUT_SECONDS"] = "3600"
        
        try:
            config = load_session_config(config_file=temp_file)
            assert config.timeout_seconds == 3600  # From environment
            assert config.max_buffer_size == 200  # From file
        finally:
            # Ensure file is closed before deletion (Windows compatibility)
            try:
                Path(temp_file).unlink()
            except PermissionError:
                # On Windows, ensure file handle is released
                import gc
                gc.collect()
                Path(temp_file).unlink()
            os.environ.pop("LUMA_SESSION_TIMEOUT_SECONDS", None)
    
    def test_nonexistent_config_file_raises_error(self):
        """Test that nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_session_config(config_file="/nonexistent/path/config.json")
    
    def test_boolean_environment_variable_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False)
        ]
        
        for env_value, expected in test_cases:
            os.environ["LUMA_SESSION_ENABLE_BUFFERING"] = env_value
            try:
                config = load_session_config()
                assert config.enable_buffering == expected
            finally:
                os.environ.pop("LUMA_SESSION_ENABLE_BUFFERING", None)
