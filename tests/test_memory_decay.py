"""
Unit tests for MemoryDecay component.

Tests the MemoryDecay component's decay functions, age calculation,
and integration with memory interface.
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

from luma.core.lifecycle.memory_decay import MemoryDecay
from luma.core.lifecycle.schemas import (
    DecayConfig,
    DecayFunctionType,
    MemoryDecayResult,
)


class TestMemoryDecayInitialization:
    """Test MemoryDecay initialization."""
    
    def test_initialization_with_all_dependencies(self):
        """Test MemoryDecay initialization with all dependencies."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        mock_logger = MagicMock()
        
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config,
            metrics_collector=mock_metrics_collector,
            logger=mock_logger
        )
        
        assert decay.memory_interface == mock_memory_interface
        assert decay.decay_config == decay_config
        assert decay.metrics_collector == mock_metrics_collector
        assert decay.logger == mock_logger
    
    def test_initialization_without_optional_dependencies(self):
        """Test MemoryDecay initialization without optional dependencies."""
        mock_memory_interface = MagicMock()
        
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        assert decay.memory_interface == mock_memory_interface
        assert decay.decay_config == decay_config
        assert decay.metrics_collector is None
        assert decay.logger is None


class TestAgeCalculation:
    """Test age calculation from timestamps."""
    
    def test_calculate_age_days_past_timestamp(self):
        """Test age calculation for past timestamp."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Create a timestamp from 10 days ago
        past_timestamp = datetime.now(UTC) - timedelta(days=10)
        timestamp_str = past_timestamp.isoformat().replace('+00:00', 'Z')
        
        age_days = decay._calculate_age_days(timestamp_str)
        
        # Allow small floating point tolerance
        assert abs(age_days - 10.0) < 0.001
    
    def test_calculate_age_days_future_timestamp(self):
        """Test age calculation for future timestamp (should be 0)."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Create a timestamp 5 days in the future
        future_timestamp = datetime.now(UTC) + timedelta(days=5)
        timestamp_str = future_timestamp.isoformat().replace('+00:00', 'Z')
        
        age_days = decay._calculate_age_days(timestamp_str)
        
        assert age_days == 0.0
    
    def test_calculate_age_days_timezone_aware(self):
        """Test age calculation with timezone-aware timestamp."""
        from datetime import timezone
        
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Create a timezone-aware timestamp (EST, UTC-5)
        est_offset = timezone(timedelta(hours=-5))
        past_timestamp = datetime.now(UTC) - timedelta(days=5)
        # Convert to EST
        est_timestamp = past_timestamp.astimezone(est_offset)
        timestamp_str = est_timestamp.isoformat()
        
        age_days = decay._calculate_age_days(timestamp_str)
        
        assert abs(age_days - 5.0) < 0.001
    
    def test_calculate_age_days_timezone_naive(self):
        """Test age calculation with timezone-naive timestamp."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Create a timezone-naive timestamp
        past_timestamp = datetime.now() - timedelta(days=3)
        timestamp_str = past_timestamp.isoformat()
        
        age_days = decay._calculate_age_days(timestamp_str)
        
        # Allow small floating point tolerance
        assert abs(age_days - 3.0) < 0.1
    
    def test_calculate_age_days_fractional_days(self):
        """Test age calculation with fractional days."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Create a timestamp 2.5 days ago
        past_timestamp = datetime.now(UTC) - timedelta(days=2, hours=12)
        timestamp_str = past_timestamp.isoformat().replace('+00:00', 'Z')
        
        age_days = decay._calculate_age_days(timestamp_str)
        
        assert abs(age_days - 2.5) < 0.001
    
    def test_calculate_age_days_invalid_timestamp(self):
        """Test age calculation with invalid timestamp."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        with pytest.raises(ValueError, match="Failed to parse timestamp"):
            decay._calculate_age_days("invalid-timestamp")


class TestExponentialDecay:
    """Test exponential decay function."""
    
    def test_exponential_decay_formula(self):
        """Test exponential decay formula: importance * e^(-decay_rate * age)."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Test with importance=1.0, age=10 days, decay_rate=0.1
        # Expected: 1.0 * e^(-0.1 * 10) = 1.0 * e^(-1) ≈ 0.3679
        importance = 1.0
        age_days = 10.0
        expected = importance * 2.718281828459045 ** (-0.1 * 10)
        
        result = decay._apply_exponential_decay(importance, age_days)
        
        assert abs(result - expected) < 0.001
    
    def test_exponential_decay_preserves_zero(self):
        """Test that zero importance stays zero."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        result = decay._apply_exponential_decay(0.0, 10.0)
        
        assert result == 0.0
    
    def test_exponential_decay_clamps_to_one(self):
        """Test that decay result is clamped to [0, 1]."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.01  # Small positive rate
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # With very small age, importance should stay close to original
        result = decay._apply_exponential_decay(0.5, 0.0)
        
        # Should be close to 0.5
        assert abs(result - 0.5) < 0.01


class TestLinearDecay:
    """Test linear decay function."""
    
    def test_linear_decay_formula(self):
        """Test linear decay formula: max(0, importance - decay_rate * age)."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.LINEAR,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Test with importance=1.0, age=5 days, decay_rate=0.1
        # Expected: max(0, 1.0 - 0.1 * 5) = max(0, 0.5) = 0.5
        importance = 1.0
        age_days = 5.0
        expected = max(0.0, importance - 0.1 * age_days)
        
        result = decay._apply_linear_decay(importance, age_days)
        
        assert result == expected
    
    def test_linear_decay_reaches_zero(self):
        """Test linear decay reaching zero."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.LINEAR,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Test with importance=0.5, age=10 days, decay_rate=0.1
        # Expected: max(0, 0.5 - 0.1 * 10) = max(0, -0.5) = 0.0
        result = decay._apply_linear_decay(0.5, 10.0)
        
        assert result == 0.0
    
    def test_linear_decay_preserves_zero(self):
        """Test that zero importance stays zero."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.LINEAR,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        result = decay._apply_linear_decay(0.0, 10.0)
        
        assert result == 0.0


class TestStepDecay:
    """Test step decay function."""
    
    def test_step_decay_formula(self):
        """Test step decay formula: importance * (1 - step_percentage)^(age / interval)."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.STEP,
            decay_rate=0.05,
            step_interval_days=7,
            step_percentage=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Test with importance=1.0, age=14 days, interval=7, percentage=0.1
        # Expected: 1.0 * (1 - 0.1)^(14 / 7) = 1.0 * 0.9^2 = 0.81
        importance = 1.0
        age_days = 14.0
        expected = importance * (1 - 0.1) ** (age_days / 7)
        
        result = decay._apply_step_decay(importance, age_days)
        
        assert abs(result - expected) < 0.001
    
    def test_step_decay_preserves_zero(self):
        """Test that zero importance stays zero."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.STEP,
            decay_rate=0.05,
            step_interval_days=7,
            step_percentage=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        result = decay._apply_step_decay(0.0, 10.0)
        
        assert result == 0.0


class TestCalculateDecayFactor:
    """Test decay factor calculation."""
    
    def test_calculate_decay_factor_exponential(self):
        """Test decay factor calculation for exponential decay."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # For age=10 days, decay_rate=0.1
        # Expected: e^(-0.1 * 10) = e^(-1) ≈ 0.3679
        factor = decay.calculate_decay_factor(10.0)
        expected = 2.718281828459045 ** (-0.1 * 10)
        
        assert abs(factor - expected) < 0.001


class TestApplyDecay:
    """Test apply_decay method."""
    
    def test_apply_decay_exponential(self):
        """Test apply_decay with exponential decay."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Mock memory retrieval
        past_timestamp = datetime.now(UTC) - timedelta(days=10)
        mock_memory = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {
                "importance": 1.0,
                "creation_timestamp": past_timestamp.isoformat().replace('+00:00', 'Z')
            },
            "timestamp": past_timestamp.isoformat().replace('+00:00', 'Z'),
            "category": "test",
            "tags": []
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = decay.apply_decay()
        
        assert isinstance(result, MemoryDecayResult)
        assert result.memories_processed == 1
        assert result.memories_updated == 1
    
    def test_apply_decay_dry_run(self):
        """Test apply_decay in dry_run mode."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Mock memory retrieval
        past_timestamp = datetime.now(UTC) - timedelta(days=10)
        mock_memory = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {
                "importance": 1.0,
                "creation_timestamp": past_timestamp.isoformat().replace('+00:00', 'Z')
            },
            "timestamp": past_timestamp.isoformat().replace('+00:00', 'Z'),
            "category": "test",
            "tags": []
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = decay.apply_decay(dry_run=True)
        
        assert isinstance(result, MemoryDecayResult)
        assert result.memories_processed == 1
        # In dry_run, store should not be called
        mock_memory_interface.store.assert_not_called()
    
    def test_apply_decay_skips_zero_importance(self):
        """Test that memories with importance=0 are skipped (not processed)."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Mock memory with importance=0
        past_timestamp = datetime.now(UTC) - timedelta(days=10)
        mock_memory = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {
                "importance": 0.0,
                "creation_timestamp": past_timestamp.isoformat().replace('+00:00', 'Z')
            },
            "timestamp": past_timestamp.isoformat().replace('+00:00', 'Z'),
            "category": "test",
            "tags": []
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = decay.apply_decay()
        
        # Zero importance memories are skipped (not processed)
        assert result.memories_processed == 0
        assert result.memories_updated == 0
    
    def test_apply_decay_logs_timestamp_errors(self):
        """Test that timestamp parsing errors are logged and skipped."""
        mock_memory_interface = MagicMock()
        mock_logger = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config,
            logger=mock_logger
        )
        
        # Mock memory with invalid timestamp
        mock_memory = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {
                "importance": 1.0,
                "creation_timestamp": "invalid-timestamp"
            },
            "timestamp": "invalid-timestamp",
            "category": "test",
            "tags": []
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = decay.apply_decay()
        
        # Logger should be called with error
        mock_logger.log.assert_called()
        # Invalid timestamp memory is skipped (not processed)
        assert result.memories_processed == 0
        assert result.memories_updated == 0
    
    def test_apply_decay_records_metrics(self):
        """Test that metrics are recorded."""
        mock_memory_interface = MagicMock()
        mock_metrics_collector = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config,
            metrics_collector=mock_metrics_collector
        )
        
        # Mock memory retrieval
        past_timestamp = datetime.now(UTC) - timedelta(days=10)
        mock_memory = {
            "id": "mem_123",
            "content": "Test content",
            "metadata": {
                "importance": 1.0,
                "creation_timestamp": past_timestamp.isoformat().replace('+00:00', 'Z')
            },
            "timestamp": past_timestamp.isoformat().replace('+00:00', 'Z'),
            "category": "test",
            "tags": []
        }
        
        mock_memory_interface.retrieve.return_value = {
            "memories": [mock_memory],
            "total_count": 1,
            "query_metadata": {}
        }
        
        result = decay.apply_decay()
        
        # Metrics should be recorded
        mock_metrics_collector.increment.assert_any_call("memory_decay.processed", 1)
        mock_metrics_collector.increment.assert_any_call("memory_decay.updated", 1)
        mock_metrics_collector.record_duration.assert_called()


class TestIntegration:
    """Integration tests for MemoryDecay."""
    
    def test_apply_decay_with_multiple_memories(self):
        """Test apply_decay with multiple memories."""
        mock_memory_interface = MagicMock()
        decay_config = DecayConfig(
            decay_function_type=DecayFunctionType.EXPONENTIAL,
            decay_rate=0.1
        )
        
        decay = MemoryDecay(
            memory_interface=mock_memory_interface,
            decay_config=decay_config
        )
        
        # Mock multiple memories
        past_timestamp_1 = datetime.now(UTC) - timedelta(days=10)
        past_timestamp_2 = datetime.now(UTC) - timedelta(days=5)
        past_timestamp_3 = datetime.now(UTC) - timedelta(days=2)
        
        mock_memories = [
            {
                "id": "mem_1",
                "content": "Content 1",
                "metadata": {
                    "importance": 1.0,
                    "creation_timestamp": past_timestamp_1.isoformat().replace('+00:00', 'Z')
                },
                "timestamp": past_timestamp_1.isoformat().replace('+00:00', 'Z'),
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_2",
                "content": "Content 2",
                "metadata": {
                    "importance": 0.8,
                    "creation_timestamp": past_timestamp_2.isoformat().replace('+00:00', 'Z')
                },
                "timestamp": past_timestamp_2.isoformat().replace('+00:00', 'Z'),
                "category": "test",
                "tags": []
            },
            {
                "id": "mem_3",
                "content": "Content 3",
                "metadata": {
                    "importance": 0.5,
                    "creation_timestamp": past_timestamp_3.isoformat().replace('+00:00', 'Z')
                },
                "timestamp": past_timestamp_3.isoformat().replace('+00:00', 'Z'),
                "category": "test",
                "tags": []
            },
        ]
        
        mock_memory_interface.retrieve.return_value = {
            "memories": mock_memories,
            "total_count": 3,
            "query_metadata": {}
        }
        
        result = decay.apply_decay()
        
        assert result.memories_processed == 3
        assert result.memories_updated == 3
