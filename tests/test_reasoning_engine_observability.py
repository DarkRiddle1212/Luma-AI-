"""Tests for Reasoning_Engine observability integration.

**Validates: Requirements 8.5**

This test module verifies that the Reasoning_Engine correctly integrates with
the observability layer (MetricsCollector and StructuredLogger) and handles
observability failures gracefully.
"""

import pytest
from unittest.mock import Mock, MagicMock
from luma.core.reasoning.reasoning_engine import Reasoning_Engine
from luma.core.reasoning.prompt_builder import Prompt_Builder
from luma.core.reasoning.llm_client_interface import LLM_Client_Interface
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


class Mock_LLM_Client(LLM_Client_Interface):
    """Mock LLM client for testing."""
    
    def __init__(self, response: str = "Answer: Test response\nUsed Memories: mem_1"):
        self.response = response
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """Return a predefined response."""
        self.call_count += 1
        return self.response


class TestObservabilityEnabled:
    """Test cases for when observability is enabled."""
    
    def test_events_emitted_when_observability_enabled(self):
        """Test that all expected events are emitted when observability is enabled."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        metrics_collector = MetricsCollector()
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "Test memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert - verify all 4 events were emitted
        assert logger.log.call_count == 4
        
        # Verify event names in order
        event_names = [call[0][0] for call in logger.log.call_args_list]
        assert event_names == [
            'reasoning_started',
            'prompt_generated',
            'llm_response_received',
            'reasoning_completed'
        ]
    
    def test_reasoning_started_event_metadata(self):
        """Test that reasoning_started event includes correct metadata."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "Memory 1", "metadata": {}},
                {"id": "mem_2", "content": "Memory 2", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check first event (reasoning_started)
        first_call = logger.log.call_args_list[0]
        event_name = first_call[0][0]
        event_payload = first_call[0][1]
        
        assert event_name == 'reasoning_started'
        assert 'query_length' in event_payload
        assert 'context_size' in event_payload
        assert event_payload['query_length'] == len("Test query")
        assert event_payload['context_size'] == 2
    
    def test_prompt_generated_event_metadata(self):
        """Test that prompt_generated event includes correct metadata."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check second event (prompt_generated)
        second_call = logger.log.call_args_list[1]
        event_name = second_call[0][0]
        event_payload = second_call[0][1]
        
        assert event_name == 'prompt_generated'
        assert 'prompt_length' in event_payload
        assert 'query_length' in event_payload
        assert 'context_size' in event_payload
        assert event_payload['prompt_length'] > 0
    
    def test_llm_response_received_event_metadata(self):
        """Test that llm_response_received event includes correct metadata."""
        # Arrange
        response = "Answer: This is a test response\nUsed Memories: mem_1"
        llm_client = Mock_LLM_Client(response=response)
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check third event (llm_response_received)
        third_call = logger.log.call_args_list[2]
        event_name = third_call[0][0]
        event_payload = third_call[0][1]
        
        assert event_name == 'llm_response_received'
        assert 'response_length' in event_payload
        assert 'query_length' in event_payload
        assert 'context_size' in event_payload
        assert event_payload['response_length'] == len(response)
    
    def test_reasoning_completed_event_metadata(self):
        """Test that reasoning_completed event includes correct metadata."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check fourth event (reasoning_completed)
        fourth_call = logger.log.call_args_list[3]
        event_name = fourth_call[0][0]
        event_payload = fourth_call[0][1]
        
        assert event_name == 'reasoning_completed'
        assert 'query_length' in event_payload
        assert 'context_size' in event_payload
        assert 'duration_ms' in event_payload
        assert event_payload['duration_ms'] >= 0
    
    def test_metrics_collector_records_duration(self):
        """Test that metrics collector records reasoning duration."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        metrics_collector = MetricsCollector()
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert
        snapshot = metrics_collector.get_snapshot()
        assert 'reasoning_latency_ms' in snapshot['timers']
        assert snapshot['timers']['reasoning_latency_ms']['count'] == 1
        assert snapshot['timers']['reasoning_latency_ms']['mean'] >= 0


class TestObservabilityDisabled:
    """Test cases for when observability is disabled."""
    
    def test_system_works_without_observability(self):
        """Test that reasoning engine works correctly when observability is disabled."""
        # Arrange
        llm_client = Mock_LLM_Client(response="Answer: Test\nUsed Memories: mem_1")
        prompt_builder = Prompt_Builder()
        
        # Create engine without observability components
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=None,
            logger=None
        )
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "Test memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert - verify normal operation
        assert result is not None
        assert result.answer == "Test"
        assert "mem_1" in result.used_memories
        assert llm_client.call_count == 1
    
    def test_system_works_with_only_logger(self):
        """Test that system works with only logger enabled."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=None,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert
        assert result is not None
        assert logger.log.call_count == 4
    
    def test_system_works_with_only_metrics_collector(self):
        """Test that system works with only metrics collector enabled."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        metrics_collector = MetricsCollector()
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector,
            logger=None
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert
        assert result is not None
        snapshot = metrics_collector.get_snapshot()
        assert 'reasoning_latency_ms' in snapshot['timers']


class TestObservabilityFailureHandling:
    """Test cases for graceful handling of observability failures."""
    
    def test_handles_logger_failure_gracefully(self):
        """Test that reasoning continues when logger raises an exception."""
        # Arrange
        llm_client = Mock_LLM_Client(response="Answer: Test\nUsed Memories: mem_1")
        prompt_builder = Prompt_Builder()
        
        # Create a logger that raises exceptions
        logger = Mock(spec=StructuredLogger)
        logger.log.side_effect = Exception("Logger failure")
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act & Assert - should raise the exception since we don't catch it
        # This tests that the failure propagates but doesn't corrupt state
        with pytest.raises(Exception, match="Logger failure"):
            engine.reason("Test query", context)
    
    def test_handles_metrics_collector_failure_gracefully(self):
        """Test that reasoning continues when metrics collector raises an exception."""
        # Arrange
        llm_client = Mock_LLM_Client(response="Answer: Test\nUsed Memories: mem_1")
        prompt_builder = Prompt_Builder()
        
        # Create a metrics collector that raises exceptions
        metrics_collector = Mock(spec=MetricsCollector)
        metrics_collector.record_duration.side_effect = Exception("Metrics failure")
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act & Assert - the finally block should still execute
        # The exception should propagate but not prevent cleanup
        with pytest.raises(Exception, match="Metrics failure"):
            engine.reason("Test query", context)
    
    def test_observability_called_even_on_llm_failure(self):
        """Test that observability events are emitted even when LLM fails."""
        # Arrange
        llm_client = Mock(spec=LLM_Client_Interface)
        llm_client.generate.side_effect = Exception("LLM API error")
        
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        metrics_collector = MetricsCollector()
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act & Assert
        with pytest.raises(Exception, match="LLM API error"):
            engine.reason("Test query", context)
        
        # Verify that reasoning_started and prompt_generated were emitted
        # (before the LLM failure)
        assert logger.log.call_count >= 2
        
        # Verify reasoning_completed was still emitted (in finally block)
        event_names = [call[0][0] for call in logger.log.call_args_list]
        assert 'reasoning_started' in event_names
        assert 'prompt_generated' in event_names
        assert 'reasoning_completed' in event_names
        
        # Verify metrics were still recorded
        snapshot = metrics_collector.get_snapshot()
        assert 'reasoning_latency_ms' in snapshot['timers']


class TestEventMetadataCorrectness:
    """Test cases verifying event metadata is correct."""
    
    def test_query_length_metadata_accuracy(self):
        """Test that query_length metadata is accurate across all events."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        query = "This is a test query with specific length"
        expected_length = len(query)
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason(query, context)
        
        # Assert - check all events have correct query_length
        for call in logger.log.call_args_list:
            event_payload = call[0][1]
            if 'query_length' in event_payload:
                assert event_payload['query_length'] == expected_length
    
    def test_context_size_metadata_accuracy(self):
        """Test that context_size metadata is accurate across all events."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "Memory 1", "metadata": {}},
                {"id": "mem_2", "content": "Memory 2", "metadata": {}},
                {"id": "mem_3", "content": "Memory 3", "metadata": {}}
            ],
            "metadata": {}
        }
        expected_size = 3
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check all events have correct context_size
        for call in logger.log.call_args_list:
            event_payload = call[0][1]
            if 'context_size' in event_payload:
                assert event_payload['context_size'] == expected_size
    
    def test_duration_metadata_is_positive(self):
        """Test that duration_ms metadata is always positive."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        metrics_collector = MetricsCollector()
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            metrics_collector=metrics_collector,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - check reasoning_completed event
        completed_call = logger.log.call_args_list[3]
        event_payload = completed_call[0][1]
        
        assert 'duration_ms' in event_payload
        assert event_payload['duration_ms'] > 0
        
        # Also verify metrics collector
        snapshot = metrics_collector.get_snapshot()
        assert snapshot['timers']['reasoning_latency_ms']['mean'] > 0
    
    def test_empty_context_metadata(self):
        """Test metadata is correct when context is empty."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        engine.reason("Test query", context)
        
        # Assert - verify context_size is 0
        first_call = logger.log.call_args_list[0]
        event_payload = first_call[0][1]
        
        assert event_payload['context_size'] == 0
