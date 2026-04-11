"""
Integration Tests for Complete Reasoning Pipeline

Tests the complete Reasoning_Engine pipeline from query input to structured result output.
These tests verify end-to-end functionality with mock LLM clients, empty and populated
contexts, and observability integration.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.2, 8.3, 8.4**
**Validates: Task 9.3 - Write integration tests for complete reasoning pipeline**
"""

import pytest
from unittest.mock import Mock
from luma.core.reasoning import (
    Reasoning_Engine,
    Prompt_Builder,
    LLM_Client_Interface,
    Reasoning_Result
)
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger


class Mock_LLM_Client(LLM_Client_Interface):
    """Mock LLM client for integration testing."""
    
    def __init__(self, response: str = None):
        """
        Initialize mock LLM client.
        
        Args:
            response: Optional predefined response. If None, generates response
                     based on the prompt content.
        """
        self.response = response
        self.prompts_received = []
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """
        Generate a mock response.
        
        Args:
            prompt: The prompt string received from the reasoning engine.
        
        Returns:
            A formatted response string with answer, used memories, and confidence.
        """
        self.prompts_received.append(prompt)
        self.call_count += 1
        
        if self.response is not None:
            return self.response
        
        # Generate response based on prompt content
        # Extract memory IDs from prompt if present
        memory_ids = []
        if "Memory ID:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if "Memory ID:" in line:
                    # Extract ID from format "Memory ID: mem_X"
                    parts = line.split("Memory ID:")
                    if len(parts) > 1:
                        mem_id = parts[1].strip().split()[0]
                        memory_ids.append(mem_id)
        
        # Build response
        answer = "This is a generated answer based on the provided context."
        used_memories_str = ", ".join(memory_ids) if memory_ids else ""
        
        response_parts = [f"Answer: {answer}"]
        if used_memories_str:
            response_parts.append(f"Used Memories: {used_memories_str}")
        response_parts.append("Confidence: 0.85")
        
        return "\n".join(response_parts)


class TestCompleteReasoningPipelineWithMockLLM:
    """Test end-to-end reasoning pipeline flow with mock LLM client."""
    
    def test_complete_pipeline_returns_reasoning_result(self):
        """Test that complete pipeline returns a Reasoning_Result object."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: Python is great\nUsed Memories: mem_1\nConfidence: 0.9"
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("What do I like?", context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "Python is great"
        assert "mem_1" in result.used_memories
        assert result.confidence == 0.9
    
    def test_pipeline_orchestrates_all_components(self):
        """Test that pipeline correctly orchestrates all components."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What programming languages do I know?"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User knows Python", "metadata": {}},
                {"id": "mem_2", "content": "User knows JavaScript", "metadata": {}}
            ],
            "metadata": {"user_id": "123"}
        }
        
        # Act
        result = engine.reason(query, context)
        
        # Assert - verify all components were used
        # 1. Prompt builder was used (query appears in prompt)
        assert llm_client.call_count == 1
        assert query in llm_client.prompts_received[0]
        
        # 2. LLM client was called
        assert len(llm_client.prompts_received) == 1
        
        # 3. Response formatter was used (result is structured)
        assert isinstance(result, Reasoning_Result)
        assert hasattr(result, 'answer')
        assert hasattr(result, 'used_memories')
        assert hasattr(result, 'confidence')
    
    def test_pipeline_passes_query_through_all_stages(self):
        """Test that query is correctly passed through all pipeline stages."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What is my favorite programming language?"
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason(query, context)
        
        # Assert - verify query made it through the pipeline
        prompt = llm_client.prompts_received[0]
        assert query in prompt
        assert isinstance(result, Reasoning_Result)
    
    def test_pipeline_handles_multiple_sequential_queries(self):
        """Test that pipeline correctly handles multiple sequential queries."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        queries = [
            "What do I like?",
            "What am I learning?",
            "What are my goals?"
        ]
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        results = [engine.reason(query, context) for query in queries]
        
        # Assert
        assert len(results) == 3
        assert llm_client.call_count == 3
        assert len(llm_client.prompts_received) == 3
        
        # Verify each query was processed independently
        for i, query in enumerate(queries):
            assert query in llm_client.prompts_received[i]
            assert isinstance(results[i], Reasoning_Result)


class TestPipelineWithEmptyContext:
    """Test reasoning pipeline with empty context."""
    
    def test_pipeline_works_with_empty_context_dict(self):
        """Test that pipeline works correctly with an empty context dictionary."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: I don't have any context about your preferences."
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Act
        result = engine.reason("What do I like?", {})
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "I don't have any context about your preferences."
        assert result.used_memories == []
    
    def test_pipeline_works_with_empty_memories_list(self):
        """Test that pipeline works with context containing empty memories list."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: No memories available to answer this question."
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [],
            "metadata": {"user_id": "123"}
        }
        
        # Act
        result = engine.reason("What do I like?", context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "No memories available to answer this question."
        assert result.used_memories == []
    
    def test_prompt_excludes_context_section_when_empty(self):
        """Test that prompt doesn't include context section when memories are empty."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert
        prompt = llm_client.prompts_received[0]
        # Prompt should not have a "Context:" section when memories are empty
        assert "Context:" not in prompt or "No relevant memories" in prompt
    
    def test_empty_context_still_produces_valid_result(self):
        """Test that empty context still produces a valid Reasoning_Result."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: General response without context\nConfidence: 0.5"
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Act
        result = engine.reason("Tell me about Python", {})
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "General response without context"
        assert result.used_memories == []
        assert result.confidence == 0.5


class TestPipelineWithPopulatedContext:
    """Test reasoning pipeline with populated context containing memories."""
    
    def test_pipeline_with_single_memory(self):
        """Test pipeline with context containing a single memory."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: You like Python\nUsed Memories: mem_1\nConfidence: 0.95"
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {
                    "id": "mem_1",
                    "content": "User's favorite language is Python",
                    "metadata": {"timestamp": "2024-01-01"}
                }
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("What do I like?", context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "You like Python"
        assert "mem_1" in result.used_memories
        assert result.confidence == 0.95
    
    def test_pipeline_with_multiple_memories(self):
        """Test pipeline with context containing multiple memories."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: You like Python and JavaScript\nUsed Memories: mem_1, mem_2\nConfidence: 0.9"
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}},
                {"id": "mem_2", "content": "User likes JavaScript", "metadata": {}},
                {"id": "mem_3", "content": "User is learning Rust", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("What programming languages do I like?", context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert "mem_1" in result.used_memories
        assert "mem_2" in result.used_memories
        assert len(result.used_memories) == 2
    
    def test_prompt_includes_all_memory_ids(self):
        """Test that prompt includes all memory IDs from context."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "Memory 1", "metadata": {}},
                {"id": "mem_2", "content": "Memory 2", "metadata": {}},
                {"id": "mem_3", "content": "Memory 3", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert
        prompt = llm_client.prompts_received[0]
        assert "mem_1" in prompt
        assert "mem_2" in prompt
        assert "mem_3" in prompt
    
    def test_prompt_includes_memory_content(self):
        """Test that prompt includes memory content from context."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User prefers Python for data science", "metadata": {}},
                {"id": "mem_2", "content": "User is learning machine learning", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason("What am I learning?", context)
        
        # Assert
        prompt = llm_client.prompts_received[0]
        assert "Python for data science" in prompt
        assert "machine learning" in prompt
    
    def test_pipeline_with_complex_context_metadata(self):
        """Test pipeline with context containing complex metadata."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {
                    "id": "mem_1",
                    "content": "User likes Python",
                    "metadata": {
                        "timestamp": "2024-01-01T10:00:00Z",
                        "source": "conversation",
                        "confidence": 0.95,
                        "tags": ["programming", "preferences"]
                    }
                }
            ],
            "metadata": {
                "user_id": "user_123",
                "session_id": "session_456",
                "device": "desktop",
                "location": "US"
            }
        }
        
        # Act
        result = engine.reason("What do I like?", context)
        
        # Assert - pipeline should handle complex metadata without errors
        assert isinstance(result, Reasoning_Result)
        assert llm_client.call_count == 1


class TestPipelineObservabilityIntegration:
    """Test reasoning pipeline integration with observability layer."""
    
    def test_pipeline_emits_all_observability_events(self):
        """Test that pipeline emits all expected observability events."""
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
        
        event_names = [call[0][0] for call in logger.log.call_args_list]
        assert event_names == [
            'reasoning_started',
            'prompt_generated',
            'llm_response_received',
            'reasoning_completed'
        ]
    
    def test_pipeline_records_timing_metrics(self):
        """Test that pipeline records timing metrics to metrics collector."""
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
        result = engine.reason("Test query", context)
        
        # Assert
        snapshot = metrics_collector.get_snapshot()
        assert 'reasoning_latency_ms' in snapshot['timers']
        assert snapshot['timers']['reasoning_latency_ms']['count'] == 1
        assert snapshot['timers']['reasoning_latency_ms']['mean'] >= 0
    
    def test_observability_events_include_correct_metadata(self):
        """Test that observability events include correct metadata."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: Test response with some content"
        )
        prompt_builder = Prompt_Builder()
        logger = Mock(spec=StructuredLogger)
        
        engine = Reasoning_Engine(
            llm_client,
            prompt_builder,
            logger=logger
        )
        
        query = "What do I like?"
        context = {
            "memories": [
                {"id": "mem_1", "content": "Memory 1", "metadata": {}},
                {"id": "mem_2", "content": "Memory 2", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason(query, context)
        
        # Assert - check reasoning_started event
        started_call = logger.log.call_args_list[0]
        started_payload = started_call[0][1]
        assert started_payload['query_length'] == len(query)
        assert started_payload['context_size'] == 2
        
        # Assert - check prompt_generated event
        prompt_call = logger.log.call_args_list[1]
        prompt_payload = prompt_call[0][1]
        assert 'prompt_length' in prompt_payload
        assert prompt_payload['prompt_length'] > 0
        
        # Assert - check llm_response_received event
        response_call = logger.log.call_args_list[2]
        response_payload = response_call[0][1]
        assert 'response_length' in response_payload
        assert response_payload['response_length'] > 0
        
        # Assert - check reasoning_completed event
        completed_call = logger.log.call_args_list[3]
        completed_payload = completed_call[0][1]
        assert 'duration_ms' in completed_payload
        assert completed_payload['duration_ms'] >= 0
    
    def test_pipeline_works_without_observability(self):
        """Test that pipeline works correctly when observability is disabled."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="Answer: Test response\nUsed Memories: mem_1"
        )
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
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "Test response"
        assert "mem_1" in result.used_memories
    
    def test_observability_events_emitted_even_on_error(self):
        """Test that observability events are emitted even when errors occur."""
        # Arrange
        llm_client = Mock(spec=LLM_Client_Interface)
        llm_client.generate.side_effect = RuntimeError("LLM API error")
        
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
        with pytest.raises(RuntimeError, match="LLM API error"):
            engine.reason("Test query", context)
        
        # Verify that events were still emitted (in finally block)
        assert logger.log.call_count >= 2
        event_names = [call[0][0] for call in logger.log.call_args_list]
        assert 'reasoning_started' in event_names
        assert 'reasoning_completed' in event_names
        
        # Verify metrics were still recorded
        snapshot = metrics_collector.get_snapshot()
        assert 'reasoning_latency_ms' in snapshot['timers']


class TestPipelineEdgeCases:
    """Test edge cases in the reasoning pipeline."""
    
    def test_pipeline_with_very_long_query(self):
        """Test pipeline handles very long queries correctly."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Create a very long query
        long_query = "What do I like? " * 100  # 1800+ characters
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason(long_query, context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert llm_client.call_count == 1
        assert long_query in llm_client.prompts_received[0]
    
    def test_pipeline_with_many_memories(self):
        """Test pipeline handles large number of memories correctly."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Create context with many memories
        memories = [
            {"id": f"mem_{i}", "content": f"Memory content {i}", "metadata": {}}
            for i in range(50)
        ]
        context = {"memories": memories, "metadata": {}}
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert llm_client.call_count == 1
        
        # Verify all memory IDs are in the prompt
        prompt = llm_client.prompts_received[0]
        for i in range(50):
            assert f"mem_{i}" in prompt
    
    def test_pipeline_with_special_characters_in_query(self):
        """Test pipeline handles special characters in query correctly."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What's my favorite language? (Python, JavaScript, or C++?)"
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason(query, context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert query in llm_client.prompts_received[0]
    
    def test_pipeline_with_unicode_characters(self):
        """Test pipeline handles unicode characters correctly."""
        # Arrange
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What do I like? 你好 🚀 café"
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes 日本語 and émojis 🎉", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        result = engine.reason(query, context)
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        prompt = llm_client.prompts_received[0]
        assert "你好" in prompt or query in prompt
    
    def test_pipeline_with_unparseable_llm_response(self):
        """Test pipeline handles unparseable LLM responses gracefully."""
        # Arrange
        llm_client = Mock_LLM_Client(
            response="This is just random text without proper formatting"
        )
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {"memories": [], "metadata": {}}
        
        # Act
        result = engine.reason("Test query", context)
        
        # Assert - should return result with raw output as answer
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "This is just random text without proper formatting"
        assert result.used_memories == []
        assert result.confidence is None


class TestPipelineProviderAgnostic:
    """Test that pipeline is provider-agnostic and works with different LLM implementations."""
    
    def test_pipeline_with_different_mock_implementations(self):
        """Test pipeline works with different LLM client implementations."""
        # Arrange
        class Custom_LLM_A(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Response from LLM A\nConfidence: 0.8"
        
        class Custom_LLM_B(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Response from LLM B\nUsed Memories: mem_1\nConfidence: 0.9"
        
        prompt_builder = Prompt_Builder()
        context = {
            "memories": [
                {"id": "mem_1", "content": "Test memory", "metadata": {}}
            ],
            "metadata": {}
        }
        
        # Act
        engine_a = Reasoning_Engine(Custom_LLM_A(), prompt_builder)
        result_a = engine_a.reason("Test query", context)
        
        engine_b = Reasoning_Engine(Custom_LLM_B(), prompt_builder)
        result_b = engine_b.reason("Test query", context)
        
        # Assert
        assert result_a.answer == "Response from LLM A"
        assert result_a.confidence == 0.8
        
        assert result_b.answer == "Response from LLM B"
        assert "mem_1" in result_b.used_memories
        assert result_b.confidence == 0.9
    
    def test_pipeline_accepts_any_llm_client_interface_implementation(self):
        """Test that pipeline accepts any LLM_Client_Interface implementation."""
        # Arrange
        class Minimal_LLM(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Minimal response"
        
        prompt_builder = Prompt_Builder()
        
        # Act - should not raise any errors
        engine = Reasoning_Engine(Minimal_LLM(), prompt_builder)
        result = engine.reason("Test", {})
        
        # Assert
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "Minimal response"
