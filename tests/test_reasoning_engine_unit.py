"""
Unit Tests for Reasoning_Engine Class

Tests the Reasoning_Engine orchestration with mocked dependencies to verify:
- Dependency injection works correctly
- The reason() method orchestrates the pipeline properly
- Components are called in the correct order
- Results are returned correctly
"""

import pytest
from luma.core.reasoning.reasoning_engine import Reasoning_Engine
from luma.core.reasoning.prompt_builder import Prompt_Builder
from luma.core.reasoning.llm_client_interface import LLM_Client_Interface
from luma.core.reasoning.schemas import Reasoning_Result


class Mock_LLM_Client(LLM_Client_Interface):
    """Mock LLM client for testing."""
    
    def __init__(self, response: str = "Answer: Test response\nUsed Memories: mem_1"):
        self.response = response
        self.last_prompt = None
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        """Return a predefined response and track the prompt."""
        self.last_prompt = prompt
        self.call_count += 1
        return self.response


class TestReasoningEngineInitialization:
    """Test suite for Reasoning_Engine initialization."""
    
    def test_initialization_with_dependencies(self):
        """Test that Reasoning_Engine initializes with injected dependencies."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        
        engine = Reasoning_Engine(
            llm_client=llm_client,
            prompt_builder=prompt_builder
        )
        
        assert engine is not None
        assert engine._llm_client is llm_client
        assert engine._prompt_builder is prompt_builder
        assert engine._response_formatter is not None
    
    def test_initialization_stores_llm_client(self):
        """Test that Reasoning_Engine stores the LLM client correctly."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        assert engine._llm_client is llm_client
    
    def test_initialization_stores_prompt_builder(self):
        """Test that Reasoning_Engine stores the prompt builder correctly."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        assert engine._prompt_builder is prompt_builder


class TestReasoningEngineOrchestration:
    """Test suite for Reasoning_Engine.reason() method orchestration."""
    
    def test_reason_returns_reasoning_result(self):
        """Test that reason() returns a Reasoning_Result object."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert isinstance(result, Reasoning_Result)
    
    def test_reason_calls_llm_client(self):
        """Test that reason() calls the LLM client."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        engine.reason("Test query", {})
        
        assert llm_client.call_count == 1
    
    def test_reason_passes_prompt_to_llm_client(self):
        """Test that reason() passes the constructed prompt to the LLM client."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What do I like?"
        context = {"memories": [], "metadata": {}}
        
        engine.reason(query, context)
        
        # Verify the LLM client received a prompt
        assert llm_client.last_prompt is not None
        assert isinstance(llm_client.last_prompt, str)
        assert len(llm_client.last_prompt) > 0
    
    def test_reason_prompt_contains_query(self):
        """Test that the prompt passed to LLM contains the query."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        query = "What programming languages do I like?"
        context = {"memories": [], "metadata": {}}
        
        engine.reason(query, context)
        
        # Verify the prompt contains the query
        assert query in llm_client.last_prompt
    
    def test_reason_with_empty_context(self):
        """Test that reason() works with empty context."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert isinstance(result, Reasoning_Result)
        assert result.answer is not None
    
    def test_reason_with_populated_context(self):
        """Test that reason() works with populated context."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}},
                {"id": "mem_2", "content": "User is learning ML", "metadata": {}}
            ],
            "metadata": {"user_id": "123"}
        }
        
        result = engine.reason("What do I like?", context)
        
        assert isinstance(result, Reasoning_Result)
        assert result.answer is not None
    
    def test_reason_prompt_contains_memory_ids(self):
        """Test that the prompt contains memory IDs from context."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        context = {
            "memories": [
                {"id": "mem_1", "content": "User likes Python", "metadata": {}},
                {"id": "mem_2", "content": "User is learning ML", "metadata": {}}
            ],
            "metadata": {}
        }
        
        engine.reason("What do I like?", context)
        
        # Verify the prompt contains memory IDs
        assert "mem_1" in llm_client.last_prompt
        assert "mem_2" in llm_client.last_prompt


class TestReasoningEngineResultFormatting:
    """Test suite for Reasoning_Engine result formatting."""
    
    def test_reason_result_has_answer(self):
        """Test that the result has an answer field."""
        llm_client = Mock_LLM_Client("Answer: Test answer")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert hasattr(result, 'answer')
        assert result.answer is not None
        assert isinstance(result.answer, str)
    
    def test_reason_result_has_used_memories(self):
        """Test that the result has a used_memories field."""
        llm_client = Mock_LLM_Client("Answer: Test\nUsed Memories: mem_1, mem_2")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert hasattr(result, 'used_memories')
        assert isinstance(result.used_memories, list)
    
    def test_reason_result_extracts_used_memories(self):
        """Test that used memories are extracted from LLM response."""
        llm_client = Mock_LLM_Client("Answer: Test\nUsed Memories: mem_1, mem_2")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert "mem_1" in result.used_memories
        assert "mem_2" in result.used_memories
    
    def test_reason_result_has_confidence(self):
        """Test that the result has a confidence field."""
        llm_client = Mock_LLM_Client("Answer: Test\nConfidence: 0.95")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert hasattr(result, 'confidence')
    
    def test_reason_result_extracts_confidence(self):
        """Test that confidence is extracted from LLM response."""
        llm_client = Mock_LLM_Client("Answer: Test\nConfidence: 0.95")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        assert result.confidence == 0.95
    
    def test_reason_handles_unparseable_response(self):
        """Test that reason() handles unparseable LLM responses gracefully."""
        llm_client = Mock_LLM_Client("Random unparseable output")
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        result = engine.reason("Test query", {})
        
        # Should return a result with the raw output as answer
        assert isinstance(result, Reasoning_Result)
        assert result.answer == "Random unparseable output"
        assert result.used_memories == []
        assert result.confidence is None


class TestReasoningEngineProviderAgnostic:
    """Test suite for provider-agnostic design."""
    
    def test_works_with_different_llm_implementations(self):
        """Test that Reasoning_Engine works with different LLM implementations."""
        # Create two different mock LLM clients
        class Mock_LLM_A(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Response from LLM A"
        
        class Mock_LLM_B(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Response from LLM B"
        
        prompt_builder = Prompt_Builder()
        
        # Test with LLM A
        engine_a = Reasoning_Engine(Mock_LLM_A(), prompt_builder)
        result_a = engine_a.reason("Test", {})
        assert "LLM A" in result_a.answer
        
        # Test with LLM B
        engine_b = Reasoning_Engine(Mock_LLM_B(), prompt_builder)
        result_b = engine_b.reason("Test", {})
        assert "LLM B" in result_b.answer
    
    def test_no_direct_dependency_on_specific_provider(self):
        """Test that Reasoning_Engine has no direct dependency on specific providers."""
        # This test verifies the architecture by checking that we can
        # instantiate the engine with any LLM_Client_Interface implementation
        
        class Custom_LLM(LLM_Client_Interface):
            def generate(self, prompt: str) -> str:
                return "Answer: Custom response"
        
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(Custom_LLM(), prompt_builder)
        
        result = engine.reason("Test", {})
        assert result.answer == "Custom response"


class TestReasoningEngineDecoupling:
    """Test suite for decoupling from storage and retrieval."""
    
    def test_accepts_context_as_generic_dict(self):
        """Test that reason() accepts context as a generic dictionary."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Test with various context structures
        contexts = [
            {},
            {"memories": []},
            {"memories": [], "metadata": {}},
            {"memories": [{"id": "1", "content": "test"}], "metadata": {"key": "value"}},
            {"custom_field": "value"},  # Non-standard structure
        ]
        
        for context in contexts:
            result = engine.reason("Test", context)
            assert isinstance(result, Reasoning_Result)
    
    def test_no_assumptions_about_context_source(self):
        """Test that Reasoning_Engine makes no assumptions about context source."""
        llm_client = Mock_LLM_Client()
        prompt_builder = Prompt_Builder()
        engine = Reasoning_Engine(llm_client, prompt_builder)
        
        # Context could come from anywhere - the engine doesn't care
        context_from_retrieval = {"memories": [{"id": "1", "content": "from retrieval"}]}
        context_from_cache = {"memories": [{"id": "2", "content": "from cache"}]}
        context_from_api = {"memories": [{"id": "3", "content": "from API"}]}
        
        result1 = engine.reason("Test", context_from_retrieval)
        result2 = engine.reason("Test", context_from_cache)
        result3 = engine.reason("Test", context_from_api)
        
        # All should work without issues
        assert isinstance(result1, Reasoning_Result)
        assert isinstance(result2, Reasoning_Result)
        assert isinstance(result3, Reasoning_Result)
