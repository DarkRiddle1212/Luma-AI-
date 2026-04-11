"""
Reasoning Engine Module

This module provides a modular reasoning layer for the Luma cognitive memory assistant
that processes user queries with injected context, constructs structured prompts,
interfaces with language models, and returns structured responses.

The module follows Clean Architecture principles to ensure modularity, testability,
and extensibility while remaining provider-agnostic and decoupled from retrieval
and storage layers.

Public Interface:
    - Reasoning_Engine: Main orchestrator for the reasoning pipeline
    - Prompt_Builder: Constructs structured prompts from queries and context
    - Response_Formatter: Formats LLM output into structured results
    - LLM_Client_Interface: Abstract interface for language model providers
    - OpenAI_LLM_Client: OpenAI API implementation of LLM_Client_Interface
    - Reasoning_Request: Data model for incoming reasoning requests
    - Reasoning_Context: Data model for injected context structure
    - Reasoning_Result: Data model for structured reasoning results

Basic Usage with OpenAI Client:
    >>> from luma.core.reasoning import (
    ...     Reasoning_Engine,
    ...     Prompt_Builder,
    ...     OpenAI_LLM_Client
    ... )
    >>> 
    >>> # Initialize components with OpenAI client
    >>> llm_client = OpenAI_LLM_Client(api_key="sk-...")
    >>> prompt_builder = Prompt_Builder()
    >>> engine = Reasoning_Engine(
    ...     llm_client=llm_client,
    ...     prompt_builder=prompt_builder
    ... )
    >>> 
    >>> # Process a query with context
    >>> context = {
    ...     "memories": [
    ...         {
    ...             "id": "mem_1",
    ...             "content": "User prefers Python for data science",
    ...             "metadata": {"timestamp": "2024-01-01"}
    ...         },
    ...         {
    ...             "id": "mem_2",
    ...             "content": "User is learning machine learning",
    ...             "metadata": {"timestamp": "2024-01-02"}
    ...         }
    ...     ],
    ...     "metadata": {"user_id": "123"}
    ... }
    >>> 
    >>> result = engine.reason(
    ...     query="What programming language should I use for ML?",
    ...     context=context
    ... )
    >>> 
    >>> print(result.answer)
    "Based on your preferences, Python is an excellent choice for machine learning..."
    >>> print(result.used_memories)
    ['mem_1', 'mem_2']
    >>> print(result.confidence)
    0.95

Usage with Mock Client for Testing:
    >>> from luma.core.reasoning import (
    ...     Reasoning_Engine,
    ...     Prompt_Builder,
    ...     LLM_Client_Interface
    ... )
    >>> 
    >>> # Create a mock LLM client for testing
    >>> class Mock_LLM_Client(LLM_Client_Interface):
    ...     def __init__(self, response: str):
    ...         self.response = response
    ...     
    ...     def generate(self, prompt: str) -> str:
    ...         return self.response
    >>> 
    >>> # Initialize engine with mock client
    >>> mock_response = '''
    ... Answer: Python is great for machine learning because it has
    ... excellent libraries like scikit-learn and TensorFlow.
    ... 
    ... Used Memories: mem_1, mem_2
    ... Confidence: 0.95
    ... '''
    >>> mock_client = Mock_LLM_Client(response=mock_response)
    >>> engine = Reasoning_Engine(
    ...     llm_client=mock_client,
    ...     prompt_builder=Prompt_Builder()
    ... )
    >>> 
    >>> # Test without making real API calls
    >>> result = engine.reason("Test query", {"memories": []})
    >>> assert "Python is great" in result.answer
    >>> assert result.used_memories == ['mem_1', 'mem_2']
    >>> assert result.confidence == 0.95

Dependency Injection Pattern:
    >>> from luma.core.reasoning import (
    ...     Reasoning_Engine,
    ...     Prompt_Builder,
    ...     OpenAI_LLM_Client,
    ...     LLM_Client_Interface
    ... )
    >>> 
    >>> # Define a factory function for creating engines
    >>> def create_reasoning_engine(
    ...     llm_client: LLM_Client_Interface,
    ...     prompt_builder: Prompt_Builder = None
    ... ) -> Reasoning_Engine:
    ...     '''Factory function demonstrating dependency injection.'''
    ...     if prompt_builder is None:
    ...         prompt_builder = Prompt_Builder()
    ...     
    ...     return Reasoning_Engine(
    ...         llm_client=llm_client,
    ...         prompt_builder=prompt_builder
    ...     )
    >>> 
    >>> # Production: Inject OpenAI client
    >>> production_engine = create_reasoning_engine(
    ...     llm_client=OpenAI_LLM_Client(api_key="sk-...")
    ... )
    >>> 
    >>> # Testing: Inject mock client
    >>> class Test_LLM_Client(LLM_Client_Interface):
    ...     def generate(self, prompt: str) -> str:
    ...         return "Answer: Test response\\nUsed Memories: mem_1"
    >>> 
    >>> test_engine = create_reasoning_engine(
    ...     llm_client=Test_LLM_Client()
    ... )
    >>> 
    >>> # Both engines have the same interface
    >>> result = test_engine.reason("Test query", {"memories": []})
    >>> assert result.answer == "Test response"

Advanced Usage with Observability:
    >>> from luma.core.reasoning import Reasoning_Engine, Prompt_Builder, OpenAI_LLM_Client
    >>> from luma.core.metrics_collector import MetricsCollector
    >>> from luma.core.structured_logger import StructuredLogger
    >>> 
    >>> # Initialize with observability components
    >>> llm_client = OpenAI_LLM_Client(api_key="sk-...")
    >>> prompt_builder = Prompt_Builder()
    >>> metrics_collector = MetricsCollector()
    >>> logger = StructuredLogger()
    >>> 
    >>> engine = Reasoning_Engine(
    ...     llm_client=llm_client,
    ...     prompt_builder=prompt_builder,
    ...     metrics_collector=metrics_collector,
    ...     logger=logger
    ... )
    >>> 
    >>> # Process query - events and metrics are automatically recorded
    >>> result = engine.reason("What do I like?", context)
    >>> 
    >>> # Check recorded metrics
    >>> latency = metrics_collector.get_metric('reasoning_latency_ms')
    >>> print(f"Reasoning took {latency}ms")
    >>> 
    >>> # Check logged events
    >>> events = logger.get_events()
    >>> assert 'reasoning_started' in [e['event'] for e in events]
    >>> assert 'reasoning_completed' in [e['event'] for e in events]

Using Data Models:
    >>> from luma.core.reasoning import Reasoning_Request, Reasoning_Context, Reasoning_Result
    >>> 
    >>> # Create structured request using data models
    >>> context = Reasoning_Context(
    ...     memories=[
    ...         {"id": "mem_1", "content": "User likes Python", "metadata": {}}
    ...     ],
    ...     metadata={"user_id": "123"}
    ... )
    >>> 
    >>> request = Reasoning_Request(
    ...     query="What programming languages do I like?",
    ...     context=context
    ... )
    >>> 
    >>> # Data models provide validation and type safety
    >>> print(request.query)
    "What programming languages do I like?"
    >>> print(request.context.memories[0]['id'])
    "mem_1"
"""

# Import data models from schemas
from luma.core.reasoning.schemas import (
    Reasoning_Request,
    Reasoning_Context,
    Reasoning_Result
)

# Import abstract interface
from luma.core.reasoning.llm_client_interface import LLM_Client_Interface

# Import concrete implementations
from luma.core.reasoning.openai_llm_client import OpenAI_LLM_Client

# Import prompt builder
from luma.core.reasoning.prompt_builder import Prompt_Builder

# Import response formatter
from luma.core.reasoning.response_formatter import Response_Formatter

# Import reasoning engine
from luma.core.reasoning.reasoning_engine import Reasoning_Engine

# Temporary backward compatibility: Import old ReasoningEngine from parent module
import sys
import importlib.util

# Load the old reasoning.py module directly to avoid circular imports
spec = importlib.util.spec_from_file_location(
    "luma.core.reasoning_old",
    "luma/core/reasoning.py"
)
if spec and spec.loader:
    reasoning_old = importlib.util.module_from_spec(spec)
    sys.modules["luma.core.reasoning_old"] = reasoning_old
    spec.loader.exec_module(reasoning_old)
    ReasoningEngine = reasoning_old.ReasoningEngine
    Intent = reasoning_old.Intent
else:
    # Fallback if module loading fails
    ReasoningEngine = None
    Intent = None

__all__ = [
    'Reasoning_Request',
    'Reasoning_Context',
    'Reasoning_Result',
    'LLM_Client_Interface',
    'OpenAI_LLM_Client',
    'Prompt_Builder',
    'Response_Formatter',
    'Reasoning_Engine',  # New reasoning engine
    'ReasoningEngine',  # Backward compatibility
    'Intent'  # Intent enum from reasoning.py
]
