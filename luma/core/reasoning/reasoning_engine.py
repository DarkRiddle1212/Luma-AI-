"""
Reasoning Engine for the Luma cognitive memory assistant.

This module provides the Reasoning_Engine class that orchestrates the complete
reasoning pipeline: prompt construction, LLM interaction, and response formatting.
The engine follows Clean Architecture principles with dependency injection for
testability and provider-agnostic design.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from .llm_client_interface import LLM_Client_Interface
from .prompt_builder import Prompt_Builder
from .response_formatter import Response_Formatter
from .schemas import Reasoning_Result

if TYPE_CHECKING:
    from luma.core.metrics_collector import MetricsCollector
    from luma.core.structured_logger import StructuredLogger
    from luma.core.memory_write.memory_write_engine import MemoryWriteEngine


class Reasoning_Engine:
    """
    Orchestrates the reasoning pipeline for processing queries with context.
    
    The Reasoning_Engine coordinates three main components:
    1. Prompt_Builder: Constructs structured prompts from queries and context
    2. LLM_Client_Interface: Sends prompts to language models and receives responses
    3. Response_Formatter: Structures LLM output into typed result objects
    
    The engine uses dependency injection for the LLM client and prompt builder,
    enabling testing without external API calls and supporting multiple LLM providers
    without code changes.
    
    Architecture:
    - Provider-agnostic: Works with any LLM_Client_Interface implementation
    - Decoupled: No dependencies on storage or retrieval layers
    - Testable: Accepts mock implementations for unit testing
    - Extensible: Designed to support future features like multi-step reasoning
    
    Extension Points:
    
    Multi-Step Reasoning:
        The engine can be extended to support iterative reasoning workflows where
        the LLM generates intermediate steps before producing a final answer.
        
        To implement multi-step reasoning:
        1. Extend the reason() method to support a loop parameter or create a
           reason_iterative() method that calls reason() multiple times
        2. Modify Prompt_Builder to include previous reasoning steps in the prompt
        3. Add a step counter and termination condition (max steps or completion signal)
        4. Store intermediate results in a reasoning chain structure
        5. Update Response_Formatter to handle step-by-step outputs
        
        Example extension pattern:
            def reason_iterative(self, query: str, context: Dict[str, Any], 
                               max_steps: int = 3) -> Reasoning_Result:
                reasoning_chain = []
                current_query = query
                
                for step in range(max_steps):
                    # Add previous steps to context
                    step_context = {**context, "previous_steps": reasoning_chain}
                    
                    # Execute reasoning step
                    result = self.reason(current_query, step_context)
                    reasoning_chain.append(result)
                    
                    # Check for completion or generate next query
                    if self._is_reasoning_complete(result):
                        break
                    current_query = self._generate_next_step_query(result)
                
                return self._consolidate_reasoning_chain(reasoning_chain)
    
    Tool-Calling:
        The engine can be extended to support tool invocation between reasoning steps,
        allowing the LLM to request external data or perform actions.
        
        To implement tool-calling:
        1. Extend LLM_Client_Interface with a generate_with_tools() method that
           accepts available tools and returns structured tool call requests
        2. Create a Tool_Registry to manage available tools and their schemas
        3. Add a Tool_Executor component to invoke tools and format results
        4. Modify the reason() method to detect tool calls in LLM responses
        5. Implement a loop: LLM generates tool calls → execute tools → 
           inject results into next prompt → LLM generates final answer
        6. Update Response_Formatter to parse tool call requests from LLM output
        
        Example extension pattern:
            def reason_with_tools(self, query: str, context: Dict[str, Any],
                                tools: List[Tool]) -> Reasoning_Result:
                tool_registry = Tool_Registry(tools)
                max_iterations = 5
                
                for iteration in range(max_iterations):
                    # Build prompt with available tools
                    prompt = self._prompt_builder.build_prompt_with_tools(
                        query, context, tool_registry.get_schemas()
                    )
                    
                    # Get LLM response (may include tool calls)
                    llm_response = self._llm_client.generate(prompt)
                    
                    # Check if LLM wants to call tools
                    tool_calls = self._response_formatter.extract_tool_calls(llm_response)
                    
                    if not tool_calls:
                        # No tools requested, return final answer
                        return self._response_formatter.format_response(llm_response)
                    
                    # Execute requested tools
                    tool_results = self._execute_tools(tool_calls, tool_registry)
                    
                    # Inject tool results into context for next iteration
                    context = {**context, "tool_results": tool_results}
                
                # Max iterations reached, return best available answer
                return self._response_formatter.format_response(llm_response)
    
    Example:
        >>> from luma.core.reasoning import (
        ...     Reasoning_Engine,
        ...     Prompt_Builder,
        ...     OpenAI_LLM_Client
        ... )
        >>> 
        >>> # Initialize with real OpenAI client
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
        ...         {"id": "mem_1", "content": "User likes Python", "metadata": {}}
        ...     ],
        ...     "metadata": {}
        ... }
        >>> result = engine.reason("What do I like?", context)
        >>> print(result.answer)
        "Based on your memories, you like Python."
        >>> print(result.used_memories)
        ['mem_1']
    
    Testing Example:
        >>> # Use mock client for testing
        >>> class Mock_LLM_Client(LLM_Client_Interface):
        ...     def generate(self, prompt: str) -> str:
        ...         return "Answer: Test response\\nUsed Memories: mem_1"
        >>> 
        >>> mock_client = Mock_LLM_Client()
        >>> engine = Reasoning_Engine(
        ...     llm_client=mock_client,
        ...     prompt_builder=Prompt_Builder()
        ... )
        >>> result = engine.reason("Test query", {})
        >>> assert result.answer == "Test response"
    """
    
    def __init__(
            self,
            llm_client: LLM_Client_Interface,
            prompt_builder: Prompt_Builder,
            metrics_collector: Optional['MetricsCollector'] = None,
            logger: Optional['StructuredLogger'] = None,
            memory_write_engine: Optional['MemoryWriteEngine'] = None
        ):
            """
            Initialize the Reasoning_Engine with injected dependencies.

            Args:
                llm_client (LLM_Client_Interface): The LLM client implementation to use
                    for generating responses. Can be OpenAI_LLM_Client, a local model
                    client, or a mock implementation for testing.
                prompt_builder (Prompt_Builder): The prompt builder to use for
                    constructing structured prompts from queries and context.
                metrics_collector (Optional[MetricsCollector]): Optional metrics collector
                    for recording performance metrics and timing information.
                logger (Optional[StructuredLogger]): Optional structured logger for
                    emitting reasoning pipeline events.
                memory_write_engine (Optional[MemoryWriteEngine]): Optional memory write
                    engine for processing and storing valuable memories after response
                    generation.

            Example:
                >>> from luma.core.reasoning import (
                ...     Reasoning_Engine,
                ...     Prompt_Builder,
                ...     OpenAI_LLM_Client
                ... )
                >>> 
                >>> llm_client = OpenAI_LLM_Client(api_key="sk-...")
                >>> prompt_builder = Prompt_Builder()
                >>> engine = Reasoning_Engine(llm_client, prompt_builder)

                >>> # With observability
                >>> from luma.core.metrics_collector import MetricsCollector
                >>> from luma.core.structured_logger import StructuredLogger
                >>> 
                >>> metrics = MetricsCollector()
                >>> logger = StructuredLogger()
                >>> engine = Reasoning_Engine(
                ...     llm_client,
                ...     prompt_builder,
                ...     metrics_collector=metrics,
                ...     logger=logger
                ... )

                >>> # With memory write engine
                >>> from luma.core.memory_write import MemoryWriteEngine
                >>> memory_write_engine = MemoryWriteEngine(extractor, scorer, writer)
                >>> engine = Reasoning_Engine(
                ...     llm_client,
                ...     prompt_builder,
                ...     memory_write_engine=memory_write_engine
                ... )
            """
            self._llm_client = llm_client
            self._prompt_builder = prompt_builder
            self._response_formatter = Response_Formatter()

            # Store observability dependencies (optional)
            self.metrics_collector = metrics_collector
            self.logger = logger

            # Store memory write engine (optional)
            self._memory_write_engine = memory_write_engine

    
    def reason(self, query: str, context: Dict[str, Any]) -> Reasoning_Result:
        """
        Process a query with context and return a structured result.
        
        This method orchestrates the complete reasoning pipeline:
        1. Constructs a structured prompt using the Prompt_Builder
        2. Sends the prompt to the LLM via the LLM_Client_Interface
        3. Formats the LLM response using the Response_Formatter
        4. Returns a structured Reasoning_Result object
        
        The method accepts context as a generic dictionary, remaining decoupled
        from specific retrieval or storage implementations. The context typically
        contains memories and metadata provided by the Context Injection Engine.
        
        When observability is enabled, the method emits events at key pipeline
        stages and records timing metrics for performance monitoring.
        
        Args:
            query (str): The user's query or question to be processed.
            context (Dict[str, Any]): Context dictionary containing memories and metadata.
                Expected structure: {"memories": [...], "metadata": {...}}
                The memories list should contain dictionaries with "id", "content",
                and "metadata" fields.
        
        Returns:
            Reasoning_Result: Structured result containing:
                - answer (str): The generated response text
                - used_memories (List[str]): Memory IDs referenced in the answer
                - confidence (Optional[float]): Confidence score if available
        
        Raises:
            Exception: May raise exceptions from the LLM client (API errors,
                authentication failures, rate limits, etc.). Callers should
                handle these appropriately.
        
        Example:
            >>> engine = Reasoning_Engine(llm_client, prompt_builder)
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
            >>> result = engine.reason(
            ...     "What programming language should I use for ML?",
            ...     context
            ... )
            >>> print(result.answer)
            "Based on your preferences, Python is an excellent choice..."
            >>> print(result.used_memories)
            ['mem_1', 'mem_2']
        """
        import time
        
        # Start timing measurement
        start_time = time.perf_counter()
        
        # Calculate metadata
        query_length = len(query)
        context_size = len(context.get('memories', []))
        
        try:
            # Emit reasoning_started event
            if self.logger is not None:
                self.logger.log('reasoning_started', {
                    'query_length': query_length,
                    'context_size': context_size
                })
            
            # Step 1: Construct prompt from query and context
            prompt = self._prompt_builder.build_prompt(query, context)
            
            # Emit prompt_generated event
            if self.logger is not None:
                self.logger.log('prompt_generated', {
                    'prompt_length': len(prompt),
                    'query_length': query_length,
                    'context_size': context_size
                })
            
            # Step 2: Send prompt to LLM and get response
            llm_response = self._llm_client.generate(prompt)
            
            # Emit llm_response_received event
            if self.logger is not None:
                self.logger.log('llm_response_received', {
                    'response_length': len(llm_response),
                    'query_length': query_length,
                    'context_size': context_size
                })
            
            # Step 3: Format the LLM response into structured result
            result = self._response_formatter.format_response(llm_response)
            
            # Step 4: Process memories with Memory Write Engine (if available)
            if self._memory_write_engine is not None:
                try:
                    memory_result = self._memory_write_engine.process(
                        user_query=query,
                        system_response=result.answer
                    )
                    
                    # Log memory write results
                    if self.logger is not None:
                        self.logger.log('memory_write_completed', {
                            'stored_count': len(memory_result.stored_memories),
                            'ignored_count': len(memory_result.ignored_memories)
                        })
                except Exception as e:
                    # Log error but don't fail the reasoning pipeline
                    if self.logger is not None:
                        self.logger.log('memory_write_error', {
                            'error': str(e)
                        })
            
            # Step 5: Return the structured result
            return result
        finally:
            # Calculate processing duration
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Record timing metric
            if self.metrics_collector is not None:
                self.metrics_collector.record_duration('reasoning_latency_ms', duration_ms)
            
            # Emit reasoning_completed event
            if self.logger is not None:
                self.logger.log('reasoning_completed', {
                    'query_length': query_length,
                    'context_size': context_size,
                    'duration_ms': duration_ms
                })


__all__ = ['Reasoning_Engine']
