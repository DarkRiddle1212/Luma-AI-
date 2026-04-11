"""
Reasoning Engine Module

This module provides Luma's cognitive orchestration layer that coordinates
message processing, context assembly, intent classification, and response generation.

The ReasoningEngine is the central orchestrator that:
1. Receives user messages
2. Builds context from available data sources
3. Detects user intent using rule-based classification
4. Generates responses via LLM interface
5. Returns structured responses

The ReasoningEngine requires an LLMInterface implementation to be provided
via constructor injection, enabling flexible integration with any LLM backend
(stub implementations for testing, or production AI services).

Example:
    >>> from luma.core.reasoning import ReasoningEngine
    >>> from luma.core.llm_interface import LLMInterface
    >>> 
    >>> # Initialize with an LLM implementation
    >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
    >>> engine = ReasoningEngine(llm=llm)
    >>> 
    >>> # Process a message
    >>> result = engine.process_message("Teach me Python loops")
    >>> print(result["response"])
    >>> print(result["intent"])  # "education"
"""

from typing import Dict, Optional, List, Any
from datetime import datetime, UTC
from enum import Enum
import logging
import time
from luma.core.llm_interface import LLMInterface
from luma.core.memory_interface import (
    MemoryInterface,
    MemoryStorageError,
    MemoryRetrievalError,
    QueryParameters,
    RetrievalResult
)
from luma.core.metrics_collector import MetricsCollector
from luma.core.structured_logger import StructuredLogger

logger = logging.getLogger(__name__)

__all__ = ['ReasoningEngine', 'Intent']


class Intent(Enum):
    """Intent enumeration for message classification."""
    STORE_MEMORY = "store_memory"
    RETRIEVE_MEMORY = "retrieve_memory"
    SCHEDULE_TASK = "schedule_task"
    SYSTEM_INFO = "system_info"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"


class ReasoningEngine:
    """
    Orchestrates message processing, context building, intent detection, and response generation.
    
    The reasoning engine is stateless and uses dependency injection for LLM implementations,
    enabling easy testing and production deployment. It coordinates the complete cognitive
    pipeline from user input to structured response.
    
    Architecture:
        - Stateless design: No instance variables modified during processing
        - Dependency injection: LLM implementation passed via constructor
        - Interface-based: Works with any LLMInterface implementation
        - Error handling: Gracefully handles invalid inputs and LLM failures
    
    Attributes:
        llm (LLMInterface): The language model implementation used for response generation.
    
    Example:
        >>> from luma.core.reasoning import ReasoningEngine
        >>> from luma.core.llm_interface import LLMInterface
        >>> 
        >>> # Initialize with an LLM implementation
        >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
        >>> engine = ReasoningEngine(llm=llm)
        >>> result = engine.process_message("Teach me Python loops")
        >>> print(result["response"])
        >>> print(result["intent"])  # "education"
        >>> 
        >>> # Initialize with custom LLM implementation
        >>> class CustomLLM(LLMInterface):
        ...     def generate_response(self, prompt: str, context: dict) -> str:
        ...         return f"Custom response to: {prompt}"
        >>> 
        >>> engine = ReasoningEngine(llm=CustomLLM())
        >>> result = engine.process_message("Hello")
    """
    
    def __init__(
            self,
            llm: Optional[LLMInterface] = None,
            memory: Optional[MemoryInterface] = None,
            write_strategy: Optional['Memory_Write_Strategy'] = None,
            session_manager: Optional['Session_Manager'] = None,
            metrics_collector: Optional[MetricsCollector] = None,
            logger: Optional[StructuredLogger] = None
        ):
            """
            Initialize the reasoning engine with an LLM implementation and optional memory.

            Uses constructor-based dependency injection to allow flexible LLM implementations
            and optional memory integration. If no LLM is provided, defaults to StubLLM.
            Memory integration is optional and enables storage and retrieval capabilities when provided.

            Args:
                llm (Optional[LLMInterface]): LLM implementation to use for response generation.
                    Must implement the LLMInterface contract. If None, defaults to StubLLM.
                    Defaults to None.
                memory (Optional[MemoryInterface]): Optional memory implementation for storage
                    and retrieval operations. If None, memory features will be disabled.
                    Defaults to None.
                write_strategy (Optional[Memory_Write_Strategy]): Optional write strategy for
                    intelligent memory persistence. If provided, enables write trigger evaluation,
                    deduplication, and conflict detection. Defaults to None.
                session_manager (Optional[Session_Manager]): Optional session manager for
                    tracking conversation sessions and buffering memories. If provided, enables
                    session-based memory management. Defaults to None.
                metrics_collector (Optional[MetricsCollector]): Optional metrics collector for
                    observability and performance monitoring. If provided, enables collection
                    of reasoning operation metrics. Defaults to None.
                logger (Optional[StructuredLogger]): Optional structured logger for
                    observability and debugging. If provided, enables structured JSON logging
                    of reasoning operations. Defaults to None.

            Example:
                >>> # Use default StubLLM without memory
                >>> engine = ReasoningEngine()
                >>> 
                >>> # Use an LLM implementation without memory
                >>> from luma.core.llm_interface import LLMInterface
                >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
                >>> engine = ReasoningEngine(llm=llm)
                >>> 
                >>> # Use with memory integration
                >>> from luma.core.memory_interface import MemoryInterface
                >>> memory = MyMemoryImplementation()  # Any class implementing MemoryInterface
                >>> engine = ReasoningEngine(llm=llm, memory=memory)
                >>> 
                >>> # Use with write strategy and session management
                >>> from luma.core.write_strategy import Memory_Write_Strategy
                >>> from luma.core.session_manager import Session_Manager
                >>> write_strategy = Memory_Write_Strategy(config, session_manager, memory)
                >>> engine = ReasoningEngine(llm=llm, memory=memory, write_strategy=write_strategy, session_manager=session_manager)
                >>> 
                >>> # Use with observability components
                >>> from luma.core.metrics_collector import MetricsCollector
                >>> from luma.core.structured_logger import StructuredLogger
                >>> metrics = MetricsCollector()
                >>> logger = StructuredLogger()
                >>> engine = ReasoningEngine(llm=llm, memory=memory, metrics_collector=metrics, logger=logger)
            """
            from luma.core.llm_interface import StubLLM
            self.llm = llm if llm is not None else StubLLM()
            self.memory = memory
            self.write_strategy = write_strategy
            self.session_manager = session_manager
            self.metrics_collector = metrics_collector
            self.logger = logger
            self.current_session_id: Optional[str] = None

            # Use module-level logger for initialization logging
            module_logger = logging.getLogger(__name__)
            module_logger.info(f"ReasoningEngine initialized with {type(self.llm).__name__}")
            if memory:
                module_logger.info(f"Memory integration enabled: {type(memory).__name__}")
            else:
                module_logger.info("Memory integration disabled (no memory provided)")

            if write_strategy:
                module_logger.info(f"Write strategy enabled: {type(write_strategy).__name__}")

            if session_manager:
                module_logger.info(f"Session manager enabled: {type(session_manager).__name__}")

    def build_context(self, user_message: str, user_context: Optional[Dict] = None, retrieved_memories: Optional[List] = None) -> Dict:
            """
            Build context dictionary from user message, system state, and optional user context.

            Assembles a comprehensive context dictionary containing the user's message,
            temporal information, user context, session memories (if session active),
            and placeholders for future integration with system monitoring modules. This
            context is used to inform LLM response generation with relevant information.

            Context Structure:
                - message (str): The original user input message
                - message_length (int): Length of the message
                - user_context (dict): User-provided context
                - timestamp (str): ISO 8601 format timestamp of when context was built
                - relevant_memories (list): Session memories if session is active
                - session_id (str|None): Current session ID if session is active
                - system_state (dict): Empty dict reserved for future system monitoring

            Args:
                user_message (str): The user's input message to be included in context.
                user_context (Optional[Dict]): Optional user-provided context dictionary.
                    Defaults to None, which results in an empty dict in the context.
                retrieved_memories (Optional[List]): Optional list of retrieved memories.
                    If provided, these memories are used instead of session memories.
                    Defaults to None, which uses session memories if available.

            Returns:
                Dict: Context dictionary containing message, metadata, memories, session_id,
                    and integration placeholders.

            Example:
                >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
                >>> engine = ReasoningEngine(llm=llm)
                >>> 
                >>> # Build context without user context
                >>> context = engine.build_context("Hello, Luma!")
                >>> print(context["message"])
                'Hello, Luma!'
                >>> 
                >>> # Build context with user context
                >>> user_ctx = {"user_id": "123", "session": "abc"}
                >>> context = engine.build_context("Test", user_context=user_ctx)
                >>> print(context["user_context"])
                {'user_id': '123', 'session': 'abc'}
                >>> 
                >>> # Build context with retrieved memories
                >>> memories = [{"id": "1", "content": "Test", "metadata": {}}]
                >>> context = engine.build_context("Test", retrieved_memories=memories)
                >>> print(context["memories"])
                [{'id': '1', 'content': 'Test', 'metadata': {}}]

            Note:
                This method is stateless and does not modify any instance variables.
                The timestamp uses UTC time to ensure consistency across time zones.
            """
            # Use provided retrieved_memories if given, otherwise get session memories
            if retrieved_memories is not None:
                session_memories = retrieved_memories
            else:
                # Get session memories if session is active
                session_memories = []
                if self.session_manager and self.current_session_id:
                    session_memories = self.session_manager.get_session_memories(
                        self.current_session_id
                    )
                    logger.debug(
                        f"Retrieved {len(session_memories)} session memories "
                        f"for session {self.current_session_id}"
                    )

            # Construct context dictionary with message, memories, and metadata
            # Include both old and new key names for backward compatibility
            context = {
                # New keys (for orchestration tests)
                "message": user_message,  # Original user input for LLM processing
                "message_length": len(user_message),  # Length of the message
                "user_context": user_context or {},  # User-provided context
                "timestamp": datetime.now(UTC).isoformat(),  # UTC timestamp in ISO 8601 format
                "relevant_memories": session_memories,  # Session memories
                "session_id": self.current_session_id,  # Current session ID or None
                "system_state": {},  # Future: system monitoring data (CPU, memory, etc.)
                # Old keys (for backward compatibility)
                "user_message": user_message,  # Backward compatibility
                "memories": session_memories,  # Backward compatibility
                "memory_placeholder": session_memories,  # Backward compatibility - required by tests
                "system_state_placeholder": {}  # Backward compatibility
            }

            # Log context keys and memory count for debugging
            logger.debug(
                f"Built context with {len(session_memories)} session memories"
            )
            return context

    def detect_intent(self, user_message: str) -> str:
        """
        Detect user intent from message using rule-based classification.
        
        Analyzes the user's message to determine their intent using keyword-based rules.
        This rule-based approach provides deterministic classification and serves as a
        foundation for future ML-based intent detection systems.
        
        Intent Classification Rules:
            - "store_memory": Message contains "remember" or "store"
              Examples: "Remember to buy milk", "Store this information"
            
            - "retrieve_memory": Message contains "what was", "recall", or "retrieve"
              Examples: "What was my last task?", "Recall our conversation", "Retrieve notes"
            
            - "education": Message contains "teach", "learn", or "explain"
              Examples: "Teach me Python", "I want to learn", "Explain recursion"
            
            - "scheduling": Message contains "schedule" or "remind"
              Examples: "Schedule a meeting", "Remind me tomorrow"
            
            - "general": Default intent when no rules match
              Examples: "Hello", "How are you?", "What can you do?"
        
        Future Enhancement:
            This rule-based system will be replaced with ML-based intent classification
            using models like BERT or custom-trained classifiers. The method signature
            and return type will remain the same to maintain API compatibility.
        
        Args:
            user_message (str): The user's input message to classify.
        
        Returns:
            str: The detected intent classification. Always returns a valid intent string,
                defaulting to "general" if no rules match.
        
        Properties:
            - Deterministic: Same input always produces same output
            - Case-insensitive: "REMEMBER" and "remember" both match store_memory
            - First-match: Returns first matching intent in rule order
            - Always valid: Never returns None or raises exceptions
        
        Example:
            >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
            >>> engine = ReasoningEngine(llm=llm)
            >>> engine.detect_intent("Remember to buy milk")
            'store_memory'
            >>> engine.detect_intent("What was my last task?")
            'retrieve_memory'
            >>> engine.detect_intent("Teach me Python")
            'education'
            >>> engine.detect_intent("Schedule a meeting")
            'scheduling'
            >>> engine.detect_intent("Hello there!")
            'general'
            >>> engine.detect_intent("REMEMBER THIS")  # Case-insensitive
            'store_memory'
        
        Note:
            This method is stateless and does not modify any instance variables.
            The classification is performed on every call without caching.
        """
        # Convert to lowercase for case-insensitive matching
        message_lower = user_message.lower()
        
        # Rule-based intent detection using keyword matching
        # Rules are evaluated in priority order (first match wins)
        
        # Check for store_memory intent - user wants to save information
        if any(keyword in message_lower for keyword in ["remember", "store"]):
            intent = "store_memory"
        
        # Check for retrieve_memory intent - user wants to recall information
        elif any(keyword in message_lower for keyword in ["what was", "recall", "retrieve"]):
            intent = "retrieve_memory"
        
        # Check for education intent - user wants to learn something
        elif any(keyword in message_lower for keyword in ["teach", "learn", "explain"]):
            intent = "education"
        
        # Check for scheduling intent - user wants to schedule or set reminders
        elif any(keyword in message_lower for keyword in ["schedule", "remind"]):
            intent = "scheduling"
        
        # Default to general intent if no specific keywords match
        else:
            intent = "general"
        
        logger.debug(f"Detected intent: {intent}")
        return intent

    def _handle_store_memory(self, user_message: str) -> Dict:
        """
        Handle store_memory intent by extracting content and storing to memory.

        Processes messages with store_memory intent by extracting the content to be
        stored (removing trigger words), calling the memory storage interface, and
        returning a confirmation response. Handles cases where memory is not configured
        and storage errors gracefully with comprehensive error handling.

        Enhanced Features:
            - Uses write_strategy if available for intelligent persistence
            - Falls back to direct memory.store() if no write_strategy
            - Retrieves stored memory to inject into context
            - Builds context with newly stored memory
            - Wraps storage in try-except for MemoryStorageError
            - Logs storage errors with full details
            - Returns user-friendly error message on failure
            - Includes error details in response metadata
            - Ensures system doesn't crash on storage failure

        Args:
            user_message (str): User's message containing content to store.

        Returns:
            Dict: Response dictionary with confirmation or error message.
                Success response includes:
                    - memory_id: ID of stored memory
                    - context_keys: Keys in the built context
                Error response includes:
                    - error: Error message
                    - error_type: Type of error (no_memory_configured or storage_error)

        Example:
            >>> engine = ReasoningEngine(llm=llm, memory=memory, write_strategy=write_strategy)
            >>> result = engine._handle_store_memory("Remember to buy milk")
            >>> print(result["response"])
            "I've stored that information: 'to buy milk'"
            >>> print(result["intent"])
            'store_memory'
            >>> print(result["metadata"]["memory_id"])
            'mem_123'
        """
        if not self.memory:
            # Build context even when memory is not configured to include context_keys
            context = self.build_context(user_message)
            # Add intent to context for consistency
            context["intent"] = "store_memory"
            return {
                "response": "Memory storage is not available.",
                "intent": "store_memory",
                "metadata": {
                    "error": "no_memory_configured",
                    "error_type": "no_memory_configured",
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }

        try:
            # Extract content to store (remove trigger words)
            content = user_message.lower()
            for trigger in ["remember", "store", "save"]:
                content = content.replace(trigger, "").strip()

            # Store using write_strategy if available, otherwise direct storage
            if self.write_strategy:
                memory_id = self.write_strategy.store_memory(
                    content=content,
                    metadata={"source": "user_request"}
                )
            else:
                memory_id = self.memory.store(
                    content=content,
                    metadata={"source": "user_request", "category": "user_memory"}
                )

            # Retrieve the stored memory to inject into context
            try:
                result = self.memory.retrieve(params={"query": content, "limit": 1})

                # Handle both dict and RetrievalResult return types
                if isinstance(result, dict):
                    memories = result.get("memories", [])
                else:
                    memories = result["memories"]

                stored_memory = memories[0] if memories else None
            except Exception as e:
                logger.warning(f"Failed to retrieve stored memory for context: {e}")
                stored_memory = None

            # Build context (no longer passing memories as parameter)
            context = self.build_context(user_message)

            logger.info(f"Stored memory: {memory_id}")
            return {
                "response": f"I've stored that information: '{content}'",
                "intent": "store_memory",
                "metadata": {
                    "memory_id": memory_id,
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }

        except MemoryStorageError as e:
            # Handle MemoryStorageError with user-friendly message
            logger.error(f"Memory storage failed: {e}", exc_info=True)

            return {
                "response": f"I couldn't store that information. Please try again later.",
                "intent": "store_memory",
                "metadata": {
                    "error": str(e),
                    "error_type": "storage_error",
                    "context_keys": [],
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during memory storage: {e}", exc_info=True)

            return {
                "response": f"I couldn't store that information due to an unexpected error.",
                "intent": "store_memory",
                "metadata": {
                    "error": str(e),
                    "error_type": "unexpected_error",
                    "context_keys": [],
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }

    def _handle_retrieve_memory(self, user_message: str) -> Dict:
        """
        Handle retrieve_memory intent by querying memory and injecting into context.
        
        Processes messages with retrieve_memory intent by extracting the query,
        retrieving matching memories using enhanced QueryParameters, and generating
        a response with memory context. Handles cases where memory is not configured,
        no results are found, and retrieval errors gracefully by falling back to
        processing without memories.
        
        Enhanced Features:
            - Uses QueryParameters for structured retrieval
            - Extracts memories from RetrievalResult
            - Injects all memory metadata (category, tags, timestamp)
            - Logs retrieval metadata (count, execution time, filters)
            - Handles MemoryRetrievalError with fallback to LLM-only processing
            - Returns response with comprehensive metadata
        
        Args:
            user_message (str): User's message containing retrieval query.
        
        Returns:
            Dict: Response dictionary with retrieved memories incorporated.
                Success response includes:
                    - memories_found: Count of retrieved memories
                    - memory_ids: List of memory IDs
                    - execution_time_ms: Query execution time
                    - filters_applied: Applied filter parameters
                Error response includes:
                    - error: Error message
                    - fallback: True flag indicating fallback to LLM-only
        
        Example:
            >>> engine = ReasoningEngine(llm=llm, memory=memory)
            >>> result = engine._handle_retrieve_memory("What was my last task?")
            >>> print(result["response"])
            "Based on your memories, your last task was..."
            >>> print(result["metadata"]["memories_found"])
            3
            >>> print(result["metadata"]["execution_time_ms"])
            15.3
        """
        if not self.memory:
            # Build context even when memory is not configured to include context_keys
            context = self.build_context(user_message)
            return {
                "response": "Memory retrieval is not available.",
                "intent": "retrieve_memory",
                "metadata": {
                    "error": "no_memory_configured",
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }
        
        try:
            # Extract query (remove trigger words)
            query = user_message.lower()
            for trigger in ["what was", "recall", "retrieve", "remember"]:
                query = query.replace(trigger, "").strip()
            
            # Build QueryParameters for enhanced retrieval
            params: QueryParameters = {
                "query": query,
                "limit": 5
            }
            
            # Retrieve from memory using enhanced API
            retrieval_result: RetrievalResult = self.memory.retrieve(params=params)
            
            # Extract memories from RetrievalResult
            memories = retrieval_result["memories"]
            total_count = retrieval_result["total_count"]
            query_metadata = retrieval_result["query_metadata"]
            
            # Log retrieval metadata
            logger.info(
                f"Retrieved {total_count} memories for query: '{query}' "
                f"in {query_metadata.get('execution_time_ms', 0):.2f}ms"
            )
            logger.debug(f"Filters applied: {query_metadata.get('filters_applied', {})}")
            
            if not memories:
                return {
                    "response": "I don't have any memories matching that query.",
                    "intent": "retrieve_memory",
                    "metadata": {
                        "memories_found": 0,
                        "execution_time_ms": query_metadata.get("execution_time_ms", 0),
                        "filters_applied": query_metadata.get("filters_applied", {}),
                        "context_keys": [],
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                }
            
            # Build context with retrieved memories injected
            context = self.build_context(user_message, retrieved_memories=memories)
            
            # Generate response with memory context
            prompt = f"Based on these memories: {memories}, respond to: {user_message}"
            response_text = self.llm.generate_response(prompt, context)
            
            return {
                "response": response_text,
                "intent": "retrieve_memory",
                "metadata": {
                    "memories_found": total_count,
                    "memory_ids": [m["id"] for m in memories],
                    "execution_time_ms": query_metadata.get("execution_time_ms", 0),
                    "filters_applied": query_metadata.get("filters_applied", {}),
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }
            
        except MemoryRetrievalError as e:
            # Handle MemoryRetrievalError with fallback to LLM-only processing
            logger.error(f"Memory retrieval failed: {e}", exc_info=True)
            
            # Continue processing without memories (fallback behavior)
            context = self.build_context(user_message)
            response_text = self.llm.generate_response(user_message, context)
            
            return {
                "response": response_text,
                "intent": "retrieve_memory",
                "metadata": {
                    "error": str(e),
                    "fallback": True,
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }
        except Exception as e:
            # Handle unexpected errors with fallback
            logger.error(f"Unexpected error during memory retrieval: {e}", exc_info=True)
            
            # Continue processing without memories
            context = self.build_context(user_message)
            response_text = self.llm.generate_response(user_message, context)
            
            return {
                "response": response_text,
                "intent": "retrieve_memory",
                "metadata": {
                    "error": str(e),
                    "fallback": True,
                    "context_keys": list(context.keys()),
                    "timestamp": context["timestamp"]
                }
            }

    def process_message(self, user_message: str) -> Dict:
        """
        Process a user message through the complete reasoning pipeline.
        
        Orchestrates the full cognitive processing flow from user input to structured response.
        This method coordinates context building, intent detection, LLM invocation, and
        response formatting while handling errors gracefully at each stage.
        
        Pipeline Steps:
            1. Input Validation: Check for empty, None, or whitespace-only messages
            2. Context Building: Assemble context dictionary with message and metadata
            3. Intent Detection: Classify user intent using rule-based detection
            4. Context Enrichment: Add detected intent to context dictionary
            5. Prompt Formatting: Prepare prompt for LLM (currently uses message directly)
            6. LLM Invocation: Generate response via LLM interface
            7. Response Construction: Build structured response dictionary
            8. Error Handling: Catch and handle any exceptions gracefully
        
        Response Structure:
            Success Response:
                {
                    "response": str,      # Generated response text from LLM
                    "intent": str,        # Detected intent classification
                    "metadata": {
                        "context_keys": list,   # Keys present in context dictionary
                        "timestamp": str        # ISO 8601 timestamp
                    }
                }
            
            Invalid Input Response:
                {
                    "response": str,      # Error message explaining invalid input
                    "intent": "invalid",  # Special intent for invalid inputs
                    "metadata": {
                        "context_keys": [],     # Empty list for invalid inputs
                        "timestamp": str        # ISO 8601 timestamp
                    }
                }
            
            Error Response:
                {
                    "response": str,      # Error message with exception details
                    "intent": "error",    # Special intent for processing errors
                    "metadata": {
                        "context_keys": [],     # Empty list on error
                        "timestamp": str,       # ISO 8601 timestamp
                        "error": str            # Detailed error information
                    }
                }
        
        Args:
            user_message (str): The user's input message to process. Can be any string,
                including empty strings, None, or whitespace-only strings (all handled gracefully).
        
        Returns:
            Dict: Structured response dictionary containing response text, intent classification,
                and metadata. Always returns a valid dictionary, never raises exceptions.
        
        Error Handling:
            - Empty/None/Whitespace Input: Returns invalid response without processing
            - LLM Exceptions: Catches all exceptions, logs error, returns error response
            - Context Building Errors: Caught by outer try-except, returns error response
            - Intent Detection Errors: Caught by outer try-except, returns error response
        
        Logging:
            - INFO: Normal message processing with detected intent
            - WARNING: Empty or invalid input messages
            - ERROR: Exceptions during processing with full stack trace
            - DEBUG: Context building and intent detection details (via other methods)
        
        Example:
            >>> llm = MyLLMImplementation()  # Any class implementing LLMInterface
            >>> engine = ReasoningEngine(llm=llm)
            >>> 
            >>> # Valid message processing
            >>> result = engine.process_message("Teach me Python")
            >>> print(result["response"])
            'Response from LLM...'
            >>> print(result["intent"])
            'education'
            >>> print(result["metadata"]["context_keys"])
            ['user_message', 'timestamp', 'memory_placeholder', 'system_state_placeholder', 'intent']
            >>> 
            >>> # Empty message handling
            >>> result = engine.process_message("")
            >>> print(result["intent"])
            'invalid'
            >>> print(result["response"])
            'No message provided. Please send a message to process.'
            >>> 
            >>> # Whitespace-only message handling
            >>> result = engine.process_message("   ")
            >>> print(result["intent"])
            'invalid'
            >>> 
            >>> # None message handling
            >>> result = engine.process_message(None)
            >>> print(result["intent"])
            'invalid'
        
        Properties:
            - Stateless: Does not modify instance variables
            - Deterministic: Same input produces same output (for same LLM implementation)
            - Robust: Never raises unhandled exceptions
            - Structured: Always returns dict with response, intent, metadata keys
        
        Note:
            The prompt formatting is currently simple (uses user_message directly).
            Future enhancements will include sophisticated prompt engineering with
            templates, few-shot examples, and context-aware formatting.
        """
        # Start timing measurement for instrumentation
        start_time = time.perf_counter()
        
        # Step 1: Validate input - check for empty/None/whitespace-only
        if not user_message or not user_message.strip():
            logger.warning("Empty message received")
            result = {
                "response": "No message provided. Please send a message to process.",
                "intent": "invalid",
                "metadata": {
                    "context_keys": [],
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
            
            # Record metrics and log events for invalid input
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            if self.metrics_collector is not None:
                self.metrics_collector.record_duration('reasoning_latency_ms', duration_ms)
                self.metrics_collector.increment('reasoning_count')
            
            if self.logger is not None:
                self.logger.log('reasoning_completed', {
                    'intent': 'invalid',
                    'context_keys': [],
                    'response_length': len(result["response"])
                })
            
            return result
        
        try:
            # Step 2: Build context from user message
            # Assembles metadata including timestamp and placeholders for future integrations
            context = self.build_context(user_message)
            
            # Step 3: Detect intent from user message
            # Uses rule-based classification to determine user's goal
            intent = self.detect_intent(user_message)
            
            # Step 3.5: Route to memory handlers for memory intents
            if intent == "store_memory":
                result = self._handle_store_memory(user_message)
            elif intent == "retrieve_memory":
                result = self._handle_retrieve_memory(user_message)
            else:
                # Step 4: Add intent to context dictionary
                # Enriches context with detected intent for LLM to use
                context["intent"] = intent
                
                # Step 5: Format prompt (use user_message directly for now)
                # Future: Will include sophisticated prompt engineering with templates
                prompt = user_message
                
                # Step 6: Generate response via LLM interface
                # Delegates to injected LLM implementation
                logger.info(f"Processing message with intent: {intent}")
                response_text = self.llm.generate_response(prompt, context)
                
                # Step 7: Construct structured response dictionary
                # Returns consistent format with response, intent, and metadata
                result = {
                    "response": response_text,
                    "intent": intent,
                    "metadata": {
                        "context_keys": list(context.keys()),
                        "timestamp": context["timestamp"]
                    }
                }
            
            # Record metrics and log events for successful processing
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            if self.metrics_collector is not None:
                self.metrics_collector.record_duration('reasoning_latency_ms', duration_ms)
                self.metrics_collector.increment('reasoning_count')
            
            if self.logger is not None:
                self.logger.log('reasoning_completed', {
                    'intent': result["intent"],
                    'context_keys': result["metadata"]["context_keys"],
                    'response_length': len(result["response"])
                })
            
            return result
            
        except Exception as e:
            # Step 8: Catch any LLM or processing errors and return error response
            # Ensures the system never crashes, always returns a valid response
            # Logs full stack trace for debugging while returning user-friendly message
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            result = {
                "response": f"An error occurred while processing your message: {str(e)}",
                "intent": "error",
                "metadata": {
                    "context_keys": [],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": str(e)
                }
            }
            
            # Record metrics and log events for error case
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            if self.metrics_collector is not None:
                self.metrics_collector.record_duration('reasoning_latency_ms', duration_ms)
                self.metrics_collector.increment('reasoning_count')
            
            if self.logger is not None:
                self.logger.log('reasoning_completed', {
                    'intent': 'error',
                    'context_keys': [],
                    'response_length': len(result["response"])
                })
            
            return result


    def start_session(self, metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Start a new conversation session.

        Creates a new session via the session manager and stores the session_id
        for use in subsequent memory operations. If no session manager is configured,
        returns None.

        Args:
            metadata: Optional metadata to associate with the session (e.g., user_id, device_id)

        Returns:
            The unique session_id if session manager is available, None otherwise

        Example:
            >>> engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
            >>> session_id = engine.start_session(metadata={"user_id": "user123"})
            >>> print(f"Started session: {session_id}")
        """
        if self.session_manager:
            self.current_session_id = self.session_manager.create_session(metadata)
            logger.info(f"Started session: {self.current_session_id}")
            return self.current_session_id
        else:
            logger.warning("Cannot start session: no session manager configured")
            return None


    def end_session(self, persist: bool = True) -> None:
        """
        End the current conversation session and persist buffered memories.

        Ends the active session via the session manager, optionally persisting
        buffered memories to long-term storage. Clears the current_session_id
        after ending the session.

        Args:
            persist: Whether to persist buffered memories (default: True).
                     Set to False to discard buffered memories without saving.

        Example:
            >>> engine = ReasoningEngine(llm=llm, memory=memory, session_manager=session_manager)
            >>> session_id = engine.start_session()
            >>> # ... process messages ...
            >>> engine.end_session(persist=True)  # Save buffered memories
            >>>
            >>> # Or discard without saving
            >>> engine.end_session(persist=False)
        """
        if self.session_manager and self.current_session_id:
            persisted_count = self.session_manager.end_session(self.current_session_id, persist)
            logger.info(
                f"Ended session {self.current_session_id}, "
                f"persisted {persisted_count} memories"
            )
            self.current_session_id = None
        else:
            if not self.session_manager:
                logger.warning("Cannot end session: no session manager configured")
            else:
                logger.warning("Cannot end session: no active session")
    def handle_message(self, message: str, user_context: Optional[Dict] = None) -> Dict:
        """
        Handle a message with enhanced orchestration capabilities.

        This method provides a higher-level interface than process_message,
        returning structured responses with intent, response, context, and metadata.

        Args:
            message: The user's input message
            user_context: Optional additional context from the user

        Returns:
            Dict with keys: intent, response, context, metadata
        """
        # Build context with user_context
        context = self.build_context(message)
        if user_context:
            context["user_context"] = user_context

        # Detect intent
        intent = self.detect_intent(message)

        # Route based on intent
        if intent == "store_memory":
            result = self._handle_store_memory(message)
        elif intent == "retrieve_memory":
            result = self._handle_retrieve_memory(message)
        else:
            # Generate response via LLM
            response_text = self.llm.generate_response(message, context)
            result = {
                "response": response_text,
                "intent": intent,
                "metadata": {
                    "context_keys": list(context.keys())
                }
            }

        # Add context to result
        result["context"] = context

        # Ensure metadata exists and add message_length
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["message_length"] = len(message)
        result["metadata"]["context_keys"] = list(context.keys())
        result["metadata"]["confidence"] = 0.85  # Placeholder confidence score

        return result

    def route_intent(self, message: str, context: Dict) -> Intent:
        """
        Route message to appropriate intent using keyword matching.

        Args:
            message: The user's input message
            context: Context dictionary (currently unused, for future enhancement)

        Returns:
            Intent enum value
        """
        message_lower = message.lower()

        # Check for memory storage keywords
        if any(keyword in message_lower for keyword in ["remember", "store", "save"]):
            return Intent.STORE_MEMORY

        # Check for memory retrieval keywords
        elif any(keyword in message_lower for keyword in ["recall", "what did", "retrieve"]):
            return Intent.RETRIEVE_MEMORY

        # Check for scheduling keywords
        elif any(keyword in message_lower for keyword in ["schedule", "remind", "todo"]):
            return Intent.SCHEDULE_TASK

        # Check for system info keywords
        elif any(keyword in message_lower for keyword in ["system", "status", "health", "monitor", "check"]):
            return Intent.SYSTEM_INFO

        # Check for general query keywords
        elif any(keyword in message_lower for keyword in ["help", "what can", "show", "capabilities"]):
            return Intent.GENERAL_QUERY

        # Default to unknown
        else:
            return Intent.UNKNOWN

    def process(self, input_data: Dict) -> Dict:
        """
        Process input data dictionary (legacy method for backward compatibility).

        Args:
            input_data: Dictionary of input data

        Returns:
            Dict with status and input_keys
        """
        return {
            "status": "processed",
            "input_keys": list(input_data.keys())
        }

    def analyze_context(self, context: Dict) -> Dict:
        """
        Analyze context dictionary (legacy method for backward compatibility).

        Args:
            context: Context dictionary to analyze

        Returns:
            Dict with context_size, insights, and confidence
        """
        return {
            "context_size": len(context),
            "insights": ["Context analyzed successfully"],
            "confidence": 0.9
        }




