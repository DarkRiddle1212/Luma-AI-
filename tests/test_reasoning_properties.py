"""
Property-Based Tests for Reasoning Engine Brain

This module implements property-based tests using Hypothesis to verify
universal correctness properties for the LLM interface and reasoning engine.

Feature: reasoning-engine-brain
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock
from luma.core.llm_interface import LLMInterface, StubLLM
from luma.core.reasoning import ReasoningEngine
from luma.core.memory_interface import MemoryInterface


# ============================================================================
# 13.1 Property Test: LLM Interface Contract Compliance (Property 1)
# ============================================================================

# Feature: reasoning-engine-brain, Property 1: LLM Interface Contract Compliance
@given(
    prompt=st.text(min_size=0, max_size=5),
    context=st.dictionaries(
        keys=st.text(min_size=1, max_size=5),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.booleans(),
            st.lists(st.text(max_size=5), max_size=5),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=5),
                values=st.text(max_size=5),
                max_size=3
            )
        ),
        min_size=0,
        max_size=10
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_llm_interface_contract_compliance_property(prompt, context):
    """
    Property: For any implementation of LLMInterface, calling generate_response 
    with valid prompt and context should return a string without raising unhandled exceptions.
    
    **Validates: Requirements 1.2, 1.3**
    
    This test verifies that:
    1. generate_response accepts any string prompt
    2. generate_response accepts any dictionary context
    3. generate_response returns a string
    4. generate_response does not raise unhandled exceptions
    """
    # Test with StubLLM implementation
    llm = StubLLM()
    
    # Call generate_response - should not raise exception
    response = llm.generate_response(prompt, context)
    
    # Verify response is a string
    assert isinstance(response, str), \
        f"generate_response must return a string, got {type(response).__name__}"
    
    # Verify response is not None
    assert response is not None, "generate_response must not return None"
    
    # Verify response has content (StubLLM always returns formatted response)
    assert len(response) > 0, "generate_response must return non-empty string"


# ============================================================================
# 13.2 Property Test: Structured Response Consistency (Property 2)
# ============================================================================

# Feature: reasoning-engine-brain, Property 2: Structured Response Consistency
@given(st.text(min_size=1, max_size=5))
@settings(max_examples=10)
@pytest.mark.property_test
def test_structured_response_consistency_property(user_message):
    """
    Property: For any valid user message, process_message should return 
    a dictionary containing "response", "intent", and "metadata" keys.
    
    **Validates: Requirements 5.5, 5.6**
    
    This test verifies that:
    1. process_message returns a dictionary
    2. The dictionary contains "response" key with string value
    3. The dictionary contains "intent" key with string value
    4. The dictionary contains "metadata" key with dictionary value
    5. The structure is consistent across all valid inputs
    """
    # Initialize reasoning engine with default StubLLM
    engine = ReasoningEngine()
    
    # Process the message
    result = engine.process_message(user_message)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), \
        f"process_message must return a dict, got {type(result).__name__}"
    
    # Verify required keys are present
    assert "response" in result, \
        "process_message result must contain 'response' key"
    assert "intent" in result, \
        "process_message result must contain 'intent' key"
    assert "metadata" in result, \
        "process_message result must contain 'metadata' key"
    
    # Verify value types
    assert isinstance(result["response"], str), \
        f"'response' must be a string, got {type(result['response']).__name__}"
    assert isinstance(result["intent"], str), \
        f"'intent' must be a string, got {type(result['intent']).__name__}"
    assert isinstance(result["metadata"], dict), \
        f"'metadata' must be a dict, got {type(result['metadata']).__name__}"
    
    # Verify response is not None or empty
    assert result["response"] is not None, "'response' must not be None"
    assert len(result["response"]) > 0, "'response' must not be empty"
    
    # Verify intent is not None or empty
    assert result["intent"] is not None, "'intent' must not be None"
    assert len(result["intent"]) > 0, "'intent' must not be empty"


# ============================================================================
# 13.3 Property Test: Intent Detection Determinism (Property 3)
# ============================================================================

# Feature: reasoning-engine-brain, Property 3: Intent Detection Determinism
@given(st.text(min_size=0, max_size=5))
@settings(max_examples=10)
@pytest.mark.property_test
def test_intent_detection_determinism_property(user_message):
    """
    Property: For any given user message, detect_intent should always 
    return the same intent classification when called multiple times.
    
    **Validates: Requirements 4.2, 4.3**
    
    This test verifies that:
    1. detect_intent is deterministic (same input → same output)
    2. detect_intent returns consistent results across multiple calls
    3. detect_intent always returns a valid string intent
    4. The classification is stable and reproducible
    
    This property is critical for:
    - Predictable system behavior
    - Debugging and testing
    - Caching and optimization
    - User experience consistency
    """
    # Initialize reasoning engine
    engine = ReasoningEngine()
    
    # Call detect_intent multiple times with the same message
    intent1 = engine.detect_intent(user_message)
    intent2 = engine.detect_intent(user_message)
    intent3 = engine.detect_intent(user_message)
    
    # Verify all calls return the same intent
    assert intent1 == intent2, \
        f"detect_intent must be deterministic: first call returned '{intent1}', second call returned '{intent2}'"
    assert intent2 == intent3, \
        f"detect_intent must be deterministic: second call returned '{intent2}', third call returned '{intent3}'"
    assert intent1 == intent3, \
        f"detect_intent must be deterministic: first call returned '{intent1}', third call returned '{intent3}'"
    
    # Verify the intent is a string
    assert isinstance(intent1, str), \
        f"detect_intent must return a string, got {type(intent1).__name__}"
    
    # Verify the intent is not None
    assert intent1 is not None, "detect_intent must not return None"
    
    # Verify the intent is not empty
    assert len(intent1) > 0, "detect_intent must return non-empty string"
    
    # Verify the intent is one of the valid intents
    valid_intents = {"store_memory", "retrieve_memory", "education", "scheduling", "general"}
    assert intent1 in valid_intents, \
        f"detect_intent must return a valid intent, got '{intent1}'. Valid intents: {valid_intents}"


# ============================================================================
# 13.4 Property Test: Error Handling Robustness (Property 4)
# ============================================================================

# Feature: reasoning-engine-brain, Property 4: Error Handling Robustness
@given(
    st.one_of(
        st.just(""),                           # Empty string
        st.just(None),                         # None value
        st.sampled_from([" ", "  ", "   ", "\t", "\n", "\t\n", "  \t  ", "\n\n", "   \t\n   "])  # Whitespace-only
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_error_handling_robustness_property(invalid_input):
    """
    Property: For any empty, None, or whitespace-only input, process_message 
    should return a valid response dictionary without raising exceptions.
    
    **Validates: Requirements 8.1, 8.4**
    
    This test verifies that:
    1. process_message handles empty strings gracefully
    2. process_message handles None values gracefully
    3. process_message handles whitespace-only strings gracefully
    4. No unhandled exceptions are raised for invalid inputs
    5. A valid response dictionary is always returned
    6. The response indicates the error appropriately
    
    This property is critical for:
    - System stability and robustness
    - Graceful degradation under invalid inputs
    - User experience (no crashes)
    - API reliability
    """
    # Initialize reasoning engine
    engine = ReasoningEngine()
    
    # Process the invalid input - should not raise exception
    result = engine.process_message(invalid_input)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), \
        f"process_message must return a dict even for invalid input, got {type(result).__name__}"
    
    # Verify required keys are present
    assert "response" in result, \
        "process_message result must contain 'response' key even for invalid input"
    assert "intent" in result, \
        "process_message result must contain 'intent' key even for invalid input"
    assert "metadata" in result, \
        "process_message result must contain 'metadata' key even for invalid input"
    
    # Verify value types
    assert isinstance(result["response"], str), \
        f"'response' must be a string, got {type(result['response']).__name__}"
    assert isinstance(result["intent"], str), \
        f"'intent' must be a string, got {type(result['intent']).__name__}"
    assert isinstance(result["metadata"], dict), \
        f"'metadata' must be a dict, got {type(result['metadata']).__name__}"
    
    # Verify response is not None or empty
    assert result["response"] is not None, "'response' must not be None"
    assert len(result["response"]) > 0, "'response' must not be empty"
    
    # Verify intent indicates invalid input
    assert result["intent"] == "invalid", \
        f"Intent for invalid input should be 'invalid', got '{result['intent']}'"
    
    # Verify metadata is present and valid
    assert "timestamp" in result["metadata"], \
        "metadata must contain 'timestamp' even for invalid input"
    assert "context_keys" in result["metadata"], \
        "metadata must contain 'context_keys' even for invalid input"


# ============================================================================
# 13.5 Property Test: Context Building Completeness (Property 5)
# ============================================================================

# Feature: reasoning-engine-brain, Property 5: Context Building Completeness
@given(st.text(min_size=0, max_size=5))
@settings(max_examples=10)
@pytest.mark.property_test
def test_context_building_completeness_property(user_message):
    """
    Property: For any user message, build_context should return a dictionary 
    containing at minimum: user_message, timestamp, memory_placeholder, 
    and system_state_placeholder keys.
    
    **Validates: Requirements 3.2, 3.3, 3.4**
    
    This test verifies that:
    1. build_context returns a dictionary
    2. The dictionary contains "user_message" key
    3. The dictionary contains "timestamp" key
    4. The dictionary contains "memory_placeholder" key
    5. The dictionary contains "system_state_placeholder" key
    6. All required keys are present for any input message
    7. The values have the correct types
    
    This property is critical for:
    - Ensuring consistent context structure across all messages
    - Guaranteeing downstream components can rely on required keys
    - Supporting future integration with memory and monitoring modules
    - Maintaining API contract stability
    """
    # Initialize reasoning engine
    engine = ReasoningEngine()
    
    # Build context from the user message
    context = engine.build_context(user_message)
    
    # Verify context is a dictionary
    assert isinstance(context, dict), \
        f"build_context must return a dict, got {type(context).__name__}"
    
    # Verify all required keys are present
    required_keys = {"user_message", "timestamp", "memory_placeholder", "system_state_placeholder"}
    actual_keys = set(context.keys())
    
    missing_keys = required_keys - actual_keys
    assert not missing_keys, \
        f"build_context must include all required keys. Missing: {missing_keys}"
    
    # Verify user_message key and value
    assert "user_message" in context, \
        "build_context result must contain 'user_message' key"
    assert context["user_message"] == user_message, \
        f"'user_message' must match input. Expected: '{user_message}', got: '{context['user_message']}'"
    
    # Verify timestamp key and value type
    assert "timestamp" in context, \
        "build_context result must contain 'timestamp' key"
    assert isinstance(context["timestamp"], str), \
        f"'timestamp' must be a string, got {type(context['timestamp']).__name__}"
    assert len(context["timestamp"]) > 0, \
        "'timestamp' must not be empty"
    
    # Verify timestamp is valid ISO format (basic check)
    # ISO format should contain 'T' separator and be reasonably long
    assert "T" in context["timestamp"], \
        f"'timestamp' should be in ISO format (contain 'T'), got: '{context['timestamp']}'"
    
    # Verify memory_placeholder key and value type
    assert "memory_placeholder" in context, \
        "build_context result must contain 'memory_placeholder' key"
    assert isinstance(context["memory_placeholder"], list), \
        f"'memory_placeholder' must be a list, got {type(context['memory_placeholder']).__name__}"
    
    # Verify system_state_placeholder key and value type
    assert "system_state_placeholder" in context, \
        "build_context result must contain 'system_state_placeholder' key"
    assert isinstance(context["system_state_placeholder"], dict), \
        f"'system_state_placeholder' must be a dict, got {type(context['system_state_placeholder']).__name__}"


# ============================================================================
# 3.10 Property Test: Backward Compatibility (Property 10)
# ============================================================================

# Feature: reasoning-memory-integration, Property 10: Backward Compatibility
@given(
    intent_type=st.sampled_from(["education", "scheduling", "general"]),
    message_content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_backward_compatibility_property(intent_type, message_content):
    """
    Property: For any message with non-memory intents (education, scheduling, general),
    the system should process the message using the existing flow without attempting
    memory operations.
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. Non-memory intents are processed correctly
    2. No memory.store() calls are made
    3. No memory.retrieve() calls are made
    4. The existing processing flow is unchanged
    5. Response structure remains consistent
    6. Memory integration doesn't break existing functionality
    """
    # Create trigger words for each intent type
    intent_triggers = {
        "education": ["teach", "learn", "explain"],
        "scheduling": ["schedule", "remind"],
        "general": ["hello", "hi", "what"]  # General triggers or neutral words
    }
    
    # Pick a trigger for the intent type
    import random
    trigger = random.choice(intent_triggers[intent_type])
    
    # Construct message with trigger + content
    message = f"{trigger} {message_content}"
    
    # Create mock memory that tracks calls
    mock_memory = Mock(spec=MemoryInterface)
    mock_memory.store = Mock(return_value="should_not_be_called")
    mock_memory.retrieve = Mock(return_value=[])
    
    # Create LLM
    llm = StubLLM()
    
    # Create ReasoningEngine with memory
    engine = ReasoningEngine(llm=llm, memory=mock_memory)
    
    # Process the message
    result = engine.process_message(message)
    
    # Verify intent was detected correctly (not store_memory or retrieve_memory)
    assert result["intent"] in ["education", "scheduling", "general"], \
        f"Intent should be non-memory intent, got '{result['intent']}'"
    
    # Verify the detected intent matches the expected type
    # (Note: general intent is default, so it might match even without trigger)
    if intent_type != "general":
        assert result["intent"] == intent_type, \
            f"Intent should be '{intent_type}', got '{result['intent']}'"
    
    # CRITICAL: Verify no memory operations were attempted
    assert not mock_memory.store.called, \
        f"memory.store() should NOT be called for {intent_type} intent, but it was called"
    
    assert not mock_memory.retrieve.called, \
        f"memory.retrieve() should NOT be called for {intent_type} intent, but it was called"
    
    # Verify response structure is consistent with existing flow
    assert isinstance(result, dict), \
        f"Result must be a dict, got {type(result)}"
    
    assert "response" in result, \
        "Result must contain 'response' key"
    assert "intent" in result, \
        "Result must contain 'intent' key"
    assert "metadata" in result, \
        "Result must contain 'metadata' key"
    
    # Verify response is valid
    assert isinstance(result["response"], str), \
        f"Response must be a string, got {type(result['response'])}"
    assert len(result["response"]) > 0, \
        "Response should not be empty"
    
    # Verify metadata structure
    assert isinstance(result["metadata"], dict), \
        f"Metadata must be a dict, got {type(result['metadata'])}"
    
    # Verify standard metadata fields are present (from existing flow)
    assert "context_keys" in result["metadata"], \
        "Metadata should contain 'context_keys' from existing flow"
    assert "timestamp" in result["metadata"], \
        "Metadata should contain 'timestamp' from existing flow"
    
    # Verify no memory-specific metadata is present
    assert "memory_id" not in result["metadata"], \
        "Metadata should NOT contain 'memory_id' for non-memory intents"
    assert "memories_found" not in result["metadata"], \
        "Metadata should NOT contain 'memories_found' for non-memory intents"
    assert "memory_ids" not in result["metadata"], \
        "Metadata should NOT contain 'memory_ids' for non-memory intents"


# Feature: reasoning-memory-integration, Property 10: Backward Compatibility
@given(
    message=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'P')),
        min_size=1,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_backward_compatibility_without_memory_property(message):
    """
    Property: For any message, the ReasoningEngine should work correctly
    even when no memory implementation is provided (memory=None).
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. ReasoningEngine works without memory dependency
    2. Non-memory intents process normally
    3. Memory intents handle missing memory gracefully
    4. No crashes occur when memory is None
    5. Backward compatibility with systems that don't use memory
    """
    # Create LLM
    llm = StubLLM()
    
    # Create ReasoningEngine WITHOUT memory (backward compatibility)
    engine = ReasoningEngine(llm=llm, memory=None)
    
    # Process the message - should not crash
    try:
        result = engine.process_message(message)
    except Exception as e:
        pytest.fail(
            f"ReasoningEngine should work without memory, "
            f"but raised {type(e).__name__}: {e}"
        )
    
    # Verify result is valid
    assert isinstance(result, dict), \
        f"Result must be a dict, got {type(result)}"
    
    assert "response" in result, \
        "Result must contain 'response' key"
    assert "intent" in result, \
        "Result must contain 'intent' key"
    assert "metadata" in result, \
        "Result must contain 'metadata' key"
    
    # Verify response is not empty
    assert isinstance(result["response"], str), \
        f"Response must be a string, got {type(result['response'])}"
    assert len(result["response"]) > 0, \
        "Response should not be empty"
    
    # If intent is memory-related, verify graceful handling
    if result["intent"] in ["store_memory", "retrieve_memory"]:
        # Should return error message about memory not being available
        assert "not available" in result["response"].lower() or \
               "no memory" in result["response"].lower() or \
               "couldn't" in result["response"].lower(), \
            f"Memory intent without memory should return informative error, got: '{result['response']}'"


# Feature: reasoning-memory-integration, Property 10: Backward Compatibility
@given(
    intent_type=st.sampled_from(["education", "scheduling", "general"]),
    message_content=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
        min_size=5,
        max_size=200
    )
)
@settings(max_examples=10)
@pytest.mark.property_test
def test_existing_flow_unchanged_property(intent_type, message_content):
    """
    Property: For any non-memory intent, the processing flow should be identical
    whether memory is provided or not.
    
    **Validates: Requirements 9.3**
    
    This test verifies that:
    1. Non-memory intents produce same results with or without memory
    2. Memory integration is truly optional
    3. Existing functionality is not affected by memory addition
    4. Response structure is consistent
    """
    # Create trigger words for each intent type
    intent_triggers = {
        "education": ["teach", "learn", "explain"],
        "scheduling": ["schedule", "remind"],
        "general": ["hello", "hi", "what"]
    }
    
    # Pick a trigger for the intent type
    import random
    trigger = random.choice(intent_triggers[intent_type])
    
    # Construct message with trigger + content
    message = f"{trigger} {message_content}"
    
    # Create LLM
    llm1 = StubLLM()
    llm2 = StubLLM()
    
    # Create two engines: one with memory, one without
    mock_memory = Mock(spec=MemoryInterface)
    mock_memory.store = Mock(return_value="mem_123")
    mock_memory.retrieve = Mock(return_value=[])
    
    engine_with_memory = ReasoningEngine(llm=llm1, memory=mock_memory)
    engine_without_memory = ReasoningEngine(llm=llm2, memory=None)
    
    # Process the same message with both engines
    result_with_memory = engine_with_memory.process_message(message)
    result_without_memory = engine_without_memory.process_message(message)
    
    # Verify both detected the same intent
    assert result_with_memory["intent"] == result_without_memory["intent"], \
        f"Intent detection should be same with/without memory. " \
        f"With memory: '{result_with_memory['intent']}', " \
        f"Without memory: '{result_without_memory['intent']}'"
    
    # Verify intent is non-memory
    assert result_with_memory["intent"] in ["education", "scheduling", "general"], \
        f"Intent should be non-memory, got '{result_with_memory['intent']}'"
    
    # Verify no memory operations were called
    assert not mock_memory.store.called, \
        "memory.store() should not be called for non-memory intents"
    assert not mock_memory.retrieve.called, \
        "memory.retrieve() should not be called for non-memory intents"
    
    # Verify both responses have the same structure
    assert set(result_with_memory.keys()) == set(result_without_memory.keys()), \
        "Response structure should be identical with/without memory"
    
    # Verify both responses are valid
    for result in [result_with_memory, result_without_memory]:
        assert isinstance(result["response"], str), \
            "Response must be a string"
        assert len(result["response"]) > 0, \
            "Response should not be empty"
        assert isinstance(result["metadata"], dict), \
            "Metadata must be a dict"
        assert "context_keys" in result["metadata"], \
            "Metadata should contain context_keys"
        assert "timestamp" in result["metadata"], \
            "Metadata should contain timestamp"
