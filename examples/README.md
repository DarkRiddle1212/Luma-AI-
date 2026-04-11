# Reasoning Engine Examples

This directory contains comprehensive examples demonstrating Luma's Reasoning Engine Brain functionality.

## Running the Examples

### Option 1: Using PYTHONPATH (Recommended for Development)

```bash
# On Linux/Mac
PYTHONPATH=. python examples/reasoning_example.py

# On Windows (PowerShell)
$env:PYTHONPATH = "."; python examples/reasoning_example.py

# On Windows (CMD)
set PYTHONPATH=.
python examples/reasoning_example.py
```

### Option 2: Install Package in Development Mode

```bash
pip install -e .
python examples/reasoning_example.py
```

## What's Demonstrated

The `reasoning_example.py` file includes 8 comprehensive examples:

1. **Basic ReasoningEngine Usage** - Simple message processing with default StubLLM
2. **Different Intent Classifications** - Demonstrates all 5 intent types (store_memory, retrieve_memory, education, scheduling, general)
3. **Error Handling** - Shows how the engine handles empty strings, whitespace, and None values
4. **Custom LLM Implementation** - Example of creating your own LLM by implementing LLMInterface
5. **Comparing StubLLM vs Custom LLM** - Side-by-side comparison of different LLM implementations
6. **Context Building and Metadata** - Explains the context structure and metadata fields
7. **Intent Detection Edge Cases** - Case-insensitivity, multiple keywords, and partial matches
8. **Simulating LLM Errors** - Demonstrates graceful error handling when LLM fails

## Example Output

Each example prints structured output showing:
- The input message
- Detected intent
- Generated response
- Metadata (context keys, timestamp)

## Next Steps

After running these examples, you can:
- Replace `StubLLM` with a real LLM implementation (OpenAI, Ollama, etc.)
- Integrate with the memory module for context enrichment
- Add ML-based intent classification
- Connect to FastAPI endpoints for API access

## Architecture

The Reasoning Engine uses:
- **Dependency Injection**: LLM implementations are injected via constructor
- **Stateless Design**: No instance variables modified during processing
- **Interface-Based Programming**: Works with any LLMInterface implementation
- **Structured Responses**: Always returns dict with response, intent, and metadata

## Related Files

- `luma/core/reasoning.py` - ReasoningEngine implementation
- `luma/core/llm_interface.py` - LLMInterface and StubLLM implementation
- `tests/test_reasoning.py` - Unit tests for reasoning engine
- `.kiro/specs/reasoning-engine-brain/` - Complete specification documents
