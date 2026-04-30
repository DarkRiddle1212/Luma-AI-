# Gemini Provider Setup & Verification Guide

## Prerequisites

1. **Get a Gemini API Key**
   - Visit: https://makersuite.google.com/app/apikey
   - Create or sign in to your Google account
   - Generate a new API key
   - Copy the key (you'll need it for configuration)

## Configuration Steps

### Step 1: Set Environment Variables

Create or update your `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and set these required variables:

```bash
# LLM Provider Configuration
LLM_PROVIDER=gemini

# Gemini API Key (REQUIRED)
GEMINI_API_KEY=your-actual-api-key-here

# Optional: Customize Gemini settings
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT=30.0
GEMINI_MAX_TOKENS=1024
GEMINI_TEMPERATURE=0.4
GEMINI_LOG_PROMPTS=false

# LLM Fallback Configuration
LLM_MAX_RETRIES=3
LLM_MAX_RESPONSE_CHARS=4000
LLM_FALLBACK_RESPONSE="I'm having trouble generating a response right now."
```

### Step 2: Install Dependencies

Ensure you have the required dependencies installed:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install just the Gemini dependency
pip install google-generativeai>=0.3.0
```

### Step 3: Verify Installation

Check that the Gemini package is installed:

```bash
python -c "import google.generativeai as genai; print(f'Gemini SDK version: {genai.__version__}')"
```

## Verification Tests

### Quick Verification (No API Key Required)

Run unit tests that don't require a real API key:

```bash
# Run all provider tests (uses mocks)
pytest tests/llm/ -v -m "not slow"

# Run specific Gemini provider tests
pytest tests/llm/test_gemini_provider.py -v
pytest tests/llm/test_gemini_provider_generate.py -v
```

Expected output: All tests should pass ✓

### Full Verification (Requires API Key)

Run end-to-end tests with the real Gemini API:

```bash
# Set your API key
export GEMINI_API_KEY=your-actual-api-key-here

# Run slow/e2e tests
pytest tests/llm/test_gemini_e2e_integration.py -v -m slow

# Or run all LLM tests including e2e
pytest tests/llm/ -v
```

Expected output:
- ✓ test_successful_generation
- ✓ test_authentication_with_valid_key
- ✓ test_timeout_handling
- ✓ test_multiple_sequential_requests
- ✓ test_response_normalization
- ✓ test_different_temperature_values
- ✓ test_different_max_tokens_values

### Interactive Verification Script

Run the verification script to test Gemini interactively:

```bash
python verify_gemini.py
```

This script will:
1. Check if GEMINI_API_KEY is set
2. Test basic generation
3. Test error handling
4. Display token usage
5. Verify response format

## Troubleshooting

### Issue: "GEMINI_API_KEY environment variable not set"

**Solution:**
```bash
# Set the environment variable
export GEMINI_API_KEY=your-actual-api-key-here

# Or add it to your .env file
echo "GEMINI_API_KEY=your-actual-api-key-here" >> .env
```

### Issue: "authentication error: 401"

**Causes:**
- Invalid API key
- API key not activated
- API key expired

**Solution:**
1. Verify your API key at https://makersuite.google.com/app/apikey
2. Generate a new API key if needed
3. Update your `.env` file with the new key

### Issue: "rate limit exceeded: 429"

**Causes:**
- Too many requests in a short time
- Free tier quota exceeded

**Solution:**
1. Wait a few minutes before retrying
2. Reduce request frequency
3. Check your quota at https://console.cloud.google.com/

### Issue: "timeout after 30.0s"

**Causes:**
- Network connectivity issues
- Gemini API is slow or overloaded
- Timeout setting too low

**Solution:**
1. Check your internet connection
2. Increase timeout in `.env`:
   ```bash
   GEMINI_TIMEOUT=60.0
   ```
3. Retry the request

### Issue: "Module 'google.generativeai' not found"

**Solution:**
```bash
pip install google-generativeai>=0.3.0
```

## Configuration Options

### Model Selection

Available models (as of 2024):
- `gemini-2.5-flash` (default) - Fast, efficient, good for most tasks
- `gemini-2.0-flash-exp` - Experimental, latest features
- `gemini-1.5-pro` - More capable, slower, higher cost

Set in `.env`:
```bash
GEMINI_MODEL=gemini-2.5-flash
```

### Temperature Control

Controls randomness in responses:
- `0.0` - Deterministic, focused
- `0.4` (default) - Balanced
- `1.0` - Creative, varied

Set in `.env`:
```bash
GEMINI_TEMPERATURE=0.4
```

### Token Limits

Maximum tokens to generate:
- `1024` (default) - Standard responses
- `2048` - Longer responses
- `4096` - Very long responses

Set in `.env`:
```bash
GEMINI_MAX_TOKENS=1024
```

### Timeout Settings

Request timeout in seconds:
- `30.0` (default) - Standard timeout
- `60.0` - Longer timeout for complex requests
- `10.0` - Quick timeout for simple requests

Set in `.env`:
```bash
GEMINI_TIMEOUT=30.0
```

## Integration Points

### Using Gemini in Your Code

The Gemini provider is automatically loaded when you initialize the application:

```python
from luma.container import initialize_application

# Initialize with Gemini (reads from environment)
engine = initialize_application()

# Use the reasoning engine
response = engine.process_message("Hello, how are you?")
print(response["response"])
```

### Manual Provider Creation

For advanced use cases, you can create the provider manually:

```python
from luma.core.llm.providers.gemini_provider import GeminiProvider
from luma.core.structured_logger import StructuredLogger

config = {
    "api_key": "your-api-key",
    "model": "gemini-2.5-flash",
    "timeout": 30.0,
    "max_tokens": 1024,
    "temperature": 0.4
}

provider = GeminiProvider(
    config=config,
    logger=StructuredLogger("gemini")
)

result = provider.generate(
    prompt="What is 2 + 2?",
    options={
        "temperature": 0.0,
        "max_tokens": 10,
        "request_id": "test-001"
    }
)

print(result["text"])
```

## Testing Strategy

### 1. Unit Tests (No API Key)
- Test configuration validation
- Test error mapping
- Test response normalization
- Test with mocked API responses

```bash
pytest tests/llm/test_gemini_provider.py -v
```

### 2. Integration Tests (No API Key)
- Test provider factory
- Test LLM client integration
- Test configuration loading

```bash
pytest tests/llm/test_llm_client_integration.py -v
pytest tests/llm/test_config_loading_integration.py -v
```

### 3. End-to-End Tests (Requires API Key)
- Test real API calls
- Test authentication
- Test timeout handling
- Test multiple requests

```bash
export GEMINI_API_KEY=your-key
pytest tests/llm/test_gemini_e2e_integration.py -v -m slow
```

## Success Criteria

✅ **Configuration**
- [ ] GEMINI_API_KEY is set in environment or .env
- [ ] google-generativeai package is installed
- [ ] LLM_PROVIDER=gemini in configuration

✅ **Unit Tests**
- [ ] All unit tests pass (pytest tests/llm/ -m "not slow")
- [ ] No import errors
- [ ] Configuration validation works

✅ **Integration Tests**
- [ ] Provider factory creates Gemini provider
- [ ] LLM client integrates with provider
- [ ] Configuration loads from environment

✅ **End-to-End Tests** (with API key)
- [ ] Successful generation with real API
- [ ] Authentication works
- [ ] Error handling works
- [ ] Response format is correct

✅ **Application Integration**
- [ ] ReasoningEngine initializes with Gemini
- [ ] Chat flow works end-to-end
- [ ] Responses are generated correctly

## Next Steps

Once Gemini is verified:

1. **Test in Application Context**
   ```bash
   # Start the API server
   luma-memory-server
   
   # Test the chat endpoint
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "message": "Hello!"}'
   ```

2. **Monitor Usage**
   - Check logs for provider events
   - Monitor token usage
   - Track error rates

3. **Optimize Configuration**
   - Adjust temperature for your use case
   - Tune max_tokens for response length
   - Set appropriate timeout values

## Support

- **Gemini API Documentation**: https://ai.google.dev/docs
- **API Key Management**: https://makersuite.google.com/app/apikey
- **Quota & Billing**: https://console.cloud.google.com/
- **Luma Issues**: File an issue in the repository
