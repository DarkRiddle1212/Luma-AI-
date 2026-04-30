# Gemini Quick Start Guide

## 🚀 5-Minute Setup

### 1. Get API Key
Visit: https://makersuite.google.com/app/apikey

### 2. Configure Environment
```bash
# Copy example config
cp .env.example .env

# Edit .env and add your key
GEMINI_API_KEY=your-actual-api-key-here
LLM_PROVIDER=gemini
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 4. Verify Setup
```bash
python verify_gemini.py
```

## ✅ Quick Verification Checklist

- [ ] API key obtained from Google AI Studio
- [ ] `.env` file created with `GEMINI_API_KEY`
- [ ] `google-generativeai` package installed
- [ ] `verify_gemini.py` runs successfully
- [ ] All tests pass: `pytest tests/llm/ -v -m "not slow"`

## 🧪 Testing Commands

```bash
# Unit tests (no API key needed)
pytest tests/llm/test_gemini_provider.py -v

# Integration tests (no API key needed)
pytest tests/llm/test_llm_client_integration.py -v

# E2E tests (requires API key)
export GEMINI_API_KEY=your-key
pytest tests/llm/test_gemini_e2e_integration.py -v -m slow

# All tests
pytest tests/llm/ -v
```

## 🔧 Common Issues

### "GEMINI_API_KEY not set"
```bash
export GEMINI_API_KEY=your-key
# or add to .env file
```

### "authentication error: 401"
- Check API key is valid
- Regenerate key at https://makersuite.google.com/app/apikey

### "rate limit exceeded: 429"
- Wait a few minutes
- Check quota at https://console.cloud.google.com/

### "Module not found: google.generativeai"
```bash
pip install google-generativeai>=0.3.0
```

## 📊 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use |
| `GEMINI_TEMPERATURE` | `0.4` | Randomness (0.0-1.0) |
| `GEMINI_MAX_TOKENS` | `1024` | Max response length |
| `GEMINI_TIMEOUT` | `30.0` | Request timeout (seconds) |

## 🎯 Usage Example

```python
from luma.container import initialize_application

# Initialize with Gemini (reads from .env)
engine = initialize_application()

# Use the engine
response = engine.process_message("What is 2 + 2?")
print(response["response"])
```

## 📚 Documentation

- Full setup guide: `GEMINI_SETUP.md`
- Verification script: `python verify_gemini.py`
- API docs: https://ai.google.dev/docs

## 🆘 Support

If verification fails:
1. Check `GEMINI_SETUP.md` for detailed troubleshooting
2. Run `python verify_gemini.py` for diagnostic info
3. Check logs for error details
4. Verify API key at https://makersuite.google.com/app/apikey
