# How to Get Your Gemini API Key - Step by Step

## 🎯 Overview
You need a free API key from Google to use Gemini. This takes about 2-3 minutes.

---

## 📝 Step-by-Step Instructions

### Step 1: Open Google AI Studio

**Click this link:** https://makersuite.google.com/app/apikey

Or copy and paste this URL into your browser:
```
https://makersuite.google.com/app/apikey
```

### Step 2: Sign In with Google

You'll see a Google sign-in page.

**Options:**
- Use your existing Google account (Gmail, etc.)
- Or create a new Google account if you don't have one

**Click:** "Sign in with Google"

### Step 3: Accept Terms (if prompted)

If this is your first time:
- You may see terms of service
- Read and accept them
- Click "Continue" or "Accept"

### Step 4: Create API Key

You'll see the "API Keys" page.

**Click the button:** "Create API Key"

You'll see options:
- **"Create API key in new project"** ← Choose this (recommended)
- Or select an existing Google Cloud project if you have one

**Click:** "Create API key in new project"

### Step 5: Copy Your API Key

A popup will appear with your API key!

**It looks like this:**
```
AIzaSyD...your-key-here...xyz123
```

**IMPORTANT:** 
- Click the **"Copy"** button (📋 icon)
- Or manually select and copy the entire key
- Keep this window open until you've saved it!

### Step 6: Save Your API Key Securely

**Option A: Save to .env file (Recommended)**

1. Open your project folder in your editor
2. Find the `.env` file (or create it if it doesn't exist)
3. Add this line:
   ```bash
   GEMINI_API_KEY=AIzaSyD...paste-your-key-here...xyz123
   ```
4. Save the file

**Option B: Save to a text file temporarily**

1. Open Notepad or any text editor
2. Paste your API key
3. Save as `gemini_key.txt` in a safe location
4. You can add it to `.env` later

**⚠️ SECURITY WARNING:**
- Never share your API key publicly
- Never commit it to Git (`.env` is in `.gitignore`)
- Never post it in forums or chat

---

## ✅ Verification

After saving your key, verify it works:

### Quick Test (Windows PowerShell)

```powershell
# Set the environment variable temporarily
$env:GEMINI_API_KEY = "your-key-here"

# Run verification
python verify_gemini.py
```

### Quick Test (Command Prompt)

```cmd
# Set the environment variable temporarily
set GEMINI_API_KEY=your-key-here

# Run verification
python verify_gemini.py
```

### Expected Output

```
======================================================================
  Step 1: Environment Check
======================================================================

[OK] GEMINI_API_KEY is set: ****xyz123
[INFO] LLM_PROVIDER: gemini
...
```

---

## 🎨 Visual Guide

```
┌─────────────────────────────────────────────────────────────┐
│  1. Visit: https://makersuite.google.com/app/apikey        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Sign in with your Google account                        │
│     [Sign in with Google]                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Click "Create API Key"                                  │
│     [+ Create API Key]                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Choose "Create API key in new project"                  │
│     ○ Create API key in new project     [Recommended]       │
│     ○ Create API key in existing project                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Copy your API key!                                      │
│                                                             │
│     Your API key:                                           │
│     AIzaSyD...your-key-here...xyz123    [📋 Copy]          │
│                                                             │
│     ⚠️  Keep this key secure!                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Add to .env file:                                       │
│                                                             │
│     GEMINI_API_KEY=AIzaSyD...your-key...xyz123             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Setting Up Your .env File

### If .env doesn't exist:

```bash
# Copy the example file
cp .env.example .env
```

### Edit .env file:

Open `.env` in your text editor and find this section:

```bash
# ============================================================================
# LLM Provider Configuration
# ============================================================================

# LLM provider selection (gemini, mock, or future providers)
LLM_PROVIDER=gemini

# Gemini API configuration (required when LLM_PROVIDER=gemini)
GEMINI_API_KEY=your-gemini-api-key-here    ← REPLACE THIS LINE
```

**Replace** `your-gemini-api-key-here` with your actual key:

```bash
GEMINI_API_KEY=AIzaSyD...your-actual-key...xyz123
```

**Save the file!**

---

## 🚀 Next Steps

After getting your API key:

### 1. Verify it works
```bash
python verify_gemini.py
```

### 2. Run tests
```bash
# Unit tests (no API calls)
pytest tests/llm/test_gemini_provider.py -v

# E2E tests (makes real API calls)
pytest tests/llm/test_gemini_e2e_integration.py -v -m slow
```

### 3. Try it in your application
```python
from luma.container import initialize_application

# Initialize with Gemini
engine = initialize_application()

# Test it
response = engine.process_message("What is 2 + 2?")
print(response["response"])
```

---

## ❓ Troubleshooting

### "I can't find the API Keys page"

**Solution:**
1. Make sure you're signed in to Google
2. Try this direct link: https://aistudio.google.com/app/apikey
3. Or go to: https://ai.google.dev/ → Click "Get API Key"

### "Create API Key button is disabled"

**Possible causes:**
- You need to accept terms of service first
- Your Google account needs verification
- Try refreshing the page

**Solution:**
1. Look for any banners or prompts to accept terms
2. Verify your Google account (check email)
3. Try a different browser

### "I lost my API key"

**Solution:**
1. Go back to https://makersuite.google.com/app/apikey
2. You'll see your existing keys listed
3. Click "Show key" to reveal it
4. Or create a new key if needed

### "API key doesn't work"

**Check:**
1. Did you copy the entire key? (starts with `AIza...`)
2. Are there any extra spaces before/after the key?
3. Did you save the `.env` file?
4. Try creating a new API key

---

## 💰 Pricing & Limits

### Free Tier (Gemini API)

**Generous free quota:**
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day

**This is plenty for:**
- Development and testing
- Personal projects
- Small applications

### Paid Tier

If you need more:
- Visit: https://console.cloud.google.com/
- Enable billing on your project
- Much higher limits available

**For most users, the free tier is sufficient!**

---

## 🔒 Security Best Practices

### ✅ DO:
- Store API key in `.env` file
- Add `.env` to `.gitignore` (already done)
- Use environment variables
- Keep backups in a secure location

### ❌ DON'T:
- Commit API keys to Git
- Share keys publicly
- Post keys in forums/chat
- Hardcode keys in source code
- Email keys in plain text

---

## 📞 Need Help?

### Google AI Studio Support
- Documentation: https://ai.google.dev/docs
- Community: https://discuss.ai.google.dev/

### Luma Project Support
- Check: `GEMINI_SETUP.md` for detailed troubleshooting
- Run: `python verify_gemini.py` for diagnostics
- Review: `GEMINI_VERIFICATION_SUMMARY.md` for complete guide

---

## ✨ Summary

1. ✅ Visit https://makersuite.google.com/app/apikey
2. ✅ Sign in with Google
3. ✅ Click "Create API Key"
4. ✅ Copy your key
5. ✅ Add to `.env` file: `GEMINI_API_KEY=your-key`
6. ✅ Run `python verify_gemini.py`

**You're done!** 🎉
