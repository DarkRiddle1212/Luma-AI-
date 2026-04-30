#!/usr/bin/env python3
"""Simple Gemini API test"""

import os
import google.generativeai as genai

# Configure API
api_key = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel('gemini-2.5-flash')

# Test generation
print("Testing Gemini API...")
print("Sending prompt: 'Say hello'")

try:
    response = model.generate_content("Say hello")
    print(f"\nSuccess! Response: {response.text}")
    print(f"\nToken usage:")
    if hasattr(response, 'usage_metadata'):
        print(f"  Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"  Response tokens: {response.usage_metadata.candidates_token_count}")
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e).__name__}")
