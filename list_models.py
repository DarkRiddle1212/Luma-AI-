#!/usr/bin/env python3
"""List available Gemini models"""

import google.generativeai as genai

# Configure API
api_key = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=api_key)

print("Available Gemini models:\n")

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            print(f"    Display name: {model.display_name}")
            print(f"    Description: {model.description[:80]}...")
            print()
except Exception as e:
    print(f"Error listing models: {e}")
