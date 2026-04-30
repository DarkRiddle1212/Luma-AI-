#!/usr/bin/env python3
"""
Gemini Provider Verification Script

This script verifies that the Gemini provider is properly configured
and working correctly. It performs a series of checks and tests to
ensure the integration is functional.

Usage:
    python verify_gemini.py

Requirements:
    - GEMINI_API_KEY environment variable must be set
    - google-generativeai package must be installed
"""

import os
import sys
from typing import Dict, Optional


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"[OK] {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"[WARN] {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"[INFO] {text}")


def check_environment() -> bool:
    """Check if required environment variables are set."""
    print_header("Step 1: Environment Check")
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print_error("GEMINI_API_KEY environment variable is not set")
        print_info("Set it with: export GEMINI_API_KEY=your-api-key")
        print_info("Or add it to your .env file")
        return False
    
    # Mask the API key for display
    masked_key = "*" * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else "****"
    print_success(f"GEMINI_API_KEY is set: {masked_key}")
    
    # Check optional environment variables
    provider = os.getenv("LLM_PROVIDER", "not set")
    model = os.getenv("GEMINI_MODEL", "not set (will use default)")
    timeout = os.getenv("GEMINI_TIMEOUT", "not set (will use default)")
    
    print_info(f"LLM_PROVIDER: {provider}")
    print_info(f"GEMINI_MODEL: {model}")
    print_info(f"GEMINI_TIMEOUT: {timeout}")
    
    return True


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    print_header("Step 2: Dependency Check")
    
    try:
        import google.generativeai as genai
        print_success(f"google-generativeai is installed (version: {genai.__version__})")
    except ImportError:
        print_error("google-generativeai package is not installed")
        print_info("Install it with: pip install google-generativeai>=0.3.0")
        return False
    
    try:
        from luma.core.llm.providers.gemini_provider import GeminiProvider
        print_success("GeminiProvider is available")
    except ImportError as e:
        print_error(f"Failed to import GeminiProvider: {e}")
        return False
    
    try:
        from luma.core.structured_logger import StructuredLogger
        print_success("StructuredLogger is available")
    except ImportError as e:
        print_error(f"Failed to import StructuredLogger: {e}")
        return False
    
    return True


def test_provider_initialization() -> Optional[object]:
    """Test that the provider can be initialized."""
    print_header("Step 3: Provider Initialization")
    
    try:
        from luma.core.llm.providers.gemini_provider import GeminiProvider
        from luma.core.structured_logger import StructuredLogger
        
        config = {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "timeout": float(os.getenv("GEMINI_TIMEOUT", "30.0")),
            "max_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
            "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.4")),
            "log_prompts": False
        }
        
        logger = StructuredLogger("gemini_verify")
        provider = GeminiProvider(config=config, logger=logger)
        
        print_success("GeminiProvider initialized successfully")
        print_info(f"Model: {config['model']}")
        print_info(f"Timeout: {config['timeout']}s")
        print_info(f"Max tokens: {config['max_tokens']}")
        print_info(f"Temperature: {config['temperature']}")
        
        return provider
        
    except Exception as e:
        print_error(f"Failed to initialize provider: {e}")
        return None


def test_basic_generation(provider: object) -> bool:
    """Test basic text generation."""
    print_header("Step 4: Basic Generation Test")
    
    try:
        prompt = "What is 2 + 2? Answer with just the number."
        options = {
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "temperature": 0.0,  # Deterministic
            "max_tokens": 10,
            "request_id": "verify-001"
        }
        
        print_info(f"Sending prompt: '{prompt}'")
        print_info("Waiting for response...")
        
        result = provider.generate(prompt, options)
        
        print_success("Generation successful!")
        print_info(f"Response: {result['text']}")
        print_info(f"Model: {result['model']}")
        print_info(f"Provider: {result['provider']}")
        print_info(f"Prompt tokens: {result['prompt_tokens']}")
        print_info(f"Completion tokens: {result['completion_tokens']}")
        
        # Verify response contains expected answer
        if "4" in result["text"]:
            print_success("Response contains expected answer (4)")
        else:
            print_warning(f"Response doesn't contain expected answer: {result['text']}")
        
        return True
        
    except Exception as e:
        print_error(f"Generation failed: {e}")
        return False


def test_error_handling(provider: object) -> bool:
    """Test error handling with invalid input."""
    print_header("Step 5: Error Handling Test")
    
    try:
        from luma.core.llm.providers.provider_interface import ProviderError
        
        # Test with very short timeout (should fail)
        print_info("Testing timeout handling with 0.001s timeout...")
        
        # Create a new provider with very short timeout
        from luma.core.llm.providers.gemini_provider import GeminiProvider
        from luma.core.structured_logger import StructuredLogger
        
        config = {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "timeout": 0.001,  # Very short timeout
            "max_tokens": 1024,
            "temperature": 0.4,
            "log_prompts": False
        }
        
        logger = StructuredLogger("gemini_verify_timeout")
        timeout_provider = GeminiProvider(config=config, logger=logger)
        
        try:
            result = timeout_provider.generate(
                "Write a long essay.",
                {
                    "model": "gemini-2.5-flash",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "request_id": "verify-timeout"
                }
            )
            print_warning("Timeout test didn't fail (network might be very fast)")
            
        except ProviderError as e:
            if e.is_transient:
                print_success("Timeout error correctly marked as transient")
            else:
                print_warning("Timeout error not marked as transient")
            
            if "timeout" in str(e).lower():
                print_success("Error message mentions timeout")
            else:
                print_warning(f"Error message doesn't mention timeout: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        return False


def test_multiple_requests(provider: object) -> bool:
    """Test multiple sequential requests."""
    print_header("Step 6: Multiple Requests Test")
    
    try:
        prompts = [
            "What is 1 + 1?",
            "What is 2 + 2?",
            "What is 3 + 3?"
        ]
        
        print_info(f"Sending {len(prompts)} sequential requests...")
        
        for i, prompt in enumerate(prompts):
            options = {
                "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                "temperature": 0.0,
                "max_tokens": 10,
                "request_id": f"verify-multi-{i:03d}"
            }
            
            result = provider.generate(prompt, options)
            print_success(f"Request {i+1}/{len(prompts)}: {result['text'].strip()}")
        
        print_success("All requests completed successfully")
        return True
        
    except Exception as e:
        print_error(f"Multiple requests test failed: {e}")
        return False


def test_application_integration() -> bool:
    """Test integration with the application container."""
    print_header("Step 7: Application Integration Test")
    
    try:
        from luma.container import initialize_application
        
        print_info("Initializing application with Gemini provider...")
        
        # Initialize application (should load Gemini from environment)
        engine = initialize_application()
        
        print_success("Application initialized successfully")
        
        # Check that LLM is configured
        if hasattr(engine, 'llm') and engine.llm is not None:
            print_success(f"LLM is configured: {type(engine.llm).__name__}")
        else:
            print_error("LLM is not configured in ReasoningEngine")
            return False
        
        # Test a simple message
        print_info("Testing message processing...")
        response = engine.process_message("Hello, what is 2 + 2?")
        
        if response and "response" in response:
            print_success("Message processed successfully")
            print_info(f"Response: {response['response'][:100]}...")
        else:
            print_error("Message processing failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Application integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: Dict[str, bool]) -> None:
    """Print a summary of all test results."""
    print_header("Verification Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        if result:
            print_success(test_name)
        else:
            print_error(test_name)
    
    print(f"\n{'=' * 70}")
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"{'=' * 70}\n")
    
    if failed == 0:
        print_success("All verification tests passed!")
        print_info("Gemini provider is properly configured and working.")
        print_info("\nNext steps:")
        print_info("  1. Start the API server: luma-memory-server")
        print_info("  2. Test the chat endpoint")
        print_info("  3. Monitor logs for provider events")
    else:
        print_error(f"{failed} verification test(s) failed.")
        print_info("\nTroubleshooting:")
        print_info("  1. Check GEMINI_SETUP.md for detailed instructions")
        print_info("  2. Verify your API key is valid")
        print_info("  3. Check your internet connection")
        print_info("  4. Review error messages above")


def main() -> int:
    """Main verification function."""
    print_header("Gemini Provider Verification")
    print_info("This script will verify your Gemini integration")
    print_info("Make sure GEMINI_API_KEY is set in your environment\n")
    
    results = {}
    
    # Step 1: Check environment
    if not check_environment():
        print_error("\nEnvironment check failed. Cannot proceed.")
        return 1
    results["Environment Check"] = True
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print_error("\nDependency check failed. Cannot proceed.")
        return 1
    results["Dependency Check"] = True
    
    # Step 3: Initialize provider
    provider = test_provider_initialization()
    if provider is None:
        print_error("\nProvider initialization failed. Cannot proceed.")
        results["Provider Initialization"] = False
        print_summary(results)
        return 1
    results["Provider Initialization"] = True
    
    # Step 4: Test basic generation
    results["Basic Generation"] = test_basic_generation(provider)
    
    # Step 5: Test error handling
    results["Error Handling"] = test_error_handling(provider)
    
    # Step 6: Test multiple requests
    results["Multiple Requests"] = test_multiple_requests(provider)
    
    # Step 7: Test application integration
    results["Application Integration"] = test_application_integration()
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
