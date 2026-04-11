#!/usr/bin/env python3
"""Run tests in batches and collect summary results."""

import subprocess
import sys
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return results."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "--no-cov", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            timeout=15  # Reduced timeout
        )
        output = result.stdout + result.stderr
        
        # Parse results
        if "passed" in output or "failed" in output:
            # Extract summary line
            for line in output.split('\n'):
                if 'passed' in line or 'failed' in line:
                    return test_file.name, line.strip()
        return test_file.name, "No results"
    except subprocess.TimeoutExpired:
        return test_file.name, "TIMEOUT"
    except Exception as e:
        return test_file.name, f"ERROR: {e}"

def main():
    test_dir = Path("tests")
    test_files = sorted(test_dir.glob("test_*.py"))
    
    print(f"Found {len(test_files)} test files")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    failed_files = []
    
    for i, test_file in enumerate(test_files, 1):  # Run all files
        print(f"[{i}/{len(test_files)}] {test_file.name[:40]:<40}", end=" ")
        name, result = run_test_file(test_file)
        print(result[:60])
        
        # Parse counts
        if "passed" in result:
            parts = result.split()
            for j, part in enumerate(parts):
                if "passed" in part and j > 0:
                    try:
                        total_passed += int(parts[j-1])
                    except:
                        pass
                if "failed" in part and j > 0:
                    try:
                        count = int(parts[j-1])
                        total_failed += count
                        if count > 0:
                            failed_files.append(name)
                    except:
                        pass
    
    print("=" * 80)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    if failed_files:
        print(f"\nFiles with failures:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
