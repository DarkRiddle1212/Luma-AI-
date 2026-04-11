#!/usr/bin/env python3
"""
Script to fix all datetime.UTC and datetime.utcnow() usage across the codebase.
"""

import os
import re
from pathlib import Path

def fix_datetime_in_file(filepath):
    """Fix datetime usage in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    modified = False
    
    # Check if file needs UTC import
    needs_utc_import = False
    if 'datetime.UTC' in content or 'datetime.utcnow()' in content:
        # Check if UTC is already imported
        if 'from datetime import' in content and 'UTC' not in content:
            needs_utc_import = True
    
    # Fix datetime.UTC references
    if 'datetime.UTC' in content:
        print(f"  Fixing datetime.UTC in {filepath}")
        content = re.sub(r'datetime\.UTC', 'UTC', content)
        modified = True
    
    # Fix datetime.utcnow() references
    if 'datetime.utcnow()' in content:
        print(f"  Fixing datetime.utcnow() in {filepath}")
        content = re.sub(r'datetime\.utcnow\(\)', 'datetime.now(UTC)', content)
        modified = True
        needs_utc_import = True
    
    # Add UTC import if needed
    if needs_utc_import and modified:
        # Find the datetime import line
        import_pattern = r'from datetime import ([^\n]+)'
        match = re.search(import_pattern, content)
        if match:
            imports = match.group(1)
            if 'UTC' not in imports:
                # Add UTC to existing import
                new_imports = imports.rstrip() + ', UTC'
                content = re.sub(import_pattern, f'from datetime import {new_imports}', content, count=1)
                print(f"  Added UTC to imports in {filepath}")
    
    # Write back if modified
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix datetime usage in all Python files."""
    # Directories to process
    dirs_to_process = ['luma', 'luma_memory', 'tests']
    
    files_modified = 0
    for dir_name in dirs_to_process:
        if not os.path.exists(dir_name):
            continue
            
        print(f"\nProcessing {dir_name}/...")
        for filepath in Path(dir_name).rglob('*.py'):
            if fix_datetime_in_file(filepath):
                files_modified += 1
    
    print(f"\n✅ Fixed {files_modified} files")

if __name__ == '__main__':
    main()
