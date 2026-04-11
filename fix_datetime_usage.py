#!/usr/bin/env python3
"""
Script to fix datetime.UTC and datetime.utcnow() usage across the codebase.
"""

import os
import re
from pathlib import Path

def fix_file(filepath):
    """Fix datetime usage in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Check if file needs UTC import
        needs_utc_import = False
        if 'datetime.UTC' in content or 'datetime.utcnow()' in content:
            needs_utc_import = True
        
        # Fix datetime.UTC references
        if 'datetime.UTC' in content:
            content = re.sub(r'datetime\.UTC', 'UTC', content)
            modified = True
        
        # Fix datetime.utcnow() references
        if 'datetime.utcnow()' in content:
            content = re.sub(r'datetime\.utcnow\(\)', 'datetime.now(UTC)', content)
            modified = True
        
        # Add UTC import if needed and not already present
        if needs_utc_import and modified:
            # Check if UTC is already imported
            if 'from datetime import' in content and ', UTC' not in content and 'UTC' not in content.split('from datetime import')[1].split('\n')[0]:
                # Find the datetime import line and add UTC
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('from datetime import') and 'UTC' not in line:
                        # Add UTC to the import
                        if line.endswith('datetime'):
                            lines[i] = line + ', UTC'
                        elif 'datetime,' in line or 'datetime ' in line:
                            # Insert UTC after datetime
                            lines[i] = line.replace('datetime', 'datetime, UTC', 1)
                        modified = True
                        break
                content = '\n'.join(lines)
        
        # Write back if modified
        if modified and content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, filepath
        
        return False, None
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, None

def main():
    """Main function to fix all Python files."""
    # Directories to process
    dirs_to_process = ['tests', 'luma', 'luma_memory']
    
    fixed_files = []
    
    for dir_name in dirs_to_process:
        if not os.path.exists(dir_name):
            continue
            
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    was_fixed, fixed_path = fix_file(filepath)
                    if was_fixed:
                        fixed_files.append(fixed_path)
    
    print(f"\nFixed {len(fixed_files)} files:")
    for f in fixed_files:
        print(f"  - {f}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
