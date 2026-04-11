#!/usr/bin/env python3
"""
Script to fix corrupted datetime imports.
"""

import os
import re

def fix_file(filepath):
    """Fix datetime import in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix corrupted imports like "from datetime, UTC import"
        content = re.sub(r'from datetime, UTC import', 'from datetime import', content)
        
        # Ensure UTC is in the import if datetime.now(UTC) or just UTC is used
        if ('datetime.now(UTC)' in content or re.search(r'\.replace\(tzinfo=UTC\)', content)) and 'from datetime import' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('from datetime import') and 'UTC' not in line:
                    # Add UTC to import
                    if line.strip().endswith('datetime'):
                        lines[i] = line.replace('datetime', 'datetime, UTC')
                    elif line.strip().endswith('timedelta'):
                        lines[i] = line.replace('timedelta', 'timedelta, UTC')
                    elif 'datetime,' in line:
                        lines[i] = line.replace('datetime,', 'datetime, UTC,')
                    elif 'timedelta,' in line:
                        lines[i] = line.replace('timedelta,', 'timedelta, UTC,')
                    break
            content = '\n'.join(lines)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function."""
    dirs_to_process = ['tests', 'luma', 'luma_memory']
    
    fixed_count = 0
    for dir_name in dirs_to_process:
        if not os.path.exists(dir_name):
            continue
            
        for root, dirs, files in os.walk(dir_name):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if fix_file(filepath):
                        print(f"Fixed: {filepath}")
                        fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
