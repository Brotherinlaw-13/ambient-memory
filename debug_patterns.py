#!/usr/bin/env python3

import re

# Get actual line from file
with open('/Users/rook/workspace/memory/telegram/the-factory-2026-02-16.md', 'r') as f:
    content = f.read()
    
lines_with_diego = [line for line in content.split('\n') if 'Diego:' in line]
line = lines_with_diego[0]  # First Diego line

print(f"Line: {repr(line)}")

# Test correct patterns
patterns_to_test = [
    r'`\d+:\d+`',  # just timestamp
    r'\*\*Diego:\*\*',  # just name with correct format
    r'`\d+:\d+`\s*\*\*Diego:\*\*',  # timestamp + name
    r'`\d+:\d+`\s*\*\*Diego:\*\*\s*(.+)',  # full pattern
    r'.*\*\*Diego:\*\*\s*(.+)',  # anywhere in line
]

for i, pattern in enumerate(patterns_to_test):
    match = re.search(pattern, line)
    result = match.group(1) if match and match.groups() else match.group(0) if match else 'No match'
    print(f'Step {i+1} ({pattern}): {result}')