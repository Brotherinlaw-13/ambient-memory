#!/usr/bin/env python3
"""Test indexing a single file."""

import requests
import json

AMBIENT_SERVER = "http://localhost:9877"
TEST_FILE = "/Users/rook/workspace/memory/telegram/hogar-2026-02-16.md"

def main():
    print("🧪 Testing indexing with single file...")
    
    # Check server
    try:
        response = requests.get(f"{AMBIENT_SERVER}/health", timeout=5)
        print(f"✅ Server: {response.json()}")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    
    # Read test file
    try:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📄 File size: {len(content):,} chars")
    except Exception as e:
        print(f"❌ File error: {e}")
        return 1
    
    # Ingest
    print("🔄 Ingesting...")
    ingest_data = {
        "text": content,
        "source": "hogar-2026-02-16.md",
        "collection": None,  # Auto-classify
        "chunk_size": 800
    }
    
    try:
        response = requests.post(
            f"{AMBIENT_SERVER}/ingest",
            json=ingest_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['total_chunks']} chunks into {result['collections_used']}")
            return 0
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return 1
    except Exception as e:
        print(f"❌ Ingest error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())