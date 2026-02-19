#!/usr/bin/env python3
"""
Index Telegram archive data into ambient-memory server.

Reads all .md files from /Users/rook/workspace/memory/telegram/,
chunks them using ConversationChunker, and ingests them via localhost:9877/ingest.
"""

import os
import sys
import json
import glob
import requests
from pathlib import Path
from datetime import datetime

# Add ambient_memory to path
sys.path.insert(0, '/Users/rook/workspace/ambient-memory/src')

from ambient_memory.chunking import ConversationChunker

TELEGRAM_DIR = "/Users/rook/workspace/memory/telegram"
AMBIENT_SERVER = "http://localhost:9877"
CHUNK_SIZE = 800  # Default chunk size

def main():
    print("🔍 Indexing Telegram archives into ambient-memory...")
    print(f"📂 Source: {TELEGRAM_DIR}")
    print(f"🌐 Server: {AMBIENT_SERVER}")
    
    # Check server is running
    try:
        response = requests.get(f"{AMBIENT_SERVER}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server health check failed: {response.status_code}")
            return 1
        print(f"✅ Server is healthy: {response.json()}")
    except Exception as e:
        print(f"❌ Cannot connect to ambient-memory server: {e}")
        print("   Make sure server is running on localhost:9877")
        return 1
    
    # Find all .md files
    md_files = glob.glob(os.path.join(TELEGRAM_DIR, "*.md"))
    print(f"📋 Found {len(md_files)} .md files to process")
    
    if not md_files:
        print("❌ No .md files found in telegram directory")
        return 1
    
    total_chunks = 0
    processed_files = 0
    failed_files = 0
    
    for file_path in sorted(md_files):
        filename = os.path.basename(file_path)
        print(f"\n📄 Processing: {filename}")
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"   ⚠️  Empty file, skipping")
                continue
            
            print(f"   📝 File size: {len(content):,} characters")
            
            # Ingest via API
            ingest_data = {
                "text": content,
                "source": filename,
                "collection": None,  # Let it auto-classify
                "chunk_size": CHUNK_SIZE
            }
            
            print(f"   🔄 Ingesting...", flush=True)
            response = requests.post(
                f"{AMBIENT_SERVER}/ingest",
                json=ingest_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                file_chunks = result["total_chunks"]
                collections_used = result["collections_used"]
                
                print(f"   ✅ Indexed {file_chunks} chunks into collections: {', '.join(collections_used)}")
                total_chunks += file_chunks
                processed_files += 1
            else:
                print(f"   ❌ Ingestion failed: {response.status_code} - {response.text}")
                failed_files += 1
                
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            failed_files += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Processed: {processed_files} files")
    print(f"   ❌ Failed: {failed_files} files") 
    print(f"   🧩 Total chunks: {total_chunks}")
    
    # Get collections info
    try:
        response = requests.get(f"{AMBIENT_SERVER}/collections", timeout=10)
        if response.status_code == 200:
            collections = response.json()
            print(f"\n📚 Collections after indexing:")
            for col in collections:
                print(f"   • {col['name']}: {col['count']} chunks")
        else:
            print(f"   ⚠️  Could not get collection stats: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Could not get collection stats: {e}")
    
    if failed_files == 0:
        print(f"\n🎉 All files processed successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed_files} files failed processing")
        return 1

if __name__ == "__main__":
    sys.exit(main())