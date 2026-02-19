#!/usr/bin/env python3
"""
Simplified evaluation test using existing embedding server.

Tests the evaluation approach without requiring full ChromaDB setup.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import httpx
from dataclasses import dataclass


@dataclass
class EvaluationQuery:
    """A query extracted from conversation data for evaluation."""
    text: str
    source_file: str
    line_number: int
    timestamp: Optional[str] = None
    context_before: List[str] = None
    context_after: List[str] = None
    
    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []


class ConversationParser:
    """Parse Telegram conversation files and extract Diego's messages."""
    
    def __init__(self):
        # Patterns to identify Diego's messages
        self.diego_patterns = [
            r'^\s*`\d+:\d+`\s*\*\*Diego\*\*[:：]\s*(.+)',  # `12:34` **Diego**: message
            r'^\s*\*\*Diego\*\*[:：]\s*(.+)',  # **Diego**: message  
            r'^\s*Diego[:：]\s*(.+)',  # Diego: message
        ]
        
        # Pattern to extract timestamps
        self.timestamp_pattern = r'`(\d{2}:\d{2})`'
    
    def extract_diego_messages(self, file_path: Path) -> List[EvaluationQuery]:
        """Extract Diego's messages from a conversation file."""
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line contains a Diego message
            diego_text = None
            
            # Look for **Diego:** format specifically (note the space after **Diego**)
            if '**Diego:**' in line:
                parts = line.split('**Diego:**', 1)
                if len(parts) > 1:
                    diego_text = parts[1].strip()
            else:
                # Try the regex patterns
                for pattern in self.diego_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match and match.groups():
                        diego_text = match.group(1).strip()
                        break
            
            if diego_text and len(diego_text) >= 10:  # Only consider substantive messages
                # Extract timestamp if present
                timestamp_match = re.search(self.timestamp_pattern, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # Get context lines (3 before, 3 after)
                context_before = []
                context_after = []
                
                for j in range(max(0, i-3), i):
                    if j < len(lines):
                        context_before.append(lines[j].strip())
                
                for j in range(i+1, min(len(lines), i+4)):
                    if j < len(lines):
                        context_after.append(lines[j].strip())
                
                queries.append(EvaluationQuery(
                    text=diego_text,
                    source_file=str(file_path),
                    line_number=i + 1,
                    timestamp=timestamp,
                    context_before=context_before,
                    context_after=context_after
                ))
        
        return queries


class SimpleMemoryTester:
    """Simple memory tester using existing embedding server."""
    
    def __init__(self, embedding_server_url: str = "http://localhost:9876"):
        self.embedding_server_url = embedding_server_url
        self.parser = ConversationParser()
    
    def test_embedding_server(self) -> bool:
        """Test if the embedding server is responding."""
        try:
            response = httpx.get(f"{self.embedding_server_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            print(f"Embedding server test failed: {e}")
            return False
    
    def query_existing_memory(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Query the existing memory system via the embedding server."""
        try:
            response = httpx.get(
                f"{self.embedding_server_url}/search",
                params={"q": query, "n": n},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Memory query failed for '{query[:50]}...': {e}")
            return []
    
    def evaluate_sample_queries(self, queries: List[EvaluationQuery], limit: int = 10) -> Dict[str, Any]:
        """Evaluate a sample of queries against the existing memory."""
        test_queries = queries[:limit]
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_queries_available": len(queries),
            "queries_tested": len(test_queries),
            "embedding_server_status": "online" if self.test_embedding_server() else "offline",
            "query_results": []
        }
        
        print(f"🔍 Testing {len(test_queries)} queries against existing memory...")
        
        for i, query in enumerate(test_queries):
            print(f"   Query {i+1}: '{query.text[:50]}...'")
            
            memory_results = self.query_existing_memory(query.text, n=3)
            
            query_result = {
                "query": query.text,
                "source": Path(query.source_file).name,
                "line_number": query.line_number,
                "timestamp": query.timestamp,
                "results_count": len(memory_results),
                "results": [
                    {
                        "snippet": result.get("snippet", "")[:200],
                        "source": result.get("source", ""),
                        "similarity": result.get("final_score", 0),
                        "collection": result.get("collection", "")
                    }
                    for result in memory_results
                ]
            }
            
            results["query_results"].append(query_result)
            
            # Basic analysis
            if memory_results:
                best_score = max(r.get("final_score", 0) for r in memory_results)
                print(f"      → {len(memory_results)} results, best score: {best_score:.3f}")
            else:
                print(f"      → No results found")
        
        # Compute basic metrics
        queries_with_results = sum(1 for qr in results["query_results"] if qr["results_count"] > 0)
        total_results = sum(qr["results_count"] for qr in results["query_results"])
        
        results["summary"] = {
            "coverage": queries_with_results / len(test_queries) if test_queries else 0,
            "avg_results_per_query": total_results / len(test_queries) if test_queries else 0,
            "queries_with_results": queries_with_results,
            "total_results": total_results
        }
        
        return results


def main():
    """Run the simplified evaluation test."""
    print("🧠 Ambient Memory Evaluation Test")
    print("=" * 50)
    
    # Set up paths
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    # Test data files
    test_files = [
        telegram_dir / "the-factory-2026-02-17.md",
        telegram_dir / "the-factory-2026-02-16.md"
    ]
    
    # Check files exist
    available_files = [f for f in test_files if f.exists()]
    if not available_files:
        print("❌ No test files found. Available files:")
        for f in telegram_dir.glob("the-factory-*.md"):
            print(f"    {f.name}")
        return
    
    print(f"📁 Using test files: {[f.name for f in available_files]}")
    
    # Initialise tester
    tester = SimpleMemoryTester()
    
    # Test embedding server
    print("\n🔍 Testing embedding server connection...")
    if not tester.test_embedding_server():
        print("❌ Embedding server not available. Make sure it's running on localhost:9876")
        return
    
    print("✅ Embedding server is online")
    
    # Extract Diego's messages
    print("\n📋 Extracting Diego's messages...")
    all_queries = []
    for file_path in available_files:
        queries = tester.parser.extract_diego_messages(file_path)
        print(f"   {file_path.name}: {len(queries)} messages found")
        all_queries.extend(queries)
    
    print(f"✅ Total Diego messages extracted: {len(all_queries)}")
    
    if not all_queries:
        print("❌ No Diego messages found in test files")
        return
    
    # Show sample queries
    print("\n📝 Sample queries:")
    for i, q in enumerate(all_queries[:3]):
        print(f"   {i+1}. '{q.text[:80]}{'...' if len(q.text) > 80 else ''}'")
        print(f"      Source: {Path(q.source_file).name}:{q.line_number}")
    
    # Run evaluation
    print("\n🧪 Running evaluation test...")
    results = tester.evaluate_sample_queries(all_queries, limit=15)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    summary = results["summary"]
    print(f"Queries Tested:        {results['queries_tested']}")
    print(f"Coverage:              {summary['coverage']:.1%} ({summary['queries_with_results']}/{results['queries_tested']} queries had results)")
    print(f"Avg Results per Query: {summary['avg_results_per_query']:.1f}")
    print(f"Total Results:         {summary['total_results']}")
    print(f"Embedding Server:      {results['embedding_server_status']}")
    
    # Save results
    results_file = eval_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Analysis
    print("\n🔍 Analysis:")
    if summary['coverage'] > 0.7:
        print("✅ Good coverage - most queries returned results")
    elif summary['coverage'] > 0.3:
        print("⚠️  Moderate coverage - some queries returned results")
    else:
        print("❌ Low coverage - few queries returned results")
    
    if summary['avg_results_per_query'] > 2:
        print("✅ Good result density")
    elif summary['avg_results_per_query'] > 0.5:
        print("⚠️  Moderate result density")
    else:
        print("❌ Low result density")
    
    print("\n✅ Test complete!")
    print("\n📝 Next steps:")
    print("   1. Fix ChromaDB dependencies to enable full ingestion")
    print("   2. Implement relevance scoring")
    print("   3. Add feedback collection")
    print("   4. Build the FastAPI server")


if __name__ == "__main__":
    main()