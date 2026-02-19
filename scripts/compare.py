#!/usr/bin/env python3
"""
Compare search results between current system (localhost:9876) and ambient-memory (localhost:9877).

Usage: python3 compare.py "query text"
"""

import sys
import json
import requests
from datetime import datetime

CURRENT_SERVER = "http://localhost:9876"  # Current production system
AMBIENT_SERVER = "http://localhost:9877"  # New ambient-memory system

def query_current_system(query, limit=5):
    """Query the current production system."""
    try:
        url = f"{CURRENT_SERVER}/search?q={query}&n={limit}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "results": data.get("results", []),
                "query": data.get("query", query),
                "system": "current"
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "system": "current"}
    except Exception as e:
        return {"success": False, "error": str(e), "system": "current"}

def query_ambient_system(query, limit=5):
    """Query the ambient-memory system."""
    try:
        payload = {
            "query": query,
            "limit": limit,
            "include_scores": True
        }
        response = requests.post(f"{AMBIENT_SERVER}/query", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "results": data.get("results", []),
                "query": data.get("query", query),
                "execution_time_ms": data.get("execution_time_ms", 0),
                "system": "ambient"
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "system": "ambient"}
    except Exception as e:
        return {"success": False, "error": str(e), "system": "ambient"}

def format_current_result(result, index):
    """Format a result from current system for display."""
    return f"""
   {index+1}. [{result.get('similarity', 0):.3f}] ({result.get('source', 'unknown')})
       {result.get('snippet', '')[:200]}{'...' if len(result.get('snippet', '')) > 200 else ''}
    """

def format_ambient_result(result, index):
    """Format a result from ambient-memory for display."""
    final_score = result.get('final_score', 0)
    semantic = result.get('semantic_similarity', 0)
    keyword = result.get('keyword_score', 0)
    source = result.get('source', 'unknown')
    collection = result.get('collection', 'unknown')
    text = result.get('text', '')
    
    return f"""
   {index+1}. [{final_score:.3f}] (sem:{semantic:.2f}, kw:{keyword:.2f}) {collection}:{source}
       {text[:200]}{'...' if len(text) > 200 else ''}
    """

def score_relevance(query, result_text):
    """Simple relevance scoring (0-5 scale)."""
    query_lower = query.lower()
    text_lower = result_text.lower()
    
    score = 0
    query_words = query_lower.split()
    
    # Word overlap scoring
    matches = sum(1 for word in query_words if word in text_lower)
    if matches > 0:
        score += min(matches / len(query_words) * 3, 3)
    
    # Exact phrase matching
    if query_lower in text_lower:
        score += 2
    
    return min(score, 5)

def compare_systems(query):
    """Compare both systems for a given query."""
    print(f"🔍 Comparing systems for query: '{query}'")
    print("=" * 80)
    
    # Query both systems
    current_result = query_current_system(query)
    ambient_result = query_ambient_system(query)
    
    # Display results
    print("\n📊 CURRENT SYSTEM (localhost:9876)")
    print("-" * 40)
    if current_result["success"]:
        if current_result["results"]:
            for i, result in enumerate(current_result["results"]):
                print(format_current_result(result, i))
        else:
            print("   No results found")
    else:
        print(f"   ❌ Error: {current_result['error']}")
    
    print("\n🆕 AMBIENT-MEMORY (localhost:9877)")
    print("-" * 40)
    if ambient_result["success"]:
        if ambient_result["results"]:
            print(f"   Execution time: {ambient_result.get('execution_time_ms', 0)}ms")
            for i, result in enumerate(ambient_result["results"]):
                print(format_ambient_result(result, i))
        else:
            print("   No results found")
    else:
        print(f"   ❌ Error: {ambient_result['error']}")
    
    # Relevance comparison
    print("\n📈 RELEVANCE COMPARISON")
    print("-" * 40)
    
    if current_result["success"] and ambient_result["success"]:
        current_scores = []
        ambient_scores = []
        
        for result in current_result["results"]:
            score = score_relevance(query, result.get("snippet", ""))
            current_scores.append(score)
        
        for result in ambient_result["results"]:
            score = score_relevance(query, result.get("text", ""))
            ambient_scores.append(score)
        
        current_avg = sum(current_scores) / len(current_scores) if current_scores else 0
        ambient_avg = sum(ambient_scores) / len(ambient_scores) if ambient_scores else 0
        
        print(f"   Current system avg relevance: {current_avg:.2f}/5")
        print(f"   Ambient-memory avg relevance: {ambient_avg:.2f}/5")
        
        if ambient_avg > current_avg:
            print(f"   🏆 Ambient-memory wins by {ambient_avg - current_avg:.2f} points")
        elif current_avg > ambient_avg:
            print(f"   🏆 Current system wins by {current_avg - ambient_avg:.2f} points")
        else:
            print("   🤝 Tie!")
    
    return {
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "current": current_result,
        "ambient": ambient_result
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 compare.py \"query text\"")
        return 1
    
    query = sys.argv[1]
    result = compare_systems(query)
    
    # Optional: save to file
    # with open("comparison_result.json", "w") as f:
    #     json.dump(result, f, indent=2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())