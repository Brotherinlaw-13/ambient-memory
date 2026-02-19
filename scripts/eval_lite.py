#!/usr/bin/env python3
"""Lightweight evaluation: uses only httpx to query localhost:9876. No heavy imports."""

import json
import re
import time
import httpx
from pathlib import Path
from collections import defaultdict

TELEGRAM_DIR = Path("/Users/rook/workspace/memory/telegram")
EMBEDDING_SERVER = "http://localhost:9876"
COLLECTIONS = ["memory_work", "memory_projects", "memory_personal", "memory_infrastructure", "memory_general", "telegram_memory", "workspace_memory"]

def extract_diego_messages(filepath):
    """Extract Diego's messages with surrounding context."""
    lines = filepath.read_text().splitlines()
    messages = []
    for i, line in enumerate(lines):
        m = re.search(r'`(\d{2}:\d{2})`\s*\*\*Diego(?:\s*\(.*?\))?:\*\*\s*(.+)', line)
        if not m:
            m = re.search(r'\*\*Diego\*\*:\s*(.+)', line)
            if m:
                ts = None
                text = m.group(1).strip()
            else:
                continue
        else:
            ts = m.group(1)
            text = m.group(2).strip()
        
        # Skip pure attachments, lone emojis
        if re.match(r'^\[📎[^\]]+\]$', text):
            continue
        if not any(c.isalpha() for c in text):
            continue
        
        # Context: ±3 lines
        ctx_before = [lines[j].strip() for j in range(max(0, i-3), i) if lines[j].strip()]
        ctx_after = [lines[j].strip() for j in range(i+1, min(len(lines), i+4)) if lines[j].strip()]
        
        messages.append({
            "text": text,
            "timestamp": ts,
            "source": filepath.name,
            "line": i,
            "context_before": ctx_before,
            "context_after": ctx_after
        })
    return messages

def query_memory(text, context=None, n_results=3):
    """Query the embedding server."""
    results_all = []
    # Search top 3 most relevant collections only to save memory
    for collection in COLLECTIONS[:3]:
        try:
            r = httpx.post(f"{EMBEDDING_SERVER}/query", json={
                "query": text,
                "collection": collection,
                "n_results": n_results
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                # Server returns a list of dicts with text/source/distance
                if isinstance(data, list):
                    for item in data:
                        dist = item.get("distance", 1.0)
                        similarity = 1 - (dist / 2.0)
                        results_all.append({
                            "document": item.get("text", "")[:200],
                            "similarity": round(similarity, 4),
                            "distance": round(dist, 4),
                            "collection": collection,
                            "source": item.get("source", "unknown")
                        })
                # ChromaDB format fallback
                elif isinstance(data, dict):
                    docs = data.get("documents", [[]])[0]
                    dists = data.get("distances", [[]])[0]
                    metas = data.get("metadatas", [[]])[0]
                    for doc, dist, meta in zip(docs, dists, metas):
                        similarity = 1 - (dist / 2.0)
                        results_all.append({
                            "document": doc[:200],
                            "similarity": round(similarity, 4),
                            "distance": round(dist, 4),
                            "collection": collection,
                            "source": meta.get("source", "unknown")
                        })
        except Exception as e:
            pass
    
    # Sort by similarity descending, take top n
    results_all.sort(key=lambda x: x["similarity"], reverse=True)
    return results_all[:n_results]

def score_relevance(query_text, query_context, result):
    """Score how relevant a result is to the query. Returns 0-1."""
    doc = result["document"].lower()
    q = query_text.lower()
    
    score = 0.0
    
    # 1. Entity overlap (strongest signal)
    query_words = set(w.strip(".,!?") for w in q.split() if len(w) > 2)
    doc_words = set(w.strip(".,!?") for w in doc.split() if len(w) > 2)
    
    # Known project/entity names
    entities = ["darwin", "chowdown", "hire space", "hirespace", "bluesky", "telegram", 
                "openclaw", "railway", "vercel", "chromadb", "mcas", "wodboard",
                "root juice", "pocket guide", "story time", "aldermor", "stitch"]
    
    query_entities = [e for e in entities if e in q]
    doc_entities = [e for e in entities if e in doc]
    
    if query_entities:
        matching = set(query_entities) & set(doc_entities)
        if matching:
            score += 0.4 * (len(matching) / len(query_entities))
        else:
            score -= 0.2  # Penalty: query about X, result about Y
    
    # 2. Word overlap
    common = query_words & doc_words
    if query_words:
        overlap_ratio = len(common) / len(query_words)
        score += 0.3 * overlap_ratio
    
    # 3. Context overlap (if query had context)
    if query_context:
        ctx_text = " ".join(query_context).lower()
        ctx_words = set(w.strip(".,!?") for w in ctx_text.split() if len(w) > 3)
        ctx_common = ctx_words & doc_words
        if ctx_words:
            score += 0.2 * (len(ctx_common) / len(ctx_words))
    
    # 4. Similarity bonus
    sim = result["similarity"]
    if sim > 0.7:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def main():
    # Check embedding server
    try:
        r = httpx.get(f"{EMBEDDING_SERVER}/health", timeout=5)
        print(f"Embedding server: {r.status_code}")
    except:
        print("ERROR: Embedding server not available")
        return
    
    # Collect all Diego messages from Feb 10-17
    all_messages = []
    for day in range(10, 18):
        f = TELEGRAM_DIR / f"the-factory-2026-02-{day}.md"
        if f.exists():
            msgs = extract_diego_messages(f)
            all_messages.extend(msgs)
            print(f"  {f.name}: {len(msgs)} messages")
    
    # Limit to 50 queries for memory-safe evaluation
    if len(all_messages) > 50:
        import random
        random.seed(42)
        all_messages = random.sample(all_messages, 50)
    
    print(f"\nEvaluating: {len(all_messages)} queries")
    
    # Evaluate
    results = []
    total_relevance = 0
    noise_count = 0
    high_precision_count = 0
    total_results = 0
    no_results = 0
    
    for i, msg in enumerate(all_messages):
        t0 = time.time()
        search_results = query_memory(msg["text"], msg["context_before"] + msg["context_after"])
        elapsed = (time.time() - t0) * 1000
        
        if not search_results:
            no_results += 1
            continue
        
        msg_scores = []
        for sr in search_results:
            rel = score_relevance(msg["text"], msg["context_before"] + msg["context_after"], sr)
            msg_scores.append(rel)
            total_results += 1
            total_relevance += rel
            if rel >= 0.4:
                high_precision_count += 1
            if rel < 0.1:
                noise_count += 1
        
        results.append({
            "query": msg["text"][:100],
            "source": msg["source"],
            "results_count": len(search_results),
            "avg_relevance": round(sum(msg_scores) / len(msg_scores), 3) if msg_scores else 0,
            "best_relevance": round(max(msg_scores), 3) if msg_scores else 0,
            "time_ms": round(elapsed),
            "top_result": search_results[0]["document"][:100] if search_results else "",
            "top_similarity": search_results[0]["similarity"] if search_results else 0
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(all_messages)}...")
    
    # Summary
    queries_with_results = len(all_messages) - no_results
    avg_relevance = (total_relevance / total_results * 100) if total_results else 0
    precision = (high_precision_count / total_results * 100) if total_results else 0
    noise_rate = (noise_count / total_results * 100) if total_results else 0
    coverage = (queries_with_results / len(all_messages) * 100) if all_messages else 0
    
    summary = {
        "phase": "Phase 6 - Real Data Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_queries": len(all_messages),
        "queries_with_results": queries_with_results,
        "total_results": total_results,
        "coverage": round(coverage, 1),
        "avg_relevance_pct": round(avg_relevance, 1),
        "high_precision_pct": round(precision, 1),
        "noise_rate_pct": round(noise_rate, 1),
        "avg_time_ms": round(sum(r["time_ms"] for r in results) / len(results)) if results else 0,
    }
    
    print(f"\n{'='*50}")
    print(f"RESULTS (Real Data via localhost:9876)")
    print(f"{'='*50}")
    print(f"Queries: {summary['total_queries']}")
    print(f"Coverage: {summary['coverage']}%")
    print(f"Avg Relevance: {summary['avg_relevance_pct']}%")
    print(f"High Precision (≥0.4): {summary['high_precision_pct']}%")
    print(f"Noise (<0.1): {summary['noise_rate_pct']}%")
    print(f"Avg Time: {summary['avg_time_ms']}ms")
    
    # Save
    out_dir = Path(__file__).parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "phase6-baseline.json", "w") as f:
        json.dump({"summary": summary, "details": results[:50]}, f, indent=2)
    
    print(f"\nSaved to eval/phase6-baseline.json")
    
    # Show best and worst
    results.sort(key=lambda x: x["avg_relevance"], reverse=True)
    print(f"\nTop 5 best matches:")
    for r in results[:5]:
        print(f"  [{r['avg_relevance']:.2f}] \"{r['query'][:60]}\" → {r['top_result'][:60]}")
    
    print(f"\nTop 5 worst (noise):")
    for r in results[-5:]:
        print(f"  [{r['avg_relevance']:.2f}] \"{r['query'][:60]}\" → {r['top_result'][:60]}")

if __name__ == "__main__":
    main()
