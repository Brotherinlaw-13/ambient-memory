#!/usr/bin/env python3
"""Test improvements over baseline. Each improvement measured independently."""

import json
import re
import time
import httpx
from pathlib import Path
import random

TELEGRAM_DIR = Path("/Users/rook/workspace/memory/telegram")
EMBEDDING_SERVER = "http://localhost:9876"
COLLECTIONS = ["memory_work", "memory_projects", "memory_personal", "memory_infrastructure", "memory_general", "telegram_memory", "workspace_memory"]

def extract_diego_messages(filepath):
    lines = filepath.read_text().splitlines()
    messages = []
    for i, line in enumerate(lines):
        m = re.search(r'`(\d{2}:\d{2})`\s*\*\*Diego(?:\s*\(.*?\))?:\*\*\s*(.+)', line)
        if not m:
            m = re.search(r'\*\*Diego\*\*:\s*(.+)', line)
            if m:
                ts, text = None, m.group(1).strip()
            else:
                continue
        else:
            ts, text = m.group(1), m.group(2).strip()
        
        if re.match(r'^\[📎[^\]]+\]$', text):
            continue
        if not any(c.isalpha() for c in text):
            continue
        
        ctx_before = [lines[j].strip() for j in range(max(0, i-3), i) if lines[j].strip()]
        ctx_after = [lines[j].strip() for j in range(i+1, min(len(lines), i+4)) if lines[j].strip()]
        
        messages.append({
            "text": text, "timestamp": ts, "source": filepath.name,
            "line": i, "context_before": ctx_before, "context_after": ctx_after
        })
    return messages

def query_memory(text, n_results=3, collections=None):
    results_all = []
    for collection in (collections or COLLECTIONS[:3]):
        try:
            r = httpx.post(f"{EMBEDDING_SERVER}/query", json={
                "query": text, "collection": collection, "n_results": n_results
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    for item in data:
                        dist = item.get("distance", 1.0)
                        results_all.append({
                            "document": item.get("text", "")[:200],
                            "similarity": round(1 - dist/2.0, 4),
                            "distance": round(dist, 4),
                            "collection": collection,
                            "source": item.get("source", "unknown")
                        })
        except:
            pass
    results_all.sort(key=lambda x: x["similarity"], reverse=True)
    return results_all[:n_results]

ENTITIES = ["darwin", "chowdown", "hire space", "hirespace", "bluesky", "telegram",
            "openclaw", "railway", "vercel", "chromadb", "mcas", "wodboard",
            "root juice", "pocket guide", "story time", "aldermor", "stitch",
            "raquel", "dante", "tristan", "tello"]

def score_relevance(query_text, context, result):
    doc = result["document"].lower()
    q = query_text.lower()
    score = 0.0
    
    query_words = set(w.strip(".,!?") for w in q.split() if len(w) > 2)
    doc_words = set(w.strip(".,!?") for w in doc.split() if len(w) > 2)
    
    query_entities = [e for e in ENTITIES if e in q]
    doc_entities = [e for e in ENTITIES if e in doc]
    
    if query_entities:
        matching = set(query_entities) & set(doc_entities)
        if matching:
            score += 0.4 * (len(matching) / len(query_entities))
        else:
            score -= 0.2
    
    common = query_words & doc_words
    if query_words:
        score += 0.3 * (len(common) / len(query_words))
    
    if context:
        ctx_text = " ".join(context).lower()
        ctx_words = set(w.strip(".,!?") for w in ctx_text.split() if len(w) > 3)
        ctx_common = ctx_words & doc_words
        if ctx_words:
            score += 0.2 * (len(ctx_common) / len(ctx_words))
    
    if result["similarity"] > 0.7:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def evaluate(messages, strategy_name, query_fn):
    """Run evaluation with a custom query function."""
    total_relevance = 0
    noise_count = 0
    high_prec = 0
    total_results = 0
    no_results = 0
    
    for msg in messages:
        results = query_fn(msg)
        if not results:
            no_results += 1
            continue
        for r in results:
            rel = score_relevance(msg["text"], msg["context_before"] + msg["context_after"], r)
            total_results += 1
            total_relevance += rel
            if rel >= 0.4:
                high_prec += 1
            if rel < 0.1:
                noise_count += 1
    
    queries_with = len(messages) - no_results
    return {
        "strategy": strategy_name,
        "coverage": round(queries_with / len(messages) * 100, 1) if messages else 0,
        "avg_relevance": round(total_relevance / total_results * 100, 1) if total_results else 0,
        "precision": round(high_prec / total_results * 100, 1) if total_results else 0,
        "noise": round(noise_count / total_results * 100, 1) if total_results else 0,
        "total_results": total_results,
        "queries_with_results": queries_with
    }

def main():
    httpx.get(f"{EMBEDDING_SERVER}/health", timeout=5)
    
    all_messages = []
    for day in range(10, 18):
        f = TELEGRAM_DIR / f"the-factory-2026-02-{day}.md"
        if f.exists():
            all_messages.extend(extract_diego_messages(f))
    
    random.seed(42)
    messages = random.sample(all_messages, min(50, len(all_messages)))
    print(f"Testing {len(messages)} queries\n")
    
    results = []
    
    # 1. Baseline (raw)
    print("Testing: Baseline...")
    r = evaluate(messages, "Baseline", lambda m: query_memory(m["text"]))
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 2. Min threshold 0.55
    print("Testing: Min threshold 0.55...")
    def with_threshold_55(msg):
        res = query_memory(msg["text"])
        return [r for r in res if r["similarity"] >= 0.55]
    r = evaluate(messages, "Min threshold 0.55", with_threshold_55)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 3. Min threshold 0.60
    print("Testing: Min threshold 0.60...")
    def with_threshold_60(msg):
        res = query_memory(msg["text"])
        return [r for r in res if r["similarity"] >= 0.60]
    r = evaluate(messages, "Min threshold 0.60", with_threshold_60)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 4. Min threshold 0.65
    print("Testing: Min threshold 0.65...")
    def with_threshold_65(msg):
        res = query_memory(msg["text"])
        return [r for r in res if r["similarity"] >= 0.65]
    r = evaluate(messages, "Min threshold 0.65", with_threshold_65)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 5. Context expansion (use context for query)
    print("Testing: Context expansion...")
    def with_context(msg):
        q = msg["text"]
        words = q.split()
        if len(words) <= 5 and msg["context_before"]:
            q = " ".join(msg["context_before"][-2:]) + " " + q
        return query_memory(q)
    r = evaluate(messages, "Context expansion (short)", with_context)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 6. Universal context (ALL queries get context)
    print("Testing: Universal context...")
    def with_universal_context(msg):
        q = msg["text"]
        if msg["context_before"]:
            q = " ".join(msg["context_before"][-2:]) + " " + q
        return query_memory(q)
    r = evaluate(messages, "Universal context", with_universal_context)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 7. Universal context + threshold 0.55
    print("Testing: Universal context + threshold 0.55...")
    def combined_55(msg):
        q = msg["text"]
        if msg["context_before"]:
            q = " ".join(msg["context_before"][-2:]) + " " + q
        res = query_memory(q)
        return [r for r in res if r["similarity"] >= 0.55]
    r = evaluate(messages, "Universal ctx + thresh 0.55", combined_55)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 8. Universal context + threshold 0.60
    print("Testing: Universal context + threshold 0.60...")
    def combined_60(msg):
        q = msg["text"]
        if msg["context_before"]:
            q = " ".join(msg["context_before"][-2:]) + " " + q
        res = query_memory(q)
        return [r for r in res if r["similarity"] >= 0.60]
    r = evaluate(messages, "Universal ctx + thresh 0.60", combined_60)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 9. Search ALL collections
    print("Testing: All 7 collections...")
    def all_collections(msg):
        return query_memory(msg["text"], collections=COLLECTIONS)
    r = evaluate(messages, "All 7 collections", all_collections)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # 10. All collections + universal context + threshold 0.55
    print("Testing: All cols + universal ctx + thresh 0.55...")
    def full_combo(msg):
        q = msg["text"]
        if msg["context_before"]:
            q = " ".join(msg["context_before"][-2:]) + " " + q
        res = query_memory(q, collections=COLLECTIONS)
        return [r for r in res if r["similarity"] >= 0.55]
    r = evaluate(messages, "Full combo", full_combo)
    results.append(r)
    print(f"  Rel={r['avg_relevance']}% Prec={r['precision']}% Noise={r['noise']}% Cov={r['coverage']}%")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Strategy':<35} {'Rel%':>6} {'Prec%':>6} {'Noise%':>7} {'Cov%':>6}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['strategy']:<35} {r['avg_relevance']:>5.1f}% {r['precision']:>5.1f}% {r['noise']:>6.1f}% {r['coverage']:>5.1f}%")
    
    # Save
    out_dir = Path(__file__).parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "improvements-comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to eval/improvements-comparison.json")

if __name__ == "__main__":
    main()
