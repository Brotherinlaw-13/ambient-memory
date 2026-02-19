#!/usr/bin/env python3
"""
Simplified Phase 3 evaluation script without full ChromaDB dependencies.
Tests the core scoring logic using the embedding server.
"""

import json
import re
import sys
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass 
class Query:
    text: str
    source_file: str
    line_number: int
    context_before: List[str] = None
    context_after: List[str] = None


class MessageExtractor:
    """Extract Diego's messages from conversation files."""
    
    def __init__(self):
        self.diego_patterns = [
            r'^\s*`\d+:\d+`\s*\*\*Diego:\*\*\s*(.+)',
            r'^\s*\*\*Diego:\*\*\s*(.+)',
            r'^\s*Diego:\s*(.+)',
            r'.*\*\*Diego:\*\*\s*(.+)',
        ]
        
        self.invalid_patterns = [
            r'^\[📎[^\]]+\]$',
            r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+$',
            r'^\*\*System\*\*:',
            r'^System:',
            r'^Bot:',
        ]
    
    def is_valid_query(self, text: str) -> bool:
        """Check if message is valid for evaluation."""
        if not text.strip():
            return False
        
        for pattern in self.invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        # Must have alphanumeric content
        import unicodedata
        text_no_emoji = ''.join(c for c in text if unicodedata.category(c) not in ['So', 'Sm'])
        text_cleaned = re.sub(r'[^\w\s]', '', text_no_emoji)
        return bool(re.search(r'[a-zA-Z0-9]', text_cleaned))
    
    def extract_from_file(self, file_path: Path) -> List[Query]:
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
            
            diego_text = None
            for pattern in self.diego_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if match.groups():
                        diego_text = match.group(1).strip()
                    else:
                        if '**Diego:**' in line:
                            parts = line.split('**Diego:**', 1)
                            if len(parts) > 1:
                                diego_text = parts[1].strip().lstrip('*').strip()
                        elif 'Diego:' in line:
                            parts = line.split('Diego:', 1)
                            if len(parts) > 1:
                                diego_text = parts[1].strip()
                    break
            
            if diego_text and self.is_valid_query(diego_text):
                # Get context
                context_before = []
                context_after = []
                
                for j in range(max(0, i-3), i):
                    if j < len(lines):
                        context_before.append(lines[j].strip())
                
                for j in range(i+1, min(len(lines), i+4)):
                    if j < len(lines):
                        context_after.append(lines[j].strip())
                
                queries.append(Query(
                    text=diego_text,
                    source_file=str(file_path),
                    line_number=i + 1,
                    context_before=context_before,
                    context_after=context_after
                ))
        
        return queries


class Phase3Scorer:
    """Reproduce Phase 3 relevance scoring logic."""
    
    def __init__(self):
        self.high_relevance_keywords = [
            "project", "work", "task", "deadline", "meeting", "client",
            "development", "build", "deploy", "feature", "bug", "issue"
        ]
        
        self.low_relevance_keywords = [
            "weather", "morning", "evening", "hello", "thanks", "ok"
        ]
        
        self.project_entities = [
            "darwin", "chowdown", "hire space", "railway", "the factory", 
            "the keep", "root juice", "tello", "stitch", "protein counter", 
            "geo", "diego", "rook"
        ]
    
    def extract_entities(self, text: str) -> set:
        """Extract entities from text."""
        words = text.split()
        entities = set()
        i = 0
        
        while i < len(words):
            word = words[i].strip(".,!?;:\"'()[]{}") 
            if word and word[0].isupper() and len(word) >= 2 and not word.isupper():
                entity_parts = [word]
                j = i + 1
                while j < len(words) and j - i < 3:
                    next_word = words[j].strip(".,!?;:\"'()[]{}")
                    if next_word and next_word[0].isupper() and len(next_word) >= 2:
                        entity_parts.append(next_word)
                        j += 1
                    else:
                        break
                entity = " ".join(entity_parts)
                entities.add(entity.lower())
                for part in entity_parts:
                    if len(part) >= 3:
                        entities.add(part.lower())
                i = j
            else:
                i += 1
        
        # Add ALL-CAPS acronyms
        for word in words:
            word = word.strip(".,!?;:\"'()[]{}")
            if word.isupper() and len(word) >= 2:
                entities.add(word.lower())
        
        return entities
    
    def score_relevance(self, query: Query, result_text: str) -> float:
        """Score relevance using Phase 3 logic."""
        result_text_lower = result_text.lower()
        query_text_lower = query.text.lower()
        
        # Basic keyword overlap
        query_words = set(query_text_lower.split())
        result_words = set(result_text_lower.split())
        exact_matches = query_words & result_words
        overlap_score = len(exact_matches) / max(len(query_words), 1)
        
        # Context overlap
        context_text = " ".join(query.context_before + query.context_after).lower()
        context_overlap = 0
        if context_text:
            context_words = set(context_text.split())
            context_matches = context_words & result_words
            context_overlap = len(context_matches) / max(len(context_words), 1)
        
        # Entity matching
        query_entities = self.extract_entities(query.text)
        result_entities = self.extract_entities(result_text)
        
        # Entity boost
        entity_boost = 0
        for entity in self.project_entities:
            if entity in query_text_lower and entity in result_text_lower:
                entity_boost += 0.4
        
        common_entities = query_entities & result_entities
        entity_boost += len(common_entities) * 0.1
        entity_boost = min(entity_boost, 0.7)
        
        # Topic relevance
        high_rel_score = sum(0.05 for kw in self.high_relevance_keywords if kw in result_text_lower)
        low_rel_penalty = sum(0.1 for kw in self.low_relevance_keywords if kw in result_text_lower and kw not in query_text_lower)
        
        # Short query handling
        if len(query_words) <= 5:
            expanded_query = query.text
            if query.context_before:
                recent_context = " ".join(query.context_before[-2:])
                expanded_query = recent_context + " " + expanded_query
            if query.context_after:
                following_context = " ".join(query.context_after[:2])
                expanded_query = expanded_query + " " + following_context
            
            expanded_words = set(expanded_query.lower().split())
            expanded_matches = expanded_words & result_words
            overlap_score = max(overlap_score, len(expanded_matches) / max(len(expanded_words), 1))
            
            context_weight = 0.3
            overlap_weight = 0.6
        else:
            context_weight = 0.1
            overlap_weight = 0.8
        
        # Final score
        final_score = (
            overlap_score * overlap_weight +
            context_overlap * context_weight +
            entity_boost +
            high_rel_score -
            low_rel_penalty
        )
        
        return max(0.0, min(1.0, final_score))


class SimpleEvaluator:
    """Simple evaluator using embedding server and mock results."""
    
    def __init__(self, embedding_server_url="http://localhost:9876"):
        self.embedding_server_url = embedding_server_url
        self.extractor = MessageExtractor()
        self.scorer = Phase3Scorer()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from server."""
        try:
            response = httpx.post(
                f"{self.embedding_server_url}/embed",
                json={"text": text, "prefix": "query"},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Embedding request failed: {e}")
            return []
    
    def mock_search_results(self, query: Query) -> List[str]:
        """
        Generate mock search results based on query content.
        This simulates what ChromaDB would return for testing scoring logic.
        """
        results = []
        
        # High relevance result (contains key terms)
        if any(proj in query.text.lower() for proj in self.scorer.project_entities):
            results.append(f"This is about {query.text[:50]}... project discussion with relevant context and details.")
        
        # Medium relevance result
        results.append(f"Some context mentioning {query.text.split()[0] if query.text.split() else 'topic'} with partial relevance.")
        
        # Low relevance/noise result
        results.append("This is a general conversation that doesn't really relate to the query much at all.")
        
        return results
    
    def evaluate_queries(self, queries: List[Query]) -> Dict[str, Any]:
        """Evaluate queries and compute metrics."""
        all_scores = []
        high_relevance = 0
        medium_relevance = 0
        low_relevance = 0
        
        print(f"Evaluating {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(queries)}")
            
            # Get embedding (to test server connectivity)
            embedding = self.get_embedding(query.text)
            if not embedding:
                continue
            
            # Generate mock results for testing
            mock_results = self.mock_search_results(query)
            
            # Score each result
            for result_text in mock_results:
                score = self.scorer.score_relevance(query, result_text)
                all_scores.append(score)
                
                if score >= 0.4:
                    high_relevance += 1
                elif score >= 0.2:
                    medium_relevance += 1
                else:
                    low_relevance += 1
        
        total_results = len(all_scores)
        if total_results == 0:
            return {"error": "No results scored"}
        
        metrics = {
            "total_queries": len(queries),
            "total_results": total_results,
            "avg_relevance_score": sum(all_scores) / len(all_scores),
            "precision_at_high": high_relevance / total_results,
            "noise_rate": low_relevance / total_results,
            "score_distribution": {
                "high_relevance": high_relevance,
                "medium_relevance": medium_relevance,
                "low_relevance": low_relevance
            }
        }
        
        return metrics


def main():
    """Run simplified Phase 3 evaluation."""
    print("🧠 Simplified Phase 3 Evaluation")
    print("=" * 40)
    
    # Check embedding server
    try:
        response = httpx.get("http://localhost:9876/health", timeout=5.0)
        if response.status_code != 200:
            print("❌ Embedding server not available at localhost:9876")
            return
        print("✅ Embedding server available")
    except Exception:
        print("❌ Cannot connect to embedding server at localhost:9876")
        return
    
    # Extract queries from test files
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    test_files = []
    for day in range(10, 18):  # Feb 10-17
        file_path = telegram_dir / f"the-factory-2026-02-{day:02d}.md"
        if file_path.exists():
            test_files.append(file_path)
    
    if not test_files:
        print("❌ No test files found")
        return
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Extract all queries
    evaluator = SimpleEvaluator()
    all_queries = []
    
    for file_path in test_files:
        queries = evaluator.extractor.extract_from_file(file_path)
        print(f"  {file_path.name}: {len(queries)} messages")
        all_queries.extend(queries)
    
    print(f"✅ Extracted {len(all_queries)} total queries")
    
    # Run evaluation
    print("\n🔍 Running evaluation...")
    metrics = evaluator.evaluate_queries(all_queries)
    
    # Display results
    print("\n" + "=" * 40)
    print("📊 SIMPLIFIED PHASE 3 EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Queries:         {metrics['total_queries']}")
    print(f"Total Results:         {metrics['total_results']}")
    print(f"Avg Relevance Score:   {metrics['avg_relevance_score']:.3f}")
    print(f"High Precision:        {metrics['precision_at_high']:.2%}")
    print(f"Noise Rate:            {metrics['noise_rate']:.2%}")
    
    # Save results
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "simplified_phase3_baseline",
        "metrics": metrics,
        "sample_queries": [q.text[:100] for q in all_queries[:10]]
    }
    
    results_file = eval_dir / "phase6-simplified-baseline.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    main()